"""Reader and editor accounts: scrypt password hashing and authentication.

Passwords are hashed with ``hashlib.scrypt`` (n=2**14, r=8, p=1, a random
16-byte salt) and stored self-describing as
``scrypt$<n>$<r>$<p>$<salt_b64>$<hash_b64>`` so the work factor can be raised
later without invalidating existing hashes (``needs_rehash`` + rehash on the
next successful sign-in). Verification is constant-time, and signing in with
an unknown email still performs one scrypt computation so response times do
not reveal which addresses have accounts.

Functions flush but never commit; callers own the transaction.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import re
import secrets

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsren.models import Role, User

SCHEME = "scrypt"
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SALT_BYTES = 16
KEY_BYTES = 64
MIN_PASSWORD_LENGTH = 8
# Long enough for any passphrase; bounded so a huge body cannot tie up a worker.
MAX_PASSWORD_LENGTH = 1024
MAX_EMAIL_LENGTH = 320
MAX_DISPLAY_NAME_LENGTH = 120

# Upper bounds on parameters read back from stored hashes. A tampered or
# corrupt hash must not be able to request gigabytes of memory.
_MAX_N = 2**20
_MAX_R = 32
_MAX_P = 16
_MAXMEM = 256 * 1024 * 1024

# Deliberately simple: one "@", no whitespace, a dot in the domain. Real
# validation happens when a briefing email is (or is not) delivered.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s.]{2,}$")

# Hash compared against when the email is unknown (timing equalisation).
_DUMMY_HASH: str | None = None


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _derive(password: str, salt: bytes, n: int, r: int, p: int, dklen: int) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, maxmem=_MAXMEM, dklen=dklen)


def hash_password(password: str) -> str:
    """Hash ``password`` with scrypt and a fresh random salt.

    Returns ``"scrypt$n$r$p$salt_b64$hash_b64"``. Length rules are enforced by
    ``create_user``/``set_password``, not here.
    """
    if not isinstance(password, str):
        raise TypeError("password must be a string")
    salt = secrets.token_bytes(SALT_BYTES)
    digest = _derive(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P, KEY_BYTES)
    return f"{SCHEME}${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def _parse(hashed: str) -> tuple[int, int, int, bytes, bytes] | None:
    parts = (hashed or "").split("$")
    if len(parts) != 6 or parts[0] != SCHEME:
        return None
    try:
        n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
        salt = base64.b64decode(parts[4], validate=True)
        expected = base64.b64decode(parts[5], validate=True)
    except (ValueError, binascii.Error):
        return None
    if not (1 < n <= _MAX_N and n & (n - 1) == 0 and 0 < r <= _MAX_R and 0 < p <= _MAX_P):
        return None
    if not salt or not expected:
        return None
    return n, r, p, salt, expected


def verify_password(password: str, hashed: str) -> bool:
    """True when ``password`` matches ``hashed``; False for wrong or malformed input (never raises)."""
    if not isinstance(password, str) or not isinstance(hashed, str):
        return False
    parsed = _parse(hashed)
    if parsed is None:
        return False
    n, r, p, salt, expected = parsed
    try:
        actual = _derive(password, salt, n, r, p, len(expected))
    except (ValueError, MemoryError):
        return False
    return hmac.compare_digest(actual, expected)


def needs_rehash(hashed: str) -> bool:
    """True when ``hashed`` was made with parameters other than the current ones."""
    parsed = _parse(hashed)
    if parsed is None:
        return True
    n, r, p, salt, expected = parsed
    return (n, r, p, len(salt), len(expected)) != (SCRYPT_N, SCRYPT_R, SCRYPT_P, SALT_BYTES, KEY_BYTES)


def normalize_email(email: str) -> str:
    """Canonical form used for storage and lookup: stripped and lower-cased."""
    return (email or "").strip().lower()


def validate_email(email: str) -> str:
    """Return the normalized email or raise ``ValueError`` if it is not a plausible address."""
    normalized = normalize_email(email)
    if not normalized or len(normalized) > MAX_EMAIL_LENGTH or not _EMAIL_RE.match(normalized):
        raise ValueError("Enter a valid email address, like name@example.com.")
    return normalized


def validate_password(password: str) -> str:
    """Return ``password`` unchanged or raise ``ValueError`` if it breaks the length rules."""
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Passwords must be at least {MIN_PASSWORD_LENGTH} characters long.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(f"Passwords must be at most {MAX_PASSWORD_LENGTH} characters long.")
    return password


def get_user_by_email(session: Session, email: str) -> User | None:
    """Look up an account by email (case-insensitive)."""
    normalized = normalize_email(email)
    if not normalized:
        return None
    return session.scalars(select(User).where(func.lower(User.email) == normalized)).first()


def create_user(session: Session, email: str, password: str, display_name: str, role: str = Role.READER) -> User:
    """Create an account.

    The email is stripped and lower-cased; it must look like an address and be
    unused. The password must be at least eight characters. ``display_name``
    defaults to the email's local part. Raises ``ValueError`` with a
    reader-facing message on any validation failure or duplicate email.
    """
    normalized = validate_email(email)
    validate_password(password)
    role_value = (role or Role.READER).strip().lower()
    if role_value not in {r.value for r in Role}:
        raise ValueError(f"Unknown role {role!r}; use one of {', '.join(r.value for r in Role)}.")
    if get_user_by_email(session, normalized) is not None:
        raise ValueError("An account with this email already exists.")
    name = " ".join((display_name or "").split()) or normalized.split("@", 1)[0]
    user = User(
        email=normalized,
        display_name=name[:MAX_DISPLAY_NAME_LENGTH],
        password_hash=hash_password(password),
        role=role_value,
    )
    session.add(user)
    session.flush()
    return user


def set_password(session: Session, user: User, new_password: str) -> User:
    """Replace a user's password (length rules apply)."""
    validate_password(new_password)
    user.password_hash = hash_password(new_password)
    session.flush()
    return user


def _dummy_hash() -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password(secrets.token_urlsafe(16))
    return _DUMMY_HASH


def authenticate(session: Session, email: str, password: str) -> User | None:
    """Return the user when ``email``/``password`` match, else None.

    Unknown emails still cost one scrypt computation (no account enumeration
    by timing). A successful sign-in upgrades hashes made with older
    parameters.
    """
    user = get_user_by_email(session, email)
    if user is None:
        verify_password(password if isinstance(password, str) else "", _dummy_hash())
        return None
    if not verify_password(password, user.password_hash):
        return None
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
        session.flush()
    return user
