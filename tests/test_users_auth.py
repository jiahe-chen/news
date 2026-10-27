"""Accounts: scrypt hashing, constant-time verification, account creation, and sign-in."""

from __future__ import annotations

import base64
import hashlib

import pytest

from newsren.users.auth import (
    authenticate,
    create_user,
    get_user_by_email,
    hash_password,
    needs_rehash,
    set_password,
    verify_password,
)


def _legacy_hash(password: str, n: int = 2**10) -> str:
    salt = b"0123456789abcdef"
    digest = hashlib.scrypt(password.encode(), salt=salt, n=n, r=8, p=1, dklen=64)
    return f"scrypt${n}$8$1${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def test_hash_format_and_parameters():
    hashed = hash_password("correct horse battery")
    scheme, n, r, p, salt_b64, hash_b64 = hashed.split("$")
    assert (scheme, n, r, p) == ("scrypt", str(2**14), "8", "1")
    assert len(base64.b64decode(salt_b64)) == 16
    assert len(base64.b64decode(hash_b64)) == 64
    assert "correct horse" not in hashed


def test_same_password_gets_a_fresh_salt():
    assert hash_password("same-password") != hash_password("same-password")


def test_verify_password_accepts_only_the_right_password():
    hashed = hash_password("s3cret-passphrase")
    assert verify_password("s3cret-passphrase", hashed)
    assert not verify_password("s3cret-passphrasE", hashed)
    assert not verify_password("", hashed)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "x",
        "scrypt$abc",
        "bcrypt$16384$8$1$AAAA$AAAA",
        "scrypt$16384$8$1$not base64!$AAAA",
        "scrypt$16383$8$1$AAAAAAAAAAAAAAAAAAAAAA==$AAAA",  # n not a power of two
        f"scrypt${2**30}$8$1$AAAAAAAAAAAAAAAAAAAAAA==$AAAA",  # absurd work factor is refused, not computed
        "scrypt$16384$8$1$$AAAA",
    ],
)
def test_verify_password_rejects_malformed_hashes(bad):
    assert verify_password("anything-at-all", bad) is False


def test_verify_password_tolerates_non_strings():
    assert verify_password(None, hash_password("password1")) is False  # type: ignore[arg-type]
    assert verify_password("password1", None) is False  # type: ignore[arg-type]


def test_needs_rehash():
    assert not needs_rehash(hash_password("password-1"))
    assert needs_rehash(_legacy_hash("password-1"))
    assert needs_rehash("garbage")
    assert verify_password("password-1", _legacy_hash("password-1"))


def test_create_user_normalizes_email_and_hashes_password(session):
    user = create_user(session, "  Reader@Example.COM ", "long-enough-pw", "  Ada   Reader ")
    assert user.id is not None
    assert user.email == "reader@example.com"
    assert user.display_name == "Ada Reader"
    assert user.role == "reader"
    assert user.password_hash.startswith("scrypt$")
    assert verify_password("long-enough-pw", user.password_hash)


def test_create_user_defaults_display_name_and_accepts_editor_role(session):
    editor = create_user(session, "desk@newsren.example", "editor-pass-1", "", role="editor")
    assert editor.display_name == "desk"
    assert editor.is_editor


def test_create_user_rejects_duplicates_case_insensitively(session):
    create_user(session, "dup@example.com", "password-one", "One")
    with pytest.raises(ValueError, match="already exists"):
        create_user(session, " DUP@example.com", "password-two", "Two")


@pytest.mark.parametrize("email", ["", "no-at-sign", "a@b", "a b@example.com", "@example.com", "a@@example.com"])
def test_create_user_rejects_bad_email(session, email):
    with pytest.raises(ValueError, match="valid email"):
        create_user(session, email, "password-long", "X")


def test_create_user_rejects_short_password_and_unknown_role(session):
    with pytest.raises(ValueError, match="at least 8"):
        create_user(session, "short@example.com", "1234567", "Short")
    with pytest.raises(ValueError, match="Unknown role"):
        create_user(session, "role@example.com", "password-long", "Role", role="superuser")


def test_authenticate(session):
    user = create_user(session, "signin@example.com", "open-sesame-1", "Sign In")
    assert authenticate(session, "SIGNIN@example.com ", "open-sesame-1") is user
    assert authenticate(session, "signin@example.com", "open-sesame-2") is None
    assert authenticate(session, "nobody@example.com", "open-sesame-1") is None
    assert get_user_by_email(session, "Signin@Example.com") is user


def test_authenticate_handles_unusable_hash(session, factory):
    user = factory.user(email="legacy@example.com", password_hash="x")
    assert user.password_hash == "x"
    assert authenticate(session, "legacy@example.com", "whatever-pass") is None


def test_authenticate_upgrades_old_parameters(session):
    user = create_user(session, "upgrade@example.com", "first-password", "Upgrade")
    user.password_hash = _legacy_hash("first-password")
    assert authenticate(session, "upgrade@example.com", "first-password") is user
    assert user.password_hash.split("$")[1] == str(2**14)
    assert verify_password("first-password", user.password_hash)


def test_set_password(session):
    user = create_user(session, "change@example.com", "old-password", "Change")
    set_password(session, user, "new-password")
    assert authenticate(session, "change@example.com", "new-password") is user
    assert authenticate(session, "change@example.com", "old-password") is None
    with pytest.raises(ValueError):
        set_password(session, user, "short")
