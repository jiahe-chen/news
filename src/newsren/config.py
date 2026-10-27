"""Runtime configuration, read once from environment variables.

Every setting has a safe default so the app runs locally with no
configuration at all (offline LLM mode, SQLite in ./data).
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    database_url: str = ""

    # LLM: "auto" uses Claude when credentials exist, otherwise the offline
    # heuristic provider. "claude" forces Claude, "offline" forces heuristics.
    llm_mode: str = "auto"
    model: str = "claude-opus-5-5"
    default_effort: str = "medium"

    # Orchestrator limits (per agent task).
    agent_max_tool_calls: int = 8
    agent_timeout_seconds: float = 120.0
    agent_max_cost_usd: float = 0.75
    agent_max_output_tokens: int = 16000

    # Editorial controls.
    require_human_approval: bool = True
    auto_deliver_personal_briefings: bool = True

    # Source access.
    contact_email: str = "newsroom@newsren.example"
    fetch_timeout_seconds: float = 20.0
    max_items_per_source: int = 30

    # Delivery.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "The Newsren <briefings@newsren.example>"
    smtp_starttls: bool = True

    # Web.
    secret_key: str = field(default_factory=lambda: secrets.token_hex(32))
    base_url: str = "http://localhost:8000"
    scheduler_enabled: bool = False
    scheduler_interval_minutes: int = 60

    @property
    def outbox_dir(self) -> Path:
        return self.data_dir / "outbox"

    @property
    def user_agent(self) -> str:
        # SEC EDGAR requires a descriptive User-Agent with a contact address.
        return f"TheNewsren/0.1 (news research; {self.contact_email})"

    def with_overrides(self, **kwargs) -> Settings:
        return replace(self, **kwargs)


def load_dotenv(path: Path = Path(".env")) -> None:
    """Read KEY=VALUE lines from ``.env`` into the environment (real env vars win)."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().removeprefix("export ").strip(), value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


def _secret_key(data_dir: Path) -> str:
    """NEWSREN_SECRET_KEY, else a random key persisted in the data dir so sessions survive restarts."""
    configured = os.environ.get("NEWSREN_SECRET_KEY")
    if configured:
        return configured
    key_file = data_dir / ".secret_key"
    try:
        if key_file.is_file():
            return key_file.read_text().strip()
        data_dir.mkdir(parents=True, exist_ok=True)
        key = secrets.token_hex(32)
        key_file.write_text(key)
        key_file.chmod(0o600)
        return key
    except OSError:
        return secrets.token_hex(32)


def load_settings() -> Settings:
    load_dotenv()
    data_dir = Path(os.environ.get("NEWSREN_DATA_DIR", "data"))
    database_url = os.environ.get("NEWSREN_DATABASE_URL") or f"sqlite:///{data_dir / 'newsren.db'}"
    secret = _secret_key(data_dir)
    return Settings(
        data_dir=data_dir,
        database_url=database_url,
        llm_mode=os.environ.get("NEWSREN_LLM", "auto").strip().lower(),
        model=os.environ.get("NEWSREN_MODEL", "claude-opus-5-5"),
        default_effort=os.environ.get("NEWSREN_EFFORT", "medium"),
        agent_max_tool_calls=_int("NEWSREN_AGENT_MAX_TOOL_CALLS", 8),
        agent_timeout_seconds=_float("NEWSREN_AGENT_TIMEOUT_SECONDS", 120.0),
        agent_max_cost_usd=_float("NEWSREN_AGENT_MAX_COST_USD", 0.75),
        agent_max_output_tokens=_int("NEWSREN_AGENT_MAX_OUTPUT_TOKENS", 16000),
        require_human_approval=_bool("NEWSREN_REQUIRE_HUMAN_APPROVAL", True),
        auto_deliver_personal_briefings=_bool("NEWSREN_AUTO_DELIVER_BRIEFINGS", True),
        contact_email=os.environ.get("NEWSREN_CONTACT_EMAIL", "newsroom@newsren.example"),
        fetch_timeout_seconds=_float("NEWSREN_FETCH_TIMEOUT_SECONDS", 20.0),
        max_items_per_source=_int("NEWSREN_MAX_ITEMS_PER_SOURCE", 30),
        smtp_host=os.environ.get("NEWSREN_SMTP_HOST", ""),
        smtp_port=_int("NEWSREN_SMTP_PORT", 587),
        smtp_user=os.environ.get("NEWSREN_SMTP_USER", ""),
        smtp_password=os.environ.get("NEWSREN_SMTP_PASSWORD", ""),
        smtp_from=os.environ.get("NEWSREN_SMTP_FROM", "The Newsren <briefings@newsren.example>"),
        smtp_starttls=_bool("NEWSREN_SMTP_STARTTLS", True),
        secret_key=secret,
        base_url=os.environ.get("NEWSREN_BASE_URL", "http://localhost:8000"),
        scheduler_enabled=_bool("NEWSREN_SCHEDULER", False),
        scheduler_interval_minutes=_int("NEWSREN_SCHEDULER_INTERVAL_MINUTES", 60),
    )


_override: Settings | None = None


@lru_cache(maxsize=1)
def _cached() -> Settings:
    return load_settings()


def get_settings() -> Settings:
    """Return the active settings (test code may install an override)."""
    return _override if _override is not None else _cached()


def set_settings(settings: Settings | None) -> None:
    """Install (or clear, with None) a settings override. Used by tests and the CLI."""
    global _override
    _override = settings
