"""LLM provider selection."""

from __future__ import annotations

import os

from newsren.config import Settings, get_settings
from newsren.llm.base import (
    Budget,
    BudgetExceeded,
    LLMProvider,
    LLMResult,
    StructuredTask,
    Tool,
    ToolResult,
    ToolRunner,
    ToolTask,
    Usage,
)
from newsren.llm.offline import OfflineProvider

__all__ = [
    "Budget",
    "BudgetExceeded",
    "LLMProvider",
    "LLMResult",
    "StructuredTask",
    "Tool",
    "ToolResult",
    "ToolRunner",
    "ToolTask",
    "Usage",
    "OfflineProvider",
    "claude_credentials_available",
    "get_provider",
    "set_provider",
]

_provider_override: LLMProvider | None = None


def claude_credentials_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def get_provider(settings: Settings | None = None) -> LLMProvider:
    """Return the configured provider (``NEWSREN_LLM`` = auto | claude | offline)."""
    if _provider_override is not None:
        return _provider_override
    settings = settings or get_settings()
    mode = settings.llm_mode
    if mode == "offline" or (mode == "auto" and not claude_credentials_available()):
        return OfflineProvider()
    from newsren.llm.claude import ClaudeProvider

    return ClaudeProvider(model=settings.model, default_effort=settings.default_effort)


def set_provider(provider: LLMProvider | None) -> None:
    """Install (or clear) a provider override, e.g. a fake Claude client in tests."""
    global _provider_override
    _provider_override = provider
