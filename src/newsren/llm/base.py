"""Provider-neutral task definitions for the agent layer.

An agent describes *what* it needs (a ``StructuredTask`` or a ``ToolTask``)
and always supplies an ``offline`` builder: a deterministic, heuristic
implementation that answers from the same evidence without an LLM. The
Claude provider uses the model; the offline provider (and the Claude provider
on refusal/outage) uses the builder. This keeps the platform usable, testable,
and honest about which output came from where (``LLMResult.provider``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel

from newsren.schemas import CitationRef

T = TypeVar("T", bound=BaseModel)


class BudgetExceeded(RuntimeError):
    """Raised when an agent task exceeds its time, tool-call, or cost budget."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens


@dataclass
class Budget:
    """Per-task limits enforced by the orchestrator and providers."""

    max_tool_calls: int = 8
    timeout_seconds: float = 120.0
    max_cost_usd: float = 0.75
    max_output_tokens: int = 16000
    started_at: float = field(default_factory=time.monotonic)

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def remaining_seconds(self) -> float:
        return self.timeout_seconds - self.elapsed()

    def check_time(self) -> None:
        if self.elapsed() > self.timeout_seconds:
            raise BudgetExceeded(f"time budget of {self.timeout_seconds:.0f}s exceeded")

    def check_cost(self, cost_usd: float) -> None:
        if cost_usd > self.max_cost_usd:
            raise BudgetExceeded(f"cost budget of ${self.max_cost_usd:.2f} exceeded (${cost_usd:.3f})")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_tool_calls": self.max_tool_calls,
            "timeout_seconds": self.timeout_seconds,
            "max_cost_usd": self.max_cost_usd,
            "max_output_tokens": self.max_output_tokens,
        }


@dataclass
class ToolResult:
    content: str
    citations: list[CitationRef] = field(default_factory=list)
    is_error: bool = False
    data: Any = None  # structured payload for offline builders


@dataclass
class Tool:
    """A retrieval tool an agent may call. Only approved sources are reachable."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], ToolResult]

    def to_api(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "strict": True,
        }


class ToolRunner:
    """Executes tools with budget accounting and a trace (shared by providers)."""

    def __init__(self, tools: list[Tool], budget: Budget):
        self._tools = {t.name: t for t in tools}
        self.budget = budget
        self.calls = 0
        self.trace: list[dict[str, Any]] = []
        self.citations: dict[str, CitationRef] = {}

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    def exhausted(self) -> bool:
        return self.calls >= self.budget.max_tool_calls

    def call(self, name: str, tool_input: dict[str, Any]) -> ToolResult:
        self.budget.check_time()
        if name not in self._tools:
            result = ToolResult(
                content=f"Unknown tool '{name}'. Available: {', '.join(self._tools)}", is_error=True
            )
        elif self.exhausted():
            result = ToolResult(
                content="Tool budget exhausted. Write your final answer now from the evidence already gathered.",
                is_error=True,
            )
        else:
            self.calls += 1
            try:
                result = self._tools[name].handler(tool_input)
            except BudgetExceeded:
                raise
            except Exception as exc:  # tool failures are reported to the model, not raised
                result = ToolResult(content=f"Tool error: {type(exc).__name__}: {exc}", is_error=True)
        for cit in result.citations:
            self.citations.setdefault(cit.url, cit)
        self.trace.append(
            {
                "tool": name,
                "input": tool_input,
                "is_error": result.is_error,
                "result_preview": result.content[:300],
                "citations": [c.url for c in result.citations],
            }
        )
        return result


@dataclass
class StructuredTask(Generic[T]):
    """One structured-output request: evidence in, validated pydantic model out."""

    name: str
    system: str
    prompt: str
    output_model: type[T]
    offline: Callable[[], T]
    effort: str | None = None
    max_tokens: int | None = None


@dataclass
class ToolTask(Generic[T]):
    """An agentic research task: the model plans, calls approved tools, then answers."""

    name: str
    system: str
    prompt: str
    tools: list[Tool]
    output_model: type[T]
    offline: Callable[[ToolRunner], T]
    effort: str | None = None
    max_tokens: int | None = None


@dataclass
class LLMResult(Generic[T]):
    output: T
    provider: str  # "claude" | "offline"
    model: str | None
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)
    # "ok" | "fallback" (offline answered after a Claude failure) | "refused"
    status: str = "ok"
    note: str | None = None
    citations: list[CitationRef] = field(default_factory=list)  # gathered by tools

    @property
    def generated_by(self) -> str:
        return self.model if self.provider == "claude" and self.model else "offline"


class LLMProvider(Protocol):
    name: str
    model: str | None

    def generate(self, task: StructuredTask[T], budget: Budget) -> LLMResult[T]: ...

    def run_tools(self, task: ToolTask[T], budget: Budget) -> LLMResult[T]: ...
