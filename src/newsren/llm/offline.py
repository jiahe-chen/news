"""Offline provider: deterministic heuristics, no network, no cost.

Used when no Claude credentials are configured (and as the fallback when a
Claude call fails or is refused). Each task carries its own heuristic builder.
"""

from __future__ import annotations

from newsren.llm.base import Budget, LLMResult, StructuredTask, T, ToolRunner, ToolTask


class OfflineProvider:
    name = "offline"
    model: str | None = None

    def generate(self, task: StructuredTask[T], budget: Budget) -> LLMResult[T]:
        budget.check_time()
        output = task.offline()
        return LLMResult(output=output, provider="offline", model=None)

    def run_tools(self, task: ToolTask[T], budget: Budget) -> LLMResult[T]:
        runner = ToolRunner(task.tools, budget)
        output = task.offline(runner)
        return LLMResult(
            output=output,
            provider="offline",
            model=None,
            tool_calls=runner.calls,
            trace=runner.trace,
            citations=list(runner.citations.values()),
        )
