"""Claude provider (Anthropic Python SDK).

* Structured outputs via ``output_config.format`` (JSON schema generated from
  the task's pydantic model with ``anthropic.transform_schema``).
* Adaptive thinking with an explicit effort level.
* Server-side refusal fallback (``fallbacks="default"``) on by default.
* Agentic research uses a manual tool loop so the orchestrator can enforce
  tool-call, time, and cost budgets between turns.
* Any API failure, refusal, or unparseable output falls back to the task's
  offline builder and is reported as ``status="fallback"``.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic
from anthropic import transform_schema
from pydantic import ValidationError

from newsren.llm.base import (
    Budget,
    BudgetExceeded,
    LLMResult,
    StructuredTask,
    T,
    ToolRunner,
    ToolTask,
    Usage,
)
from newsren.llm.pricing import cost_usd

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"

AGENT_GROUND_RULES = (
    "You work for The Newsren, an editorially controlled news intelligence platform. "
    "Use only the evidence provided or retrieved through your tools; never invent sources, "
    "numbers, quotes, dates, or URLs. Cite evidence by its citation id. When credible sources "
    "disagree, describe the disagreement instead of choosing a side. Label forecasts, scenarios, "
    "and projections as such. Distinguish preprints from peer-reviewed work, vendor claims from "
    "independent evaluations, and backtests from live results. If something cannot be verified "
    "from the evidence, say so plainly."
)


def _usage_from(response: Any) -> Usage:
    u = getattr(response, "usage", None)
    if u is None:
        return Usage()
    return Usage(
        input_tokens=getattr(u, "input_tokens", 0) or 0,
        output_tokens=getattr(u, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
    )


def _final_text(response: Any) -> str:
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")


class ClaudeProvider:
    name = "claude"

    def __init__(self, model: str, default_effort: str = "medium", client: Any | None = None):
        self.model = model
        self.default_effort = default_effort
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = anthropic.Anthropic(max_retries=2)
        return self._client

    # ------------------------------------------------------------------ helpers

    def _system(self, system: str) -> list[dict[str, Any]]:
        # Stable prefix first so prompt caching can reuse it across requests.
        return [
            {"type": "text", "text": f"{AGENT_GROUND_RULES}\n\n{system}", "cache_control": {"type": "ephemeral"}}
        ]

    def _output_config(self, task_effort: str | None, output_model: type) -> dict[str, Any]:
        return {
            "effort": task_effort or self.default_effort,
            "format": {"type": "json_schema", "schema": transform_schema(output_model)},
        }

    def _create(self, budget: Budget, **kwargs: Any) -> Any:
        timeout = max(5.0, budget.remaining_seconds())
        return self.client.beta.messages.create(
            model=self.model,
            thinking={"type": "adaptive"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
            timeout=timeout,
            **kwargs,
        )

    def _fallback(self, task: StructuredTask[T] | ToolTask[T], budget: Budget, note: str, result: LLMResult[T]):
        log.warning("Claude task %s fell back to offline builder: %s", task.name, note)
        if isinstance(task, ToolTask):
            runner = ToolRunner(task.tools, Budget(**{**budget.as_dict()}))
            output = task.offline(runner)
            result.tool_calls += runner.calls
            result.trace.extend(runner.trace)
            result.citations = list({c.url: c for c in [*result.citations, *runner.citations.values()]}.values())
        else:
            output = task.offline()
        result.output = output
        result.provider = "offline"
        result.status = "refused" if note.startswith("refusal") else "fallback"
        result.note = note
        return result

    # ------------------------------------------------------------------ API

    def generate(self, task: StructuredTask[T], budget: Budget) -> LLMResult[T]:
        result: LLMResult[T] = LLMResult(output=None, provider="claude", model=self.model)  # type: ignore[arg-type]
        try:
            budget.check_time()
            response = self._create(
                budget,
                max_tokens=min(task.max_tokens or budget.max_output_tokens, budget.max_output_tokens),
                system=self._system(task.system),
                messages=[{"role": "user", "content": task.prompt}],
                output_config=self._output_config(task.effort, task.output_model),
            )
        except BudgetExceeded:
            raise
        except anthropic.APIError as exc:
            return self._fallback(task, budget, f"api error: {type(exc).__name__}", result)

        result.llm_calls = 1
        result.usage = _usage_from(response)
        result.model = getattr(response, "model", None) or self.model
        result.cost_usd = cost_usd(result.model, result.usage)

        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            return self._fallback(task, budget, f"refusal ({category or 'unspecified'})", result)
        if response.stop_reason == "max_tokens":
            return self._fallback(task, budget, "truncated output (max_tokens)", result)
        try:
            result.output = task.output_model.model_validate_json(_final_text(response))
        except (ValidationError, ValueError) as exc:
            return self._fallback(task, budget, f"invalid structured output: {exc}"[:300], result)
        return result

    def run_tools(self, task: ToolTask[T], budget: Budget) -> LLMResult[T]:
        result: LLMResult[T] = LLMResult(output=None, provider="claude", model=self.model)  # type: ignore[arg-type]
        runner = ToolRunner(task.tools, budget)
        messages: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
        tools = [t.to_api() for t in task.tools]
        # Each tool call costs at most one more model turn; +2 leaves room for
        # the final answer after the tool budget is spent.
        max_turns = budget.max_tool_calls + 2

        try:
            for _turn in range(max_turns):
                budget.check_time()
                response = self._create(
                    budget,
                    max_tokens=min(task.max_tokens or budget.max_output_tokens, budget.max_output_tokens),
                    system=self._system(task.system),
                    messages=messages,
                    tools=tools,
                    output_config=self._output_config(task.effort, task.output_model),
                )
                result.llm_calls += 1
                result.usage.add(_usage_from(response))
                result.model = getattr(response, "model", None) or self.model
                result.cost_usd = cost_usd(result.model, result.usage)
                budget.check_cost(result.cost_usd)

                if response.stop_reason == "refusal":
                    category = getattr(getattr(response, "stop_details", None), "category", None)
                    result.tool_calls, result.trace = runner.calls, runner.trace
                    return self._fallback(task, budget, f"refusal ({category or 'unspecified'})", result)

                tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
                if response.stop_reason != "tool_use" or not tool_uses:
                    if response.stop_reason == "max_tokens":
                        result.tool_calls, result.trace = runner.calls, runner.trace
                        return self._fallback(task, budget, "truncated output (max_tokens)", result)
                    result.output = task.output_model.model_validate_json(_final_text(response))
                    break

                # Append the full assistant content (thinking blocks included) unchanged.
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in tool_uses:
                    tool_input = block.input if isinstance(block.input, dict) else {}
                    outcome = runner.call(block.name, tool_input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": outcome.content,
                            "is_error": outcome.is_error,
                        }
                    )
                messages.append({"role": "user", "content": tool_results})
            else:
                result.tool_calls, result.trace = runner.calls, runner.trace
                return self._fallback(task, budget, "turn limit reached without a final answer", result)
        except BudgetExceeded as exc:
            result.tool_calls, result.trace = runner.calls, runner.trace
            result.citations = list(runner.citations.values())
            return self._fallback(task, budget, f"budget: {exc}", result)
        except anthropic.APIError as exc:
            result.tool_calls, result.trace = runner.calls, runner.trace
            return self._fallback(task, budget, f"api error: {type(exc).__name__}", result)
        except (ValidationError, ValueError) as exc:
            result.tool_calls, result.trace = runner.calls, runner.trace
            return self._fallback(task, budget, f"invalid structured output: {exc}"[:300], result)

        result.tool_calls = runner.calls
        result.trace = runner.trace
        result.citations = list(runner.citations.values())
        return result
