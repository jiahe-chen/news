"""Orchestrator and shared agent plumbing.

The orchestrator is the single entry point through which every agent task
runs. It selects the provider, enforces time/tool/cost budgets, records an
``AgentRun`` row (tokens, cost, duration, tool trace) for the cost and
efficiency metrics, and guarantees a result: if a budget is exceeded the
task's offline builder answers and the run is marked accordingly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic

from sqlalchemy import select
from sqlalchemy.orm import Session

from newsren.config import Settings, get_settings
from newsren.llm import (
    Budget,
    BudgetExceeded,
    LLMProvider,
    LLMResult,
    StructuredTask,
    ToolRunner,
    ToolTask,
    get_provider,
)
from newsren.llm.base import T
from newsren.models import AgentRun, AgentRunStatus, Article, GeneratedContent, utcnow
from newsren.schemas import CitationRef, EvidenceDoc
from newsren.textutil import iso, normalize_space, truncate

log = logging.getLogger(__name__)

_STATUS = {"ok": AgentRunStatus.OK, "fallback": AgentRunStatus.FALLBACK, "refused": AgentRunStatus.REFUSED}


@dataclass
class AgentOutcome(Generic[T]):
    output: T
    run: AgentRun
    result: LLMResult[T]

    @property
    def generated_by(self) -> str:
        return self.result.generated_by


class Orchestrator:
    def __init__(self, session: Session, provider: LLMProvider | None = None, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()
        self.provider = provider or get_provider(self.settings)

    def budget(self, **overrides: Any) -> Budget:
        s = self.settings
        values = {
            "max_tool_calls": s.agent_max_tool_calls,
            "timeout_seconds": s.agent_timeout_seconds,
            "max_cost_usd": s.agent_max_cost_usd,
            "max_output_tokens": s.agent_max_output_tokens,
        }
        values.update(overrides)
        return Budget(**values)

    def run(
        self,
        agent: str,
        task: StructuredTask[T] | ToolTask[T],
        *,
        subject_type: str | None = None,
        subject_id: int | None = None,
        budget: Budget | None = None,
    ) -> AgentOutcome[T]:
        budget = budget or self.budget()
        run = AgentRun(
            agent=agent,
            task=truncate(task.name, 300),
            status=AgentRunStatus.RUNNING,
            provider=self.provider.name,
            model=self.provider.model,
            subject_type=subject_type,
            subject_id=subject_id,
            budget=budget.as_dict(),
        )
        self.session.add(run)
        self.session.flush()
        started = time.monotonic()
        try:
            if isinstance(task, ToolTask):
                result = self.provider.run_tools(task, budget)
            else:
                result = self.provider.generate(task, budget)
            run.status = _STATUS.get(result.status, AgentRunStatus.OK)
            run.error = result.note
        except BudgetExceeded as exc:
            log.warning("Agent %s exceeded budget: %s", agent, exc)
            result = self._offline_fallback(task, budget)
            run.status = AgentRunStatus.BUDGET_EXCEEDED
            run.error = str(exc)
        except Exception as exc:
            run.status = AgentRunStatus.ERROR
            run.error = f"{type(exc).__name__}: {exc}"[:2000]
            run.finished_at = utcnow()
            run.duration_ms = int((time.monotonic() - started) * 1000)
            self.session.flush()
            raise

        run.provider = result.provider
        run.model = result.model if result.provider == "claude" else None
        run.llm_calls = result.llm_calls
        run.tool_calls = result.tool_calls
        run.retrieval_requests = sum(1 for t in result.trace if not t.get("is_error"))
        run.input_tokens = result.usage.input_tokens
        run.output_tokens = result.usage.output_tokens
        run.cache_read_tokens = result.usage.cache_read_tokens
        run.cache_write_tokens = result.usage.cache_write_tokens
        run.cost_usd = round(result.cost_usd, 6)
        run.trace = result.trace[:50]
        run.finished_at = utcnow()
        run.duration_ms = int((time.monotonic() - started) * 1000)
        self.session.flush()
        return AgentOutcome(output=result.output, run=run, result=result)

    @staticmethod
    def _offline_fallback(task: StructuredTask[T] | ToolTask[T], budget: Budget) -> LLMResult[T]:
        if isinstance(task, ToolTask):
            runner = ToolRunner(task.tools, Budget(**budget.as_dict()))
            output = task.offline(runner)
            return LLMResult(
                output=output,
                provider="offline",
                model=None,
                tool_calls=runner.calls,
                trace=runner.trace,
                status="fallback",
                citations=list(runner.citations.values()),
            )
        return LLMResult(output=task.offline(), provider="offline", model=None, status="fallback")


# --------------------------------------------------------------------------
# Evidence packs: the only material an agent may cite
# --------------------------------------------------------------------------


def citation_for(article: Article, cid: str, snippet: str = "") -> CitationRef:
    return CitationRef(
        id=cid,
        url=article.url,
        title=article.title,
        publisher=article.source.name if article.source else "",
        published_at=iso(article.published_at),
        retrieved_at=iso(article.retrieved_at),
        article_id=article.id,
        snippet=truncate(snippet or article.summary or article.content or "", 280),
        trust_tier=article.trust_tier,
        doc_type=article.doc_type,
    )


def article_text(article: Article, max_chars: int = 4000) -> str:
    parts = [article.title]
    summary = normalize_space(article.summary)
    content = normalize_space(article.content)
    if summary and not content.startswith(summary.rstrip("…").rstrip(".")):
        parts.append(article.summary or "")
    if content:
        parts.append(article.content or "")
    return truncate("\n\n".join(parts), max_chars)


def build_evidence_pack(
    articles: Sequence[Article], max_chars_per_doc: int = 4000, start: int = 1
) -> list[EvidenceDoc]:
    """Number source documents c1..cN. Duplicates are skipped (original kept)."""
    docs: list[EvidenceDoc] = []
    seen: set[int] = set()
    for article in articles:
        key = article.duplicate_of_id or article.id
        if key in seen:
            continue
        seen.add(key)
        cid = f"c{start + len(docs)}"
        docs.append(
            EvidenceDoc(citation=citation_for(article, cid), text=article_text(article, max_chars_per_doc))
        )
    return docs


def render_evidence(docs: Sequence[EvidenceDoc]) -> str:
    """Prompt-ready rendering of an evidence pack."""
    blocks = []
    for doc in docs:
        c = doc.citation
        meta = ", ".join(
            x
            for x in [
                c.publisher,
                f"published {c.published_at}" if c.published_at else "publication date unknown",
                f"retrieved {c.retrieved_at}" if c.retrieved_at else "",
                f"source type: {c.doc_type}",
                f"trust tier: {c.trust_tier}",
            ]
            if x
        )
        blocks.append(f'<source id="{c.id}">\n{c.title}\n({meta})\n{c.url}\n\n{doc.text}\n</source>')
    return "\n\n".join(blocks)


def citations_of(docs: Sequence[EvidenceDoc]) -> list[CitationRef]:
    return [d.citation for d in docs]


def valid_citation_ids(docs: Sequence[EvidenceDoc]) -> set[str]:
    return {d.citation.id for d in docs}


# --------------------------------------------------------------------------
# Generated-content cache
# --------------------------------------------------------------------------


def cache_key(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:64]


def get_cached(session: Session, kind: str, key: str) -> GeneratedContent | None:
    return session.scalars(
        select(GeneratedContent)
        .where(GeneratedContent.kind == kind, GeneratedContent.cache_key == key)
        .order_by(GeneratedContent.generated_at.desc())
    ).first()


def get_or_create_content(
    session: Session,
    kind: str,
    key_parts: Sequence[Any],
    build: Callable[[], tuple[dict[str, Any], list[CitationRef], AgentOutcome[Any] | None]],
    *,
    brief_id: int | None = None,
    story_id: int | None = None,
    article_id: int | None = None,
    audience: str | None = None,
    params: dict[str, Any] | None = None,
    refresh: bool = False,
) -> GeneratedContent:
    """Return cached generated content or build, store, and return it."""
    key = cache_key(kind, *key_parts)
    if not refresh:
        existing = get_cached(session, kind, key)
        if existing is not None:
            return existing
    content, citations, outcome = build()
    row = GeneratedContent(
        kind=kind,
        cache_key=key,
        brief_id=brief_id,
        story_id=story_id,
        article_id=article_id,
        audience=audience,
        params=params or {},
        content=content,
        citations=[c.model_dump() for c in citations],
        generated_by=outcome.generated_by if outcome else "offline",
        agent_run_id=outcome.run.id if outcome else None,
    )
    session.add(row)
    session.flush()
    return row
