"""The editorially controlled news-intelligence workflow.

Discover -> Connect -> Explain -> Verify (quality gate) -> Output, run by the
orchestrator with per-task budgets. Each step is independent so one failure
(e.g. a source outage) never blocks the rest; errors are collected in the report.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from newsren.agents.base import Orchestrator
from newsren.config import Settings, get_settings
from newsren.models import Article, ArticleEntity, Brief, BriefStatus, Story

log = logging.getLogger(__name__)


@dataclass
class PipelineReport:
    new_articles: int = 0
    duplicates: int = 0
    sources_ok: int = 0
    sources_failed: int = 0
    demo_articles_loaded: int = 0
    entities_tagged: int = 0
    stories_changed: int = 0
    stories_refreshed: int = 0
    briefs_generated: int = 0
    briefs_in_review: int = 0
    briefs_published: int = 0
    finance: dict[str, Any] = field(default_factory=dict)
    monitoring: dict[str, Any] = field(default_factory=dict)
    briefings: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _step(report: PipelineReport, name: str, session: Session, fn, *args, **kwargs):
    """Run one pipeline step inside a savepoint; record and contain failures."""
    try:
        with session.begin_nested():
            return fn(*args, **kwargs)
    except Exception as exc:  # contain: one failing step must not stop the workflow
        log.exception("Pipeline step %s failed", name)
        report.errors.append(f"{name}: {type(exc).__name__}: {exc}")
        return None


def tag_untagged_articles(session: Session, limit: int = 2000) -> int:
    from newsren.intelligence.entities import tag_article_entities

    tagged_ids = select(ArticleEntity.article_id).distinct()
    articles = session.scalars(
        select(Article).where(Article.id.not_in(tagged_ids), Article.duplicate_of_id.is_(None)).limit(limit)
    ).all()
    for article in articles:
        tag_article_entities(session, article)
    return len(articles)


def stories_needing_briefs(session: Session, limit: int) -> list[Story]:
    """Stories whose latest brief is missing or older than their last update."""
    stories = session.scalars(select(Story).order_by(Story.last_updated_at.desc())).all()
    out: list[Story] = []
    for story in stories:
        latest = session.scalars(
            select(Brief)
            .where(Brief.story_id == story.id, Brief.kind == "story")
            .order_by(Brief.generated_at.desc())
        ).first()
        if latest is None or latest.generated_at < story.last_updated_at:
            out.append(story)
        if len(out) >= limit:
            break
    return out


def run_pipeline(
    session: Session,
    *,
    settings: Settings | None = None,
    network: bool = True,
    demo: bool = False,
    max_briefs: int = 25,
    finance: bool = True,
    monitor: bool = True,
    briefings: bool = True,
) -> PipelineReport:
    from newsren.agents.brief import generate_brief
    from newsren.ingest.fixtures import load_demo_data
    from newsren.ingest.pipeline import ingest_all
    from newsren.ingest.sources import sync_sources
    from newsren.intelligence.clustering import assign_stories
    from newsren.intelligence.timeline import refresh_story

    settings = settings or get_settings()
    report = PipelineReport()
    orchestrator = Orchestrator(session, settings=settings)

    # 1. Discover: approved sources only.
    _step(report, "sync_sources", session, sync_sources, session)
    if demo:
        loaded = _step(report, "demo_data", session, load_demo_data, session)
        report.demo_articles_loaded = loaded or 0
    if network:
        ingest = _step(report, "ingest", session, ingest_all, session, settings)
        if ingest is not None:
            report.new_articles = ingest.new_articles
            report.duplicates = ingest.duplicates
            report.sources_ok = ingest.sources_ok
            report.sources_failed = ingest.sources_failed
            report.errors.extend(ingest.errors[:20])

    # 2. Connect: entities, story clustering, timelines and story intelligence.
    report.entities_tagged = _step(report, "entities", session, tag_untagged_articles, session) or 0
    changed = _step(report, "clustering", session, assign_stories, session) or []
    report.stories_changed = len(changed)
    for story in changed:
        if _step(report, f"refresh_story:{story.id}", session, refresh_story, session, story, orchestrator):
            report.stories_refreshed += 1

    # 3-4. Explain + Verify: AI Story Briefs pass through the editorial quality gate.
    for story in stories_needing_briefs(session, max_briefs):
        brief = _step(report, f"brief:{story.id}", session, generate_brief, session, story, orchestrator)
        if brief is None:
            continue
        report.briefs_generated += 1
        if brief.status == BriefStatus.APPROVED:
            report.briefs_published += 1
        else:
            report.briefs_in_review += 1

    # Finance trackers (funding rounds, AI-in-finance claims, optional EDGAR refresh).
    if finance:
        from newsren.finance.service import refresh_finance

        report.finance = (
            _step(report, "finance", session, refresh_finance, session, orchestrator, settings, network) or {}
        )

    # 5. Output: opt-in monitoring and low-risk personal briefings.
    if monitor:
        from newsren.agents.monitor import run_monitoring

        report.monitoring = _step(report, "monitoring", session, run_monitoring, session, orchestrator) or {}
    if briefings:
        from newsren.agents.briefing import run_briefings

        report.briefings = _step(report, "briefings", session, run_briefings, session, orchestrator) or {}

    session.flush()
    return report
