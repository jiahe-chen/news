"""Builds hand-made briefs over the fictional sample articles (shared by the Section 5 tests).

Loaded by the test modules with ``importlib`` so the briefs do not depend on
``generate_brief``: each brief's citations are ``CitationRef`` dicts pointing
at articles created here, and its claims are ``BriefClaim`` rows.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from newsren.agents.base import citation_for
from newsren.intelligence.events import classify_event
from newsren.models import Article, Brief, BriefClaim, Source, Story, StoryEvent, utcnow

DATA = Path(__file__).parent
ARTICLES: dict[str, dict[str, Any]] = json.loads((DATA / "articles.json").read_text())["articles"]
BRIEFS: dict[str, dict[str, Any]] = json.loads((DATA / "briefs.json").read_text())["briefs"]


class Newsroom:
    """Creates sources, articles, stories, and briefs from the sample files."""

    def __init__(self, factory: Any):
        self.factory = factory
        self.session = factory.session
        self.sources: dict[str, Source] = {}
        self.articles: dict[str, Article] = {}

    def article(self, key: str) -> Article:
        if key not in self.articles:
            spec = ARTICLES[key]
            source = self.sources.get(spec["publisher"])
            if source is None:
                source = self.factory.source(
                    vertical=spec["vertical"],
                    trust_tier=spec["trust_tier"],
                    doc_type=spec["doc_type"],
                    name=spec["publisher"],
                )
                self.sources[spec["publisher"]] = source
            self.articles[key] = self.factory.article(
                spec["title"],
                spec["body"],
                source=source,
                vertical=spec["vertical"],
                doc_type=spec["doc_type"],
                trust_tier=spec["trust_tier"],
                summary=spec["summary"],
                published_at=utcnow() - timedelta(days=spec["days_ago"], hours=len(self.articles)),
            )
        return self.articles[key]

    def brief(self, name: str, **overrides: Any) -> Brief:
        spec = {**BRIEFS[name], **overrides}
        articles = [self.article(key) for key in spec["articles"]]
        story = None
        if spec.get("story"):
            story = Story(title=spec["headline"], vertical=spec["vertical"])
            self.session.add(story)
            for article in articles:
                article.story = story
                event_type = classify_event(article)
                article.event_type = event_type
                story.events.append(
                    StoryEvent(
                        article=article,
                        occurred_at=article.published_at,
                        event_type=event_type,
                        title=article.title,
                        description=article.summary,
                    )
                )
        brief = Brief(
            story=story,
            vertical=spec["vertical"],
            kind=spec.get("kind", "story"),
            headline=spec["headline"],
            summary=spec["summary"],
            why_it_matters=spec.get("why_it_matters"),
            sections=spec.get("sections", []),
            uncertainty=spec.get("uncertainty", []),
            disagreements=spec.get("disagreements", []),
            watch_next=spec.get("watch_next", []),
            citations=[citation_for(a, f"c{i}").model_dump() for i, a in enumerate(articles, 1)],
        )
        brief.claims = [
            BriefClaim(position=i, **{"claim_type": "fact", "support_status": "supported", **claim})
            for i, claim in enumerate(spec["claims"])
        ]
        self.session.add(brief)
        self.session.flush()
        return brief
