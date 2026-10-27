"""Database schema for The Newsren.

Every generated item keeps a record of its sources (``citations`` JSON in the
``CitationRef`` shape from ``newsren.schemas``), publication timestamps,
retrieval time, and the evidence supporting its main claims.

Enumerations are stored as plain strings; the allowed values live in the
``StrEnum`` classes below so code and templates share one vocabulary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """Naive UTC timestamp (SQLite has no timezone type; we store UTC everywhere)."""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


# --------------------------------------------------------------------------
# Vocabularies
# --------------------------------------------------------------------------


class Vertical(StrEnum):
    TECHNOLOGY = "technology"
    EDUCATION = "education"
    INDUSTRY = "industry"
    FINANCE = "finance"


VERTICAL_LABELS = {
    Vertical.TECHNOLOGY: "Technology",
    Vertical.EDUCATION: "Education",
    Vertical.INDUSTRY: "Industry",
    Vertical.FINANCE: "Finance",
}


class SourceKind(StrEnum):
    RSS = "rss"
    ARXIV = "arxiv"
    EDGAR = "edgar"
    ERIC = "eric"
    MANUAL = "manual"
    DEMO = "demo"


class TrustTier(StrEnum):
    PRIMARY = "primary"  # the original record: regulator, filing, paper, official release
    INDEPENDENT = "independent"  # independent reporting or third-party evaluation
    VENDOR = "vendor"  # company marketing / press office describing its own product
    COMMENTARY = "commentary"  # opinion, columns, analysis pieces


class DocType(StrEnum):
    NEWS = "news"
    PRESS_RELEASE = "press_release"
    RESEARCH_PREPRINT = "research_preprint"
    RESEARCH_PEER_REVIEWED = "research_peer_reviewed"
    FILING = "filing"
    EARNINGS_CALL = "earnings_call"
    REGULATORY = "regulatory"
    POLICY = "policy"
    EDUCATION_REPORT = "education_report"
    ADVISORY = "advisory"
    OPINION = "opinion"
    BLOG = "blog"


class EventType(StrEnum):
    ANNOUNCEMENT = "announcement"
    RESEARCH_PUBLICATION = "research_publication"
    INDEPENDENT_EVALUATION = "independent_evaluation"
    BENCHMARK = "benchmark"
    PRODUCT_LAUNCH = "product_launch"
    ADOPTION = "adoption"
    REGULATION = "regulation"
    POLICY = "policy"
    FUNDING = "funding"
    EARNINGS = "earnings"
    FILING = "filing"
    ENFORCEMENT = "enforcement"
    CORRECTION = "correction"
    SECURITY_ADVISORY = "security_advisory"
    OPINION = "opinion"
    ANALYSIS = "analysis"
    OTHER = "other"


class Audience(StrEnum):
    STUDENT = "student"
    EDUCATOR = "educator"
    PROFESSIONAL = "professional"


class Depth(StrEnum):
    INTRO = "intro"
    STANDARD = "standard"
    TECHNICAL = "technical"


class BriefStatus(StrEnum):
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"  # approved and published
    REJECTED = "rejected"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ClaimType(StrEnum):
    FACT = "fact"
    INTERPRETATION = "interpretation"
    FORECAST = "forecast"
    PERFORMANCE = "performance"  # a performance/result claim (must carry a label)
    OPINION = "opinion"


class SupportStatus(StrEnum):
    SUPPORTED = "supported"  # backed by at least one cited source
    CORROBORATED = "corroborated"  # backed by two or more independent sources
    CONTESTED = "contested"  # credible sources disagree
    UNVERIFIED = "unverified"  # no source in hand confirms it


class PerformanceLabel(StrEnum):
    BACKTEST = "backtest"
    SIMULATED = "simulated"
    LIVE = "live"


class AIFinanceClass(StrEnum):
    ANNOUNCEMENT = "announcement"
    PILOT = "pilot"
    BACKTEST = "backtest"
    LIVE_DEPLOYMENT = "live_deployment"
    REGULATORY_FINDING = "regulatory_finding"


class Role(StrEnum):
    READER = "reader"
    EDITOR = "editor"
    ADMIN = "admin"


class AgentRunStatus(StrEnum):
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"
    BUDGET_EXCEEDED = "budget_exceeded"
    REFUSED = "refused"
    FALLBACK = "fallback"  # Claude unavailable/refused; offline heuristic answered


# --------------------------------------------------------------------------
# Sources and articles (Discover)
# --------------------------------------------------------------------------


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(80), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(20), default=SourceKind.RSS)
    url: Mapped[str] = mapped_column(String(1000))
    homepage: Mapped[str | None] = mapped_column(String(500))
    vertical: Mapped[str] = mapped_column(String(20))
    trust_tier: Mapped[str] = mapped_column(String(20), default=TrustTier.INDEPENDENT)
    default_doc_type: Mapped[str] = mapped_column(String(40), default=DocType.NEWS)
    # Approval is the gate for agent access: agents only read approved sources.
    approved: Mapped[bool] = mapped_column(Boolean, default=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    license_note: Mapped[str | None] = mapped_column(Text)
    data_delay_note: Mapped[str | None] = mapped_column(String(300))
    last_fetched_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_status: Mapped[str | None] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    articles: Mapped[list[Article]] = relationship(back_populates="source")


class Article(Base):
    __tablename__ = "articles"
    __table_args__ = (
        Index("ix_articles_vertical_published", "vertical", "published_at"),
        Index("ix_articles_story", "story_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    url: Mapped[str] = mapped_column(String(1000))
    canonical_url: Mapped[str] = mapped_column(String(1000), unique=True)
    title: Mapped[str] = mapped_column(String(500))
    author: Mapped[str | None] = mapped_column(String(300))
    summary: Mapped[str | None] = mapped_column(Text)  # publisher-provided abstract/dek
    content: Mapped[str | None] = mapped_column(Text)  # plain text body when licensed/available
    published_at: Mapped[datetime | None] = mapped_column(DateTime)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    vertical: Mapped[str] = mapped_column(String(20))
    doc_type: Mapped[str] = mapped_column(String(40), default=DocType.NEWS)
    trust_tier: Mapped[str] = mapped_column(String(20), default=TrustTier.INDEPENDENT)
    language: Mapped[str] = mapped_column(String(10), default="en")
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    simhash: Mapped[str] = mapped_column(String(16), default="0" * 16)  # 64-bit hex
    duplicate_of_id: Mapped[int | None] = mapped_column(ForeignKey("articles.id"))
    story_id: Mapped[int | None] = mapped_column(ForeignKey("stories.id"))
    event_type: Mapped[str | None] = mapped_column(String(40))
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    # Identifiers such as arxiv_id, doi, cik, accession, ticker, eric_id.
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    source: Mapped[Source] = relationship(back_populates="articles")
    story: Mapped[Story | None] = relationship(back_populates="articles")
    entities: Mapped[list[ArticleEntity]] = relationship(back_populates="article", cascade="all, delete-orphan")

    @property
    def is_duplicate(self) -> bool:
        return self.duplicate_of_id is not None

    @property
    def display_time(self) -> datetime:
        return self.published_at or self.retrieved_at


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (UniqueConstraint("normalized", "kind"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    normalized: Mapped[str] = mapped_column(String(200), index=True)
    # organization | person | product | policy | place | concept | ticker
    kind: Mapped[str] = mapped_column(String(30), default="organization")

    mentions: Mapped[list[ArticleEntity]] = relationship(back_populates="entity")


class ArticleEntity(Base):
    __tablename__ = "article_entities"

    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id"), primary_key=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("entities.id"), primary_key=True)
    salience: Mapped[float] = mapped_column(Float, default=1.0)

    article: Mapped[Article] = relationship(back_populates="entities")
    entity: Mapped[Entity] = relationship(back_populates="mentions")


# --------------------------------------------------------------------------
# Story Intelligence (Connect)
# --------------------------------------------------------------------------


class Story(Base):
    __tablename__ = "stories"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    vertical: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), default="developing")  # developing|active|dormant
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    # Story-intelligence summary: what happened first, what is new, what is
    # disputed, and how the evidence changed. Refreshed when articles join.
    what_happened_first: Mapped[str | None] = mapped_column(Text)
    whats_new: Mapped[str | None] = mapped_column(Text)
    disputed: Mapped[list[Any]] = mapped_column(JSON, default=list)
    evidence_changes: Mapped[list[Any]] = mapped_column(JSON, default=list)
    key_entities: Mapped[list[Any]] = mapped_column(JSON, default=list)
    keywords: Mapped[list[Any]] = mapped_column(JSON, default=list)
    version: Mapped[int] = mapped_column(Integer, default=1)
    # Fingerprint of member article ids at last refresh (change detection).
    member_fingerprint: Mapped[str] = mapped_column(String(64), default="")
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)

    articles: Mapped[list[Article]] = relationship(back_populates="story")
    events: Mapped[list[StoryEvent]] = relationship(
        back_populates="story", cascade="all, delete-orphan", order_by="StoryEvent.occurred_at"
    )
    briefs: Mapped[list[Brief]] = relationship(back_populates="story")


class StoryEvent(Base):
    __tablename__ = "story_events"
    __table_args__ = (UniqueConstraint("story_id", "article_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    story_id: Mapped[int] = mapped_column(ForeignKey("stories.id"))
    article_id: Mapped[int | None] = mapped_column(ForeignKey("articles.id"))
    occurred_at: Mapped[datetime] = mapped_column(DateTime)
    event_type: Mapped[str] = mapped_column(String(40), default=EventType.OTHER)
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(Text)
    is_correction: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    story: Mapped[Story] = relationship(back_populates="events")
    article: Mapped[Article | None] = relationship()


# --------------------------------------------------------------------------
# AI Story Brief + editorial quality gate (Explain, Verify, Review)
# --------------------------------------------------------------------------


class Brief(Base):
    __tablename__ = "briefs"

    id: Mapped[int] = mapped_column(primary_key=True)
    story_id: Mapped[int | None] = mapped_column(ForeignKey("stories.id"))
    article_id: Mapped[int | None] = mapped_column(ForeignKey("articles.id"))
    vertical: Mapped[str] = mapped_column(String(20))
    # "story" (AI Story Brief) | "earnings" | "regulation" | ... (product that produced it)
    kind: Mapped[str] = mapped_column(String(40), default="story")
    headline: Mapped[str] = mapped_column(String(500))
    summary: Mapped[str] = mapped_column(Text)
    why_it_matters: Mapped[str | None] = mapped_column(Text)
    what_changed: Mapped[str | None] = mapped_column(Text)
    key_points: Mapped[list[Any]] = mapped_column(JSON, default=list)
    # Expandable sections, e.g. What happened? / What is technically new? /
    # What evidence exists? / What should developers watch next?
    sections: Mapped[list[Any]] = mapped_column(JSON, default=list)
    uncertainty: Mapped[list[Any]] = mapped_column(JSON, default=list)
    disagreements: Mapped[list[Any]] = mapped_column(JSON, default=list)
    watch_next: Mapped[list[Any]] = mapped_column(JSON, default=list)
    citations: Mapped[list[Any]] = mapped_column(JSON, default=list)  # list[CitationRef]
    status: Mapped[str] = mapped_column(String(20), default=BriefStatus.DRAFT, index=True)
    risk_level: Mapped[str] = mapped_column(String(10), default=RiskLevel.LOW)
    requires_human_review: Mapped[bool] = mapped_column(Boolean, default=True)
    check_results: Mapped[list[Any]] = mapped_column(JSON, default=list)  # list[CheckResult]
    generated_by: Mapped[str] = mapped_column(String(80), default="offline")
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    published_at: Mapped[datetime | None] = mapped_column(DateTime)
    version: Mapped[int] = mapped_column(Integer, default=1)
    reviewer_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    review_notes: Mapped[str | None] = mapped_column(Text)
    review_started_at: Mapped[datetime | None] = mapped_column(DateTime)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime)
    agent_run_id: Mapped[int | None] = mapped_column(ForeignKey("agent_runs.id"))
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)

    story: Mapped[Story | None] = relationship(back_populates="briefs")
    article: Mapped[Article | None] = relationship()
    claims: Mapped[list[BriefClaim]] = relationship(
        back_populates="brief", cascade="all, delete-orphan", order_by="BriefClaim.position"
    )

    @property
    def is_published(self) -> bool:
        return self.status == BriefStatus.APPROVED


class BriefClaim(Base):
    __tablename__ = "brief_claims"

    id: Mapped[int] = mapped_column(primary_key=True)
    brief_id: Mapped[int] = mapped_column(ForeignKey("briefs.id"))
    position: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    claim_type: Mapped[str] = mapped_column(String(20), default=ClaimType.FACT)
    support_status: Mapped[str] = mapped_column(String(20), default=SupportStatus.UNVERIFIED)
    citation_ids: Mapped[list[Any]] = mapped_column(JSON, default=list)  # CitationRef.id values
    # Required for performance claims (finance guardrail): backtest/simulated/live.
    performance_label: Mapped[str | None] = mapped_column(String(20))
    performance_period: Mapped[str | None] = mapped_column(String(120))
    benchmark: Mapped[str | None] = mapped_column(String(200))
    notes: Mapped[str | None] = mapped_column(Text)
    high_impact: Mapped[bool] = mapped_column(Boolean, default=False)
    # Trust metric: editors sample claims and record whether the linked
    # sources actually support them.
    editor_verdict: Mapped[str | None] = mapped_column(String(20))  # accurate|inaccurate
    verdict_at: Mapped[datetime | None] = mapped_column(DateTime)

    brief: Mapped[Brief] = relationship(back_populates="claims")


class EditorialAction(Base):
    __tablename__ = "editorial_actions"

    id: Mapped[int] = mapped_column(primary_key=True)
    brief_id: Mapped[int] = mapped_column(ForeignKey("briefs.id"), index=True)
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    # submit | approve | reject | correct | claim_verdict | regenerate | auto_check
    action: Mapped[str] = mapped_column(String(30))
    notes: Mapped[str | None] = mapped_column(Text)
    duration_seconds: Mapped[float | None] = mapped_column(Float)  # review time
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Correction(Base):
    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(primary_key=True)
    brief_id: Mapped[int] = mapped_column(ForeignKey("briefs.id"), index=True)
    description: Mapped[str] = mapped_column(Text)
    material: Mapped[bool] = mapped_column(Boolean, default=False)
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class GeneratedContent(Base):
    """Cached output of an explanation/analysis agent or editorial product.

    kind examples: "learning", "impact", "evidence", "product:research_paper_decoder".
    """

    __tablename__ = "generated_content"
    __table_args__ = (Index("ix_generated_lookup", "kind", "cache_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(80))
    cache_key: Mapped[str] = mapped_column(String(64))
    brief_id: Mapped[int | None] = mapped_column(ForeignKey("briefs.id"))
    story_id: Mapped[int | None] = mapped_column(ForeignKey("stories.id"))
    article_id: Mapped[int | None] = mapped_column(ForeignKey("articles.id"))
    audience: Mapped[str | None] = mapped_column(String(20))
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    content: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    citations: Mapped[list[Any]] = mapped_column(JSON, default=list)
    generated_by: Mapped[str] = mapped_column(String(80), default="offline")
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    agent_run_id: Mapped[int | None] = mapped_column(ForeignKey("agent_runs.id"))


# --------------------------------------------------------------------------
# Readers, personalization, monitoring (Output, Follow)
# --------------------------------------------------------------------------


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    display_name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(300))
    role: Mapped[str] = mapped_column(String(20), default=Role.READER)
    audience: Mapped[str] = mapped_column(String(20), default=Audience.PROFESSIONAL)
    depth: Mapped[str] = mapped_column(String(20), default=Depth.STANDARD)
    email_opt_in: Mapped[bool] = mapped_column(Boolean, default=False)
    # daily | weekly | when_meaningful
    briefing_frequency: Mapped[str] = mapped_column(String(20), default="when_meaningful")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_briefing_at: Mapped[datetime | None] = mapped_column(DateTime)

    interests: Mapped[list[Interest]] = relationship(back_populates="user", cascade="all, delete-orphan")
    follows: Mapped[list[Follow]] = relationship(back_populates="user", cascade="all, delete-orphan")

    @property
    def is_editor(self) -> bool:
        return self.role in (Role.EDITOR, Role.ADMIN)


class Interest(Base):
    __tablename__ = "interests"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    # topic | question | organization | sector
    kind: Mapped[str] = mapped_column(String(20), default="topic")
    value: Mapped[str] = mapped_column(String(500))
    vertical: Mapped[str | None] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    user: Mapped[User] = relationship(back_populates="interests")


class Follow(Base):
    """A reader's request to follow a question, story, company, or sector."""

    __tablename__ = "follows"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    # topic | story | company | sector
    target_type: Mapped[str] = mapped_column(String(20), default="topic")
    story_id: Mapped[int | None] = mapped_column(ForeignKey("stories.id"))
    entity_name: Mapped[str | None] = mapped_column(String(200))
    vertical: Mapped[str | None] = mapped_column(String(20))
    keywords: Mapped[list[Any]] = mapped_column(JSON, default=list)
    # FollowCriteria (newsren.schemas) as JSON.
    criteria: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime)
    check_count: Mapped[int] = mapped_column(Integer, default=0)

    user: Mapped[User] = relationship(back_populates="follows")
    story: Mapped[Story | None] = relationship()
    alerts: Mapped[list[Alert]] = relationship(back_populates="follow", cascade="all, delete-orphan")


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (UniqueConstraint("follow_id", "dedup_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    follow_id: Mapped[int] = mapped_column(ForeignKey("follows.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    article_id: Mapped[int | None] = mapped_column(ForeignKey("articles.id"))
    story_id: Mapped[int | None] = mapped_column(ForeignKey("stories.id"))
    event_type: Mapped[str | None] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(String(500))
    reason: Mapped[str] = mapped_column(Text)
    matched_criteria: Mapped[list[Any]] = mapped_column(JSON, default=list)
    citations: Mapped[list[Any]] = mapped_column(JSON, default=list)
    # Prevents repeated alerts for the same development (story event or
    # near-duplicate article).
    dedup_key: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    read_at: Mapped[datetime | None] = mapped_column(DateTime)
    emailed_at: Mapped[datetime | None] = mapped_column(DateTime)
    feedback: Mapped[str | None] = mapped_column(String(20))  # useful | unwanted

    follow: Mapped[Follow] = relationship(back_populates="alerts")
    article: Mapped[Article | None] = relationship()


class Briefing(Base):
    """A personalized My Newsren briefing."""

    __tablename__ = "briefings"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(300))
    period_start: Mapped[datetime | None] = mapped_column(DateTime)
    period_end: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    content: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # BriefingOut
    citations: Mapped[list[Any]] = mapped_column(JSON, default=list)
    item_count: Mapped[int] = mapped_column(Integer, default=0)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    generated_by: Mapped[str] = mapped_column(String(80), default="offline")
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime)
    delivery_channel: Mapped[str | None] = mapped_column(String(20))  # web | email
    opened_at: Mapped[datetime | None] = mapped_column(DateTime)
    agent_run_id: Mapped[int | None] = mapped_column(ForeignKey("agent_runs.id"))


class Question(Base):
    """An Ask Newsren research request and its sourced answer."""

    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    session_key: Mapped[str | None] = mapped_column(String(64), index=True)
    brief_id: Mapped[int | None] = mapped_column(ForeignKey("briefs.id"))
    story_id: Mapped[int | None] = mapped_column(ForeignKey("stories.id"))
    article_id: Mapped[int | None] = mapped_column(ForeignKey("articles.id"))
    audience: Mapped[str | None] = mapped_column(String(20))
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str | None] = mapped_column(Text)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # ResearchAnswerOut
    citations: Mapped[list[Any]] = mapped_column(JSON, default=list)
    tool_trace: Mapped[list[Any]] = mapped_column(JSON, default=list)
    refused: Mapped[bool] = mapped_column(Boolean, default=False)
    agent_run_id: Mapped[int | None] = mapped_column(ForeignKey("agent_runs.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Feedback(Base):
    """Reader ratings that feed the usefulness metric."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    # briefing | answer | brief | learning | impact | evidence
    target_type: Mapped[str] = mapped_column(String(20))
    target_id: Mapped[int] = mapped_column(Integer)
    answered_question: Mapped[bool | None] = mapped_column(Boolean)
    saved_time: Mapped[bool | None] = mapped_column(Boolean)
    rating: Mapped[int | None] = mapped_column(Integer)  # 1..5
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class QuizAttempt(Base):
    """Comprehension + transfer results (education metric)."""

    __tablename__ = "quiz_attempts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    content_id: Mapped[int] = mapped_column(ForeignKey("generated_content.id"))
    # "newsren" (read the adapted explanation) or "summary" (read a plain summary)
    condition: Mapped[str] = mapped_column(String(20), default="newsren")
    comprehension_correct: Mapped[int] = mapped_column(Integer, default=0)
    comprehension_total: Mapped[int] = mapped_column(Integer, default=0)
    transfer_correct: Mapped[int] = mapped_column(Integer, default=0)
    transfer_total: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class PageView(Base):
    """Minimal privacy-preserving analytics (return visits to followed stories)."""

    __tablename__ = "page_views"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    path: Mapped[str] = mapped_column(String(300))
    story_id: Mapped[int | None] = mapped_column(ForeignKey("stories.id"))
    brief_id: Mapped[int | None] = mapped_column(ForeignKey("briefs.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# --------------------------------------------------------------------------
# Orchestrator accounting
# --------------------------------------------------------------------------


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    agent: Mapped[str] = mapped_column(String(80), index=True)
    task: Mapped[str] = mapped_column(String(300))
    status: Mapped[str] = mapped_column(String(20), default=AgentRunStatus.RUNNING)
    provider: Mapped[str] = mapped_column(String(20), default="offline")
    model: Mapped[str | None] = mapped_column(String(80))
    subject_type: Mapped[str | None] = mapped_column(String(30))
    subject_id: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0)
    retrieval_requests: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    budget: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    trace: Mapped[list[Any]] = mapped_column(JSON, default=list)
    error: Mapped[str | None] = mapped_column(Text)


# --------------------------------------------------------------------------
# Finance section
# --------------------------------------------------------------------------


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    ticker: Mapped[str | None] = mapped_column(String(12), unique=True)
    cik: Mapped[str | None] = mapped_column(String(10), unique=True)  # zero-padded 10 digits
    sector: Mapped[str | None] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class EarningsReport(Base):
    """Earnings Decoder output: reported figures vs prior periods and guidance.

    Every figure carries the filing/transcript it was read from.
    """

    __tablename__ = "earnings_reports"
    __table_args__ = (UniqueConstraint("company_id", "fiscal_period", "form"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"))
    fiscal_period: Mapped[str] = mapped_column(String(20))  # e.g. FY2026Q2
    form: Mapped[str] = mapped_column(String(10))  # 10-Q | 10-K | 8-K
    filed_at: Mapped[datetime | None] = mapped_column(DateTime)
    accession: Mapped[str | None] = mapped_column(String(30))
    filing_url: Mapped[str | None] = mapped_column(String(500))
    # list[EarningsMetric]
    metrics: Mapped[list[Any]] = mapped_column(JSON, default=list)
    # list[GuidanceItem]
    guidance: Mapped[list[Any]] = mapped_column(JSON, default=list)
    narrative: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    data_delay_note: Mapped[str | None] = mapped_column(String(300))
    brief_id: Mapped[int | None] = mapped_column(ForeignKey("briefs.id"))
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)

    company: Mapped[Company] = relationship()
    brief: Mapped[Brief | None] = relationship()


class FundingRound(Base):
    __tablename__ = "funding_rounds"

    id: Mapped[int] = mapped_column(primary_key=True)
    company_name: Mapped[str] = mapped_column(String(200))
    sector: Mapped[str | None] = mapped_column(String(120))
    # pre_seed | seed | series_a | series_b | series_c | series_d_plus | growth | debt | grant | other | undisclosed
    stage: Mapped[str] = mapped_column(String(30), default="undisclosed")
    amount_usd: Mapped[float | None] = mapped_column(Float)
    amount_text: Mapped[str | None] = mapped_column(String(100))  # as reported, e.g. "€20 million"
    investors: Mapped[list[Any]] = mapped_column(JSON, default=list)
    lead_investors: Mapped[list[Any]] = mapped_column(JSON, default=list)
    announced_at: Mapped[datetime | None] = mapped_column(DateTime)
    article_id: Mapped[int | None] = mapped_column(ForeignKey("articles.id"))
    source_url: Mapped[str] = mapped_column(String(1000))
    source_title: Mapped[str | None] = mapped_column(String(500))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    extraction_confidence: Mapped[float] = mapped_column(Float, default=0.5)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)  # editor-checked
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)


class AIFinanceClaim(Base):
    """AI-in-Finance Reality Check: one classified claim about AI in finance."""

    __tablename__ = "ai_finance_claims"

    id: Mapped[int] = mapped_column(primary_key=True)
    firm: Mapped[str] = mapped_column(String(200))
    # trading_markets | asset_wealth | vc_pe | founders | regulation | other
    area: Mapped[str] = mapped_column(String(30), default="other")
    claim_text: Mapped[str] = mapped_column(Text)
    classification: Mapped[str] = mapped_column(String(30))  # AIFinanceClass
    performance_label: Mapped[str | None] = mapped_column(String(20))  # PerformanceLabel
    period: Mapped[str | None] = mapped_column(String(120))
    benchmark: Mapped[str | None] = mapped_column(String(200))
    costs_assumed: Mapped[str | None] = mapped_column(String(300))
    authors_caveats: Mapped[str | None] = mapped_column(Text)
    evidence_note: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str] = mapped_column(String(1000))
    source_title: Mapped[str] = mapped_column(String(500))
    published_at: Mapped[datetime | None] = mapped_column(DateTime)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    article_id: Mapped[int | None] = mapped_column(ForeignKey("articles.id"))
    is_seed: Mapped[bool] = mapped_column(Boolean, default=False)  # from the plan's research table
