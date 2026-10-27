"""Pydantic contracts for agent inputs/outputs.

These models are the structured-output schemas sent to Claude (via
``client.beta.messages.parse(output_format=...)``) and the shapes the offline
provider must return. They are also what gets stored in JSON columns.

Conventions:
* ``citation_ids`` reference ``CitationRef.id`` values ("c1", "c2", ...) from
  the evidence pack given to the agent. Agents may only cite what they were
  given; editorial checks reject unknown ids.
* Keep fields required-with-defaults so offline builders stay terse.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


class CitationRef(BaseModel):
    id: str = Field(description='Stable id within this item, e.g. "c1".')
    url: str
    title: str
    publisher: str = ""
    published_at: str | None = Field(default=None, description="ISO-8601 publication time, if known.")
    retrieved_at: str | None = Field(default=None, description="ISO-8601 time The Newsren retrieved it.")
    article_id: int | None = None
    snippet: str = Field(default="", description="Short quote/extract supporting the claim.")
    trust_tier: str = "independent"
    doc_type: str = "news"


class EvidenceDoc(BaseModel):
    """One source document handed to an agent (the 'evidence pack')."""

    citation: CitationRef
    text: str = Field(description="Title + summary + (truncated) body text.")


# --------------------------------------------------------------------------
# AI Story Brief (Stage 1)
# --------------------------------------------------------------------------

ClaimTypeLit = Literal["fact", "interpretation", "forecast", "performance", "opinion"]
SupportLit = Literal["supported", "corroborated", "contested", "unverified"]
PerformanceLit = Literal["backtest", "simulated", "live"]
RiskLit = Literal["low", "medium", "high"]


class ClaimOut(BaseModel):
    text: str
    claim_type: ClaimTypeLit = "fact"
    support_status: SupportLit = "supported"
    citation_ids: list[str] = Field(default_factory=list)
    performance_label: PerformanceLit | None = Field(
        default=None, description="Required when claim_type is 'performance'."
    )
    performance_period: str | None = None
    benchmark: str | None = None
    high_impact: bool = Field(default=False, description="Claim could materially affect readers' decisions.")
    notes: str | None = None


class SectionOut(BaseModel):
    title: str
    body: str
    citation_ids: list[str] = Field(default_factory=list)


class DisagreementPosition(BaseModel):
    position: str
    citation_ids: list[str] = Field(default_factory=list)


class DisagreementOut(BaseModel):
    topic: str
    positions: list[DisagreementPosition] = Field(default_factory=list)


class BriefOut(BaseModel):
    headline: str
    summary: str = Field(description="One concise paragraph: what happened.")
    why_it_matters: str
    what_changed: str = Field(default="", description="What is new compared with earlier coverage.")
    key_points: list[str] = Field(default_factory=list)
    sections: list[SectionOut] = Field(default_factory=list)
    claims: list[ClaimOut] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)
    disagreements: list[DisagreementOut] = Field(default_factory=list)
    watch_next: list[str] = Field(default_factory=list)
    risk_level: RiskLit = "low"


# --------------------------------------------------------------------------
# Evidence and Bias Inspection (2.5)
# --------------------------------------------------------------------------


class SourceRole(BaseModel):
    citation_id: str
    role: Literal["original", "secondary", "vendor_claim", "independent_evaluation", "commentary"]
    note: str = ""


class ClaimComparison(BaseModel):
    claim: str
    agreeing_citation_ids: list[str] = Field(default_factory=list)
    disagreeing_citation_ids: list[str] = Field(default_factory=list)
    status: SupportLit = "unverified"
    note: str = ""


class StatementOut(BaseModel):
    text: str
    citation_ids: list[str] = Field(default_factory=list)


class EvidenceReportOut(BaseModel):
    original_sources: list[SourceRole] = Field(default_factory=list)
    claim_comparisons: list[ClaimComparison] = Field(default_factory=list)
    facts: list[StatementOut] = Field(default_factory=list)
    interpretations: list[StatementOut] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    framing_notes: list[str] = Field(
        default_factory=list,
        description=(
            "How sources frame the story differently. Describe differences in emphasis and "
            "evidence; never label a source 'biased' merely for holding a different view."
        ),
    )
    study_design: StudyDesignOut | None = None


class StudyDesignOut(BaseModel):
    """Used by the Education Evidence Checker / research inspection."""

    design: str = ""
    participants: str = ""
    outcome_measures: list[str] = Field(default_factory=list)
    comparison_groups: str = ""
    limitations: list[str] = Field(default_factory=list)
    peer_reviewed: bool | None = None
    measures_transfer: bool | None = Field(
        default=None,
        description="Does the study measure transfer to unfamiliar problems (vs. speed/completion)?",
    )


EvidenceReportOut.model_rebuild()


# --------------------------------------------------------------------------
# Ask Newsren research agent (Stage 2)
# --------------------------------------------------------------------------


class ResearchAnswerOut(BaseModel):
    answer: str
    plan: list[str] = Field(default_factory=list, description="Research steps the agent took.")
    verified: list[StatementOut] = Field(default_factory=list)
    could_not_verify: list[str] = Field(default_factory=list)
    citations: list[CitationRef] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None


# --------------------------------------------------------------------------
# News-to-Learning (2.3)
# --------------------------------------------------------------------------


class GlossaryItem(BaseModel):
    term: str
    definition: str


class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    answer_index: int
    explanation: str = ""
    kind: Literal["comprehension", "transfer"] = "comprehension"


class ClassroomActivity(BaseModel):
    title: str
    duration_minutes: int = 15
    steps: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)


class LearningOut(BaseModel):
    audience: Literal["student", "educator", "professional"]
    reading_time_minutes: int = 5
    title: str
    explainer: str = Field(description="Main explanation adapted to the audience.")
    key_concepts: list[GlossaryItem] = Field(default_factory=list)
    glossary: list[GlossaryItem] = Field(default_factory=list)
    quiz: list[QuizQuestion] = Field(default_factory=list)
    discussion_questions: list[str] = Field(default_factory=list)
    classroom_activities: list[ClassroomActivity] = Field(default_factory=list)
    professional_implications: list[str] = Field(default_factory=list)
    advanced_reading: str = Field(default="", description="Deeper technical reading for advanced learners.")
    citation_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Industry Impact (2.4) and vertical editorial products
# --------------------------------------------------------------------------


class EffectOut(BaseModel):
    description: str
    evidence_status: Literal["supported", "speculative", "scenario"] = "speculative"
    citation_ids: list[str] = Field(default_factory=list)
    time_horizon: str = ""


class StakeholderOut(BaseModel):
    name: str
    kind: Literal["industry", "occupation", "decision", "company", "public"] = "industry"
    effects: list[EffectOut] = Field(default_factory=list)


class ImpactOut(BaseModel):
    question: str
    summary: str
    stakeholders: list[StakeholderOut] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list, description="What professionals should investigate.")
    caveats: list[str] = Field(default_factory=list)


class ProductSection(BaseModel):
    heading: str
    body: str
    label: str = Field(default="", description='Evidence label, e.g. "supported", "hypothesis", "projection".')
    citation_ids: list[str] = Field(default_factory=list)


class ProductOut(BaseModel):
    """Generic output of a vertical editorial product (Research Paper Decoder, ...)."""

    product: str
    title: str
    summary: str
    sections: list[ProductSection] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list, description='e.g. "preprint", "projection", "backtest".')
    caveats: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Story Intelligence (2.2)
# --------------------------------------------------------------------------


class TimelineEventOut(BaseModel):
    article_id: int
    event_type: str
    title: str
    description: str = ""


class StoryUpdateOut(BaseModel):
    title: str
    what_happened_first: str
    whats_new: str
    disputed: list[str] = Field(default_factory=list)
    evidence_changes: list[str] = Field(default_factory=list)
    events: list[TimelineEventOut] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Follow This Story / My Newsren (2.1, 2.6)
# --------------------------------------------------------------------------


class FollowCriteria(BaseModel):
    """Reader-chosen notification criteria for a follow."""

    event_types: list[str] = Field(
        default_factory=list, description="Only alert on these EventType values (empty = any material event)."
    )
    exclude_doc_types: list[str] = Field(default_factory=lambda: ["opinion"])
    require_independent: bool = Field(default=False, description="Require a non-vendor source.")
    require_primary: bool = False
    custom: str = Field(default="", description="Free-text condition, checked by the monitoring agent.")


class MonitorDecisionOut(BaseModel):
    matches: bool
    event_type: str = "other"
    is_material: bool = False
    is_opinion: bool = False
    criteria_met: list[str] = Field(default_factory=list)
    reason: str = ""


class BriefingItem(BaseModel):
    headline: str
    what_changed: str
    why_it_matters: str
    citation_ids: list[str] = Field(default_factory=list)
    story_id: int | None = None
    brief_id: int | None = None
    interest: str = ""


class BriefingSection(BaseModel):
    heading: str
    items: list[BriefingItem] = Field(default_factory=list)


class BriefingOut(BaseModel):
    title: str
    intro: str
    sections: list[BriefingSection] = Field(default_factory=list)
    closing_note: str = ""


# --------------------------------------------------------------------------
# Finance
# --------------------------------------------------------------------------


class EarningsMetric(BaseModel):
    name: str
    value: float | None
    unit: str = "USD"
    period: str = ""
    prior_value: float | None = None
    prior_period: str = ""
    change_pct: float | None = None
    xbrl_tag: str = ""
    source_url: str = ""
    source_quote: str = Field(default="", description="Quote/locator in the filing for this number.")


class GuidanceItem(BaseModel):
    metric: str
    guidance: str
    actual: str = ""
    comparison: Literal["above", "within", "below", "not_comparable", "unknown"] = "unknown"
    source_url: str = ""
    source_quote: str = ""


class FundingExtractOut(BaseModel):
    is_funding_round: bool
    company_name: str = ""
    sector: str = ""
    stage: str = "undisclosed"
    amount_text: str = ""
    amount_usd: float | None = None
    investors: list[str] = Field(default_factory=list)
    lead_investors: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class AIFinanceClaimOut(BaseModel):
    is_ai_finance_claim: bool
    firm: str = ""
    area: Literal["trading_markets", "asset_wealth", "vc_pe", "founders", "regulation", "other"] = "other"
    claim_text: str = ""
    classification: Literal["announcement", "pilot", "backtest", "live_deployment", "regulatory_finding"] = (
        "announcement"
    )
    performance_label: PerformanceLit | None = None
    period: str = ""
    benchmark: str = ""
    costs_assumed: str = ""
    authors_caveats: str = ""
    evidence_note: str = ""


class CheckResult(BaseModel):
    """One automated editorial quality-gate check."""

    name: str
    passed: bool
    severity: Literal["info", "warning", "blocker"] = "warning"
    message: str = ""
    details: list[str] = Field(default_factory=list)
