"""Evidence and Bias Inspection: source roles, claim comparisons, framing, study design, caching."""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsren import db
from newsren.agents.base import Orchestrator, build_evidence_pack
from newsren.agents.brief import generate_brief, source_docs
from newsren.agents.evidence import (
    AGENT_NAME,
    CONTENT_KIND,
    inspect_evidence,
    labels_source_biased,
    sanitize_report,
)
from newsren.config import Settings, set_settings
from newsren.editorial.workflow import record_correction
from newsren.ingest.fixtures import load_demo_data
from newsren.intelligence.clustering import assign_stories
from newsren.intelligence.entities import tag_article_entities
from newsren.intelligence.timeline import refresh_story
from newsren.llm import OfflineProvider, set_provider
from newsren.llm.claude import ClaudeProvider
from newsren.models import (
    AgentRun,
    Article,
    Brief,
    DocType,
    GeneratedContent,
    Story,
    TrustTier,
    User,
    Vertical,
    utcnow,
)
from newsren.schemas import (
    ClaimComparison,
    EvidenceReportOut,
    SourceRole,
    StatementOut,
)

CITE = re.compile(r"\[(c\d+)\]")
BIASED = re.compile(r"\bbias(ed)?\b", re.IGNORECASE)

LUMEN = "Lumen Labs launches Lumen-4"
HALBROOK = "Halbrook study: AI practice assistant"
CORVANE = "Corvane Semiconductor to add 40,000 wafers"
NORTHBRIDGE = "Northbridge researchers: AI portfolio agents"


@pytest.fixture(scope="module")
def story_db(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("evidence-stories")
    settings = Settings(data_dir=root, database_url=f"sqlite:///{root / 'stories.db'}", llm_mode="offline")
    set_settings(settings)
    set_provider(OfflineProvider())
    db.reset_engine()
    try:
        db.init_db()
        with db.session_scope() as session:
            load_demo_data(session)
            for article in session.scalars(select(Article).where(Article.duplicate_of_id.is_(None))):
                tag_article_entities(session, article)
            for story in assign_stories(session):
                refresh_story(session, story)
    finally:
        db.reset_engine()
        set_provider(None)
        set_settings(None)
    return root / "stories.db"


@pytest.fixture()
def stories(tmp_path, story_db) -> Iterator[Session]:
    target = tmp_path / "stories.db"
    shutil.copyfile(story_db, target)
    settings = Settings(data_dir=tmp_path, database_url=f"sqlite:///{target}", llm_mode="offline")
    set_settings(settings)
    set_provider(OfflineProvider())
    db.reset_engine()
    db.init_db()
    session = db.new_session()
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        db.reset_engine()
        set_provider(None)
        set_settings(None)


def brief_for(session: Session, title_start: str) -> Brief:
    article = session.scalars(select(Article).where(Article.title.startswith(title_start))).one()
    return generate_brief(session, article.story)


def ids_by_title(content: GeneratedContent) -> dict[str, str]:
    return {c["title"]: c["id"] for c in content.citations}


def cid_for(content: GeneratedContent, title_start: str) -> str:
    return next(cid for title, cid in ids_by_title(content).items() if title.startswith(title_start))


def report_of(content: GeneratedContent) -> EvidenceReportOut:
    return EvidenceReportOut.model_validate(content.content)


def all_ids(report: EvidenceReportOut) -> set[str]:
    ids = {r.citation_id for r in report.original_sources}
    for comp in report.claim_comparisons:
        ids |= set(comp.agreeing_citation_ids) | set(comp.disagreeing_citation_ids)
    for st in [*report.facts, *report.interpretations]:
        ids |= set(st.citation_ids)
    texts = [*report.uncertainties, *report.missing_evidence, *report.framing_notes]
    texts += [c.note for c in report.claim_comparisons]
    return ids | {cid for t in texts for cid in CITE.findall(t)}


# --------------------------------------------------------------------------
# The plan's example: a university press release versus the study's actual results
# --------------------------------------------------------------------------


def test_ai_learning_storyline_roles_and_study_design(stories):
    brief = brief_for(stories, HALBROOK)
    content = inspect_evidence(stories, brief)

    assert content.kind == CONTENT_KIND and content.brief_id == brief.id and content.story_id == brief.story_id
    assert content.generated_by == "offline" and content.params == {"brief_version": 1}
    report = report_of(content)
    assert all_ids(report) <= {c["id"] for c in content.citations}

    roles = {r.citation_id: r.role for r in report.original_sources}
    preprint = cid_for(content, "Effects of an AI practice assistant")
    release = cid_for(content, "Halbrook study")
    clarification = cid_for(content, "Clarification:")
    dispatch = cid_for(content, "AI tutor study")
    assert roles[preprint] == "original"
    assert roles[release] == "vendor_claim" and roles[clarification] == "vendor_claim"
    assert roles[dispatch] == "secondary"
    assert len(report.original_sources) == len(content.citations)

    design = report.study_design
    assert design is not None
    assert "randomized" in design.design.lower() and "eight weeks" in design.design
    assert "212" in design.participants
    assert any("transfer" in m for m in design.outcome_measures)
    assert any("time to complete" in m for m in design.outcome_measures)
    assert "without hints" in design.comparison_groups
    assert any("single district" in lim for lim in design.limitations)
    assert any("teachers knew" in lim.lower() for lim in design.limitations)
    assert design.peer_reviewed is False
    assert design.measures_transfer is True

    # The press release's headline claim is contested by the study and by independent reporting.
    contested = [c for c in report.claim_comparisons if c.status == "contested"]
    assert contested
    assert any(preprint in c.disagreeing_citation_ids for c in contested)
    assert all(release in c.agreeing_citation_ids or clarification in c.agreeing_citation_ids for c in contested)

    # Framing: emphasis differences, described with both sides cited, never a bias label.
    notes = report.framing_notes
    assert notes and not any(BIASED.search(n) for n in notes)
    speed = next(n for n in notes if "speed or completion" in n)
    assert f"[{release}]" in speed and f"[{preprint}]" in speed and "transfer" in speed
    assert any("equity" in n and f"[{release}]" in n for n in notes)

    facts = " ".join(f.text for f in report.facts)
    assert "22 minutes" in facts
    assert any("strong evidence" in i.text for i in report.interpretations)
    assert any("Peer review" in m for m in report.missing_evidence)
    assert any("preprint" in u for u in report.uncertainties)


def test_technology_storyline_separates_vendor_claims_from_independent_evaluation(stories):
    brief = brief_for(stories, LUMEN)
    content = inspect_evidence(stories, brief)
    report = report_of(content)
    roles = {r.citation_id: r.role for r in report.original_sources}

    assert roles[cid_for(content, "Independent evaluation of Lumen-4")] == "independent_evaluation"
    assert roles[cid_for(content, "Lumen Labs launches")] == "vendor_claim"
    assert roles[cid_for(content, "Lumen-4 Technical Report")] == "vendor_claim"
    assert roles[cid_for(content, "Opinion:")] == "commentary"
    assert roles[cid_for(content, "Lumen Labs unveils")] == "secondary"
    assert report.study_design is None
    assert any("second independent evaluation" in m for m in report.missing_evidence)

    by_claim = {c.claim: c for c in report.claim_comparisons}
    assert [c.claim for c in report.claim_comparisons] == [c.text for c in brief.claims]
    vendor_claim = next(c for text, c in by_claim.items() if text.startswith("Lumen Labs released Lumen-4"))
    assert vendor_claim.status == "contested"
    assert cid_for(content, "Independent evaluation of Lumen-4") in vendor_claim.disagreeing_citation_ids
    opinion = next(c for c in report.claim_comparisons if c.claim.startswith("Opinion"))
    assert "not evidence itself" in opinion.note
    assert not any(BIASED.search(n) for n in report.framing_notes)


@pytest.mark.parametrize("title_start", [LUMEN, HALBROOK, CORVANE, NORTHBRIDGE])
def test_every_vertical_report_is_cited_and_free_of_bias_labels(stories, title_start):
    brief = brief_for(stories, title_start)
    content = inspect_evidence(stories, brief)
    report = report_of(content)
    valid = {c["id"] for c in content.citations}
    assert all_ids(report) <= valid
    assert {r.citation_id for r in report.original_sources} == valid
    assert len(report.claim_comparisons) == len(brief.claims)
    assert all(st.citation_ids for st in [*report.facts, *report.interpretations])
    assert report.facts
    assert not any(labels_source_biased(n) for n in report.framing_notes)
    for comp in report.claim_comparisons:
        if comp.status == "contested":
            assert comp.disagreeing_citation_ids
        if comp.status in ("supported", "corroborated"):
            assert comp.agreeing_citation_ids


def test_finance_report_flags_backtest_and_missing_live_record(stories):
    content = inspect_evidence(stories, brief_for(stories, NORTHBRIDGE))
    report = report_of(content)
    assert any("live track record" in m for m in report.missing_evidence)
    perf = [c for c in report.claim_comparisons if "Performance result labelled backtest" in c.note]
    assert perf and all("not evidence of future returns" in c.note for c in perf)


# --------------------------------------------------------------------------
# Caching and the evidence pack
# --------------------------------------------------------------------------


def test_report_is_cached_per_brief_version(stories):
    brief = brief_for(stories, HALBROOK)
    first = inspect_evidence(stories, brief)
    runs = stories.scalar(select(func.count()).select_from(AgentRun).where(AgentRun.agent == AGENT_NAME))
    assert runs == 1

    again = inspect_evidence(stories, brief)
    assert again.id == first.id
    assert stories.scalar(select(func.count()).select_from(AgentRun).where(AgentRun.agent == AGENT_NAME)) == 1

    refreshed = inspect_evidence(stories, brief, refresh=True)
    assert refreshed.id != first.id
    assert refreshed.cache_key == first.cache_key

    editor = User(email="editor@example.com", display_name="Ed", password_hash="x", role="editor")
    stories.add(editor)
    stories.flush()
    record_correction(stories, brief, editor, "Clarified the speed figure.", material=False)
    assert brief.version == 2
    corrected = inspect_evidence(stories, brief)
    assert corrected.cache_key != first.cache_key
    assert corrected.params == {"brief_version": 2}
    rows = stories.scalar(
        select(func.count()).select_from(GeneratedContent).where(GeneratedContent.kind == CONTENT_KIND)
    )
    assert rows == 3


def test_pack_adds_story_members_beyond_the_brief_citations(session, factory):
    news = factory.source(name="Daily Wire")
    vendor = factory.source(trust_tier=TrustTier.VENDOR, doc_type=DocType.PRESS_RELEASE, name="Acme Newsroom")
    story = Story(title="Acme Model Z", vertical=Vertical.TECHNOLOGY)
    session.add(story)
    now = utcnow()
    articles = [
        factory.article(
            "Acme launches Model Z",
            "Acme says Model Z answers 90% of test questions.",
            source=vendor,
            published_at=now - timedelta(days=12),
        )  # fmt: skip
    ]
    for i in range(10):
        articles.append(
            factory.article(
                f"Model Z report {i}",
                f"Reporters looked at Model Z deployment number {i} in {10 + i} offices.",
                source=news,
                published_at=now - timedelta(days=11 - i),
            )
        )
    for art in articles:
        art.story = story
    session.flush()

    brief = generate_brief(session, story)
    assert len(brief.citations) == 8
    content = inspect_evidence(session, brief)
    ids = [c["id"] for c in content.citations]
    assert ids[:8] == [c["id"] for c in brief.citations]
    assert len(ids) == 11 and ids[8:] == ["c9", "c10", "c11"]
    report = report_of(content)
    assert [r.citation_id for r in report.original_sources] == ids


def test_brief_without_a_story_or_stored_articles(session):
    brief = Brief(
        vertical=Vertical.FINANCE,
        kind="earnings",
        headline="Example Corp quarterly results",
        summary="Example Corp reported revenue of $1.2 billion [c1].",
        citations=[
            {
                "id": "c1",
                "url": "https://www.sec.gov/Archives/edgar/data/1/000000000126000001.txt",
                "title": "Example Corp 10-Q",
                "publisher": "SEC EDGAR",
                "snippet": "Revenue was $1.2 billion for the quarter, compared with $1.0 billion a year earlier.",
                "trust_tier": "primary",
                "doc_type": "filing",
            }
        ],
    )
    session.add(brief)
    session.flush()
    report = report_of(inspect_evidence(session, brief))
    assert [(r.citation_id, r.role) for r in report.original_sources] == [("c1", "original")]
    assert report.study_design is None
    assert report.facts and report.facts[0].citation_ids == ["c1"]


# --------------------------------------------------------------------------
# Output hygiene and the Claude path
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "flagged"),
    [
        ("The university's release is biased toward its own lab.", True),
        ("A biased source should be ignored.", True),
        ("This outlet's coverage is propaganda.", True),
        ("The study notes possible selection bias in which classrooms volunteered.", False),
        ("The release leads with speed; the preprint leads with the transfer test.", False),
    ],
)
def test_labels_source_biased(text, flagged):
    assert labels_source_biased(text) is flagged


def _docs(session, factory):
    vendor = factory.source(trust_tier=TrustTier.VENDOR, doc_type=DocType.PRESS_RELEASE, name="Vendor Newsroom")
    news = factory.source(name="Wire One")
    other = factory.source(name="Wire Two")
    a = factory.article("Vendor launches tool", "The vendor says its tool is 40% faster.", source=vendor)
    b = factory.article("Wire One on the tool", "Wire One found the tool 40% faster in its test.", source=news)
    c = factory.article("Wire Two on the tool", "Wire Two found the tool 40% faster in its test.", source=other)
    pack = build_evidence_pack([b, c, a])
    return source_docs(pack, {x.id: x for x in (a, b, c)})


def test_sanitize_report_enforces_ids_statuses_and_framing(session, factory):
    docs = _docs(session, factory)  # c1 Wire One, c2 Wire Two, c3 vendor
    raw = EvidenceReportOut(
        original_sources=[
            SourceRole(citation_id="c9", role="original"),
            SourceRole(citation_id="c1", role="secondary"),
        ],
        claim_comparisons=[
            ClaimComparison(claim="Tool is faster", agreeing_citation_ids=["c3", "c1"], status="corroborated"),
            ClaimComparison(claim="Tool is safe", agreeing_citation_ids=["c1"], status="contested"),
            ClaimComparison(claim="Tool is cheap", agreeing_citation_ids=["c8"], status="supported"),
            ClaimComparison(claim="Tool is 40% faster", agreeing_citation_ids=["c1", "c2"], status="corroborated"),
        ],
        facts=[StatementOut(text="A fact [c1][c5].", citation_ids=["c1", "c5"]), StatementOut(text="No source.")],
        framing_notes=[
            "Vendor Newsroom is biased in favour of its product [c3].",
            "The vendor leads with speed [c3]; Wire One adds its own test [c1][c6].",
        ],
    )
    clean = sanitize_report(raw, docs)
    assert [r.citation_id for r in clean.original_sources] == ["c1", "c2", "c3"]
    assert clean.original_sources[2].role == "vendor_claim"  # filled by the rule-based classifier
    statuses = [c.status for c in clean.claim_comparisons]
    assert statuses == ["supported", "supported", "unverified", "corroborated"]
    assert clean.facts == [StatementOut(text="A fact [c1].", citation_ids=["c1"])]
    assert clean.framing_notes == ["The vendor leads with speed [c3]; Wire One adds its own test [c1]."]


class _FakeMessages:
    def __init__(self, text: str):
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=900, output_tokens=400, cache_read_input_tokens=0, cache_creation_input_tokens=0
            ),
            model="claude-opus-5-5",
        )


def test_claude_report_is_sanitized_before_caching(session, factory):
    vendor = factory.source(trust_tier=TrustTier.VENDOR, doc_type=DocType.PRESS_RELEASE, name="Acme Newsroom")
    news = factory.source(name="Daily Wire")
    story = Story(title="Acme tool", vertical=Vertical.TECHNOLOGY)
    session.add(story)
    for art in (
        factory.article("Acme launches tool", "Acme says its tool is 40% faster.", source=vendor),
        factory.article(
            "Wire tests Acme tool", "Daily Wire tested the tool and found it 12% faster.", source=news
        ),
    ):
        art.story = story
    session.flush()
    brief = generate_brief(session, story)

    report = EvidenceReportOut(
        original_sources=[SourceRole(citation_id="c2", role="vendor_claim", note="Acme's release.")],
        claim_comparisons=[
            ClaimComparison(claim=c.text, agreeing_citation_ids=["c1", "c12"], status="supported")
            for c in brief.claims
        ],
        facts=[StatementOut(text="Daily Wire measured 12% [c1].", citation_ids=["c1"])],
        framing_notes=[
            "Acme's release is clearly biased.",
            "Acme leads with 40% [c2]; Daily Wire found 12% [c1].",
        ],
    ).model_dump_json()
    client = SimpleNamespace(beta=SimpleNamespace(messages=_FakeMessages(report)))
    orchestrator = Orchestrator(session, provider=ClaudeProvider(model="claude-opus-5-5", client=client))

    content = inspect_evidence(session, brief, orchestrator=orchestrator)

    kwargs = client.beta.messages.calls[0]
    assert "Evidence and Bias Inspection" in kwargs["system"][0]["text"]
    assert "Never call a source" in kwargs["system"][0]["text"]
    assert "Brief claims to compare" in kwargs["messages"][0]["content"]
    assert content.generated_by == "claude-opus-5-5"
    clean = report_of(content)
    assert clean.framing_notes == ["Acme leads with 40% [c2]; Daily Wire found 12% [c1]."]
    assert all(c.agreeing_citation_ids == ["c1"] for c in clean.claim_comparisons)
    assert {r.citation_id for r in clean.original_sources} == {"c1", "c2"}
