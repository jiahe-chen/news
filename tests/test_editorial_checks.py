"""Editorial quality gate: automated checks over hand-built briefs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from newsren.agents.base import citation_for
from newsren.editorial.checks import CHECK_NAMES, assess_risk, failed_checks, run_checks
from newsren.models import Article, Brief, BriefClaim, Source, SourceKind
from newsren.schemas import CheckResult

DATA = Path(__file__).parent / "data" / "editorial" / "newsroom.json"
LICENSE = "Headline + publisher summary + link; full text not stored."


class Newsroom:
    """Builds sources, articles, and briefs from the fictional newsroom sample."""

    def __init__(self, factory):
        self.factory = factory
        self.session = factory.session
        self.specs: dict[str, dict[str, str]] = json.loads(DATA.read_text())["articles"]
        self.sources: dict[str, Source] = {}
        self.articles: dict[str, Article] = {}

    def article(self, key: str) -> Article:
        if key not in self.articles:
            spec = self.specs[key]
            source = self.sources.get(spec["publisher"])
            if source is None:
                source = self.factory.source(
                    vertical=spec["vertical"],
                    trust_tier=spec["trust_tier"],
                    doc_type=spec["doc_type"],
                    name=spec["publisher"],
                )
                source.license_note = LICENSE
                self.sources[spec["publisher"]] = source
            self.articles[key] = self.factory.article(
                spec["title"],
                spec["body"],
                source=source,
                vertical=spec["vertical"],
                doc_type=spec["doc_type"],
                trust_tier=spec["trust_tier"],
            )
        return self.articles[key]

    def citations(self, keys: list[str]) -> list[dict[str, Any]]:
        return [citation_for(self.article(k), f"c{i}").model_dump() for i, k in enumerate(keys, 1)]

    def brief(
        self,
        keys: list[str],
        claims: list[dict[str, Any]],
        *,
        vertical: str = "technology",
        citations: list[dict[str, Any]] | None = None,
        **fields: Any,
    ) -> Brief:
        values: dict[str, Any] = {
            "headline": "Lumen-4 launch: independent test finds a lower score than the company's figure",
            "summary": (
                "Lumen Labs released Lumen-4, a math tutoring model, and says it solves 82% of competition "
                "problems. An independent evaluation of 1,200 held-out problems found it solved 61%."
            ),
            "why_it_matters": "Schools weighing AI math tutors need independent evidence, not only vendor figures.",
            "sections": [
                {
                    "title": "What happened?",
                    "body": "Lumen Labs released Lumen-4 on Monday.",
                    "citation_ids": ["c1"],
                }
            ],
        }
        values.update(fields)
        brief = Brief(
            vertical=vertical, citations=citations if citations is not None else self.citations(keys), **values
        )
        brief.claims = [
            BriefClaim(position=i, **{"support_status": "supported", **claim}) for i, claim in enumerate(claims)
        ]
        self.session.add(brief)
        self.session.flush()
        return brief


TECH_KEYS = ["vendor_launch", "independent_eval"]
TECH_CLAIMS: list[dict[str, Any]] = [
    {"text": "Lumen Labs released the Lumen-4 math tutoring model on Monday.", "citation_ids": ["c1"]},
    {"text": "Lumen Labs says Lumen-4 solves 82% of competition math problems.", "citation_ids": ["c1"]},
    {
        "text": "An independent evaluation of 1,200 held-out problems found Lumen-4 solved 61% of them.",
        "citation_ids": ["c2"],
    },
    {
        "text": "The independent result is lower than the company's figure, which may reflect different test sets.",
        "claim_type": "interpretation",
        "citation_ids": ["c1", "c2"],
    },
]

FINANCE_KEYS = ["bank_note", "finance_news"]
FINANCE_FIELDS: dict[str, Any] = {
    "headline": "Harbor Bank's AI agents beat a 60/40 portfolio in a backtest, not in live trading",
    "summary": (
        "Harbor Bank researchers reported that AI portfolio agents beat a 60/40 portfolio in a 20-year "
        "backtest; the bank has not reported live results."
    ),
    "why_it_matters": "A backtest shows how a strategy would have done on past data; it is not a live track record.",
    "sections": [
        {"title": "What was reported", "body": "A backtest covering 2005 to 2025.", "citation_ids": ["c1"]},
    ],
}
FINANCE_CLAIMS: list[dict[str, Any]] = [
    {
        "text": (
            "Harbor Bank researchers reported that AI portfolio agents beat a 60/40 portfolio by 1.8 "
            "percentage points a year in a 2005-2025 backtest."
        ),
        "claim_type": "performance",
        "performance_label": "backtest",
        "performance_period": "2005-2025",
        "benchmark": "60/40 stock/bond portfolio",
        "citation_ids": ["c1"],
    },
    {"text": "Harbor Bank has not reported live results for the agents.", "citation_ids": ["c2"]},
    {
        "text": "The authors cautioned that an in-sample result is not proof of future outperformance.",
        "citation_ids": ["c1"],
    },
]


@pytest.fixture()
def newsroom(factory) -> Newsroom:
    return Newsroom(factory)


def by_name(checks: list[CheckResult]) -> dict[str, CheckResult]:
    return {c.name: c for c in checks}


def failing(checks: list[CheckResult]) -> list[str]:
    return [c.name for c in checks if not c.passed]


# --------------------------------------------------------------------------- clean briefs


def test_clean_technology_brief_passes_every_check(session, newsroom) -> None:
    brief = newsroom.brief(TECH_KEYS, TECH_CLAIMS)
    checks = run_checks(session, brief)
    assert [c.name for c in checks] == list(CHECK_NAMES)
    assert failing(checks) == [], [c for c in checks if not c.passed]
    assert assess_risk(brief, checks) == "low"


def test_clean_finance_brief_passes_but_is_always_high_risk(session, newsroom) -> None:
    brief = newsroom.brief(FINANCE_KEYS, FINANCE_CLAIMS, vertical="finance", **FINANCE_FIELDS)
    checks = run_checks(session, brief)
    assert failing(checks) == [], [c for c in checks if not c.passed]
    assert assess_risk(brief, checks) == "high"


def test_run_checks_is_read_only(session, newsroom) -> None:
    brief = newsroom.brief(TECH_KEYS, TECH_CLAIMS)
    before = (brief.status, brief.risk_level, list(brief.check_results), brief.version)
    run_checks(session, brief)
    assert (brief.status, brief.risk_level, list(brief.check_results), brief.version) == before


# --------------------------------------------------------------------------- citations


def test_fact_claim_without_citation_is_a_blocker(session, newsroom) -> None:
    claims = [
        *TECH_CLAIMS,
        {"text": "Lumen-4 is already used in 300 schools.", "citation_ids": []},
        {"text": "Some teachers will find it useful.", "claim_type": "opinion", "citation_ids": []},
    ]
    check = by_name(run_checks(session, newsroom.brief(TECH_KEYS, claims)))["citation_coverage"]
    assert not check.passed and check.severity == "blocker"
    assert len(check.details) == 1
    assert "Claim 5" in check.details[0] and "300 schools" in check.details[0]


def test_uncited_high_impact_interpretation_is_a_blocker(session, newsroom) -> None:
    claims = [
        *TECH_CLAIMS,
        {"text": "Districts may adopt it.", "claim_type": "interpretation", "high_impact": True},
    ]
    check = by_name(run_checks(session, newsroom.brief(TECH_KEYS, claims)))["citation_coverage"]
    assert not check.passed and "(high-impact)" in check.details[0]


def test_unknown_citation_ids_are_blockers(session, newsroom) -> None:
    claims = [*TECH_CLAIMS, {"text": "Lumen-4 solved 61% of problems.", "citation_ids": ["c2", "c9"]}]
    brief = newsroom.brief(
        TECH_KEYS,
        claims,
        sections=[{"title": "What evidence exists?", "body": "An evaluation.", "citation_ids": ["c7"]}],
        disagreements=[
            {
                "topic": "Score",
                "positions": [
                    {"position": "82%", "citation_ids": ["c1"]},
                    {"position": "61%", "citation_ids": ["c8"]},
                ],
            }
        ],
    )
    check = by_name(run_checks(session, brief))["unknown_citation_ids"]
    assert not check.passed and check.severity == "blocker"
    text = " ".join(check.details)
    assert "c9" in text and "c7" in text and "c8" in text and "c2" not in text.replace("c2,", "")


def test_brief_without_sources_is_blocked(session, newsroom) -> None:
    claims = [{"text": "Lumen Labs released Lumen-4.", "citation_ids": []}]
    checks = by_name(run_checks(session, newsroom.brief([], claims, citations=[])))
    assert not checks["sources_cited"].passed and checks["sources_cited"].severity == "blocker"
    assert checks["approved_sources"].passed  # nothing to judge


def test_missing_headline_and_summary_block(session, newsroom) -> None:
    check = by_name(run_checks(session, newsroom.brief(TECH_KEYS, TECH_CLAIMS, headline="", summary=" ")))[
        "content_present"
    ]
    assert not check.passed and len(check.details) == 2


def test_claims_recorded_warns_when_brief_has_no_claims(session, newsroom) -> None:
    checks = run_checks(session, newsroom.brief(TECH_KEYS, []))
    assert failing(checks) == ["claims_recorded"]
    assert assess_risk(newsroom.brief(TECH_KEYS, []), checks) == "medium"


def test_claim_support_flags_figures_missing_from_cited_sources(session, newsroom) -> None:
    claims = [
        *TECH_CLAIMS,
        {"text": "The independent test found Lumen-4 solved 75% of problems.", "citation_ids": ["c2"]},
    ]
    check = by_name(run_checks(session, newsroom.brief(TECH_KEYS, claims)))["claim_support"]
    assert not check.passed and check.severity == "warning"
    assert "75" in check.details[0] and "c2" in check.details[0]


def test_claim_support_flags_claims_unrelated_to_their_source(session, newsroom) -> None:
    claims = [
        *TECH_CLAIMS,
        {
            "text": "Regulators opened an antitrust inquiry into semiconductor pricing practices.",
            "citation_ids": ["c1"],
        },
    ]
    check = by_name(run_checks(session, newsroom.brief(TECH_KEYS, claims)))["claim_support"]
    assert not check.passed and "little wording in common" in check.details[0]


# --------------------------------------------------------------------------- provenance


def test_unapproved_source_is_a_blocker(session, newsroom, factory) -> None:
    rogue_source = factory.source(name="Unvetted Blog", approved=False)
    rogue_source.license_note = LICENSE
    rogue = factory.article(
        "Lumen-4 leaks", "Unverified screenshots of Lumen-4 appeared online.", source=rogue_source
    )
    citations = [*newsroom.citations(TECH_KEYS), citation_for(rogue, "c3").model_dump()]
    claims = [*TECH_CLAIMS, {"text": "Screenshots of Lumen-4 appeared online.", "citation_ids": ["c3"]}]
    check = by_name(run_checks(session, newsroom.brief(TECH_KEYS, claims, citations=citations)))[
        "approved_sources"
    ]
    assert not check.passed and check.severity == "blocker"
    assert len(check.details) == 1 and "c3" in check.details[0] and "not approved" in check.details[0]


def test_unknown_domain_fails_and_allowlisted_primary_record_passes(session, newsroom) -> None:
    citations = [
        *newsroom.citations(TECH_KEYS),
        {
            "id": "c3",
            "url": "https://random-blog.example/post/1",
            "title": "A blog post",
            "retrieved_at": "2026-10-01T00:00:00Z",
        },
        {
            "id": "c4",
            "url": "https://www.sec.gov/Archives/edgar/data/1/000000000126000001.txt",
            "title": "Form 8-K",
            "retrieved_at": "2026-10-01T00:00:00Z",
            "published_at": "2026-09-30T00:00:00Z",
            "doc_type": "filing",
            "trust_tier": "primary",
        },
    ]
    checks = by_name(run_checks(session, newsroom.brief(TECH_KEYS, TECH_CLAIMS, citations=citations)))
    approved = checks["approved_sources"]
    assert not approved.passed and len(approved.details) == 1 and "random-blog.example" in approved.details[0]
    licensing = checks["licensing_notes"]
    assert not licensing.passed and [d for d in licensing.details if "c4" in d] == []


def test_demo_sources_pass_when_their_rows_are_approved(session, newsroom) -> None:
    demo = Source(
        slug="demo-lumen-labs",
        name="Lumen Labs Newsroom (demo)",
        kind=SourceKind.DEMO,
        url="https://lumenlabs.example/feed",
        homepage="https://lumenlabs.example",
        vertical="technology",
        trust_tier="vendor",
        default_doc_type="press_release",
        approved=True,
        license_note="Fictional demo content written for The Newsren.",
    )
    session.add(demo)
    session.flush()
    citations = [
        {
            "id": "c1",
            "url": "https://lumenlabs.example/stories/lumen-4-launch",
            "title": "Lumen Labs releases Lumen-4",
            "publisher": demo.name,
            "published_at": "2026-10-01T09:00:00Z",
            "retrieved_at": "2026-10-01T10:00:00Z",
            "snippet": "Lumen Labs released Lumen-4, a math tutoring model, on Monday.",
            "trust_tier": "vendor",
            "doc_type": "press_release",
        }
    ]
    claims = [{"text": "Lumen Labs released the Lumen-4 math tutoring model on Monday.", "citation_ids": ["c1"]}]
    brief = newsroom.brief([], claims, citations=citations, is_demo=True, sections=[])
    checks = by_name(run_checks(session, brief))
    assert checks["approved_sources"].passed
    assert "demo" in checks["approved_sources"].message
    assert checks["licensing_notes"].passed

    demo.active = False  # paused sources stay approved for items already published
    session.flush()
    assert by_name(run_checks(session, brief))["approved_sources"].passed

    demo.approved = False
    session.flush()
    assert not by_name(run_checks(session, brief))["approved_sources"].passed


def test_malformed_links_and_records_are_blockers(session, newsroom) -> None:
    citations = [
        *newsroom.citations(TECH_KEYS),
        {"id": "c3", "url": "lumenlabs.example/no-scheme", "title": "No scheme"},
        {"id": "c4", "url": "https://bad host.example/x", "title": "Space in host"},
        {"id": "c5", "url": "https://ok.example/x"},
    ]
    check = by_name(run_checks(session, newsroom.brief(TECH_KEYS, TECH_CLAIMS, citations=citations)))[
        "well_formed_links"
    ]
    assert not check.passed and check.severity == "blocker"
    text = " ".join(check.details)
    assert "c3" in text and "c4" in text and "Citation record c5 is incomplete" in text


def test_missing_timestamps_and_licensing_notes(session, newsroom) -> None:
    citations = newsroom.citations(TECH_KEYS)
    citations[0]["retrieved_at"] = None
    citations[1]["published_at"] = None
    newsroom.sources["The Circuit Ledger"].license_note = None
    checks = by_name(run_checks(session, newsroom.brief(TECH_KEYS, TECH_CLAIMS, citations=citations)))
    assert not checks["retrieval_timestamps"].passed and checks["retrieval_timestamps"].severity == "warning"
    assert "c1" in checks["retrieval_timestamps"].details[0]
    assert not checks["publication_dates"].passed and checks["publication_dates"].severity == "info"
    assert not checks["licensing_notes"].passed and "The Circuit Ledger" in checks["licensing_notes"].details[0]


# --------------------------------------------------------------------------- labelling


def test_unlabelled_preprint_is_flagged(session, newsroom) -> None:
    keys = [*TECH_KEYS, "preprint"]
    claims = [
        *TECH_CLAIMS,
        {"text": "The evaluation used 1,200 held-out competition problems.", "citation_ids": ["c3"]},
    ]
    check = by_name(run_checks(session, newsroom.brief(keys, claims)))["preprint_labelling"]
    assert not check.passed and check.severity == "warning"
    assert "c3" in check.details[0]


@pytest.mark.parametrize(
    "field_values",
    [
        {"summary": "An independent preprint found Lumen-4 solved 61% of 1,200 held-out problems."},
        {"uncertainty": ["The evaluation has not yet been peer reviewed."]},
        {"uncertainty": ["The evaluation is not peer-reviewed."]},
    ],
)
def test_labelled_preprint_passes(session, newsroom, field_values) -> None:
    keys = [*TECH_KEYS, "preprint"]
    claims = [
        *TECH_CLAIMS,
        {"text": "The evaluation used 1,200 held-out competition problems.", "citation_ids": ["c3"]},
    ]
    assert by_name(run_checks(session, newsroom.brief(keys, claims, **field_values)))["preprint_labelling"].passed


def test_contested_claim_requires_described_disagreement(session, newsroom) -> None:
    keys = [*TECH_KEYS, "independent_eval_2"]
    contested = {
        "text": "Lumen Labs says Lumen-4 is its most capable model for every math topic.",
        "support_status": "contested",
        "citation_ids": ["c1", "c3"],
    }
    claims = [*TECH_CLAIMS, contested]
    check = by_name(run_checks(session, newsroom.brief(keys, claims)))["disagreements_described"]
    assert not check.passed and check.severity == "warning" and "contested" in check.details[0]

    one_sided = [
        {"topic": "Is Lumen-4 strong across topics?", "positions": [{"position": "Yes", "citation_ids": ["c1"]}]}
    ]
    check = by_name(run_checks(session, newsroom.brief(keys, claims, disagreements=one_sided)))[
        "disagreements_described"
    ]
    assert not check.passed and "fewer than two positions" in check.details[0]

    both_sides = [
        {
            "topic": "Is Lumen-4 strong across topics?",
            "positions": [
                {"position": "Lumen Labs calls it its most capable model.", "citation_ids": ["c1"]},
                {"position": "Northfield AI Lab found it weaker on geometry.", "citation_ids": ["c3"]},
            ],
        }
    ]
    assert by_name(run_checks(session, newsroom.brief(keys, claims, disagreements=both_sides)))[
        "disagreements_described"
    ].passed


def test_forecast_claims_must_be_labelled(session, newsroom) -> None:
    unlabelled = {
        "text": "Lumen Labs will release Lumen-5 next year.",
        "claim_type": "forecast",
        "citation_ids": ["c1"],
    }
    check = by_name(run_checks(session, newsroom.brief(TECH_KEYS, [*TECH_CLAIMS, unlabelled])))[
        "forecast_labelling"
    ]
    assert not check.passed and check.severity == "warning" and "Lumen-5" in check.details[0]

    labelled = {**unlabelled, "text": "Lumen Labs expects to release Lumen-5 next year."}
    assert by_name(run_checks(session, newsroom.brief(TECH_KEYS, [*TECH_CLAIMS, labelled])))[
        "forecast_labelling"
    ].passed


def test_vendor_capability_claim_needs_attribution(session, newsroom) -> None:
    bare = {"text": "Lumen-4 solves 82% of competition math problems.", "citation_ids": ["c1"]}
    check = by_name(run_checks(session, newsroom.brief(TECH_KEYS, [*TECH_CLAIMS, bare])))["vendor_attribution"]
    assert not check.passed and "company's own materials" in check.details[0]


def test_corroborated_requires_two_independent_publishers(session, newsroom) -> None:
    keys = [*TECH_KEYS, "independent_eval_2"]
    weak = {
        "text": "Lumen Labs released Lumen-4 on Monday.",
        "support_status": "corroborated",
        "citation_ids": ["c1"],
    }
    check = by_name(run_checks(session, newsroom.brief(keys, [*TECH_CLAIMS, weak])))["support_status"]
    assert not check.passed and "corroborated" in check.details[0]

    strong = {
        "text": "Independent evaluations found Lumen-4 solved 61% and 64% of the problems they tested.",
        "support_status": "corroborated",
        "citation_ids": ["c2", "c3"],
    }
    assert by_name(run_checks(session, newsroom.brief(keys, [*TECH_CLAIMS, strong])))["support_status"].passed


def test_unverified_fact_is_flagged(session, newsroom) -> None:
    claim = {"text": "Lumen-4 was trained on school data.", "support_status": "unverified", "citation_ids": ["c1"]}
    check = by_name(run_checks(session, newsroom.brief(TECH_KEYS, [*TECH_CLAIMS, claim])))["support_status"]
    assert not check.passed and "unverified" in check.details[0]


def test_education_research_brief_must_name_population_and_outcomes(session, newsroom) -> None:
    claims = [
        {
            "text": "In a randomized study of 600 ninth-grade students, AI tutoring users scored 4 points higher.",
            "citation_ids": ["c1"],
        }
    ]
    vague = newsroom.brief(
        ["learning_study"],
        [{"text": "An AI tutor helped with algebra.", "citation_ids": ["c1"]}],
        vertical="education",
        headline="AI tutor helps with algebra",
        summary="An AI tutor helped with algebra in a new study.",
        why_it_matters="AI tutors are spreading.",
        sections=[],
    )
    check = by_name(run_checks(session, vague))["study_population"]
    assert not check.passed and len(check.details) == 2

    specific = newsroom.brief(
        ["learning_study"],
        claims,
        vertical="education",
        headline="AI tutoring and ninth-grade algebra",
        summary="A randomized study of 600 ninth-grade students found AI tutoring users scored higher on an algebra test.",
        why_it_matters="It did not measure transfer to unfamiliar problems.",
        sections=[],
    )
    assert by_name(run_checks(session, specific))["study_population"].passed


# --------------------------------------------------------------------------- guardrails


FINANCE_VIOLATION = "Analysts upgraded Harbor Bank to buy with a $250 price target."


def test_finance_guardrail_lint_is_a_blocker(session, newsroom) -> None:
    fields = {**FINANCE_FIELDS, "summary": f"{FINANCE_FIELDS['summary']} {FINANCE_VIOLATION}"}
    brief = newsroom.brief(FINANCE_KEYS, FINANCE_CLAIMS, vertical="finance", **fields)
    checks = run_checks(session, brief)
    check = by_name(checks)["guardrail_lint"]
    assert not check.passed and check.severity == "blocker"
    assert len(check.details) == 2
    assert all(d.startswith("Summary:") for d in check.details)
    assert [c.name for c in failed_checks(checks, "blocker")] == ["guardrail_lint"]


def test_finance_lint_lists_offending_claims(session, newsroom) -> None:
    claims = [*FINANCE_CLAIMS, {"text": "AI agents can reliably beat the market.", "citation_ids": ["c1"]}]
    check = by_name(
        run_checks(session, newsroom.brief(FINANCE_KEYS, claims, vertical="finance", **FINANCE_FIELDS))
    )["guardrail_lint"]
    assert not check.passed and check.details[0].startswith("Claim 4:")


def test_non_finance_brief_gets_overclaiming_warning_not_finance_rules(session, newsroom) -> None:
    tech = newsroom.brief(TECH_KEYS, TECH_CLAIMS, why_it_matters=FINANCE_VIOLATION)
    assert by_name(run_checks(session, tech))["guardrail_lint"].passed

    overclaim = newsroom.brief(TECH_KEYS, TECH_CLAIMS, why_it_matters="The evaluation proves that AI tutors work.")
    check = by_name(run_checks(session, overclaim))["guardrail_lint"]
    assert not check.passed and check.severity == "warning"
    assert check.details[0].startswith("Why it matters: Overstated certainty")


def test_unlabelled_performance_claim_blocks_finance_and_warns_elsewhere(session, newsroom) -> None:
    unlabelled = {
        "text": "Harbor Bank's AI agents beat a 60/40 portfolio by 1.8 percentage points a year.",
        "claim_type": "performance",
        "citation_ids": ["c1"],
    }
    claims = [FINANCE_CLAIMS[1], unlabelled]
    finance = by_name(
        run_checks(session, newsroom.brief(FINANCE_KEYS, claims, vertical="finance", **FINANCE_FIELDS))
    )
    check = finance["performance_labels"]
    assert not check.passed and check.severity == "blocker"
    assert check.details[0].startswith("Claim 2:") and "backtest, simulated, or live" in check.details[0]
    # The text names its benchmark (a 60/40 portfolio) but not the period or the backtest/live label.
    assert "period" in check.details[0] and "benchmark" not in check.details[0]

    tech = by_name(
        run_checks(session, newsroom.brief(FINANCE_KEYS, claims, vertical="technology", **FINANCE_FIELDS))
    )
    assert not tech["performance_labels"].passed and tech["performance_labels"].severity == "warning"


# --------------------------------------------------------------------------- risk


def test_high_impact_claims_are_flagged_and_raise_risk(session, newsroom) -> None:
    claims = [dict(TECH_CLAIMS[0]), *TECH_CLAIMS[1:]]
    claims[2] = {**claims[2], "high_impact": True}
    brief = newsroom.brief(TECH_KEYS, claims)
    checks = run_checks(session, brief)
    check = by_name(checks)["high_impact_claims"]
    assert not check.passed and check.severity == "info" and "Claim 3" in check.details[0]
    assert failed_checks(checks, "blocker") == [] and failed_checks(checks, "warning") == []
    assert assess_risk(brief, checks) == "high"

    claims[2] = {**claims[2], "support_status": "unverified"}
    weak = by_name(run_checks(session, newsroom.brief(TECH_KEYS, claims)))["high_impact_claims"]
    assert weak.severity == "warning" and "not firmly supported" in weak.details[0]


def test_assess_risk_levels_and_stored_dicts(session, newsroom) -> None:
    brief = newsroom.brief(TECH_KEYS, TECH_CLAIMS)
    warning = CheckResult(name="x", passed=False, severity="warning")
    blocker = CheckResult(name="y", passed=False, severity="blocker")
    info = CheckResult(name="z", passed=False, severity="info")
    assert assess_risk(brief, [info]) == "low"
    assert assess_risk(brief, [warning.model_dump()]) == "medium"
    assert assess_risk(brief, [warning, blocker.model_dump()]) == "high"
    brief.vertical = "finance"
    assert assess_risk(brief, []) == "high"
