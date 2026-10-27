"""Each editorial product's offline output: non-empty, labelled, citing only evidence-pack ids."""

from __future__ import annotations

import importlib.util
import re
import sys
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from newsren.agents.base import Orchestrator
from newsren.agents.products import PRODUCTS, available_products, product_context, run_product, sanitize_product
from newsren.llm.base import LLMResult
from newsren.models import AgentRun, GeneratedContent
from newsren.schemas import ProductOut, ProductSection


def _load_newsroom():
    name = "explain_newsroom"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).parent / "data" / "explain" / "newsroom.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.fixture()
def newsroom(factory):
    return _load_newsroom().Newsroom(factory)


BRIEFS = [
    "model_launch",
    "tutor_launch",
    "learning_study",
    "policy",
    "semiconductor",
    "earnings",
    "funding",
    "backtest",
    "enforcement",
]


def _check(row: GeneratedContent, key: str) -> ProductOut:
    out = ProductOut.model_validate(row.content)
    valid = {c["id"] for c in row.citations}
    assert row.kind == f"product:{key}" and out.product == key
    assert out.title and out.summary and out.sections and out.caveats
    assert out.labels and set(s.label for s in out.sections) <= set(out.labels)
    for section in out.sections:
        assert section.heading and section.body and section.label, section
        assert set(section.citation_ids) <= valid, section
        assert set(re.findall(r"\[(c\d+)\]", section.body)) <= valid, section
    assert set(re.findall(r"\[(c\d+)\]", out.summary)) <= valid
    return out


@pytest.mark.parametrize("name", BRIEFS)
def test_every_applicable_product_is_labelled_and_cites_valid_ids(session, newsroom, name):
    brief = newsroom.brief(name)
    products = available_products(brief)
    assert products
    for spec in products:
        out = _check(run_product(session, spec.key, brief), spec.key)
        assert any(section.citation_ids for section in out.sections), spec.key


def test_runs_through_orchestrator_and_caches(session, newsroom):
    brief = newsroom.brief("semiconductor")
    first = run_product(session, "company_watch", brief)
    assert run_product(session, "company_watch", brief).id == first.id
    assert run_product(session, "company_watch", brief, refresh=True).id != first.id
    runs = session.scalars(select(AgentRun).where(AgentRun.agent == "product:company_watch")).all()
    assert len(runs) == 2 and runs[0].subject_type == "brief" and runs[0].subject_id == brief.id


def test_launch_reality_check(session, newsroom):
    out = _check(
        run_product(session, "launch_reality_check", newsroom.brief("model_launch")), "launch_reality_check"
    )
    checks = [s for s in out.sections if s.heading.startswith("Claim check")]
    assert checks and any(s.label == "independently tested" for s in checks)
    tested = next(s for s in checks if s.label == "independently tested")
    assert "Vendor claim:" in tested.body and "Independent evidence:" in tested.body
    assert len(tested.citation_ids) == 2
    body = " ".join(s.body for s in out.sections if s.label != "correction")
    # The independent score a correction withdrew (81.3%) is never used as evidence.
    assert "81.3%" not in body
    assert any(s.label == "correction" for s in out.sections)
    assert "preprint" in out.labels

    tutor = _check(
        run_product(session, "launch_reality_check", newsroom.brief("tutor_launch")), "launch_reality_check"
    )
    flagged = [s for s in tutor.sections if s.label == "no third-party evidence"]
    assert flagged and "no third-party evidence" in tutor.labels
    assert "lack third-party evidence" in tutor.summary


def test_research_paper_decoder(session, newsroom):
    out = _check(
        run_product(session, "research_paper_decoder", newsroom.brief("learning_study")), "research_paper_decoder"
    )
    headings = " | ".join(s.heading for s in out.sections)
    for part in (
        "status",
        "research question",
        "method",
        "findings",
        "limitations",
        "causal claim",
        "applications",
    ):
        assert part in headings, part
    status = next(s for s in out.sections if s.heading.endswith("status"))
    assert status.label == "preprint" and "peer review" in status.body
    causal = next(s for s in out.sections if "causal" in s.heading)
    assert "by chance" in causal.body and "other populations" in causal.body
    applications = next(s for s in out.sections if "applications" in s.heading)
    assert applications.label == "hypothesis"
    findings = next(s for s in out.sections if s.heading.endswith("findings"))
    assert "not statistically significant" in findings.body and "preprint" in findings.label


def test_technology_timeline_uses_story_events_in_order(session, newsroom):
    brief = newsroom.brief("model_launch")
    out = _check(run_product(session, "technology_timeline", brief), "technology_timeline")
    dated = [s for s in out.sections if "·" in s.heading]
    assert len(dated) == len(brief.story.events)
    dates = [datetime.strptime(s.heading.split(" · ")[0], "%b %d, %Y") for s in dated]
    assert dates == sorted(dates)
    labels = {s.label for s in dated}
    assert {"independent evaluation", "correction"} <= labels
    assert any(s.heading == "How the evidence changed" for s in out.sections)


def test_policy_impact_explorer_parses_timeline(session, newsroom):
    out = _check(
        run_product(session, "policy_impact_explorer", newsroom.brief("policy")), "policy_impact_explorer"
    )
    headings = [s.heading for s in out.sections]
    assert "Who is affected?" in headings and "What remains uncertain?" in headings
    timeline = [s for s in out.sections if " · " in s.heading]
    phrases = [s.heading.split(" · ")[1] for s in timeline]
    assert phrases[:4] == ["March 2027", "2027-28", "fall 2028", "fall 2029"]
    assert all(s.label == "scheduled" for s in timeline)
    assert "Adopted" in out.summary
    uncertain = next(s for s in out.sections if s.heading == "What remains uncertain?")
    assert uncertain.label == "open question" and "has not decided" in uncertain.body


def test_classroom_discussion_builder_has_age_bands(session, newsroom):
    out = _check(
        run_product(session, "classroom_discussion_builder", newsroom.brief("learning_study")),
        "classroom_discussion_builder",
    )
    headings = [s.heading for s in out.sections]
    assert headings[0] == "Verified facts to share"
    for band in ("Middle school (ages 11-14)", "High school (ages 14-18)", "University"):
        section = next(s for s in out.sections if s.heading == band)
        assert "Discussion questions:" in section.body and "Activity:" in section.body
    middle = next(s for s in out.sections if s.heading.startswith("Middle"))
    university = next(s for s in out.sections if s.heading == "University")
    assert "Activity: Fact or open question?" in middle.body and "Activity: Evidence audit" in university.body
    facts = out.sections[0].body
    assert "learn algebra 30% faster" not in facts.lower()
    assert any(s.label == "open question" for s in out.sections)


def test_education_evidence_checker(session, newsroom):
    out = _check(
        run_product(session, "education_evidence_checker", newsroom.brief("learning_study")),
        "education_evidence_checker",
    )
    by_heading = {s.heading: s for s in out.sections}
    matched = [s for s in out.sections if s.label == "promotional claim vs. measured outcome"]
    assert matched and "What the study measured" in matched[0].body and "22 minutes" in matched[0].body
    speed = by_heading["Speed or transfer?"]
    assert "not statistically significant" in speed.body and "transfer test" in speed.body
    assert "212" in by_heading["Participants"].body or "109 students" in by_heading["Participants"].body
    assert "comparison" in by_heading["Comparison group"].body.lower()
    assert by_heading["Study design and status"].label == "preprint"


def test_company_watch_lists_documented_actions(session, newsroom):
    out = _check(run_product(session, "company_watch", newsroom.brief("semiconductor")), "company_watch")
    orgs = [s.heading.split(":")[0] for s in out.sections if s.heading.endswith("documented actions")]
    assert {"Corvane Semiconductor", "Orrin Lithography", "Pinecrest Mobile"} <= set(orgs)
    assert "Calder" not in orgs
    for section in out.sections:
        if section.heading.endswith("documented actions"):
            assert re.search(r"[A-Z][a-z]{2} \d{1,2}, \d{4}:", section.body)
    assert any(s.heading == "Side by side" for s in out.sections)
    assert any("absence of a documented action" in c for c in out.caveats)


def test_supply_chain_map_is_all_hypotheses(session, newsroom):
    out = _check(
        run_product(session, "supply_chain_ripple_map", newsroom.brief("semiconductor")), "supply_chain_ripple_map"
    )
    assert {s.label for s in out.sections} == {"hypothesis"}
    headings = [s.heading for s in out.sections]
    assert any(h.startswith("Upstream") for h in headings) and any(h.startswith("Downstream") for h in headings)
    for section in out.sections:
        assert "hypothesis" in section.body.lower() or "scenario" in section.body.lower()
    assert any("not a confirmed causal effect" in c for c in out.caveats)


def test_skills_and_career_trends_are_signals(session, newsroom):
    out = _check(
        run_product(session, "skills_career_trends", newsroom.brief("semiconductor")), "skills_career_trends"
    )
    assert all(s.label.startswith("signal") for s in out.sections)
    body = " ".join(s.body for s in out.sections)
    assert "technician" in body and "16-week" in body
    assert "not a prediction" in out.labels and "not predictions" in out.summary


class _FakeClaude:
    name = "claude"
    model = "fake-claude"

    def __init__(self, output: ProductOut):
        self.output = output

    def generate(self, task, budget):
        return LLMResult(output=self.output, provider="claude", model=self.model)

    def run_tools(self, task, budget):  # pragma: no cover - not used by products
        raise NotImplementedError


def test_model_output_is_sanitized(session, newsroom):
    brief = newsroom.brief("semiconductor")
    raw = ProductOut(
        product="wrong",
        title="Ripple map",
        summary="Map [c77].",
        sections=[
            ProductSection(
                heading="Upstream",
                body="Tool makers will win big [c4][c77].",
                label="fact",
                citation_ids=["c4", "c77"],
            ),
            ProductSection(heading="Empty", body=" "),
        ],
        labels=["fact"],
    )
    row = run_product(
        session, "supply_chain_ripple_map", brief, orchestrator=Orchestrator(session, provider=_FakeClaude(raw))
    )
    out = ProductOut.model_validate(row.content)
    assert out.product == "supply_chain_ripple_map" and row.generated_by == "fake-claude"
    assert [s.heading for s in out.sections] == ["Upstream"]
    section = out.sections[0]
    assert section.label == "hypothesis" and section.citation_ids == ["c4"] and "[c77]" not in section.body
    assert out.labels[0] == "hypothesis" and out.summary == "Map."


def test_model_forecast_stated_as_fact_is_relabelled(session, newsroom):
    brief = newsroom.brief("semiconductor")
    ctx = product_context(session, PRODUCTS["company_watch"], brief)
    raw = ProductOut(
        product="company_watch",
        title="Watch",
        summary="Two companies.",
        sections=[
            ProductSection(
                heading="Corvane",
                body="Corvane's capacity will triple by 2027 [c1].",
                label="documented",
                citation_ids=["c1"],
            ),
            ProductSection(
                heading="Orrin", body="Orrin said lead times grew [c4].", label="documented", citation_ids=["c4"]
            ),
        ],
    )
    out = sanitize_product(raw, ctx)
    assert [s.label for s in out.sections] == ["projection or claim (not established)", "documented"]


def test_empty_model_output_falls_back_to_offline(session, newsroom):
    brief = newsroom.brief("policy")
    ctx = product_context(session, PRODUCTS["policy_impact_explorer"], brief)
    out = sanitize_product(ProductOut(product="x", title="", summary="", sections=[]), ctx)
    assert out.sections and out.sections[0].heading == "Who is affected?"
