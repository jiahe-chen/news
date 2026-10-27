"""Funding Round Tracker: rule-based extraction, tracker updates, trend aggregation, and the agent path."""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import func, select

from newsren.agents.base import Orchestrator
from newsren.finance.funding import (
    AGENT_NAME,
    STAGES,
    extract_funding,
    extract_funding_from_text,
    funding_trends,
    infer_sector,
    looks_like_funding,
    parse_amounts,
    update_funding_tracker,
)
from newsren.llm import LLMResult
from newsren.models import AgentRun, DocType, FundingRound, GeneratedContent, Vertical, utcnow
from newsren.schemas import FundingExtractOut

# --------------------------------------------------------------------------- extraction (table-driven)

POSITIVE = [
    (
        "Quillfield raises $45 million in a Series B led by Harrow Ventures to expand AI feedback tools for teachers",
        "Edtech startup Quillfield raised $45 million in a Series B round led by Harrow Ventures, with participation "
        "from Larchmont Capital and Saltmarsh Partners. The round brings Quillfield's total funding to $63 million.",
        dict(
            company_name="Quillfield",
            stage="series_b",
            amount_text="$45 million",
            amount_usd=45e6,
            lead_investors=["Harrow Ventures"],
            investors=["Harrow Ventures", "Larchmont Capital", "Saltmarsh Partners"],
            sector="Edtech",
        ),
    ),
    (
        "GridLoom secures $80 million Series C co-led by Aster Fund and Halden Growth",
        "Grid-software company GridLoom raised an $80 million Series C to expand software that shifts electricity "
        "demand for utilities.",
        dict(
            company_name="GridLoom",
            stage="series_c",
            amount_usd=80e6,
            lead_investors=["Aster Fund", "Halden Growth"],
            sector="Climate & energy",
        ),
    ),
    (
        "Tarnwell Compute raises $220 million Series D led by Orbit Growth Partners to build AI accelerators",
        "Tarnwell Compute, which designs chips for AI models, has raised $220 million in a Series D round led by Orbit "
        "Growth Partners. Corvane Ventures, the investment arm of Corvane Semiconductor, which manufactures "
        "Tarnwell's chips, also participated.",
        dict(
            company_name="Tarnwell Compute",
            stage="series_d_plus",
            investors=["Orbit Growth Partners", "Corvane Ventures"],
            sector="Semiconductors",
        ),
    ),
    (
        "Tessellate Robotics raises $12 million seed round led by Granite Peak Capital for warehouse picking robots",
        "Tessellate Robotics raised a $12 million seed round led by Granite Peak Capital, with Copperleaf Angels "
        "participating, to build robots that pick mixed items in warehouses.",
        dict(
            company_name="Tessellate Robotics",
            stage="seed",
            investors=["Granite Peak Capital", "Copperleaf Angels"],
            sector="Robotics",
        ),
    ),
    (
        "Acme AI raises $1.2B in growth round led by Big Fund",
        "",
        dict(
            company_name="Acme AI",
            stage="growth",
            amount_text="$1.2B",
            amount_usd=1.2e9,
            lead_investors=["Big Fund"],
        ),
    ),
    (
        "Berlin-based Klimax secures €20m Series A from Blue Ventures and Red Capital",
        "",
        dict(
            company_name="Klimax",
            stage="series_a",
            amount_text="€20m",
            amount_usd=None,
            investors=["Blue Ventures", "Red Capital"],
        ),
    ),
    (
        "Startup Foo closes £3.4 million pre-seed round",
        "",
        dict(company_name="Foo", stage="pre_seed", amount_usd=None),
    ),
    (
        "MedCo lands $8 million in venture debt from Silicon Lenders",
        "",
        dict(company_name="MedCo", stage="debt", amount_usd=8e6, investors=["Silicon Lenders"]),
    ),
    (
        "Vela Health raises 20 million euros in Series B funding",
        "",
        dict(company_name="Vela Health", stage="series_b", amount_text="20 million euros", amount_usd=None),
    ),
    (
        "PayFlow gets $15 million Series A led by Ribbit Partners, with participation from Sequoia and Accel",
        "",
        dict(
            company_name="PayFlow",
            stage="series_a",
            lead_investors=["Ribbit Partners"],
            investors=["Ribbit Partners", "Sequoia", "Accel"],
        ),
    ),
    (
        "Copperline Security raises $6.5 million seed round to monitor VPN appliances",
        "",
        dict(sector="Cybersecurity"),
    ),
    (
        "Lattice Labs raises $12 million from Granite Fund",
        "",
        dict(company_name="Lattice Labs", amount_usd=12e6, investors=["Granite Fund"]),
    ),
]


@pytest.mark.parametrize(("title", "body", "expected"), POSITIVE, ids=[p[0][:40] for p in POSITIVE])
def test_extracts_funding_rounds(title, body, expected):
    out = extract_funding_from_text(title, body)
    assert out.is_funding_round
    assert out.confidence >= 0.6
    assert out.stage in STAGES
    for field, value in expected.items():
        assert getattr(out, field) == value, field


NEGATIVE = [
    # trend roundups, opinion, earnings, fund closes, contracts, negations, prices
    ("AI education startups raised more money in the third quarter, but in fewer deals", "", None),
    ("Corvane raised its full-year revenue guidance to $3.8 billion", "", None),
    ("Harrow Ventures closes $300 million third fund", "", None),
    ("Defense startup Orion secures $50 million contract with the Army", "", None),
    ("Shares of Acme closed at $45.20 on Friday", "", None),
    ("Lumen Labs raises prices for its API", "", None),
    ("Calder district raises $20 million bond for school repairs", "Voters approved the bond.", None),
    ("Independent evaluation of Lumen-4", "The OEC received no funding from Lumen Labs.", None),
    ("Acme has not raised a Series A round yet", "", None),
    ("Regulator fines adviser $310,000 for claiming AI agents it did not use", "", None),
    ("Opinion: Quillfield raises $45 million, but what for?", "A columnist argues...", DocType.OPINION),
]


@pytest.mark.parametrize(("title", "body", "doc_type"), NEGATIVE, ids=[n[0][:40] for n in NEGATIVE])
def test_rejects_non_funding_text(title, body, doc_type):
    assert not extract_funding_from_text(title, body, doc_type=doc_type).is_funding_round


@pytest.mark.parametrize(
    ("text", "amount_text", "usd"),
    [
        ("raised $45 million", "$45 million", 45e6),
        ("raised $1.2B", "$1.2B", 1.2e9),
        ("raised €20m", "€20m", None),
        ("raised US$12m", "US$12m", 12e6),
        ("raised $500K", "$500K", 5e5),
        ("raised £3.4 million", "£3.4 million", None),
        ("raised 50 million dollars", "50 million dollars", 50e6),
        ("raised $2,500,000", "$2,500,000", 2.5e6),
    ],
)
def test_parse_amounts_converts_only_usd(text, amount_text, usd):
    amounts = parse_amounts(text)
    assert amounts and amounts[0].text == amount_text and amounts[0].usd == usd


def test_parse_amounts_ignores_small_bare_dollar_figures():
    assert parse_amounts("shares closed at $45.20") == []


@pytest.mark.parametrize(
    ("title", "sector"),
    [
        ("AI tutoring startup for teachers and classrooms raises funds", "Edtech"),
        ("Neobank for freelancers adds payments and lending", "Fintech"),
        ("AI model startup builds large language model tools", "AI & machine learning"),
        ("Clinical diagnostics company for hospitals", "Health & biotech"),
        ("Something entirely different", "Other"),
    ],
)
def test_infer_sector(title, sector):
    assert infer_sector(title) == sector


# --------------------------------------------------------------------------- tracker


@pytest.fixture()
def finance_source(factory):
    return factory.source(vertical=Vertical.FINANCE, name="Seed & Series Wire")


def make(factory, source, title, body="", **kwargs):
    return factory.article(title, body, source=source, **kwargs)


def test_update_funding_tracker_logs_confident_rounds_once(session, factory, finance_source):
    now = utcnow()
    quill = make(
        factory,
        finance_source,
        "Quillfield raises $45 million in a Series B led by Harrow Ventures",
        "Edtech startup Quillfield raised $45 million in a Series B round led by Harrow Ventures, with participation "
        "from Larchmont Capital and Saltmarsh Partners.",
        published_at=now - timedelta(days=3),
    )
    quill.is_demo = True
    make(
        factory,
        finance_source,
        "Opinion: a big round buys time",
        "Quillfield raises $45 million...",
        doc_type=DocType.OPINION,
    )
    make(factory, finance_source, "Corvane reports third-quarter revenue of $3.42 billion", "Revenue rose 18%.")
    edu = factory.source(vertical=Vertical.EDUCATION)
    make(factory, edu, "District raises $20 million bond for schools", "Voters approved the bond.")
    # Independent coverage of the same round from another outlet.
    other_outlet = factory.source(vertical=Vertical.TECHNOLOGY, name="Tech Daily")
    make(
        factory,
        other_outlet,
        "Quillfield lands $45 million Series B for teacher feedback AI",
        "Quillfield said the round was led by Harrow Ventures.",
        published_at=now - timedelta(days=2),
    )

    created = update_funding_tracker(session)
    assert [r.company_name for r in created] == ["Quillfield"]
    row = created[0]
    assert row.article_id == quill.id
    assert (row.source_url, row.source_title) == (quill.url, quill.title)
    assert row.announced_at == quill.published_at
    assert row.stage == "series_b" and row.amount_usd == 45e6 and row.amount_text == "$45 million"
    assert row.lead_investors == ["Harrow Ventures"]
    assert row.investors == ["Harrow Ventures", "Larchmont Capital", "Saltmarsh Partners"]
    assert row.sector == "Edtech" and row.is_demo and row.extraction_confidence >= 0.6

    assert update_funding_tracker(session) == []
    assert session.scalar(select(func.count()).select_from(FundingRound)) == 1


def test_update_funding_tracker_with_explicit_articles(session, factory, finance_source):
    a = make(
        factory, finance_source, "Tessellate Robotics raises $12 million seed round led by Granite Peak Capital"
    )
    b = make(factory, finance_source, "Markets close higher on Friday")
    created = update_funding_tracker(session, articles=[a, b])
    assert [r.company_name for r in created] == ["Tessellate Robotics"]
    assert update_funding_tracker(session, articles=[a]) == []


def _round(session, name, stage, usd, investors, days_ago, sector="Edtech", text=None):
    row = FundingRound(
        company_name=name,
        sector=sector,
        stage=stage,
        amount_usd=usd,
        amount_text=text or (f"${usd / 1e6:g} million" if usd else None),
        investors=investors,
        lead_investors=investors[:1],
        announced_at=utcnow() - timedelta(days=days_ago),
        source_url=f"https://news.example/{name}",
        source_title=f"{name} raises",
    )
    session.add(row)
    return row


def test_funding_trends_counts_and_sums_without_ranking_startups(session):
    _round(session, "Alpha", "seed", 5e6, ["Granite Peak", "Mosaic Angels"], 5)
    _round(session, "Beta", "series_a", 20e6, ["Granite Peak"], 10, sector="Fintech")
    _round(
        session, "Gamma", "series_a", None, ["Harrow Ventures", "Granite Peak"], 20, sector="Fintech", text="€15m"
    )
    _round(session, "Delta", "series_b", 40e6, ["Harrow Ventures"], 200)  # outside the window
    session.flush()

    trends = funding_trends(session, days=90)
    assert trends["total_rounds"] == 3
    assert trends["total_disclosed_usd"] == 25e6
    assert trends["rounds_with_disclosed_usd"] == 2
    assert trends["by_sector"] == [
        {"sector": "Fintech", "rounds": 2, "disclosed_usd": 20e6},
        {"sector": "Edtech", "rounds": 1, "disclosed_usd": 5e6},
    ]
    assert [(s["stage"], s["rounds"]) for s in trends["by_stage"]] == [("seed", 1), ("series_a", 2)]
    assert trends["by_stage"][1]["label"] == "Series A"
    assert trends["top_investors"][0] == ("Granite Peak", 3)
    assert ("Harrow Ventures", 1) in trends["top_investors"]
    assert [r["company_name"] for r in trends["recent"]] == ["Alpha", "Beta", "Gamma"], "newest first, not by size"
    assert trends["recent"][2]["amount_text"] == "€15m" and trends["recent"][2]["amount_usd"] is None
    assert "not a ranking of startups" in trends["note"]
    assert not any("rank" in key for key in trends)

    assert funding_trends(session, days=365)["total_rounds"] == 4


def test_funding_trends_can_exclude_demo_rows(session):
    _round(session, "Demo Co", "seed", 1e6, ["X"], 1).is_demo = True
    session.flush()
    assert funding_trends(session)["includes_demo"] is True
    assert funding_trends(session, include_demo=False)["total_rounds"] == 0


# --------------------------------------------------------------------------- agent path


class FakeClaude:
    """Returns a scripted FundingExtractOut, including names that are not in the article."""

    name = "claude"
    model = "claude-test"

    def __init__(self, output: FundingExtractOut):
        self.output = output
        self.calls = 0

    def generate(self, task, budget):
        self.calls += 1
        assert task.output_model is FundingExtractOut
        assert '<source id="c1">' in task.prompt
        assert "never" in task.system.lower()
        return LLMResult(output=self.output, provider="claude", model=self.model, llm_calls=1)

    def run_tools(self, task, budget):  # pragma: no cover - not used
        raise NotImplementedError


def test_offline_orchestrator_path_records_agent_runs(session, factory, finance_source):
    article = make(
        factory, finance_source, "GridLoom secures $80 million Series C co-led by Aster Fund and Halden Growth"
    )
    make(factory, finance_source, "Central bank holds rates steady")
    created = update_funding_tracker(session, orchestrator=Orchestrator(session))
    assert [r.company_name for r in created] == ["GridLoom"]
    runs = session.scalars(select(AgentRun).where(AgentRun.agent == AGENT_NAME)).all()
    assert len(runs) == 1, "articles that fail the cheap pre-filter never reach the agent"
    assert runs[0].subject_type == "article" and runs[0].subject_id == article.id


def test_agent_output_is_checked_against_the_article(session, factory, finance_source):
    article = make(
        factory,
        finance_source,
        "Klimax secures €20m Series A led by Blue Ventures",
        "Klimax, a battery recycler, secured €20m in a Series A led by Blue Ventures.",
    )
    fake = FakeClaude(
        FundingExtractOut(
            is_funding_round=True,
            company_name="Klimax",
            sector="Climate & energy",
            stage="series_a",
            amount_text="€20m",
            amount_usd=21_500_000,  # a currency conversion the article does not state
            investors=["Blue Ventures", "Imaginary Capital"],
            lead_investors=["Blue Ventures"],
            confidence=0.9,
        )
    )
    orchestrator = Orchestrator(session, provider=fake)
    created = update_funding_tracker(session, articles=[article], orchestrator=orchestrator)
    assert len(created) == 1
    row = created[0]
    assert row.investors == ["Blue Ventures"], "names absent from the article are dropped"
    assert row.amount_usd is None and row.amount_text == "€20m"
    cached = session.scalars(select(GeneratedContent).where(GeneratedContent.article_id == article.id)).all()
    assert len(cached) == 1 and cached[0].generated_by == "claude-test"

    # A model answer is cached per article version: no second call.
    from newsren.finance.funding import extract_funding_with_agent

    assert extract_funding_with_agent(session, article, orchestrator).company_name == "Klimax"
    assert fake.calls == 1


def test_agent_hallucinated_company_is_rejected(session, factory, finance_source):
    article = make(factory, finance_source, "Startup raises funding round", "A startup raised a seed round.")
    fake = FakeClaude(FundingExtractOut(is_funding_round=True, company_name="Nonexistent Labs", confidence=0.9))
    assert (
        update_funding_tracker(session, articles=[article], orchestrator=Orchestrator(session, provider=fake))
        == []
    )


def test_extract_funding_reads_article_fields(factory, finance_source):
    article = make(
        factory,
        finance_source,
        "Copperline Security raises $6.5 million seed round",
        "Cybersecurity startup Copperline Security raised $6.5 million in a seed round from Mosaic Angels Network and "
        "Fieldstone Ventures.",
    )
    assert looks_like_funding(article)
    out = extract_funding(article)
    assert out.investors == ["Mosaic Angels Network", "Fieldstone Ventures"]
    assert out.lead_investors == []
