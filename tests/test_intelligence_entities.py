"""Entity extraction and tagging (Story Intelligence)."""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from newsren.intelligence.entities import (
    ORG,
    entity_head,
    extract_entities,
    is_generic_name,
    normalize_entity,
    tag_article_entities,
)
from newsren.models import ArticleEntity, Entity


def kinds(text: str) -> dict[str, str]:
    return dict(extract_entities(text))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The Securities and Exchange Commission charged two advisers.", {"SEC": "organization"}),
        ("J.P. Morgan and JPMorgan Chase are the same bank.", {"JPMorgan": "organization"}),
        ("Nvidia Corp. supplies chips; NVIDIA shares rose.", {"NVIDIA": "organization"}),
        ("A paper posted to arXiv tested the model.", {"arXiv": "organization"}),
        (
            "Deutsche Bank works with Google Cloud on surveillance.",
            {"Deutsche Bank": "organization", "Google Cloud": "organization"},
        ),
        ("The U.S. Department of Education issued guidance.", {"Department of Education": "organization"}),
        (
            "FINRA and the Federal Reserve Board published reports.",
            {"FINRA": "organization", "Federal Reserve": "organization"},
        ),
        (
            "TSMC and Samsung Electronics compete with Intel.",
            {"TSMC": "organization", "Samsung": "organization", "Intel": "organization"},
        ),
    ],
)
def test_gazetteer_aliases_map_to_canonical_names(text, expected):
    found = kinds(text)
    for name, kind in expected.items():
        assert found.get(name) == kind, (name, found)


def test_tickers_are_extracted_once_and_exchange_prefix_is_not_an_entity():
    found = extract_entities("Kestrel Semiconductor (NASDAQ: KSTL) rose; $KSTL gained 4% while NYSE: ABCD fell.")
    assert ("KSTL", "ticker") in found
    assert ("ABCD", "ticker") in found
    assert sum(1 for name, _ in found if name == "KSTL") == 1
    assert "Nasdaq" not in dict(found)


def test_multiword_names_skip_sentence_initial_words_months_and_days():
    text = (
        "On Monday, researchers at the Open Evaluation Institute evaluated the model. "
        "In March 2024 the Harborview Bank pilot began. The company said demand was strong."
    )
    found = kinds(text)
    assert found["Open Evaluation Institute"] == "organization"
    assert found["Harborview Bank"] == "organization"
    for noise in ("On Monday", "Monday", "In March", "March", "The", "The company"):
        assert noise not in found


@pytest.mark.parametrize(
    ("text", "name", "kind"),
    [
        ("Schools must comply with the Every Student Succeeds Act.", "Every Student Succeeds Act", "policy"),
        ("The White House rescinded Executive Order 14110 last year.", "Executive Order 14110", "policy"),
        ("Lawmakers debated the Algorithmic Fairness Act in committee.", "Algorithmic Fairness Act", "policy"),
        ("Nimbus Labs released Atlas-3 to schools.", "Atlas-3", "product"),
        ("OpenAI shipped GPT-5 while Google released Gemini 2.5 Pro.", "Gemini 2.5 Pro", "product"),
        ("Developers compared it with Llama 4 on coding tasks.", "Llama 4", "product"),
        ("Chief Executive Maria Lopez said the model would ship in May.", "Maria Lopez", "person"),
        ("The fund is run by Kestrel Capital Partners in Ohio.", "Kestrel Capital Partners", "organization"),
        ("Kestrel Semiconductor will expand its fab in Arizona.", "Arizona", "place"),
        ("Brightpath Robotics raised money from investors.", "Brightpath Robotics", "organization"),
        (
            "Riverside Unified School District approved the pilot.",
            "Riverside Unified School District",
            "organization",
        ),
    ],
)
def test_kind_cues(text, name, kind):
    assert kinds(text).get(name) == kind


def test_generic_acronyms_are_ignored_but_specific_ones_kept():
    found = kinds("The CEO said AI and GPU demand rose, according to CISA and the OECD.")
    assert "CEO" not in found and "AI" not in found and "GPU" not in found
    assert found["CISA"] == "organization"
    assert found["OECD"] == "organization"


@pytest.mark.parametrize(
    ("text", "absent"),
    [
        # Real live items (NIST News, SEC EDGAR, SEC press): identifiers and technologies are not names.
        (
            "New Chip-Based ‘Tape Measure’ for Light Could Revolutionize Mobile Timekeeping and Enhance National "
            "Security. The new device could help provide vital navigation signals when GPS is either unavailable "
            "or has been jammed.",
            "GPS",
        ),
        (
            "HUNTINGTON BANCSHARES INC /MD/ (CIK 0000049196) filed Form 10-Q (quarterly report) with the SEC on "
            "2026-10-27, accession number 0000049196-26-000090.",
            "CIK",
        ),
        (
            "The U.S. Securities and Exchange Commission (SEC) and the French Autorité des Marchés Financiers (AMF) "
            "announced that they have entered into a Memorandum of Understanding (MOU) designed to facilitate "
            "consultation.",
            "MOU",
        ),
        (
            "SEC to Host Virtual National Compliance Outreach Seminar for Investment Companies and Investment "
            "Advisers. The Securities and Exchange Commission’s Compliance Outreach Program announced today that it "
            "will host a virtual national seminar for investment companies and investment advisers.",
            "Investment Advisers",
        ),
    ],
)
def test_generic_identifiers_and_classes_are_not_entities(text, absent):
    found = kinds(text)
    assert absent not in found
    assert "Investment Companies" not in found


def test_is_generic_name_matches_the_extractor_vocabulary():
    for name in ("GPS", "CIK", "Investment Advisers", "AI", "Investment Companies and Funds"):
        assert is_generic_name(name), name
    for name in ("SEC", "Federal Reserve", "Investment Advisers Act", "Fidelity Investments", "PayPal Holdings"):
        assert not is_generic_name(name), name


def test_generic_class_words_are_not_brand_heads():
    assert entity_head("Investment Company Institute", ORG) is None
    assert entity_head("Compliance Outreach Program", ORG) is None
    assert entity_head("Fidelity Investments", ORG) == "fidelity"
    assert entity_head("Corvane Semiconductor", ORG) == "corvane"


def test_title_case_headline_does_not_produce_topic_noise():
    found = kinds("Nimbus Labs Launches Atlas-3 Math Model for High Schools")
    assert found == {"Nimbus Labs": "organization", "Atlas-3": "product"}


def test_short_forms_resolve_to_the_full_name():
    text = (
        "Nimbus Labs released Atlas-3. Researchers at the Open Evaluation Institute (OEI) evaluated it. "
        "OEI found lower scores, and Nimbus disputed the result. Analyst Priya Raman said Raman's team agreed."
    )
    names = [name for name, _ in extract_entities(text)]
    assert "Nimbus" not in names
    assert "OEI" not in names
    assert names.count("Nimbus Labs") == 1
    assert "Open Evaluation Institute" in names


def test_empty_text():
    assert extract_entities("") == []
    assert extract_entities(None) == []


def test_normalize_entity_dedupes_case_punctuation_and_legal_suffixes():
    assert normalize_entity("Nvidia Corp.") == normalize_entity("NVIDIA")
    assert normalize_entity("The Goldman Sachs Group, Inc.") == "goldman sachs"
    assert normalize_entity("Kestrel's") == "kestrel"
    assert normalize_entity("Debevoise & Plimpton") == "debevoise and plimpton"


def test_tag_article_entities_upserts_with_title_salience(session, factory):
    art = factory.article(
        "SEC charges Harborview Bank over AI claims",
        "The Securities and Exchange Commission said Harborview Bank overstated its use of AI. "
        "FINRA also reviewed the bank's surveillance agents built with Google Cloud.",
    )
    entities = tag_article_entities(session, art)
    names = [e.name for e in entities]
    assert names[:2] == ["SEC", "Harborview Bank"]  # title mentions weigh most
    links = {link.entity.name: link.salience for link in art.entities}
    assert links["SEC"] == 1.0
    assert links["FINRA"] < links["Harborview Bank"]
    assert all(0 < s <= 1 for s in links.values())

    # Re-tagging is idempotent: no duplicate entities or links.
    count_before = session.scalar(select(func.count()).select_from(Entity))
    tag_article_entities(session, art)
    assert session.scalar(select(func.count()).select_from(Entity)) == count_before
    assert session.scalar(select(func.count()).select_from(ArticleEntity)) == len(entities)


def test_tagging_shares_entities_across_articles_and_drops_stale_links(session, factory):
    a1 = factory.article("FINRA publishes AI agent observations", "FINRA described agent risks for member firms.")
    a2 = factory.article(
        "Broker fined by FINRA", "The Financial Industry Regulatory Authority fined Acme Securities."
    )
    tag_article_entities(session, a1)
    tag_article_entities(session, a2)
    finra = session.scalars(select(Entity).where(Entity.normalized == "finra")).all()
    assert len(finra) == 1
    assert {link.article_id for link in finra[0].mentions} == {a1.id, a2.id}

    a2.title = "Broker settles case"
    a2.summary = a2.content = "Acme Securities settled the matter."
    tag_article_entities(session, a2)
    assert "FINRA" not in {link.entity.name for link in a2.entities}
    assert "Acme Securities" in {link.entity.name for link in a2.entities}
