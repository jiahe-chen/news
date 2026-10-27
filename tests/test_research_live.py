"""Ask Newsren offline answers on live-shaped data (real headlines and feed summaries).

Each test pins a problem found by asking realistic reader questions against
the live archive (545 articles from the approved sources): headline-only
regulator releases answered as "no coverage", off-topic "closest coverage"
for questions the archive cannot answer, abstract labels and datelines in
quotes, unknown dates, follow-ups built from title-case words, arXiv and
EDGAR queries that missed the question, and duplicate citations of one filing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from newsren.agents import tools
from newsren.agents.research import ask, clean_sentences, sanitize_answer
from newsren.agents.tools import HttpResponse
from newsren.models import Article, Source, utcnow
from newsren.schemas import CitationRef, ResearchAnswerOut, StatementOut
from newsren.textutil import canonical_url, content_hash, normalize_space, simhash_hex
from test_tools_ranking import load_live_articles  # shared loader for tests/data/research/live_articles.json

DATA = Path(__file__).parent / "data" / "research"


@pytest.fixture()
def live(session) -> dict[int, Article]:
    return load_live_articles(session)


def _result(row) -> ResearchAnswerOut:
    result = ResearchAnswerOut.model_validate(row.result)
    ids = {c.id for c in result.citations}
    assert [c.id for c in result.citations] == [f"c{i}" for i in range(1, len(ids) + 1)]
    assert set(re.findall(r"\[(c\d+)\]", result.answer)) <= ids
    for statement in result.verified:
        assert statement.citation_ids and set(statement.citation_ids) <= ids
    return result


def _quoted_parts(statement: str) -> list[str]:
    """The verbatim material of a statement: the quoted headline and each sentence after the attribution."""
    attribution = r"(?:\d{4}(?:-\d{2}-\d{2})?|date unknown)\)(?: (?:reports|states|says|argues))?: "
    body = re.split(attribution, statement, maxsplit=1)[-1]
    parts = re.split(r"[“”]|…|(?<=[.!?])\s+(?=[A-Z])", body)
    return [p.strip(" .") for p in parts if len(p.strip(" .")) > 15]


def _assert_verbatim(session, result: ResearchAnswerOut) -> None:
    """Every quoted headline and sentence appears in the archived article the statement cites."""
    by_id = {c.id: c for c in result.citations}
    for statement in result.verified:
        cit = by_id[statement.citation_ids[0]]
        if not cit.article_id:
            continue
        article = session.get(Article, cit.article_id)
        source_text = normalize_space(f"{article.title}. {article.summary} {article.content}")
        for part in _quoted_parts(statement.text):
            assert part in source_text, f"not in [A{article.id}]: {part!r}"


# --------------------------------------------------------------------------
# Sentence hygiene (pure)
# --------------------------------------------------------------------------


def test_clean_sentences_removes_feed_furniture(live):
    eric = clean_sentences(live[61].summary)
    assert eric[0].startswith("Current knowledge and research on students' utilization")
    assert not any(re.match(r"(Purpose|Design/methodology/approach|Findings)\s*:", s) for s in eric)
    assert "This research used survey-based research to investigate generative AI literacy" in " ".join(eric)

    # Dateline removed; "Inc. (UBS Financial)" is not a sentence break.
    assert clean_sentences(live[174].summary) == [
        "FINRA has fined UBS Financial Services Inc. (UBS Financial) $20 million for anti-money laundering (AML) "
        "violations."
    ]
    # "Sept. 29-30" is not a sentence break either.
    assert clean_sentences(live[168].summary)[0].startswith("At its meeting Sept. 29-30, the FINRA Board")
    assert clean_sentences(live[134].summary)[0].startswith("Fifty-nine school districts")

    # Sponsor notes, reprint notes, newsletter intros, and EDGAR feed metadata are not reporting.
    assert not any("brought to you by" in s for s in clean_sentences(live[319].summary))
    assert not any("originally published" in s for s in clean_sentences(live[91].summary))
    assert not any("today’s edition" in s for s in clean_sentences(live[487].summary))
    edgar = clean_sentences(live[179].summary)
    assert len(edgar) == 1 and edgar[0].startswith("PayPal Holdings, Inc. (CIK 0001633917) filed Form 10-Q")

    # A feed summary cut off mid-phrase ends at its last full clause.
    sec = clean_sentences(live[248].summary)[0]
    assert sec.endswith("from 35 investors through their fund…") and "Beyond" not in sec


def test_sanitize_keeps_paragraphs_and_list_lines():
    real = CitationRef(id="A1", url="https://news.example/a", title="T", publisher="P", article_id=1, snippet="s")
    output = ResearchAnswerOut(
        answer="Lead point [c1]\n\nAlso relevant:\n- Second  point [c1]\n- Third point [c9]",
        verified=[StatementOut(text="Lead point", citation_ids=["c1"])],
        citations=[CitationRef(id="c1", url="https://news.example/a", title="x")],
    )
    clean = sanitize_answer(output, [real])
    assert clean.answer == "Lead point [c1]\n\nAlso relevant:\n- Second point [c1]\n- Third point"


# --------------------------------------------------------------------------
# Questions the archive answers
# --------------------------------------------------------------------------


def test_headline_only_regulator_releases_answer_the_question(session, live):
    """Federal Reserve press-release items carry only a headline; they used to yield "no coverage"."""
    row = ask(session, "What did the Federal Reserve announce about enforcement actions?")
    result = _result(row)
    assert not result.answer.startswith("The Newsren archive has no coverage")
    assert len(result.verified) >= 3
    for statement in result.verified:
        assert "Federal Reserve Board" in statement.text and "enforcement action" in statement.text
        assert "“Federal Reserve Board" in statement.text, "the headline is quoted"
    assert all(c.publisher.startswith("Federal Reserve") for c in result.citations)
    assert any("headline and feed summary" in item for item in result.could_not_verify)
    _assert_verbatim(session, result)


def test_sec_charges_are_answered_directly_and_labelled_as_allegations(session, live):
    row = ask(session, "What has the SEC charged firms with recently?")
    result = _result(row)
    assert not result.answer.startswith(("The Newsren archive does not directly answer", "The retrieved sources"))
    assert len(result.citations) >= 3
    assert all(c.title.startswith("SEC Charges") for c in result.citations)
    assert "allegations" in result.answer
    assert not any("firms" in item for item in result.could_not_verify), "a generic noun is not a gap"
    assert not any("EDGAR" in item for item in result.could_not_verify), "enforcement is not a filings question"
    # "Recently": newest first.
    dates = [c.published_at for c in result.citations]
    assert dates == sorted(dates, reverse=True)
    # Follow-ups name a party from the releases, not title-case fragments.
    joined = " ".join(result.follow_up_questions)
    assert "Meyer Global Management" in joined
    assert not re.search(r"about (Securities|Exchange Commission|Owen|Kobe)\?", joined)
    _assert_verbatim(session, result)


def test_specific_terms_beat_generic_ones(session, live):
    """Before: an SEC statistics release ("published") answered a NIST post-quantum question."""
    nist = _result(ask(session, "What did NIST publish about post-quantum cryptography?"))
    assert [c.title for c in nist.citations] == [
        "NIST Report Provides Guidelines for Transition to Post-Quantum Cryptography Standards"
    ]
    assert "migration activities for federal agencies" in nist.answer, "the summary explains the headline"

    ups = _result(ask(session, "What is UPS changing about its rates?"))
    assert ups.answer.startswith(
        "Supply Chain Dive (news report, 2026-10-26): “UPS plans 5.9% rate increase on Dec. 21.”"
    )
    assert "start-ups" not in ups.answer

    kroger = _result(ask(session, "How is Kroger using RFID and computer vision?"))
    assert [c.title for c in kroger.citations] == [
        "Kroger taps RFID, computer vision data to boost inventory management"
    ]
    _assert_verbatim(session, kroger)


def test_research_abstracts_are_quoted_without_labels_and_with_years(session, live):
    row = ask(session, "What does recent research say about students' AI literacy?")
    result = _result(row)
    assert len(result.citations) >= 2 and all(
        "Literacy" in c.title or "literacy" in c.title for c in result.citations
    )
    assert not re.search(r"(Purpose|Design/methodology/approach|Findings)\s*:", result.answer)
    assert "date unknown" not in result.answer
    assert re.search(r"\(peer-reviewed research, 202[56]\)", result.answer)
    _assert_verbatim(session, result)


def test_multi_part_question_adds_a_source_for_the_missing_part(session, live):
    row = ask(session, "What education issues are on the ballot in the 2026 election?")
    result = _result(row)
    titles = [c.title for c in result.citations]
    assert titles[0] == "Election 2026: The education issues we’re watching"
    assert any("IPS school board" in t for t in titles), "the ballot part is covered by another source"
    assert not any("Electric Truck" in t for t in titles)
    assert not any("ballot" in item for item in result.could_not_verify)
    # The lede of that item is a photo caption; the next sentence is quoted instead.
    assert "running against vouchers" in result.answer and "speaks outside" not in result.answer
    _assert_verbatim(session, result)


def test_answer_is_a_lead_plus_list_with_notes(session, live):
    row = ask(session, "What is happening with the UK Budget and income tax?")
    result = _result(row)
    paragraphs = result.answer.split("\n\n")
    assert paragraphs[0].startswith("BBC News - Business (news report, 2026-10-2")
    assert paragraphs[1].startswith("Also relevant:\n- ")
    assert all("Budget" in s.text for s in result.verified)
    assert not any("“UK”" in item for item in result.could_not_verify), "a region is not a missing topic"
    _assert_verbatim(session, result)


def test_independent_evidence_gap_is_worded_as_a_property_of_the_sources(session, live):
    row = ask(session, "Is there independent evidence that AI tutors improve learning?")
    result = _result(row)
    assert result.answer.startswith("None of the retrieved sources is identified as an independent evaluation.")
    assert any(
        item.startswith("None of the retrieved sources is identified as an independent evaluation")
        for item in result.could_not_verify
    )
    assert "tutoring" in result.answer
    assert not any("Multi-Agent" in c.title for c in result.citations), "an unrelated 'independent' paper"


# --------------------------------------------------------------------------
# Questions the archive cannot answer
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "gap"),
    [
        ("What did the European Central Bank decide about interest rates?", "European Central Bank"),
        ("What is the status of NASA's Mars Sample Return mission?", "Mars Sample Return"),
    ],
)
def test_no_coverage_is_reported_without_off_topic_sources(session, live, question, gap):
    """Before: these listed unrelated "closest coverage" (startup deals, an SEC charge, an eclipse feature)."""
    result = _result(ask(session, question))
    assert result.citations == [] and result.verified == []
    assert result.answer.startswith("The Newsren archive has no coverage that addresses this question")
    assert result.could_not_verify[0].startswith("No retrieved source addresses the question")
    assert f"No archived source mentions “{gap}”." in result.could_not_verify


def test_unanswered_figure_is_not_framed_as_an_answer(session, live):
    """Before: an Audi plant's capacity was quoted for Tesla's deliveries. On the full live archive this
    question gets no coverage; on this sample the on-subject Tesla item may appear, but only as related."""
    result = _result(ask(session, "How many vehicles did Tesla deliver in the third quarter?"))
    assert [c.title for c in result.citations] == ["Tesla begins humanoid production in California"]
    assert "so the question is not answered directly. The closest related coverage:" in result.answer
    assert result.could_not_verify[0] == "None of the retrieved sources answers the question directly."
    assert any("“deliver”" in item for item in result.could_not_verify)


def test_related_coverage_is_labelled_when_the_question_is_not_answered(session, live):
    """The subject is covered but not the part asked about: say so first, then offer only on-subject items."""
    result = _result(ask(session, "What did Kroger say about its quarterly earnings?"))
    assert result.answer.startswith(
        "The retrieved sources do not address “quarterly” or “earnings”, so the question is not answered directly."
    )
    assert "closest related coverage" in result.answer
    assert [c.title for c in result.citations] == [
        "Kroger taps RFID, computer vision data to boost inventory management"
    ], "the PayPal 10-Q matches 'quarterly' but not the subject"
    assert result.could_not_verify[0] == "None of the retrieved sources answers the question directly."


def test_advice_request_on_live_data_is_refused_with_education(session, live):
    row = ask(session, "Should I buy PayPal stock?")
    result = _result(row)
    assert row.refused and result.refused and row.agent_run_id is None
    assert result.citations and result.citations[0].title.startswith("PayPal Holdings, Inc. files Form 10-Q")
    assert "buy" not in result.answer.lower().split("won't tell you whether to buy")[-1].split(".")[0]


# --------------------------------------------------------------------------
# Primary sources (network tools, replayed from saved responses)
# --------------------------------------------------------------------------


@pytest.fixture()
def fake_network(monkeypatch):
    bodies = {
        tools.ARXIV_API_URL: (DATA / "arxiv_ai_literacy.xml").read_bytes(),
        tools.SEC_TICKERS_URL: (DATA / "sec_company_tickers_sample.json").read_bytes(),
        tools.SEC_SUBMISSIONS_URL.format(cik="0001045810"): (
            DATA / "sec_submissions_nvda_sample.json"
        ).read_bytes(),
    }
    calls: list[str] = []

    def fake(url, settings, **kwargs):
        calls.append(url)
        body = next(v for k, v in bodies.items() if url.startswith(k))
        return HttpResponse(url=url, status_code=200, content_type="application/octet-stream", content=body)

    tools.clear_caches()
    monkeypatch.setattr(tools, "http_get", fake)
    yield calls
    tools.clear_caches()


def test_research_question_adds_an_arxiv_preprint(session, live, fake_network):
    row = ask(session, "What does recent research say about students' AI literacy?", allow_network=True)
    result = _result(row)
    arxiv_call = next(u for u in fake_network if u.startswith(tools.ARXIV_API_URL))
    assert "all%3A%22ai+literacy%22" in arxiv_call and "research" not in arxiv_call.split("search_query=")[1]
    papers = [c for c in result.citations if c.url.startswith("https://arxiv.org/")]
    assert papers and papers[0].doc_type == "research_preprint"
    assert any(c.article_id for c in result.citations), "archived ERIC research is still used"
    assert "is a preprint and not peer reviewed" in result.answer


def test_filing_question_finds_the_company_by_name_and_cites_the_filing_once(session, live, fake_network):
    # An EDGAR feed item for the same 10-Q that the live listing returns (accession from the saved sample).
    source = session.query(Source).filter_by(slug="sec-edgar-10q").one_or_none()
    if source is None:
        source = Source(
            slug="sec-edgar-10q",
            name="SEC EDGAR current filings - Form 10-Q",
            url="https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=10-Q",
            homepage="https://www.sec.gov/",
            vertical="finance",
            trust_tier="primary",
            default_doc_type="filing",
        )
        session.add(source)
    summary = (
        "NVIDIA CORP (CIK 0001045810) filed Form 10-Q (quarterly report) with the SEC on 2026-08-26, accession "
        "number 0001045810-26-000075. Source: SEC EDGAR current filings feed; accepted 2026-08-26T20:31:12Z. "
        "EDGAR times are filing acceptance times."
    )
    url = "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000075/0001045810-26-000075-index.htm"
    session.add(
        Article(
            source=source,
            url=url,
            canonical_url=canonical_url(url),
            title="NVIDIA CORP files Form 10-Q (quarterly report)",
            summary=summary,
            content="",
            published_at=utcnow(),
            vertical="finance",
            doc_type="filing",
            trust_tier="primary",
            content_hash=content_hash("nvda-10q", summary),
            simhash=simhash_hex(summary),
        )
    )
    session.flush()

    row = ask(session, "What did NVIDIA report in its latest 10-Q filing?", allow_network=True)
    result = _result(row)
    assert {"tool": "search_sec_filings", "input": {"company_or_ticker": "NVIDIA"}}.items() <= next(
        t for t in row.tool_trace if t["tool"] == "search_sec_filings"
    ).items()
    assert result.answer.startswith(
        "SEC EDGAR shows NVIDIA CORP's most recent Form 10-Q (quarterly report) was filed 2026-08-26 for the "
        "period ending 2026-07-26."
    )
    urls = [c.url for c in result.citations]
    assert urls[0].endswith("nvda-20260726.htm") and urls.count(urls[0]) == 1
    assert not any(c.article_id for c in result.citations), "the archived feed copy of the same 10-Q is not cited"
    assert all(c.publisher == "SEC EDGAR" for c in result.citations)
    assert any("figures inside the filings were not extracted" in item for item in result.could_not_verify)
