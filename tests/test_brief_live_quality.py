"""AI Story Brief wording on real data: regressions from the live run of 2026-10-27.

The articles in ``tests/data/briefs/live_samples.json`` are copied verbatim from
the live database (545 articles from 27 approved feeds, offline LLM mode).
Publisher feeds store only a headline and a short summary; ERIC and arXiv
summaries are abstracts. Each test pins one problem the offline builder had on
that data: paper titles used as a sentence subject, abstract labels, sentence
fragments, captions and bare quotations, key points repeating the summary,
single-source stories that did not say so, uncited "what changed" text, and
cross-citations between unrelated documents.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from newsren.agents.base import build_evidence_pack
from newsren.agents.brief import (
    SUMMARY_SENTENCES,
    _context,
    _OfflineBrief,
    _story_event_types,
    generate_brief,
    remap_cited_units,
    short_point,
)
from newsren.intelligence.timeline import member_articles, refresh_story
from newsren.models import Brief, ClaimType, Story, utcnow

SAMPLES = json.loads((Path(__file__).parent / "data" / "briefs" / "live_samples.json").read_text())
ROWS = SAMPLES["articles"]
CITE = re.compile(r"\[c\d+\]")
LABELS = re.compile(
    r"(?:^|[.\]] )(?:Purpose|Background|Objectives?|Abstract|Design(?:/\w+)*|Findings|Originality(?:/value)?)\s*:"
)


def live_story(session, factory, name: str, *, refresh: bool = True) -> Story:
    """A story holding one sample storyline's live articles, stored as the feeds stored them (no body), with
    the key entities and keywords clustering gave it in the live run."""
    rows = [r for r in ROWS if r["story"] == name]
    assert rows, name
    base = utcnow() - timedelta(days=1)
    live = SAMPLES["stories"][name]
    story = Story(
        title=rows[0]["title"],
        vertical=rows[0]["vertical"],
        key_entities=live["key_entities"],
        keywords=live["keywords"],
    )
    session.add(story)
    sources = {}
    for i, row in enumerate(rows):
        key = (row["publisher"], row["doc_type"])
        if key not in sources:
            sources[key] = factory.source(
                vertical=row["vertical"],
                trust_tier=row["trust_tier"],
                doc_type=row["doc_type"],
                name=row["publisher"],
            )
        published = (
            datetime.fromisoformat(row["published_at"]) if row["published_at"] else base + timedelta(minutes=i)
        )
        article = factory.article(
            row["title"],
            "",
            source=sources[key],
            vertical=row["vertical"],
            doc_type=row["doc_type"],
            trust_tier=row["trust_tier"],
            summary=row["summary"],
            published_at=published,
        )
        article.story = story
    session.flush()
    if refresh:
        refresh_story(session, story)
    return story


def texts(brief: Brief) -> list[str]:
    return [
        brief.summary,
        brief.why_it_matters or "",
        brief.what_changed or "",
        *brief.key_points,
        *brief.uncertainty,
        *brief.watch_next,
        *(s["body"] for s in brief.sections),
        *(c.text for c in brief.claims),
    ]


def summary_sentences(brief: Brief) -> set[str]:
    parts = re.split(r"(?<=\]\.)\s+|(?<=\])\s+(?=[A-Z“])", brief.summary)
    return {CITE.sub("", p).strip().rstrip(".").strip().lower() for p in parts if p.strip()}


def plain(text: str) -> str:
    return CITE.sub("", text).strip().rstrip(".").strip().lower()


# --------------------------------------------------------------------------
# Research abstracts (ERIC)
# --------------------------------------------------------------------------


def test_two_eric_papers_brief_reads_like_a_research_digest(session, factory):
    story = live_story(session, factory, "eric_literacy")
    brief = generate_brief(session, story)

    # Before: "Schools, teachers, and families weighing Exploring AI Literacy, Exploring Generative AI Literacy,
    # Using Generative AI need to know what was actually measured, and for whom."
    assert brief.why_it_matters.startswith(
        "Schools, teachers, and families weighing these studies need to know what was actually measured, and for whom."
    )
    assert "Exploring" not in brief.why_it_matters and "Using Generative AI" not in brief.why_it_matters
    assert "Findings from one study population or setting" in brief.why_it_matters

    # Before: "the story has moved from an independent evaluation to research publication".
    assert "independent evaluation" not in brief.what_changed
    assert "A second study on the same topic was published" in brief.what_changed

    # Before: "Purpose: Current knowledge and research ..." in the summary, claims, and key points.
    for text in texts(brief):
        assert not LABELS.search(text), text
    # The summary says what the studies did, not the background an abstract opens with.
    assert brief.summary.startswith(
        "This research used survey-based research to investigate generative AI literacy"
    )
    assert "595 secondary level students" in brief.summary
    assert "…" not in brief.summary

    # Key points: distinct from the summary and short.
    summary = summary_sentences(brief)
    assert brief.key_points
    for point in brief.key_points:
        assert plain(point) not in summary, point
        assert len(CITE.sub("", point)) <= 260, point

    # A claim about one paper is not cited to the other paper because both use "students" and "generative AI".
    university = next(c for c in brief.claims if "how university students report using generative AI" in c.text)
    assert university.citation_ids == ["c1"]
    assert any("abstracts are available here" in u for u in brief.uncertainty)
    failed = [c["name"] for c in brief.check_results if not c["passed"]]
    assert "claim_support" not in failed and "study_population" not in failed, brief.check_results


def test_research_aims_scenarios_and_arguments_are_not_forecasts(session, factory):
    scenarios = generate_brief(session, live_story(session, factory, "ethical_scenarios"))
    survey = next(c for c in scenarios.claims if "six GenAI ethical scenarios" in c.text)
    assert survey.claim_type == ClaimType.FACT
    assert "projection" not in survey.text

    civic = generate_brief(session, live_story(session, factory, "civic_literacy"))
    vital = [c for c in civic.claims if "will be vital to support a healthy democracy" in c.text]
    assert vital and all(c.claim_type == ClaimType.INTERPRETATION for c in vital)
    assert all("projection or stated plan" not in t for t in texts(civic) if "healthy democracy" in t)
    assert not any("projections cited here" in w for w in civic.watch_next)


def test_first_person_in_abstracts_becomes_the_authors(session, factory):
    middle = generate_brief(session, live_story(session, factory, "unsettled_middle"))
    assert "the authors offer a critical-ethical stance" in middle.summary
    world = generate_brief(session, live_story(session, factory, "world_language"))
    # Words in quotation marks are left alone; "our on-going self-study" outside them is not.
    joined = " ".join(texts(world))
    assert "part of their on-going self-study" in joined
    assert '"core practice"' in joined
    for text in texts(middle) + texts(world):
        assert not re.search(r"\b(?:we|our|We|Our)\b", re.sub(r"“[^”]*”|\"[^\"]*\"", "", text)), text


def test_limitations_of_a_tool_are_not_the_studys_limitations(session, factory):
    brief = generate_brief(session, live_story(session, factory, "pilot_study"))
    assert not any("potentials, and limitations of generative AI tools" in u for u in brief.uncertainty)
    assert any("single study" in u for u in brief.uncertainty)
    # The pilot study's own caveat (gains did not transfer) is kept.
    assert any("did not consistently transfer" in u for u in brief.uncertainty)


# --------------------------------------------------------------------------
# Publisher feeds: headline and a short summary only
# --------------------------------------------------------------------------


def test_school_board_story_keeps_dates_whole_and_skips_captions(session, factory):
    brief = generate_brief(session, live_story(session, factory, "denver_board"))
    joined = " ".join(texts(brief))
    # Before: "... appoint a new member on Nov [c1]." and the claim "19 following the resignation of DJ Torres."
    assert "on Nov. 19 following the resignation of DJ Torres" in brief.summary
    assert not any(re.match(r"^\d+ following", c.text) for c in brief.claims)
    assert not re.search(r"\bNov \[c\d\]", joined)
    # Before: a photo caption ("... talks with students in August as school board members ... listen in").
    assert "listen in" not in joined
    # Before: "weighing Denver Superintendent Alex, Denver Public Schools Superintendent Alex Marrero, DJ Torres".
    assert "weighing" not in brief.why_it_matters and "Superintendent Alex," not in brief.why_it_matters
    assert "two news reports from Chalkbeat" in brief.why_it_matters
    assert brief.uncertainty[0].startswith("Only one source has reported this so far (Chalkbeat)")
    assert "Chalkbeat's headline: “Denver Superintendent Alex Marrero earns 88 points" in brief.summary


def test_single_source_teaser_is_attributed_and_flagged(session, factory):
    brief = generate_brief(session, live_story(session, factory, "dune_trailer"))
    # Before: the summary was the bare quotation "After years of combat, I've become the oppressor ...".
    assert (
        brief.summary == "Ars Technica's headline: “Warner Bros. drops final trailer for Dune: Part Three” [c1]."
    )
    assert "After years of combat" not in " ".join(texts(brief))
    # Before: "Claims about Dune, Warner Bros, Chani bear on developers, buyers, and institutions ..."
    assert "Claims about" not in brief.why_it_matters
    assert "the evidence so far is a single news report from Ars Technica [c1]" in brief.why_it_matters
    assert any(u.startswith("Only one source has reported this so far (Ars Technica)") for u in brief.uncertainty)
    assert any("headline and a short summary are available here" in u for u in brief.uncertainty)
    # Before: "Primary documents or further reporting that confirm or update these figures" with no figures.
    assert brief.watch_next == [
        "Reporting from other outlets, or primary documents, that confirm or update this account [c1]."
    ]
    evidence = next(s for s in brief.sections if s["title"] == "What evidence exists?")
    assert evidence["body"].startswith("The evidence so far is a single news report from Ars Technica")


def test_funding_teaser_reports_the_headline_figures_without_misnamed_subjects(session, factory):
    brief = generate_brief(session, live_story(session, factory, "disruptive_fund"))
    # Before: "For investors, founders, and finance professionals following ElevenLabs, Groq, ..."
    assert brief.why_it_matters.startswith(
        "For investors, founders, and finance professionals, the story shows how to read"
    )
    reported = next(s for s in brief.sections if s["title"] == "What was reported?")
    assert "$10B fund" in reported["body"]  # was "The sources in hand do not address this yet."
    assert "not investment advice" in brief.why_it_matters


def test_security_flaw_report_reads_as_a_security_story(session, factory):
    brief = generate_brief(session, live_story(session, factory, "uefi_flaw"))
    assert brief.why_it_matters.startswith("Organizations that run the affected software or hardware may face")
    assert "no official advisory is in the evidence yet" in brief.why_it_matters
    assert "Claims about Earth" not in brief.why_it_matters
    assert any("Updated advisories" in w for w in brief.watch_next)


def test_subject_names_come_from_proper_names_not_title_words(session, factory):
    google = generate_brief(session, live_story(session, factory, "android_stores"))
    assert "following Google," in google.why_it_matters
    budget = generate_brief(session, live_story(session, factory, "bbc_budget"))
    # Before: "Developments at Budget, Chancellor John Healey ripple through suppliers, customers, and workers."
    assert budget.why_it_matters.startswith("Developments like this can ripple through suppliers")
    assert "scenarios, not confirmed outcomes" in budget.why_it_matters


def test_question_or_short_headlines_are_not_statements(session, factory):
    budget = generate_brief(session, live_story(session, factory, "bbc_budget"))
    assert not any("what could be in the Budget?”" in c.text for c in budget.claims)
    assert any(
        c.text.startswith("BBC News - Business' headline: “Minimum wage set to rise") for c in budget.claims
    )
    # A headline that is a plan is labelled as one.
    wage = next(c for c in budget.claims if "Minimum wage set to rise" in c.text)
    assert wage.claim_type == ClaimType.FORECAST
    tech_life = generate_brief(session, live_story(session, factory, "tech_life"))
    assert "“Tech Life”" not in tech_life.summary


def test_key_points_do_not_repeat_the_summary_verbatim(session, factory):
    for name in ("eric_literacy", "three_papers", "denver_board", "disruptive_fund", "android_stores"):
        brief = generate_brief(session, live_story(session, factory, name))
        summary = summary_sentences(brief)
        repeats = [p for p in brief.key_points if plain(p) in summary]
        assert brief.key_points, name
        # A repeat is only allowed when the brief has nothing else to list.
        assert not repeats or len(brief.key_points) == 1, (name, repeats)
        assert len(brief.summary) and brief.summary.count("[c") <= SUMMARY_SENTENCES + 1


# --------------------------------------------------------------------------
# Filings and story-summary citations
# --------------------------------------------------------------------------


def test_filings_by_different_companies_do_not_corroborate_each_other(session, factory):
    story = live_story(session, factory, "edgar_filings")
    brief = generate_brief(session, story)
    knowles = next(c for c in brief.claims if c.text.startswith("Knowles Corp"))
    assert knowles.citation_ids == ["c1"]  # before: ['c1', 'c2', 'c3'] (ENSIGN's and Zurn's filings)
    joined = " ".join(texts(brief))
    assert "Source: SEC EDGAR current filings feed" not in joined
    assert "EDGAR times are filing acceptance times" not in joined
    assert brief.watch_next == ["Later filings or official releases that update this information [c1]."]


def test_story_summary_sentences_citing_documents_outside_the_pack_are_dropped():
    # The live EDGAR story's "what is new", whose citations [c10]-[c12] point past the brief's 8-document pack.
    text = (
        "Since the first report, the story has moved from a filing to earnings [c12]. On Oct 27, 2026 (earnings), "
        "SEC EDGAR current filings - Form 8-K (regulatory filing) reported: “RUSH ENTERPRISES INC \\TX\\ (CIK "
        "0001012019) filed Form 8-K (current report) with the SEC on 2026-10-27” [c12]. On Oct 27, 2026, SEC EDGAR "
        "current filings - Form 10-Q (regulatory filing) reported: “Knowles Corp filed Form 10-Q” [c1]."
    )
    kept = remap_cited_units(text, {"c1": "c1", "c2": "c2"})
    assert kept == (
        "On Oct 27, 2026, SEC EDGAR current filings - Form 10-Q (regulatory filing) reported: “Knowles Corp filed "
        "Form 10-Q” [c1]."
    )
    assert remap_cited_units(text, {"c3": "c1"}) == ""


def test_what_changed_never_carries_uncited_statements(session, factory):
    story = live_story(session, factory, "edgar_filings")
    story.whats_new = "Since the first report, the story has moved from a filing to earnings [c12]."
    brief = generate_brief(session, story)
    for sentence in re.split(r"(?<=\.)\s+(?=[A-Z])", brief.what_changed):
        assert CITE.search(sentence), sentence


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def test_short_point_trims_to_the_main_clause_without_fragments():
    # Live abstract sentences (ERIC).
    assert short_point(
        "To explore secondary level students' AI literacy, AI reliance, and ethical practices when using "
        "generative AI, the researchers collected data from a sample of 595 secondary level students from a "
        "private institution using an adopted 20-item Generative AI Literacy Assessment (GLAT) and a researcher "
        "made tool for AI reliance and ethical practices with 5-point rating scale."
    ).startswith("The researchers collected data from a sample of 595 secondary level students")
    # A list after the opening phrase is not a main clause ("Analytical, and reflective learning, ...").
    through = (
        "Through experiential, analytical, and reflective learning, students build confidence and competence in "
        "interacting with various AI tools, acquire skills in AI-human communication and prompting, and develop "
        "critical thinking abilities to identify and mitigate stereotypes generated by AI."
    )
    assert short_point(through).startswith("Through experiential, analytical, and reflective learning, students")
    # A clause inside parentheses is never cut open.
    survey = short_point(
        "The authors collected 808 survey responses on critical thinking disposition and skills, reliance "
        "behaviours (a self-developed and validated scale, including reflective use, cautious use, thoughtless "
        "use, and collaborative use), trust towards AI, and AI literacy from undergraduates after a "
        "problem-solving task with Generative AI."
    )
    assert survey.count("(") == survey.count(")")
    # A trailing participle clause goes; a projection label stays.
    assert (
        short_point(
            "Importantly, 56.7% of university guidelines posit that the opportunities of generative AI for higher "
            "education outweigh the risks, underscoring the potential transformative impact on teaching and "
            "research and the need for institutional guidance on assessment."
        )
        == "Importantly, 56.7% of university guidelines posit that the opportunities of generative AI for higher "
        "education outweigh the risks."
    )


def test_offline_builder_names_no_subject_for_research_only_stories(session, factory):
    story = live_story(session, factory, "eric_literacy")
    members = member_articles(story)
    types = _story_event_types(story, members)
    evidence = build_evidence_pack(members)
    builder = _OfflineBrief(_context(story, members, evidence, types))
    assert builder.research_only and builder.subject() is None
