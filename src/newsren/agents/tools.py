"""Approved retrieval tools for the Ask Newsren research agent (reader loop).

The agent never browses freely. It gets five tools, each limited to an
approved source (plan §6: "Restrict agent access to approved source APIs and
retrieval tools"):

* ``search_archive`` — full-text search over articles The Newsren ingested
  from approved, active sources. Syndicated copies collapse onto the original.
* ``get_article`` — the (truncated) text of one archived article.
* ``search_research_papers`` — the arXiv API, an approved research index.
  Every result is labelled a preprint: arXiv does not peer review.
* ``search_sec_filings`` — SEC EDGAR company submissions (data.sec.gov),
  resolved from a ticker, CIK, or company name via SEC's ticker file.
* ``fetch_approved_url`` — one document, only when
  ``newsren.ingest.sources.is_approved_url`` allows its domain; redirects are
  followed by hand and every hop is re-checked.

Each handler returns a ``ToolResult``: numbered, human-readable text for the
model (handle, title, publisher, dates, source type, trust tier, URL, snippet)
plus ``CitationRef``s for provenance and a structured ``data`` payload for
offline builders. Handles: ``A<article_id>`` for archive articles,
``arXiv:<id>`` for papers, ``SEC:<accession>`` for filings, and the URL for
fetched pages. ``ResearchToolbox.retrieved_text`` records what each source
said, so the research agent can check the quotes a model attaches to its
citations.

Network tools honour ``allow_network``: when it is False (offline mode,
tests) they return an error result explaining that network retrieval is
disabled. All input schemas are strict (``additionalProperties: false``,
every property required, optional inputs nullable via ``anyOf`` with
``null``) because ``Tool.to_api`` sends ``strict: true``; numeric bounds are
stated in descriptions and enforced in the handlers, since strict schemas do
not support ``minimum``/``maximum``.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import httpx
from sqlalchemy import func, select
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from newsren.agents.base import article_text, citation_for
from newsren.config import Settings, get_settings
from newsren.db import search_article_ids
from newsren.ingest.fetchers import FeedParseError, FetchedItem, parse_arxiv
from newsren.ingest.sources import is_approved_url
from newsren.llm import Tool, ToolResult
from newsren.models import Article, DocType, Source, TrustTier, Vertical, utcnow
from newsren.schemas import CitationRef
from newsren.textutil import (
    best_snippet,
    canonical_url,
    domain_of,
    iso,
    normalize_space,
    parse_datetime,
    split_sentences,
    strip_html,
    tokenize,
    truncate,
)

ARCHIVE_RESULT_LIMIT = 8
ARCHIVE_CANDIDATES = 200
ARTICLE_TEXT_LIMIT = 6000
FETCH_TEXT_LIMIT = 6000
PAPER_RESULT_LIMIT = 6
FILING_RESULT_LIMIT = 10
SNIPPET_CHARS = 280
MAX_SINCE_DAYS = 3650
MAX_FETCH_BYTES = 3_000_000
MAX_JSON_BYTES = 40_000_000  # SEC submissions for large filers run to several MB
MAX_REDIRECTS = 5
SEC_TICKER_CACHE_SECONDS = 24 * 3600

ARXIV_API_URL = "https://export.arxiv.org/api/query"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

VERTICALS: tuple[str, ...] = tuple(v.value for v in Vertical)

DOC_TYPE_LABELS: dict[str, str] = {
    DocType.NEWS: "news report",
    DocType.PRESS_RELEASE: "press release (the organization's own account)",
    DocType.RESEARCH_PREPRINT: "preprint, not peer reviewed",
    DocType.RESEARCH_PEER_REVIEWED: "peer-reviewed research",
    DocType.FILING: "regulatory filing",
    DocType.EARNINGS_CALL: "earnings call transcript",
    DocType.REGULATORY: "regulator release",
    DocType.POLICY: "policy document",
    DocType.EDUCATION_REPORT: "education report",
    DocType.ADVISORY: "security advisory",
    DocType.OPINION: "opinion piece",
    DocType.BLOG: "blog post",
}

TRUST_TIER_LABELS: dict[str, str] = {
    TrustTier.PRIMARY: "primary source (the original record)",
    TrustTier.INDEPENDENT: "independent reporting or evaluation",
    TrustTier.VENDOR: "vendor (a company describing itself)",
    TrustTier.COMMENTARY: "commentary",
}

# Words that make a reader's question a question but say nothing about its
# topic; dropping them keeps archive searches and term checks focused.
QUESTION_FILLER = frozenset(
    """
    tell explain explained happen happened happening happens know think really mean means meant thing things
    anything something everything latest recent recently current currently update updates story stories
    article articles newsren please exactly actually going does did done doing use used using way ways kind
    part matter matters mattered important need needs want wants show shows showed give given find found
    sure question questions answer anyone someone people lot lots
    """.split()
)

# Primary-record domains on the allow-list and what they publish.
_PRIMARY_DOMAIN_TYPES: tuple[tuple[str, str], ...] = (
    ("sec.gov", DocType.FILING),
    ("arxiv.org", DocType.RESEARCH_PREPRINT),
    ("federalreserve.gov", DocType.REGULATORY),
    ("finra.org", DocType.REGULATORY),
    ("ies.ed.gov", DocType.EDUCATION_REPORT),
    ("nces.ed.gov", DocType.EDUCATION_REPORT),
    ("ed.gov", DocType.POLICY),
    ("cisa.gov", DocType.ADVISORY),
    ("nist.gov", DocType.REGULATORY),
)

# SEC forms that carry company news; routine forms (insider Forms 3/4/5,
# Form 144, prospectus supplements) are counted but not listed.
MATERIAL_FORMS: dict[str, str] = {
    "10-K": "annual report",
    "10-K/A": "amended annual report",
    "10-KT": "transition annual report",
    "10-Q": "quarterly report",
    "10-Q/A": "amended quarterly report",
    "8-K": "current report",
    "8-K/A": "amended current report",
    "20-F": "foreign private issuer annual report",
    "20-F/A": "amended foreign private issuer annual report",
    "40-F": "Canadian issuer annual report",
    "6-K": "foreign private issuer current report",
    "DEF 14A": "proxy statement",
    "S-1": "registration statement",
    "S-1/A": "amended registration statement",
    "F-1": "foreign issuer registration statement",
    "S-3": "shelf registration statement",
    "S-4": "registration statement for a merger or exchange offer",
    "ARS": "annual report to shareholders",
    "11-K": "employee benefit plan annual report",
}

EIGHT_K_ITEMS: dict[str, str] = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.05": "Material Cybersecurity Incidents",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.02": "Departure or Election of Directors or Officers; Compensation Arrangements",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}

_ROUTINE_FORM_LABELS: dict[str, str] = {
    "3": "insider ownership",
    "4": "insider transactions",
    "5": "insider ownership",
    "144": "proposed insider sales",
}

_ARXIV_ID = re.compile(r"^(?:arxiv:\s*)?(\d{4}\.\d{4,5})(v\d+)?$", re.IGNORECASE)
_ARXIV_FIELD = re.compile(r"\b(?:ti|au|abs|co|jr|cat|rn|id|all):", re.IGNORECASE)
_HTML_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_BOILERPLATE_BLOCKS = re.compile(r"(?is)<(nav|footer|noscript|svg|form|aside|script|style)\b.*?</\1\s*>")
_MAIN_BLOCK = re.compile(r"(?is)<main\b[^>]*>(.*)</main\s*>")
_ARTICLE_BLOCK = re.compile(r"(?is)<article\b[^>]*>(.*)</article\s*>")
_CHARSET = re.compile(r"charset=([\w-]+)", re.IGNORECASE)
_COMPANY_SUFFIXES = frozenset(
    """
    inc incorporated corp corporation co company companies ltd limited plc llc lp holdings holding group the
    sa ag nv se aktiengesellschaft
    """.split()
)


# --------------------------------------------------------------------------
# Small helpers shared with the research agent
# --------------------------------------------------------------------------


def query_terms(text: str | None) -> list[str]:
    """Distinct topic words of a query or question, in order (stopwords, filler, possessives removed)."""
    out: list[str] = []
    for token in tokenize(text):
        if token.endswith("'s") or token.endswith("’s"):
            token = token[:-2]
        if len(token) > 1 and token not in QUESTION_FILLER and token not in out:
            out.append(token)
    return out


# --------------------------------------------------------------------------
# Query analysis shared by archive ranking and the offline research builder
# --------------------------------------------------------------------------

_SUFFIXES = (
    "ational", "ations", "ation", "ately", "ating", "ated", "ates", "ate", "ings", "ing", "ions", "ion", "ments",
    "ment", "ness", "ies", "ied", "ers", "er", "es", "ed", "ly", "s", "y",
)  # fmt: skip

# Words that shape a question ("what did X announce", "is Y getting ...") or name a generic category
# ("firms", "issues") rather than its subject. They still count toward relevance, but weakly, and their
# absence is never reported as a gap.
WEAK_TERMS = frozenset(
    """
    confirm confirmed verify verified prove proved proven compare compared differ affect affected affects change
    changed changes changing work worked cause caused describe described expect expected reach reached become
    became get gets getting got make file filed announce announced announces announcing announcement publish
    published publishes publishing release released releases status decide decided decides go goes went going
    plan plans planned planning say says saying said report reported reporting happen happening tell told look
    looking stand know learn affecting doing done warn warns warned warning urge urges urged ask asks asked call
    calls called
    firm firms company companies business businesses organization organizations organisation organisations
    people group groups entity entities individual individuals thing things issue issues option options detail
    details
    """.split()
)

# Words that describe the kind of evidence asked for rather than the topic. They rank weakly; only the
# ones that name a checkable property ("independent", "peer-reviewed") are reported when unsupported.
META_TERMS = frozenset(
    """
    research researchers study studies paper papers preprint preprints evidence independent independently
    findings finding data analysis literature results evaluation evaluations peer-reviewed review reviews
    """.split()
)
UNREPORTED_META_TERMS = META_TERMS - {"independent", "independently", "peer-reviewed", "evaluation", "evaluations"}

# Acronyms and their spelled-out forms (and the reverse). Acronym aliases match case-sensitively.
TERM_ALIASES: dict[str, tuple[str, ...]] = {
    "sec": ("Securities and Exchange Commission",),
    "securities and exchange commission": ("SEC",),
    "fed": ("Federal Reserve",),
    "federal reserve": ("Fed",),
    "nist": ("National Institute of Standards and Technology",),
    "ncsc": ("National Cyber Security Centre", "National Cyber Security Center"),
    "finra": ("Financial Industry Regulatory Authority",),
    "ai": ("artificial intelligence",),
    "artificial intelligence": ("AI",),
    "ecb": ("European Central Bank",),
    "european central bank": ("ECB",),
    "uk": ("United Kingdom", "U.K.", "Britain", "British"),
    "eu": ("European Union",),
    "pqc": ("post-quantum cryptography",),
    "ipo": ("initial public offering",),
    "fcc": ("Federal Communications Commission",),
    "ftc": ("Federal Trade Commission",),
    "cisa": ("Cybersecurity and Infrastructure Security Agency",),
    "llm": ("large language model",),
    "llms": ("large language models",),
}

# Capitalized words that open a question; they are not names.
_QUESTION_OPENERS = frozenset(
    """
    what how why when where which who whom whose is are was were do does did will would can could should shall
    has have had may might must tell explain show give list please summarize summarise describe compare any
    """.split()
)
# Leading region acronyms are split off a capitalized run ("UK Budget" -> "UK" + "Budget").
_REGION_PREFIXES = frozenset({"UK", "US", "U.S.", "U.K.", "EU", "USA"})
_REGION_NAMES = frozenset(p.lower() for p in _REGION_PREFIXES)
_NAME_WORD = r"(?:[A-Z][\w\-&.]*[\w&]|[A-Z])"
_NAME_RUN = re.compile(rf"{_NAME_WORD}(?:\s+(?:(?:of|and|for|&|de|del|la|le)\s+)?{_NAME_WORD})*")
_ACRONYM_WORD = re.compile(r"^[A-Z][A-Z0-9&]{1,7}$")


def stem_word(word: str) -> str:
    """Crude suffix stripping so "evaluation"/"evaluated" and "benchmarks"/"benchmark" meet."""
    w = word.lower()
    if w.endswith("'s") or w.endswith("’s"):
        w = w[:-2]
    for suffix in _SUFFIXES:
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            return w[: -len(suffix)]
    return w


def text_stems(text: str | None) -> set[str]:
    """Stems of the topic words in ``text``."""
    return {stem_word(t) for t in tokenize(text)}


# Endings that turn a word into a related form ("robot" -> "robotic", "literate" -> "literacy"); a longer
# word that merely starts with a shorter one ("elect" -> "electric") is a different word.
_DERIVED_ENDINGS = frozenset(
    "ic ics ical al ally ist ists ism ive ives ity or ors ry ries ship e es ce ces cy ian ians".split()
)


def stem_supported(term_stem: str, stems: set[str]) -> bool:
    """Exact stem match, or a derived form ("robot" ~ "robotic") when the shorter stem has 4+ letters, is at
    least half as long as the longer one, and the longer one only adds a derivational ending."""
    if term_stem in stems:
        return True
    if len(term_stem) < 4:
        return False
    for s in stems:
        if s[:2] != term_stem[:2]:
            continue
        short, long_ = (s, term_stem) if len(s) <= len(term_stem) else (term_stem, s)
        if (
            len(short) >= 4
            and 2 * len(short) >= len(long_)
            and long_.startswith(short)
            and long_[len(short) :] in _DERIVED_ENDINGS
        ):
            return True
    return False


def _phrase_pattern(phrase: str, *, case_sensitive: bool) -> re.Pattern[str]:
    words = [re.escape(w) for w in re.findall(r"[\w&.]+", phrase)]
    body = r"[\s\-]+".join(words)
    return re.compile(rf"(?<![\w]){body}(?:['’]s)?(?![\w])", 0 if case_sensitive else re.IGNORECASE)


@dataclass(frozen=True)
class QueryUnit:
    """One thing a question asks about: a word, a multi-word name, or an acronym (with aliases)."""

    text: str  # display form, as the reader wrote it
    stem: str  # stem key (words) or lower-cased text (names, acronyms)
    kind: str  # "word" | "name" | "acronym"
    weight: float = 1.0
    weak: bool = False  # question-shaping verb ("announce", "getting")
    meta: bool = False  # evidence-type word ("research", "independent")
    patterns: tuple[re.Pattern[str], ...] = ()
    df: int | None = None  # archive documents mentioning it (None when not measured)

    @property
    def is_name(self) -> bool:
        return self.kind in ("name", "acronym")

    def matches(self, text: str, stems: set[str] | None = None) -> bool:
        """True when ``text`` mentions this unit (stems of ``text`` may be passed to save work)."""
        if not text:
            return False
        if self.patterns:
            if any(p.search(text) for p in self.patterns):
                return True
            if self.kind != "word":
                return False
        return stem_supported(self.stem, stems if stems is not None else text_stems(text))


@dataclass
class QueryProfile:
    """The weighted units of a question or search query.

    Weights are inverse document frequencies over the archive (rare words such
    as "UPS" or "post-quantum" count more than "learning" or "rates"), scaled
    down for question-shaping verbs and evidence-type words.
    """

    units: list[QueryUnit]

    @property
    def total(self) -> float:
        return sum(u.weight for u in self.units) or 1.0

    @property
    def anchors(self) -> list[QueryUnit]:
        """Units an archived source must mention to be on topic: the names, or else the most specific words.

        Units no archived article mentions cannot anchor anything (they are
        still counted, as gaps, in ``coverage``).
        """
        return self._anchors(archive=True)

    def _anchors(self, archive: bool) -> list[QueryUnit]:
        if archive and self.subject_absent:
            return []
        present = [u for u in self.units if u.df != 0] if archive else list(self.units)
        names = [u for u in present if u.is_name and not u.meta]
        if names:
            top = max(u.weight for u in names)
            # A very common name ("AI") anchors only together with the most specific other words.
            strong = [u for u in names if u.weight >= 0.5 * top]
            words = [u for u in present if not u.is_name and not u.weak and not u.meta]
            if words and top < 0.6 * max(u.weight for u in words):
                best = max(u.weight for u in words)
                strong += [u for u in words if u.weight >= 0.75 * best]
            return strong
        topical = [u for u in present if not u.weak and not u.meta] or present
        if not topical:
            return []
        best = max(u.weight for u in topical)
        return [u for u in topical if u.weight >= 0.6 * best]

    def matched(self, text: str) -> list[QueryUnit]:
        stems = text_stems(text)
        return [u for u in self.units if u.matches(text, stems)]

    def coverage(self, title: str, body: str = "", *, count_absent: bool = False) -> float:
        """Weighted share of the units mentioned (a title mention counts fully, a body mention 0.8).

        A word that no archived article contains cannot tell archived sources apart, so by default it counts
        only for a document that has it (an arXiv abstract, say); it is still reported as a gap. Names always
        count: a source that lacks the question's subject does not answer it. ``count_absent=True`` gives the
        share of the whole question instead.
        """
        if not self.units:
            return 0.0
        t_stems, b_stems = text_stems(title), text_stems(body)
        got = total = 0.0
        for u in self.units:
            if u.matches(title, t_stems):
                got += u.weight
                total += u.weight
            elif u.matches(body, b_stems):
                got += 0.8 * u.weight
                total += u.weight
            elif count_absent or not (u.kind == "word" and u.df == 0):
                total += u.weight
        return got / total if total else 0.0

    @property
    def subject_absent(self) -> bool:
        """True when the question names its subject and no archived article mentions any of those names."""
        names = [u for u in self.units if u.is_name and not u.meta and u.text.lower() not in _REGION_NAMES]
        return bool(names) and all(u.df == 0 for u in names)

    def on_topic(self, text: str, *, archive: bool = True) -> bool:
        """Whether ``text`` mentions the question's subject. ``archive=False`` for documents from outside the
        archive (arXiv, EDGAR), where archive document frequencies say nothing about what they mention."""
        if archive and self.subject_absent:
            return False
        anchors = self._anchors(archive)
        if not anchors:
            return True
        stems = text_stems(text)
        return any(u.matches(text, stems) for u in anchors)

    def fts_terms(self) -> list[str]:
        """Words for an OR full-text query: every unit word plus the words of its aliases."""
        out: list[str] = []
        for u in self.units:
            for word in [*re.findall(r"[\w\-]+", u.text), *_alias_words(u)]:
                w = word.lower()
                if len(w) > 1 and w not in out:
                    out.append(w)
        return out


def _alias_words(unit: QueryUnit) -> list[str]:
    words: list[str] = []
    for alias in TERM_ALIASES.get(unit.text.lower(), ()):
        words.extend(t for t in tokenize(alias) if t not in words)
    return words


def _name_runs(text: str) -> list[tuple[int, int, str]]:
    """(start, end, text) of capitalized runs that read as names, question openers removed."""
    runs: list[tuple[int, int, str]] = []
    for match in _NAME_RUN.finditer(text):
        words = match.group(0).split()
        # Drop leading question words ("Is Intel", "What NIST") and trailing connectors.
        opener = False
        while words and words[0].lower() in _QUESTION_OPENERS:
            words = words[1:]
            opener = True
        while words and words[-1].lower() in {"of", "and", "for", "&", "de", "del", "la", "le"}:
            words = words[:-1]
        if not words:
            continue
        sentence_start = not opener and (
            match.start() == 0 or bool(re.search(r"[.!?:]\s*$", text[: match.start()]))
        )
        if len(words) == 1 and sentence_start and not _ACRONYM_WORD.match(words[0]):
            continue  # ordinary capitalization at the start of a sentence
        # "UK Budget" -> "UK" + "Budget"
        if len(words) > 1 and words[0] in _REGION_PREFIXES:
            runs.append((match.start(), match.start() + len(words[0]), words[0]))
            words = words[1:]
        name = " ".join(words).rstrip(".")
        if len(name) >= 2:
            runs.append((match.start(), match.end(), name))
    return runs


def _fts_count(session: Session, expression: str) -> int:
    try:
        value = session.execute(
            sql_text("SELECT count(*) FROM articles_fts WHERE articles_fts MATCH :q"), {"q": expression}
        ).scalar()
    except Exception:  # noqa: BLE001 - a malformed MATCH expression must never break a search
        return 0
    return int(value or 0)


def _fts_phrase(text: str) -> str:
    words = [w for w in re.findall(r"[\w]+", text.lower()) if w]
    return '"' + " ".join(words) + '"' if words else ""


def _unit_weight(session: Session | None, unit_text: str, n_docs: int) -> tuple[float, int | None]:
    """(inverse document frequency weight, document frequency) of a unit; (1.0, None) without an archive."""
    if session is None or n_docs <= 0:
        return 1.0, None
    expressions = [_fts_phrase(unit_text), *(_fts_phrase(a) for a in TERM_ALIASES.get(unit_text.lower(), ()))]
    expression = " OR ".join(e for e in expressions if e)
    df = _fts_count(session, expression) if expression else 0
    return max(0.25, math.log((n_docs + 1) / (df + 1)) + 0.25), df


def query_profile(session: Session | None, query: str | None) -> QueryProfile:
    """Weighted units of ``query``: capitalized names and acronyms (kept whole, with aliases) plus topic words.

    With a session, each unit is weighted by its inverse document frequency in
    the archive; without one, all units weigh 1. Question-shaping verbs weigh
    0.3 and evidence-type words 0.5 of that.
    """
    text = normalize_space(query)
    units: list[QueryUnit] = []
    covered: set[str] = set()
    for _start, _end, name in _name_runs(text):
        key = name.lower()
        if key in covered or key in QUESTION_FILLER:
            continue
        covered.add(key)
        covered.update(t.lower() for t in re.findall(r"[\w\-]+", name))
        acronym = bool(_ACRONYM_WORD.match(name))
        patterns = [_phrase_pattern(name, case_sensitive=acronym)]
        for alias in TERM_ALIASES.get(key, ()):
            patterns.append(_phrase_pattern(alias, case_sensitive=bool(_ACRONYM_WORD.match(alias))))
        if acronym:
            kind = "acronym"
        elif " " in name or "-" in name:
            kind = "name"
        else:
            # A single capitalized word is a proper noun: it matches as written or in capitals ("Kroger's",
            # "KROGER"), never as a common noun ("Carrier" is not "the carrier's rate bump").
            kind = "name"
            forms = "|".join(re.escape(f) for f in dict.fromkeys([name, name.upper()]))
            patterns = [re.compile(rf"(?<![\w])(?:{forms})(?:['’]s|s)?(?![\w])"), *patterns[1:]]
        stem = stem_word(name) if kind == "name" and " " not in name and "-" not in name else key
        units.append(QueryUnit(text=name, stem=stem, kind=kind, patterns=tuple(patterns)))
    for term in query_terms(text):
        if term in covered or term.isdigit():
            continue
        covered.add(term)
        alias_patterns = tuple(
            _phrase_pattern(a, case_sensitive=bool(_ACRONYM_WORD.match(a))) for a in TERM_ALIASES.get(term, ())
        )
        units.append(
            QueryUnit(
                text=term,
                stem=stem_word(term),
                kind="word",
                weak=term in WEAK_TERMS or stem_word(term) in {stem_word(w) for w in WEAK_TERMS},
                meta=term in META_TERMS,
                patterns=alias_patterns,
            )
        )
    n_docs = 0
    if session is not None:
        try:
            n_docs = int(session.scalar(select(func.count()).select_from(Article)) or 0)
        except Exception:  # noqa: BLE001
            n_docs = 0
    weighted: list[QueryUnit] = []
    for unit in units:
        weight, df = _unit_weight(session, unit.text, n_docs)
        if unit.weak:
            weight *= 0.3
        elif unit.meta:
            weight *= 0.5
        weighted.append(
            QueryUnit(unit.text, unit.stem, unit.kind, round(weight, 3), unit.weak, unit.meta, unit.patterns, df)
        )
    return QueryProfile(weighted)


def doc_type_label(doc_type: str | None) -> str:
    """Reader-facing label for a ``DocType`` value."""
    return DOC_TYPE_LABELS.get(doc_type or "", doc_type or "source")


def trust_tier_label(trust_tier: str | None) -> str:
    """Reader-facing label for a ``TrustTier`` value."""
    return TRUST_TIER_LABELS.get(trust_tier or "", trust_tier or "unknown")


def article_handle(article_id: int) -> str:
    """The handle tools use for an archived article, e.g. ``A42``."""
    return f"A{article_id}"


def _day(value: datetime | str | None) -> str:
    dt = parse_datetime(value) if isinstance(value, str) else value
    return dt.strftime("%Y-%m-%d") if dt else ""


def _int_arg(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _error(message: str) -> ToolResult:
    return ToolResult(content=message, is_error=True)


# --------------------------------------------------------------------------
# HTTP layer (patched in tests; every request carries the descriptive User-Agent)
# --------------------------------------------------------------------------


class RetrievalError(RuntimeError):
    """A network retrieval failed or was refused (unapproved redirect, too large, ...)."""


@dataclass
class HttpResponse:
    url: str
    status_code: int
    content_type: str
    content: bytes
    truncated: bool = False

    @property
    def text(self) -> str:
        match = _CHARSET.search(self.content_type or "")
        encoding = match.group(1) if match else "utf-8"
        try:
            return self.content.decode(encoding, errors="replace")
        except LookupError:
            return self.content.decode("utf-8", errors="replace")


def http_get(
    url: str,
    settings: Settings,
    *,
    accept: str = "*/*",
    allow: Callable[[str], bool] | None = None,
    max_bytes: int = MAX_FETCH_BYTES,
    partial_ok: bool = False,
) -> HttpResponse:
    """GET ``url`` with The Newsren's User-Agent, following redirects only to URLs ``allow`` accepts.

    Reads at most ``max_bytes``: beyond that the body is cut short when
    ``partial_ok`` (and ``truncated`` is set), otherwise ``RetrievalError``.
    Network failures also raise ``RetrievalError``; HTTP error statuses are
    returned for the caller to report.
    """
    headers = {"User-Agent": settings.user_agent, "Accept": accept}
    current = url
    try:
        with httpx.Client(
            timeout=settings.fetch_timeout_seconds, headers=headers, follow_redirects=False
        ) as client:
            for _hop in range(MAX_REDIRECTS + 1):
                if allow is not None and not allow(current):
                    raise RetrievalError(f"redirected to {current}, which is not an approved source")
                with client.stream("GET", current) as response:
                    if response.is_redirect and response.headers.get("location"):
                        current = str(response.url.join(response.headers["location"]))
                        continue
                    chunks: list[bytes] = []
                    size = 0
                    truncated = False
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
                            if not partial_ok:
                                raise RetrievalError(f"response from {current} exceeds {max_bytes} bytes")
                            chunks.append(chunk[: max(0, len(chunk) - (size - max_bytes))])
                            truncated = True
                            break
                        chunks.append(chunk)
                    return HttpResponse(
                        url=str(response.url),
                        status_code=response.status_code,
                        content_type=response.headers.get("content-type", ""),
                        content=b"".join(chunks),
                        truncated=truncated,
                    )
    except httpx.HTTPError as exc:
        raise RetrievalError(f"{type(exc).__name__}: {exc}") from exc
    raise RetrievalError(f"too many redirects starting from {url}")


def _get_json(url: str, settings: Settings) -> Any:
    response = http_get(url, settings, accept="application/json", max_bytes=MAX_JSON_BYTES)
    if response.status_code == 404:
        raise RetrievalError(f"no data at {url} (HTTP 404)")
    if response.status_code != 200:
        raise RetrievalError(f"HTTP {response.status_code} from {url}")
    try:
        return json.loads(response.content)
    except ValueError as exc:
        raise RetrievalError(f"invalid JSON from {url}") from exc


# --------------------------------------------------------------------------
# Archive search
# --------------------------------------------------------------------------


def _approved_source(article: Article) -> bool:
    src = article.source
    return src is not None and bool(src.approved) and bool(src.active)


def archive_search(
    session: Session,
    query: str,
    *,
    vertical: str | None = None,
    since_days: int | None = None,
    limit: int = ARCHIVE_RESULT_LIMIT,
) -> list[Article]:
    """Best-matching archived articles from approved, active sources (duplicates collapse to the original).

    Full-text search (with acronym aliases, e.g. SEC / Securities and Exchange
    Commission) finds candidates; they are then ranked by how much of the
    query they cover, weighting rare words above common ones and headline
    mentions above body mentions, with the full-text score breaking ties.
    """
    return [article for article, _ in ranked_archive_search(session, query, vertical, since_days, limit)]


def ranked_archive_search(
    session: Session,
    query: str,
    vertical: str | None = None,
    since_days: int | None = None,
    limit: int = ARCHIVE_RESULT_LIMIT,
    profile: QueryProfile | None = None,
) -> list[tuple[Article, float]]:
    """``archive_search`` with each article's query coverage (0 to 1)."""
    terms = query_terms(query)
    if not terms:
        return []
    profile = profile or query_profile(session, query)
    fts_terms = profile.fts_terms() or terms
    hits = search_article_ids(session, " ".join(fts_terms), limit=ARCHIVE_CANDIDATES)
    if not hits:
        return []
    rows = {a.id: a for a in session.scalars(select(Article).where(Article.id.in_([aid for aid, _ in hits])))}
    cutoff = utcnow() - timedelta(days=since_days) if since_days else None
    scored: list[tuple[float, int, Article, float]] = []
    seen: set[int] = set()
    for rank, (aid, _score) in enumerate(hits):
        article = rows.get(aid)
        if article is not None and article.duplicate_of_id is not None:
            article = session.get(Article, article.duplicate_of_id)
        if article is None or article.id in seen:
            continue
        seen.add(article.id)
        if not _approved_source(article):
            continue
        if vertical and article.vertical != vertical:
            continue
        if cutoff is not None and article.display_time < cutoff:
            continue
        coverage = profile.coverage(article.title or "", _article_sentences(article))
        if not profile.on_topic(f"{article.title} {_article_sentences(article)}"):
            coverage *= 0.5
        # The full-text rank only breaks near-ties between equally covering articles.
        score = coverage + 0.05 * (1 - rank / len(hits))
        scored.append((score, rank, article, coverage))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(article, round(coverage, 3)) for _, _, article, coverage in scored[:limit]]


def published_label(article: Article) -> str:
    """Publication date as YYYY-MM-DD, the year alone when the source gives only a year, else ""."""
    published = _day(article.published_at)
    if published:
        return published
    year = (article.extra or {}).get("publication_year")
    try:
        return str(int(year)) if year else ""
    except (TypeError, ValueError):
        return ""


def _article_meta(article: Article) -> str:
    published = published_label(article)
    parts = [
        article.source.name if article.source else "",
        f"published {published}" if published else "publication date unknown",
        f"retrieved {_day(article.retrieved_at)}",
        f"type: {doc_type_label(article.doc_type)}",
        f"trust tier: {trust_tier_label(article.trust_tier)}",
        f"vertical: {article.vertical}",
    ]
    return " · ".join(p for p in parts if p)


def _evidence_notes(article: Article) -> list[str]:
    """Labelling reminders for the model (preprint, vendor claim, opinion, correction, data delay)."""
    notes: list[str] = []
    extra = article.extra or {}
    if article.doc_type == DocType.RESEARCH_PREPRINT:
        notes.append(
            extra.get("peer_review_note") or "Preprint: not peer reviewed. Do not overstate causality from it."
        )
    elif article.doc_type == DocType.RESEARCH_PEER_REVIEWED:
        notes.append("Peer-reviewed research.")
    if article.trust_tier == TrustTier.VENDOR or article.doc_type == DocType.PRESS_RELEASE:
        notes.append(
            "This is the organization's own account: attribute its claims to it and treat capability or "
            "performance claims as claims until independently checked."
        )
    if article.doc_type == DocType.OPINION or article.trust_tier == TrustTier.COMMENTARY:
        notes.append("Opinion/commentary: an interpretation, not evidence that something happened.")
    if re.match(r"\s*(correction|clarification)\b", article.title or "", re.IGNORECASE):
        notes.append("This item corrects or clarifies earlier reporting.")
    if article.source is not None and article.source.data_delay_note:
        notes.append(f"Data note: {article.source.data_delay_note}")
    return notes


def _article_sentences(article: Article) -> str:
    """Summary + body without the title, for snippets."""
    return normalize_space(f"{article.summary or ''} {article.content or ''}")


# --------------------------------------------------------------------------
# arXiv
# --------------------------------------------------------------------------


def arxiv_terms(query: str, max_terms: int = 5) -> list[str]:
    """Topic terms for an arXiv keyword search.

    Words that describe the kind of evidence ("research", "study",
    "independent evidence") or shape the question ("say", "announce") are
    dropped, because requiring them in every abstract hides relevant papers;
    a short acronym is kept together with the word after it ("AI literacy",
    "AI tutors") because on its own it matches most of arXiv.
    """
    q = normalize_space(query)
    words = [w.lower().strip("'’-") for w in re.findall(r"[A-Za-z0-9][\w\-'’]*", q)]
    words = [w[:-2] if w.endswith(("'s", "’s")) else w.rstrip("'’") for w in words]
    keep = set(query_terms(q)) - META_TERMS - WEAK_TERMS
    terms: list[str] = []
    i = 0
    while i < len(words):
        word = words[i]
        if word not in keep:
            i += 1
            continue
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if len(word) <= 3 and nxt in keep:
            terms.append(f"{word} {nxt}")
            i += 2
            continue
        terms.append(word)
        i += 1
    out: list[str] = []
    for term in terms:
        if term not in out:
            out.append(term)
    return out[:max_terms] or query_terms(q)[:max_terms]


def arxiv_query_params(query: str, max_terms: int = 5) -> dict[str, Any]:
    """arXiv API parameters for a free-text query, an arXiv id, or a query already in arXiv syntax."""
    q = normalize_space(query)
    id_match = _ARXIV_ID.match(q)
    if id_match:
        return {"id_list": id_match.group(1), "max_results": 1}
    if _ARXIV_FIELD.search(q):
        search_query = q
    else:
        terms = arxiv_terms(q, max_terms) or [q]
        search_query = " AND ".join(f'all:"{t}"' if "-" in t or " " in t else f"all:{t}" for t in terms)
    return {
        "search_query": search_query,
        "start": 0,
        "max_results": PAPER_RESULT_LIMIT,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }


def _arxiv_entry(item: FetchedItem, index: int, query: str) -> tuple[str, CitationRef]:
    extra = item.extra or {}
    arxiv_id = extra.get("arxiv_id") or item.url.rsplit("/", 1)[-1]
    authors = extra.get("authors") or []
    author_text = ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else "")
    updated = _day(extra.get("updated_at"))
    submitted = _day(item.published_at)
    status = "Preprint, not peer reviewed by arXiv."
    if extra.get("journal_ref"):
        status += f" Authors report a journal reference ({extra['journal_ref']}); not verified."
    meta = " · ".join(
        p
        for p in [
            f"authors: {author_text}" if author_text else "",
            f"submitted {submitted}" if submitted else "",
            f"last updated {updated}" if updated and updated != submitted else "",
            f"categories: {', '.join(extra.get('categories') or [])}" if extra.get("categories") else "",
        ]
        if p
    )
    version = extra.get("arxiv_version") or ""
    lines = [
        f"[{index}] [arXiv:{arxiv_id}] {item.title}" + (f" ({version})" if version else ""),
        f"    {meta}" if meta else "",
        f"    Status: {status}",
        f"    URL: {item.url}",
        f"    Abstract: {truncate(item.summary, 900)}",
    ]
    citation = CitationRef(
        id=f"arXiv:{arxiv_id}",
        url=item.url,
        title=item.title,
        publisher="arXiv",
        published_at=iso(item.published_at),
        retrieved_at=iso(utcnow()),
        snippet=best_snippet(item.summary or "", query, SNIPPET_CHARS) or truncate(item.summary, SNIPPET_CHARS),
        trust_tier=TrustTier.PRIMARY,
        doc_type=DocType.RESEARCH_PREPRINT,
    )
    return "\n".join(line for line in lines if line), citation


# --------------------------------------------------------------------------
# SEC EDGAR
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CompanyRecord:
    cik: str  # zero-padded, 10 digits
    ticker: str
    title: str


_sec_cache: dict[str, tuple[float, list[CompanyRecord]]] = {}


def clear_caches() -> None:
    """Forget the cached SEC ticker file (tests; or after SEC updates it)."""
    _sec_cache.clear()


def parse_company_tickers(data: Mapping[str, Any]) -> list[CompanyRecord]:
    """Rows of SEC's ``company_tickers.json`` in SEC's order (largest companies first)."""
    records: list[CompanyRecord] = []
    rows = data.values() if isinstance(data, Mapping) else data
    for row in rows:
        try:
            cik = str(int(row["cik_str"])).zfill(10)
            ticker = str(row["ticker"]).strip().upper()
        except (KeyError, TypeError, ValueError):
            continue
        records.append(CompanyRecord(cik=cik, ticker=ticker, title=str(row.get("title") or "")))
    return records


def _company_words(name: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", name.lower().replace("&", " and "))
    return [w for w in words if w not in _COMPANY_SUFFIXES]


def resolve_company(records: Iterable[CompanyRecord], query: str) -> list[CompanyRecord]:
    """Companies matching a ticker, CIK, or name, best first (one row per CIK); empty when nothing matches.

    Exact ticker or CIK wins; otherwise names match when every query word
    appears in the registered name (legal suffixes like "Inc" ignored), with
    exact names first.
    """
    q = normalize_space(query)
    if not q:
        return []
    records = list(records)
    # Tickers are never all digits, so a number is a CIK.
    cik_match = re.fullmatch(r"(?i)(?:cik\s*)?(\d{1,10})", q)
    if cik_match:
        cik = cik_match.group(1).zfill(10)
        return [r for r in records if r.cik == cik][:1]
    ticker = q.lstrip("$").upper().replace(".", "-").replace("/", "-")
    exact = [r for r in records if r.ticker == ticker]
    if exact:
        return exact[:1]
    words = _company_words(q)
    if not words:
        return []
    scored: list[tuple[int, int, CompanyRecord]] = []
    for order, record in enumerate(records):
        title_words = _company_words(record.title)
        if not title_words:
            continue
        if title_words == words:
            score = 3
        elif title_words[: len(words)] == words:
            score = 2
        elif all(w in title_words for w in words):
            score = 1
        else:
            continue
        scored.append((-score, order, record))
    scored.sort(key=lambda item: (item[0], item[1]))
    out: list[CompanyRecord] = []
    seen: set[str] = set()
    for _, _, record in scored:
        if record.cik not in seen:
            seen.add(record.cik)
            out.append(record)
    return out


def filing_url(cik: str, accession: str, document: str | None = None) -> str:
    """EDGAR archive URL of a filing's primary document (or its index when the document is unknown)."""
    base = f"{SEC_ARCHIVES_URL}/{int(cik)}/{accession.replace('-', '')}/"
    return f"{base}{document}" if document else base


def recent_material_filings(
    submissions: Mapping[str, Any], cik: str, limit: int = FILING_RESULT_LIMIT
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Newest material filings from a ``submissions`` payload, plus counts of the routine forms skipped."""
    recent = (submissions.get("filings") or {}).get("recent") or {}
    columns = {k: v for k, v in recent.items() if isinstance(v, list)}
    n = min((len(v) for v in columns.values()), default=0)
    rows = [{k: v[i] for k, v in columns.items()} for i in range(n)]
    rows.sort(key=lambda r: str(r.get("filingDate") or ""), reverse=True)
    filings: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for row in rows:
        form = str(row.get("form") or "").strip()
        if form not in MATERIAL_FORMS:
            skipped[form or "unknown"] += 1
            continue
        if len(filings) >= limit:
            continue
        accession = str(row.get("accessionNumber") or "")
        if not accession:
            continue
        items = [i.strip() for i in str(row.get("items") or "").split(",") if i.strip()]
        filings.append(
            {
                "form": form,
                "form_description": MATERIAL_FORMS[form],
                "filed": str(row.get("filingDate") or ""),
                "report_date": str(row.get("reportDate") or ""),
                "accession": accession,
                "items": [f"{i} {EIGHT_K_ITEMS[i]}" if i in EIGHT_K_ITEMS else i for i in items],
                "url": filing_url(cik, accession, str(row.get("primaryDocument") or "") or None),
                "index_url": filing_url(cik, accession),
                "description": str(row.get("primaryDocDescription") or ""),
            }
        )
    return filings, skipped


def html_main_text(raw_html: str) -> str:
    """Readable text of an HTML page: the ``<main>``/``<article>`` element when present, without navigation."""
    cleaned = _BOILERPLATE_BLOCKS.sub(" ", raw_html)
    for pattern in (_MAIN_BLOCK, _ARTICLE_BLOCK):
        match = pattern.search(cleaned)
        if match:
            text = strip_html(match.group(1))
            if len(text) >= 200:
                return text
    return strip_html(cleaned)


def _skipped_summary(skipped: Counter[str]) -> str:
    if not skipped:
        return ""
    parts = []
    for form, count in skipped.most_common(5):
        label = _ROUTINE_FORM_LABELS.get(form.removesuffix("/A"))
        if label is None and form.startswith("424B"):
            label = "prospectus supplements"
        parts.append(f"{count} × {form}" + (f" ({label})" if label else ""))
    rest = sum(skipped.values()) - sum(c for _, c in skipped.most_common(5))
    if rest:
        parts.append(f"{rest} other")
    return f"Not listed: {sum(skipped.values())} other filings, mostly routine ({', '.join(parts)})."


# --------------------------------------------------------------------------
# Tool schemas (strict)
# --------------------------------------------------------------------------


def _nullable(schema: dict[str, Any], description: str) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}], "description": description}


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


SEARCH_ARCHIVE_SCHEMA = _object(
    {
        "query": {
            "type": "string",
            "description": "Keywords to search for: organizations, products, people, policies, or topics. "
            "Short keyword queries work better than full questions.",
        },
        "vertical": _nullable(
            {"type": "string", "enum": list(VERTICALS)},
            "Restrict to one news section, or null for all sections.",
        ),
        "since_days": _nullable(
            {"type": "integer"},
            f"Only articles published in the last N days (1 to {MAX_SINCE_DAYS}), or null for any date.",
        ),
    }
)
GET_ARTICLE_SCHEMA = _object(
    {"article_id": {"type": "integer", "description": "The number in an archive handle, e.g. 42 for [A42]."}}
)
SEARCH_PAPERS_SCHEMA = _object(
    {
        "query": {
            "type": "string",
            "description": "Keywords (title words, method, model or dataset names), an arXiv id such as "
            "2412.20138, or arXiv query syntax such as ti:TradingAgents.",
        }
    }
)
SEARCH_FILINGS_SCHEMA = _object(
    {
        "company_or_ticker": {
            "type": "string",
            "description": "A stock ticker (NVDA), SEC CIK number, or registered company name (NVIDIA Corp).",
        }
    }
)
FETCH_URL_SCHEMA = _object(
    {"url": {"type": "string", "description": "An absolute http(s) URL on an approved source domain."}}
)


# --------------------------------------------------------------------------
# The toolbox
# --------------------------------------------------------------------------


@dataclass
class ResearchToolbox:
    """The approved tools bound to one database session, plus a record of what they retrieved."""

    session: Session
    settings: Settings = field(default_factory=get_settings)
    allow_network: bool = True
    # canonical URL -> the text the agent was shown for that source
    retrieved_text: dict[str, str] = field(default_factory=dict)

    def tools(self) -> list[Tool]:
        """The five approved retrieval tools, with strict input schemas."""
        return [
            Tool(
                name="search_archive",
                description=(
                    "Search The Newsren's archive of articles ingested from approved sources (news outlets, "
                    "regulators, research indexes, company newsrooms). Returns up to 8 numbered results, best "
                    "match first, each with an archive handle such as [A42], title, publisher, publication "
                    "and retrieval dates, source type (news, press release, preprint, filing, regulator "
                    "release, opinion, ...), trust tier, URL, and a matching snippet. Syndicated copies are "
                    "collapsed into the original. Start every research task here. Snippets are not enough to "
                    "support a claim: read the articles you rely on with get_article."
                ),
                input_schema=SEARCH_ARCHIVE_SCHEMA,
                handler=self.search_archive,
            ),
            Tool(
                name="get_article",
                description=(
                    "Read one archived article by its handle number (42 for [A42]): title, publisher, dates, "
                    "source type, trust tier, labelling notes (preprint, vendor claim, opinion, correction), "
                    "URL, and the text, truncated to about 6,000 characters."
                ),
                input_schema=GET_ARTICLE_SCHEMA,
                handler=self.get_article,
            ),
            Tool(
                name="search_research_papers",
                description=(
                    "Search arXiv (an approved research index) for research papers. Use it when the question "
                    "concerns a study, paper, model evaluation, or benchmark methodology. Returns up to 6 "
                    "papers with arXiv id, title, authors, submission date, abstract, and URL. arXiv papers "
                    "are preprints that have not been peer reviewed; say so when you rely on one."
                ),
                input_schema=SEARCH_PAPERS_SCHEMA,
                handler=self.search_research_papers,
            ),
            Tool(
                name="search_sec_filings",
                description=(
                    "List a US-listed company's most recent material SEC EDGAR filings (10-K, 10-Q, 8-K with "
                    "item descriptions, 20-F, 6-K, proxy and registration statements): form, filing date, "
                    "period, accession number, and the primary document URL. Use it for questions about "
                    "earnings, filings, or disclosures, then read a filing with fetch_approved_url."
                ),
                input_schema=SEARCH_FILINGS_SCHEMA,
                handler=self.search_sec_filings,
            ),
            Tool(
                name="fetch_approved_url",
                description=(
                    "Fetch one document from an approved source: domains of The Newsren's registered sources "
                    "and primary-record sites such as sec.gov, arxiv.org, federalreserve.gov, finra.org, and "
                    "ed.gov. Returns the page title and plain text, truncated to about 6,000 characters. Any "
                    "other domain is refused. PDFs are not supported."
                ),
                input_schema=FETCH_URL_SCHEMA,
                handler=self.fetch_approved_url,
            ),
        ]

    # ------------------------------------------------------------ bookkeeping

    def _remember(self, url: str, text: str) -> None:
        try:
            key = canonical_url(url)
        except ValueError:
            return
        existing = self.retrieved_text.get(key, "")
        if text and text not in existing:
            self.retrieved_text[key] = f"{existing}\n{text}".strip()

    def text_for(self, url: str) -> str:
        """Everything the tools showed for ``url`` in this session (empty when never retrieved)."""
        try:
            return self.retrieved_text.get(canonical_url(url), "")
        except ValueError:
            return ""

    def _network_disabled(self, tool: str, what: str) -> ToolResult:
        return _error(
            f"{tool} is unavailable: network retrieval is disabled in this deployment (offline mode), so "
            f"{what} cannot be consulted right now. Answer from The Newsren archive (search_archive, "
            f"get_article) and list anything that needs {what} under could_not_verify."
        )

    # ------------------------------------------------------------ archive

    def search_archive(self, args: dict[str, Any]) -> ToolResult:
        """Tool handler: full-text search over approved archived articles."""
        query = normalize_space(args.get("query"))
        vertical = args.get("vertical") or None
        since_days = _int_arg(args.get("since_days"))
        if vertical is not None and vertical not in VERTICALS:
            return _error(f"Unknown vertical {vertical!r}. Use one of: {', '.join(VERTICALS)}, or null.")
        if since_days is not None:
            since_days = max(1, min(since_days, MAX_SINCE_DAYS))
        if not query_terms(query):
            return _error("search_archive needs a query with at least one meaningful keyword.")

        ranked = ranked_archive_search(self.session, query, vertical=vertical, since_days=since_days)
        articles = [article for article, _ in ranked]
        relevance = {article.id: cov for article, cov in ranked}
        filters = [f"vertical={vertical}" if vertical else "", f"last {since_days} days" if since_days else ""]
        filter_text = f"; filters: {', '.join(f for f in filters if f)}" if any(filters) else ""
        header = (
            f'Archive search for "{query}": {len(articles)} result{"s" if len(articles) != 1 else ""} '
            f"(approved sources only; syndicated copies collapsed into the original{filter_text})."
        )
        if not articles:
            return ToolResult(
                content=f"{header}\nNo archived article matches. Try different or fewer keywords, or "
                "remove the filters.",
                data=[],
            )
        lines = [header, "Cite a result by its URL. Read it with get_article before relying on it.", ""]
        citations: list[CitationRef] = []
        data: list[dict[str, Any]] = []
        for index, article in enumerate(articles, start=1):
            handle = article_handle(article.id)
            snippet = (
                best_snippet(_article_sentences(article), query, SNIPPET_CHARS)
                or truncate(article.summary, SNIPPET_CHARS)
                or truncate(article.title, SNIPPET_CHARS)
            )
            lines.extend(
                [
                    f"[{index}] [{handle}] {article.title}",
                    f"    {_article_meta(article)}",
                    f"    URL: {article.url}",
                    f'    Snippet: "{snippet}"',
                ]
            )
            citation = citation_for(article, handle, snippet)
            citations.append(citation)
            self._remember(article.url, f"{article.title}\n{snippet}")
            data.append(
                {
                    "article_id": article.id,
                    "handle": handle,
                    "title": article.title,
                    "url": article.url,
                    "snippet": snippet,
                    "vertical": article.vertical,
                    "doc_type": article.doc_type,
                    "trust_tier": article.trust_tier,
                    "published_at": iso(article.published_at),
                    "date_label": published_label(article),
                    "story_id": article.story_id,
                    "relevance": relevance.get(article.id, 0.0),
                    "citation": citation,
                }
            )
        return ToolResult(content="\n".join(lines), citations=citations, data=data)

    def get_article(self, args: dict[str, Any]) -> ToolResult:
        """Tool handler: one archived article's text (truncated) with labels and its citation."""
        article_id = _int_arg(args.get("article_id"))
        if article_id is None:
            return _error("get_article needs an integer article_id (42 for [A42]).")
        article = self.session.get(Article, article_id)
        if article is None:
            return _error(f"No archived article has id {article_id}. Use search_archive to find article handles.")
        notes: list[str] = []
        if article.duplicate_of_id is not None:
            original = self.session.get(Article, article.duplicate_of_id)
            if original is not None:
                notes.append(
                    f"[A{article.id}] is a syndicated copy; showing the original "
                    f"[{article_handle(original.id)}] instead."
                )
                article = original
        if not _approved_source(article):
            return _error(f"Article [A{article.id}] is not from an approved, active source and cannot be used.")

        handle = article_handle(article.id)
        full = article_text(article, max_chars=10**9)
        text = article_text(article, max_chars=ARTICLE_TEXT_LIMIT)
        notes.extend(_evidence_notes(article))
        if len(full) > len(text):
            notes.append(f"Text truncated to {ARTICLE_TEXT_LIMIT} of {len(full)} characters.")
        lines = [
            f"[{handle}] {article.title}",
            _article_meta(article),
            f"URL: {article.url}",
            *(f"Note: {n}" for n in notes),
            "",
            text,
        ]
        self._remember(article.url, text)
        citation = citation_for(article, handle)
        data = {
            "article_id": article.id,
            "handle": handle,
            "title": article.title,
            "url": article.url,
            "publisher": article.source.name if article.source else "",
            "published_at": iso(article.published_at),
            "date_label": published_label(article),
            "doc_type": article.doc_type,
            "trust_tier": article.trust_tier,
            "vertical": article.vertical,
            "story_id": article.story_id,
            "summary": truncate(article.summary, ARTICLE_TEXT_LIMIT),
            "content": truncate(article.content, ARTICLE_TEXT_LIMIT),
            "text": text,
            "citation": citation,
        }
        return ToolResult(content="\n".join(lines), citations=[citation], data=data)

    # ------------------------------------------------------------ arXiv

    def search_research_papers(self, args: dict[str, Any]) -> ToolResult:
        """Tool handler: arXiv API search; every result is labelled a preprint."""
        query = normalize_space(args.get("query"))
        if not query:
            return _error("search_research_papers needs a query.")
        if not self.allow_network:
            return self._network_disabled("search_research_papers", "arXiv")
        params = arxiv_query_params(query)
        items, error = self._arxiv(params)
        broadened = ""
        if error is None and not items and "search_query" in params and not _ARXIV_FIELD.search(query):
            terms = arxiv_terms(query)
            if len(terms) > 2:
                narrower = " AND ".join(f'all:"{t}"' if "-" in t or " " in t else f"all:{t}" for t in terms[:2])
                params = {**params, "search_query": narrower}
                items, error = self._arxiv(params)
                broadened = f' No paper matched every keyword; broadened to "{" ".join(terms[:2])}".'
        if error is not None:
            return _error(error)
        header = (
            f'arXiv search for "{query}": {len(items)} paper{"s" if len(items) != 1 else ""} '
            f"(arXiv API, an approved research index).{broadened}"
        )
        if not items:
            return ToolResult(content=f"{header}\nNo papers found. Try other keywords or an arXiv id.", data=[])
        lines = [
            header,
            'Every arXiv paper is a preprint: arXiv does not peer review submissions. Say "preprint" when '
            "you rely on one and do not overstate causality.",
            "",
        ]
        citations: list[CitationRef] = []
        data: list[dict[str, Any]] = []
        for index, item in enumerate(items[:PAPER_RESULT_LIMIT], start=1):
            block, citation = _arxiv_entry(item, index, query)
            lines.extend([block, ""])
            citations.append(citation)
            self._remember(item.url, f"{item.title}\n{item.summary}")
            data.append(
                {
                    "arxiv_id": citation.id.removeprefix("arXiv:"),
                    "title": item.title,
                    "url": item.url,
                    "summary": item.summary or "",
                    "authors": (item.extra or {}).get("authors") or [],
                    "published_at": citation.published_at,
                    "doc_type": DocType.RESEARCH_PREPRINT,
                    "trust_tier": TrustTier.PRIMARY,
                    "citation": citation,
                }
            )
        return ToolResult(content="\n".join(lines).rstrip(), citations=citations, data=data)

    def _arxiv(self, params: dict[str, Any]) -> tuple[list[FetchedItem], str | None]:
        url = f"{ARXIV_API_URL}?{urlencode(params)}"
        try:
            response = http_get(url, self.settings, accept="application/atom+xml")
        except RetrievalError as exc:
            return [], f"arXiv could not be reached: {exc}"
        try:
            items = parse_arxiv(response.content)
        except FeedParseError as exc:
            return [], f"arXiv rejected the query: {exc}"
        if response.status_code != 200 and not items:
            return [], f"arXiv returned HTTP {response.status_code}."
        return items, None

    # ------------------------------------------------------------ SEC EDGAR

    def _sec_companies(self) -> list[CompanyRecord]:
        cached = _sec_cache.get(SEC_TICKERS_URL)
        if cached and time.monotonic() - cached[0] < SEC_TICKER_CACHE_SECONDS:
            return cached[1]
        records = parse_company_tickers(_get_json(SEC_TICKERS_URL, self.settings))
        if records:
            _sec_cache[SEC_TICKERS_URL] = (time.monotonic(), records)
        return records

    def search_sec_filings(self, args: dict[str, Any]) -> ToolResult:
        """Tool handler: a company's recent material SEC EDGAR filings."""
        name = normalize_space(args.get("company_or_ticker"))
        if not name:
            return _error("search_sec_filings needs a ticker, CIK, or company name.")
        if not self.allow_network:
            return self._network_disabled("search_sec_filings", "SEC EDGAR")
        try:
            matches = resolve_company(self._sec_companies(), name)
        except RetrievalError as exc:
            return _error(f"SEC EDGAR could not be reached: {exc}")
        if not matches:
            return _error(
                f'SEC EDGAR lists no company matching "{name}". Try the exact ticker (e.g. NVDA) or the '
                "registered company name. Companies that are not SEC registrants have no EDGAR filings."
            )
        company = matches[0]
        try:
            submissions = _get_json(SEC_SUBMISSIONS_URL.format(cik=company.cik), self.settings)
        except RetrievalError as exc:
            return _error(f"SEC EDGAR submissions for CIK {company.cik} could not be retrieved: {exc}")
        filings, skipped = recent_material_filings(submissions, company.cik)
        registered = str(submissions.get("name") or company.title)
        tickers = submissions.get("tickers") or [company.ticker]
        retrieved = iso(utcnow())
        lines = [
            f"SEC EDGAR filings for {registered} (ticker {', '.join(map(str, tickers[:3]))}, CIK {company.cik}); "
            f"primary source: data.sec.gov submissions, retrieved {retrieved}. EDGAR lists filings after "
            "acceptance; this is not real-time market data.",
        ]
        if len(matches) > 1:
            others = "; ".join(f"{m.title} ({m.ticker})" for m in matches[1:4])
            lines.append(f'Other companies matching "{name}": {others}.')
        if not filings:
            lines.append("No recent 10-K, 10-Q, 8-K, or other material filings are listed.")
        else:
            lines.append(f"The {len(filings)} most recent material filings, newest first:")
        skipped_text = _skipped_summary(skipped)
        if skipped_text:
            lines.append(skipped_text)
        lines.append("")
        citations: list[CitationRef] = []
        for index, filing in enumerate(filings, start=1):
            period = f", period ending {filing['report_date']}" if filing["report_date"] else ""
            items = f"\n    Items: {'; '.join(filing['items'])}" if filing["items"] else ""
            lines.append(
                f"[{index}] [SEC:{filing['accession']}] Form {filing['form']} ({filing['form_description']}) "
                f"filed {filing['filed']}{period}{items}\n    URL: {filing['url']}"
            )
            summary = (
                f"{registered} Form {filing['form']} ({filing['form_description']}) filed {filing['filed']}"
                f"{period}" + (f"; items {'; '.join(filing['items'])}" if filing["items"] else "")
            )
            citations.append(
                CitationRef(
                    id=f"SEC:{filing['accession']}",
                    url=filing["url"],
                    title=f"{registered} Form {filing['form']} ({filing['form_description']}), filed {filing['filed']}",
                    publisher="SEC EDGAR",
                    published_at=iso(parse_datetime(filing["filed"])),
                    retrieved_at=retrieved,
                    snippet=truncate(summary, SNIPPET_CHARS),
                    trust_tier=TrustTier.PRIMARY,
                    doc_type=DocType.FILING,
                )
            )
            self._remember(filing["url"], summary)
        if filings:
            lines.extend(
                [
                    "",
                    "Read a filing with fetch_approved_url(url); sec.gov is an approved primary source. Quote the "
                    "filing for every number you report.",
                ]
            )
        data = {"company": registered, "cik": company.cik, "tickers": tickers, "filings": filings}
        return ToolResult(content="\n".join(lines).rstrip(), citations=citations, data=data)

    # ------------------------------------------------------------ approved URLs

    def _source_for_domain(self, domain: str) -> Source | None:
        for src in self.session.scalars(select(Source).where(Source.approved.is_(True), Source.active.is_(True))):
            for url in (src.homepage, src.url):
                d = domain_of(url or "")
                if d and (domain == d or domain.endswith("." + d)):
                    return src
        return None

    def _page_citation(self, url: str, title: str, text: str) -> CitationRef:
        try:
            article = self.session.scalars(
                select(Article).where(Article.canonical_url == canonical_url(url))
            ).first()
        except ValueError:
            article = None
        if article is not None and article.duplicate_of_id is None:
            return citation_for(article, article_handle(article.id))
        domain = domain_of(url)
        source = self._source_for_domain(domain)
        doc_type = next(
            (dt for d, dt in _PRIMARY_DOMAIN_TYPES if domain == d or domain.endswith("." + d)),
            source.default_doc_type if source else DocType.NEWS,
        )
        is_primary_domain = any(domain == d or domain.endswith("." + d) for d, _ in _PRIMARY_DOMAIN_TYPES)
        trust = (
            TrustTier.PRIMARY if is_primary_domain else (source.trust_tier if source else TrustTier.INDEPENDENT)
        )
        lead = next((s for s in split_sentences(text) if len(s) > 60), "") or text
        return CitationRef(
            id=url,
            url=url,
            title=title or url,
            publisher=source.name if source else domain,
            published_at=None,
            retrieved_at=iso(utcnow()),
            snippet=truncate(lead, SNIPPET_CHARS),
            trust_tier=trust,
            doc_type=doc_type,
        )

    def fetch_approved_url(self, args: dict[str, Any]) -> ToolResult:
        """Tool handler: fetch one page from an approved domain (redirects re-checked) as plain text."""
        url = normalize_space(args.get("url"))
        if not url.lower().startswith(("http://", "https://")):
            return _error("fetch_approved_url needs an absolute http(s) URL.")
        if not is_approved_url(self.session, url):
            return _error(
                f"Refused: {url} is not an approved source. Only domains of The Newsren's registered sources "
                "and primary-record sites (for example sec.gov, arxiv.org, federalreserve.gov, finra.org, "
                "ed.gov) can be fetched. Use search_archive for coverage from other outlets."
            )
        if not self.allow_network:
            return self._network_disabled("fetch_approved_url", "live web pages")
        try:
            response = http_get(
                url,
                self.settings,
                accept="text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.8,*/*;q=0.5",
                allow=lambda u: is_approved_url(self.session, u),
                partial_ok=True,
            )
        except RetrievalError as exc:
            return _error(f"Could not fetch {url}: {exc}")
        if response.status_code >= 400:
            return _error(f"{url} returned HTTP {response.status_code}.")
        content_type = response.content_type.lower()
        if "pdf" in content_type or response.content[:5] == b"%PDF-":
            return _error(
                f"{url} is a PDF, which this tool cannot read. Look for an HTML version (for SEC filings, the "
                ".htm primary document listed by search_sec_filings)."
            )
        raw = response.text.lstrip("\ufeff")
        if "html" in content_type or raw.lstrip()[:200].lower().startswith(("<!doctype html", "<html")):
            match = _HTML_TITLE.search(raw)
            title = normalize_space(strip_html(match.group(1))) if match else ""
            text = html_main_text(raw)
        elif any(t in content_type for t in ("text/", "json", "xml")) or not content_type:
            title, text = "", raw
        else:
            return _error(f"{url} has an unsupported content type ({response.content_type}).")
        text = normalize_space(text)
        if not text:
            return _error(f"{url} returned no readable text.")
        shown = truncate(text, FETCH_TEXT_LIMIT)
        final_url = response.url or url
        citation = self._page_citation(final_url, title, text)
        self._remember(url, shown)
        self._remember(final_url, shown)
        notes = []
        if final_url != url:
            notes.append(f"Redirected to {final_url}.")
        if len(text) > len(shown) or response.truncated:
            notes.append(f"Text truncated to {FETCH_TEXT_LIMIT} characters.")
        lines = [
            f"{title or final_url}",
            f"{citation.publisher} · retrieved {_day(utcnow())} · type: {doc_type_label(citation.doc_type)} · "
            f"trust tier: {trust_tier_label(citation.trust_tier)}",
            f"URL: {final_url}",
            *(f"Note: {n}" for n in notes),
            "",
            shown,
        ]
        data = {"url": final_url, "title": title, "text": shown, "citation": citation}
        return ToolResult(content="\n".join(lines), citations=[citation], data=data)


def build_research_tools(
    session: Session, settings: Settings | None = None, allow_network: bool = True
) -> list[Tool]:
    """The Ask Newsren tools (archive, article, arXiv, SEC EDGAR, approved URL) with strict JSON schemas.

    With ``allow_network=False`` the arXiv, EDGAR, and URL tools return an
    error result explaining that network retrieval is disabled.
    """
    return ResearchToolbox(session, settings or get_settings(), allow_network).tools()
