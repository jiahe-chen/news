"""Regulation Watch (plan §3, finance products): who is affected, what changes, and when.

``regulation_watch`` lists recent regulatory and enforcement actions found in
ingested articles: regulator documents (SEC, FINRA, Federal Reserve releases,
state and federal education agencies, and the demo corpus's fictional
regulators) plus news reports of such actions. Each item says which regulator
acted, what kind of action it was (enforcement, rule, guidance, statement)
and at what stage (proposed, final, settled), quotes the sentences that say
who is affected and what changes, and parses compliance, effective, and
comment dates.

Extraction is heuristic and deterministic: sentences are quoted from the
source, never paraphrased, so every line can be checked against its
citation. Dates given relative to the publication date ("comments are due
within 60 days") are converted and marked as estimates; dates relative to a
future event ("six months after the final notice") are kept as phrases with
no date. News reports of the same action are attached to the regulator's own
document as coverage instead of being listed twice. Nothing here is legal or
investment advice; the regulator's own text sets the obligations.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from newsren.agents.base import citation_for
from newsren.intelligence.events import classify_event
from newsren.models import Article, DocType, EventType, TrustTier, utcnow
from newsren.textutil import iso, jaccard, normalize_space, split_sentences, tokenize, truncate

REGULATOR_GROUPS: tuple[str, ...] = ("SEC", "FINRA", "Federal Reserve", "Department of Education", "other")
ACTION_TYPES: tuple[str, ...] = ("enforcement", "rule", "guidance", "statement")
ACTION_LABELS: dict[str, str] = {
    "enforcement": "Enforcement action",
    "rule": "Rule or requirement",
    "guidance": "Guidance",
    "statement": "Statement",
}
STAGE_LABELS: dict[str, str] = {
    "proposed": "Proposed (not final)",
    "final": "Final or adopted",
    "settled": "Settled",
    "announced": "Announced",
}
# Regulator kinds that belong on the Finance page.
FINANCE_REGULATOR_KINDS = frozenset(
    {
        "securities regulator",
        "self-regulatory organization",
        "central bank",
        "banking regulator",
        "derivatives regulator",
        "consumer finance regulator",
    }
)
WATCH_NOTE = (
    "Regulation Watch quotes regulators' own documents and news reports of their actions. It is news and "
    "education, not legal advice; the regulator's text sets the obligations, and proposals can change before "
    "they are final."
)
MAX_SENTENCES = 3
MAX_ITEMS = 50

_SKIP_DOC_TYPES = frozenset(
    {
        DocType.OPINION.value,
        DocType.FILING.value,
        DocType.EARNINGS_CALL.value,
        DocType.ADVISORY.value,
        DocType.RESEARCH_PREPRINT.value,
        DocType.RESEARCH_PEER_REVIEWED.value,
        DocType.BLOG.value,
    }
)
_REGULATOR_DOC_TYPES = frozenset({DocType.REGULATORY.value, DocType.POLICY.value})
_ACTION_EVENTS = frozenset({EventType.REGULATION.value, EventType.ENFORCEMENT.value, EventType.POLICY.value})


# --------------------------------------------------------------------------
# Regulators
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Regulator:
    name: str
    group: str
    kind: str
    pattern: re.Pattern[str]


_KNOWN_REGULATORS: tuple[_Regulator, ...] = (
    _Regulator(
        "SEC", "SEC", "securities regulator", re.compile(r"\bSecurities and Exchange Commission\b|\bSEC\b")
    ),
    _Regulator(
        "FINRA",
        "FINRA",
        "self-regulatory organization",
        re.compile(r"\bFINRA\b|\bFinancial Industry Regulatory Authority\b"),
    ),
    _Regulator(
        "Federal Reserve",
        "Federal Reserve",
        "central bank",
        re.compile(r"\bFederal Reserve\b|\bFederal Open Market Committee\b|\bFOMC\b|\bthe Fed\b"),
    ),
    _Regulator(
        "Department of Education",
        "Department of Education",
        "education agency",
        re.compile(r"\b(?:U\.S\.\s+)?Department of Education\b|\bEducation Department\b"),
    ),
    _Regulator(
        "CFTC",
        "other",
        "derivatives regulator",
        re.compile(r"\bCommodity Futures Trading Commission\b|\bCFTC\b"),
    ),
    _Regulator(
        "CFPB",
        "other",
        "consumer finance regulator",
        re.compile(r"\bConsumer Financial Protection Bureau\b|\bCFPB\b"),
    ),
    _Regulator(
        "OCC",
        "other",
        "banking regulator",
        re.compile(r"\bOffice of the Comptroller of the Currency\b|\bOCC\b"),
    ),
    _Regulator(
        "FDIC", "other", "banking regulator", re.compile(r"\bFederal Deposit Insurance Corporation\b|\bFDIC\b")
    ),
    _Regulator("FTC", "other", "consumer protection agency", re.compile(r"\bFederal Trade Commission\b|\bFTC\b")),
    _Regulator("European Central Bank", "other", "central bank", re.compile(r"\bEuropean Central Bank\b|\bECB\b")),
    _Regulator("Bank of England", "other", "central bank", re.compile(r"\bBank of England\b")),
)

# Named agencies ("Markets Integrity Commission", "Calder State Board of Education").
_AGENCY = re.compile(
    r"\b((?:[A-Z][\w'’-]*\s+){1,5}(?:Commission|Authority|Agency|Bureau|Council|Administration|Office|"
    r"Board(?:\s+of\s+(?:[A-Z]\w+\s?)+)?|Department(?:\s+of\s+(?:[A-Z]\w+\s?)+)?))"
)
_LEADING_ARTICLE = re.compile(r"^(?:The|A|An|This|That)\s+")
_KIND_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("education agency", re.compile(r"\beducation\b|\bschools?\b|\bdistricts?\b|\bstudents?\b", re.IGNORECASE)),
    (
        "central bank",
        re.compile(r"\bcentral bank\b|\bmonetary\b|\breserve bank\b|\binterest rates?\b", re.IGNORECASE),
    ),
    (
        "self-regulatory organization",
        re.compile(r"\bmember (?:firms?|broker-dealers?)\b|\bself-regulatory\b|\bbroker-dealer\b", re.IGNORECASE),
    ),
    (
        "securities regulator",
        re.compile(
            r"\bsecurities\b|\binvestment advis[eo]rs?\b|\bregistered investment\b|\bmarkets?\b|\binvestors?\b",
            re.IGNORECASE,
        ),
    ),
    ("banking regulator", re.compile(r"\bbanks?\b|\bbanking\b|\bdeposit\b", re.IGNORECASE)),
)


def _agency_kind(name: str, context: str) -> str:
    for kind, rx in _KIND_CUES:
        if rx.search(name):
            return kind
    for kind, rx in _KIND_CUES:
        if rx.search(context):
            return kind
    return "regulator"


def _named_agency(text: str) -> str | None:
    for m in _AGENCY.finditer(text or ""):
        name = _LEADING_ARTICLE.sub("", normalize_space(m.group(1)))
        if len(name.split()) >= 2:
            return name
    return None


def _known(text: str) -> _Regulator | None:
    hits = [(m.start(), r) for r in _KNOWN_REGULATORS if (m := r.pattern.search(text or ""))]
    return min(hits, key=lambda h: h[0])[1] if hits else None


def identify_regulator(article: Article) -> tuple[str, str, str] | None:
    """(name, group, kind) of the regulator behind an article, or None when no regulator is identified.

    ``group`` is one of ``REGULATOR_GROUPS`` (SEC, FINRA, Federal Reserve,
    Department of Education, other). Known regulators named in the title
    come first, then the publishing source when it is itself a regulator
    (a primary regulatory or policy document), then regulators and named
    agencies in the summary and body.
    """
    title, summary, content = article.title or "", article.summary or "", article.content or ""
    context = f"{title} {summary} {content[:1500]}"
    if (reg := _known(title)) is not None:
        return reg.name, reg.group, reg.kind
    source = article.source
    if source is not None and article.trust_tier == TrustTier.PRIMARY and article.doc_type in _REGULATOR_DOC_TYPES:
        if (reg := _known(source.name)) is not None:
            return reg.name, reg.group, reg.kind
        return source.name, "other", _agency_kind(source.name, context)
    for text in (summary, content):
        if (reg := _known(text)) is not None:
            return reg.name, reg.group, reg.kind
        if (name := _named_agency(text)) is not None:
            if (reg := _known(name)) is not None:
                return reg.name, reg.group, reg.kind
            return name, "other", _agency_kind(name, context)
    if (name := _named_agency(title)) is not None:
        return name, "other", _agency_kind(name, context)
    return None


# --------------------------------------------------------------------------
# Action type and stage
# --------------------------------------------------------------------------

_ACTION_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "enforcement",
        re.compile(
            r"\bcharg(?:es|ed|ing)\b|\bsettle(?:s|d|ment)?\b|\bpenalt(?:y|ies)\b|\bfine[sd]?\b|\bcensure[sd]?\b|"
            r"\bcease[- ]and[- ]desist\b|\bsanction(?:s|ed)?\b|\bdisgorgement\b|\benforcement\s+action\b|"
            r"\bbarred\b|\bexpelled\b|\bconsent\s+order\b|\bsue[sd]\b|\blawsuit\b|\bindict(?:ed|ment)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "guidance",
        re.compile(
            r"\bguidance\b|\bregulatory\s+notice\b|\bnotice\s+to\s+members\b|\brisk\s+alert\b|\bstaff\s+bulletin\b|"
            r"\bFAQs?\b|\bfrequently\s+asked\s+questions\b|\bsupervisory\s+(?:letter|expectations|guidance)\b|"
            r"\binterpretive\b|\bno-action\b|\bexpectations\b|\bdear\s+colleague\b",
            re.IGNORECASE,
        ),
    ),
    (
        "rule",
        re.compile(
            r"\brules?\b|\brulemaking\b|\bregulations?\b|\brequirements?\b|\bamendments?\b|\bmandates?\b|"
            r"\bstandards\b|\brequest(?:s|ing)?\s+comment\b",
            re.IGNORECASE,
        ),
    ),
    (
        "statement",
        re.compile(
            r"\bstatements?\b|\bpolicy\s+decision\b|\bmonetary\s+policy\b|\binterest\s+rates?\b|"
            r"\bfederal\s+funds\s+rate\b|\btestimony\b|\btestif(?:y|ies|ied)\b|\bspeech\b|\bremarks\b|\bminutes\b",
            re.IGNORECASE,
        ),
    ),
)
_STAGE_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "settled",
        re.compile(
            r"\bsettled\b|\bsettle(?:s|ment)\b|\bwithout\s+admitting\s+or\s+denying\b|\bagreed\s+to\s+(?:pay|a)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "proposed",
        re.compile(
            r"\bpropos(?:e|es|ed|al|als|ing)\b|\brequest(?:s|ing)?\s+comment\b|\bcomments?\s+(?:are|is)\s+due\b|"
            r"\bpublic\s+comment\b|\bdraft\b",
            re.IGNORECASE,
        ),
    ),
    (
        "final",
        re.compile(
            r"\badopt(?:s|ed)?\b|\bfinal\s+(?:rule|rules|guidance|order|notice)\b|\bapprov(?:es|ed)\b|\bvoted\b|"
            r"\bfinali[sz](?:es|ed)\b|\btakes?\s+effect\b|\bissued\b",
            re.IGNORECASE,
        ),
    ),
)


def _first_cue(cues: Sequence[tuple[str, re.Pattern[str]]], texts: Sequence[str]) -> str | None:
    for text in texts:
        for name, rx in cues:
            if rx.search(text or ""):
                return name
    return None


def classify_action(title: str, summary: str = "", content: str = "") -> tuple[str | None, str]:
    """(action_type, stage) for a regulatory text: the title decides first, then the summary, then the body.

    ``action_type`` is None when nothing marks the text as an enforcement
    action, rule, guidance, or substantive statement (for example an investor
    education announcement).
    """
    action = _first_cue(_ACTION_CUES, [title, summary, content])
    stage = _first_cue(_STAGE_CUES, [f"{title} {summary}", content]) or "announced"
    if action != "enforcement" and stage == "settled":
        stage = "announced"
    return action, stage


# --------------------------------------------------------------------------
# Who is affected / what changes
# --------------------------------------------------------------------------

_AFFECTED_EXPLICIT = re.compile(
    r"\bwho\s+is\s+affected\b|\bappl(?:y|ies|ied)\s+to\b|\bwould\s+apply\b|\bdoes\s+not\s+apply\b|"
    r"\bsubject\s+to\b|\bcovered\s+(?:firms|entities|institutions)\b",
    re.IGNORECASE,
)
_AFFECTED_PARTIES = re.compile(
    r"\b(?:firms?|member\s+firms|advis[eo]rs?|investment\s+advis[eo]rs?|broker[- ]dealers?|banks?|lenders?|"
    r"credit\s+unions?|issuers?|public\s+companies|companies|funds?|fund\s+managers|asset\s+managers|exchanges|"
    r"platforms|investors|clients|customers|consumers|borrowers|schools?|school\s+districts?|districts?|"
    r"charter\s+schools|students|teachers|colleges|universities|institutions|vendors|providers)\b",
    re.IGNORECASE,
)
# Obligations, prohibitions, and sanctions count double; softer future-tense cues count once.
_STRONG_CHANGE = re.compile(
    r"\brequir(?:e|es|ed|ing|ement|ements)\b|\bprohibit(?:s|ed|ing)?\b|\bbans?\b|\bmust\b|\bshall\b|"
    r"\bwould\s+(?:be\s+expected\s+to|have\s+to|need\s+to|require)\b|\bexpected\s+to\b|\bmay\s+not\b|"
    r"\bcould\s+not\b|\bcannot\b|\bagreed\s+to\b|\bordered\s+to\b|\bpenalt(?:y|ies)\b|\bcensure\b",
    re.IGNORECASE,
)
_WEAK_CHANGE = re.compile(r"\bwill\b|\bwould\b|\btak(?:e|es|ing)\s+effect\b|\ballow(?:s|ed)?\b", re.IGNORECASE)
MAX_AFFECTED = 2


def _pick(sentences: Sequence[str], scorer, limit: int) -> list[str]:
    """The ``limit`` best-scoring sentences (score > 0), in their original order."""
    scored = [(scorer(s), i, s) for i, s in enumerate(sentences)]
    best = sorted((x for x in scored if x[0] > 0), key=lambda x: (-x[0], x[1]))[:limit]
    return [s for _, _, s in sorted(best, key=lambda x: x[1])]


def _affected_score(sentence: str) -> int:
    return 3 * bool(_AFFECTED_EXPLICIT.search(sentence)) + len(set(_AFFECTED_PARTIES.findall(sentence.lower())))


def _change_score(sentence: str) -> int:
    return 2 * len(_STRONG_CHANGE.findall(sentence)) + len(_WEAK_CHANGE.findall(sentence))


def who_and_what(sentences: Sequence[str]) -> tuple[list[str], list[str]]:
    """(who is affected, what changes): sentences quoted from the source.

    Sentences that say explicitly whom a measure applies to are kept for
    "who is affected"; the strongest obligation, prohibition, or sanction
    sentences among the rest describe what changes; then sentences naming
    affected parties (firms, advisers, broker-dealers, banks, schools,
    students...) fill "who is affected".
    """
    explicit = [s for s in sentences if _AFFECTED_EXPLICIT.search(s)]
    changes = _pick([s for s in sentences if s not in explicit], _change_score, MAX_SENTENCES)
    affected = _pick([s for s in sentences if s not in changes], _affected_score, MAX_AFFECTED)
    return [truncate(s, 320) for s in affected], [truncate(s, 320) for s in changes]


# --------------------------------------------------------------------------
# When: compliance, effective, and comment dates
# --------------------------------------------------------------------------

_MONTH_NAMES = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTH_NAMES |= {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name}
_MONTH_NAMES["sept"] = 9
_MONTH = r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December|" \
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\.?"  # fmt: skip
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "eighteen": 18, "twenty": 20, "thirty": 30,
    "forty-five": 45, "forty": 40, "sixty": 60, "ninety": 90, "hundred twenty": 120,
}  # fmt: skip
_COUNT = r"(?P<count>\d{1,3}|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")"

_DATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "day",
        re.compile(rf"\b{_MONTH}\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<year>\d{{4}})\b", re.IGNORECASE),
    ),
    ("day", re.compile(rf"\b(?P<day>\d{{1,2}})\s+{_MONTH}\s+(?P<year>\d{{4}})\b", re.IGNORECASE)),
    ("iso", re.compile(r"\b(?P<year>\d{4})-(?P<m>\d{2})-(?P<day>\d{2})\b")),
    ("month", re.compile(rf"\b{_MONTH}\s+(?P<year>\d{{4}})\b", re.IGNORECASE)),
    ("season", re.compile(r"\b(?P<season>spring|summer|fall|autumn|winter)\s+(?:of\s+)?(?P<year>\d{4})\b", re.I)),
    (
        "relative",
        re.compile(
            rf"(?:\b(?P<lead>within|in|after|for|of)\s+(?:the\s+next\s+)?)?\b{_COUNT}[\s-]+"
            r"(?P<unit>(?:business\s+|calendar\s+)?days?|weeks?|months?|years?)\b"
            r"(?P<after>\s+(?:after|of|following|from)\s+(?:the\s+)?(?P<anchor>[\w\s'-]{3,40}?)(?=[.,;:)]|$))?",
            re.IGNORECASE,
        ),
    ),
    (
        "relative",
        re.compile(rf"\b{_COUNT}-(?P<unit>day|week|month)\s+(?P<lead>comment)\s+period\b", re.IGNORECASE),
    ),
)
_COMMENT_CUE = re.compile(
    r"\bcomments?\s+(?:are|is|must\s+be|should\s+be)\s+(?:due|received|submitted|accepted)\b|"
    r"\bcomments?\s+(?:due|deadline)\b|\bcomment\s+(?:period|deadline)\b|"
    r"\bpublic\s+comment\s+(?:is\s+open|period|closes|ends)\b|\bopen\s+for\s+(?:public\s+)?comments?\b",
    re.IGNORECASE,
)
_COMPLIANCE_CUE = re.compile(
    r"\bcompliance\s+(?:date|deadline|period)\b|\bmust\s+comply\b|\bcomply\s+(?:by|with)\b|\bin\s+compliance\b",
    re.IGNORECASE,
)
_EFFECTIVE_CUE = re.compile(
    r"\beffective\b|\btak(?:e|es|ing)\s+effect\b|\bgo(?:es)?\s+into\s+effect\b|\bin\s+effect\b|"
    r"\bbeginning\b|\bbegins?\b|\bstarting\b|\bstarts?\b|\bapplies\b|\bwould\s+apply\b|\bphased?\s+in\b",
    re.IGNORECASE,
)
_BY_BEFORE = re.compile(r"\b(?:by|no\s+later\s+than|before)\s*$", re.IGNORECASE)
_DEADLINE_CUE = re.compile(r"\bdeadline\b|\bdue\b", re.IGNORECASE)


def _count(raw: str) -> int | None:
    raw = raw.lower()
    return int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw)


def _when_kind(sentence: str, start: int, end: int) -> str | None:
    """What a date in a sentence is for: comment deadline, compliance, effective date, or another deadline.

    Cues are read within the date's clause (semicolons separate the items of
    a list, each with its own date).
    """
    clause_start = sentence.rfind(";", 0, start) + 1
    clause_end = sentence.find(";", end)
    clause_end = len(sentence) if clause_end < 0 else clause_end
    before = sentence[max(clause_start, start - 160) : start]
    around = sentence[max(clause_start, start - 160) : min(clause_end, end + 60)]
    if _COMMENT_CUE.search(around):
        return "comment_deadline"
    if _COMPLIANCE_CUE.search(around):
        return "compliance"
    if _BY_BEFORE.search(before):
        return "deadline"
    if _EFFECTIVE_CUE.search(around):
        return "effective"
    return "deadline" if _DEADLINE_CUE.search(around) else None


def _add_months(day: date, months: int) -> date:
    month = day.month - 1 + months
    year, month = day.year + month // 12, month % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def parse_when(text: str, published: datetime | None = None) -> list[dict[str, Any]]:
    """Compliance, effective, comment, and deadline dates stated in ``text``.

    Returns dicts ``{kind, phrase, date, estimated, sentence}`` in order of
    appearance. ``kind`` is ``comment_deadline``, ``compliance``,
    ``effective``, or ``deadline`` ("by March 31, 2027"); dates without such a
    cue are ignored. ``date`` is ISO-8601 at the precision the text gives
    (``2027-07-01``, ``2027-03``, or ``2028`` for "fall 2028"), or None when a
    date is relative to a future event ("six months after the final notice").
    Durations counted from the publication date ("within 60 days") are
    converted using ``published`` and marked ``estimated``.
    """
    found: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None, str]] = set()
    for sentence in split_sentences(text):
        taken: list[tuple[int, int]] = []
        for shape, rx in _DATE_PATTERNS:
            for m in rx.finditer(sentence):
                if any(m.start() < e and s < m.end() for s, e in taken):
                    continue
                kind = _when_kind(sentence, m.start(), m.end())
                if shape == "relative" and (m.groupdict().get("lead") or "").lower() == "comment":
                    kind = "comment_deadline"
                if kind is None:
                    continue
                taken.append((m.start(), m.end()))
                when, estimated = _resolve(shape, m, published)
                if shape == "relative" and when is None and m.groupdict().get("anchor") is None and not published:
                    continue
                phrase = normalize_space(m.group(0))
                key = (kind, when, "") if when else (kind, None, phrase.lower())
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    {
                        "kind": kind,
                        "phrase": phrase,
                        "date": when,
                        "estimated": estimated,
                        "sentence": truncate(sentence, 320),
                    }
                )
    return found


def _resolve(shape: str, m: re.Match[str], published: datetime | None) -> tuple[str | None, bool]:
    groups = m.groupdict()
    if shape == "day":
        month = _MONTH_NAMES.get(groups["month"].lower().rstrip("."))
        try:
            return date(int(groups["year"]), month or 0, int(groups["day"])).isoformat(), False
        except ValueError:
            return None, False
    if shape == "iso":
        try:
            return date(int(groups["year"]), int(groups["m"]), int(groups["day"])).isoformat(), False
        except ValueError:
            return None, False
    if shape == "month":
        month = _MONTH_NAMES.get(groups["month"].lower().rstrip("."))
        return (f"{int(groups['year']):04d}-{month:02d}" if month else None), False
    if shape == "season":
        return f"{int(groups['year']):04d}", True
    # Relative durations: only convertible when counted from the publication itself.
    anchor = (groups.get("anchor") or "").lower()
    count = _count(groups["count"])
    if count is None or published is None:
        return None, False
    if anchor and not re.search(r"\b(?:publication|today|date of this|this notice|issuance)\b", anchor):
        return None, False
    unit = groups["unit"].lower()
    start = published.date()
    if "day" in unit:
        return (start + timedelta(days=count)).isoformat(), True
    if "week" in unit:
        return (start + timedelta(weeks=count)).isoformat(), True
    if "month" in unit:
        return _add_months(start, count).isoformat(), True
    return _add_months(start, 12 * count).isoformat(), True


# --------------------------------------------------------------------------
# Regulation Watch
# --------------------------------------------------------------------------


_AFTER_QUOTE = re.compile(r"(?<=[.!?][”\"’])\s+(?=[A-Z0-9“\"(])")


def _sentences(article: Article) -> list[str]:
    sentences = split_sentences(article.content or "") or split_sentences(article.summary or "")
    # split_sentences keeps "... called “AI washing.” According to ..." together; split after closing quotes too.
    parts = [part for sentence in sentences for part in _AFTER_QUOTE.split(sentence)]
    return [s for s in dict.fromkeys(normalize_space(p) for p in parts) if len(s) > 20]


def _is_candidate(article: Article) -> bool:
    if article.duplicate_of_id is not None or article.doc_type in _SKIP_DOC_TYPES:
        return False
    if article.trust_tier == TrustTier.COMMENTARY:
        return False
    if article.doc_type in _REGULATOR_DOC_TYPES:
        return True
    return (article.event_type or classify_event(article)) in _ACTION_EVENTS


def _is_regulator_document(article: Article) -> bool:
    return article.doc_type in _REGULATOR_DOC_TYPES and article.trust_tier == TrustTier.PRIMARY


_NAME_TOKEN = re.compile(r"\b[A-Z][a-z]{3,}[A-Za-z]*\b")
_COMMON_CAPS = frozenset(
    {"the", "this", "that", "after", "under", "today", "according", "without", "board", "commission", "regulator"}
)


def _distinctive(article: Article, regulator: str) -> set[str]:
    reg = {w.lower() for w in regulator.split()}
    words = {w.lower() for w in _NAME_TOKEN.findall(f"{article.title} {article.summary or ''}")}
    return {w for w in words if w not in reg and w not in _COMMON_CAPS}


def _regulator_tokens(name: str) -> set[str]:
    return {t for t in tokenize(re.sub(r"['’]s\b", "", name)) if t not in {"of", "the", "state", "board"}} or set(
        tokenize(name)
    )


def _same_regulator(a: str, b: str) -> bool:
    """ "Calder's Board of Education" and "Calder State Board of Education" are the same regulator."""
    return a == b or jaccard(_regulator_tokens(a), _regulator_tokens(b)) >= 0.5


def _action_similarity(lead: Article, other: Article, regulator: str) -> float:
    """How likely two articles report the same action (0 = unrelated)."""
    if lead.story_id is not None and lead.story_id == other.story_id:
        return 1.0
    words = jaccard(
        tokenize(f"{lead.title} {lead.summary or ''}"), tokenize(f"{other.title} {other.summary or ''}")
    )
    names = 0.3 if _distinctive(lead, regulator) & _distinctive(other, regulator) else 0.0
    score = words + names
    return score if score >= 0.2 else 0.0


def _item(article: Article, regulator: tuple[str, str, str], action: str, stage: str) -> dict[str, Any]:
    name, group, kind = regulator
    published = article.published_at or article.retrieved_at
    sentences = _sentences(article)
    affected, changes = who_and_what(sentences)
    when = parse_when(f"{article.summary or ''}\n\n{article.content or ''}", article.published_at)
    caveats: list[str] = []
    if stage == "proposed":
        caveats.append("Proposed: the regulator may change it before it is final.")
    if any(w["estimated"] for w in when):
        caveats.append(
            "Dates marked as estimates are counted from the publication date or stated only as a season; check "
            "the regulator's notice for the exact date."
        )
    if any(w["date"] is None for w in when):
        caveats.append("Some dates depend on a future step (for example a final notice) and are not yet known.")
    if action == "enforcement" and re.search(r"without\s+admitting\s+or\s+denying", article.content or "", re.I):
        caveats.append("Settled without admitting or denying the findings.")
    if not _is_regulator_document(article):
        caveats.append("Reported by news coverage; the regulator's own release is the primary record.")
    if article.is_demo:
        caveats.append("Fictional demo content.")
    lede = sentences[0] if sentences else article.summary or article.title
    return {
        "article_id": article.id,
        "title": article.title,
        "url": article.url,
        "publisher": article.source.name if article.source else "",
        "published_at": iso(published),
        "summary": truncate(article.summary or lede, 400),
        "vertical": article.vertical,
        "doc_type": article.doc_type,
        "trust_tier": article.trust_tier,
        "story_id": article.story_id,
        "is_demo": bool(article.is_demo),
        "regulator": name,
        "regulator_group": group,
        "regulator_kind": kind,
        "action_type": action,
        "action_label": ACTION_LABELS[action],
        "stage": stage,
        "stage_label": STAGE_LABELS[stage],
        "who_is_affected": affected,
        "what_changes": changes,
        "when": when,
        "citations": [citation_for(article, "c1", snippet=truncate(lede, 280)).model_dump()],
        "coverage": [],
        "caveats": caveats,
    }


def _attach_coverage(item: dict[str, Any], article: Article) -> None:
    cid = f"c{len(item['citations']) + 1}"
    item["citations"].append(citation_for(article, cid).model_dump())
    item["coverage"].append(
        {
            "article_id": article.id,
            "title": article.title,
            "url": article.url,
            "publisher": article.source.name if article.source else "",
            "published_at": iso(article.published_at or article.retrieved_at),
            "citation_id": cid,
        }
    )
    # A news report can add a date or a party the regulator's summary left out.
    known = {(w["kind"], w["date"]) for w in item["when"]}
    for mention in parse_when(f"{article.summary or ''}\n\n{article.content or ''}", article.published_at):
        if (mention["kind"], mention["date"]) not in known:
            item["when"].append({**mention, "citation_id": cid})


def regulation_watch(
    session: Session,
    days: int = 60,
    *,
    vertical: str | None = None,
    now: datetime | None = None,
    limit: int = MAX_ITEMS,
) -> list[dict[str, Any]]:
    """Recent regulatory and enforcement actions: who is affected, what changes, and when.

    Considers non-duplicate, non-opinion articles from the last ``days`` days
    (optionally one ``vertical``) that are regulator documents (doc type
    regulatory/policy) or report a regulation, enforcement, or policy event.
    Each item is a dict with ``article_id, title, url, publisher,
    published_at`` (ISO), ``summary, vertical, doc_type, trust_tier,
    story_id, is_demo``, ``regulator`` (display name), ``regulator_group``
    (SEC / FINRA / Federal Reserve / Department of Education / other),
    ``regulator_kind`` (e.g. "securities regulator", "self-regulatory
    organization", "central bank", "education agency"), ``action_type``
    (enforcement / rule / guidance / statement) with ``action_label``,
    ``stage`` (proposed / final / settled / announced) with ``stage_label``,
    ``who_is_affected`` and ``what_changes`` (sentences quoted from the
    source), ``when`` (see ``parse_when``), ``citations`` (CitationRef dicts:
    the item's own document first, then coverage), ``coverage`` (news reports
    of the same action), and ``caveats``. Announcements that are not
    regulatory actions (investor-education launches, research releases) are
    left out. Newest first.
    """
    now = now or utcnow()
    since = now - timedelta(days=days)
    when = func.coalesce(Article.published_at, Article.retrieved_at)
    query = (
        select(Article)
        .where(
            Article.duplicate_of_id.is_(None),
            Article.doc_type.not_in(sorted(_SKIP_DOC_TYPES)),
            when >= since,
            when <= now + timedelta(days=1),
        )
        .order_by(when.desc(), Article.id.desc())
    )
    if vertical:
        query = query.where(Article.vertical == str(vertical))
    candidates = [a for a in session.scalars(query) if _is_candidate(a)]
    # Regulators' own documents first, so news reports attach to them as coverage.
    candidates.sort(key=lambda a: not _is_regulator_document(a))

    items: list[tuple[dict[str, Any], Article]] = []
    for article in candidates:
        regulator = identify_regulator(article)
        if regulator is None:
            continue
        action, stage = classify_action(article.title or "", article.summary or "", article.content or "")
        if action is None:
            continue
        if not _is_regulator_document(article):
            scored = [
                (_action_similarity(lead, article, regulator[0]), item)
                for item, lead in items
                if _same_regulator(item["regulator"], regulator[0])
            ]
            best = max(scored, key=lambda pair: pair[0], default=(0.0, None))
            if best[1] is not None and best[0] > 0:
                _attach_coverage(best[1], article)
                continue
        items.append((_item(article, regulator, action, stage), article))
    ordered = sorted(items, key=lambda pair: (pair[0]["published_at"] or "", pair[1].id or 0), reverse=True)
    return [item for item, _ in ordered[:limit]]


def finance_regulation_items(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The Regulation Watch items that belong on the Finance page (finance vertical or a financial regulator)."""
    return [
        item
        for item in items
        if item.get("vertical") == "finance"
        or item.get("regulator_group") in ("SEC", "FINRA", "Federal Reserve")
        or item.get("regulator_kind") in FINANCE_REGULATOR_KINDS
    ]
