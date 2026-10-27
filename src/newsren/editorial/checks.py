"""Editorial quality gate: automated checks every brief passes through before publication.

``run_checks`` returns one ``CheckResult`` per check, always in the same order
and always present (a check that does not apply passes with an explanatory
message), so editors and templates see a stable checklist. Severities:

* ``blocker`` — the brief cannot be approved without an explicit editor
  override (missing or unknown citations, unapproved sources, malformed links,
  finance guardrail violations, unlabelled finance performance claims);
* ``warning`` — a human should look before publication (labelling, attribution,
  provenance gaps); any failed warning raises the risk level to medium;
* ``info`` — context for the reviewer (missing publication dates, high-impact
  claims, which always route the brief to a human).

The checks are deterministic and read-only: they never modify the brief.

Check names (stable identifiers): ``content_present``, ``sources_cited``,
``claims_recorded``, ``citation_coverage``, ``unknown_citation_ids``,
``claim_support``, ``approved_sources``, ``well_formed_links``,
``retrieval_timestamps``, ``publication_dates``, ``licensing_notes``,
``support_status``, ``vendor_attribution``, ``disagreements_described``,
``preprint_labelling``, ``forecast_labelling``, ``study_population``,
``guardrail_lint``, ``performance_labels``, ``high_impact_claims``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from newsren.agents.base import article_text
from newsren.editorial.guardrails import is_hedged, lint_text, performance_claim_issues
from newsren.ingest.sources import is_approved_url
from newsren.models import (
    Article,
    Brief,
    BriefClaim,
    ClaimType,
    DocType,
    RiskLevel,
    Source,
    SourceKind,
    SupportStatus,
    TrustTier,
    Vertical,
)
from newsren.schemas import CheckResult, CitationRef
from newsren.textutil import domain_of, normalize_space, tokenize, truncate

BLOCKER = "blocker"
WARNING = "warning"
INFO = "info"

CHECK_NAMES: tuple[str, ...] = (
    "content_present",
    "sources_cited",
    "claims_recorded",
    "citation_coverage",
    "unknown_citation_ids",
    "claim_support",
    "approved_sources",
    "well_formed_links",
    "retrieval_timestamps",
    "publication_dates",
    "licensing_notes",
    "support_status",
    "vendor_attribution",
    "disagreements_described",
    "preprint_labelling",
    "forecast_labelling",
    "study_population",
    "guardrail_lint",
    "performance_labels",
    "high_impact_claims",
)

_PREPRINT_LABEL = re.compile(
    r"\bpre-?prints?\b|\bnot\s+(?:yet\s+)?(?:been\s+)?peer[- ]reviewed\b|"
    r"\bnot\s+(?:yet\s+)?(?:undergone|gone\s+through|passed|completed)\s+peer\s+review\b|"
    r"\b(?:without|before|awaiting|pending|ahead\s+of)\s+peer\s+review\b|\bunreviewed\b|\bnot\s+peer\s+reviewed\b",
    re.IGNORECASE,
)
_ATTRIBUTED = re.compile(
    r"\b(?:says?|said|claims?|claimed|according\s+to|announced|reports?|reported|states?|stated|describes?|"
    r"described|calls?|called|caution(?:s|ed)?|warn(?:s|ed)?|notes?|noted|acknowledg(?:es|ed)|"
    r"disclos(?:es|ed)|wrote|estimates?|touts?|touted|advertis\w*|promot\w*|marketing|press\s+release|"
    r"self-reported|company-reported|vendor|its\s+own)\b",
    re.IGNORECASE,
)
# Capability or performance language in a vendor-sourced claim: comparatives, superlatives, numbers.
_CAPABILITY = re.compile(
    r"\b(?:better|best|faster|fastest|more\s+accurate|most|leading|state[- ]of[- ]the[- ]art|breakthrough|"
    r"outperform\w*|beats?|superior|improv\w*|revolution\w*|unmatched|unrivalled|unrivaled)\b|"
    r"\d+(?:\.\d+)?\s*(?:%|percent\b|x\b|times\b)",
    re.IGNORECASE,
)
_POPULATION = re.compile(
    r"\b(?:students?|pupils?|learners?|teachers?|educators?|participants?|schools?|districts?|grades?|graders?|"
    r"undergraduates?|children|adults|sample|cohorts?|classrooms?|families|parents|respondents?|subjects)\b|"
    r"\bn\s*=\s*\d",
    re.IGNORECASE,
)
_OUTCOME = re.compile(
    r"\b(?:scores?|outcomes?|measured|measures?|tests?|assessments?|achievement|completion|transfer|grades|"
    r"attendance|gains?|proficiency|retention|graduation|enrollment|results?)\b",
    re.IGNORECASE,
)
_STUDY_DOC_TYPES = frozenset(
    {DocType.RESEARCH_PREPRINT.value, DocType.RESEARCH_PEER_REVIEWED.value, DocType.EDUCATION_REPORT.value}
)
_CITATION_MARKER = re.compile(r"\[(?:c\d+(?:\s*,\s*c\d+)*)\]", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.,])\d+(?:\.\d+)?(?![\w])")
_MIN_OVERLAP = 0.25


# --------------------------------------------------------------------------
# Context shared by the checks
# --------------------------------------------------------------------------


@dataclass
class _Context:
    session: Session
    brief: Brief
    vertical: str
    citations: list[CitationRef]
    malformed_records: list[str]
    claims: list[BriefClaim]
    texts: list[tuple[str, str]]
    _sources_by_domain: dict[str, list[Source]] | None = None
    _articles: dict[int, Article | None] = field(default_factory=dict)

    @property
    def is_finance(self) -> bool:
        return self.vertical == Vertical.FINANCE

    @property
    def by_id(self) -> dict[str, CitationRef]:
        return {c.id: c for c in self.citations}

    def article(self, cit: CitationRef) -> Article | None:
        if cit.article_id is None:
            return None
        if cit.article_id not in self._articles:
            self._articles[cit.article_id] = self.session.get(Article, cit.article_id)
        return self._articles[cit.article_id]

    def source_for(self, cit: CitationRef) -> Source | None:
        """The ``Source`` row behind a citation: via its article, else by URL domain."""
        article = self.article(cit)
        if article is not None and article.source is not None:
            return article.source
        domain = domain_of(cit.url) if cit.url else ""
        if not domain:
            return None
        if self._sources_by_domain is None:
            index: dict[str, list[Source]] = {}
            for src in self.session.scalars(select(Source)):
                for url in (src.url, src.homepage):
                    d = domain_of(url) if url else ""
                    if d:
                        index.setdefault(d, []).append(src)
            self._sources_by_domain = index
        for known, sources in self._sources_by_domain.items():
            if domain == known or domain.endswith("." + known):
                approved = [s for s in sources if s.approved]
                return (approved or sources)[0]
        return None

    def evidence_text(self, cit: CitationRef) -> str:
        """All stored text for a citation: its snippet plus the article it points to."""
        parts = [cit.title, cit.snippet]
        article = self.article(cit)
        if article is not None:
            parts.append(article_text(article, max_chars=20000))
        return normalize_space(" ".join(p for p in parts if p))


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_as_text(v) for k, v in value.items() if k != "citation_ids")
    if isinstance(value, list | tuple):
        return " ".join(_as_text(v) for v in value)
    return str(value)


def _section_parts(section: Any) -> tuple[str, str, list[str]]:
    if isinstance(section, dict):
        title = str(section.get("title") or section.get("heading") or "Section")
        return title, str(section.get("body") or ""), [str(c) for c in section.get("citation_ids") or []]
    return "Section", _as_text(section), []


def _disagreement_parts(item: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(item, dict):
        positions = [p if isinstance(p, dict) else {"position": _as_text(p)} for p in item.get("positions") or []]
        return str(item.get("topic") or "Disagreement"), positions
    return _as_text(item), []


def _parse_citations(raw: Sequence[Any]) -> tuple[list[CitationRef], list[str]]:
    citations: list[CitationRef] = []
    malformed: list[str] = []
    for i, item in enumerate(raw or [], 1):
        try:
            citations.append(CitationRef.model_validate(item))
        except ValidationError:
            data = item if isinstance(item, dict) else {}
            cid = str(data.get("id") or f"#{i}")
            malformed.append(f"Citation record {cid} is incomplete (needs id, url, and title).")
            citations.append(
                CitationRef(id=cid, url=str(data.get("url") or ""), title=str(data.get("title") or "(untitled)"))
            )
    return citations, malformed


def _brief_texts(brief: Brief, claims: Sequence[BriefClaim]) -> list[tuple[str, str]]:
    texts = [
        ("Headline", brief.headline or ""),
        ("Summary", brief.summary or ""),
        ("Why it matters", brief.why_it_matters or ""),
        ("What changed", brief.what_changed or ""),
    ]
    texts += [(f"Key point {i}", _as_text(p)) for i, p in enumerate(brief.key_points or [], 1)]
    for section in brief.sections or []:
        title, body, _ = _section_parts(section)
        texts.append((f"Section '{truncate(title, 60)}'", f"{title}. {body}" if body else title))
    texts += [(f"Claim {i}", c.text or "") for i, c in enumerate(claims, 1)]
    texts += [(f"Uncertainty {i}", _as_text(u)) for i, u in enumerate(brief.uncertainty or [], 1)]
    for item in brief.disagreements or []:
        topic, positions = _disagreement_parts(item)
        body = " ".join(_as_text(p.get("position")) for p in positions)
        texts.append((f"Disagreement '{truncate(topic, 60)}'", f"{topic}. {body}".strip()))
    texts += [(f"Watch next {i}", _as_text(w)) for i, w in enumerate(brief.watch_next or [], 1)]
    return [(loc, t) for loc, t in texts if normalize_space(t)]


# Column defaults only apply at flush, so read unflushed claims as the database would store them.
def _ctype(claim: BriefClaim) -> str:
    return claim.claim_type or ClaimType.FACT.value


def _status(claim: BriefClaim) -> str:
    return claim.support_status or SupportStatus.UNVERIFIED.value


def _claim_label(index: int, claim: BriefClaim) -> str:
    return f'Claim {index}: "{truncate(claim.text, 100)}"'


def _cit_label(cit: CitationRef) -> str:
    return f"{cit.id} ({truncate(cit.title, 70)})"


def _result(name: str, severity: str, failures: list[str], ok_message: str, fail_message: str) -> CheckResult:
    return CheckResult(
        name=name,
        passed=not failures,
        severity=severity,
        message=fail_message if failures else ok_message,
        details=failures,
    )


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------


def _check_content_present(ctx: _Context) -> CheckResult:
    missing = [
        label
        for label, value in (("headline", ctx.brief.headline), ("summary", ctx.brief.summary))
        if not normalize_space(value)
    ]
    return _result(
        "content_present",
        BLOCKER,
        [f"The brief has no {m}." for m in missing],
        "Headline and summary are present.",
        "The brief is missing its headline or summary.",
    )


def _check_sources_cited(ctx: _Context) -> CheckResult:
    return _result(
        "sources_cited",
        BLOCKER,
        [] if ctx.citations else ["No source documents are recorded for this brief."],
        f"{len(ctx.citations)} source(s) recorded.",
        "Every generated item must keep a record of its sources.",
    )


def _check_claims_recorded(ctx: _Context) -> CheckResult:
    return _result(
        "claims_recorded",
        WARNING,
        [] if ctx.claims else ["No claims were extracted, so the brief cannot be checked claim by claim."],
        f"{len(ctx.claims)} claim(s) recorded for claim-level checks.",
        "The brief records no claims; check its statements against the sources by hand.",
    )


def _check_citation_coverage(ctx: _Context) -> CheckResult:
    must_cite = {ClaimType.FACT.value, ClaimType.PERFORMANCE.value}
    failures = [
        _claim_label(i, c) + (" (high-impact)" if c.high_impact else f" ({_ctype(c)})") + " cites no source."
        for i, c in enumerate(ctx.claims, 1)
        if (_ctype(c) in must_cite or c.high_impact) and not c.citation_ids
    ]
    return _result(
        "citation_coverage",
        BLOCKER,
        failures,
        "Every fact and performance claim cites at least one source.",
        f"{len(failures)} claim(s) that must cite a source have no citation.",
    )


def _check_unknown_citation_ids(ctx: _Context) -> CheckResult:
    known = set(ctx.by_id)
    failures: list[str] = []
    ids = [c.id for c in ctx.citations]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        failures.append(f"Citation id {dup} is used for more than one source.")
    for i, claim in enumerate(ctx.claims, 1):
        unknown = [str(cid) for cid in claim.citation_ids or [] if str(cid) not in known]
        if unknown:
            failures.append(f"{_claim_label(i, claim)} cites unknown id(s) {', '.join(unknown)}.")
    for section in ctx.brief.sections or []:
        title, _, cids = _section_parts(section)
        unknown = [cid for cid in cids if cid not in known]
        if unknown:
            failures.append(f"Section '{title}' cites unknown id(s) {', '.join(unknown)}.")
    for item in ctx.brief.disagreements or []:
        topic, positions = _disagreement_parts(item)
        unknown = [str(cid) for p in positions for cid in p.get("citation_ids") or [] if str(cid) not in known]
        if unknown:
            failures.append(f"Disagreement '{topic}' cites unknown id(s) {', '.join(unknown)}.")
    return _result(
        "unknown_citation_ids",
        BLOCKER,
        failures,
        "All citation ids refer to recorded sources.",
        "Some citation ids do not match any recorded source.",
    )


def _numbers(text: str) -> set[float]:
    values: set[float] = set()
    for raw in _NUMBER.findall(_CITATION_MARKER.sub(" ", text)):
        try:
            values.add(float(raw.replace(",", "")))
        except ValueError:
            continue
    return values


def _stems(text: str) -> set[str]:
    return {t[:6] for t in tokenize(text)}


def _check_claim_support(ctx: _Context) -> CheckResult:
    by_id = ctx.by_id
    failures: list[str] = []
    unchecked = 0
    for i, claim in enumerate(ctx.claims, 1):
        cited = [by_id[str(cid)] for cid in claim.citation_ids or [] if str(cid) in by_id]
        if not cited:
            continue
        evidence = " ".join(ctx.evidence_text(c) for c in cited)
        if len(tokenize(evidence)) < 5:
            unchecked += 1
            continue
        ids = ", ".join(c.id for c in cited)
        # Single-digit integers are too common (ordinals, list items) to verify reliably.
        claim_numbers = {n for n in _numbers(claim.text) if n >= 10 or n != int(n)}
        missing = sorted(claim_numbers - _numbers(evidence))
        if missing:
            shown = ", ".join(f"{n:g}" for n in missing)
            failures.append(f"{_claim_label(i, claim)}: figure(s) {shown} not found in cited source(s) {ids}.")
            continue
        stems = _stems(_CITATION_MARKER.sub(" ", claim.text))
        if len(stems) >= 4 and len(stems & _stems(evidence)) / len(stems) < _MIN_OVERLAP:
            failures.append(f"{_claim_label(i, claim)}: little wording in common with cited source(s) {ids}.")
    note = f" ({unchecked} claim(s) could not be checked: no stored source text.)" if unchecked else ""
    return _result(
        "claim_support",
        WARNING,
        failures,
        f"Claims' figures and key terms appear in their cited sources.{note}",
        f"{len(failures)} claim(s) may not be supported by the sources they cite; verify against the originals.",
    )


def _check_approved_sources(ctx: _Context) -> CheckResult:
    failures: list[str] = []
    demo = 0
    for cit in ctx.citations:
        source = ctx.source_for(cit)
        # An approved source row also counts: ingested articles can live on a different domain from
        # the feed (CDNs, syndication hosts), and demo or paused sources stay approved for past items.
        approved = (bool(cit.url) and is_approved_url(ctx.session, cit.url)) or bool(source and source.approved)
        if not approved:
            reason = "its source is not approved" if source is not None else "not on an approved source domain"
            failures.append(f"{_cit_label(cit)}: {cit.url or '(no URL)'} — {reason}.")
        elif source is not None and source.kind == SourceKind.DEMO:
            demo += 1
    demo_note = f" ({demo} from fictional demo sources on reserved .example domains.)" if demo else ""
    return _result(
        "approved_sources",
        BLOCKER,
        failures,
        f"All sources are approved.{demo_note}",
        f"{len(failures)} source(s) are not on the approved source list.",
    )


def _link_problem(url: str) -> str | None:
    if not url:
        return "no URL recorded"
    if any(ch.isspace() for ch in url) or any(ch in url for ch in '<>"'):
        return "contains spaces or invalid characters"
    try:
        parts = urlsplit(url)
    except ValueError:
        return "cannot be parsed"
    if parts.scheme not in ("http", "https"):
        return "not an http(s) link"
    host = parts.hostname or ""
    if not host or "." not in host or host.startswith(".") or host.endswith("."):
        return "has no valid host"
    return None


def _check_well_formed_links(ctx: _Context) -> CheckResult:
    failures = list(ctx.malformed_records)
    for cit in ctx.citations:
        problem = _link_problem(cit.url)
        if problem:
            failures.append(f"{_cit_label(cit)}: link {problem}.")
    return _result(
        "well_formed_links",
        BLOCKER,
        failures,
        "All source links are well-formed http(s) URLs.",
        "Some source links are missing or malformed.",
    )


def _check_retrieval_timestamps(ctx: _Context) -> CheckResult:
    failures = [f"{_cit_label(c)} has no retrieval time." for c in ctx.citations if not c.retrieved_at]
    return _result(
        "retrieval_timestamps",
        WARNING,
        failures,
        "Every source records when The Newsren retrieved it.",
        "Some sources do not record a retrieval time.",
    )


def _check_publication_dates(ctx: _Context) -> CheckResult:
    failures = [f"{_cit_label(c)} has no publication date." for c in ctx.citations if not c.published_at]
    return _result(
        "publication_dates",
        INFO,
        failures,
        "Every source records its publication date.",
        "Some sources have no known publication date; show 'date unknown' to readers.",
    )


def _check_licensing_notes(ctx: _Context) -> CheckResult:
    failures: list[str] = []
    for cit in ctx.citations:
        source = ctx.source_for(cit)
        if source is not None:
            if not normalize_space(source.license_note):
                failures.append(f"{_cit_label(cit)}: source '{source.name}' has no licensing note.")
        elif not (cit.url and is_approved_url(ctx.session, cit.url)):
            failures.append(f"{_cit_label(cit)}: no source record, so its licensing terms are unknown.")
        # Otherwise: an allow-listed primary record (regulator, filing, research index) with public terms.
    return _result(
        "licensing_notes",
        WARNING,
        failures,
        "Licensing terms are recorded for every source.",
        "Some sources have no recorded licensing terms; confirm we may use them.",
    )


def _publisher_key(cit: CitationRef) -> str:
    return normalize_space(cit.publisher).lower() or domain_of(cit.url) or cit.id


def _check_support_status(ctx: _Context) -> CheckResult:
    by_id = ctx.by_id
    failures: list[str] = []
    for i, claim in enumerate(ctx.claims, 1):
        status = _status(claim)
        cited = [by_id[str(cid)] for cid in claim.citation_ids or [] if str(cid) in by_id]
        if status == SupportStatus.CORROBORATED:
            publishers = {_publisher_key(c) for c in cited}
            independent = [c for c in cited if c.trust_tier != TrustTier.VENDOR]
            if len(publishers) < 2 or not independent:
                failures.append(
                    f"{_claim_label(i, claim)} is marked corroborated but does not cite two independent sources."
                )
        elif (
            status in (SupportStatus.SUPPORTED, SupportStatus.CONTESTED)
            and not claim.citation_ids
            # Uncited facts and performance claims are already blockers under citation_coverage.
            and _ctype(claim) not in (ClaimType.FACT, ClaimType.PERFORMANCE)
            and not claim.high_impact
        ):
            failures.append(f"{_claim_label(i, claim)} is marked {status} but cites no source.")
        elif status == SupportStatus.UNVERIFIED and _ctype(claim) in (ClaimType.FACT, ClaimType.PERFORMANCE):
            failures.append(
                f"{_claim_label(i, claim)} is presented as a {_ctype(claim)} but is unverified; "
                "say it could not be verified or remove it."
            )
    return _result(
        "support_status",
        WARNING,
        failures,
        "Support labels match the sources cited.",
        "Some claims' support labels overstate the evidence.",
    )


def _check_vendor_attribution(ctx: _Context) -> CheckResult:
    by_id = ctx.by_id
    failures: list[str] = []
    for i, claim in enumerate(ctx.claims, 1):
        if _ctype(claim) not in (ClaimType.FACT, ClaimType.PERFORMANCE):
            continue
        cited = [by_id[str(cid)] for cid in claim.citation_ids or [] if str(cid) in by_id]
        vendor_only = bool(cited) and all(
            c.trust_tier == TrustTier.VENDOR or c.doc_type == DocType.PRESS_RELEASE for c in cited
        )
        if vendor_only and _CAPABILITY.search(claim.text) and not _ATTRIBUTED.search(claim.text):
            failures.append(
                f"{_claim_label(i, claim)} rests only on the company's own materials; attribute it "
                '("the company says") or cite an independent evaluation.'
            )
    return _result(
        "vendor_attribution",
        WARNING,
        failures,
        "Vendor capability claims are attributed or independently supported.",
        "Some vendor claims are stated as fact without attribution.",
    )


def _check_disagreements(ctx: _Context) -> CheckResult:
    contested = [_claim_label(i, c) for i, c in enumerate(ctx.claims, 1) if _status(c) == SupportStatus.CONTESTED]
    disagreements = [_disagreement_parts(d) for d in ctx.brief.disagreements or []]
    failures: list[str] = []
    if contested and not disagreements:
        failures += [
            f"{label} is contested, but the brief does not describe the disagreement." for label in contested
        ]
    for topic, positions in disagreements:
        if len(positions) < 2:
            failures.append(f"Disagreement '{topic}' describes fewer than two positions.")
        uncited = [p for p in positions if not p.get("citation_ids")]
        if uncited:
            failures.append(f"Disagreement '{topic}': {len(uncited)} position(s) cite no source.")
    return _result(
        "disagreements_described",
        WARNING,
        failures,
        "Disagreements between sources are described with each side's sources."
        if disagreements
        else "No contested claims.",
        "Where credible sources disagree, describe each position with its sources instead of choosing one.",
    )


def _check_preprint_labelling(ctx: _Context) -> CheckResult:
    preprints = [c for c in ctx.citations if c.doc_type == DocType.RESEARCH_PREPRINT]
    if not preprints:
        return _result("preprint_labelling", WARNING, [], "No preprints cited.", "")
    labelled = any(_PREPRINT_LABEL.search(text) for _, text in ctx.texts)
    failures = [] if labelled else [f"{_cit_label(c)} is a preprint." for c in preprints]
    return _result(
        "preprint_labelling",
        WARNING,
        failures,
        "Preprints are identified as not yet peer reviewed.",
        "The brief cites a preprint but never tells readers it is a preprint / not yet peer reviewed.",
    )


def _check_forecast_labelling(ctx: _Context) -> CheckResult:
    failures = [
        f"{_claim_label(i, c)} is a forecast but is not labelled as one (projected, expects, could, scenario...)."
        for i, c in enumerate(ctx.claims, 1)
        if _ctype(c) == ClaimType.FORECAST and not is_hedged(c.text)
    ]
    return _result(
        "forecast_labelling",
        WARNING,
        failures,
        "Forecasts and scenarios are labelled as projections.",
        "Some forecasts read as facts; label them as projections or scenarios.",
    )


def _check_study_population(ctx: _Context) -> CheckResult:
    studies = [c for c in ctx.citations if c.doc_type in _STUDY_DOC_TYPES]
    if ctx.vertical != Vertical.EDUCATION or not studies:
        return _result("study_population", WARNING, [], "Not an education research brief.", "")
    text = " ".join(t for _, t in ctx.texts)
    failures: list[str] = []
    if not _POPULATION.search(text):
        failures.append("The brief does not say who was studied (students, grade level, schools, sample size).")
    if not _OUTCOME.search(text):
        failures.append("The brief does not say which outcomes were measured.")
    return _result(
        "study_population",
        WARNING,
        failures,
        "The studied population and measured outcomes are named.",
        "Name the studied population and measured outcomes before generalizing results.",
    )


def _check_guardrail_lint(ctx: _Context) -> CheckResult:
    failures = [f"{loc}: {violation}" for loc, text in ctx.texts for violation in lint_text(text, ctx.vertical)]
    if ctx.is_finance:
        return _result(
            "guardrail_lint",
            BLOCKER,
            failures,
            "No ratings, price targets, personalized advice, guaranteed returns, market-beating claims, "
            "or forecasts stated as fact.",
            "Finance guardrail violations: The Newsren's finance coverage is news and education only.",
        )
    return _result(
        "guardrail_lint",
        WARNING,
        failures,
        "No overclaiming language found.",
        "Overclaiming language: avoid stating certainty, guarantees, or causation the evidence does not support.",
    )


def _check_performance_labels(ctx: _Context) -> CheckResult:
    failures = [
        f"{_claim_label(i, c)}: {' '.join(issues)}"
        for i, c in enumerate(ctx.claims, 1)
        if (issues := performance_claim_issues(c))
    ]
    return _result(
        "performance_labels",
        BLOCKER if ctx.is_finance else WARNING,
        failures,
        "Performance claims are labelled backtest, simulated, or live, with period, benchmark, and source.",
        "Every performance claim needs a backtest/simulated/live label, its period, benchmark, and source.",
    )


def _check_high_impact(ctx: _Context) -> CheckResult:
    flagged = [(i, c) for i, c in enumerate(ctx.claims, 1) if c.high_impact]
    weak = [
        (i, c)
        for i, c in flagged
        if not c.citation_ids or _status(c) in (SupportStatus.UNVERIFIED, SupportStatus.CONTESTED)
    ]
    details = [
        _claim_label(i, c)
        + (" is high-impact and not firmly supported." if (i, c) in weak else " is high-impact.")
        for i, c in flagged
    ]
    return CheckResult(
        name="high_impact_claims",
        passed=not flagged,
        severity=WARNING if weak else INFO,
        message=(
            "No high-impact claims."
            if not flagged
            else f"{len(flagged)} high-impact claim(s): a human editor must verify them before publication."
        ),
        details=details,
    )


_CHECKS: tuple[Callable[[_Context], CheckResult], ...] = (
    _check_content_present,
    _check_sources_cited,
    _check_claims_recorded,
    _check_citation_coverage,
    _check_unknown_citation_ids,
    _check_claim_support,
    _check_approved_sources,
    _check_well_formed_links,
    _check_retrieval_timestamps,
    _check_publication_dates,
    _check_licensing_notes,
    _check_support_status,
    _check_vendor_attribution,
    _check_disagreements,
    _check_preprint_labelling,
    _check_forecast_labelling,
    _check_study_population,
    _check_guardrail_lint,
    _check_performance_labels,
    _check_high_impact,
)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def run_checks(session: Session, brief: Brief) -> list[CheckResult]:
    """Run every editorial quality-gate check on ``brief`` and return the results in ``CHECK_NAMES`` order."""
    citations, malformed = _parse_citations(brief.citations or [])
    claims = sorted(brief.claims or [], key=lambda c: (c.position or 0, c.id or 0))
    ctx = _Context(
        session=session,
        brief=brief,
        vertical=str(brief.vertical or "").lower(),
        citations=citations,
        malformed_records=malformed,
        claims=claims,
        texts=_brief_texts(brief, claims),
    )
    return [check(ctx) for check in _CHECKS]


def _field(check: CheckResult | dict[str, Any], name: str) -> Any:
    return check.get(name) if isinstance(check, dict) else getattr(check, name)


def failed_checks(checks: Sequence[CheckResult | dict[str, Any]], severity: str | None = None) -> list[Any]:
    """Checks that did not pass (optionally of one severity); accepts models or stored dicts."""
    return [
        c for c in checks if not _field(c, "passed") and (severity is None or _field(c, "severity") == severity)
    ]


def assess_risk(brief: Brief, checks: Sequence[CheckResult | dict[str, Any]]) -> str:
    """Risk level for routing: high for finance, any high-impact claim, or any blocker; medium for warnings."""
    if str(brief.vertical or "").lower() == Vertical.FINANCE:
        return RiskLevel.HIGH.value
    if any(c.high_impact for c in brief.claims or []) or failed_checks(checks, BLOCKER):
        return RiskLevel.HIGH.value
    if failed_checks(checks, WARNING):
        return RiskLevel.MEDIUM.value
    return RiskLevel.LOW.value
