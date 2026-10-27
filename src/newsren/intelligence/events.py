"""Rule-based event classification for story timelines and monitoring.

``classify_event(article)`` maps an article to an ``EventType`` value using
its ``doc_type``, ``trust_tier``, title, and summary. It is pure and fast so
monitoring can call it on every candidate article.

Rules are checked in this order; the first that fires wins:

1. **correction** - the piece corrects, retracts, or clarifies earlier
   reporting ("Correction:", "issued a correction", "retracted",
   "an earlier version of this article"). Outranks everything, including
   opinion, because a correction changes the evidence whatever its format.
2. **opinion / analysis by format** - ``doc_type`` opinion, ``trust_tier``
   commentary, or an explicit label ("Opinion:", "Op-ed", "Analysis:").
   An opinion column *about* a benchmark is not itself a benchmark, which
   the plan's Follow This Story example ("not whenever someone writes
   another opinion article") depends on. Cue-based opinion/analysis
   detection (rule 12) stays low in the order.
3. **enforcement** - regulators charging, fining, settling, or issuing
   cease-and-desist orders; lawsuits only when a public enforcer sues.
4. **security_advisory** - ``doc_type`` advisory, CVE ids, vulnerabilities,
   exploited flaws, data breaches.
5. **earnings / filing** - ``doc_type`` earnings_call, or quarterly-results
   language -> earnings; ``doc_type`` filing or SEC form numbers
   (10-K, 10-Q, 8-K, S-1 ...) -> filing.
6. **funding** - money raised in a round ("raises $40 million",
   "Series B", "seed round").
7. **independent_evaluation / benchmark** - evaluation language
   ("benchmark", "evaluated", "audit", "replicate", "leaderboard").
   ``independent_evaluation`` requires explicit third-party wording
   ("independent", "third-party", "researchers at X evaluated", "could not
   replicate") or a non-vendor source that is not merely relaying the
   company's own claim. A vendor's own benchmark claim (vendor trust tier or
   press release, or reporting that attributes the scores to the company)
   is ``benchmark``. Research papers need the evaluation cue in the title.
8. **research_publication** - ``doc_type`` research_preprint,
   research_peer_reviewed, or education_report; "study", "paper",
   "preprint", "peer-reviewed", "randomized".
9. **regulation / policy** - ``doc_type`` regulatory -> regulation;
   ``doc_type`` policy -> policy; legally binding rule-making language
   (rule, regulation, law, bill, "AI Act", compliance) -> regulation;
   government or institutional policy and guidance (curriculum, standards,
   monetary policy, guidance for schools) -> policy.
10. **product_launch** - launches, unveils, releases, "now available",
    "general availability", open-sourcing.
11. **adoption** - adopts, deploys, pilots, "rolls out to staff/schools",
    partnerships and contracts, "goes live".
12. **opinion / analysis by cue** - "why", "what it means", "explainer",
    "the case for/against", "lessons".
13. **announcement** - press releases or vendor sources, "announces",
    "plans to", "unveils plan".
14. **other**.
"""

from __future__ import annotations

import re
from typing import Protocol

from newsren.models import DocType, EventType, TrustTier
from newsren.textutil import normalize_space


class EventSubject(Protocol):
    """Anything with the article fields the rules read (``Article`` or a stand-in)."""

    title: str
    summary: str | None
    content: str | None
    doc_type: str
    trust_tier: str


def _rx(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


_CORRECTION = _rx(
    r"^\s*(?:correction|corrected|clarification|retraction|erratum)\b",
    r"\b(?:correction|clarification|erratum|editor's note)\s*:",
    r"\b(?:issues?|issued|publish(?:es|ed)?|posts?|posted|runs?|ran|appends?|appended)\s+(?:a\s+)?"
    r"(?:correction|erratum|retraction|clarification)\b",
    r"\bretract(?:s|ed|ion|ing)?\b",
    r"\bcorrect(?:s|ed|ing)\s+(?:its|their|his|her|an?|the)\s+(?:earlier\s+|previous\s+|prior\s+)?"
    r"(?:claim|figure|report|statement|benchmark|results?|numbers?|data|paper|article|story|score|scores)\b",
    r"\b(?:an?|the)\s+earlier\s+version\s+of\s+this\s+(?:article|story|report|post|release)\b",
    r"\bwithdr(?:aws|ew|awn)\s+(?:its|the|their)\s+(?:paper|claim|claims|results?|benchmark|study|preprint)\b",
)

_OPINION_LABEL = _rx(
    r"^\s*(?:opinion|op-ed|column|commentary|editorial|viewpoint|perspective|letter)\s*[:|\-–—]",
    r"\bop-ed\b",
)
_ANALYSIS_LABEL = _rx(r"^\s*(?:analysis|explainer|news analysis)\s*[:|\-–—]")

_REGULATOR = (
    r"SEC|FINRA|FTC|CFTC|CFPB|OCC|FDIC|DOJ|Justice Department|Department of Justice|attorneys? general|"
    r"regulators?|Federal Reserve|Fed|state of \w+|prosecutors?|commission|agency"
)
_ENFORCEMENT = _rx(
    r"\benforcement\s+action\b",
    r"\bcharged\b",
    r"\b(?:settled|settles|settle|brings?|brought|files?|filed)\s+(?:civil\s+|criminal\s+|fraud\s+)?charges\b",
    r"\bcharges\s+against\b",
    r"\b(?:fined|fines)\b",
    r"\b(?:civil|monetary)\s+penalt(?:y|ies)\b",
    r"\bpenali[sz](?:ed|es|ing)\b",
    r"\bcease[- ]and[- ]desist\b",
    r"\bdisgorgement\b",
    r"\bconsent\s+(?:order|decree)\b",
    r"\b(?:indicted|indictment|sanctioned|censured|barred\s+from)\b",
    rf"\b(?:{_REGULATOR})\s+(?:sues|sued|suing|charges)\b",
    rf"\b(?:sues|sued|lawsuit)\b.{{0,60}}\b(?:{_REGULATOR})\b",
)

_SECURITY = _rx(
    r"\bCVE-\d{4}-\d{3,}\b",
    r"\bsecurity\s+(?:advisory|bulletin|flaw|hole|breach)\b",
    r"\bvulnerabilit(?:y|ies)\b",
    r"\bzero[- ]day\b",
    r"\bactively\s+exploited\b",
    r"\b(?:data|security)\s+breach\b",
    r"\bransomware\b",
    r"\bmalware\b",
    r"\bpatch(?:es|ed)?\b[^.]{0,30}?\b(?:flaw|bug|vulnerabilit\w*|exploit\w*)",
    r"\bCISA\s+(?:warns|adds|issues|orders)\b",
)

# Weak cues count only in the title; strong cues anywhere in title + summary.
_EARNINGS_TITLE = _rx(r"\bearnings\b", r"\bquarterly\s+results\b")
_EARNINGS = _rx(
    r"\bquarterly\s+(?:results|revenue|profit|loss|earnings)\b",
    r"\b(?:Q[1-4]|first[- ]quarter|second[- ]quarter|third[- ]quarter|fourth[- ]quarter|full[- ]year)\s+"
    r"(?:results|revenue|sales|profit|loss|earnings)\b",
    r"\bfiscal\s+(?:first|second|third|fourth)\s+quarter\b",
    r"\bresults\s+of\s+operations\b",
    r"\bearnings\s+(?:call|release|report)\b",
    r"\b(?:earnings|diluted)\s+per\s+share\b",
    r"\b(?:beats?|miss(?:es|ed)?|tops?|topped)\s+(?:analyst\s+|wall\s+street\s+)?(?:estimates|expectations)\b",
    r"\bnet\s+income\b",
    r"\brevenue\s+(?:rose|fell|grew|declined|increased|decreased|jumped|climbed|dropped)\b",
)
_FILING = _rx(
    r"\b(?:10-K|10-Q|8-K|S-1|F-1|S-4|20-F|6-K|13F|13D|13G|DEF\s?14A|424B\d|Form\s+4|Form\s+D)\b",
    r"\b(?:files?|filed|filing)\s+(?:with|to)\s+the\s+(?:SEC|Securities and Exchange Commission)\b",
    r"\bregulatory\s+filing\b",
    r"\bprospectus\b",
    r"\bfiles?\s+(?:confidentially\s+)?for\s+(?:an?\s+)?IPO\b",
)

_FUNDING = _rx(
    r"\b(?:raises?|raised|raising|secures?|secured|lands?|landed|closes?|closed|nabs?|bags?|attracts?)\b"
    r"[^.]{0,40}?(?:\$|€|£|¥|US\$)\s?\d",
    r"\b(?:raises?|raised|secures?|secured)\s+(?:\d[\d.,]*\s*(?:million|billion|m|bn)|funding|capital)\b",
    r"\bSeries\s+[A-H]\b",
    r"\b(?:pre-seed|seed|growth|venture|funding|financing|investment|equity)\s+(?:round|funding|financing)\b",
    r"\bfunding\s+round\b",
    r"\b(?:led|co-led)\s+by\b[^.]{0,80}\b(?:Capital|Ventures|Partners|Horowitz|Combinator|Fund)\b",
    r"\bvaluation\s+of\s+(?:\$|€|£)",
)

_EVAL = _rx(
    # Finance "benchmark rate/index/portfolio" is a reference point, not an evaluation.
    r"\bbenchmark(?:s|ed|ing)?\b(?!\s+(?:interest\s+)?(?:rates?|index|indexes|indices|yields?|portfolios?|"
    r"bonds?|treasur(?:y|ies)|prices?|lending|loans?|stock|equity|S&P))",
    r"\bevaluat(?:e|es|ed|ing|ion|ions)\b",
    r"\bleaderboard\b",
    r"\bhead-to-head\b",
    r"\bred[- ]team(?:s|ed|ing)?\b",
    r"\baudit(?:s|ed|ing)?\b",
    r"\breplicat(?:e|es|ed|ion|ing)\b",
    r"\breproduc(?:e|ed|ibility|tion)\b",
    r"\b(?:tests?|testing|trials?)\s+(?:show|showed|shows|found|find|finds|suggest|suggests)\b",
    r"\b(?:independent(?:ly)?|third-party|external)\s+(?:tests?|tested|testing|review)\b",
    r"\bscor(?:es|ed|ing)\s+\d",
    r"\bresearchers\s+(?:at|from)\s+[^.]{1,80}?\b(?:evaluated|tested|assessed|benchmarked|audited|replicated)\b",
)
_THIRD_PARTY = _rx(
    r"\bindependent(?:ly)?\b",
    r"\bthird[- ]party\b",
    r"\bresearchers\s+(?:at|from)\s+[^.]{1,80}?\b(?:evaluated|tested|assessed|benchmarked|audited|"
    r"replicated|examined|reviewed|found)\b",
    r"\b(?:external|outside)\s+(?:evaluat|audit|review|test)",
    r"\bcould\s+not\s+(?:replicate|reproduce)\b",
    r"\bfailed\s+to\s+(?:replicate|reproduce)\b",
)
# "not yet independently verified", "no third-party tests": negated independence is not independence.
_NEGATED_THIRD_PARTY = _rx(
    r"\b(?:not\s+(?:yet\s+|been\s+)?|no\s+|without\s+(?:an?\s+)?|lack(?:s|ing)?\s+(?:an?\s+)?|absent\s+|"
    r"(?:awaits?|awaiting|pending|needs?|requires?)\s+(?:an?\s+)?)(?:independent(?:ly)?|third[- ]party|outside|"
    r"external)\b"
    r"(?:\s+\w+)?"
)
_VENDOR_ATTRIBUTION = _rx(
    r"\b(?:says|said)\s+its\b",
    r"\bclaim(?:s|ed|ing)?\s+(?:that\s+)?(?:its|the\s+model|top|state[- ]of[- ]the[- ]art|record|best)\b",
    r"\baccording\s+to\s+the\s+(?:company|firm|vendor|startup|lab)\b",
    r"\b(?:company|vendor|self)[- ]reported\b",
    r"\b(?:internal|in-house|its\s+own)\s+(?:benchmark|test|evaluation)s?\b",
)

_RESEARCH_TITLE = _rx(
    r"\bstud(?:y|ies)\b",
    r"\bpaper\b(?!\s+trading)",
    r"\bresearchers?\b",
    r"^\s*survey\s*:",
    r"\bsurvey\s+(?:finds|shows|of)\b",
)
_RESEARCH = _rx(
    r"\bpreprint\b",
    r"\barXiv\b",
    r"\bpeer[- ]reviewed\b",
    r"\bpublished\s+in\s+(?:the\s+)?(?:journal|Nature|Science|Proceedings|PNAS|The\s+Lancet)\b",
    r"\b(?:working|research)\s+paper\b",
    r"\brandomi[sz]ed\s+(?:controlled\s+)?(?:trial|study|experiment)\b",
    r"\bmeta-analysis\b",
)

_REGULATION_TITLE = _rx(
    r"\bregulat(?:ion|ions|ors?|ory)\b", r"\brules?\b", r"\blaw\b", r"\bbill\b", r"\blegislation\b"
)
_REGULATION = _rx(
    r"\b(?:proposed|final|new|interim)\s+rules?\b",
    r"\brule-?making\b",
    r"\bregulators?\s+(?:propose|proposes|proposed|adopt|adopts|adopted|issue|issues|issued|approve|approves)\b",
    r"\b(?:SEC|FINRA|CFTC|FTC|CFPB|OCC|FDIC|European\s+Commission)\s+"
    r"(?:proposes|adopts|issues|finalizes|approves|publishes|releases|updates)\b",
    r"\boversight\s+report\b",
    r"\bsigned\s+into\s+law\b",
    r"\b(?:passes|passed|votes?\s+on|introduces?|introduced)\b[^.]{0,40}\b(?:bill|act|law)\b",
    r"\bAI\s+Act\b",
    r"\bcompliance\s+(?:deadline|requirements?|date)\b",
    r"\bexecutive\s+order\b",
)
_POLICY_TITLE = _rx(
    r"\bpolic(?:y|ies)\b",
    r"\bguidance\b",
    r"\bcurriculum\b",
    r"\bstandards\b",
    r"\bmandates?\b",
    r"\bbans?\b",
    r"\brequirements?\b",
)
_POLICY = _rx(
    r"\bmonetary\s+policy\b",
    r"\binterest[- ]rate\s+decision\b",
    r"\b(?:raises|cuts|holds|lowers|leaves)\s+(?:interest\s+)?rates\b",
    r"\bfederal\s+funds\s+rate\b",
    r"\bguidance\s+(?:for|to)\s+(?:schools|districts|teachers|educators|colleges|universities|states)\b",
    r"\b(?:board|department)\s+of\s+education\s+(?:issues|issued|releases|released|announces|announced|"
    r"adopts|adopted|approves|approved|votes|voted)\b",
    r"\bcurriculum\s+(?:standards|framework|changes?|requirements?)\b",
    r"\bgrant\s+program\b",
)

_LAUNCH = _rx(
    r"\blaunch(?:es|ed|ing)?\b",
    r"\bunveil(?:s|ed|ing)?\b(?!\s+(?:a\s+)?plans?\b)",
    r"\bdebut(?:s|ed)?\b",
    r"\bintroduc(?:es|ed|ing)\b",
    r"\breleas(?:es|ed|ing)\b(?!\s+(?:its\s+)?(?:quarterly|results|earnings|guidance|minutes))",
    r"\bnow\s+available\b",
    r"\bgenerally\s+available\b|\bgeneral\s+availability\b",
    r"\bopen[- ]sourc(?:es|ed|ing)\b",
    r"\bavailable\s+(?:today|now|in\s+preview)\b",
)
_ADOPTEES = (
    r"staff|employees|workers|students|teachers|schools|districts|classrooms|bankers|advisers|advisors|"
    r"analysts|users|customers|campuses"
)
_ADOPTION = _rx(
    r"\badopt(?:s|ed|ing|ion)?\b",
    r"\bdeploy(?:s|ed|ing|ment)?\b",
    rf"\broll(?:s|ed)?\s+out\b[^.]{{0,60}}?\b(?:to|across|for)\s+(?:its\s+|about\s+|some\s+|all\s+)?"
    rf"(?:[\d,]+\s+)?(?:\w+\s+)?(?:{_ADOPTEES})\b",
    r"\bpilot(?:s|ed|ing)?\b",
    r"\bpartner(?:s|ed|ship)?\s+with\b",
    r"\bsign(?:s|ed)?\s+(?:a\s+)?(?:multi-year\s+|\$[\d.,]+\s*(?:million|billion)\s+)?(?:deal|contract|agreement)\b",
    r"\b(?:begin|began|begins|start|starts|started)\s+using\b",
    r"\b(?:goes|went)\s+live\b",
    r"\bin\s+production\b",
    r"\bintegrat(?:es|ed|ing)\b[^.]{0,40}\binto\b",
)
_ANALYSIS_CUE = _rx(
    r"^\s*(?:why|how|what)\b",
    r"\bwhat\s+[^.;:]{0,60}?\b(?:would\s+|will\s+|could\s+)?means?\b",
    r"\bexplain(?:er|ed)\b",
    r"\bthe\s+case\s+(?:for|against)\b",
    r"\blessons\s+(?:from|for)\b",
    r"\banalysis\b",
    r"\bdeep\s+dive\b",
)
_OPINION_CUE = _rx(r"\bopinion\b", r"\bcolumn(?:ist)?\b", r"\bcommentary\b", r"\beditorial\b", r"\bwe\s+believe\b")
_ANNOUNCEMENT = _rx(
    r"\bannounc(?:es|ed|ing|ement)\b",
    r"\bunveils?\s+(?:a\s+)?plans?\b",
    r"\bplans?\s+to\b",
    r"\bstatement\b",
    r"\bpledges?\b",
    r"\bcommits?\s+to\b",
)

_RESEARCH_DOCS = {DocType.RESEARCH_PREPRINT, DocType.RESEARCH_PEER_REVIEWED}
# Official records are classified by what they are, never as evaluations.
_RECORD_DOCS = {DocType.REGULATORY, DocType.POLICY, DocType.FILING, DocType.EARNINGS_CALL, DocType.ADVISORY}


def _text_of(article: EventSubject) -> tuple[str, str, str]:
    title = normalize_space(article.title)
    summary = normalize_space(article.summary)
    if not summary:
        summary = normalize_space((article.content or "")[:600])
    return title, summary, (f"{title}. {summary}" if summary else title)


def _first_pos(rx: re.Pattern[str], text: str) -> int:
    m = rx.search(text)
    return m.start() if m else -1


def classify_event_with_reason(article: EventSubject) -> tuple[str, str]:
    """(EventType value, rule name) for ``article``. See the module docstring for the rule order."""
    title, summary, text = _text_of(article)
    doc_type = (article.doc_type or "").lower()
    tier = (article.trust_tier or "").lower()
    is_vendor = tier == TrustTier.VENDOR or doc_type == DocType.PRESS_RELEASE

    # 1. Corrections outrank everything.
    if _CORRECTION.search(text):
        return EventType.CORRECTION, "correction"

    # 2. Commentary by format.
    if _ANALYSIS_LABEL.search(title):
        return EventType.ANALYSIS, "analysis_label"
    if doc_type == DocType.OPINION or tier == TrustTier.COMMENTARY or _OPINION_LABEL.search(title):
        if _ANALYSIS_CUE.search(title) and not _OPINION_CUE.search(title):
            return EventType.ANALYSIS, "commentary_analysis"
        return EventType.OPINION, "commentary"

    # 3-4. Enforcement and security advisories.
    if _ENFORCEMENT.search(text):
        return EventType.ENFORCEMENT, "enforcement"
    if doc_type == DocType.ADVISORY or _SECURITY.search(text):
        return EventType.SECURITY_ADVISORY, "security_advisory"

    # 5. Earnings and filings.
    if doc_type == DocType.EARNINGS_CALL:
        return EventType.EARNINGS, "earnings_doc"
    if doc_type == DocType.FILING:
        if _EARNINGS.search(text):
            return EventType.EARNINGS, "filing_earnings"
        return EventType.FILING, "filing_doc"
    if _EARNINGS.search(text) or _EARNINGS_TITLE.search(title):
        return EventType.EARNINGS, "earnings"
    if _FILING.search(text):
        return EventType.FILING, "filing"

    # 6. Funding rounds.
    if _FUNDING.search(text):
        return EventType.FUNDING, "funding"

    # 7. Evaluations and benchmarks. A cue only in the summary of a launch or
    # announcement story describes that launch, not a new evaluation.
    launch_in_title = _first_pos(_LAUNCH, title)
    eval_in_title = _first_pos(_EVAL, title)
    if doc_type in _RECORD_DOCS:
        eval_hit = False
    elif launch_in_title >= 0 and (eval_in_title < 0 or launch_in_title < eval_in_title):
        eval_hit = False  # "X unveils Y; benchmark claims unverified" reports a launch
    elif doc_type in _RESEARCH_DOCS:
        eval_hit = eval_in_title >= 0
    else:
        eval_hit = eval_in_title >= 0 or bool(_EVAL.search(summary) and not _ANNOUNCEMENT.search(title))
    if eval_hit:
        if is_vendor:
            if eval_in_title >= 0:
                return EventType.BENCHMARK, "vendor_benchmark"
        elif _THIRD_PARTY.search(_NEGATED_THIRD_PARTY.sub(" ", text)):
            return EventType.INDEPENDENT_EVALUATION, "third_party_evaluation"
        elif _VENDOR_ATTRIBUTION.search(text):
            return EventType.BENCHMARK, "reported_vendor_benchmark"
        elif tier in (TrustTier.INDEPENDENT, TrustTier.PRIMARY):
            return EventType.INDEPENDENT_EVALUATION, "non_vendor_evaluation"

    # 8. Research publications.
    if doc_type in _RESEARCH_DOCS or doc_type == DocType.EDUCATION_REPORT:
        return EventType.RESEARCH_PUBLICATION, "research_doc"
    if not is_vendor and (_RESEARCH.search(text) or _RESEARCH_TITLE.search(title)):
        return EventType.RESEARCH_PUBLICATION, "research"

    # 9. Regulation and policy.
    if doc_type == DocType.REGULATORY:
        return EventType.REGULATION, "regulatory_doc"
    if doc_type == DocType.POLICY:
        return EventType.POLICY, "policy_doc"
    if _REGULATION.search(text) or _REGULATION_TITLE.search(title):
        return EventType.REGULATION, "regulation"
    if _POLICY.search(text) or _POLICY_TITLE.search(title):
        return EventType.POLICY, "policy"

    # 10-11. Launches and adoption; when the title has both, the earlier verb wins.
    launch_at, adopt_at = _first_pos(_LAUNCH, title), _first_pos(_ADOPTION, title)
    if launch_at >= 0 and (adopt_at < 0 or launch_at < adopt_at):
        return EventType.PRODUCT_LAUNCH, "product_launch"
    if adopt_at >= 0:
        return EventType.ADOPTION, "adoption"
    if _LAUNCH.search(summary):
        return EventType.PRODUCT_LAUNCH, "product_launch"
    if _ADOPTION.search(summary):
        return EventType.ADOPTION, "adoption"

    # 12. Opinion and analysis by cue.
    if _OPINION_CUE.search(title):
        return EventType.OPINION, "opinion_cue"
    if _ANALYSIS_CUE.search(title):
        return EventType.ANALYSIS, "analysis_cue"

    # 13-14.
    if is_vendor or _ANNOUNCEMENT.search(text):
        return EventType.ANNOUNCEMENT, "announcement"
    return EventType.OTHER, "other"


def classify_event(article: EventSubject) -> str:
    """``EventType`` value for ``article`` (rule-based, pure, fast)."""
    return str(classify_event_with_reason(article)[0])


EVENT_LABELS: dict[str, str] = {
    EventType.ANNOUNCEMENT: "announcement",
    EventType.RESEARCH_PUBLICATION: "research publication",
    EventType.INDEPENDENT_EVALUATION: "independent evaluation",
    EventType.BENCHMARK: "benchmark claim",
    EventType.PRODUCT_LAUNCH: "product launch",
    EventType.ADOPTION: "adoption",
    EventType.REGULATION: "regulation",
    EventType.POLICY: "policy",
    EventType.FUNDING: "funding round",
    EventType.EARNINGS: "earnings",
    EventType.FILING: "filing",
    EventType.ENFORCEMENT: "enforcement action",
    EventType.CORRECTION: "correction",
    EventType.SECURITY_ADVISORY: "security advisory",
    EventType.OPINION: "opinion",
    EventType.ANALYSIS: "analysis",
    EventType.OTHER: "development",
}


def event_label(event_type: str | None) -> str:
    """Reader-facing label for an ``EventType`` value."""
    return EVENT_LABELS.get(event_type or EventType.OTHER, (event_type or "development").replace("_", " "))
