"""Editorial guardrails: the finance notice, the advice redirect, and a language linter.

The plan's finance section is news and education only: no buy/sell/hold calls,
price targets, portfolio recommendations, or personalized investment advice;
every performance claim labelled backtest, simulated, or live with its period,
benchmark, and source; forecasts never presented as facts; and never a claim
that an AI agent can reliably beat the market. Research and education coverage
must not overstate certainty or causality.

``lint_text`` is deterministic and rule-based. Each rule knows which verticals
it applies to and which kinds of context make a match legitimate reporting
rather than a violation:

* negation just before the match ("does not prove", "nothing here is a
  recommendation to buy");
* attribution in the same sentence ("the firm claimed guaranteed returns",
  "analysts said shares will soar") or a quoted phrase;
* hedging in the same sentence for forecasts ("analysts expect prices will rise").

Rating calls and price targets are flagged even when attributed, because the
plan rules them out of finance coverage entirely; an editor can still approve
such a brief with an explicit override note.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from newsren.models import BriefClaim, ClaimType, PerformanceLabel, Vertical
from newsren.schemas import ClaimOut
from newsren.textutil import normalize_space, truncate

FINANCE_DISCLAIMER = (
    "For information and education only: this is not investment advice. The Newsren's finance coverage "
    "reports and explains the news; nothing here is a recommendation to buy, sell, or hold any security, "
    "and nothing here is tailored to your personal circumstances. Performance figures are labelled as "
    "backtest, simulated, or live, with their period, benchmark, and source, and backtested or simulated "
    "results are not evidence of future returns. Market data is timestamped and may be delayed."
)

ADVICE_REFUSAL = (
    "The Newsren doesn't give personalized investment advice, so we won't tell you whether to buy, sell, "
    "or hold a particular security, offer price targets, or suggest how to allocate your money. "
    "What we can do is help you understand the news behind your question: what the company or fund "
    "actually reported in its filings, earnings materials, or announcements; how those figures compare "
    "with guidance and earlier periods; the concepts involved (for example valuation, dilution, "
    "backtesting versus live results, drawdown, and risk); and the questions an analyst would look at "
    "before forming a view. For a decision about your own money, consider speaking with a licensed "
    "financial adviser, such as a registered investment adviser or a fiduciary financial planner, who "
    "can take your situation into account."
)

_I = re.IGNORECASE

# ---------------------------------------------------------------- context cues

# Negation in the few words before a match turns a claim into its denial.
_NEGATION = re.compile(
    r"\b(?:not|no|never|nothing|none|nor|neither|without|cannot|unproven|whether|rather\s+than|instead\s+of|"
    r"fail(?:s|ed)?\s+to|unlikely\s+to|doubt(?:s|ful)?)\b|n['’]t\b",
    _I,
)

# Reported speech: the sentence attributes the statement to someone else.
_ATTRIBUTION = re.compile(
    r"\b(?:claim(?:s|ed|ing)?|said|says|say|advertis(?:e|es|ed|ing)|marketed|marketing|tout(?:s|ed|ing)?|"
    r"according\s+to|alleg(?:e|es|ed|edly|ation|ations)|charg(?:e|es|ed)|assert(?:s|ed)?|boast(?:s|ed)?|"
    r"pitch(?:es|ed)?|purport(?:s|ed|edly)?|misleading(?:ly)?|falsely|told|wrote|argu(?:e|es|ed)|"
    r"insist(?:s|ed)?|suggest(?:s|ed)|press\s+release|headline|pledg(?:e|es|ed)|promis(?:e|es|ed|ing))\b",
    _I,
)

# Hedging / labelling words that mark a statement as a forecast, projection, or scenario.
_HEDGE = re.compile(
    r"\b(?:could|may|might|would|if|unless|likely|unlikely|possibl[ey]|potential(?:ly)?|perhaps|"
    r"expect(?:s|ed|ing|ation|ations)?|forecast(?:s|ed|ing|er|ers)?|project(?:s|ed|ing|ion|ions)|"
    r"estimat(?:e|es|ed|ing|ion|ions)|predict(?:s|ed|ing|ion|ions)?|anticipat(?:e|es|ed|ing)|"
    r"scenario(?:s)?|outlook|guidance|consensus|plan(?:s|ned)?\s+to|aim(?:s|ed)?\s+to|target(?:s|ed|ing)?|"
    r"hope(?:s|d)?\s+to|intend(?:s|ed)?\s+to|believ(?:e|es|ed)|assum(?:e|es|ed|ing|ption|ptions)|"
    r"hypothe\w*|uncertain\w*|risk\s+that)\b",
    _I,
)

_EVIDENCE_OF_CAUSATION = re.compile(r"\b(?:randomi[sz]ed|controlled\s+trial|rct|experiment(?:al)?)\b", _I)

# Sentence boundaries (a period followed by a digit, as in "$1.2 billion", is not one).
_BOUNDARY = re.compile(r"[.!?;]\s+|\n")
_QUOTES = re.compile(r"[\"“”]")
_NEGATION_WINDOW = 60


@dataclass(frozen=True)
class _Rule:
    key: str
    label: str
    advice: str
    patterns: tuple[re.Pattern[str], ...]
    verticals: frozenset[str] | None = None  # None: every vertical
    negation: bool = True
    attribution: bool = False
    hedge: bool = False
    # Text immediately before the match that makes it legitimate (e.g. "returns above risk-free returns").
    exempt_before: re.Pattern[str] | None = None
    # Sentence-level evidence that makes the match legitimate (e.g. a randomized trial for causal language).
    exempt_sentence: re.Pattern[str] | None = None


def _rx(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, _I) for p in patterns)


_FINANCE = frozenset({Vertical.FINANCE.value})
_PROJECTION_VERTICALS = frozenset({Vertical.FINANCE.value, Vertical.INDUSTRY.value})

# "hold" is also a central-bank verb ("reiterated its plan to hold rates steady").
_HOLD = (
    r"hold(?!\s+(?:rates?|interest|steady|its|their|the|off|on|back|firm|still|fire|talks|hearings?|meetings?))"
)
_RATING_WORD = (
    rf"(?:strong\s+)?(?:buy|sell|outperform|underperform|overweight|underweight|neutral|{_HOLD})(?![-\w])"
)

_MARKET_MOVE = (
    r"(?:rise|fall|soar|crash|surge|plunge|plummet|skyrocket|double|triple|halve|collapse|rally|tumble|jump|"
    r"drop|climb|decline|rebound|recover|spike|sink|slump|grow|shrink|increase|decrease)"
)
_MARKET_SUBJECT = (
    r"(?:prices?|shares|stocks?|markets?|index|indices|valuations?|rates?|yields?|revenues?|earnings|sales|"
    r"profits?|margins?|demand|supply|costs?|output|capacity|bitcoin|crypto(?:currenc(?:y|ies))?|the\s+dollar|"
    r"inflation|unemployment|gdp|economy|growth|spending)"
)

_RULES: tuple[_Rule, ...] = (
    # ------------------------------------------------------------ finance only
    _Rule(
        key="rating_call",
        label="Buy/sell/hold call",
        advice="The Newsren does not issue ratings or recommendations on securities; report the underlying facts.",
        verticals=_FINANCE,
        patterns=_rx(
            r"\bstrong\s+(?:buy|sell)\b",
            r"\b(?:buy|sell|hold|outperform|underperform|overweight|underweight|market[- ]perform|sector[- ]perform|"
            r"equal[- ]weight)\s+(?:rating|recommendation|call)s?\b",
            r"\b(?:rated|upgrad(?:e|es|ed|ing)|downgrad(?:e|es|ed|ing)|reiterat(?:e|es|ed|ing)|"
            r"initiat(?:e|es|ed|ing)\s+coverage)\b[^.;\n]{0,40}?\b(?:to|as|at|with)\s+(?:an?\s+)?" + _RATING_WORD,
            r"\b(?:is|are|looks?\s+like|remains?|stays?|it['’]s)\s+an?\s+(?:strong\s+)?(?:buy|sell)(?![-\w])",
            r"\brecommend(?:s|ed|ing|ation)?\s+(?:to\s+)?(?:buy|sell|" + _HOLD + r")(?:ing)?(?![-\w])",
            r"\b(?:top|best)\s+(?:stock\s+)?picks?\b|\bstock\s+picks?\b",
        ),
    ),
    _Rule(
        key="price_target",
        label="Price target",
        advice="The Newsren does not publish price targets.",
        verticals=_FINANCE,
        patterns=_rx(
            r"\$\s?\d[\d,.]*(?:\s*(?:k|m|bn|b|billion|million))?\s+(?:price\s+target|target\s+price)\b",
            r"\b(?:price\s+target|target\s+price)s?\s+(?:of|to|at|is|was|from|near|around)\s+"
            r"(?:about\s+|around\s+|roughly\s+)?\$?\s?\d[\d,]*(?:\.\d+)?",
            r"\b(?:rais(?:e|es|ed|ing)|lift(?:s|ed|ing)?|cut(?:s|ting)?|lower(?:s|ed|ing)?|set(?:s|ting)?|"
            r"maintain(?:s|ed|ing)?|reiterat(?:e|es|ed|ing)|boost(?:s|ed|ing)?|trim(?:s|med|ming)?|hike[sd]?)\s+"
            r"(?:its|their|our|the|a|his|her)\s+(?:\w+\s+){0,2}?(?:price\s+target|target\s+price)\b",
            r"\b(?:our|my)\s+(?:\w+\s+)?(?:price\s+target|target\s+price)\b",
        ),
    ),
    _Rule(
        key="personal_advice",
        label="Personalized investment advice",
        advice="Explain the concepts and evidence; do not tell readers what to buy, sell, hold, or how to allocate.",
        verticals=_FINANCE,
        patterns=_rx(
            r"\byou\s+(?:should|must|need\s+to|ought\s+to|might\s+want\s+to|may\s+want\s+to|had\s+better)\s+"
            r"(?:consider\s+|seriously\s+consider\s+|definitely\s+|probably\s+)?(?:invest(?:ing)?|buy(?:ing)?|"
            r"sell(?:ing)?|hold(?:ing)?\s+(?:on\s+to\s+)?(?:the|your|this|these|shares|stocks?)|short(?:ing)?|"
            r"dump(?:ing)?|load(?:ing)?\s+up|allocat(?:e|ing)|rebalanc(?:e|ing)|"
            r"mov(?:e|ing)\s+your\s+(?:money|savings|portfolio|401)|"
            r"put(?:ting)?\s+(?:your|some|more|all)\s+(?:money|savings|cash|capital)|get(?:ting)?\s+(?:in|out)\b|"
            r"add(?:ing)?\s+(?:to\s+)?(?:your\s+)?(?:position|holdings|stake))",
            r"\b(?:investors|readers|retail\s+investors|savers|traders|everyone)\s+(?:should|must|need\s+to|"
            r"ought\s+to)\s+(?:buy|sell|hold|invest|short|dump|load\s+up|accumulate|avoid\s+(?:the\s+)?"
            r"(?:stock|shares)|get\s+(?:in|out)|pile\s+into|rotate\s+into)\b",
            r"\b(?:we|i)\s+(?:recommend|suggest|advise)\s+(?:that\s+(?:you|investors|readers)\s+)?(?:buy|buying|"
            r"sell|selling|hold|holding|short|shorting|invest|investing|allocat\w+|you\s+(?:buy|sell|hold|invest))\b",
            r"\byour\s+(?:portfolio|401\s*\(?k\)?|ira|retirement\s+(?:savings|account|money)|savings|nest\s+egg|"
            r"investments?)\s+(?:should|needs?\s+to|must|ought\s+to)\b",
            r"\b(?:allocate|put|invest|move|keep)\s+(?:about\s+|around\s+|at\s+least\s+|no\s+more\s+than\s+|"
            r"up\s+to\s+)?\d+\s*(?:%|percent)\s+of\s+your\b",
            r"\b(?:now\s+is\s+(?:a|the)\s+(?:good|great|right|perfect|ideal)\s+time|it['’]s\s+(?:a\s+)?"
            r"(?:good|great|the\s+right)\s+time)\s+to\s+(?:buy|sell|invest|get\s+in|load\s+up|take\s+profits?)\b",
            r"\b(?:buy|sell|short)\s+(?:the\s+dip|(?:it|them|the\s+stock|shares)\s+(?:right\s+)?now|(?:right\s+)?now)"
            r"\b(?![\s,–—-]*pay\s+later)",
        ),
    ),
    # ------------------------------------------------------------ finance + industry
    _Rule(
        key="forecast_as_fact",
        label="Forecast stated as fact",
        advice="Label it as a forecast or projection and attribute it (e.g. 'the company projects', 'analysts expect').",
        verticals=_PROJECTION_VERTICALS,
        attribution=True,
        hedge=True,
        patterns=_rx(
            # Filler words may hold decimals ("$1.2") but never cross a sentence boundary.
            rf"\b{_MARKET_SUBJECT}\b(?:\s+[\w'’$%,/-]+(?:\.\d+[\w%]*)?){{0,5}}?\s+"
            rf"(?:will|(?:is|are)\s+(?:going|set|certain|sure|bound|poised|guaranteed)\s+to)\s+"
            rf"(?:\w+\s+){{0,2}}?{_MARKET_MOVE}\b",
        ),
    ),
    # ------------------------------------------------------------ every vertical
    _Rule(
        key="guaranteed_returns",
        label="Guaranteed or risk-free return claim",
        advice="No investment return is guaranteed or risk-free; attribute the claim to whoever made it or remove it.",
        attribution=True,
        exempt_before=re.compile(
            r"(?:above|over|minus|than|with|against|versus|vs\.?|exceed\w*|excess\s+of|relative\s+to)\s+(?:the\s+)?$",
            _I,
        ),
        patterns=_rx(
            r"\b(?:guarantee[sd]?|guaranteeing|assured)\s+(?:a\s+|an\s+)?(?:\w+\s+){0,2}?(?:returns?|profits?|"
            r"gains?|income|yields?|payouts?)\b",
            r"\b(?:risk[- ]free|riskless|no[- ]risk|zero[- ]risk)\s+(?:\w+\s+)?(?:returns?|profits?|gains?|income|"
            r"investments?|bets?|trades?|money)\b",
            r"\bcan(?:not|['’]t)\s+lose\b",
            r"\bsure[- ](?:thing|bet|fire)\b",
        ),
    ),
    _Rule(
        key="beat_market",
        label="Claim that an AI agent or strategy beats the market",
        advice=(
            "Never state that an AI agent or strategy can reliably beat the market; report the specific result "
            "with its label (backtest, simulated, live), period, and benchmark."
        ),
        attribution=True,
        patterns=_rx(
            r"\b(?:can|will|always|consistently|reliably|routinely|guaranteed\s+to|able\s+to|proven\s+to|sure\s+to|"
            r"certain\s+to)\s+(?:\w+\s+){0,2}?(?:beat|beats|outperform|outperforms|outsmart|outsmarts)\s+(?:the\s+)?"
            r"(?:overall\s+|broader\s+|stock\s+)?(?:market|markets|s&p(?:\s*500)?|index|indexes|indices|nasdaq|dow)\b"
            r"(?!['’]s|\s+(?:expectations|estimates|consensus|forecasts?))",
            r"\bmarket[- ]beating\b",
            r"\b(?:beat|beats|outperform|outperforms)\s+the\s+market\s+(?:every|each)\s+(?:year|quarter|month|time)\b",
        ),
    ),
    _Rule(
        key="overclaim_proof",
        label="Overstated certainty",
        advice="Research supports or suggests; say what the study measured and how strong the evidence is.",
        attribution=True,
        hedge=True,
        patterns=_rx(
            r"\b(?:proves?|proved|proven)\s+(?:that\b|beyond\b|conclusively\b)",
            r"\b(?:has|have|is|are|was|were|been)\s+proven\s+to\b",
            r"\bproof\s+that\b",
            r"\b(?:scientifically|clinically|empirically)\s+proven\b",
            r"\b(?:definitively|conclusively|undeniably|indisputably|unequivocally)\s+(?:shows?|showed|shown|proves?|"
            r"proved|establish(?:es|ed)?|demonstrates?|demonstrated|confirms?|confirmed)\b",
            r"\b(?:settles|settled)\s+the\s+(?:debate|question|matter)\b",
        ),
    ),
    _Rule(
        key="overclaim_guarantee",
        label="Guarantee language",
        advice="Avoid promising outcomes; describe what the evidence shows and its limits.",
        attribution=True,
        patterns=_rx(
            r"\bguarantee[sd]?\s+(?:to\s+\w+|that\b|(?:better|higher|improved|faster|success|results?|outcomes?|"
            r"learning|grades|scores|accuracy|safety)\b)",
            r"\b100\s*(?:%|percent)\s+(?:accurate|effective|safe|secure|reliable|certain)\b",
            r"\b(?:will|is\s+going\s+to)\s+(?:definitely|certainly|undoubtedly|inevitably|surely)\b",
        ),
    ),
    _Rule(
        key="causal_overclaim",
        label="Causal claim about outcomes",
        advice=(
            "Say whether the study design supports causation (randomized, with a comparison group) or describe "
            "the result as an association, and name the population studied."
        ),
        attribution=True,
        hedge=True,
        exempt_sentence=_EVIDENCE_OF_CAUSATION,
        patterns=_rx(
            r"\b(?:causes?|caused|causing)\s+(?:students\s+to\s+(?:learn|score|perform)|(?:better|higher|improved|"
            r"greater|increased|lower|reduced|faster)\s+(?:learning|test\s+scores?|scores?|grades|outcomes?|"
            r"achievement|performance|results|retention))",
            r"\b(?:leads?|led)\s+to\s+(?:better|higher|improved|greater|increased)\s+(?:learning|test\s+scores?|"
            r"scores?|grades|outcomes?|achievement|retention)\b",
            r"\b(?:boosts?|boosted|improves?|improved|raises?|raised)\s+(?:student\s+)?(?:learning|test\s+scores?|"
            r"achievement|grades)\s+(?:for\s+all|for\s+every|in\s+every|everywhere|across\s+the\s+board)\b",
        ),
    ),
)

RULE_KEYS: tuple[str, ...] = tuple(rule.key for rule in _RULES)


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    s = 0
    for m in _BOUNDARY.finditer(text, 0, start):
        s = m.end()
    nxt = _BOUNDARY.search(text, end)
    return s, (nxt.start() if nxt else len(text))


def _inside_quotes(prefix: str) -> bool:
    return len(_QUOTES.findall(prefix)) % 2 == 1


def _is_exempt(rule: _Rule, text: str, m: re.Match[str]) -> bool:
    s, e = _sentence_bounds(text, m.start(), m.end())
    before = text[max(s, m.start() - _NEGATION_WINDOW) : m.start()]
    rest_of_sentence = text[s : m.start()] + " " + text[m.end() : e]
    if rule.negation and _NEGATION.search(before):
        return True
    if rule.exempt_before is not None and rule.exempt_before.search(text[s : m.start()]):
        return True
    if rule.attribution and (_ATTRIBUTION.search(rest_of_sentence) or _inside_quotes(text[s : m.start()])):
        return True
    if rule.hedge and _HEDGE.search(rest_of_sentence):
        return True
    return rule.exempt_sentence is not None and bool(rule.exempt_sentence.search(text[s:e]))


def _applies(rule: _Rule, vertical: str) -> bool:
    return rule.verticals is None or vertical in rule.verticals


def lint_text(text: str | None, vertical: str | None) -> list[str]:
    """Return human-readable guardrail violations in ``text`` for a brief in ``vertical``.

    Finance text is checked against every rule (ratings, price targets,
    personalized advice, guaranteed returns, market-beating claims, forecasts
    stated as fact, overclaiming). Industry adds forecasts stated as fact to
    the overclaiming rules that apply everywhere (proof/guarantee language,
    causal overclaims, guaranteed returns, market-beating claims).
    Each violation reads ``'<label>: "<excerpt>" — <what to do>'``.
    """
    text = normalize_space(text)
    if not text:
        return []
    vertical_key = str(vertical or "").strip().lower()
    taken: list[tuple[int, int]] = []
    found: list[tuple[int, str]] = []
    seen: set[tuple[str, str]] = set()
    for rule in _RULES:
        if not _applies(rule, vertical_key):
            continue
        for pattern in rule.patterns:
            for m in pattern.finditer(text):
                if any(m.start() < end and start < m.end() for start, end in taken):
                    continue
                if _is_exempt(rule, text, m):
                    continue
                excerpt = truncate(m.group(0), 90)
                key = (rule.key, excerpt.lower())
                if key in seen:
                    continue
                seen.add(key)
                taken.append((m.start(), m.end()))
                found.append((m.start(), f'{rule.label}: "{excerpt}" — {rule.advice}'))
    return [message for _, message in sorted(found)]


def is_hedged(text: str | None) -> bool:
    """True when ``text`` labels itself as a forecast, projection, estimate, or scenario."""
    return bool(_HEDGE.search(text or ""))


# ---------------------------------------------------------------- advice requests

_FIN_NOUN = (
    r"(?:stocks?|shares?|equit(?:y|ies)|etfs?|index\s+funds?|mutual\s+funds?|funds?|bonds?|options?|calls|puts|"
    r"crypto(?:currenc(?:y|ies))?|bitcoin|btc|ether(?:eum)?|coins?|tokens?|reits?|securities|security|"
    r"positions?|portfolio|the\s+dip|ipo)"
)
_ASKER = r"(?:i|we|you|investors|people|one|someone(?:\s+like\s+me)?|my\s+\w+)"

_ADVICE_PATTERNS: tuple[re.Pattern[str], ...] = _rx(
    # "should I sell my shares", "should investors invest in ..."
    rf"\bshould\s+{_ASKER}\s+(?:still\s+|just\s+|now\s+|really\s+|also\s+)?(?:invest|short|sell|hold|dump|"
    r"exit|cash\s+out|get\s+out\s+of|get\s+into|pull\s+out|take\s+profits?|add\s+to|double\s+down|load\s+up|"
    r"rebalance|allocate|diversify|average\s+down|move\s+my|put\s+my)\b",
    # "is NVDA a good investment", "is now a good time to buy"
    r"\bis\s+(?:it|now|this|that|[^\s?.!]+(?:\s+[^\s?.!]+){0,3}?)\s+(?:a\s+|an\s+)?(?:good|great|bad|smart|safe|wise|"
    r"solid|sound|risky|terrible|decent)\s+(?:long[- ]term\s+|short[- ]term\s+)?(?:investment|buy|bet|stock|pick|"
    r"time\s+to\s+(?:buy|sell|invest|get\s+in))\b"
    r"(?!\s+for\s+(?:schools?|districts?|students?|teachers?|universities|colleges|states?|cities|governments?|"
    r"taxpayers|hospitals?|the\s+(?:state|city|country|district|government|school)))",
    r"\b(?:good|bad|smart|wise)\s+idea\s+to\s+(?:buy|sell|invest|short|hold)\b",
    r"\bworth\s+(?:buying|investing\s+in|holding|selling|shorting)\b",
    # "what stocks should I buy", "which AI stocks to buy"
    r"\b(?:what|which)\s+(?:[\w$&.'-]+\s+){0,3}?(?:stocks?|shares|etfs?|funds?|bonds?|cryptos?|"
    r"cryptocurrenc(?:y|ies)|coins?|tokens?|companies|startups?|sectors?|options)\s+(?:should\s+(?:i|we|you|"
    r"investors|one|people|someone|my)|would\s+you\s+(?:buy|recommend|pick)|do\s+you\s+(?:recommend|like|"
    r"suggest)|to\s+(?:buy|invest|pick|own|hold|short|sell)|(?:is|are)\s+(?:the\s+)?(?:best|safest|good|worth\s+"
    r"(?:buying|investing|owning))|will\s+(?:go\s+up|rise|soar|double|outperform))",
    # "best stocks to buy now", "top AI ETFs for 2027"
    r"\b(?:best|top|safest|hottest)\s+(?:[\w$&.-]+\s+){0,2}?(?:stocks?|etfs?|funds?|cryptos?|coins?|tokens?|"
    r"investments?|shares|picks)\s+(?:to\s+(?:buy|own|hold|invest)|for\s+(?:me|my|beginners|retirement|20\d\d|"
    r"the\s+(?:long|short)\s+term|this\s+year|next\s+year)|(?:right\s+)?now)\b",
    # "how should I invest my 401k", "how do I invest in AI"
    r"\bhow\s+(?:should|do|can|would|could)\s+(?:i|we)\s+(?:best\s+)?(?:invest|allocate|diversify|rebalance|"
    r"hedge|position\s+(?:my|our))\b",
    r"\bwhere\s+(?:should|do|can|could)\s+(?:i|we)\s+(?:invest|put|park|keep|move)\s+(?:my|our)\s+(?:money|"
    r"savings|cash|401\s*\(?k\)?|ira|retirement|funds|portfolio|capital)\b",
    r"\bhow\s+much\s+(?:money\s+|of\s+my\s+\w+\s+)?should\s+(?:i|we)\s+(?:invest|put|allocate|buy|bet|keep)\b",
    r"\bwhen\s+should\s+(?:i|we)\s+(?:buy|sell|invest|get\s+(?:in|out)|cash\s+out|take\s+profits?)\b",
    r"\bis\s+(?:it|now)\s+(?:the\s+)?(?:right\s+)?time\s+to\s+(?:buy|sell|invest|get\s+(?:in|out)|cash\s+out|"
    r"take\s+profits?)\b",
    r"\btoo\s+late\s+to\s+(?:buy|sell|invest|get\s+in)\b",
    # rating requests: "buy or sell?", "is NVDA a buy?"
    r"\b(?:buy|sell|hold)\s+or\s+(?:buy|sell|hold)\b",
    r"\bis\s+(?:[^\s?.!]+\s+){1,3}?an?\s+(?:strong\s+)?(?:buy|sell|hold)(?![-\w])",
    # price predictions: "will Nvidia stock go up?"
    r"\bwill\s+(?:[^\s?.!]+\s+){0,3}?(?:stocks?|shares|share\s+price|stock\s+price|bitcoin|btc|crypto|"
    r"ether(?:eum)?|tokens?)\s+(?:go\s+(?:up|down)|rise|fall|rebound|recover|crash|soar|double|triple|"
    r"keep\s+(?:rising|falling|going\s+up)|hit\s+\$?\d|reach\s+\$?\d|bounce\s+back)\b",
    # recommendation requests: "recommend some ETFs", "stock tips"
    r"\b(?:recommend|suggest)\s+(?:me\s+)?(?:(?:a|some|any|good|the\s+best)\s+)*(?:stocks?|etfs?|funds?|"
    r"investments?|cryptos?|coins?|shares|portfolio)\b",
    r"\b(?:stock|crypto|investment)\s+(?:picks?|tips?|recommendations?|ideas)\b",
)

# "should I buy X": advice unless X is plainly a consumer purchase ("a new laptop").
_SHOULD_BUY = re.compile(
    r"\bshould\s+(?:i|we|you|investors|people|one|someone)\s+(?:still\s+|just\s+|now\s+|really\s+)?"
    r"(?:buy|purchase|pick\s+up)\b(?P<rest>[^?.!]*)",
    _I,
)
_CONSUMER_OBJECT = re.compile(r"^\s*(?:a|an|the\s+new|a\s+new|some\s+new|one\s+of\s+the\s+new)\b", _I)

_MY_ACCOUNT = re.compile(
    r"\b(?:my|our)\s+(?:401\s*\(?k\)?|403\s*\(?b\)?|ira|roth(?:\s+ira)?|portfolio|retirement(?:\s+(?:savings|"
    r"account|fund|money))?|brokerage(?:\s+account)?|savings|pension|nest\s+egg|investments?|stocks?|shares|"
    r"money)\b",
    _I,
)
_ADVICE_CUE = re.compile(
    r"\b(?:should|recommend|advi[cs]e|best\s+(?:way|move|option)|what\s+(?:to|do\s+i|should\s+i)\s+do|"
    r"allocate|rebalance|switch|invest)\b",
    _I,
)

# Price-target requests ("price target for NVDA") versus concept questions ("what is a price target?").
_PRICE_TARGET_REQUEST = (
    re.compile(r"\b(?:price\s+targets?|target\s+prices?)\s+(?:for|on|of)\s+(?!(?:a|an|any)\b)\S", _I),
    re.compile(
        r"\b(?:your|the\s+current|the\s+latest|analysts['’]?|its|their)\s+(?:\w+\s+)?(?:price\s+target|"
        r"target\s+price)",
        _I,
    ),
    re.compile(r"\bwhat['’]?s?\s+(?:is\s+)?(?:the|your)\s+(?:\w+\s+)?(?:price\s+target|target\s+price)", _I),
    # Tickers are upper case, so this pattern is deliberately case-sensitive.
    re.compile(r"(?:\$[A-Za-z]{1,5}|\b[A-Z]{2,5})(?:['’]s)?\s+(?:price\s+target|target\s+price)"),
)
_CONCEPT_QUESTION = re.compile(
    r"^\s*(?:what\s+(?:is|are|does|do)\s+(?:a|an)\b|how\s+(?:are|is|do|does)\b.*\b(?:set|calculated|determined|"
    r"work|made|derived)\b|what\s+does\b.*\bmean\b|(?:please\s+)?(?:explain|define)\b|definition\s+of)",
    _I,
)


def detect_advice_request(question: str | None) -> bool:
    """True when a reader asks for personalized investment advice, a rating, a price target, or a price call.

    Concept questions ("what is a price target?", "how do analysts set price
    targets?") and news questions ("how did the SEC rule affect brokers?") are
    not advice requests and get a normal sourced answer.
    """
    q = normalize_space(question)
    if not q:
        return False
    if any(p.search(q) for p in _ADVICE_PATTERNS):
        return True
    for m in _SHOULD_BUY.finditer(q):
        rest = m.group("rest")
        if not _CONSUMER_OBJECT.match(rest) or re.search(rf"\b{_FIN_NOUN}\b", rest, _I):
            return True
    if _MY_ACCOUNT.search(q) and _ADVICE_CUE.search(q):
        return True
    return not _CONCEPT_QUESTION.search(q) and any(p.search(q) for p in _PRICE_TARGET_REQUEST)


# ---------------------------------------------------------------- performance claims


_PERFORMANCE_TEXT = re.compile(
    r"\b(?:annuali[sz]ed\s+returns?|total\s+returns?|excess\s+returns?|risk[- ]adjusted\s+returns?|"
    r"sharpe\s+ratios?|max(?:imum)?\s+drawdowns?|cagr|alpha\s+of)\b"
    r"|\b(?:returned|gained|lost)\s+[-+]?\d+(?:\.\d+)?\s*(?:%|percent)"
    r"|[-+]?\d+(?:\.\d+)?\s*(?:%|percent)\s+(?:annual(?:ized)?\s+)?(?:returns?|gains?|losses)\b"
    r"|\b(?:outperform(?:ed|s|ing)?|beat|beats|beating|underperform(?:ed|s|ing)?|trailed|lagged)\s+"
    r"(?:(?:a|an|the|its|their)\s+)?(?:[\w/&.-]+\s+){0,3}?(?:portfolio|index|s&p(?:\s*500)?|fund|60/40|"
    r"market(?!['’]s|\s+(?:expectations|estimates|consensus|forecasts?)))\b",
    _I,
)

_VALID_LABELS = frozenset(label.value for label in PerformanceLabel)

# What a reader sees in the claim text itself also counts as labelling.
_TEXT_LABEL = re.compile(
    r"\b(?:back-?test(?:s|ed|ing)?|simulat(?:ed|ion|ions)|hypothetical|paper[- ]trading|"
    r"live\s+(?:track\s+record|trading|results?|returns?|performance|deployment|portfolio)|real\s+money)\b",
    _I,
)
_TEXT_PERIOD = re.compile(
    r"\b(?:19|20)\d{2}\b|\b\d+[- ](?:year|month|quarter|week|day)s?\b|"
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty)[- ](?:year|month|quarter|decade)s?\b|"
    r"\b(?:a|one|two|three)?[- ]?decades?\b|\bQ[1-4]\b|\bfiscal\b|\b(?:past|last)\s+\d*\s*(?:years?|months?|quarters?)\b",
    _I,
)
_TEXT_BENCHMARK = re.compile(
    r"\b(?:benchmark|60/40|s&p(?:\s*500)?|index|indices|msci|nasdaq|dow|russell|ftse|treasur(?:y|ies)|"
    r"buy[- ]and[- ]hold|risk[- ]free\s+rate)\b|"
    r"\b(?:stock|bond|stock-and-bond|balanced|passive|equal[- ]weight(?:ed)?|benchmark)\s+portfolio\b",
    _I,
)


def looks_like_performance(text: str | None) -> bool:
    """True when ``text`` describes investment or strategy performance (returns, beating a benchmark...)."""
    return bool(_PERFORMANCE_TEXT.search(text or ""))


def performance_claim_issues(claim: BriefClaim | ClaimOut) -> list[str]:
    """Problems with a performance claim's labelling (a ``BriefClaim`` or ``ClaimOut``).

    A claim counts as a performance claim when it is typed ``performance``,
    carries a performance label, or its text describes investment/strategy
    performance. Such claims must be labelled backtest, simulated, or live and
    state the period, the benchmark, and at least one source. The structured
    fields (``performance_label``, ``performance_period``, ``benchmark``) are
    preferred; wording a reader can see in the claim itself ("in a 2005-2024
    backtest against a 60/40 portfolio") also satisfies the requirement.
    """
    text = claim.text or ""
    typed = str(claim.claim_type or "") == ClaimType.PERFORMANCE
    label = (claim.performance_label or "").strip().lower()
    if not (typed or label or looks_like_performance(text)):
        return []
    issues: list[str] = []
    if label and label not in _VALID_LABELS:
        issues.append(f"Performance label '{claim.performance_label}' is not one of backtest, simulated, or live.")
    elif not label and not _TEXT_LABEL.search(text):
        why = "" if typed else " (the text describes investment or strategy performance)"
        issues.append(f"Not labelled backtest, simulated, or live{why}.")
    if not normalize_space(claim.performance_period) and not _TEXT_PERIOD.search(text):
        issues.append("Does not state the period the result covers.")
    if not normalize_space(claim.benchmark) and not _TEXT_BENCHMARK.search(text):
        issues.append("Does not name the benchmark the result is compared with.")
    if not list(claim.citation_ids or []):
        issues.append("Cites no source for the result.")
    return issues
