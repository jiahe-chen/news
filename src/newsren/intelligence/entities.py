"""Entity extraction and tagging for Story Intelligence (Connect).

Heuristic, dependency-free named-entity extraction tuned for technology,
education, industry, and finance news:

* a gazetteer of well-known organizations, regulators, research indexes,
  products, policies, and places, with aliases
  ("Securities and Exchange Commission" -> "SEC", "Nvidia Corp." -> "NVIDIA");
* stock tickers ("$ABCD", "NASDAQ: ABCD", "(NYSE: XYZ)");
* model/product names with version numbers ("GPT-5", "Llama 4", "Atlas-3");
* capitalized multi-word sequences ("Nimbus Labs", "Department of Education"),
  skipping sentence-initial common words, months, and days;
* acronyms ("CISA", "OEI"), minus generic ones ("AI", "CEO", "GPU").

Kinds (organization | person | product | policy | place | ticker) come from
simple cues: organizational suffixes (Inc, Corp, Labs, University, Bank,
Agency ...), policy words (Act, Rule, Bill, Regulation ...), versioned
product names, honorifics/job titles and speech verbs for people.

Headlines written in Title Case are handled conservatively: names already
found in the body are matched, and otherwise only sequences with an explicit
cue (suffix, gazetteer, version number) are kept.

Names are normalized for dedupe (``normalize_entity``): case, punctuation,
possessives, a leading "the", and legal suffixes are ignored, and gazetteer
aliases map to one canonical name.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace

from sqlalchemy import select
from sqlalchemy.orm import Session

from newsren.models import Article, ArticleEntity, Entity
from newsren.textutil import STOPWORDS, normalize_space

# --------------------------------------------------------------------------
# Gazetteer: (canonical name, kind, aliases as they appear in text)
# --------------------------------------------------------------------------

ORG, PERSON, PRODUCT, POLICY, PLACE, TICKER = (
    "organization",
    "person",
    "product",
    "policy",
    "place",
    "ticker",
)

GAZETTEER: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    # Regulators, central banks, government bodies
    ("SEC", ORG, ("SEC", "Securities and Exchange Commission", "U.S. Securities and Exchange Commission")),
    ("FINRA", ORG, ("FINRA", "Financial Industry Regulatory Authority")),
    ("Federal Reserve", ORG, ("Federal Reserve", "Federal Reserve Board", "Federal Reserve System", "Fed")),
    ("FOMC", ORG, ("FOMC", "Federal Open Market Committee")),
    ("CFTC", ORG, ("CFTC", "Commodity Futures Trading Commission")),
    ("FTC", ORG, ("FTC", "Federal Trade Commission")),
    ("CFPB", ORG, ("CFPB", "Consumer Financial Protection Bureau")),
    ("OCC", ORG, ("OCC", "Office of the Comptroller of the Currency")),
    ("FDIC", ORG, ("FDIC", "Federal Deposit Insurance Corporation")),
    ("U.S. Treasury", ORG, ("U.S. Treasury", "Treasury Department", "Department of the Treasury")),
    ("Department of Justice", ORG, ("Department of Justice", "Justice Department", "DOJ")),
    ("Department of Commerce", ORG, ("Department of Commerce", "Commerce Department")),
    (
        "Department of Education",
        ORG,
        ("Department of Education", "U.S. Department of Education", "Education Department"),
    ),
    ("Institute of Education Sciences", ORG, ("Institute of Education Sciences", "IES")),
    ("ERIC", ORG, ("ERIC", "Education Resources Information Center")),
    ("NCES", ORG, ("NCES", "National Center for Education Statistics")),
    ("NIST", ORG, ("NIST", "National Institute of Standards and Technology")),
    ("National Science Foundation", ORG, ("National Science Foundation", "NSF")),
    ("CISA", ORG, ("CISA", "Cybersecurity and Infrastructure Security Agency")),
    ("European Commission", ORG, ("European Commission",)),
    ("European Central Bank", ORG, ("European Central Bank", "ECB")),
    ("Bank of England", ORG, ("Bank of England",)),
    ("Bank of Japan", ORG, ("Bank of Japan", "BOJ")),
    ("IMF", ORG, ("IMF", "International Monetary Fund")),
    ("World Bank", ORG, ("World Bank",)),
    ("OECD", ORG, ("OECD",)),
    ("UNESCO", ORG, ("UNESCO",)),
    ("United Nations", ORG, ("United Nations", "UN")),
    # AI and technology companies
    ("OpenAI", ORG, ("OpenAI",)),
    ("Anthropic", ORG, ("Anthropic",)),
    ("Google DeepMind", ORG, ("Google DeepMind", "DeepMind")),
    ("Google Cloud", ORG, ("Google Cloud",)),
    ("Google", ORG, ("Google",)),
    ("Alphabet", ORG, ("Alphabet",)),
    ("Microsoft", ORG, ("Microsoft",)),
    ("Meta", ORG, ("Meta Platforms", "Meta")),
    ("Apple", ORG, ("Apple",)),
    ("Amazon Web Services", ORG, ("Amazon Web Services", "AWS")),
    ("Amazon", ORG, ("Amazon",)),
    ("NVIDIA", ORG, ("NVIDIA", "Nvidia")),
    ("AMD", ORG, ("AMD", "Advanced Micro Devices")),
    ("Intel", ORG, ("Intel",)),
    (
        "TSMC",
        ORG,
        (
            "TSMC",
            "Taiwan Semiconductor Manufacturing Company",
            "Taiwan Semiconductor Manufacturing",
            "Taiwan Semiconductor",
        ),
    ),
    ("Samsung", ORG, ("Samsung Electronics", "Samsung")),
    ("SK Hynix", ORG, ("SK Hynix", "SK hynix")),
    ("Micron", ORG, ("Micron Technology", "Micron")),
    ("ASML", ORG, ("ASML",)),
    ("Arm Holdings", ORG, ("Arm Holdings",)),
    ("Qualcomm", ORG, ("Qualcomm",)),
    ("Broadcom", ORG, ("Broadcom",)),
    ("IBM", ORG, ("IBM",)),
    ("Oracle", ORG, ("Oracle",)),
    ("Salesforce", ORG, ("Salesforce",)),
    ("Mistral AI", ORG, ("Mistral AI",)),
    ("xAI", ORG, ("xAI",)),
    ("Hugging Face", ORG, ("Hugging Face",)),
    ("Baidu", ORG, ("Baidu",)),
    ("Alibaba", ORG, ("Alibaba",)),
    ("Tencent", ORG, ("Tencent",)),
    ("Huawei", ORG, ("Huawei",)),
    ("ByteDance", ORG, ("ByteDance",)),
    ("Tesla", ORG, ("Tesla",)),
    ("Cisco", ORG, ("Cisco",)),
    ("Palantir", ORG, ("Palantir",)),
    ("Databricks", ORG, ("Databricks",)),
    ("GitHub", ORG, ("GitHub",)),
    # Education organizations
    ("Khan Academy", ORG, ("Khan Academy",)),
    ("Coursera", ORG, ("Coursera",)),
    ("Pearson", ORG, ("Pearson",)),
    ("Chegg", ORG, ("Chegg",)),
    ("Duolingo", ORG, ("Duolingo",)),
    ("College Board", ORG, ("College Board",)),
    ("MIT", ORG, ("MIT", "Massachusetts Institute of Technology")),
    ("Stanford University", ORG, ("Stanford University", "Stanford")),
    ("Harvard University", ORG, ("Harvard University", "Harvard")),
    ("Carnegie Mellon University", ORG, ("Carnegie Mellon University", "Carnegie Mellon")),
    ("UC Berkeley", ORG, ("UC Berkeley", "University of California, Berkeley")),
    # Finance
    ("JPMorgan", ORG, ("JPMorgan Chase", "JPMorgan", "J.P. Morgan", "JP Morgan")),
    ("Goldman Sachs", ORG, ("Goldman Sachs",)),
    ("Morgan Stanley", ORG, ("Morgan Stanley",)),
    ("Deutsche Bank", ORG, ("Deutsche Bank",)),
    ("Bank of America", ORG, ("Bank of America",)),
    ("Citigroup", ORG, ("Citigroup", "Citi")),
    ("Wells Fargo", ORG, ("Wells Fargo",)),
    ("BlackRock", ORG, ("BlackRock",)),
    ("Vanguard", ORG, ("Vanguard",)),
    ("Fidelity Investments", ORG, ("Fidelity Investments",)),
    ("Charles Schwab", ORG, ("Charles Schwab",)),
    ("UBS", ORG, ("UBS",)),
    ("HSBC", ORG, ("HSBC",)),
    ("Barclays", ORG, ("Barclays",)),
    ("Nasdaq", ORG, ("Nasdaq", "NASDAQ")),
    ("NYSE", ORG, ("NYSE", "New York Stock Exchange")),
    ("Bloomberg", ORG, ("Bloomberg",)),
    ("Thomson Reuters", ORG, ("Thomson Reuters",)),
    ("Reuters", ORG, ("Reuters",)),
    ("The New York Times", ORG, ("The New York Times", "New York Times")),
    ("Financial Times", ORG, ("Financial Times",)),
    ("Sequoia Capital", ORG, ("Sequoia Capital", "Sequoia")),
    ("Andreessen Horowitz", ORG, ("Andreessen Horowitz", "a16z")),
    ("Y Combinator", ORG, ("Y Combinator",)),
    ("SoftBank", ORG, ("SoftBank",)),
    ("Tiger Global", ORG, ("Tiger Global",)),
    ("Accel", ORG, ("Accel",)),
    ("Delphia", ORG, ("Delphia",)),
    ("Global Predictions", ORG, ("Global Predictions",)),
    ("Debevoise & Plimpton", ORG, ("Debevoise & Plimpton", "Debevoise")),
    ("Mercer", ORG, ("Mercer",)),
    ("Deloitte", ORG, ("Deloitte",)),
    ("Natixis", ORG, ("Natixis",)),
    # Research indexes
    ("arXiv", ORG, ("arXiv", "ArXiv", "arxiv.org")),
    ("SSRN", ORG, ("SSRN",)),
    ("NBER", ORG, ("NBER", "National Bureau of Economic Research")),
    # Products
    ("ChatGPT", PRODUCT, ("ChatGPT",)),
    ("Claude", PRODUCT, ("Claude",)),
    ("Gemini", PRODUCT, ("Gemini",)),
    ("Llama", PRODUCT, ("Llama",)),
    ("Copilot", PRODUCT, ("GitHub Copilot", "Microsoft Copilot", "Copilot")),
    ("LLM Suite", PRODUCT, ("LLM Suite",)),
    ("TradingAgents", PRODUCT, ("TradingAgents",)),
    ("Raise Agent", PRODUCT, ("Raise Agent",)),
    ("Khanmigo", PRODUCT, ("Khanmigo",)),
    ("CUDA", PRODUCT, ("CUDA",)),
    ("Azure", PRODUCT, ("Microsoft Azure", "Azure")),
    ("iPhone", PRODUCT, ("iPhone",)),
    ("Android", PRODUCT, ("Android",)),
    # Policies and laws
    ("EU AI Act", POLICY, ("EU AI Act", "AI Act")),
    ("CHIPS and Science Act", POLICY, ("CHIPS and Science Act", "CHIPS Act")),
    ("Regulation Best Interest", POLICY, ("Regulation Best Interest", "Reg BI")),
    ("GDPR", POLICY, ("GDPR", "General Data Protection Regulation")),
    ("FERPA", POLICY, ("FERPA",)),
    ("COPPA", POLICY, ("COPPA",)),
    ("Title IX", POLICY, ("Title IX",)),
    ("Every Student Succeeds Act", POLICY, ("Every Student Succeeds Act", "ESSA")),
    ("Dodd-Frank Act", POLICY, ("Dodd-Frank Act", "Dodd-Frank")),
    ("Sarbanes-Oxley Act", POLICY, ("Sarbanes-Oxley Act", "Sarbanes-Oxley")),
    # Places
    ("United States", PLACE, ("United States", "U.S.", "USA", "US")),
    ("United Kingdom", PLACE, ("United Kingdom", "U.K.", "UK", "Britain")),
    ("European Union", PLACE, ("European Union", "EU")),
    ("Europe", PLACE, ("Europe",)),
    ("China", PLACE, ("China",)),
    ("Taiwan", PLACE, ("Taiwan",)),
    ("Japan", PLACE, ("Japan",)),
    ("South Korea", PLACE, ("South Korea",)),
    ("India", PLACE, ("India",)),
    ("Germany", PLACE, ("Germany",)),
    ("France", PLACE, ("France",)),
    ("Canada", PLACE, ("Canada",)),
    ("Mexico", PLACE, ("Mexico",)),
    ("Singapore", PLACE, ("Singapore",)),
    ("Israel", PLACE, ("Israel",)),
    ("Netherlands", PLACE, ("Netherlands",)),
    ("Arizona", PLACE, ("Arizona",)),
    ("Texas", PLACE, ("Texas",)),
    ("California", PLACE, ("California",)),
    ("Ohio", PLACE, ("Ohio",)),
    ("New York", PLACE, ("New York City", "New York")),
    ("New Jersey", PLACE, ("New Jersey",)),
    ("New Zealand", PLACE, ("New Zealand",)),
    ("Washington", PLACE, ("Washington",)),
    ("Silicon Valley", PLACE, ("Silicon Valley",)),
    ("Wall Street", PLACE, ("Wall Street",)),
    ("Beijing", PLACE, ("Beijing",)),
    ("Shanghai", PLACE, ("Shanghai",)),
    ("Hong Kong", PLACE, ("Hong Kong",)),
    ("Tokyo", PLACE, ("Tokyo",)),
    ("Seoul", PLACE, ("Seoul",)),
    ("Taipei", PLACE, ("Taipei",)),
    ("Hsinchu", PLACE, ("Hsinchu",)),
    ("London", PLACE, ("London",)),
    ("Brussels", PLACE, ("Brussels",)),
    ("Frankfurt", PLACE, ("Frankfurt",)),
    ("Paris", PLACE, ("Paris",)),
    ("Berlin", PLACE, ("Berlin",)),
)

# --------------------------------------------------------------------------
# Cue vocabularies
# --------------------------------------------------------------------------

LEGAL_SUFFIXES = frozenset(
    "inc incorporated corp corporation co company ltd limited llc plc sa ag nv se gmbh holdings group".split()
)

ORG_WORDS = frozenset(
    """
    Inc Inc. Corp Corporation Co Company Ltd LLC PLC Group Holdings Labs Lab Laboratories Laboratory
    University College Institute Academy School District Schools Board Bank Bancorp Agency Commission
    Department Ministry Bureau Authority Council Committee Office Administration Association Society
    Foundation Federation Center Centre Exchange Fund Capital Partners Ventures Investments Securities
    Advisors Advisers Financial Insurance Trust Technologies Technology Systems Software Solutions Networks
    Devices Instruments Industries Analytics Dynamics Aerospace Biosciences Therapeutics Pharmaceuticals
    Robotics Semiconductor Semiconductors Electronics Computing Materials Energy Motors Media Studios
    Research AI Cloud Services Consortium Coalition Alliance Union Reserve Court Senate Congress Parliament
    Hospital Press
    """.split()
)

# Words after which "Nimbus Labs Atlas-3" splits into "Nimbus Labs" + "Atlas-3".
_SPLITTING_ORG_WORDS = frozenset(
    "Inc Corp Corporation Ltd LLC PLC Labs Lab Laboratories Group Holdings Technologies Systems".split()
)

ORG_HEAD_WORDS = frozenset(
    "University Department Ministry Office Bureau Board Bank Institute College Academy Commission".split()
)

POLICY_SUFFIX = frozenset(
    """
    Act Acts Rule Rules Bill Regulation Regulations Directive Law Laws Order Amendment Amendments Policy
    Framework Guidance Standard Standards Mandate Statute Treaty Accord Code Plan
    """.split()
)
POLICY_PREFIX = frozenset({"Title", "Regulation", "Executive", "Rule"})

PRODUCT_SUFFIX = frozenset(
    """
    Model Models Chip Chips GPU Processor Accelerator Suite Agent Agents Copilot Assistant App Tutor
    Platform Phone OS API SDK Engine Headset Glasses Browser Chatbot
    """.split()
)
MODEL_TIER_WORDS = frozenset("Pro Ultra Max Mini Turbo Flash Nano Plus Opus Sonnet Haiku Large Small Lite".split())

PLACE_SUFFIX = frozenset("City County Province Valley Island Islands Republic Kingdom Peninsula".split())

# Job titles and honorifics: stripped from the front of a name and a cue that a person follows.
TITLE_WORDS = frozenset(
    """
    CEO CFO CTO COO Chief Executive President Chair Chairman Chairwoman Chairperson Secretary Director
    Professor Prof Dr Mr Ms Mrs Senator Sen Rep Representative Governor Gov Minister Commissioner
    Superintendent Founder Co-founder Cofounder Analyst Researcher Economist Spokesperson Spokesman
    Spokeswoman Judge Mayor Dean Provost Vice Deputy
    """.split()
)

ROLE_WORDS = frozenset(
    """
    analyst analysts researcher researchers economist professor executive officer spokesperson spokesman
    spokeswoman director founder cofounder co-founder president chair chairman chairwoman superintendent
    principal teacher student commissioner secretary author columnist reporter editor scientist engineer
    partner investor lawyer attorney strategist manager said says told according
    """.split()
)

CONNECTORS = frozenset({"of", "for", "and", "on", "de", "du", "van", "von", "der", "da", "la", "le"})

MONTHS_DAYS = frozenset(
    """
    January February March April May June July August September October November December Jan Feb Mar Apr
    Jun Jul Aug Sep Sept Oct Nov Dec Monday Tuesday Wednesday Thursday Friday Saturday Sunday Mon Tue Tues
    Wed Thu Thur Thurs Fri Sat Sun Today Tonight Yesterday Tomorrow
    """.split()
)

# Capitalized words that open sentences without naming anything.
COMMON_INITIAL = frozenset(
    """
    New After Before Despite While When Although Though Since Because As At In On For With Without Under
    Over During Following According Amid Among Across Earlier Later Last Next This That These Those Its
    Their Some Many Most Several Few All Both Each Every Other Another Such Officials Researchers Analysts
    Students Teachers Educators Investors Regulators Schools Shares Stocks Companies Lawmakers Critics
    Experts Parents Users Customers Developers Banks Firms Startups However Meanwhile Still Separately Now
    Here There Why How What Who Where Yes No Update Updated Breaking Exclusive Opinion Analysis Correction
    Clarification Editorial Commentary Overall Instead Indeed Also Then Further Finally First Second Third
    Already Even Only Just Nearly About More Less Read See Watch Note Please Click Sign Subscribe Unlike
    Like If Once Until Unless Whether Given Asked Speaking Writing Looking Based Using Under Upon Within
    Together Similarly Likewise Additionally Moreover Notably Importantly Specifically Currently Recently
    Previously Today's Our We You They He She It Its I Startup Startups Chipmaker Chipmakers Automaker
    Automakers Lender Lenders Insurer Retailer Brokerage Firm Company Regulator District Districts State
    States Officials Executives Agency Analyst Investor Teacher Student Parent Lawmaker Rival Rivals
    """.split()
)

# Words that, on their own or only together, are topics rather than names.
GENERIC_WORDS = frozenset(
    """
    artificial intelligence machine learning deep generative large language model models data technology tech
    education finance financial industry industrial science research report reports study studies students
    student schools school teachers teacher high middle elementary math mathematics reading writing
    computer computing chip chips semiconductor semiconductors market markets stock stocks bank banks
    policy policies regulation regulations rule rules law laws government federal state states public
    private national international global local university universities college colleges internet web
    online digital cloud software hardware network security cyber cybersecurity privacy health climate
    energy economy economic inflation interest rate rates earnings revenue profit quarter quarterly annual
    fiscal ai ml llm llms agent agents agentic chief executive officer officers vice president senior sharpe
    regulatory notice notices algebra geometry calculus biology chemistry physics specialty
    managing director big small new old early late best top latest first final big tech news update
    breaking exclusive opinion analysis commentary editorial correction press release statement summary
    overview introduction conclusion abstract background methods results discussion limitations figure
    table appendix section part chapter series round stage phase grade page item form exhibit north south
    east west central eastern western northern southern american chinese european japanese korean
    taiwanese german french british indian canadian mexican russian israeli asian african english
    spanish dutch swiss italian australian
    """.split()
)

EXCLUDED_PREFIXES = frozenset(
    """
    Series Form Item Section Figure Table Phase Stage Grade Round Chapter Page Exhibit Appendix Part Version
    Versions Release Build Patch Tracking
    """.split()
)

GENERIC_ACRONYMS = frozenset(
    """
    AI AGI ML LLM LLMS NLP GPU GPUS CPU CPUS TPU TPUS NPU API APIS SDK CEO CEOS CFO CTO COO CIO CISO VP SVP
    EVP IPO IPOS ETF ETFS EPS GAAP ROI KPI KPIS YOY QOQ USD EUR GBP JPY CNY ESG STEM MBA OK TV PC PCS PDF FAQ
    HTML URL AM PM ET PT GMT UTC EST PST EDT PDT CET FY Q1 Q2 Q3 Q4 H1 H2 R&D B2B B2C SAAS IT HR PR II III
    IV VI VII VIII IX XI XII AML KYC GDP CPI PCE SAT ACT AP ID IOT AR VR XR EV EVS OS RAM SSD HBM DRAM NAND
    RL RAG ASIC FPGA SOC NEW THE AND FOR LLC INC CORP LTD PLC USA DC NYC LA SF PHD BA BS MA MS JD MD RN K12
    CO2 CEO's FAQS AKA TBD TBA FYI ASAP DIY EOD PDT CVE CVES SEO UI UX QA CI CD OSS VPN VPNS
    """.split()
)

HEADLINE_BREAK = frozenset(
    """
    launch launches launched unveil unveils unveiled announce announces announced say says said raise raises
    raised report reports reported release releases released add adds win wins get gets plan plans open
    opens hit hits beat beats miss misses cut cuts sue sues sued file files filed join joins buy buys
    sell sells acquire acquires partner partners expand expands introduce introduces debut debuts sign signs
    face faces drop drops rise rises fall falls top tops lead leads back backs warn warns charge charges
    fine fines settle settles approve approves propose proposes adopt adopts ban bans delay delays pause
    pauses test tests find finds show shows issue issues publish publishes roll rolls take takes make
    makes go goes see sees claim claims dispute disputes question questions correct corrects retract
    retracts outperform outperforms beats tops new first latest after amid over into from as to with
    without against how why what when who will could should may might can is are was were be its their
    vs versus study report model models chip chips math students schools school teachers update opinion
    analysis correction exclusive breaking deal deals plan amid ahead behind gains slips surges jumps
    """.split()
)

FIRST_NAMES = frozenset(
    """
    James John Robert Michael William David Richard Joseph Thomas Charles Christopher Daniel Matthew Anthony
    Mark Donald Steven Paul Andrew Joshua Kenneth Kevin Brian George Timothy Ronald Edward Jason Jeffrey Ryan
    Jacob Gary Nicholas Eric Jonathan Stephen Larry Justin Scott Brandon Benjamin Samuel Gregory Alexander
    Frank Patrick Raymond Jack Dennis Jerry Tyler Aaron Jose Adam Nathan Henry Douglas Zachary Peter Kyle
    Noah Ethan Jeremy Walter Christian Keith Roger Terry Austin Sean Gerald Carl Harold Dylan Arthur Lawrence
    Jordan Jesse Bryan Billy Bruce Gabriel Joe Logan Alan Juan Albert Willie Elijah Wayne Randy Vincent
    Mary Patricia Jennifer Linda Elizabeth Barbara Susan Jessica Sarah Karen Lisa Nancy Betty Sandra Margaret
    Ashley Kimberly Emily Donna Michelle Carol Amanda Melissa Deborah Stephanie Dorothy Rebecca Sharon Laura
    Cynthia Amy Kathleen Angela Shirley Brenda Emma Anna Pamela Nicole Samantha Katherine Christine Helen
    Debra Rachel Carolyn Janet Maria Catherine Heather Diane Olivia Julie Joyce Victoria Ruth Virginia Lauren
    Kelly Christina Joan Evelyn Judith Andrea Hannah Megan Cheryl Jacqueline Martha Madison Teresa Gloria
    Sara Janice Ann Kathryn Abigail Sophia Frances Jean Alice Judy Isabella Julia Grace Amber Denise Danielle
    Marilyn Beverly Charlotte Natalie Theresa Diana Brittany Doris Kayla Alexis Lori Marie Priya Raj Wei Li
    Ming Jun Hiroshi Yuki Kenji Akira Sanjay Anil Arjun Ravi Deepa Aisha Fatima Omar Ahmed Mohamed Mohammed
    Ali Hassan Carlos Luis Miguel Javier Sofia Lucia Elena Ana Mateo Diego Pablo Lars Sven Ingrid Hans
    Klaus Pierre Jean-Pierre Marc Claire Francois Giulia Marco Luca Chen Min Ji-woo Seo-yeon Jensen Satya
    Sundar Sam Dario Demis Lisa Jamie Jane Gary Mira Elon Tim Jerome Janet Gary Christine Lael Michelle
    """.split()
)

_ABBREVIATIONS = frozenset(
    "Dr Mr Ms Mrs Prof St Inc Corp Co Ltd Jr Sr vs No Gov Sen Rep Gen Lt Col Mt U.S U.K e.g i.e etc approx".split()
)

# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def normalize_entity(name: str) -> str:
    """Normalized key used to dedupe entity names.

    Lower-cases, drops periods, possessives, a leading "the", and trailing
    legal suffixes ("Inc", "Corp", "Group" ...); "&" counts as "and".
    """
    s = normalize_space(name)
    s = re.sub(r"[’']s\b", "", s)
    s = s.replace("&", " and ").lower().replace(".", "")
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    words = s.split()
    if len(words) > 1 and words[0] == "the":
        words = words[1:]
    while len(words) > 1 and words[-1] in LEGAL_SUFFIXES:
        words.pop()
    return " ".join(words)[:200]


ALIAS_INDEX: dict[str, tuple[str, str]] = {}
for _canonical, _kind, _aliases in GAZETTEER:
    for _alias in (*_aliases, _canonical):
        ALIAS_INDEX.setdefault(normalize_entity(_alias), (_canonical, _kind))


def canonical_entity(name: str) -> tuple[str, str] | None:
    """(canonical name, kind) if ``name`` is a known gazetteer entity or alias."""
    return ALIAS_INDEX.get(normalize_entity(name))


# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

_ALIAS_TO_CANONICAL: dict[str, tuple[str, str]] = {}
for _canonical, _kind, _aliases in GAZETTEER:
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL.setdefault(_alias, (_canonical, _kind))

_GAZETTEER_RE = re.compile(
    r"(?<![\w\-])(?:"
    + "|".join(re.escape(a) for a in sorted(_ALIAS_TO_CANONICAL, key=len, reverse=True))
    + r")(?![\w\-]|\.\w)"
)

_EXCHANGES = r"NYSE American|NYSE Arca|NYSE|NASDAQ|Nasdaq|AMEX|TSX|LSE|HKEX|TWSE|KRX|OTCQX|OTC"
_TICKER_EXCHANGE_RE = re.compile(rf"\b(?:{_EXCHANGES})\s*:\s*([A-Z]{{1,5}}(?:\.[A-Z])?)\b")
_TICKER_DOLLAR_RE = re.compile(r"(?<![\w$])\$([A-Z]{1,5}(?:\.[A-Z])?)\b")

# Well-known model families followed by a version or tier ("GPT-5", "Llama 4", "Claude Opus 5").
_MODEL_FAMILY_RE = re.compile(
    r"(?<![\w\-])(?:GPT|Llama|Gemini|Claude|Mistral|Qwen|DeepSeek|Phi|Gemma|Grok)"
    r"(?:(?:[\s\-](?:\d+(?:\.\d+)*[a-z]?|o\d*)\b)|(?:\s(?:Opus|Sonnet|Haiku|Pro|Ultra|Flash|Nano|Mini|Turbo))+)"
    r"(?:\s(?:\d+(?:\.\d+)*|Opus|Sonnet|Haiku|Pro|Ultra|Flash|Nano|Mini|Turbo)\b)*"
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*(?:[.&][A-Za-z0-9][A-Za-z0-9'’\-]*)*")
_VERSIONED_RE = re.compile(r"^[A-Z][A-Za-z]*-?\d+(?:\.\d+)*[A-Za-z]*$")
_NOT_PRODUCT_RE = re.compile(
    r"^(?:Q[1-4]|FY\d+|H[12]|K-12|COVID-19|[SF]-\d+|\d+-[KQ]|CVE-.*|G\d+|Rule\d*|Form\d*)$"
)
_VERSION_TOKEN_RE = re.compile(r"^\d+(?:\.\d+)*[a-z]?$")
_ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9&]{1,6}s?$")
_NAME_WORD_RE = re.compile(r"^[A-Z][a-z]+(?:[-'’][A-Z]?[a-z]+)*$")


# --------------------------------------------------------------------------
# Extraction internals
# --------------------------------------------------------------------------

_CONFIDENCE_GAZETTEER = 3
_CONFIDENCE_CUE = 2
_CONFIDENCE_DEFAULT = 1


@dataclass(frozen=True)
class Mention:
    """One entity mention: display name, kind, normalized key, character offset."""

    name: str
    kind: str
    key: str
    start: int
    confidence: int = _CONFIDENCE_DEFAULT


@dataclass
class _Tok:
    word: str
    start: int
    end: int
    gap: str  # raw text between the previous token and this one
    possessive: bool


def _mask(text: str, start: int, end: int) -> str:
    # Same-length mask of non-space characters so later scans see a hard break.
    return text[:start] + "|" * (end - start) + text[end:]


def _tokens(text: str) -> list[_Tok]:
    out: list[_Tok] = []
    prev_end = 0
    for m in _TOKEN_RE.finditer(text):
        raw = m.group()
        word = raw
        possessive = False
        if word.endswith(("'s", "’s")):
            word, possessive = word[:-2], True
        elif word.endswith(("'", "’")):
            word, possessive = word[:-1], True
        word = word.rstrip("-")
        if word:
            out.append(_Tok(word, m.start(), m.end(), text[prev_end : m.start()], possessive))
        prev_end = m.end()
    return out


def _starts_sentence(toks: list[_Tok], i: int) -> bool:
    if i == 0:
        return True
    gap = toks[i].gap
    if "\n" in gap:
        return True
    if re.search(r"[.!?:;]", gap):
        prev = toks[i - 1].word
        return not (prev in _ABBREVIATIONS and gap.strip() == ".")
    return False


def _is_cap(word: str) -> bool:
    if not word or word.isdigit():
        return False
    if word[0].isupper():
        return True
    # camelCase product names such as "iPhone" or "eSIM".
    return word[0].islower() and len(word) > 2 and word[1].isupper()


def _is_acronym(word: str) -> bool:
    if not _ACRONYM_RE.match(word):
        return False
    upper = word.upper()
    base = upper[:-1] if word.endswith("s") else upper
    if upper in GENERIC_ACRONYMS or base in GENERIC_ACRONYMS:
        return False
    if word.lower() in STOPWORDS:
        return False
    return sum(c.isupper() for c in word) >= 2


def _is_versioned(word: str) -> bool:
    return bool(_VERSIONED_RE.match(word)) and not _NOT_PRODUCT_RE.match(word) and any(c.isdigit() for c in word)


def _is_title_case(toks: list[_Tok]) -> bool:
    content = [t.word for t in toks if t.word.isalpha() and len(t.word) >= 4 and t.word.lower() not in STOPWORDS]
    if len(content) < 3:
        return False
    return all(w[0].isupper() for w in content)


def _sentences(toks: list[_Tok]) -> list[list[_Tok]]:
    sentences: list[list[_Tok]] = []
    for i, tok in enumerate(toks):
        if _starts_sentence(toks, i) or not sentences:
            sentences.append([])
        sentences[-1].append(tok)
    return sentences


def _kind_for(words: list[str], person_cue: bool) -> tuple[str, int]:
    first, last = words[0], words[-1]
    if last in POLICY_SUFFIX or first in POLICY_PREFIX and len(words) > 1:
        return POLICY, _CONFIDENCE_CUE
    versioned = any(_is_versioned(w) for w in words) or (len(words) > 1 and _VERSION_TOKEN_RE.match(last))
    if versioned or last in PRODUCT_SUFFIX or (first[0].islower() and _is_cap(first)):
        return PRODUCT, _CONFIDENCE_CUE
    if any(w in ORG_WORDS for w in words) or first in ORG_HEAD_WORDS:
        return ORG, _CONFIDENCE_CUE
    if last in PLACE_SUFFIX:
        return PLACE, _CONFIDENCE_CUE
    name_like = 2 <= len(words) <= 3 and all(_NAME_WORD_RE.match(w) for w in words)
    # A following "said" is not enough: companies say things too ("Orrin Lithography says").
    if name_like and (person_cue or first in FIRST_NAMES):
        return PERSON, _CONFIDENCE_CUE
    if len(words) == 1 and _is_acronym(first):
        return ORG, _CONFIDENCE_CUE
    return ORG, _CONFIDENCE_DEFAULT


def _split_run(words: list[_Tok]) -> list[list[_Tok]]:
    """Split a raw capitalized run at "and" (unless part of a name) and after org suffixes."""
    parts: list[list[_Tok]] = [[]]
    for i, tok in enumerate(words):
        if tok.word in MONTHS_DAYS:
            parts.append([])
            continue
        if tok.word == "and":
            # Keep "and" inside institutional names ("Commerce and Trade Commission"),
            # split lists of names ("Larchmont Capital and Saltmarsh Partners").
            before = words[i - 1].word if i > 0 else ""
            if words[-1].word not in _AND_HEADS or before in ORG_WORDS:
                parts.append([])
                continue
        parts[-1].append(tok)
        nxt = words[i + 1].word if i + 1 < len(words) else ""
        if (
            tok.word in _SPLITTING_ORG_WORDS
            and nxt
            and nxt not in CONNECTORS
            and nxt not in ORG_WORDS
            and nxt.lower() not in LEGAL_SUFFIXES
        ):
            parts.append([])
    return [p for p in parts if p]


_NUMBER_WORDS = frozenset(
    "three four five six seven eight nine ten eleven twelve dozen dozens hundreds thousands millions".split()
)
_STRIP_WORDS = (STOPWORDS - LEGAL_SUFFIXES - {"new", "first"}) | _NUMBER_WORDS
_AND_HEADS = POLICY_SUFFIX | frozenset(
    """
    Commission Department Administration Agency Bureau Board Committee Council Institute University Exchange
    Laboratory Association Society Foundation Center Centre Authority Office Ministry
    """.split()
)


def _trim(words: list[_Tok], sentence_initial: bool) -> tuple[list[_Tok], bool]:
    """Drop leading/trailing words that are not part of a name; report a person cue."""
    person_cue = False
    strip_words = _STRIP_WORDS
    while words:
        w = words[0].word
        lw = w.lower()
        if w == "Executive" and len(words) > 1 and words[1].word == "Order":
            break
        if w in TITLE_WORDS:
            person_cue = True
        elif not (
            lw in strip_words
            or w in MONTHS_DAYS
            or w in CONNECTORS
            or (sentence_initial and w in COMMON_INITIAL)
            or w in {"The", "A", "An"}
        ):
            break
        words = words[1:]
        sentence_initial = False
    while words and (words[-1].word in CONNECTORS or words[-1].word.lower() in strip_words):
        words = words[:-1]
    return words, person_cue


def _run_mentions(
    sentence: list[_Tok],
    title_case: bool,
    proper_singles: set[str],
    out: list[Mention],
) -> None:
    """Capitalized-sequence and acronym mentions within one sentence."""
    i = 0
    n = len(sentence)
    while i < n:
        tok = sentence[i]
        if not _is_cap(tok.word) or (title_case and tok.word.lower() in HEADLINE_BREAK):
            i += 1
            continue
        # Collect a run: cap words joined by whitespace, connectors allowed between cap words.
        run = [tok]
        j = i + 1
        while j < n:
            cur = sentence[j]
            if cur.gap.strip() not in ("", "&") or run[-1].possessive:
                break
            if title_case and cur.word.lower() in HEADLINE_BREAK:
                break
            if _is_cap(cur.word):
                run.append(cur)
                j += 1
                continue
            if (
                cur.word in CONNECTORS
                and j + 1 < n
                and _is_cap(sentence[j + 1].word)
                and not sentence[j + 1].gap.strip()
            ):
                run.append(cur)
                j += 1
                continue
            break
        # Attach a trailing version number ("Llama 4", "Executive Order 14110").
        if j < n and not sentence[j].gap.strip() and not run[-1].possessive:
            v = sentence[j].word
            if _VERSION_TOKEN_RE.match(v) and not re.match(r"^(19|20)\d\d$", v) and len(v) <= 6:
                run.append(sentence[j])
                j += 1
        sentence_initial = run[0] is sentence[0]
        # "analyst Ines Marlowe", "said Ines Marlowe": a lowercase role word right before the name.
        role_before = i > 0 and not tok.gap.strip() and sentence[i - 1].word.lower() in ROLE_WORDS
        for part in _split_run(run):
            part_initial = sentence_initial and part[0] is run[0]
            words, person_cue = _trim(part, part_initial)
            if not words:
                continue
            person_cue = person_cue or (role_before and part[0] is run[0])
            _emit(words, person_cue, part_initial and words[0] is part[0], title_case, proper_singles, out)
        i = j


def _emit(
    toks: list[_Tok],
    person_cue: bool,
    sentence_initial: bool,
    title_case: bool,
    proper_singles: set[str],
    out: list[Mention],
) -> None:
    words = [t.word for t in toks]
    if words[0] in EXCLUDED_PREFIXES:
        return
    if len(words) <= 2 and words[0].upper() in GENERIC_ACRONYMS:
        return  # "CI 24", "Q3 2026"
    # Model/product runs end at the versioned token (plus tier words): "Atlas-3 Math" -> "Atlas-3".
    for k, w in enumerate(words):
        if _is_versioned(w):
            end = k + 1
            while end < len(words) and (words[end] in MODEL_TIER_WORDS or _VERSION_TOKEN_RE.match(words[end])):
                end += 1
            words = words[:end]
            break
    # Trailing generic acronyms are descriptors, not part of the name: "Pallas Gateway VPN" -> "Pallas Gateway".
    while (
        len(words) > 1
        and words[-1] != "AI"
        and words[-1].rstrip("s").isupper()
        and words[-1].upper().rstrip("S") in GENERIC_ACRONYMS
    ):
        words = words[:-1]
    content = [w for w in words if w not in CONNECTORS]
    if all(w.lower() in GENERIC_WORDS or w.lower() in STOPWORDS for w in content):
        return
    name = " ".join(words)
    if len(name) < 2:
        return
    known = canonical_entity(name)
    if known:
        out.append(Mention(known[0], known[1], normalize_entity(known[0]), toks[0].start, _CONFIDENCE_GAZETTEER))
        return
    if len(words) == 1:
        w = words[0]
        if _is_acronym(w):
            out.append(Mention(w, ORG, normalize_entity(w), toks[0].start, _CONFIDENCE_CUE))
            return
        if _is_versioned(w) or (w[0].islower() and _is_cap(w)):
            out.append(Mention(w, PRODUCT, normalize_entity(w), toks[0].start, _CONFIDENCE_CUE))
            return
        if any(c.isdigit() for c in w) or ("-" in w and w.split("-")[0].lower() in GENERIC_WORDS | {"ai"}):
            return  # identifiers ("DEMO-2026-0142") or compounds ("AI-assisted", "English-language"), not names
        if w in MONTHS_DAYS or w in TITLE_WORDS or w in COMMON_INITIAL or len(w) < 3:
            return
        # A bare cue word ("Bank", "Agent", "Act") or a generic acronym ("GPU") names nothing.
        if (
            w in ORG_WORDS
            or w in PRODUCT_SUFFIX
            or w in POLICY_SUFFIX
            or w.upper().rstrip("S") in GENERIC_ACRONYMS
        ):
            return
        # A lone capitalized word at a sentence start (or in a headline) is only a
        # name if it also appears capitalized mid-sentence somewhere in the text.
        if (sentence_initial or title_case) and w not in proper_singles:
            return
    kind, confidence = _kind_for(words, person_cue)
    if title_case and confidence < _CONFIDENCE_CUE:
        return
    out.append(Mention(name, kind, normalize_entity(name), toks[0].start, confidence))


def _resolve_aliases(mentions: list[Mention]) -> list[Mention]:
    """Map short forms to longer names in the same text.

    "Nimbus" -> "Nimbus Labs", "Westhollow Unified" -> "Westhollow Unified
    School District", "Lopez" -> "Maria Lopez", and initialisms such as
    "OEI" -> "Open Evaluation Institute". Version-specific names are never
    targets ("Pallas Gateway" stays itself next to "Pallas Gateway 9.4").
    """
    distinct = list({m.key: m for m in mentions}.values())
    counts: dict[str, int] = {}
    for m in mentions:
        counts[m.key] = counts.get(m.key, 0) + 1
    longer = [
        m
        for m in distinct
        if " " in m.name
        and m.kind in (ORG, PRODUCT, POLICY)
        and not _VERSION_TOKEN_RE.match(m.name.split()[-1])
        and not any(_is_versioned(w) for w in m.name.split()[1:])
    ]
    # Prefer organizations, then the most-mentioned, then the longest name.
    longer.sort(key=lambda m: (m.kind != ORG, -counts[m.key], -len(m.name)))
    by_last: dict[str, Mention] = {}
    by_initials: dict[str, Mention] = {}
    for m in distinct:
        words = m.name.split()
        if len(words) < 2:
            continue
        if m.kind == PERSON:
            by_last.setdefault(words[-1], m)
        initials = "".join(w[0] for w in words if w[0].isupper())
        if len(initials) >= 2:
            by_initials.setdefault(initials, m)

    targets: dict[str, Mention] = {}
    for m in distinct:
        if m.confidence >= _CONFIDENCE_GAZETTEER or m.kind in (PERSON, TICKER):
            continue
        words = m.name.split()
        if len(words) == 1 and (m.name.lower() in GENERIC_WORDS or m.name in ORG_WORDS):
            continue
        target = next(
            (t for t in longer if len(t.name.split()) > len(words) and t.name.split()[: len(words)] == words),
            None,
        )
        if target is None and len(words) == 1:
            target = by_last.get(m.name)
            if target is None and m.name.isupper():
                target = by_initials.get(m.name)
        if target is not None and target.key != m.key:
            targets[m.key] = target
    return [
        Mention(t.name, t.kind, t.key, m.start, t.confidence) if (t := targets.get(m.key)) else m for m in mentions
    ]


def entity_head(name: str, kind: str) -> str | None:
    """Brand head of a multi-word organization/product name ("Corvane Semiconductor" -> "corvane").

    Lets clustering connect "Corvane" in one article with "Corvane
    Semiconductor" in another. None for generic first words ("Open", "State").
    """
    if kind not in (ORG, PRODUCT):
        return None
    words = normalize_entity(name).split()
    if len(words) < 2:
        return None
    head = words[0]
    if (
        len(head) < 3
        or head in STOPWORDS
        or head in GENERIC_WORDS
        or head.capitalize() in COMMON_INITIAL
        or head.capitalize() in ORG_HEAD_WORDS
        or head in _GENERIC_HEADS
    ):
        return None
    return head


_GENERIC_HEADS = frozenset(
    """
    open united general global international national american first central digital civic regional community
    public state federal north south east west great royal standard advanced applied smart quantum future next
    pacific atlantic western eastern northern southern metropolitan city county district board department
    university institute school college bank center centre office council committee association society
    foundation agency commission ministry bureau markets market capital ventures partners investments fab
    plant factory line project program
    """.split()
)


def extract_mentions(text: str | None, known: dict[str, tuple[str, str]] | None = None) -> list[Mention]:
    """All entity mentions in ``text`` (with repeats), in order of appearance.

    ``known`` maps normalized keys to (name, kind) for names already found
    elsewhere (e.g. an article body), so Title Case headlines can be matched
    against them.
    """
    if not text or not text.strip():
        return []
    mentions: list[Mention] = []
    masked = text

    # 1. Tickers ("NASDAQ: ABCD", "$ABCD").
    for rx in (_TICKER_EXCHANGE_RE, _TICKER_DOLLAR_RE):
        for m in rx.finditer(masked):
            symbol = m.group(1)
            mentions.append(Mention(symbol, TICKER, symbol.lower(), m.start(), _CONFIDENCE_GAZETTEER))
            masked = _mask(masked, m.start(), m.end())

    # 2. Versioned model families ("GPT-5", "Claude Opus 5").
    for m in _MODEL_FAMILY_RE.finditer(masked):
        name = normalize_space(m.group())
        mentions.append(Mention(name, PRODUCT, normalize_entity(name), m.start(), _CONFIDENCE_GAZETTEER))
        masked = _mask(masked, m.start(), m.end())

    # 3. Gazetteer names and aliases.
    for m in _GAZETTEER_RE.finditer(masked):
        canonical, kind = _ALIAS_TO_CANONICAL[m.group()]
        mentions.append(Mention(canonical, kind, normalize_entity(canonical), m.start(), _CONFIDENCE_GAZETTEER))
        masked = _mask(masked, m.start(), m.end())

    # 4. Names found elsewhere (used for Title Case headlines).
    if known:
        for key, (name, kind) in sorted(known.items(), key=lambda kv: -len(kv[1][0])):
            if len(name) < 3:
                continue
            for m in re.finditer(rf"(?<![\w\-]){re.escape(name)}(?![\w\-])", masked):
                mentions.append(Mention(name, kind, key, m.start(), _CONFIDENCE_CUE))
                masked = _mask(masked, m.start(), m.end())

    # 5. Capitalized sequences and acronyms, sentence by sentence.
    toks = _tokens(masked)
    sentences = _sentences(toks)
    # Words capitalized mid-sentence in running prose are evidence of a proper noun.
    proper_singles = {
        s[k].word for s in sentences if not _is_title_case(s) for k in range(1, len(s)) if _is_cap(s[k].word)
    }
    for sentence in sentences:
        shouting = sum(t.word.isupper() for t in sentence) > 0.6 * len(sentence) and len(sentence) >= 4
        if shouting:
            continue
        _run_mentions(sentence, _is_title_case(sentence), proper_singles, mentions)

    mentions.sort(key=lambda m: m.start)
    return _resolve_aliases(mentions)


def _dedupe(mentions: Iterable[Mention]) -> list[Mention]:
    best: dict[str, Mention] = {}
    first: dict[str, int] = {}
    for m in mentions:
        if not m.key:
            continue
        first.setdefault(m.key, m.start)
        cur = best.get(m.key)
        if cur is None or m.confidence > cur.confidence:
            best[m.key] = m
    return sorted(best.values(), key=lambda m: first[m.key])


def extract_entities(text: str | None) -> list[tuple[str, str]]:
    """Named entities in ``text`` as (name, kind), deduplicated, in order of first appearance.

    Kinds: organization, person, product, policy, place, ticker.
    """
    return [(m.name, m.kind) for m in _dedupe(extract_mentions(text))]


# --------------------------------------------------------------------------
# Tagging
# --------------------------------------------------------------------------

MAX_ENTITIES_PER_ARTICLE = 20
_TITLE_WEIGHT = 3.0
_LEDE_BONUS = 0.75
_BODY_CAP = 6  # repeated mentions beyond this add nothing


def _article_body(article: Article) -> str:
    summary = normalize_space(article.summary)
    content = normalize_space(article.content)
    if summary and content.startswith(summary.rstrip("…").rstrip(".")[:120]):
        summary = ""
    return "\n".join(x for x in (summary, (article.content or "")[:20000]) if x)


def entity_saliences(article: Article) -> list[tuple[str, str, str, float]]:
    """(name, kind, normalized key, salience 0..1) for an article, most salient first.

    Title mentions weigh three times a body mention; an early (lede) mention
    adds a bonus; salience is scaled so the top entity scores 1.0.
    """
    body = _article_body(article)
    body_mentions = extract_mentions(body)
    known = {m.key: (m.name, m.kind) for m in _dedupe(body_mentions)}
    title_mentions = extract_mentions(article.title or "", known=known)
    # Resolve short forms across title and body ("Orrin" in the body, "Orrin Lithography" in the title).
    offset = len(article.title or "") + 1
    combined = _resolve_aliases([*title_mentions, *(replace(m, start=m.start + offset) for m in body_mentions)])
    title_mentions = combined[: len(title_mentions)]
    body_mentions = [replace(m, start=m.start - offset) for m in combined[len(title_mentions) :]]

    info: dict[str, Mention] = {m.key: m for m in _dedupe([*title_mentions, *body_mentions])}
    raw: dict[str, float] = dict.fromkeys(info, 0.0)
    for key in {m.key for m in title_mentions}:
        raw[key] += _TITLE_WEIGHT
    counts: dict[str, int] = {}
    for m in body_mentions:
        counts[m.key] = counts.get(m.key, 0) + 1
    lede_end = min(len(body), 400)
    for key, count in counts.items():
        raw[key] += min(count, _BODY_CAP)
        if any(m.key == key and m.start < lede_end for m in body_mentions):
            raw[key] += _LEDE_BONUS
    top = max(raw.values(), default=0.0) or 1.0
    ranked = sorted(raw.items(), key=lambda kv: (-kv[1], info[kv[0]].start))
    return [(info[k].name, info[k].kind, k, round(v / top, 3)) for k, v in ranked if v > 0]


def _get_or_create_entity(session: Session, name: str, kind: str, key: str) -> Entity:
    entity = session.scalars(select(Entity).where(Entity.normalized == key, Entity.kind == kind)).first()
    if entity is None:
        entity = Entity(name=name[:200], normalized=key, kind=kind)
        session.add(entity)
        session.flush()
    return entity


def tag_article_entities(session: Session, article: Article) -> list[Entity]:
    """Extract entities for ``article`` and upsert ``Entity`` + ``ArticleEntity`` rows.

    Idempotent: re-tagging updates saliences and removes links to entities no
    longer found. Returns the article's entities, most salient first (at most
    ``MAX_ENTITIES_PER_ARTICLE``).
    """
    scored = entity_saliences(article)[:MAX_ENTITIES_PER_ARTICLE]
    existing = {link.entity_id: link for link in article.entities}
    entities: list[Entity] = []
    keep: set[int] = set()
    for name, kind, key, salience in scored:
        entity = _get_or_create_entity(session, name, kind, key)
        if entity.id in keep:
            continue
        keep.add(entity.id)
        entities.append(entity)
        link = existing.get(entity.id)
        if link is None:
            article.entities.append(ArticleEntity(entity=entity, salience=salience))
        else:
            link.salience = salience
    for entity_id, link in existing.items():
        if entity_id not in keep:
            article.entities.remove(link)
    session.flush()
    return entities


def article_entity_keys(session: Session, article: Article, limit: int = 12) -> list[str]:
    """Normalized entity keys of an article, most salient first (tags it if untagged)."""
    if not article.entities:
        tag_article_entities(session, article)
    links = sorted(article.entities, key=lambda link: -link.salience)
    return [link.entity.normalized for link in links[:limit]]
