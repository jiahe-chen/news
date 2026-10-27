"""Editorial guardrails: language lint, advice-request detection, and performance-claim labelling."""

from __future__ import annotations

import pytest

from newsren.editorial.guardrails import (
    ADVICE_REFUSAL,
    FINANCE_DISCLAIMER,
    detect_advice_request,
    is_hedged,
    lint_text,
    looks_like_performance,
    performance_claim_issues,
)
from newsren.models import BriefClaim
from newsren.schemas import ClaimOut

# (vertical, text, expected violation label or None for clean text)
LINT_CASES = [
    # --- finance: ratings, price targets, personalized advice
    ("finance", "Analysts upgraded the stock to buy after the earnings call.", "Buy/sell/hold call"),
    ("finance", "We rate Lumen Labs a strong buy.", "Buy/sell/hold call"),
    ("finance", "At these levels the shares are a buy.", "Buy/sell/hold call"),
    ("finance", "Three brokers downgraded the shares to hold.", "Buy/sell/hold call"),
    ("finance", "The bank carries an outperform rating from two firms.", "Buy/sell/hold call"),
    ("finance", "Harbor Bank set a $250 price target on the stock.", "Price target"),
    ("finance", "Analysts raised their price target to $150.", "Price target"),
    ("finance", "Our target price is $98.", "Price target"),
    ("finance", "You should invest in index funds before the next rate cut.", "Personalized investment advice"),
    ("finance", "Investors should buy bank stocks ahead of the stress tests.", "Personalized investment advice"),
    ("finance", "Allocate 20% of your portfolio to AI chipmakers.", "Personalized investment advice"),
    ("finance", "Now is a great time to buy regional banks.", "Personalized investment advice"),
    ("finance", "Buy the dip while it lasts.", "Personalized investment advice"),
    # --- finance: returns, market-beating claims, forecasts stated as fact
    ("finance", "The fund offers guaranteed returns of 8% a year.", "Guaranteed or risk-free return claim"),
    ("finance", "It is a risk-free investment for retirees.", "Guaranteed or risk-free return claim"),
    ("finance", "AI agents can reliably beat the market.", "Claim that an AI agent or strategy beats the market"),
    ("finance", "The startup sells a market-beating trading bot.", "Claim that an AI agent or strategy beats"),
    ("finance", "Shares will soar to $300 next year.", "Forecast stated as fact"),
    ("finance", "Interest rates will fall in the spring.", "Forecast stated as fact"),
    # --- industry: projections must be labelled; ratings are not industry rules
    ("industry", "Chip prices will soar to record levels in 2027.", "Forecast stated as fact"),
    ("industry", "Global capacity is set to double by 2028.", "Forecast stated as fact"),
    # --- every vertical: overclaiming
    ("education", "The study proves that AI tutoring improves math scores.", "Overstated certainty"),
    ("education", "The program is scientifically proven to boost reading.", "Overstated certainty"),
    ("education", "The app guarantees better grades for every student.", "Guarantee language"),
    ("education", "AI tutoring led to higher test scores.", "Causal claim about outcomes"),
    ("technology", "The new model is 100% accurate on medical questions.", "Guarantee language"),
    (
        "technology",
        "This AI trading agent will beat the market.",
        "Claim that an AI agent or strategy beats the market",
    ),
    # --- clean: negated, attributed, hedged, educational, or out-of-vertical text
    ("finance", FINANCE_DISCLAIMER, None),
    ("finance", ADVICE_REFUSAL, None),
    ("finance", "A price target is an analyst's estimate of where a stock could trade in 12 months.", None),
    ("finance", "The study does not show that AI agents can reliably beat the market.", None),
    ("finance", "Backtests cannot prove that a strategy will beat the market.", None),
    ("finance", "The SEC said the adviser promised guaranteed returns to clients.", None),
    ("finance", 'The firm\'s ads described its fund as "risk-free returns for everyone".', None),
    ("finance", "The company expects revenue will rise to $5 billion next year.", None),
    ("finance", "Revenue will rise to $5 billion next year, the company said.", None),
    ("finance", "Under the bank's stress scenario, shares could fall 30%.", None),
    ("finance", "The Fed reiterated its plan to hold rates steady.", None),
    ("finance", "Affirm, a buy now, pay later lender, reported revenue of $800 million.", None),
    ("finance", "The bank's analysts initiated a buy-side research program.", None),
    ("finance", "Moody's downgraded the lender's credit rating to junk.", None),
    ("finance", "The Sharpe ratio compares returns with risk-free returns.", None),
    ("finance", "The firm will decline to comment on the filing.", None),
    ("finance", "Harbor Bank's backtest beat a 60/40 portfolio from 2005 to 2025.", None),
    ("industry", "Analysts expect chip prices will rise next year.", None),
    ("technology", "Chip prices will soar to record levels in 2027.", None),
    ("technology", "Analysts upgraded the stock to buy.", None),
    ("technology", "The model proves theorems in Lean and solved 61% of problems.", None),
    ("education", "The press release claims the study proves that tutoring improves scores.", None),
    ("education", "The study does not prove that AI tutoring improves scores.", None),
    ("education", "This trial settled the debate over AI tutors.", "Overstated certainty"),
    ("technology", "Independent replication by a second group would help settle the question.", None),
    ("technology", "If the replication proves that the gap is real, Lumen Labs may revise its claim.", None),
    ("education", "In a randomized controlled trial, AI tutoring led to higher test scores.", None),
    ("education", "AI tutoring may lead to higher test scores, the authors suggest.", None),
    ("education", "Texas guarantees admission to students in the top 10% of their class.", None),
    ("finance", "", None),
]


@pytest.mark.parametrize(("vertical", "text", "expected"), LINT_CASES)
def test_lint_text(vertical: str, text: str, expected: str | None) -> None:
    violations = lint_text(text, vertical)
    if expected is None:
        assert violations == [], violations
    else:
        assert violations, f"expected a {expected!r} violation"
        assert any(v.startswith(expected) for v in violations), violations


def test_lint_violation_is_readable_and_quotes_the_offending_words() -> None:
    [violation] = lint_text("Analysts raised their price target to $150.", "finance")
    assert violation.startswith('Price target: "price target to $150"')
    assert "does not publish price targets" in violation


def test_lint_reports_each_problem_once_in_text_order() -> None:
    text = "Upgraded to buy with a $250 price target. Shares will soar. Upgraded to buy with a $250 price target."
    labels = [v.split(":")[0] for v in lint_text(text, "finance")]
    assert labels == ["Buy/sell/hold call", "Price target", "Forecast stated as fact"]


def test_lint_accepts_vertical_enum_and_unknown_vertical() -> None:
    from newsren.models import Vertical

    assert lint_text("We rate it a strong buy.", Vertical.FINANCE)
    assert lint_text("We rate it a strong buy.", None) == []
    assert lint_text("The study proves that tutoring works.", None)


ADVICE_REQUESTS = [
    "should I buy NVDA",
    "Should I buy NVDA after earnings?",
    "should i sell my Tesla shares?",
    "Is Lumen Labs a good investment?",
    "is X a good investment",
    "Is now a good time to buy bank stocks?",
    "What stocks should I buy this year?",
    "what stocks should I",
    "Which AI stocks should I invest in?",
    "How should I invest my 401k?",
    "how should I invest my 401(k)?",
    "Where should I put my savings?",
    "How much should I invest in bitcoin?",
    "My portfolio is down 20%, what should I do?",
    "What's the price target for NVDA?",
    "price target for Lumen Labs",
    "NVDA price target?",
    "Give me your top stock picks",
    "Is NVDA a buy?",
    "Buy or sell Circuit Ledger after earnings?",
    "Will Nvidia stock go up after earnings?",
    "Is it too late to buy AI stocks?",
    "Can you recommend some ETFs?",
    "Should investors sell bank stocks now?",
    "Should I buy the dip?",
    "Is it a good idea to short Harbor Bank?",
    "Is Harbor Bank stock worth buying?",
]

NOT_ADVICE_REQUESTS = [
    "What is a price target?",
    "what is a price target?",
    "How do analysts set price targets?",
    "How are price targets for stocks set?",
    "how did the SEC rule affect brokers",
    "How did the SEC rule affect my 401k?",
    "What did Lumen Labs report for quarterly revenue?",
    "Is the backtest result the same as live performance?",
    "Why did the bank say its AI agents beat a 60/40 portfolio?",
    "What should investors know about the new FINRA guidance?",
    "Should schools buy AI tutoring software?",
    "Is the AI tutoring program a good investment for schools?",
    "What does a strong buy rating mean?",
    "Explain the Sharpe ratio",
    "What is dilution in a funding round?",
    "Which companies did the SEC fine for AI washing?",
    "Will the Fed cut rates?",
    "How do I read a 10-Q filing?",
    "Should I buy a new laptop for school?",
    "What is the difference between a backtest and a live track record?",
    "Should I trust this benchmark?",
    "",
]


@pytest.mark.parametrize("question", ADVICE_REQUESTS)
def test_detect_advice_request_true(question: str) -> None:
    assert detect_advice_request(question) is True


@pytest.mark.parametrize("question", NOT_ADVICE_REQUESTS)
def test_detect_advice_request_false(question: str) -> None:
    assert detect_advice_request(question) is False


def test_disclaimer_and_refusal_cover_the_plan() -> None:
    disclaimer = FINANCE_DISCLAIMER.lower()
    for phrase in ("not investment advice", "buy, sell, or hold", "backtest, simulated, or live", "benchmark"):
        assert phrase in disclaimer
    assert "delayed" in disclaimer and "timestamped" in disclaimer
    refusal = ADVICE_REFUSAL.lower()
    assert "personalized investment advice" in refusal
    for angle in ("filings", "concepts", "analyst", "licensed financial adviser"):
        assert angle in refusal


@pytest.mark.parametrize(
    ("text", "hedged"),
    [
        ("Revenue is projected to reach $5 billion.", True),
        ("Analysts forecast a slowdown.", True),
        ("The company expects margins to narrow.", True),
        ("Prices could fall if demand weakens.", True),
        ("In a downside scenario, output drops 10%.", True),
        ("Revenue will reach $5 billion.", False),
    ],
)
def test_is_hedged(text: str, hedged: bool) -> None:
    assert is_hedged(text) is hedged


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("AI agents beat a 60/40 portfolio by 1.8 points a year.", True),
        ("The strategy returned 14% in 2025.", True),
        ("The fund's annualized return was 9%.", True),
        ("It outperformed the S&P 500 over two decades.", True),
        ("The model outperformed rivals on the MMLU benchmark.", False),
        ("Quarterly revenue beat the market's expectations.", False),
        ("JPMorgan rolled out LLM Suite to about 50,000 staff.", False),
    ],
)
def test_looks_like_performance(text: str, expected: bool) -> None:
    assert looks_like_performance(text) is expected


def test_performance_claim_issues_for_unlabelled_claim() -> None:
    claim = ClaimOut(text="The AI agents performed well for the bank.", claim_type="performance")
    issues = performance_claim_issues(claim)
    assert len(issues) == 4
    assert any("backtest, simulated, or live" in i for i in issues)
    assert any("period" in i for i in issues)
    assert any("benchmark" in i for i in issues)
    assert any("source" in i for i in issues)


def test_performance_claim_issues_clean_when_fully_labelled() -> None:
    claim = ClaimOut(
        text="In a backtest, AI agents beat a 60/40 portfolio by 1.8 points a year.",
        claim_type="performance",
        performance_label="backtest",
        performance_period="2005-2025",
        benchmark="60/40 stock/bond portfolio",
        citation_ids=["c1"],
    )
    assert performance_claim_issues(claim) == []


def test_performance_claim_issues_accepts_labels_stated_in_the_text() -> None:
    claim = ClaimOut(
        text="In a 2005-2024 backtest, Northbridge's AI agents beat a 60/40 portfolio.",
        claim_type="fact",
        citation_ids=["c1"],
    )
    assert performance_claim_issues(claim) == []
    over_two_decades = ClaimOut(
        text="Northbridge's AI agents outperformed a 60/40 benchmark over two decades.", citation_ids=["c1"]
    )
    assert performance_claim_issues(over_two_decades) == [
        "Not labelled backtest, simulated, or live (the text describes investment or strategy performance)."
    ]


def test_performance_claim_issues_catches_untyped_performance_text_on_orm_claim() -> None:
    claim = BriefClaim(
        text="The fund returned 14% last year.",
        claim_type="fact",
        citation_ids=["c1"],
        performance_label=None,
    )
    issues = performance_claim_issues(claim)
    assert issues[0].startswith("Not labelled backtest, simulated, or live (the text describes")


def test_performance_claim_issues_rejects_unknown_label_and_ignores_non_performance() -> None:
    bad = BriefClaim(
        text="Agents beat the index.",
        claim_type="performance",
        performance_label="paper-trading",
        performance_period="2024",
        benchmark="S&P 500",
        citation_ids=["c1"],
    )
    assert performance_claim_issues(bad) == [
        "Performance label 'paper-trading' is not one of backtest, simulated, or live."
    ]
    plain = BriefClaim(text="The SEC fined Delphia $225,000.", claim_type="fact", citation_ids=["c1"])
    assert performance_claim_issues(plain) == []
