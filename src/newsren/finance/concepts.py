"""Finance concepts for learners (plan §4: "Finance concepts for learners").

Short, neutral, educational explainers that finance stories link to, so a
reader who meets "backtest", "Sharpe ratio", or "post-money valuation" in a
story can learn what it means and what to watch out for. Examples are
explicitly hypothetical; nothing here is investment, legal, or tax advice.

``concepts_in_text`` links text to concepts by matching each concept's
aliases (literal phrases, case-insensitive except for all-caps acronyms) and
patterns (regular expressions for phrases that vary, such as "guides
fourth-quarter revenue"). Generic words that are common outside finance
(for example "benchmark" in an AI-model story, or "guidance" in a regulatory
notice) are deliberately matched only in finance-specific phrasing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cache

from newsren.textutil import slugify


@dataclass(frozen=True)
class Concept:
    slug: str
    term: str
    short: str
    explainer: str  # 2-4 paragraphs separated by blank lines
    example: str  # a worked, clearly hypothetical example
    pitfalls: tuple[str, ...] = ()
    related: tuple[str, ...] = ()  # slugs of related concepts
    category: str = "investing"
    aliases: tuple[str, ...] = ()  # literal phrases that link text to this concept
    patterns: tuple[str, ...] = field(default=(), repr=False)  # extra regular expressions

    @property
    def paragraphs(self) -> list[str]:
        """The explainer split into paragraphs (for templates)."""
        return [p.strip() for p in self.explainer.split("\n\n") if p.strip()]


CATEGORY_LABELS: dict[str, str] = {
    "investing": "Testing investment claims",
    "startups": "Startups and venture funding",
    "earnings": "Earnings and company filings",
    "regulation": "Rules for AI in finance",
}

DISCLAIMER = (
    "These explainers are educational. They are not investment, legal, or tax advice, and the examples "
    "use made-up numbers."
)


CONCEPTS: list[Concept] = [
    # ------------------------------------------------------------------ testing investment claims
    Concept(
        slug="backtesting",
        term="Backtesting",
        short="Testing a trading or investment strategy on historical data to see how it would have performed.",
        explainer=(
            "A backtest applies a strategy's rules to past market data and records the trades and returns it "
            "would have produced. It is a simulation: no real money was invested, no real orders were filled, "
            "and the strategy's designers already knew how history turned out.\n\n"
            "Backtests are a reasonable first step for checking whether an idea is coherent, and researchers "
            "use them constantly. But a backtest is evidence about the past under assumptions, not a track "
            "record. Its result depends on the period chosen, the costs assumed, the benchmark compared "
            "against, and how many variations were tried before the reported one.\n\n"
            "When a firm reports that an AI system or strategy 'beat the market', the first questions are: "
            "was this a backtest, a simulation on live data (paper trading), or real money; over what period; "
            "against which benchmark; and with what costs? The Newsren labels every performance claim as "
            "backtest, simulated, or live for this reason."
        ),
        example=(
            "Hypothetical example: a researcher tests a rule that buys stocks after three days of gains, using "
            "2005-2024 prices. The backtest shows 9% a year versus 7% for an index. Before treating that as "
            "meaningful, you would ask how many other rules were tried, whether trading costs were included, "
            "and whether the rule still works on years it was not designed on."
        ),
        pitfalls=(
            "Treating a backtest as a track record, or as a reason to expect the strategy to keep working.",
            "Ignoring transaction costs, taxes, and the market impact of trading.",
            "Not asking how many strategies were tested before the winner was reported.",
            "Using information that was not available at the time (look-ahead bias).",
        ),
        related=(
            "overfitting",
            "in-sample-vs-out-of-sample",
            "look-ahead-bias",
            "survivorship-bias",
            "paper-trading",
        ),
        aliases=("historical simulation",),
        patterns=(r"\bback[\s-]?test(?:s|ed|ing|er)?\b",),
    ),
    Concept(
        slug="overfitting",
        term="Overfitting",
        short="When a model or strategy fits the noise in past data so closely that it fails on new data.",
        explainer=(
            "Every data set contains real patterns and random noise. A model with many adjustable settings, "
            "or a researcher who tries many variations, can end up matching the noise: the model looks "
            "excellent on the data it was built on and disappoints on anything new.\n\n"
            "In finance this risk is high because markets are noisy and history is short. If you test a "
            "thousand strategies, some will look brilliant purely by chance. Reporting only the best one, "
            "without saying how many were tried, overstates the evidence. This is sometimes called data "
            "snooping or data mining.\n\n"
            "Common defences are keeping data aside for an out-of-sample test, preferring simpler models, "
            "adjusting for the number of strategies tested, and waiting for live or paper-trading results "
            "before drawing conclusions."
        ),
        example=(
            "Hypothetical example: an AI model is tuned with 200 settings on 2010-2020 data and returns 15% a "
            "year in that period. On 2021-2024 data it was never tuned on, it returns 4%, below a simple index "
            "fund. The gap suggests the model learned the noise of 2010-2020."
        ),
        pitfalls=(
            "Judging a model only by its fit to the data it was trained or tuned on.",
            "Not disclosing how many model variants were tried.",
            "Assuming that more complex models are always better.",
        ),
        related=("backtesting", "in-sample-vs-out-of-sample", "look-ahead-bias"),
        aliases=(
            "overfitting",
            "overfit",
            "overfitted",
            "over-fitting",
            "curve fitting",
            "curve-fitting",
            "data snooping",
        ),
    ),
    Concept(
        slug="in-sample-vs-out-of-sample",
        term="In-sample vs. out-of-sample",
        short="In-sample data is used to build or tune a model; out-of-sample data is held back to test it fairly.",
        explainer=(
            "An in-sample result measures how well a model performs on the same data used to design it. It "
            "almost always looks better than reality, because design choices were made with that data in "
            "view. An out-of-sample result uses data the model never saw while it was being built, which is "
            "a fairer test.\n\n"
            "Researchers often split history into a training period and a later test period, or use "
            "'walk-forward' testing, where the model is repeatedly rebuilt on past data and tested on the next "
            "slice. Even this can leak information if researchers peek at test results and go back to "
            "adjust the model.\n\n"
            "The strongest out-of-sample evidence is time that has not happened yet: paper trading or live "
            "results recorded after the strategy was fixed. When authors caution that a result is in-sample, "
            "they are saying it is not proof of future performance."
        ),
        example=(
            "Hypothetical example: a team defines market 'regimes' using all data from 2005 to 2024, then "
            "shows that switching portfolios by regime beat a 60/40 mix over the same years. That is an "
            "in-sample result. An out-of-sample check would define regimes using only data available at each "
            "date and test on years not used to design the rules."
        ),
        pitfalls=(
            "Reading an in-sample result as a forecast.",
            "Re-using the test period to tune the model, which turns it into in-sample data.",
            "Choosing a test period that happens to suit the strategy.",
        ),
        related=("backtesting", "overfitting", "look-ahead-bias", "paper-trading"),
        aliases=("in-sample", "out-of-sample", "walk-forward", "holdout period", "hold-out period"),
    ),
    Concept(
        slug="look-ahead-bias",
        term="Look-ahead bias",
        short="Using information in a test that would not have been available at the time a decision was made.",
        explainer=(
            "Look-ahead bias creeps into a backtest when the simulated strategy 'knows' something it could "
            "not have known on the date it supposedly traded. Examples include using revised economic data "
            "instead of the first published figures, using a company's annual results before the filing date, "
            "or defining market conditions with hindsight.\n\n"
            "The bias can be subtle: data vendors often store the latest version of a figure, not what was "
            "known at the time. Point-in-time data, which records what was known on each date, is the usual "
            "remedy.\n\n"
            "Look-ahead bias makes results look better than any real investor could have achieved, so it is "
            "worth asking how each input was timed."
        ),
        example=(
            "Hypothetical example: a backtest buys a stock on 1 January because its full-year earnings rose, "
            "but those earnings were not filed until March. A real investor could not have acted on them in "
            "January, so the backtest's gains from that rule are not achievable."
        ),
        pitfalls=(
            "Using revised data instead of what was published at the time.",
            "Defining signals or regimes with the full sample period.",
            "Ignoring filing and publication delays.",
        ),
        related=("backtesting", "in-sample-vs-out-of-sample", "survivorship-bias"),
        aliases=("look-ahead bias", "lookahead bias", "look-ahead", "hindsight bias", "point-in-time data"),
    ),
    Concept(
        slug="survivorship-bias",
        term="Survivorship bias",
        short="Drawing conclusions only from the companies or funds that survived, ignoring those that failed.",
        explainer=(
            "If a study looks only at companies that exist today, it leaves out those that went bankrupt, "
            "were delisted, or were acquired at low prices. Their absence makes past returns look higher and "
            "risk look lower than an investor would really have experienced.\n\n"
            "The same problem affects fund performance studies (closed funds disappear from databases) and "
            "startup stories (we hear about the winners). A fair test includes everything that was available "
            "to invest in at the start of the period, including what later failed."
        ),
        example=(
            "Hypothetical example: a backtest of 'today's 500 largest companies' over the past 20 years shows "
            "strong returns. But it only includes firms that grew large enough to be in today's list; firms "
            "that shrank or failed during those years are missing, so the result is flattered."
        ),
        pitfalls=(
            "Building a historical test from today's index members.",
            "Judging a sector by its most successful startups.",
            "Using fund databases that drop closed funds.",
        ),
        related=("backtesting", "look-ahead-bias", "benchmark"),
        aliases=("survivorship bias", "survivor bias"),
    ),
    Concept(
        slug="sharpe-ratio",
        term="Sharpe ratio",
        short="Return earned above a risk-free rate per unit of volatility; a common risk-adjusted performance measure.",
        explainer=(
            "The Sharpe ratio divides a portfolio's excess return (its return minus a risk-free rate such as "
            "Treasury bills) by the standard deviation of those returns, a measure of how much they swing. "
            "A higher ratio means more return per unit of volatility over the period measured. It is usually "
            "quoted on an annual basis.\n\n"
            "It is useful for comparing strategies with different levels of risk, but it has limits. It "
            "treats upside and downside swings alike, can understate the risk of rare large losses, and is "
            "sensitive to the period chosen and to how often returns are measured. A Sharpe ratio from a "
            "backtest inherits all of the backtest's weaknesses."
        ),
        example=(
            "Hypothetical example: portfolio A returns 8% a year with 10% volatility; portfolio B returns 10% "
            "with 20% volatility; the risk-free rate is 3%. A's Sharpe ratio is (8 - 3) / 10 = 0.5 and B's is "
            "(10 - 3) / 20 = 0.35, so A earned more return per unit of volatility even though B returned more."
        ),
        pitfalls=(
            "Comparing Sharpe ratios computed over different periods or frequencies.",
            "Forgetting that a backtested Sharpe ratio is not a live result.",
            "Assuming volatility captures all risks, such as rare crashes or illiquidity.",
        ),
        related=("maximum-drawdown", "benchmark", "backtesting"),
        aliases=("Sharpe ratio", "Sharpe ratios", "risk-adjusted return", "risk-adjusted returns"),
    ),
    Concept(
        slug="maximum-drawdown",
        term="Maximum drawdown",
        short="The largest peak-to-trough fall in a portfolio's value before it recovered to a new high.",
        explainer=(
            "A drawdown measures how far a portfolio fell from its previous high point. The maximum drawdown "
            "is the worst such fall over a period. It captures something averages hide: how painful the worst "
            "stretch was for someone who held on.\n\n"
            "Maximum drawdown depends heavily on the period studied; a test that skips a major crisis can "
            "look much safer than one that includes it. It also says nothing about how long recovery took, "
            "so it is often reported alongside the time to recover."
        ),
        example=(
            "Hypothetical example: a portfolio grows from $100 to $150, falls to $105, then climbs to $160. "
            "The drawdown from the $150 peak to the $105 trough is 30%, which is its maximum drawdown if no "
            "larger fall occurred."
        ),
        pitfalls=(
            "Comparing drawdowns measured over periods with different crises.",
            "Ignoring how long it took to recover.",
            "Treating a backtested drawdown as the worst that can happen.",
        ),
        related=("sharpe-ratio", "benchmark", "backtesting"),
        aliases=("maximum drawdown", "max drawdown", "drawdown", "drawdowns", "peak-to-trough"),
    ),
    Concept(
        slug="benchmark",
        term="Benchmark (e.g., a 60/40 portfolio)",
        short="A reference portfolio or index used to judge whether a strategy did better or worse than a simple alternative.",
        explainer=(
            "Performance only means something relative to an alternative. A benchmark is that alternative: an "
            "index such as a broad stock market index, or a simple mix such as the '60/40 portfolio' of 60% "
            "stocks and 40% bonds that many balanced funds resemble.\n\n"
            "A fair benchmark matches the strategy's risk and what it invests in. Comparing a stock-heavy "
            "strategy with a bond-heavy benchmark, or choosing a benchmark after seeing results, can make "
            "ordinary performance look exceptional. Reports should name the benchmark, its exact mix, and how "
            "often it was rebalanced.\n\n"
            "Beating a benchmark in a backtest is different from beating it with real money over time, after "
            "costs."
        ),
        example=(
            "Hypothetical example: an AI-driven stock portfolio returned 9% a year in a backtest, while a "
            "60/40 benchmark returned 7%. Because the AI portfolio held only stocks, a fairer comparison might "
            "be an all-stock index, which returned 10% over the same years."
        ),
        pitfalls=(
            "Accepting a benchmark that takes less risk than the strategy.",
            "Not checking whether the benchmark was chosen after seeing the results.",
            "Ignoring costs and fees that the benchmark would not incur.",
        ),
        related=("sharpe-ratio", "maximum-drawdown", "backtesting", "transaction-costs"),
        aliases=("60/40", "benchmark portfolio", "benchmark portfolios", "benchmark index", "market benchmark"),
    ),
    Concept(
        slug="transaction-costs",
        term="Transaction costs",
        short="The costs of trading (commissions, bid-ask spreads, market impact, slippage, taxes) that reduce real returns.",
        explainer=(
            "Every trade costs something: explicit fees and commissions, the bid-ask spread between buying and "
            "selling prices, slippage when prices move before an order fills, and market impact when a large "
            "order pushes the price. Taxes on gains are another drag. Costs are often quoted in basis points "
            "(hundredths of a percent).\n\n"
            "Strategies that trade often are especially sensitive to costs. A backtest that assumes low or "
            "zero costs can show profits that disappear in practice, so reports should state the costs "
            "assumed and whether taxes were included."
        ),
        example=(
            "Hypothetical example: a strategy earns 0.10% gross per trade and trades 200 times a year. With "
            "assumed costs of 0.05% per trade its net gain halves; at realistic costs of 0.12% per trade it "
            "loses money."
        ),
        pitfalls=(
            "Assuming trades fill at the last quoted price.",
            "Ignoring taxes and fees.",
            "Assuming costs stay low as the amount invested grows.",
        ),
        related=("backtesting", "benchmark"),
        aliases=(
            "transaction costs",
            "transaction cost",
            "trading costs",
            "trading cost",
            "slippage",
            "bid-ask spread",
            "basis points per trade",
            "market impact",
        ),
    ),
    Concept(
        slug="paper-trading",
        term="Paper trading (simulated trading)",
        short="Running a strategy on live market data with simulated orders and no real money.",
        explainer=(
            "Paper trading sits between a backtest and a live track record. The strategy is fixed, then makes "
            "simulated trades as new market data arrives, so the results are genuinely out-of-sample in time. "
            "No real orders are placed and no client money is at risk.\n\n"
            "Paper results are stronger evidence than a backtest but still simulated: they do not capture "
            "real fills, market impact, operational errors, or the behaviour of a strategy managing large "
            "sums. Results should be labelled as simulated, not live."
        ),
        example=(
            "Hypothetical example: after a strong backtest, a bank runs its AI portfolio agents for 12 months "
            "on live prices with a notional $10 million. The agents gain 5% while the benchmark gains 6%. This "
            "is a simulated result, and it tells the bank more than the backtest did."
        ),
        pitfalls=(
            "Describing paper-trading results as live performance.",
            "Assuming simulated fills match real ones.",
            "Judging a strategy on a short paper-trading period.",
        ),
        related=("backtesting", "in-sample-vs-out-of-sample", "transaction-costs"),
        aliases=(
            "paper trading",
            "paper-trading",
            "paper trade",
            "paper trades",
            "simulated trading",
            "live track record",
        ),
    ),
    # ------------------------------------------------------------------ startups and venture funding
    Concept(
        slug="funding-rounds",
        term="Funding rounds (pre-seed, seed, Series A, B, C...)",
        short="The stages in which startups raise money from investors, usually in exchange for shares.",
        explainer=(
            "Startups typically raise money in rounds. Pre-seed and seed rounds fund early product work; "
            "Series A, B, C and later rounds usually fund growth once there is evidence of demand. Growth "
            "rounds and debt financing appear later. The labels are conventions, not legal categories, and "
            "the size of a 'Series A' varies widely by sector and year.\n\n"
            "A round announcement typically names the amount, the lead investor (who usually negotiates the "
            "terms), and other participants. It often omits the valuation and terms, which matter a great "
            "deal to founders and earlier investors.\n\n"
            "A large round tells you investors were persuaded; it does not by itself show that a company has "
            "paying customers, profits, or a lasting advantage. Aggregate funding trends can also swing on a "
            "few large deals."
        ),
        example=(
            "Hypothetical example: a startup raises a $3 million seed round led by an angel group, then 18 "
            "months later a $20 million Series A led by a venture firm, with the seed investors participating "
            "to keep their ownership from shrinking too much."
        ),
        pitfalls=(
            "Treating round size as a measure of success or quality.",
            "Comparing stage labels across sectors as if they meant the same amounts.",
            "Ignoring undisclosed terms such as valuation and liquidation preferences.",
        ),
        related=("dilution", "term-sheet", "pre-money-vs-post-money-valuation", "liquidation-preference"),
        category="startups",
        aliases=(
            "funding round",
            "funding rounds",
            "seed round",
            "seed funding",
            "pre-seed",
            "venture round",
            "growth round",
            "venture funding",
            "venture capital",
        ),
        patterns=(r"(?-i:\bSeries [A-H]\b)",),
    ),
    Concept(
        slug="dilution",
        term="Dilution",
        short="The reduction in existing owners' percentage stake when a company issues new shares.",
        explainer=(
            "When a company sells new shares to raise money, or grants stock options to employees, the total "
            "number of shares rises. Existing shareholders keep the same number of shares but own a smaller "
            "percentage of the company. That is dilution.\n\n"
            "Dilution is not automatically bad: if the new money makes the company more valuable, a smaller "
            "slice can be worth more than the old, larger one. But founders and early employees should "
            "understand how each round, option pool, and convertible instrument changes ownership. For public "
            "companies, dilution also explains why diluted earnings per share can be lower than basic EPS."
        ),
        example=(
            "Hypothetical example: a founder owns 6 million of a startup's 10 million shares (60%). The startup "
            "issues 2.5 million new shares to investors, bringing the total to 12.5 million. The founder still "
            "has 6 million shares, now 48% of the company."
        ),
        pitfalls=(
            "Forgetting option pools and convertible notes when estimating ownership.",
            "Looking only at percentage owned rather than the value of the stake.",
            "Assuming later rounds will not dilute earlier investors.",
        ),
        related=("pre-money-vs-post-money-valuation", "term-sheet", "eps", "funding-rounds"),
        category="startups",
        aliases=("dilution", "dilutive", "dilute", "dilutes", "diluting", "option pool"),
    ),
    Concept(
        slug="term-sheet",
        term="Term sheet",
        short="A summary of the main terms of an investment, agreed before the final legal documents are drafted.",
        explainer=(
            "A term sheet sets out the deal: how much is invested, at what valuation, which class of shares "
            "investors receive, liquidation preferences, board seats, voting and protective rights, "
            "anti-dilution protection, and any option pool expansion. Most of its terms are not legally "
            "binding until the final agreements are signed, though clauses such as confidentiality or an "
            "exclusivity period often are.\n\n"
            "Two offers with the same headline valuation can be very different once preferences, board "
            "control, and option-pool terms are compared. Founders usually review term sheets with a lawyer."
        ),
        example=(
            "Hypothetical example: offer A values a startup at $40 million post-money with a 1x non-participating "
            "liquidation preference. Offer B values it at $50 million but with a 2x participating preference "
            "and an enlarged option pool counted before the investment. Depending on the eventual sale price, "
            "offer A can leave founders with more."
        ),
        pitfalls=(
            "Comparing offers on headline valuation alone.",
            "Overlooking which terms are binding.",
            "Ignoring how the option pool is counted.",
        ),
        related=("pre-money-vs-post-money-valuation", "liquidation-preference", "dilution", "funding-rounds"),
        category="startups",
        aliases=("term sheet", "term sheets"),
    ),
    Concept(
        slug="pre-money-vs-post-money-valuation",
        term="Pre-money vs. post-money valuation",
        short="A company's agreed value just before (pre-money) and just after (post-money) new investment.",
        explainer=(
            "Pre-money valuation is what investors and founders agree the company is worth before the new "
            "money arrives. Post-money valuation is the pre-money value plus the new investment. The investor's "
            "ownership stake equals the investment divided by the post-money valuation.\n\n"
            "Headline valuations are negotiated prices for a private, illiquid stake, often with preferred "
            "terms attached, so they are not the same as a market price for every share. Many announcements "
            "do not disclose valuation at all."
        ),
        example=(
            "Hypothetical example: investors put $5 million into a startup at a $20 million pre-money "
            "valuation. The post-money valuation is $25 million, and the investors own $5M / $25M = 20% of the "
            "company."
        ),
        pitfalls=(
            "Mixing up pre-money and post-money figures when computing ownership.",
            "Treating a private-round valuation as a market price.",
            "Ignoring preferred terms that make headline valuations look higher.",
        ),
        related=("dilution", "term-sheet", "liquidation-preference"),
        category="startups",
        aliases=("pre-money", "post-money", "premoney", "postmoney", "valuation", "valuations"),
    ),
    Concept(
        slug="liquidation-preference",
        term="Liquidation preference",
        short="An investor's right to be paid back a set amount before common shareholders when a company is sold or wound up.",
        explainer=(
            "Venture investors usually buy preferred shares. A liquidation preference says that if the company "
            "is sold or liquidated, preferred holders receive a set amount, commonly 1x their investment, before "
            "common shareholders such as founders and employees receive anything.\n\n"
            "With a non-participating preference, investors choose either their preference or their "
            "percentage share of the proceeds. With a participating preference they get the preference and "
            "then also share in what remains. Preferences from several rounds can stack, so in a modest sale "
            "common shareholders may receive little even when the headline price sounds large."
        ),
        example=(
            "Hypothetical example: investors put in $10 million with a 1x non-participating preference for "
            "25% of a company that later sells for $30 million. They take the larger of $10 million or 25% of "
            "$30 million ($7.5 million), so they take $10 million and $20 million remains for everyone else."
        ),
        pitfalls=(
            "Assuming sale proceeds are split purely by ownership percentage.",
            "Overlooking participating or multiple (2x, 3x) preferences.",
            "Ignoring how preferences from several rounds stack.",
        ),
        related=("term-sheet", "pre-money-vs-post-money-valuation", "dilution"),
        category="startups",
        aliases=(
            "liquidation preference",
            "liquidation preferences",
            "participating preferred",
            "preferred shares",
        ),
    ),
    # ------------------------------------------------------------------ earnings and filings
    Concept(
        slug="earnings-guidance",
        term="Earnings guidance",
        short="A company's own forecast of future results, such as next quarter's revenue range.",
        explainer=(
            "Many listed companies publish guidance: management's expectations for revenue, margins, or "
            "earnings in a coming period, usually as a range. Guidance appears in earnings releases and on "
            "earnings calls, typically with a warning that actual results may differ.\n\n"
            "Guidance is a forecast, not a fact. Reports should say it is the company's estimate, and when "
            "results arrive, compare them with what the company guided, not only with analysts' estimates. "
            "Companies can also set guidance conservatively so that results look better by comparison.\n\n"
            "Guidance is not part of the tagged XBRL financial data companies file with the SEC; it has to be "
            "read from the release or call transcript itself."
        ),
        example=(
            "Hypothetical example: a company guided third-quarter revenue of $3.25-3.40 billion and reported "
            "$3.42 billion, above the top of its own range. Its new guidance of $3.6-3.8 billion for the next "
            "quarter is a projection and should be presented as one."
        ),
        pitfalls=(
            "Reporting guidance as if it were a result.",
            "Confusing company guidance with analyst consensus estimates.",
            "Ignoring the assumptions and risks the company attaches to its guidance.",
        ),
        related=("eps", "revenue-vs-net-income", "sec-forms-10q-10k-8k"),
        category="earnings",
        aliases=(
            "earnings guidance",
            "revenue guidance",
            "guidance range",
            "forward-looking statements",
            "forward-looking statement",
        ),
        patterns=(
            r"\b(?:earnings|revenue|sales|EPS|margin|profit|financial|annual|full[\s-]year|quarterly|"
            r"(?:first|second|third|fourth)[\s-]quarter)\s+(?:guidance|outlook)\b",
            r"\bguid(?:ed|es|ing)\s+(?:\w+[\s-]){0,3}(?:revenue|earnings|EPS|sales|margins?)\b",
            r"\b(?:raised|lowered|cut|reaffirmed|maintained|withdrew)\s+(?:its\s+)?(?:\w+\s+)?guidance\b",
        ),
    ),
    Concept(
        slug="eps",
        term="Earnings per share (EPS)",
        short="Net income attributable to common shareholders divided by the number of shares; diluted EPS counts potential shares too.",
        explainer=(
            "Earnings per share divides a company's net income available to common shareholders by the "
            "weighted average number of common shares outstanding during the period. Diluted EPS also counts "
            "shares that could be created from stock options, convertible securities, and similar instruments, "
            "so it is usually equal to or lower than basic EPS.\n\n"
            "EPS can rise because profits grew or because the share count fell, for example through buybacks. "
            "Quarterly EPS figures do not simply add up to the annual figure, because share counts change. "
            "Companies also report 'adjusted' EPS that excludes some items; those adjustments are chosen by "
            "management and should be compared with the official (GAAP) figure."
        ),
        example=(
            "Hypothetical example: a company earns $50 million in a quarter and has 100 million shares "
            "outstanding on average, so basic EPS is $0.50. If options could add 5 million shares, diluted EPS "
            "is $50M / 105M = about $0.48."
        ),
        pitfalls=(
            "Comparing adjusted EPS with another company's GAAP EPS.",
            "Ignoring buybacks that raise EPS without higher profits.",
            "Adding quarterly EPS figures to get an annual figure.",
        ),
        related=("revenue-vs-net-income", "dilution", "earnings-guidance"),
        category="earnings",
        aliases=("earnings per share", "per diluted share", "diluted EPS", "EPS"),
    ),
    Concept(
        slug="revenue-vs-net-income",
        term="Revenue vs. net income",
        short="Revenue is the money a company brings in from sales (the top line); net income is what is left after all costs (the bottom line).",
        explainer=(
            "Revenue, or sales, is the total a company earns from its products and services in a period. "
            "Net income, or profit, is what remains after subtracting the cost of goods sold, operating "
            "expenses, interest, and taxes. Between them sit gross profit (revenue minus the direct cost of "
            "goods) and operating income (gross profit minus operating expenses).\n\n"
            "A company can grow revenue quickly while losing money, or shrink revenue while profits rise. "
            "Stories that cite one figure should say which it is, and comparisons are clearest against the "
            "same period a year earlier, because many businesses are seasonal."
        ),
        example=(
            "Hypothetical example: a startup's revenue grows from $10 million to $15 million in a year, a 50% "
            "increase, but its costs grow from $14 million to $21 million. Its net loss widens from $4 million "
            "to $6 million despite the revenue growth."
        ),
        pitfalls=(
            "Calling revenue 'profit' or 'earnings'.",
            "Comparing a seasonal quarter with the previous quarter instead of the same quarter a year earlier.",
            "Ignoring one-off items that swing net income.",
        ),
        related=("operating-margin", "eps", "earnings-guidance"),
        category="earnings",
        aliases=(
            "net income",
            "net loss",
            "net profit",
            "bottom line",
            "top line",
            "gross profit",
            "profitable",
            "profitability",
            "year over year",
            "year-over-year",
        ),
    ),
    Concept(
        slug="operating-margin",
        term="Operating margin (and gross margin)",
        short="Operating income as a share of revenue; gross margin is gross profit as a share of revenue.",
        explainer=(
            "Margins express profit as a percentage of revenue. Gross margin (revenue minus the direct cost of "
            "what was sold, divided by revenue) shows how much each sale contributes before overheads. "
            "Operating margin (operating income divided by revenue) also subtracts research, sales, and "
            "administrative costs, showing how profitable the core business is before interest and taxes.\n\n"
            "Margins make companies of different sizes easier to compare, but norms differ widely across "
            "industries: software and chip design often have high gross margins, retailers low ones. A falling "
            "margin can reflect price cuts, rising costs, or deliberate investment."
        ),
        example=(
            "Hypothetical example: a company has revenue of $200 million, cost of goods of $120 million, and "
            "operating expenses of $50 million. Gross margin is (200 - 120) / 200 = 40%, and operating margin "
            "is (200 - 120 - 50) / 200 = 15%."
        ),
        pitfalls=(
            "Comparing margins across industries with different cost structures.",
            "Confusing percentage-point changes with percentage changes.",
            "Ignoring one-time charges inside operating income.",
        ),
        related=("revenue-vs-net-income", "eps"),
        category="earnings",
        aliases=(
            "operating margin",
            "operating margins",
            "operating income",
            "operating profit",
            "operating loss",
            "gross margin",
            "gross margins",
        ),
    ),
    Concept(
        slug="sec-forms-10q-10k-8k",
        term="Forms 10-Q, 10-K and 8-K",
        short="The main SEC reports: quarterly (10-Q), annual (10-K), and current reports on material events (8-K).",
        explainer=(
            "US-listed companies file periodic reports with the Securities and Exchange Commission. The 10-K is "
            "the annual report, with audited financial statements, a description of the business, and risk "
            "factors. The 10-Q covers each of the first three fiscal quarters, with financial statements that "
            "are reviewed but not audited. Deadlines depend on company size: roughly 60 to 90 days after the "
            "fiscal year for a 10-K and 40 to 45 days after the quarter for a 10-Q.\n\n"
            "An 8-K reports material events between periodic reports, generally within four business days: "
            "for example a merger agreement, a change of executives, or results of operations (Item 2.02), "
            "which is how many companies furnish their earnings press releases.\n\n"
            "Filings are public on SEC EDGAR, which makes them the primary source for figures in earnings "
            "coverage. Press releases and calls add context and guidance, but the filing is the record."
        ),
        example=(
            "Hypothetical example: a company with a December fiscal year publishes an earnings press release "
            "on an 8-K in early February, files its 10-K with audited annual results in late February, and "
            "files 10-Qs after its March, June, and September quarters."
        ),
        pitfalls=(
            "Assuming a press release contains everything in the filing.",
            "Forgetting that 10-Q figures are unaudited.",
            "Missing amendments (10-K/A, 10-Q/A) that correct earlier filings.",
        ),
        related=("xbrl", "earnings-guidance", "eps"),
        category="earnings",
        aliases=(
            "10-Q",
            "10-K",
            "8-K",
            "Form 10-Q",
            "Form 10-K",
            "Form 8-K",
            "SEC filing",
            "SEC filings",
            "EDGAR",
        ),
    ),
    Concept(
        slug="xbrl",
        term="XBRL",
        short="eXtensible Business Reporting Language: machine-readable tags on the numbers in company filings.",
        explainer=(
            "XBRL labels each number in a filing with a standard concept (for example us-gaap:Revenues or "
            "us-gaap:NetIncomeLoss), the period it covers, and its unit. The SEC requires financial statements "
            "in 10-Qs and 10-Ks to be tagged this way, which lets anyone retrieve reported figures directly "
            "from EDGAR's data APIs.\n\n"
            "Tagged data still needs care. Companies choose among similar tags and change them over time, may "
            "use custom tags, and repeat prior-period comparatives in later filings (sometimes restated). "
            "Quarterly figures are not always tagged separately in annual reports. And guidance, commentary, "
            "and non-GAAP measures are generally not in the tagged financial statements. The Newsren cites "
            "the tag, form, filing date, and period for every XBRL figure it shows."
        ),
        example=(
            "Hypothetical example: a 10-Q tags revenue of $20.5 million as us-gaap:Revenues for the period "
            "April 1 to June 30, and also tags the prior year's figure for the same quarter as a comparative. "
            "A data tool can read both from the same filing."
        ),
        pitfalls=(
            "Assuming all companies use the same tag for the same idea.",
            "Ignoring restated comparatives in later filings.",
            "Treating XBRL data as real-time; it arrives with the filing.",
        ),
        related=("sec-forms-10q-10k-8k", "revenue-vs-net-income"),
        category="earnings",
        aliases=("XBRL", "Inline XBRL", "iXBRL"),
    ),
    # ------------------------------------------------------------------ rules for AI in finance
    Concept(
        slug="ai-washing",
        term="AI washing",
        short="Overstating or misrepresenting how much a product or firm actually uses artificial intelligence.",
        explainer=(
            "'AI washing' describes marketing that claims AI capabilities a company does not have or does not "
            "use as described. For investment firms this is not only a reputational problem: statements to "
            "clients and investors about how money is managed are subject to anti-fraud and marketing rules.\n\n"
            "In March 2024 the SEC settled charges against two investment advisers, Delphia and Global "
            "Predictions, for false and misleading statements about their use of AI; they agreed to civil "
            "penalties of $225,000 and $175,000 respectively.\n\n"
            "When a firm claims an 'AI-powered' process, useful questions are what the AI actually does, "
            "whether it is in production or a pilot, and whether any independent party has checked the claim."
        ),
        example=(
            "Hypothetical example: an adviser's website says portfolios are 'run by autonomous AI agents', "
            "but staff actually rebalance them quarterly with a spreadsheet. If a regulator finds the claim "
            "false, the firm can face penalties regardless of how its portfolios performed."
        ),
        pitfalls=(
            "Taking 'AI-powered' marketing at face value.",
            "Confusing a pilot or research project with a deployed system.",
            "Assuming AI involvement says anything about investment quality.",
        ),
        related=("reg-bi-fiduciary-duty", "investment-adviser-registration", "backtesting"),
        category="regulation",
        aliases=("AI washing", "AI-washing", "AI washed"),
    ),
    Concept(
        slug="reg-bi-fiduciary-duty",
        term="Regulation Best Interest and fiduciary duty",
        short="The conduct standards for brokers (Reg BI) and investment advisers (fiduciary duty) when they make recommendations.",
        explainer=(
            "In the United States, broker-dealers recommending securities or strategies to retail customers "
            "must follow the SEC's Regulation Best Interest (Reg BI), in force since June 2020. It requires "
            "them to act in the customer's best interest when recommending, and sets disclosure, care, "
            "conflict-of-interest, and compliance obligations.\n\n"
            "Registered investment advisers owe their clients a fiduciary duty under the Investment Advisers "
            "Act of 1940, made up of a duty of care and a duty of loyalty, which applies to the whole "
            "relationship.\n\n"
            "These obligations do not disappear when a recommendation is produced or influenced by AI. Firms "
            "remain responsible for supervising the tools they use, and regulators have said existing rules "
            "apply to AI-assisted activity."
        ),
        example=(
            "Hypothetical example: a brokerage's AI assistant suggests a high-fee product to a retiree when a "
            "cheaper, similar option exists. Under Reg BI, the firm, not the software, is accountable for "
            "whether that recommendation was in the customer's best interest."
        ),
        pitfalls=(
            "Assuming AI-generated recommendations fall outside existing rules.",
            "Confusing the broker standard (Reg BI) with the adviser standard (fiduciary duty).",
            "Treating general education as if it were a personalised recommendation, or vice versa.",
        ),
        related=("investment-adviser-registration", "ai-washing"),
        category="regulation",
        aliases=(
            "Regulation Best Interest",
            "Reg BI",
            "fiduciary duty",
            "fiduciary duties",
            "fiduciary",
            "best interest obligation",
            "best interest obligations",
            "broker-dealer",
            "broker-dealers",
        ),
    ),
    Concept(
        slug="investment-adviser-registration",
        term="Investment adviser registration",
        short="Who must register as an investment adviser, and why personalised advice for pay matters.",
        explainer=(
            "Under the Investment Advisers Act of 1940, a person or firm that, for compensation, is in the "
            "business of advising others about securities is generally an investment adviser and must "
            "register with the SEC or a state regulator, depending mainly on its size. Registered advisers "
            "owe fiduciary duties and face disclosure, recordkeeping, and examination requirements.\n\n"
            "There are exclusions, notably for publishers of bona fide news and educational content of general "
            "and regular circulation that is not tailored to individual readers. Legal commentators note that "
            "an AI tool giving personalised securities advice for compensation can raise adviser registration "
            "questions, and a tool that places trades for users can raise broker-dealer questions.\n\n"
            "This is why The Newsren's finance section offers education and news only: no personalised advice, "
            "buy/sell/hold calls, price targets, portfolio recommendations, or trade execution."
        ),
        example=(
            "Hypothetical example: a newsletter explaining how bond prices react to interest rates is general "
            "education. An app that charges a fee and tells each user which funds to buy based on their own "
            "finances is giving personalised advice and may need to register."
        ),
        pitfalls=(
            "Assuming software is exempt because no human gives the advice.",
            "Blurring general education and personalised recommendations.",
            "Treating this explainer as legal advice; firms consult counsel.",
        ),
        related=("reg-bi-fiduciary-duty", "ai-washing"),
        category="regulation",
        aliases=(
            "investment adviser",
            "investment advisers",
            "investment advisor",
            "investment advisors",
            "registered investment adviser",
            "Investment Advisers Act",
            "Advisers Act",
            "RIA",
        ),
    ),
]

_BY_SLUG: dict[str, Concept] = {c.slug: c for c in CONCEPTS}
_BY_TERM: dict[str, Concept] = {slugify(c.term): c for c in CONCEPTS}

# Short or alternative slugs readers and other modules use.
SLUG_ALIASES: dict[str, str] = {
    "backtest": "backtesting",
    "in-sample": "in-sample-vs-out-of-sample",
    "out-of-sample": "in-sample-vs-out-of-sample",
    "sharpe": "sharpe-ratio",
    "drawdown": "maximum-drawdown",
    "max-drawdown": "maximum-drawdown",
    "60-40": "benchmark",
    "costs": "transaction-costs",
    "guidance": "earnings-guidance",
    "earnings-per-share": "eps",
    "revenue-vs-profit": "revenue-vs-net-income",
    "revenue": "revenue-vs-net-income",
    "net-income": "revenue-vs-net-income",
    "gross-margin": "operating-margin",
    "10-q": "sec-forms-10q-10k-8k",
    "10-k": "sec-forms-10q-10k-8k",
    "8-k": "sec-forms-10q-10k-8k",
    "sec-filings": "sec-forms-10q-10k-8k",
    "pre-money": "pre-money-vs-post-money-valuation",
    "post-money": "pre-money-vs-post-money-valuation",
    "valuation": "pre-money-vs-post-money-valuation",
    "reg-bi": "reg-bi-fiduciary-duty",
    "fiduciary-duty": "reg-bi-fiduciary-duty",
    "investment-adviser": "investment-adviser-registration",
    "series-a": "funding-rounds",
    "seed": "funding-rounds",
    "simulated": "paper-trading",
}


def get_concept(slug: str) -> Concept | None:
    """Look up a concept by slug, a short alias ("drawdown", "guidance"), or its term."""
    if not slug:
        return None
    key = slugify(slug)
    return _BY_SLUG.get(key) or _BY_SLUG.get(SLUG_ALIASES.get(key, "")) or _BY_TERM.get(key)


def concepts_by_category() -> dict[str, list[Concept]]:
    """Concepts grouped by category key, in ``CATEGORY_LABELS`` order."""
    grouped: dict[str, list[Concept]] = {key: [] for key in CATEGORY_LABELS}
    for concept in CONCEPTS:
        grouped.setdefault(concept.category, []).append(concept)
    return grouped


def related_concepts(concept: Concept) -> list[Concept]:
    """Resolved related concepts (unknown slugs are skipped)."""
    return [c for slug in concept.related if (c := get_concept(slug)) is not None]


def _phrase_regex(phrase: str) -> tuple[str, bool]:
    """Regex for a literal alias; all-caps acronyms (EPS, XBRL, RIA) match case-sensitively."""
    tokens = [re.escape(t) for t in re.split(r"[\s\-]+", phrase.strip()) if t]
    body = r"[\s\-]*".join(tokens)
    letters = [ch for ch in phrase if ch.isalpha()]
    case_sensitive = sum(ch.isupper() for ch in letters) >= 2 and not any(ch.islower() for ch in letters)
    start = r"(?<![\w-])" if phrase[:1].isalnum() else ""
    end = r"(?![\w-])" if phrase[-1:].isalnum() else ""
    return f"{start}{body}{end}", case_sensitive


@cache
def _matchers() -> tuple[tuple[Concept, re.Pattern[str] | None, re.Pattern[str] | None], ...]:
    out = []
    for concept in CONCEPTS:
        insensitive: list[str] = list(concept.patterns)
        sensitive: list[str] = []
        for alias in concept.aliases or (concept.term,):
            regex, case_sensitive = _phrase_regex(alias)
            (sensitive if case_sensitive else insensitive).append(regex)
        out.append(
            (
                concept,
                re.compile("|".join(f"(?:{p})" for p in insensitive), re.IGNORECASE) if insensitive else None,
                re.compile("|".join(f"(?:{p})" for p in sensitive)) if sensitive else None,
            )
        )
    return tuple(out)


def concept_mentions(text: str | None) -> list[tuple[Concept, str]]:
    """(concept, first matched phrase) pairs in order of first appearance in ``text``."""
    if not text:
        return []
    found: list[tuple[int, Concept, str]] = []
    for concept, insensitive, sensitive in _matchers():
        hits = [m for rx in (insensitive, sensitive) if rx is not None and (m := rx.search(text))]
        if hits:
            first = min(hits, key=lambda m: m.start())
            found.append((first.start(), concept, first.group(0)))
    found.sort(key=lambda item: item[0])
    return [(concept, phrase) for _, concept, phrase in found]


def concepts_in_text(text: str | None, limit: int | None = None) -> list[Concept]:
    """Concepts mentioned in ``text`` (word-boundary matches on terms and synonyms), in order of first mention."""
    concepts = [concept for concept, _ in concept_mentions(text)]
    return concepts[:limit] if limit is not None else concepts
