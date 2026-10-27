# Plan traceability

How each part of *The Newsren: Agentic News Intelligence Plan* (October 2026)
is implemented, where readers and editors find it, and which tests cover it.

## §1 Product vision — the news intelligence workflow

| Plan element | Implementation | UI | Tests |
|---|---|---|---|
| Sources: news, research papers, policy, filings, earnings, regulatory releases, education reports | `ingest/data/sources.json` (27 verified approved sources), `ingest/fetchers.py` (RSS/Atom, arXiv API, SEC EDGAR Atom, ERIC API) | `/sources` | `test_ingest_parsers.py`, `test_ingest_sources.py` |
| Discover: monitor sources | `ingest/pipeline.py::ingest_all`, `pipeline.py::run_pipeline`, `scheduler.py`, `newsren pipeline` | `/editor/pipeline` | `test_ingest_pipeline.py` |
| Verify: cross-check claims | `agents/brief.py` (support status: supported / corroborated / contested / unverified), `agents/evidence.py`, `editorial/checks.py` | brief claims, "Inspect evidence" | `test_brief_generation.py`, `test_evidence_inspection.py`, `test_editorial_checks.py` |
| Connect: track developments | `intelligence/clustering.py`, `intelligence/timeline.py` | `/stories/{id}`, "Connected stories" | `test_intelligence_*.py` |
| Output: briefings, explainers, timelines, alerts, research suggestions | `agents/briefing.py`, `agents/learning.py`, `agents/monitor.py`, `agents/products.py` | `/me`, tools on `/briefs/{id}` | `test_briefing_build.py`, `test_learning_*.py`, `test_monitor_follow.py`, `test_products_*.py` |
| Reader loop: ask → agent gathers evidence → sourced answer | `agents/research.py` + `agents/tools.py` (budgeted tool loop) | `/ask`, "Ask about this story" | `test_research_ask.py`, `test_tools_research.py`, `test_web_tools.py` |
| "Merely calling an LLM to generate a summary does not make a feature agentic" | Agents have a defined goal, approved tools only, and a controlled evidence process: `agents/base.py::Orchestrator` (budgets, run records), evidence packs (agents may only cite what they were given), `llm/claude.py` manual tool loop | `/editor/runs` (tool traces) | `test_research_ask.py` (fake-Claude tool loop + budget exhaustion), `test_brief_claude.py` |

## §2 Six agentic features

| Feature | Implementation | UI | Tests |
|---|---|---|---|
| 2.1 My Newsren (highest priority): topics, questions, organizations, technical depth; personalized briefing explaining what changed and why it matters | `users/service.py` (interests, preferences), `agents/briefing.py::build_briefing` (relevance × recency × novelty, depth-aware, "when meaningful" frequency), `delivery/email.py` | `/me`, `/me/briefings/{id}`, opt-in email | `test_users_service.py`, `test_briefing_build.py`, `test_delivery_email.py`, `test_web_account.py` |
| 2.2 Story Intelligence: evolving stories, what happened first / new / disputed / evidence changes | `intelligence/clustering.py` (TF-IDF + entities + headline anchors, filings by company), `intelligence/timeline.py::refresh_story` (story-intelligence agent), `related_stories` (cross-vertical) | `/stories/{id}`, "Connected stories" (merged storyline: model announcement → evaluation → product launch → school adoption → regulation) | `test_intelligence_clustering*.py`, `test_intelligence_anchors.py`, `test_intelligence_timeline.py` |
| 2.3 News-to-Learning: explainers, glossary, quizzes, discussion questions, classroom activities for students / educators / professionals | `agents/learning.py` (comprehension + transfer quiz, educator activities, professional implications, advanced reading) | "Learn this topic" + "Explain for" selector | `test_learning_materials.py`, `test_learning_quiz.py` |
| 2.4 Industry Impact: implications for industries, occupations, decisions; supported vs speculative | `agents/impact.py` (effects labelled supported / speculative / scenario) | "Industry impact" (custom questions) | `test_impact_analysis.py` |
| 2.5 Evidence and Bias Inspection: original sources, cross-source comparison, uncertainty, facts vs interpretations; no "biased" label for a different viewpoint | `agents/evidence.py` (source roles, claim comparisons, framing notes, study design: participants, outcome measures, comparison groups, transfer vs speed) | "Inspect evidence" | `test_evidence_inspection.py` (incl. framing notes never label a source "biased") |
| 2.6 Follow This Story: notify only when an event satisfies the reader's criteria | `users/service.py::create_follow` (criteria derived from the question), `agents/monitor.py::check_follow` (event types, independence, primary, opinion excluded, custom criterion agent, near-duplicate / same-development dedup) | "Follow developments", `/me` alerts with useful / not useful feedback | `test_monitor_follow.py` (the plan's "independent benchmark, not another opinion article" example), `test_web_account.py` |

## §3 Verticals and editorial products

| Vertical / product | Implementation | Tests |
|---|---|---|
| Technology: vendor claims vs third-party evaluations, technical trade-offs, capability evolution | brief sections "What happened? / What is technically new? / What evidence exists? / What should developers watch next?" (`agents/brief.py`) | `test_brief_generation.py` |
| Launch Reality Check, Research Paper Decoder, Technology Evolution Timeline | `agents/products.py` | `test_products_outputs.py` |
| Education: policy, curriculum, AI literacy, research on learning; age-appropriate explainers | education brief sections; learning agent | `test_brief_generation.py`, `test_learning_materials.py` |
| Policy Impact Explorer, Classroom Discussion Builder, Education Evidence Checker | `agents/products.py` | `test_products_outputs.py` |
| Industry: companies, supply chains, scenarios labelled as such, questions to investigate | industry brief sections; impact agent | `test_impact_analysis.py` |
| Company and Competitor Watch, Supply-Chain Ripple Map (every section a hypothesis), Skills and Career Trends | `agents/products.py` | `test_products_outputs.py` |
| Finance: Earnings Decoder (every number quoted from the filing), Funding Round Tracker (no ranking), AI-in-Finance Reality Check, Regulation Watch | `finance/edgar.py`, `finance/funding.py`, `finance/ai_claims.py`, `finance/regulation.py`, `agents/products.py` | `test_finance_*.py`, `test_products_finance.py`, `test_web_finance.py` |
| Illustrative use cases (model launch, AI-learning study, semiconductor capacity, bank backtest) | fictional demo corpus `ingest/data/demo_articles.json` covers each one end to end | `test_ingest_fixtures.py` |

## §4 Finance section: research basis, features, guardrails

| Requirement | Implementation | Tests |
|---|---|---|
| What agentic AI is doing in finance (FINRA, Deutsche Bank/Google Cloud, Goldman, TradingAgents, JPMorgan LLM Suite and 2026 backtest, vendor surveys, Metal Raise Agent, SEC AI-washing) | `finance/ai_claims.py::SEED_CLAIMS` with the plan's sources; `/finance/ai-reality-check` | `test_finance_data_ai_claims.py` |
| Earnings and filings brief vs guidance and prior period, each figure linked | `finance/edgar.py` (XBRL company facts → quarterly metrics with filing links and tags), `earnings_brief` | `test_finance_data_edgar.py`, `test_finance_briefs_earnings.py` |
| Startup and VC tracker framed as news and learning | `finance/funding.py::funding_trends` (counts/sums only), `/finance/funding` ("not an investment ranking") | `test_finance_data_funding*.py` |
| AI-in-finance reality check: announcement / pilot / backtest / live deployment / regulatory finding | `finance/ai_claims.py::classify_claim` | `test_finance_data_ai_claims*.py` |
| Finance concepts for learners linked to stories | `finance/concepts.py` (24 concepts), `/finance/concepts/{slug}` lists stories using each concept | `test_finance_data_concepts.py`, `test_web_finance.py` |
| Follow a company or sector: filings, earnings, funding, enforcement — not price moves | company/sector follows (`/finance` form → `POST /follows`), monitor event types | `test_web_finance.py`, `test_monitor_follow.py` |
| No buy/sell/hold, price targets, portfolio recommendations, personalized advice, trade execution | `editorial/guardrails.py::lint_text` (blocker in the quality gate), `detect_advice_request` + `ADVICE_REFUSAL` in Ask Newsren; no brokerage integration exists | `test_editorial_guardrails.py`, `test_research_ask.py`, `test_web_finance.py` (every finance page lint-clean) |
| Notice on every finance page | `FINANCE_DISCLAIMER` via `m.finance_notice` on every finance route, finance briefs, finance products, briefings with finance items | `test_web_finance.py`, `test_web_public.py`, `test_web_tools.py` |
| Label every performance claim backtest / simulated / live with period, benchmark, source | `ClaimOut.performance_label/period/benchmark`, `editorial/guardrails.py::performance_claim_issues`, `checks.py` | `test_editorial_checks.py` |
| Never present forecasts as facts; never say an AI agent can reliably beat the market | guardrail lint patterns; claim type `forecast` requires hedged wording | `test_editorial_guardrails.py` |
| Licensed or public data only; timestamp market data and note delays | registry license notes; EDGAR `data_delay_note`; retrieval timestamps on every citation | `test_ingest_sources.py`, `test_finance_data_edgar.py` |
| Human editorial review for any finance story | `editorial/workflow.py::submit_for_review` (finance always requires review regardless of settings) | `test_editorial_workflow.py` |

## §5 User experience

| Requirement | Where |
|---|---|
| Recognizable editorial homepage; reading stays central | `/` (lead story, per-vertical sections, developing stories) — `templates/public/home.html` |
| Navigation: Technology, Education, Industry, Finance | masthead nav in `templates/base.html` |
| Agent entry point ("Understand the next technology breakthrough…") | homepage + `/ask` + per-vertical prompts |
| Audience selector "Explain for" Student / Educator / Professional | masthead switch (`POST /audience`, cookie + profile), Learn tool tabs |
| Four tools: Connected stories, Learn this topic, Inspect evidence, Follow developments | tool tab bar on every brief (plus Industry impact and Ask about this story), htmx-loaded with no-JS fallbacks |
| Not an intrusive chatbot | the agent appears only via the entry point and tools — `test_web_public.py`, `test_web_smoke.py` |

## §6 Technical architecture and critical requirements

| Requirement | Implementation | Tests |
|---|---|---|
| Orchestrator: selects task, coordinates tools, enforces time and budget limits | `agents/base.py::Orchestrator`, `llm/base.py::Budget`, `ToolRunner` | `test_research_ask.py` (budget), `test_metrics_compute.py` (run accounting) |
| Source discovery / evidence verification / story intelligence / news explanation / editorial quality gate / output | see §1–§2 rows | — |
| Every generated item retains sources, publication timestamps, retrieval time, and supporting evidence; describe disagreements | `CitationRef` (url, publisher, published_at, retrieved_at, snippet, tier, doc type) on briefs, claims, generated content, answers, alerts, briefings; `Brief.disagreements` | `test_editorial_checks.py` (citation coverage, contested-without-disagreement) |
| Preprints vs peer review; avoid overstating causality; label forecasts/scenarios; identify studied populations | `editorial/checks.py` (preprint, forecast, study-population checks), brief and product prompts/builders | `test_editorial_checks.py`, `test_brief_generation.py` |
| Restrict agents to approved source APIs and retrieval tools | `ingest/sources.py::is_approved_url`, `agents/tools.py` (archive, article, arXiv, SEC EDGAR, approved-domain fetch only) | `test_tools_research.py` |
| Human approval for publishing; automatic delivery only for low-risk personal briefings | `editorial/workflow.py`, `NEWSREN_REQUIRE_HUMAN_APPROVAL`, `agents/briefing.py::deliver_briefing` | `test_editorial_workflow.py`, `test_briefing_build.py` |
| No unrestricted browsing, unapproved scraping, social posting, agent-triggered financial or educational decisions | no such tools exist; fetch tool refuses unapproved domains | `test_tools_research.py` |

## §7 MVP and build order

| Stage | Delivered |
|---|---|
| 1 Foundation: approved feeds, dedup, four categories, citations, AI summaries | ✔ ingestion, SimHash/URL/content dedup, four verticals, citations everywhere, AI Story Briefs |
| 2 Ask and explain: "Ask about this story", source-based Q&A, student/professional modes | ✔ Ask Newsren (story-seeded and global), News-to-Learning for three audiences |
| 3 Connect and follow: timelines, topic tracking, change detection | ✔ story clustering and timelines, follows with criteria, `story_changes_since`, alert dedup |
| 4 Briefings and depth: email briefings, lesson builder, industry impact, finance trackers, analytics | ✔ opt-in email briefings, learning materials + classroom builder, impact reports, earnings/funding/AI-claims trackers, `/analytics` |
| "Build three features first": AI Story Brief + Ask Newsren; News-to-Learning; Follow a Topic | ✔ all three, forming the loop discover → understand → investigate → follow up |

## §8 Positioning

Basic discovery and explainers are free and need no account; accounts add
saved follows, personal briefings, and email delivery. Team workspaces and
exportable reports are left for later testing, as the plan suggests; the JSON
API (`/api/v1`) is the basis for exports.

## §9 Success metrics

| Metric | Implementation (`metrics.py`) | UI |
|---|---|---|
| Trust: sampled claims accurately supported; material corrections | editor claim verdicts (`/editor/briefs/{id}`), corrections | `/analytics` |
| Usefulness: did the briefing answer the question / save time | `POST /feedback` on answers, briefings, learning, impact, evidence | `/analytics` |
| Education: comprehension and transfer vs ordinary summaries | `QuizAttempt` (condition `newsren` vs `summary`; comprehension vs transfer questions) | `/analytics` |
| Engagement: return visits to followed stories; useful vs unwanted alerts | `PageView` (signed-in readers only), alert feedback | `/analytics` |
| Editorial efficiency: time saved per approved briefing incl. review cost | review durations from `EditorialAction`, estimated time saved | `/analytics` |
| Cost: inference and retrieval cost per story, query, monitoring task | `AgentRun` tokens/cost per agent | `/analytics`, `/editor/runs` |

Article volume and chatbot usage are intentionally not reported as primary metrics.
