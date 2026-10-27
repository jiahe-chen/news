# Module contracts

Public APIs every module exposes. Modules call each other **only** through
these functions. All functions take an SQLAlchemy `Session` as first argument,
`flush()` but never `commit()` (callers own transactions), and must work in
offline LLM mode (`NEWSREN_LLM=offline`), which is what tests use.

Foundation (already implemented — do not change signatures; additive changes
only if strictly needed and reported):

| Module | Provides |
|---|---|
| `newsren.config` | `Settings`, `get_settings()`, `set_settings()` |
| `newsren.db` | `init_db()`, `new_session()`, `session_scope()`, `search_article_ids(session, query, limit, mode)`, `fts_query()` |
| `newsren.models` | All ORM models + vocab enums (`Vertical`, `DocType`, `TrustTier`, `EventType`, `Audience`, `BriefStatus`, `ClaimType`, `SupportStatus`, `PerformanceLabel`, `AIFinanceClass`, `Role`, ...) |
| `newsren.schemas` | Pydantic agent output contracts (`BriefOut`, `EvidenceReportOut`, `ResearchAnswerOut`, `LearningOut`, `ImpactOut`, `ProductOut`, `StoryUpdateOut`, `FollowCriteria`, `MonitorDecisionOut`, `BriefingOut`, finance shapes, `CheckResult`, `CitationRef`, `EvidenceDoc`) |
| `newsren.llm` | `StructuredTask`, `ToolTask`, `Tool`, `ToolResult`, `ToolRunner`, `Budget`, `get_provider()`, `set_provider()`, `OfflineProvider`; `newsren.llm.claude.ClaudeProvider` |
| `newsren.agents.base` | `Orchestrator(session).run(agent_name, task, subject_type=, subject_id=, budget=)` → `AgentOutcome(output, run, result, generated_by)`; `build_evidence_pack(articles)`, `render_evidence(docs)`, `citations_of(docs)`, `valid_citation_ids(docs)`, `citation_for(article, cid)`, `article_text(article)`, `get_or_create_content(...)`, `cache_key(...)` |
| `newsren.textutil` | `tokenize`, `top_keywords`, `tfidf_vectors`, `cosine`, `jaccard`, `split_sentences`, `extractive_summary`, `best_snippet`, `canonical_url`, `domain_of`, `content_hash`, `simhash_hex`, `hamming`, `parse_datetime`, `iso`, `slugify`, `strip_html`, `truncate` |

Every agent call goes through `Orchestrator.run` (never call a provider
directly) and every agent supplies an `offline` builder that produces a
genuinely useful, evidence-grounded result from the same inputs (extractive,
rule-based) — not placeholder text. Offline output must only cite citation
ids that exist in the evidence pack.

---

## 1. Ingestion — `newsren.ingest` (Discover)

`newsren/ingest/sources.py`
- `APPROVED_SOURCES_PATH` → `newsren/ingest/data/sources.json` (registry of approved sources: slug, name, kind, url, homepage, vertical, trust_tier, default_doc_type, license_note, data_delay_note).
- `sync_sources(session) -> list[Source]` — upsert registry rows into `sources`.
- `is_approved_url(session, url) -> bool` — True only if the URL's domain belongs to an approved, active source (homepage/url domain) or an allow-listed primary domain (sec.gov, arxiv.org, federalreserve.gov, finra.org, ies.ed.gov ...). Used by agents' fetch tool.

`newsren/ingest/fetchers.py`
- `FetchedItem` dataclass: url, title, summary, content, author, published_at (datetime|None), doc_type (optional override), extra (dict).
- `fetch_source(source: Source, settings) -> list[FetchedItem]` — dispatch on `source.kind` (rss | arxiv | edgar | eric). Uses `httpx` with `settings.user_agent`, timeout, and the max-items limit. Never raises for network errors: raises `FetchError` which the pipeline records on `source.last_status`.
- Parsers are pure functions on bytes/str so tests run on saved samples: `parse_rss(raw, source)`, `parse_arxiv(raw)`, `parse_edgar_atom(raw)`, `parse_eric(json)`.

`newsren/ingest/dedup.py`
- `find_duplicate(session, canonical_url, content_hash, simhash) -> Article | None` — exact URL / exact content / SimHash Hamming distance ≤ 3 among articles from the last 14 days.

`newsren/ingest/pipeline.py`
- `IngestReport` dataclass: sources_ok, sources_failed, new_articles, duplicates, errors (list[str]).
- `store_item(session, source, item) -> tuple[Article, bool]` — normalize, classify doc_type/trust tier (item override > source default; title cues like "opinion", "preprint"), dedupe (duplicates are stored with `duplicate_of_id` set, created=False semantics documented), return (article, is_new).
- `ingest_all(session, settings=None, source_slugs=None) -> IngestReport`.

`newsren/ingest/fixtures.py`
- `load_demo_data(session) -> int` — loads `newsren/ingest/data/demo_articles.json`: clearly fictional demo articles (`is_demo=True`, sources `kind="demo"`, `example` domains) covering all four verticals and the plan's use cases (model launch + independent benchmark + school adoption + regulation storyline; AI-learning study press release vs. study; semiconductor capacity announcement; bank backtest vs. live; funding rounds; earnings release; SEC/FINRA actions; education policy). Idempotent. ≥ 40 articles, several multi-article storylines spanning days, at least one opinion piece per storyline, near-duplicate syndicated copies to exercise dedup.

## 2. Story Intelligence — `newsren.intelligence` (Connect)

`newsren/intelligence/entities.py`
- `extract_entities(text) -> list[tuple[str, str]]` (name, kind) — heuristic: known-org gazetteer + capitalized n-grams + tickers ($ABC / NYSE: ABC).
- `tag_article_entities(session, article) -> list[Entity]`.

`newsren/intelligence/events.py`
- `classify_event(article) -> str` — `EventType` value by rules (doc_type + title/summary cues: "benchmark", "evaluation", "launch", "funding", "raises", "10-Q", "enforcement", "charged", "correction", "advisory", "opinion" ...). Pure and fast; used by monitoring and timelines.

`newsren/intelligence/clustering.py`
- `assign_stories(session, articles=None, window_days=21, threshold=0.28) -> list[Story]` — incremental clustering: non-duplicate articles without `story_id` join the best-matching existing story (TF-IDF cosine on title+summary combined with entity Jaccard, same vertical preferred, recency window) or start a new story. Returns stories that changed.
- `related_stories(session, story, limit=5) -> list[Story]` — cross-story connections via shared entities/keywords (across verticals).

`newsren/intelligence/timeline.py`
- `refresh_story(session, story, orchestrator=None, force=False) -> bool` — if membership changed (`member_fingerprint`), rebuild `StoryEvent` rows (one per member article, `classify_event`), and run the **story intelligence agent** (`StructuredTask[StoryUpdateOut]`, agent name `"story_intelligence"`) to set title, what_happened_first, whats_new, disputed, evidence_changes; bump `version`. Returns True if changed.
- `story_changes_since(story, since: datetime) -> list[StoryEvent]` — change detection helper for alerts/briefings.

## 3. Brief, Evidence, Editorial — `newsren.agents.brief`, `newsren.agents.evidence`, `newsren.editorial` (Explain, Verify, Gate)

`newsren/agents/brief.py`
- `generate_brief(session, story: Story, orchestrator=None, regenerate=False) -> Brief` — AI Story Brief for a story (agent `"story_brief"`): evidence pack from the story's articles (primary/independent sources first, max ~8), `BriefOut` → `Brief` + `BriefClaim` rows, citations stored as `CitationRef` dicts. Vertical-specific expandable sections (technology: What happened? / What is technically new? / What evidence exists? / What should developers watch next?; education: who is affected / what the evidence shows / what remains uncertain; industry: what happened / who is connected / scenarios (labelled) / questions to investigate; finance: what was reported / versus prior period or guidance / what it means / concepts). Then runs `editorial.workflow.submit_for_review`. Returns brief.
- `brief_for_story(session, story) -> Brief | None` — latest brief.

`newsren/agents/evidence.py`
- `inspect_evidence(session, brief: Brief, orchestrator=None, refresh=False) -> GeneratedContent` (kind `"evidence"`, content = `EvidenceReportOut`): original sources vs. secondary, claim-by-claim comparison across independent reporting, facts vs. interpretations, uncertainty, missing evidence, framing notes that never label a source "biased" for its viewpoint; study design block when a research/education study is involved.

`newsren/editorial/guardrails.py`
- `FINANCE_DISCLAIMER: str`
- `lint_text(text, vertical) -> list[str]` — violations: buy/sell/hold calls, price targets, personalized advice, "guaranteed returns", "beat the market" claims, forecasts stated as facts.
- `detect_advice_request(question) -> bool` — "should I buy/sell", "is X a good investment", "what stock", "how should I invest my".
- `ADVICE_REFUSAL: str` — educational redirect text.
- `performance_claim_issues(claim: BriefClaim | ClaimOut) -> list[str]` — performance claims missing label/period/benchmark/source.

`newsren/editorial/checks.py`
- `run_checks(session, brief) -> list[CheckResult]` — citation coverage (every fact/performance claim cites ≥1 id), unknown citation ids, sources approved (domain), links well-formed, retrieval timestamps present, licensing note present for sources, disagreement described when claims contested, preprint labelling (research_preprint cited → text mentions preprint/not peer reviewed), forecast labelling, finance guardrail lint, performance labels, high-impact claim flag. Severities: blocker/warning/info.
- `assess_risk(brief, checks) -> str` — high for finance or any high-impact claim or blocker; medium for warnings; else low.

`newsren/editorial/workflow.py`
- `submit_for_review(session, brief, settings=None) -> Brief` — runs checks, sets `check_results`, `risk_level`, `requires_human_review` (always True for finance, high risk, or when `settings.require_human_approval`), status `in_review` (or `approved` + `published_at` when no human review needed and no blockers). Logs `EditorialAction("submit"/"auto_check")`.
- `approve(session, brief, editor: User, notes="") -> Brief` (refuses if blockers unless notes explain override; records review duration from `review_started_at`), `reject(session, brief, editor, notes)`, `start_review(session, brief, editor)`, `record_correction(session, brief, editor, description, material)`, `set_claim_verdict(session, claim, editor, verdict)`.
- `review_queue(session) -> list[Brief]`, `published_briefs(session, vertical=None, limit=30) -> list[Brief]`.

## 4. Ask Newsren — `newsren.agents.tools`, `newsren.agents.research` (Reader loop)

`newsren/agents/tools.py` — approved retrieval tools (`Tool` objects with strict JSON schemas, `additionalProperties: false`):
- `search_archive(query, vertical?, since_days?)` — FTS over ingested approved articles; returns numbered results with CitationRefs.
- `get_article(article_id)` — full text (truncated) + citation.
- `search_research_papers(query)` — arXiv API (approved research index).
- `search_sec_filings(company_or_ticker)` — SEC EDGAR company submissions (data.sec.gov) recent filings.
- `fetch_approved_url(url)` — only if `is_approved_url`; otherwise error "not an approved source".
- `build_research_tools(session, settings, allow_network=True) -> list[Tool]`. Offline/test mode: network tools return an error result instead of fetching when `allow_network=False`.

`newsren/agents/research.py`
- `ask(session, question, *, brief=None, story=None, article=None, audience=None, user=None, session_key=None, orchestrator=None, allow_network=None) -> Question` — finance advice requests are refused with the educational redirect (still answered educationally, `refused=True`); otherwise runs `ToolTask[ResearchAnswerOut]` (agent `"ask_newsren"`): plan → search → read → answer with citations, verified statements, and could-not-verify list. Offline builder: plan from the question, archive search seeded with brief/story context, extractive answer from best-matching sentences with citations, explicit could-not-verify items for question terms without support. Stores `Question` row.

## 5. Explanation agents and editorial products — `newsren.agents.learning`, `newsren.agents.impact`, `newsren.agents.products`

`newsren/agents/learning.py`
- `learning_materials(session, brief, audience, orchestrator=None, refresh=False) -> GeneratedContent` (kind `"learning"`, content `LearningOut`): student (plain-language 5-minute explainer, glossary, comprehension + transfer quiz), educator (classroom discussion guide, activities, age-appropriate questions), professional (implications, advanced technical reading). Only verified (cited) facts.
- `record_quiz_attempt(session, content, answers: list[int], user=None, condition="newsren") -> QuizAttempt`.

`newsren/agents/impact.py`
- `industry_impact(session, brief, question=None, orchestrator=None, refresh=False) -> GeneratedContent` (kind `"impact"`, content `ImpactOut`): stakeholders (industries, occupations, decisions) with effects labelled supported (cited) / speculative / scenario.

`newsren/agents/products.py`
- `PRODUCTS: dict[str, ProductSpec]` — the plan's editorial products: `launch_reality_check`, `research_paper_decoder`, `technology_timeline` (technology); `policy_impact_explorer`, `classroom_discussion_builder`, `education_evidence_checker` (education); `company_watch`, `supply_chain_ripple_map`, `skills_career_trends` (industry); `earnings_decoder`, `funding_round_tracker`, `ai_finance_reality_check`, `regulation_watch` (finance). `ProductSpec`: key, name, vertical, description, reader_value, applies_to(brief) -> bool, and a builder.
- `available_products(brief) -> list[ProductSpec]`.
- `run_product(session, key, brief, orchestrator=None, refresh=False) -> GeneratedContent` (kind `f"product:{key}"`, content `ProductOut`). Products whose data lives elsewhere (earnings/funding/ai-finance trackers, timeline) may delegate to those modules and wrap results into `ProductOut`.

## 6. Finance — `newsren.finance`

`newsren/finance/concepts.py`
- `CONCEPTS: list[Concept]` (slug, term, short, explainer, example, pitfalls, related) — at least backtesting, overfitting, in-sample vs out-of-sample, Sharpe ratio, drawdown, benchmark, dilution, term sheet, pre/post-money valuation, guidance, EPS, revenue vs. profit, 10-Q/10-K/8-K, AI washing.
- `get_concept(slug)`, `concepts_in_text(text) -> list[Concept]` (links stories to concepts).

`newsren/finance/edgar.py`
- `company_facts(cik, settings) -> dict` (data.sec.gov/api/xbrl/companyfacts), `company_submissions(cik, settings) -> dict`, `lookup_cik(ticker, settings) -> str | None` (company_tickers.json); pure parsers `extract_quarterly_metrics(facts) -> list[EarningsMetric]` (Revenues, NetIncomeLoss, EarningsPerShareDiluted, OperatingIncomeLoss... latest quarter vs prior quarter and same quarter prior year; each metric's `source_url` points to the filing (accession) and `source_quote` names the XBRL tag, form, and period).
- `build_earnings_report(session, company, settings, facts=None) -> EarningsReport` + `earnings_brief(session, report, orchestrator=None) -> Brief` (vertical finance, kind `"earnings"`, goes through review).

`newsren/finance/funding.py`
- `extract_funding(article) -> FundingExtractOut` (rule-based: "raises $X million Series B led by ..."; LLM optional via orchestrator) and `update_funding_tracker(session, articles=None, orchestrator=None) -> list[FundingRound]`; `funding_trends(session, days=90) -> dict` (by sector, stage, top investors by count — never ranks startups as investments).

`newsren/finance/ai_claims.py`
- `SEED_CLAIMS` from the plan's research table (FINRA Jan 2026 agent types; Deutsche Bank/Google Cloud surveillance agents; Goldman exploring; TradingAgents research prototype (backtest); JPMorgan LLM Suite ~50,000 staff (live deployment); JPMorgan 2026 regime-based portfolio agents (backtest, two-decade, 60/40 benchmark, authors' caveat); SEC AI-washing actions Delphia $225,000 / Global Predictions $175,000 (regulatory finding); vendor surveys 16%/8% (announcement/survey); Metal Raise Agent Aug 2026 (announcement)) with the plan's source citations.
- `seed_ai_claims(session) -> int`, `classify_claim(article, orchestrator=None) -> AIFinanceClaimOut`, `update_ai_claims(session, articles=None, orchestrator=None) -> list[AIFinanceClaim]`.

`newsren/finance/regulation.py`
- `regulation_watch(session, days=60) -> list[dict]` — recent regulatory/enforcement articles (SEC, FINRA, Fed) with who is affected / what changes / when (heuristic extraction; agent optional).

## 7. Readers, personalization, monitoring, delivery, metrics — `newsren.users`, `newsren.agents.monitor`, `newsren.agents.briefing`, `newsren.delivery`, `newsren.metrics`

`newsren/users/auth.py`
- `hash_password(pw) -> str` (hashlib.scrypt, random salt), `verify_password(pw, hashed) -> bool`, `create_user(session, email, password, display_name, role="reader") -> User`, `authenticate(session, email, password) -> User | None`.

`newsren/users/service.py`
- `set_preferences(session, user, audience=None, depth=None, email_opt_in=None, briefing_frequency=None)`, `add_interest(session, user, kind, value, vertical=None) -> Interest`, `remove_interest(session, user, interest_id)`.
- `create_follow(session, user, question, *, target_type="topic", story=None, entity_name=None, vertical=None, criteria: FollowCriteria | None = None) -> Follow` — derives keywords from the question (and criteria from phrases like "independent benchmark", "not opinion", "peer-reviewed", "filing", "earnings", "funding", "enforcement").
- `deactivate_follow`, `user_alerts(session, user, unread_only=False)`, `mark_alert(session, alert, feedback=None, read=True)`.

`newsren/agents/monitor.py` (Follow This Story)
- `check_follow(session, follow, orchestrator=None, now=None) -> list[Alert]` — candidates = non-duplicate articles retrieved since `last_checked_at` (or created_at) matching keywords/story/entity; drop excluded doc types (opinion by default); `classify_event`; apply criteria (event types, independence, primary); optional `custom` criterion judged by the agent (`StructuredTask[MonitorDecisionOut]`, agent `"follow_monitor"`; offline: keyword rule); dedup by story-event / near-duplicate (`dedup_key`); create `Alert` with reason + citations. Updates `last_checked_at`, `check_count`.
- `run_monitoring(session, orchestrator=None) -> dict` (counts).

`newsren/agents/briefing.py` (My Newsren)
- `build_briefing(session, user, orchestrator=None, now=None, force=False) -> Briefing | None` — gathers published briefs/stories/alerts matching the user's interests (topics, questions, organizations) since `last_briefing_at`, at the user's depth/audience; returns None when nothing meaningful changed and `briefing_frequency == "when_meaningful"` (unless force). Agent `"personal_briefing"` (`BriefingOut`) explains what changed and why it matters with citations.
- `deliver_briefing(session, briefing, user, settings=None) -> Briefing` — web always; email when `user.email_opt_in` via `newsren.delivery.email.send_email` (low-risk automatic delivery allowed by `settings.auto_deliver_personal_briefings`).
- `run_briefings(session, orchestrator=None) -> dict`.

`newsren/delivery/email.py`
- `send_email(to, subject, html, text, settings=None) -> str` — SMTP when configured, else writes `.eml` to `settings.outbox_dir` and returns the path/message-id. `render_briefing_email(briefing, user) -> tuple[subject, html, text]`.

`newsren/metrics.py`
- `trust_metrics(session)` (share of editor-sampled claims judged accurate; material corrections per 100 published briefs; claim citation coverage), `usefulness_metrics` (answered-question rate, saved-time rate, mean rating), `education_metrics` (comprehension/transfer accuracy by condition newsren vs summary), `engagement_metrics` (return visits to followed stories, useful vs unwanted alert rate), `editorial_metrics` (median review time, approval/rejection counts, estimated time saved), `cost_metrics` (avg cost per story brief / per question / per monitoring check from `AgentRun`; offline runs cost 0), `all_metrics(session) -> dict`.
