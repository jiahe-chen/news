# Service API index (generated)

Public functions/classes/constants per module with the first docstring line.


## `newsren.ingest.sources`

- `approved_domains(session: Session) -> set[str]`
- `is_approved_url(session: Session, url: str) -> bool` — True only for http(s) URLs on an approved source domain or the primary allow-list.
- `load_registry(path: Path | None=None) -> list[dict[str, Any]]` — Read and validate the approved source registry (``sources.json``).
- `sync_sources(session: Session, path: Path | None=None) -> list[Source]` — Upsert registry rows into ``sources`` (by slug) and return them.
- `full_text_allowed(source: Source) -> bool` — Whether the license lets us store an item's full body text.

## `newsren.ingest.pipeline`

- const `MIN_TOKENS_FOR_CONTENT_HASH`
- const `MIN_TOKENS_FOR_SIMHASH`
- const `SUMMARY_MAX_CHARS`
- const `CONTENT_MAX_CHARS`
- const `MAX_ITEM_ERRORS_PER_SOURCE`
- class `IngestReport` — fields: `sources_ok: int`, `sources_failed: int`, `new_articles: int`, `duplicates: int`, `errors: list[str]`, `already_seen: int`, `per_source: dict[str, dict[str, Any]]` — Outcome of one ``ingest_all`` run.
- class `Classification` — fields: `doc_type: str`, `trust_tier: str`, `cues: list[str]`, `flags: dict[str, bool]`
- `classify_item(source: Source, item: FetchedItem) -> Classification` — Decide doc_type and trust tier for an item (see module docstring for precedence).
- `store_item(session: Session, source: Source, item: FetchedItem, now: datetime | None=None) -> tuple[Article, bool]` — Normalize, classify, deduplicate, and store one fetched item.
- `ingest_source(session: Session, source: Source, settings: Settings, report: IngestReport, now: datetime | None=None) -> None` — Fetch one source and store its items, recording the outcome on the source and report.
- `ingest_all(session: Session, settings: Settings | None=None, source_slugs: Iterable[str] | None=None) -> IngestReport` — Fetch every approved, active source (optionally only ``source_slugs``).

## `newsren.ingest.fixtures`

- const `DEMO_LICENSE_NOTE`
- `load_demo_dataset(path: Path | None=None) -> dict[str, Any]` — Read and validate the demo corpus; resolves syndicated copies into full articles.
- `demo_article_url(source_homepage: str, key: str, title: str) -> str` — Stable URL for a demo article (independent of load date, so reloading is idempotent).
- `load_demo_data(session: Session, now: datetime | None=None, path: Path | None=None) -> int` — Load the fictional demo corpus; returns the number of articles newly stored.

## `newsren.intelligence.entities`

- const `LEGAL_SUFFIXES`
- const `ORG_WORDS`
- const `ORG_HEAD_WORDS`
- const `POLICY_SUFFIX`
- const `POLICY_PREFIX`
- const `PRODUCT_SUFFIX`
- const `MODEL_TIER_WORDS`
- const `PLACE_SUFFIX`
- const `TITLE_WORDS`
- const `ROLE_WORDS`
- const `CONNECTORS`
- const `MONTHS_DAYS`
- const `COMMON_INITIAL`
- const `GENERIC_WORDS`
- const `EXCLUDED_PREFIXES`
- const `GENERIC_ACRONYMS`
- const `HEADLINE_BREAK`
- const `FIRST_NAMES`
- `normalize_entity(name: str) -> str` — Normalized key used to dedupe entity names.
- `canonical_entity(name: str) -> tuple[str, str] | None` — (canonical name, kind) if ``name`` is a known gazetteer entity or alias.
- class `Mention` — fields: `name: str`, `kind: str`, `key: str`, `start: int`, `confidence: int` — One entity mention: display name, kind, normalized key, character offset.
- `entity_head(name: str, kind: str) -> str | None` — Brand head of a multi-word organization/product name ("Corvane Semiconductor" -> "corvane").
- `extract_mentions(text: str | None, known: dict[str, tuple[str, str]] | None=None) -> list[Mention]` — All entity mentions in ``text`` (with repeats), in order of appearance.
- `extract_entities(text: str | None) -> list[tuple[str, str]]` — Named entities in ``text`` as (name, kind), deduplicated, in order of first appearance.
- const `MAX_ENTITIES_PER_ARTICLE`
- `entity_saliences(article: Article) -> list[tuple[str, str, str, float]]` — (name, kind, normalized key, salience 0..1) for an article, most salient first.
- `tag_article_entities(session: Session, article: Article) -> list[Entity]` — Extract entities for ``article`` and upsert ``Entity`` + ``ArticleEntity`` rows.
- `article_entity_keys(session: Session, article: Article, limit: int=12) -> list[str]` — Normalized entity keys of an article, most salient first (tags it if untagged).

## `newsren.intelligence.events`

- class `EventSubject` — fields: `title: str`, `summary: str | None`, `content: str | None`, `doc_type: str`, `trust_tier: str` — Anything with the article fields the rules read (``Article`` or a stand-in).
- `classify_event_with_reason(article: EventSubject) -> tuple[str, str]` — (EventType value, rule name) for ``article``. See the module docstring for the rule order.
- `classify_event(article: EventSubject) -> str` — ``EventType`` value for ``article`` (rule-based, pure, fast).
- `event_label(event_type: str | None) -> str` — Reader-facing label for an ``EventType`` value.

## `newsren.intelligence.clustering`

- const `TEXT_WEIGHT`
- const `ENTITY_WEIGHT`
- const `CROSS_VERTICAL_MARGIN`
- const `CENTROID_MEMBERS`
- const `ARTICLE_ENTITY_LIMIT`
- const `STORY_CORE_ENTITIES`
- const `STORY_ENTITY_LIMIT`
- const `KEY_ENTITY_LIMIT`
- const `KEYWORD_LIMIT`
- const `ACTIVE_MEMBER_COUNT`
- const `MIN_RELATED_SCORE`
- const `ANCHORED_THRESHOLD`
- const `ANCHOR_MIN_TEXT`
- const `ANCHOR_MAX_DF_SHARE`
- const `RELATED_CANDIDATE_LIMIT`
- `article_time(article: Article) -> datetime` — When the article entered the story: publication time, else retrieval time.
- `article_doc(article: Article) -> str` — Text used for story similarity: title + summary (or the start of the body).
- `similarity_text(text: str) -> str` — Lower-cased text with light plural stemming ("schools" -> "school") so TF-IDF matches word forms.
- `clean_title(title: str) -> str` — Headline without format labels ("Opinion:", "Exclusive:" ...) or trailing exclamation marks.
- `best_member_title(articles: Sequence[Article]) -> str` — Most representative, most neutral member headline (a provisional story title).
- `title_anchors(session: Session, article: Article) -> set[str]` — Organization/product keys (or brand heads) whose names appear in the article's headline.
- `match_score(session: Session, article: Article, story: Story) -> float` — Similarity of ``article`` to ``story`` as used by ``assign_stories`` (no vertical adjustment).
- `assign_stories(session: Session, articles: Iterable[Article] | None=None, window_days: int=21, threshold: float=0.28) -> list[Story]` — Attach unassigned articles to existing stories or start new ones.
- `related_stories(session: Session, story: Story, limit: int=5) -> list[Story]` — Other stories (any vertical) connected to ``story`` by shared entities and keywords.

## `newsren.intelligence.timeline`

- const `AGENT_NAME`
- const `MAX_EVIDENCE_DOCS`
- const `EVIDENCE_CHARS_PER_DOC`
- const `MAX_DISPUTED`
- const `MAX_EVIDENCE_CHANGES`
- const `LATEST_COUNT`
- const `SYSTEM_PROMPT`
- `member_articles(story: Story) -> list[Article]` — Non-duplicate member articles, oldest first.
- `member_fingerprint(articles: Iterable[Article]) -> str` — SHA-256 of the sorted member article ids.
- `story_changes_since(story: Story, since: datetime) -> list[StoryEvent]` — Timeline events that occurred after ``since`` or were added to the story after it, oldest first.
- `offline_story_update(docs: Sequence[EvidenceDoc], members: Sequence[Article], types: dict[int, str]) -> StoryUpdateOut` — Extractive story update from the evidence pack (no LLM); cites only ids in ``docs``.
- `refresh_story(session: Session, story: Story, orchestrator: Orchestrator | None=None, force: bool=False) -> bool` — Rebuild a story's timeline and intelligence summary if its membership changed.
- `story_citations(story: Story) -> list[CitationRef]` — The citations ``[c1]``, ``[c2]`` ... used in the story's summary fields, in evidence-pack order.
- `update_stories(session: Session, articles: Iterable[Article] | None=None, orchestrator: Orchestrator | None=None, *, window_days: int=21, threshold: float=0.28) -> list[Story]` — Cluster unassigned articles (``assign_stories``) and refresh every story that changed.

## `newsren.agents.brief`

- const `AGENT_NAME`
- const `MAX_EVIDENCE_DOCS`
- const `EVIDENCE_CHARS_PER_DOC`
- const `MAX_CLAIMS`
- const `MAX_KEY_POINTS`
- const `MAX_UNCERTAINTY`
- const `MAX_WATCH_NEXT`
- const `SUMMARY_SENTENCES`
- const `SYSTEM_PROMPT`
- const `ATTRIBUTION`
- const `FORECAST`
- const `CONTRADICTION`
- const `EVALUATION_CUE`
- `tier_rank(article_or_doc: Article | SourceDoc) -> int` — 0 primary, 1 independent, 2 vendor, 3 commentary (opinion is commentary whatever its source tier).
- `numbers_in(text: str) -> set[str]` — Figures in ``text`` normalized for comparison ("11,500" -> "11500", "92.1%" -> "92.1").
- `key_terms(text: str) -> set[str]` — Content-word stems plus ``#``-prefixed figures, for comparing statements across sources.
- `similar(a: str, b: str, threshold: float=0.6) -> bool` — Two statements say much the same thing (shared content stems relative to the shorter one).
- `has_cue(pattern: re.Pattern[str], text: str) -> bool` — True when ``pattern`` matches ``text`` somewhere that is not directly negated ("did not dispute").
- `is_forecast(text: str) -> bool` — Forward-looking (a projection, plan, target, or guidance) rather than a reported result.
- `is_interpretive(text: str) -> bool` — Evaluative or causal wording ("breakthrough", "suggests", "should"), not a measured or attributed statement.
- `is_performance(text: str) -> bool` — Investment or strategy performance (returns, beating a benchmark portfolio), not an AI benchmark score.
- `performance_label(sentence: str, doc_text: str='') -> str | None` — backtest / simulated / live as the source states it (sentence first, then the document), else None.
- `performance_period(sentence: str, doc_text: str='') -> str | None` — The period a performance result covers ("January 2005 through December 2024", "two-decade").
- `performance_benchmark(sentence: str, doc_text: str='') -> str | None` — The benchmark a performance result is compared with ("60/40 portfolio", "S&P 500").
- class `SourceDoc` — fields: `cid: str`, `citation: CitationRef`, `article: Article | None`, `title: str`, `sentences: list[str]`, `event_type: str`, `_terms: list[set[str]] | None` — One evidence-pack document prepared for rule-based analysis.
- `source_docs(docs: Sequence[EvidenceDoc], articles: dict[int, Article], event_types: dict[int, str] | None=None) -> list[SourceDoc]` — Prepare evidence-pack documents for analysis: body sentences (title removed) and event types.
- `corrected_values(docs: Sequence[SourceDoc]) -> set[str]` — Figures a correction in the pack says were wrong ("84.3%, not 81.3%" -> {"81.3"}).
- `usable(doc: SourceDoc, sentence: str, wrong: AbstractSet[str]) -> bool` — A sentence may be reused unless it repeats a figure a correction withdrew (outside the correction).
- class `Match` — fields: `doc: SourceDoc`, `sentence: str`, `shared: int`, `ratio: float`, `figures: int` — A sentence in another document that agrees or disagrees with a statement.
- `agreeing_sentence(claim: str, doc: SourceDoc, wrong: AbstractSet[str]=frozenset()) -> Match | None` — The sentence in ``doc`` that states the same thing as ``claim`` (shared key terms and figures), if any.
- `contradicting_sentence(claim: str, claim_doc: SourceDoc, doc: SourceDoc, wrong: AbstractSet[str]=frozenset()) -> Match | None` — A sentence in ``doc`` that disputes ``claim`` (from ``claim_doc``), or that ``claim`` disputes.
- `distinct_independent_publishers(docs: Iterable[SourceDoc]) -> set[str]` — Publishers among ``docs`` that count toward corroboration (not the vendor, its press office, or commentary).
- `possessive(name: str) -> str` — Possessive form of a name: "Open Eval's", "EdArchive Preprints'".
- `attribute(doc: SourceDoc, sentence: str) -> str` — Make a vendor sentence read as the organization's claim, and an opinion as an argument.
- `label_forecast(sentence: str) -> str` — Append a projection label to a forward-looking sentence that does not already hedge itself.
- `cite(text: str, cids: Iterable[str]) -> str` — Append inline citation markers ("... [c1][c3].") to a sentence.
- `scrub_citations(text: str, valid: set[str]) -> str` — Remove inline ``[cN]`` markers for ids that are not in the evidence pack.
- `uncertainty_items(docs: Sequence[SourceDoc], wrong: AbstractSet[str], limit: int=MAX_UNCERTAINTY) -> list[str]` — Structural gaps (preprints, vendor-only evidence) plus hedged and limitation sentences, each cited.
- `select_pack_articles(members: Sequence[Article], event_types: dict[int, str]) -> list[Article]` — At most ``MAX_EVIDENCE_DOCS`` members, primary and independent first, then vendor, then commentary.
- `remap_citations(text: str | None, mapping: dict[str, str]) -> str` — Rewrite ``[cN]`` markers through ``mapping``; markers without a mapping are dropped.
- class `StoryContext` — fields: `story: Story | None`, `vertical: str`, `docs: list[SourceDoc]`, `timeline: list[StoryEvent]`, `first: str`, `new: str`, `disputed: list[str]`, `evidence_changes: list[str]`, `keywords: list[str]`, `entities: list[str]` — Everything the brief builders know about a story: its pack, timeline, and story-intelligence summary.
- `build_prompt(ctx: StoryContext, evidence: Sequence[EvidenceDoc]) -> str` — The user message for the ``story_brief`` agent: story context, timeline, source notes, and evidence.
- class `Candidate` — fields: `doc: SourceDoc`, `index: int`, `sentence: str`, `score: float` — A scored source sentence.
- class `SentencePicker` — Scores and picks source sentences for extractive output (shared by the brief and evidence builders).
- `offline_brief(ctx: StoryContext) -> BriefOut` — Extractive, rule-based AI Story Brief from the evidence pack; cites only ids in the pack.
- `sanitize_brief(out: BriefOut, docs: Sequence[SourceDoc]) -> BriefOut` — Enforce the evidence and labelling rules on an agent's ``BriefOut``.
- `brief_for_story(session: Session, story: Story) -> Brief | None` — The latest AI Story Brief for ``story`` (any status), or None.
- `generate_brief(session: Session, story: Story, orchestrator: Orchestrator | None=None, regenerate: bool=False) -> Brief` — Generate (or return the current) AI Story Brief for ``story`` and submit it for editorial review.

## `newsren.agents.evidence`

- const `AGENT_NAME`
- const `CONTENT_KIND`
- const `EVIDENCE_CHARS_PER_DOC`
- const `MAX_EXTRA_DOCS`
- const `MAX_FACTS`
- const `MAX_INTERPRETATIONS`
- const `MAX_FRAMING_NOTES`
- const `SYSTEM_PROMPT`
- `evidence_pack(session: Session, brief: Brief) -> tuple[list[EvidenceDoc], dict[int, Article]]` — The brief's own sources (ids unchanged) followed by the story's other members (new ids).
- `build_prompt(brief: Brief, evidence: Sequence[EvidenceDoc], docs: Sequence[SourceDoc]) -> str` — The user message for the ``evidence_inspector`` agent: the brief's claims, source notes, and evidence.
- `source_role(doc: SourceDoc) -> SourceRole` — Classify a document's role in the evidence, with a note on what it contributes.
- `offline_report(brief: Brief, docs: Sequence[SourceDoc]) -> EvidenceReportOut` — Rule-based evidence inspection from the pack; cites only ids in the pack.
- `labels_source_biased(text: str) -> bool` — True when ``text`` labels a source biased (rather than describing what it emphasizes).
- `sanitize_report(report: EvidenceReportOut, docs: Sequence[SourceDoc]) -> EvidenceReportOut` — Drop unknown citation ids, fix support statuses to the evidence rules, and remove bias labels.
- `inspect_evidence(session: Session, brief: Brief, orchestrator: Orchestrator | None=None, refresh: bool=False) -> GeneratedContent` — Evidence and Bias Inspection for ``brief`` (cached per brief id and version).

## `newsren.agents.research`

- const `AGENT_NAME`
- const `MAX_READ`
- const `MAX_CONTEXT_SOURCES`
- const `MAX_UNSUPPORTED_TERMS`
- const `SYSTEM_PROMPT`
- class `ResearchContext` — fields: `kind: str`, `headline: str`, `summary: str`, `points: list[str]`, `vertical: str | None`, `citations: list[CitationRef]`, `article_ids: list[int]`, `keywords: list[str]`, `article: Article | None` — What the reader was looking at when they asked; the agent starts from it.
- `build_context(session: Session, *, brief: Brief | None=None, story: Story | None=None, article: Article | None=None) -> ResearchContext | None` — Context seed from the brief, story, or article the reader asked from (None without one).
- `offline_answer(runner: ToolRunner, question: str, ctx: ResearchContext | None=None, audience: str | None=None, network: bool=False) -> ResearchAnswerOut` — Extractive Ask Newsren answer built only from what the approved tools return.
- `sanitize_answer(output: ResearchAnswerOut, allowed: Sequence[CitationRef], texts: Mapping[str, str] | None=None) -> ResearchAnswerOut` — Keep only citations to ``allowed`` sources, renumber them c1.., and repair everything that refers to them.
- `ask(session: Session, question: str, *, brief: Brief | None=None, story: Story | None=None, article: Article | None=None, audience: str | None=None, user: User | None=None, session_key: str | None=None, orchestrator: Orchestrator | None=None, allow_network: bool | None=None) -> Question` — Answer a reader's question with a sourced, verified research answer and store it as a ``Question``.
- `questions_for(session: Session, *, user: User | None=None, session_key: str | None=None, limit: int=20) -> list[Question]` — A reader's recent questions (by account or anonymous session key), newest first.

## `newsren.agents.tools`

- const `ARCHIVE_RESULT_LIMIT`
- const `ARCHIVE_CANDIDATES`
- const `ARTICLE_TEXT_LIMIT`
- const `FETCH_TEXT_LIMIT`
- const `PAPER_RESULT_LIMIT`
- const `FILING_RESULT_LIMIT`
- const `SNIPPET_CHARS`
- const `MAX_SINCE_DAYS`
- const `MAX_FETCH_BYTES`
- const `MAX_JSON_BYTES`
- const `MAX_REDIRECTS`
- const `SEC_TICKER_CACHE_SECONDS`
- const `ARXIV_API_URL`
- const `SEC_TICKERS_URL`
- const `SEC_SUBMISSIONS_URL`
- const `SEC_ARCHIVES_URL`
- const `QUESTION_FILLER`
- `query_terms(text: str | None) -> list[str]` — Distinct topic words of a query or question, in order (stopwords, filler, possessives removed).
- `doc_type_label(doc_type: str | None) -> str` — Reader-facing label for a ``DocType`` value.
- `trust_tier_label(trust_tier: str | None) -> str` — Reader-facing label for a ``TrustTier`` value.
- `article_handle(article_id: int) -> str` — The handle tools use for an archived article, e.g. ``A42``.
- class `RetrievalError` — A network retrieval failed or was refused (unapproved redirect, too large, ...).
- class `HttpResponse` — fields: `url: str`, `status_code: int`, `content_type: str`, `content: bytes`, `truncated: bool`
- `http_get(url: str, settings: Settings, *, accept: str='*/*', allow: Callable[[str], bool] | None=None, max_bytes: int=MAX_FETCH_BYTES, partial_ok: bool=False) -> HttpResponse` — GET ``url`` with The Newsren's User-Agent, following redirects only to URLs ``allow`` accepts.
- `archive_search(session: Session, query: str, *, vertical: str | None=None, since_days: int | None=None, limit: int=ARCHIVE_RESULT_LIMIT) -> list[Article]` — Best-matching archived articles from approved, active sources (duplicates collapse to the original).
- `arxiv_query_params(query: str, max_terms: int=5) -> dict[str, Any]` — arXiv API parameters for a free-text query, an arXiv id, or a query already in arXiv syntax.
- class `CompanyRecord` — fields: `cik: str`, `ticker: str`, `title: str`
- `clear_caches() -> None` — Forget the cached SEC ticker file (tests; or after SEC updates it).
- `parse_company_tickers(data: Mapping[str, Any]) -> list[CompanyRecord]` — Rows of SEC's ``company_tickers.json`` in SEC's order (largest companies first).
- `resolve_company(records: Iterable[CompanyRecord], query: str) -> list[CompanyRecord]` — Companies matching a ticker, CIK, or name, best first (one row per CIK); empty when nothing matches.
- `filing_url(cik: str, accession: str, document: str | None=None) -> str` — EDGAR archive URL of a filing's primary document (or its index when the document is unknown).
- `recent_material_filings(submissions: Mapping[str, Any], cik: str, limit: int=FILING_RESULT_LIMIT) -> tuple[list[dict[str, Any]], Counter[str]]` — Newest material filings from a ``submissions`` payload, plus counts of the routine forms skipped.
- `html_main_text(raw_html: str) -> str` — Readable text of an HTML page: the ``<main>``/``<article>`` element when present, without navigation.
- const `SEARCH_ARCHIVE_SCHEMA`
- const `GET_ARTICLE_SCHEMA`
- const `SEARCH_PAPERS_SCHEMA`
- const `SEARCH_FILINGS_SCHEMA`
- const `FETCH_URL_SCHEMA`
- class `ResearchToolbox` — fields: `session: Session`, `settings: Settings`, `allow_network: bool`, `retrieved_text: dict[str, str]` — The approved tools bound to one database session, plus a record of what they retrieved.
- `build_research_tools(session: Session, settings: Settings | None=None, allow_network: bool=True) -> list[Tool]` — The Ask Newsren tools (archive, article, arXiv, SEC EDGAR, approved URL) with strict JSON schemas.

## `newsren.agents.learning`

- const `AGENT_NAME`
- const `CONTENT_KIND`
- const `EVIDENCE_CHARS_PER_DOC`
- const `MAX_EXTRA_DOCS`
- const `CITE_MARK`
- const `RESEARCH_DOC_TYPES`
- const `VERIFIED_STATUSES`
- const `FACT_TYPES`
- `strip_citation_marks(text: str | None) -> str` — Remove inline ``[cN]`` markers (and the space before them).
- `scrub_citation_marks(text: str | None, valid: Iterable[str]) -> str` — Keep inline ``[cN]`` markers only for ids in ``valid``.
- `citation_marks_in(text: str | None) -> list[str]` — Citation ids referenced inline in ``text``, in order of first appearance.
- `cite(text: str, cids: Iterable[str]) -> str` — Append ``[cN]`` markers before the final period of ``text``.
- `clean_sentence(text: str | None) -> str` — A sentence without citation markers, section labels ("Results: "), or brief attribution prefixes.
- `ensure_period(text: str) -> str`
- `sentence_case(text: str) -> str`
- class `PackDoc` — fields: `citation: CitationRef`, `text: str`, `article: Article | None` — One evidence document (citation + text) with the article it came from, when known.
- class `EvidencePack` — fields: `brief: Brief`, `docs: list[PackDoc]` — The documents an agent may cite for a brief, keyed by citation id.
- `tier_rank(article: Article) -> int`
- `brief_evidence(session: Session, brief: Brief, *, cited_ids: Iterable[str] | None=None, include_story: bool=True, extra_articles: Sequence[Article]=(), max_extra: int=MAX_EXTRA_DOCS, chars: int=EVIDENCE_CHARS_PER_DOC) -> EvidencePack` — The brief's sources under their own citation ids, then other story members and extras under new ids.
- `brief_text(brief: Brief, *, include_sources: bool=True) -> str` — Headline, summary, sections, claims, and (optionally) cited titles/snippets as one string.
- `lead_subject(brief: Brief, pack: EvidencePack | None=None) -> str` — The organization or product the brief is about (first named in the headline), else a generic phrase.
- class `DateMention` — fields: `phrase: str`, `when: date | None`, `kind: str` — A date phrase in a sentence and its approximate calendar date (None when relative and unanchored).
- `find_dates(sentence: str, anchor: datetime | None=None) -> list[DateMention]` — Date phrases in ``sentence`` in order of appearance, without overlaps.
- `date_label(when: date | None) -> str`
- `plain_words(text: str) -> str` — Swap a few formal words for everyday ones (keeps the first letter's case).
- `short_sentences(text: str, max_words: int=22) -> list[str]` — Split long sentences at semicolons and clause boundaries (", but", ", which", dashes).
- `simplify(text: str, max_words: int=22) -> str` — Plain-language rewrite of ``text`` (shorter sentences, everyday words); citation markers are dropped.
- class `Fact` — fields: `text: str`, `cids: tuple[str, ...]`, `claim_type: str`, `performance_label: str | None` — A verified claim from the brief, as taught: clean text plus the ids that support it.
- class `OpenItem` — fields: `text: str`, `cids: tuple[str, ...]`, `status: str` — A contested or unverified claim, presented only as an open question.
- `verified_facts(brief: Brief, valid: set[str] | None=None) -> list[Fact]` — Supported or corroborated fact/performance claims with at least one (valid) citation.
- `open_items(brief: Brief, valid: set[str] | None=None) -> list[OpenItem]` — Contested claims first, then unverified ones (never taught as facts).
- class `Lesson` — fields: `key: str`, `term: str`, `definition: str`, `cue: str`, `question: str`, `correct: str`, `wrong: tuple[str, str, str]`, `explanation: str`, `analogy: str`, `misconception: str`, `implication: str`, `discussion: str`, `verticals: frozenset[str] | None` — A teachable idea matched to stories by cue words.
- `is_fallback(lesson: Lesson) -> bool` — True for the generic lessons used only to top up a quiz (not matched to the story).
- `matched_lessons(lessons: Sequence[Lesson]) -> list[Lesson]` — The lessons matched to the story by cues (fallbacks only when nothing matched).
- `select_lessons(text: str, vertical: str, limit: int, minimum: int=0) -> list[Lesson]` — Lessons whose cues appear in ``text`` (most cue hits first, the vertical's own lessons favoured).
- class `Term` — fields: `term: str`, `pattern: str`, `definition: str`, `verticals: frozenset[str] | None` — An editorial glossary entry for a standard term (not a claim about any organization).
- `source_definition(name: str, kind: str, sentences: Sequence[tuple[PackDoc, str]]) -> tuple[str, str] | None` — A definition of ``name`` stated in the sources ("X is a ...", "X, a ...", "X (ABC)") and its citation id.
- `glossary_items(vertical: str, focus_text: str, pack: EvidencePack, *, limit: int=8, concepts: Sequence[Concept]=(), max_names: int=3) -> tuple[list[GlossaryItem], list[str]]` — Glossary for ``focus_text``: finance concepts, standard terms, and names the sources define.
- `numbers_in(text: str) -> list[_Number]` — Money amounts, percentages, and plain numbers in ``text`` (years excluded).
- `comprehension_questions(facts: Sequence[Fact], pack: EvidencePack, limit: int) -> list[QuizQuestion]` — Questions answerable from the taught facts: figure and name fill-ins, and "which statement is supported".
- `transfer_questions(lessons: Sequence[Lesson], limit: int) -> list[QuizQuestion]` — One transfer question per lesson: the idea applied to an unfamiliar situation.
- `discussion_prompts(vertical: str, subject: str, lessons: Sequence[Lesson], open_list: Sequence[OpenItem], band: str='high_school', limit: int=4) -> list[str]` — Age-appropriate discussion questions for a band ("middle_school", "high_school", "university").
- `classroom_activities(vertical: str, subject: str, facts: Sequence[Fact], open_list: Sequence[OpenItem], pack: EvidencePack, lessons: Sequence[Lesson], *, band: str='high_school', limit: int=2) -> list[ClassroomActivity]` — One or two activities built from the story's verified facts and sources.
- const `SYSTEM_PROMPT`
- `normalize_audience(audience: str | Audience) -> str` — Validate an audience value ("student", "educator", "professional").
- class `LearningContext` — fields: `brief: Brief`, `audience: str`, `pack: EvidencePack`, `facts: list[Fact]`, `open_list: list[OpenItem]`, `concepts: list[Concept]`, `lessons: list[Lesson]`, `subject: str`, `focus_text: str`, `uncertainty: list[str]`
- `learning_context(session: Session, brief: Brief, audience: str) -> LearningContext` — Evidence (only articles cited by verified claims, under the brief's ids) and teaching assets for ``brief``.
- `build_prompt(ctx: LearningContext) -> str`
- `professional_implications(ctx: LearningContext, limit: int=6) -> list[str]` — Documented facts with decision relevance, lesson implications, watch-next items, and open questions.
- `advanced_reading(ctx: LearningContext, max_sentences: int=5) -> tuple[str, list[str]]` — A dense technical paragraph from the most technical source sentences, plus a concept explainer.
- `offline_learning(ctx: LearningContext) -> LearningOut` — Extractive, rule-based learning material for ``ctx.audience`` from verified facts only.
- `sanitize_learning(out: LearningOut, ctx: LearningContext) -> LearningOut` — Enforce the evidence and audience rules on any provider's output.
- `scrub_citation_marks_keep_lines(text: str | None, valid: Iterable[str]) -> str` — ``scrub_citation_marks`` applied per line, so paragraph and list breaks survive.
- `learning_materials(session: Session, brief: Brief, audience: str | Audience, orchestrator: Orchestrator | None=None, refresh: bool=False) -> GeneratedContent` — News-to-Learning material for ``brief`` and ``audience`` (cached per brief id, version, and audience).
- `record_quiz_attempt(session: Session, content: GeneratedContent, answers: list[int], user: User | None=None, condition: str='newsren') -> QuizAttempt` — Grade ``answers`` (one option index per quiz question, in order) and store a ``QuizAttempt``.

## `newsren.agents.impact`

- const `AGENT_NAME`
- const `CONTENT_KIND`
- const `MAX_STAKEHOLDERS`
- const `MAX_SOURCED_EFFECTS`
- const `MAX_OPEN_QUESTIONS`
- class `Stakeholder` — fields: `name: str`, `kind: str`, `pattern: str`, `scenario: str`, `question: str` — A group that a development can affect, with the cue pattern that finds it in the evidence.
- class `ImpactContext` — fields: `brief: Brief`, `question: str`, `pack: EvidencePack`, `subject: str`, `named: list[Stakeholder]`
- `default_question(vertical: str) -> str` — The default impact question for a vertical.
- `impact_context(session: Session, brief: Brief, question: str) -> ImpactContext`
- `classify_effect(doc: PackDoc, sentence: str) -> EffectOut` — A sourced effect: supported when stated as fact or documented rule, speculative when projected or opinion.
- `offline_impact(ctx: ImpactContext) -> ImpactOut` — Rule-based impact analysis: lexicon stakeholders found in the evidence, sourced effects, and scenarios.
- const `SYSTEM_PROMPT`
- `build_prompt(ctx: ImpactContext) -> str`
- `sanitize_impact(out: ImpactOut, ctx: ImpactContext) -> ImpactOut` — Enforce citation and labelling rules on any provider's output.
- `industry_impact(session: Session, brief: Brief, question: str | None=None, orchestrator: Orchestrator | None=None, refresh: bool=False) -> GeneratedContent` — Industry Impact analysis for ``brief`` (cached per brief id, version, and question).

## `newsren.agents.products`

- const `CONTENT_KIND_PREFIX`
- const `MAX_SECTIONS`
- class `BriefSignals` — fields: `vertical: str`, `kind: str`, `text: str`, `doc_types: frozenset[str]`, `events: frozenset[str]`, `event_count: int`, `organizations: tuple[str, ...]`
- `brief_signals(brief: Brief) -> BriefSignals` — Vertical, cited document types, story events, cue text, and organizations of a brief.
- class `ProductContext` — fields: `session: Session`, `brief: Brief`, `spec: ProductSpec`, `pack: EvidencePack`, `subject: str`, `data: dict[str, Any]`
- `quoted(doc: PackDoc, sentence: str) -> str` — A sentence attributed to its source, with its citation marker (no extra quotes around a quotation).
- `plain(doc: PackDoc, sentence: str) -> str`
- `make_section(heading: str, items: Sequence[tuple[PackDoc, str]], label: str, *, lead: str='', quote: bool=False) -> ProductSection | None` — A section from (doc, sentence) items (None when there are no items and no lead).
- `corrected_values(pack: EvidencePack) -> set[str]` — Figures a correction says were wrong ("was 84.3%, not 81.3%" -> {"81.3%"}).
- `build_launch_reality_check(ctx: ProductContext) -> ProductOut` — Vendor capability claims set against independent benchmarks; claims without third-party evidence flagged.
- `build_research_paper_decoder(ctx: ProductContext) -> ProductOut` — Hypothesis, method, findings, limitations, and possible applications of each research paper in the evidence.
- `timeline_events(ctx: ProductContext) -> list[_Event]` — Story timeline events (``StoryEvent`` rows) mapped to pack docs; cited articles when the story has none.
- `build_technology_timeline(ctx: ProductContext) -> ProductOut` — Releases, evaluations, corrections, and adoption milestones in date order.
- `is_proposal(doc: PackDoc) -> bool` — A proposed (not yet final) policy or rule, judged from the document's title and lede.
- `build_policy_impact_explorer(ctx: ProductContext) -> ProductOut` — Who an education policy affects, its implementation timeline (dates parsed from the sources), and open questions.
- `build_classroom_discussion_builder(ctx: ProductContext) -> ProductOut` — Age-banded questions and activities (middle school, high school, university) from verified facts only.
- `build_education_evidence_checker(ctx: ProductContext) -> ProductOut` — Measured learning outcomes versus promotional claims: speed versus transfer, participants, comparison group.
- const `ACTION_VERBS`
- `acting_organizations(pack: EvidencePack, limit: int=4, exclude: Iterable[str]=()) -> list[str]` — Organizations that take documented actions in the evidence, most widely mentioned first.
- `build_company_watch(ctx: ProductContext) -> ProductOut` — Publicly documented actions per organization, with dates and citations, side by side.
- `build_supply_chain_ripple_map(ctx: ProductContext) -> ProductOut` — Upstream and downstream links documented in the sources, with possible consequences labelled as hypotheses.
- `build_skills_career_trends(ctx: ProductContext) -> ProductOut` — Occupations, emerging skills, and training options in the news, labelled as signals rather than predictions.
- `linked_earnings_report(session: Session, brief: Brief) -> EarningsReport | None` — The EarningsReport linked to the brief, or the latest one for a company named in its headline or summary.
- `build_earnings_decoder(ctx: ProductContext) -> ProductOut` — Reported figures versus guidance and prior periods, with the filing or release quoted for every number.
- `build_funding_round_tracker(ctx: ProductContext) -> ProductOut` — Rounds from public announcements (stage, amount, investors) and sector trends, never a ranking.
- `build_ai_finance_reality_check(ctx: ProductContext) -> ProductOut` — Each source's AI-in-finance claim classified as announcement, pilot, backtest, live deployment, or regulatory finding.
- `build_regulation_watch(ctx: ProductContext) -> ProductOut` — What a regulator did, who is affected, what changes, and when (compliance dates parsed from the text).
- class `ProductSpec` — fields: `key: str`, `name: str`, `vertical: str`, `description: str`, `reader_value: str`, `applies: Callable[[BriefSignals], bool]`, `builder: Callable[[ProductContext], ProductOut]`, `instructions: str`, `default_label: str`, `force_label: str | None`, `prepare: Callable[[ProductContext], None] | None`, `extra_articles: Callable[[Session, Brief], list[Article]] | None`, `max_extra: int` — An editorial product: who it serves, when it applies, how it is built offline, and how Claude is briefed.
- `available_products(brief: Brief) -> list[ProductSpec]` — Products that apply to ``brief``: its own vertical's products first, then cross-vertical ones.
- `build_prompt(ctx: ProductContext) -> str`
- `sanitize_product(out: ProductOut, ctx: ProductContext) -> ProductOut` — Enforce the product's evidence and labelling rules on any provider's output.
- `product_context(session: Session, spec: ProductSpec, brief: Brief) -> ProductContext` — Evidence pack and precomputed product data for running ``spec`` on ``brief``.
- `get_product(key: str) -> ProductSpec` — The product spec for ``key``; raises ``KeyError`` with the valid keys when unknown.
- `run_product(session: Session, key: str, brief: Brief, orchestrator: Orchestrator | None=None, refresh: bool=False) -> GeneratedContent` — Run editorial product ``key`` on ``brief`` (cached per brief id and version).

## `newsren.agents.monitor`

- const `AGENT_NAME`
- const `CREATION_GRACE`
- const `STORY_EVENT_WINDOW`
- const `MAX_CANDIDATES_PER_CHECK`
- const `MAX_AGENT_DECISIONS_PER_CHECK`
- const `FTS_LIMIT`
- const `NEAR_DUPLICATE_HAMMING`
- const `NEAR_DUPLICATE_JACCARD`
- const `ALERTED_ARTICLES_COMPARED`
- const `AGENT_EVIDENCE_CHARS`
- const `WEAK_KEYWORDS`
- const `SYSTEM_PROMPT`
- `text_stems(text: str | None) -> set[str]` — Stemmed content tokens of ``text``; hyphenated names also contribute their parts.
- `keyword_stems(keyword: str) -> list[str]` — Stemmed tokens of a keyword or phrase (empty when it has no content words).
- `keyword_in(keyword: str, stems: set[str]) -> bool` — True when every content word of ``keyword`` occurs in ``stems``.
- `is_strong_keyword(keyword: str) -> bool` — Multi-word phrases and specific single words identify a subject; generic single words do not.
- `is_price_move(article: Article) -> bool` — True when the headline or summary is about a share-price move.
- class `SubjectMatch` — fields: `phrase: str`, `keywords: list[str]` — Why a candidate is about what the reader follows.
- `topic_match(keywords: Sequence[str], article: Article) -> list[str]` — Follow keywords ``article`` matches under the min-overlap rule, or [] when it does not match.
- `entity_names(name: str) -> list[str]` — The followed name plus its gazetteer canonical name and aliases.
- `mentions_any(names: Sequence[str], text: str) -> bool` — True when any of ``names`` occurs in ``text`` as a whole word (case-insensitive).
- `entity_article_ids(session: Session, names: Sequence[str]) -> set[int]` — Ids of articles tagged with any of ``names`` (normalized entity match).
- class `RuleVerdict` — fields: `passed: bool`, `met: list[str]`, `reason: str`
- `apply_rules(article: Article, event_type: str, criteria: FollowCriteria, corrects_alerted: bool=False) -> RuleVerdict` — Deterministic criteria: excluded formats, event types, independence, primary sources.
- `offline_custom_decision(custom: str, article: Article, event_type: str, opinion_excluded: bool=True) -> MonitorDecisionOut` — Keyword rule over the reader's free-text condition (the agent's offline builder).
- `judge_custom(orchestrator: Orchestrator, follow: Follow, subject: _Subject, criteria: FollowCriteria, article: Article, event_type: str, rules_met: Sequence[str]) -> MonitorDecisionOut` — Run the ``follow_monitor`` agent on one candidate for the follow's free-text condition.
- `alert_reason(article: Article, event_type: str, criteria_met: Sequence[str], custom_reason: str='') -> str` — Plain-language reason: what was published, by whom, and which of the reader's criteria it met.
- `check_follow(session: Session, follow: Follow, orchestrator: Orchestrator | None=None, now: datetime | None=None) -> list[Alert]` — Check one follow for new developments that meet the reader's criteria; return the new alerts.
- `render_alert_email(follow: Follow, alerts: Sequence[Alert], user: User, settings: Settings | None=None) -> tuple[str, str, str]` — Render one check's alerts as ``(subject, html, text)``; every alert links to its source.
- `email_alerts(session: Session, follow: Follow, alerts: Sequence[Alert], settings: Settings | None=None, now: datetime | None=None) -> str | None` — Email one digest of ``alerts`` to an opted-in reader and set ``emailed_at``; returns the send reference.
- `run_monitoring(session: Session, orchestrator: Orchestrator | None=None, now: datetime | None=None) -> dict[str, int]` — Check every active follow; returns counts (follows_checked, alerts_created, emails_sent, errors).

## `newsren.agents.briefing`

- const `AGENT_NAME`
- const `DEFAULT_PERIOD`
- const `CANDIDATE_LIMIT`
- const `MAX_ITEMS`
- const `MAX_ITEMS_PER_SECTION`
- const `MAX_ALERT_ITEMS`
- const `MIN_RELEVANCE`
- const `MEANINGFUL_SCORE`
- const `RECENCY_HALF_LIFE_DAYS`
- const `RECENCY_FLOOR`
- const `REPEAT_STORY_NOVELTY`
- const `VERTICAL_MISMATCH`
- const `WHOLE_VERTICAL_RELEVANCE`
- const `FTS_LIMIT`
- const `MAX_TECHNICAL_CLAIMS`
- const `ALERTS_HEADING`
- const `MIN_INTERVAL`
- const `OPINION_ONLY_WEIGHT`
- const `SYSTEM_PROMPT`
- class `CitationMerger` — Registers sources from many briefs under shared ids, deduplicated by URL (or article id).
- `renumber(output: BriefingOut, refs: dict[str, dict[str, Any]]) -> tuple[BriefingOut, list[dict[str, Any]]]` — Keep only cited sources and renumber them c1, c2 ... in order of first use across items.
- class `Lens` — fields: `key: str`, `heading: str`, `kind: str`, `keywords: list[str]`, `names: list[str]`, `story_id: int | None`, `vertical: str | None`, `fts_article_ids: set[int]`, `entity_article_ids: set[int]` — One interest or follow the briefing is organised around.
- `reader_lenses(session: Session, user: User) -> list[Lens]` — The reader's interests, then active follows, as matching lenses (duplicates removed).
- class `Candidate` — fields: `brief: Brief`, `text: str`, `stems: set[str]`, `headline_stems: set[str]`, `article_ids: set[int]`, `when: datetime`
- `relevance(lens: Lens, cand: Candidate) -> float` — How strongly a brief matches one lens, 0..1 (0 = no match).
- `opinion_only(brief: Brief) -> bool` — True when every source the brief cites is opinion or commentary.
- `recency(when: datetime, now: datetime) -> float` — Exponential decay with a 3-day half-life, floored so a week-old brief still counts.
- `previous_inclusions(session: Session, user: User) -> _Previous` — Briefs, alerts, and stories included in any earlier briefing for this reader.
- class `PlannedItem` — fields: `lens: Lens`, `score: float`, `vertical: str`, `item: BriefingItem`, `allowed_ids: list[str]`, `material: str`, `brief: Brief | None`, `alert: Alert | None`, `merged_alerts: list[Alert]`
- class `PlannedSection` — fields: `heading: str`, `items: list[PlannedItem]`
- `validate_output(output: BriefingOut, sections: Sequence[PlannedSection], offline: BriefingOut, finance: bool) -> BriefingOut` — Force the planned structure onto the agent's output and fall back item by item when it strays.
- `build_briefing(session: Session, user: User, orchestrator: Orchestrator | None=None, now: datetime | None=None, force: bool=False) -> Briefing | None` — Build (and store, undelivered) the reader's personal briefing, or None when nothing is new.
- `deliver_briefing(session: Session, briefing: Briefing, user: User, settings: Settings | None=None) -> Briefing` — Deliver a briefing: always on the web; by email when the reader opted in and automatic delivery is on.
- `briefing_due(user: User, now: datetime | None=None) -> bool` — Whether a scheduled run should try a briefing for this reader (minimum interval per frequency).
- `run_briefings(session: Session, orchestrator: Orchestrator | None=None, now: datetime | None=None) -> dict[str, int]` — Build and deliver briefings for every reader with interests or active follows; returns counts.

## `newsren.editorial.guardrails`

- const `FINANCE_DISCLAIMER`
- const `ADVICE_REFUSAL`
- `lint_text(text: str | None, vertical: str | None) -> list[str]` — Return human-readable guardrail violations in ``text`` for a brief in ``vertical``.
- `is_hedged(text: str | None) -> bool` — True when ``text`` labels itself as a forecast, projection, estimate, or scenario.
- `detect_advice_request(question: str | None) -> bool` — True when a reader asks for personalized investment advice, a rating, a price target, or a price call.
- `looks_like_performance(text: str | None) -> bool` — True when ``text`` describes investment or strategy performance (returns, beating a benchmark...).
- `performance_claim_issues(claim: BriefClaim | ClaimOut) -> list[str]` — Problems with a performance claim's labelling (a ``BriefClaim`` or ``ClaimOut``).

## `newsren.editorial.checks`

- const `BLOCKER`
- const `WARNING`
- const `INFO`
- `run_checks(session: Session, brief: Brief) -> list[CheckResult]` — Run every editorial quality-gate check on ``brief`` and return the results in ``CHECK_NAMES`` order.
- `failed_checks(checks: Sequence[CheckResult | dict[str, Any]], severity: str | None=None) -> list[Any]` — Checks that did not pass (optionally of one severity); accepts models or stored dicts.
- `assess_risk(brief: Brief, checks: Sequence[CheckResult | dict[str, Any]]) -> str` — Risk level for routing: high for finance, any high-impact claim, or any blocker; medium for warnings.

## `newsren.editorial.workflow`

- const `OVERRIDE_PREFIX`
- const `CLAIM_VERDICTS`
- `review_reasons(brief: Brief, checks: list[CheckResult], settings: Settings) -> list[str]` — Why a brief needs a human editor (empty when it may be published automatically).
- `blocking_checks(brief: Brief) -> list[dict]` — The brief's stored check results that block approval.
- `submit_for_review(session: Session, brief: Brief, settings: Settings | None=None) -> Brief` — Run the quality gate and route the brief: to the human review queue, or straight to publication.
- `start_review(session: Session, brief: Brief, editor: User) -> Brief` — Record that ``editor`` opened the brief for review (starts the review-time clock).
- `approve(session: Session, brief: Brief, editor: User, notes: str='') -> Brief` — Approve and publish a brief.
- `reject(session: Session, brief: Brief, editor: User, notes: str) -> Brief` — Reject a brief (it leaves the queue and is not published); ``notes`` must say why.
- `record_correction(session: Session, brief: Brief, editor: User, description: str, material: bool=False) -> Correction` — Record a correction to a brief (shown to readers) and bump its version.
- `set_claim_verdict(session: Session, claim: BriefClaim, editor: User, verdict: str) -> BriefClaim` — Record an editor's judgement of whether a claim's linked sources support it (trust metric sample).
- `review_queue(session: Session) -> list[Brief]` — Briefs awaiting a human decision (in review or draft): high risk first, then oldest first.
- `published_briefs(session: Session, vertical: str | None=None, limit: int=30) -> list[Brief]` — Approved (published) briefs, newest first, optionally for one vertical.
- `editorial_history(session: Session, brief: Brief) -> list[EditorialAction]` — Every editorial action on a brief, oldest first (audit trail for the review page).

## `newsren.finance.concepts`

- class `Concept` — fields: `slug: str`, `term: str`, `short: str`, `explainer: str`, `example: str`, `pitfalls: tuple[str, ...]`, `related: tuple[str, ...]`, `category: str`, `aliases: tuple[str, ...]`, `patterns: tuple[str, ...]`
- const `DISCLAIMER`
- `get_concept(slug: str) -> Concept | None` — Look up a concept by slug, a short alias ("drawdown", "guidance"), or its term.
- `concepts_by_category() -> dict[str, list[Concept]]` — Concepts grouped by category key, in ``CATEGORY_LABELS`` order.
- `related_concepts(concept: Concept) -> list[Concept]` — Resolved related concepts (unknown slugs are skipped).
- `concept_mentions(text: str | None) -> list[tuple[Concept, str]]` — (concept, first matched phrase) pairs in order of first appearance in ``text``.
- `concepts_in_text(text: str | None, limit: int | None=None) -> list[Concept]` — Concepts mentioned in ``text`` (word-boundary matches on terms and synonyms), in order of first mention.

## `newsren.finance.edgar`

- const `COMPANY_TICKERS_URL`
- const `SUBMISSIONS_URL`
- const `COMPANY_FACTS_URL`
- const `ARCHIVES_BASE`
- const `DATA_DELAY_NOTE`
- const `GUIDANCE_NOTE`
- const `COMPARISON_NOTE`
- const `PERIODIC_FORMS`
- const `EARNINGS_RELEASE_FORMS`
- const `TICKER_CACHE_SECONDS`
- const `QUARTER_DAYS`
- const `NINE_MONTH_DAYS`
- const `ANNUAL_DAYS`
- const `YEAR_TOLERANCE_DAYS`
- class `EdgarError` — Raised when EDGAR cannot be reached or returns unusable data.
- `normalize_cik(cik: str | int) -> str` — Return a CIK as the zero-padded 10-digit string EDGAR uses ("CIK0000320193" -> "0000320193").
- `normalize_ticker(ticker: str) -> str` — Upper-case a ticker and use EDGAR's class separator ("brk.b" -> "BRK-B").
- `parse_company_tickers(data: Mapping[str, Any]) -> dict[str, tuple[str, str]]` — Map ticker -> (10-digit CIK, company title) from SEC's ``company_tickers.json``.
- `lookup_cik(ticker: str, settings: Settings | None=None, client: httpx.Client | None=None) -> str | None` — Return the 10-digit CIK for a ticker, or None if SEC does not list it.
- `company_submissions(cik: str | int, settings: Settings | None=None, client: httpx.Client | None=None) -> dict[str, Any]` — Company profile and recent filings (data.sec.gov/submissions).
- `company_facts(cik: str | int, settings: Settings | None=None, client: httpx.Client | None=None) -> dict[str, Any]` — All XBRL facts a company has reported (data.sec.gov/api/xbrl/companyfacts).
- `filing_index_url(cik: str | int, accession: str) -> str` — EDGAR filing index for an accession, e.g. .../data/1325618/000162828026051402/.
- `filing_document_url(cik: str | int, accession: str, document: str) -> str` — URL of one document (e.g. the primary 10-Q HTML) inside a filing.
- class `XbrlFact` — fields: `tag: str`, `unit: str`, `start: date | None`, `end: date`, `value: float`, `accession: str`, `form: str`, `filed: date`, `fiscal_year: int | None`, `fiscal_period: str | None` — One us-gaap fact from a 10-Q/10-K as listed in companyfacts.
- class `MetricSpec` — fields: `key: str`, `name: str`, `tags: tuple[str, ...]`, `unit: str`, `display_unit: str`, `additive: bool`
- `parse_facts(facts: Mapping[str, Any], tag: str, unit: str, as_of: date | None=None) -> list[XbrlFact]` — 10-Q/10-K duration facts for one us-gaap tag and unit, optionally only those filed by ``as_of``.
- class `PeriodicFiling` — fields: `accession: str`, `form: str`, `filed: date`, `period_start: date | None`, `period_end: date`, `fiscal_year: int | None`, `fiscal_period: str | None`, `is_annual: bool` — The 10-Q/10-K that the latest metrics come from.
- `latest_periodic_filing(facts: Mapping[str, Any], as_of: date | None=None) -> PeriodicFiling | None` — Identify the most recent 10-Q/10-K covered by the metric tags (latest period end, then filing date).
- class `QuarterlyMetric` — fields: `key: str`, `period_kind: Literal['quarter', 'fiscal_year']`, `form: str`, `filed: str`, `accession: str`, `derived: bool`, `comparison: Literal['same_quarter_prior_year', 'prior_quarter', 'prior_fiscal_year', '']`, `prior_source_url: str`, `prior_quarter_value: float | None`, `prior_quarter_period: str`, `prior_quarter_change_pct: float | None`, `prior_quarter_source_url: str` — ``EarningsMetric`` plus structured provenance and the second comparison.
- `change_pct(current: float | None, prior: float | None) -> float | None` — Percentage change, or None when not meaningful (missing figure, prior <= 0, or current is a loss).
- `format_value(value: float | None, unit: str='USD') -> str` — Exact figure for quotes: "$20,466,000", "-$1,250,000", or "$0.41 per share".
- `extract_quarterly_metrics(facts: Mapping[str, Any], *, as_of: date | None=None) -> list[QuarterlyMetric]` — Latest-quarter metrics from a companyfacts payload, each with prior-period comparisons and provenance.
- `recent_filings(submissions: Mapping[str, Any]) -> list[dict[str, Any]]` — Rows of ``filings.recent`` (column arrays) as dicts, newest first.
- `find_earnings_release(submissions: Mapping[str, Any], filing: PeriodicFiling, cik: str) -> dict[str, Any] | None` — The 8-K (Item 2.02, results of operations) that announced this period's results, if listed.
- `get_or_create_company(session: Session, name: str, ticker: str | None=None, cik: str | int | None=None, sector: str | None=None) -> Company` — Find a company by CIK, then ticker, then exact name (without conflicting identifiers); else create it.
- `build_earnings_report(session: Session, company: Company, settings: Settings | None=None, facts: Mapping[str, Any] | None=None, submissions: Mapping[str, Any] | None=None, *, guidance: Iterable[GuidanceItem | Mapping[str, Any]] | None=None, as_of: date | None=None) -> EarningsReport` — Create or update the ``EarningsReport`` for the company's latest 10-Q/10-K.
- const `EARNINGS_AGENT_NAME`
- const `EARNINGS_BRIEF_KIND`
- const `MAX_BRIEF_FIGURES`
- const `MAX_COMPANY_ARTICLES`
- const `MAX_AGENT_CLAIMS`
- const `RELEASE_WINDOW_BEFORE_DAYS`
- const `RELEASE_WINDOW_AFTER_DAYS`
- const `EARNINGS_SYSTEM_PROMPT`
- `period_text(period: str | None) -> str` — A readable period: "2026-04-01..2026-06-30" -> "April 1 to June 30, 2026".
- `fiscal_label(fiscal_period: str | None) -> str` — A readable fiscal period: "FY2026Q2" -> "fiscal 2026 Q2", "FY2025" -> "fiscal 2025".
- `company_name_variants(name: str) -> list[str]` — Names an article may use for a company: "Corvane Semiconductor Inc." -> [full, "Corvane Semiconductor", "Corvane"].
- `extract_guidance(text: str, *, report_quarter: int | None=None, source_url: str='', actuals: Mapping[str, tuple[float, str]] | None=None) -> list[GuidanceItem]` — Guidance ranges quoted in an earnings release (rule-based; never inferred).
- `earnings_brief(session: Session, report: EarningsReport, orchestrator: Orchestrator | None=None, *, regenerate: bool=False) -> Brief` — Write the Earnings and Filings Brief for ``report`` and send it to human review.

## `newsren.finance.funding`

- const `AGENT_NAME`
- const `CACHE_KIND`
- const `MIN_CONFIDENCE`
- const `TRACKED_VERTICALS`
- const `SKIP_DOC_TYPES`
- const `DUPLICATE_WINDOW_DAYS`
- const `TRENDS_NOTE`
- class `Amount` — fields: `text: str`, `usd: float | None`, `start: int`
- `parse_amounts(text: str) -> list[Amount]` — Money amounts in reading order ("$45 million", "$1.2B", "€20m", "20 million euros").
- `detect_stage(text: str) -> str` — Round stage from text; "undisclosed" when no stage is named.
- `infer_sector(title: str, body: str='') -> str` — Sector label by weighted keyword hits (headline/summary x3); "Other" when nothing matches.
- `parse_names(fragment: str, max_names: int=8) -> list[str]` — Organisation names from a list fragment ("Aster Fund and Halden Growth, with ..." -> two names).
- `extract_investors(text: str) -> tuple[list[str], list[str]]` — (lead investors, all investors) named in ``text``; leads come first in the full list.
- `extract_funding_from_text(title: str, body: str='', *, doc_type: str | None=None) -> FundingExtractOut` — Rule-based funding-round extraction from a headline and body text.
- `extract_funding(article: Article) -> FundingExtractOut` — Rule-based extraction of a funding round from an article (offline builder of ``funding_extractor``).
- `looks_like_funding(article: Article) -> bool` — Cheap pre-filter before calling the extraction agent.
- const `SYSTEM_PROMPT`
- `extract_funding_with_agent(session: Session, article: Article, orchestrator: Orchestrator) -> FundingExtractOut` — Run the ``funding_extractor`` agent for one article, cached per article version.
- `update_funding_tracker(session: Session, articles: Iterable[Article] | None=None, orchestrator: Orchestrator | None=None) -> list[FundingRound]` — Log funding rounds from finance/industry/technology articles not yet tracked; return the new rows.
- `funding_trends(session: Session, days: int=90, *, now: datetime | None=None, include_demo: bool=True, recent_limit: int=20) -> dict[str, Any]` — Aggregate logged rounds over the last ``days``: counts and disclosed-USD sums by sector, stage, investor.

## `newsren.finance.ai_claims`

- const `AGENT_NAME`
- const `CACHE_KIND`
- const `CANDIDATE_VERTICALS`
- const `SKIP_DOC_TYPES`
- const `DUPLICATE_WINDOW_DAYS`
- class `SeedClaim` — fields: `firm: str`, `area: str`, `claim_text: str`, `classification: str`, `source_url: str`, `source_title: str`, `published_at: datetime | None`, `published_label: str`, `performance_label: str | None`, `period: str | None`, `benchmark: str | None`, `costs_assumed: str | None`, `authors_caveats: str | None`, `evidence_note: str | None` — One documented claim from the plan's research table, with its source.
- `seed_ai_claims(session: Session) -> int` — Insert (or refresh) the plan's seed claims; returns how many were newly inserted. Idempotent.
- `ai_finance_sentences(article: Article) -> list[str]` — Sentences in which an AI cue and a finance cue co-occur (the evidence that a claim is about AI in finance).
- `mentions_ai_in_finance(article: Article) -> bool` — Cheap pre-filter: does any sentence pair an AI cue with a finance cue?
- `classify_claim_rules(article: Article) -> AIFinanceClaimOut` — Rule-based classification of an article's AI-in-finance claim (offline builder).
- const `SYSTEM_PROMPT`
- `classify_claim(article: Article, orchestrator: Orchestrator | None=None) -> AIFinanceClaimOut` — Classify an article's AI-in-finance claim; uses the ``ai_finance_classifier`` agent when given an orchestrator.
- `update_ai_claims(session: Session, articles: Iterable[Article] | None=None, orchestrator: Orchestrator | None=None) -> list[AIFinanceClaim]` — Classify qualifying finance/technology/industry articles and store new claims (idempotent per article).
- `claims_by_classification(session: Session, include_seed: bool=True) -> list[dict[str, Any]]` — Claims grouped for the reality-check page: [{classification, label, help, claims: [AIFinanceClaim]}].

## `newsren.finance.regulation`

- const `FINANCE_REGULATOR_KINDS`
- const `WATCH_NOTE`
- const `MAX_SENTENCES`
- const `MAX_ITEMS`
- `identify_regulator(article: Article) -> tuple[str, str, str] | None` — (name, group, kind) of the regulator behind an article, or None when no regulator is identified.
- `classify_action(title: str, summary: str='', content: str='') -> tuple[str | None, str]` — (action_type, stage) for a regulatory text: the title decides first, then the summary, then the body.
- const `MAX_AFFECTED`
- `who_and_what(sentences: Sequence[str]) -> tuple[list[str], list[str]]` — (who is affected, what changes): sentences quoted from the source.
- `parse_when(text: str, published: datetime | None=None) -> list[dict[str, Any]]` — Compliance, effective, comment, and deadline dates stated in ``text``.
- `regulation_watch(session: Session, days: int=60, *, vertical: str | None=None, now: datetime | None=None, limit: int=MAX_ITEMS) -> list[dict[str, Any]]` — Recent regulatory and enforcement actions: who is affected, what changes, and when.
- `finance_regulation_items(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]` — The Regulation Watch items that belong on the Finance page (finance vertical or a financial regulator).

## `newsren.finance.service`

- const `FUNDING_WINDOW_DAYS`
- const `REGULATION_WINDOW_DAYS`
- const `EARNINGS_LIMIT`
- const `CLAIMS_PER_CLASSIFICATION`
- const `MAX_EDGAR_COMPANIES`
- const `EDGAR_PAUSE_SECONDS`
- const `DATA_NOTE`
- `finance_overview(session: Session, *, now: datetime | None=None) -> dict[str, Any]` — Everything the Finance page shows, as plain data.
- `refresh_finance(session: Session, orchestrator: Orchestrator | None=None, settings: Settings | None=None, network: bool=False) -> dict[str, Any]` — Refresh the finance trackers; returns counts of what changed and any errors.

## `newsren.users.auth`

- const `SCHEME`
- const `SCRYPT_N`
- const `SCRYPT_R`
- const `SCRYPT_P`
- const `SALT_BYTES`
- const `KEY_BYTES`
- const `MIN_PASSWORD_LENGTH`
- const `MAX_PASSWORD_LENGTH`
- const `MAX_EMAIL_LENGTH`
- const `MAX_DISPLAY_NAME_LENGTH`
- `hash_password(password: str) -> str` — Hash ``password`` with scrypt and a fresh random salt.
- `verify_password(password: str, hashed: str) -> bool` — True when ``password`` matches ``hashed``; False for wrong or malformed input (never raises).
- `needs_rehash(hashed: str) -> bool` — True when ``hashed`` was made with parameters other than the current ones.
- `normalize_email(email: str) -> str` — Canonical form used for storage and lookup: stripped and lower-cased.
- `validate_email(email: str) -> str` — Return the normalized email or raise ``ValueError`` if it is not a plausible address.
- `validate_password(password: str) -> str` — Return ``password`` unchanged or raise ``ValueError`` if it breaks the length rules.
- `get_user_by_email(session: Session, email: str) -> User | None` — Look up an account by email (case-insensitive).
- `create_user(session: Session, email: str, password: str, display_name: str, role: str=Role.READER) -> User` — Create an account.
- `set_password(session: Session, user: User, new_password: str) -> User` — Replace a user's password (length rules apply).
- `authenticate(session: Session, email: str, password: str) -> User | None` — Return the user when ``email``/``password`` match, else None.

## `newsren.users.service`

- const `BRIEFING_FREQUENCIES`
- const `INTEREST_KINDS`
- const `FOLLOW_TARGETS`
- const `ALERT_FEEDBACK`
- const `FEEDBACK_TARGETS`
- const `MAX_INTEREST_LENGTH`
- const `MAX_QUESTION_LENGTH`
- const `MAX_QUESTION_KEYWORDS`
- const `MAX_FOLLOW_KEYWORDS`
- const `FILLER_WORDS`
- `set_preferences(session: Session, user: User, audience: str | None=None, depth: str | None=None, email_opt_in: bool | str | None=None, briefing_frequency: str | None=None) -> User` — Update the reader's "Explain for" audience, technical depth, email opt-in, and briefing frequency.
- `add_interest(session: Session, user: User, kind: str, value: str, vertical: str | None=None) -> Interest` — Add a topic, question, organization, or sector to the reader's interests.
- `remove_interest(session: Session, user: User, interest_id: int) -> bool` — Remove one of the reader's interests. Returns False if it does not exist or belongs to someone else.
- const `PEER_REVIEW_NOTE`
- const `PUBLISHED_STUDY_NOTE`
- `derive_criteria(question: str) -> FollowCriteria` — Notification criteria implied by the reader's question (rule-based, see module docstring).
- `derive_keywords(question: str, limit: int=MAX_QUESTION_KEYWORDS) -> list[str]` — Subject keywords for a follow question: phrases first, then remaining terms, filler removed.
- `describe_criteria(criteria: FollowCriteria | dict[str, Any]) -> list[str]` — Plain-language lines describing what will (and will not) trigger an alert, for the reader's UI.
- `create_follow(session: Session, user: User, question: str, *, target_type: str='topic', story: Story | None=None, entity_name: str | None=None, vertical: str | None=None, criteria: FollowCriteria | None=None) -> Follow` — Follow a question, story, company, or sector over time.
- `deactivate_follow(session: Session, follow: Follow | User, follow_id: int | None=None) -> Follow | None` — Stop monitoring a follow (its alerts are kept).
- `user_follows(session: Session, user: User, active_only: bool=True) -> list[Follow]` — The reader's follows, newest first.
- `user_alerts(session: Session, user: User, unread_only: bool=False, limit: int | None=None) -> list[Alert]` — The reader's alerts, newest first (optionally only unread ones).
- `unread_alert_count(session: Session, user: User) -> int` — Number of unread alerts (for navigation badges).
- `mark_alert(session: Session, alert: Alert, feedback: str | None=None, read: bool=True) -> Alert` — Mark an alert read/unread and optionally record the reader's verdict.
- `record_feedback(session: Session, user: User | None, target_type: str, target_id: int, *, answered_question: bool | None=None, saved_time: bool | None=None, rating: int | None=None, comment: str | None=None) -> Feedback` — Record whether a briefing/answer/brief answered the reader's question and saved them time.
- `interests_by_kind(interests: Sequence[Interest]) -> dict[str, list[Interest]]` — Group interests by kind (topic/question/organization/sector) in that order, for display.

## `newsren.delivery.email`

- const `SMTP_TIMEOUT_SECONDS`
- const `MAX_SUBJECT_LENGTH`
- const `FOOTER_TEXT`
- const `NO_SOURCE_TEXT`
- class `EmailDeliveryError` — SMTP delivery failed (the message never contains credentials).
- `preferences_url(settings: Settings | None=None) -> str` — Where readers manage briefing email preferences or unsubscribe.
- `build_message(to: str, subject: str, html_body: str, text_body: str, settings: Settings | None=None, headers: Mapping[str, str] | None=None) -> EmailMessage` — Build the multipart/alternative message (text first, HTML preferred by capable clients).
- `send_email(to: str, subject: str, html: str, text: str, settings: Settings | None=None, *, headers: Mapping[str, str] | None=None) -> str` — Send one email and return its Message-ID (SMTP) or the ``.eml`` path written to the outbox.
- `render_briefing_email(briefing: Briefing, user: User, settings: Settings | None=None) -> tuple[str, str, str]` — Render a personal briefing as ``(subject, html, text)``.

## `newsren.metrics`

- const `DEFAULT_BASELINE_MANUAL_MINUTES`
- const `RETURN_VISIT_MIN_GAP_MINUTES`
- const `CONDITIONS`
- const `STORY_BRIEF_AGENT`
- const `QUESTION_AGENT`
- const `MONITOR_AGENT`
- `trust_metrics(session: Session) -> dict[str, Any]` — Sampled-claim accuracy, material corrections per 100 published briefs, and claim citation coverage.
- `usefulness_metrics(session: Session) -> dict[str, Any]` — Reader feedback: answered-question rate, saved-time rate, and mean 1-5 rating, overall and by item type.
- `education_metrics(session: Session) -> dict[str, Any]` — Comprehension and transfer accuracy by reading condition, and the Newsren-minus-summary difference.
- `engagement_metrics(session: Session, min_gap_minutes: int=RETURN_VISIT_MIN_GAP_MINUTES) -> dict[str, Any]` — Return visits to followed stories and the useful-versus-unwanted alert rate.
- `editorial_metrics(session: Session, baseline_manual_minutes: float=DEFAULT_BASELINE_MANUAL_MINUTES) -> dict[str, Any]` — Human review time, approval/rejection counts, corrections, and estimated time saved per approved brief.
- `cost_metrics(session: Session) -> dict[str, Any]` — Inference and retrieval cost per agent, per story brief, per question, and per monitoring check.
- `all_metrics(session: Session, baseline_manual_minutes: float=DEFAULT_BASELINE_MANUAL_MINUTES) -> dict[str, Any]` — Every metric group in one JSON-serialisable dict (for the analytics page and ``newsren metrics``).

## `newsren.pipeline`

- class `PipelineReport` — fields: `new_articles: int`, `duplicates: int`, `sources_ok: int`, `sources_failed: int`, `demo_articles_loaded: int`, `entities_tagged: int`, `stories_changed: int`, `stories_refreshed: int`, `briefs_generated: int`, `briefs_in_review: int`, `briefs_published: int`, `finance: dict[str, Any]`, `monitoring: dict[str, Any]`, `briefings: dict[str, Any]`, `errors: list[str]`
- `tag_untagged_articles(session: Session, limit: int=2000) -> int`
- `stories_needing_briefs(session: Session, limit: int) -> list[Story]` — Stories whose latest brief is missing or older than their last update.
- `run_pipeline(session: Session, *, settings: Settings | None=None, network: bool=True, demo: bool=False, max_briefs: int=25, finance: bool=True, monitor: bool=True, briefings: bool=True) -> PipelineReport`
