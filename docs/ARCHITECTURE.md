# Architecture

The Newsren is an **editorially controlled multi-agent workflow**, not a
collection of unrestricted chatbots (plan §6). Agents have a defined goal,
access only to approved retrieval tools, and run inside budgets set by the
orchestrator. Every generated item keeps its sources, publication timestamps,
retrieval time, and the evidence behind its main claims.

```
             ┌──────────────────────────── Orchestrator ─────────────────────────────┐
             │  task selection · provider (Claude / offline) · time, tool-call and  │
             │  cost budgets · AgentRun accounting (tokens, cost, duration, trace)  │
             └───────────────────────────────────────────────────────────────────────┘
 Approved      ┌──────────┐   ┌──────────────┐   ┌───────────────┐   ┌───────────────┐   ┌────────────┐
 sources  ───► │ Discover │──►│   Connect    │──►│    Explain    │──►│ Quality gate  │──►│   Output   │
 RSS, arXiv,   │ ingest,  │   │ entities,    │   │ AI Story Brief│   │ automated     │   │ published  │
 SEC EDGAR,    │ dedup,   │   │ clustering,  │   │ learning,     │   │ checks +      │   │ stories,   │
 ERIC, Fed,    │ FTS index│   │ timelines,   │   │ impact,       │   │ human review  │   │ briefings, │
 regulators    └──────────┘   │ story intel. │   │ products      │   │ (finance: all)│   │ alerts     │
                              └──────────────┘   └───────────────┘   └───────────────┘   └────────────┘
                                         ▲                                                    │
                                         └──── Reader loop: Ask Newsren · Follow This Story ◄──┘
```

## Components (plan §6 table → code)

| Plan component | Module(s) | Notes |
|---|---|---|
| Orchestrator | `newsren/agents/base.py` (`Orchestrator`), `newsren/pipeline.py` | Every agent call goes through `Orchestrator.run`, which enforces `Budget` (max tool calls, wall-clock timeout, max cost, max output tokens) and records an `AgentRun`. `run_pipeline` sequences the workflow; each step runs in a savepoint so one failure never blocks the rest. |
| Source Discovery | `newsren/ingest/*` | Approved registry (`ingest/data/sources.json`), fetchers for RSS/Atom, arXiv, SEC EDGAR, ERIC; canonical URLs, content hashes and SimHash near-duplicate detection; SQLite FTS5 archive index. |
| Evidence Verification | `newsren/agents/evidence.py`, `newsren/editorial/checks.py` | Provenance (original vs secondary vs vendor vs independent vs commentary), dates, corroboration across independent sources, study design inspection. |
| Story Intelligence | `newsren/intelligence/*` | Entity extraction, incremental clustering into evolving stories, event classification, timelines, change detection, story-intelligence agent (what happened first, what's new, what's disputed, how evidence changed). Clustering combines TF-IDF text similarity with entity overlap and adds rules learned from live feeds: a distinctive subject named in both headlines lowers the bar (headline anchors); per-source boilerplate words and entities, publisher self-names, and repeated series titles are ignored; two releases from one publisher join only when they share a named subject; SEC filings cluster only by company (CIK). `related_stories` links stories across verticals, which the "Connected stories" tool renders as one storyline. |
| News Explanation | `newsren/agents/brief.py`, `learning.py`, `impact.py`, `products.py` | AI Story Brief, News-to-Learning (student/educator/professional), Industry Impact (supported vs speculative), 13 vertical editorial products. |
| Editorial quality gate | `newsren/editorial/*` | Citation coverage, approved-source and link checks, licensing notes, preprint and forecast labelling, finance guardrail lint, performance labels; human review queue; corrections; claim sampling for the trust metric. |
| Output | `newsren/agents/briefing.py`, `monitor.py`, `newsren/delivery/email.py`, `newsren/web/*` | Published stories, My Newsren personalized briefings (web + opt-in email), Follow This Story alerts that fire only when reader criteria are met. |

## LLM layer

`newsren/llm/` defines provider-neutral tasks:

* `StructuredTask[T]` — evidence in, validated pydantic model `T` out.
* `ToolTask[T]` — the model plans, calls approved tools, and then answers in `T`.

Each task carries an **offline builder**: a deterministic, extractive/rule-based
implementation over the same evidence. The **Claude provider**
(`newsren/llm/claude.py`) uses `claude-opus-5-5` (configurable via
`NEWSREN_MODEL`) with adaptive thinking, an explicit effort level, structured
outputs (`output_config.format` with a JSON schema generated from the pydantic
contract), prompt caching on the stable system prefix, and server-side refusal
fallbacks (`fallbacks="default"`). Agentic research uses a manual tool loop so
the orchestrator can enforce tool-call, time, and cost budgets between turns.
If Claude is unavailable, refuses, or returns invalid output, the offline
builder answers and the run is recorded as `fallback`/`refused` — the reader
always gets an evidence-grounded result and the system is honest about which
engine produced it.

Without `ANTHROPIC_API_KEY` the platform runs fully offline (that is how the
test-suite runs).

## Guardrails (plan §4, §6)

* Agents can only read approved sources: `newsren/ingest/sources.py::is_approved_url`
  gates the fetch tool; archive search only covers ingested approved sources.
  No unrestricted browsing, no social posting, no agent-triggered financial or
  educational decisions.
* Human approval is required before publication (configurable for non-finance
  low-risk items; **always** required for finance). Personal briefings are the
  only automatic delivery, and they are built only from published briefs and
  the reader's own alerts.
* Finance: disclaimer on every finance page, no buy/sell/hold calls, price
  targets, or personalized advice (advice requests in Ask Newsren are
  redirected to education), every performance claim labelled
  backtest/simulated/live with period, benchmark, and source, forecasts never
  presented as facts, public/licensed data only with timestamps and delay notes.
* Evidence: disagreements are described, not resolved silently; preprints are
  labelled; forecasts and supply-chain implications are labelled as
  projections/scenarios; education results name the studied population and
  outcome before generalizing; sources are never labelled "biased" merely for
  a different viewpoint.

## Data model

See `newsren/models.py`. Key tables: `sources`, `articles` (+ `articles_fts`),
`entities`, `stories`, `story_events`, `briefs`, `brief_claims`,
`editorial_actions`, `corrections`, `generated_content` (cached learning /
impact / evidence / product outputs), `users`, `interests`, `follows`,
`alerts`, `briefings`, `questions`, `feedback`, `quiz_attempts`, `page_views`,
`agent_runs`, and finance tables `companies`, `earnings_reports`,
`funding_rounds`, `ai_finance_claims`.

## Metrics (plan §9)

`newsren/metrics.py` computes trust (sampled-claim accuracy, material
corrections, citation coverage), usefulness (reader ratings), education
(comprehension and transfer accuracy, Newsren explanation vs plain summary),
engagement (return visits to followed stories, useful vs unwanted alerts),
editorial efficiency (review time, estimated time saved), and cost (average
inference + retrieval cost per story, question, and monitoring check from
`agent_runs`). Article volume and chatbot usage are deliberately not headline
metrics.
