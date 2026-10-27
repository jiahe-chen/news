# The Newsren

**Agentic news intelligence for technology, education, industry, and finance.**

The Newsren moves from "a website that publishes news" to "an AI news agent
that researches, explains, connects, and helps readers act on news". It
monitors approved sources, cross-checks claims, connects developments into
evolving stories, explains them for students, educators, and professionals,
and follows readers' questions over time. Every claim links to its sources.

> Implementation of *The Newsren: Agentic News Intelligence Plan* (October 2026).
> See [docs/PLAN_TRACEABILITY.md](docs/PLAN_TRACEABILITY.md) for a section-by-section
> map from the plan to code and tests.

## What it does

| Plan feature | Where |
|---|---|
| **AI Story Brief + Ask Newsren** — one concise summary, key evidence, related stories, and a research assistant grounded in cited sources | `/briefs/{id}`, `/ask`, "Ask about this story" tool |
| **News-to-Learning** — one-click student, educator, and professional explanations, quizzes (comprehension + transfer), classroom discussion guides | "Learn this topic" tool |
| **Follow This Story / Follow a Topic** — save a question with criteria; alerts only when a development meets them (e.g. *an independent benchmark, not another opinion piece*) | "Follow developments" tool, `/me` |
| **My Newsren** — personal research agent: interests, technical depth, briefings that explain what changed and why it matters | `/me`, email briefings (opt-in) |
| **Story Intelligence** — related developments grouped into evolving stories with timelines: what happened first, what's new, what's disputed, how the evidence changed | `/stories/{id}`, "Connected stories" tool |
| **Evidence and Bias Inspection** — original vs secondary sources, claims compared across independent reporting, facts vs interpretations, study design — without labelling a source "biased" for its viewpoint | "Inspect evidence" tool |
| **Industry Impact** — implications for industries, occupations, and decisions, labelled supported vs speculative | "Industry impact" tool |
| **13 vertical editorial products** — Launch Reality Check, Research Paper Decoder, Technology Evolution Timeline, Policy Impact Explorer, Classroom Discussion Builder, Education Evidence Checker, Company and Competitor Watch, Supply-Chain Ripple Map, Skills and Career Trends, Earnings Decoder, Funding Round Tracker, AI-in-Finance Reality Check, Regulation Watch | product panel on each brief; `/finance/*` |
| **Finance section** with guardrails — earnings vs prior periods with every number quoted from the filing (SEC EDGAR XBRL), funding tracker (never a ranking), AI-in-finance claims classified announcement / pilot / backtest / live / regulatory finding, finance concepts for learners, regulation watch | `/finance` |
| **Editorial quality gate** — automated checks + human review queue (finance always reviewed), corrections, claim sampling | `/editor` |
| **Success metrics** (plan §9) — trust, usefulness, education, engagement, editorial efficiency, cost | `/analytics`, `newsren metrics` |

## Quick start

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run newsren demo      # fictional demo corpus -> full workflow, offline (no API key, no network)
uv run newsren serve     # http://localhost:8000
```

`newsren demo` loads clearly-labelled **fictional** demo articles covering every
plan use case (an AI model launch followed by independent evaluation, a
product launch, school adoption, and regulation; a university press release
vs. the actual study; a semiconductor capacity announcement; a bank's
backtest; funding rounds; earnings; enforcement; an education policy), runs
the whole workflow, and creates two accounts:

| Role | Email | Password |
|---|---|---|
| Editor | `editor@newsren.example` | `editor-demo-pass` |
| Reader | `reader@newsren.example` | `reader-demo-pass` |

Finance and high-risk drafts stay in the editor's review queue to show the
human-review gate. (Use `--keep-queue` to leave every draft for review.)

### Live sources

```bash
uv run newsren init                 # register the approved source registry
uv run newsren ingest               # fetch all approved sources
uv run newsren pipeline             # discover -> connect -> explain -> verify -> output
uv run newsren finance --ticker MSFT   # Earnings Decoder from SEC EDGAR XBRL facts
```

The registry (`src/newsren/ingest/data/sources.json`) holds 27 verified,
approved sources: arXiv (cs.AI/CL/LG), BBC, Ars Technica, MIT Technology
Review, NIST, UK NCSC (technology); ERIC (IES), Chalkbeat, The Hechinger
Report, EdSurge, Inside Higher Ed, The 74 (education); BBC Business, Supply
Chain Dive, Manufacturing Dive, Semiconductor Engineering, EE Times, IEEE
Spectrum (industry); SEC press releases, SEC EDGAR 8-K and 10-Q feeds, Federal
Reserve, FINRA, TechCrunch Venture, Crunchbase News (finance). Publisher feeds
are stored as headline + summary + link (no full text); public-domain
government records are stored in full.

### Using Claude

Set `ANTHROPIC_API_KEY` (or put it in `.env`; see [.env.example](.env.example)).
With credentials present (`NEWSREN_LLM=auto`, the default) every agent runs on
**Claude Opus 5.5** (`claude-opus-5-5`, configurable with `NEWSREN_MODEL`) with
adaptive thinking, structured outputs validated against pydantic contracts,
prompt caching, and server-side refusal fallback. The research agent runs a
budgeted tool loop over approved retrieval tools only (archive search, article
reader, arXiv, SEC EDGAR, approved-domain fetch).

Without credentials, the **offline evidence engine** answers every task
deterministically from the same evidence (extractive, rule-based). The UI
always says which engine drafted an item.

## Commands

| Command | Purpose |
|---|---|
| `newsren init` | Create the database and register approved sources |
| `newsren demo [--network] [--keep-queue]` | Demo corpus + full workflow + demo accounts |
| `newsren ingest [--source SLUG]` | Fetch approved sources |
| `newsren pipeline [--no-network] [--max-briefs N]` | Run the whole workflow |
| `newsren monitor` | Check every follow for material developments |
| `newsren briefings` | Build and deliver due personal briefings |
| `newsren finance [--network] [--ticker T]` | Refresh finance trackers / build an Earnings Decoder report |
| `newsren create-user --email E --password P [--role editor]` | Create an account |
| `newsren metrics` | Print plan §9 metrics as JSON |
| `newsren serve [--host H] [--port P]` | Run the web app |

Run the workflow on a schedule with cron (`newsren pipeline`) or in-process
with `NEWSREN_SCHEDULER=true`.

## How it works

```
Approved sources ─► Discover ─► Connect ─► Explain ─► Quality gate ─► Output
 (RSS, arXiv,       ingest,     entities,   briefs,     checks +         published stories,
  EDGAR, ERIC,      dedup,      stories,    learning,   human review     briefings, alerts
  regulators)       index       timelines   impact      (finance: all)        │
                                                                              ▼
                              Reader loop: Ask Newsren · Follow This Story · My Newsren
```

An **orchestrator** runs every agent task with time, tool-call, and cost
budgets and records each run (tokens, cost, duration, tool trace). Agents only
see an evidence pack of approved sources and may only cite what they were
given. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Module
contracts: [docs/CONTRACTS.md](docs/CONTRACTS.md).

### Guardrails

- Every generated item keeps its sources, publication timestamps, retrieval
  time, and the evidence for its main claims; disagreements between credible
  sources are described, not silently resolved.
- Preprints are distinguished from peer-reviewed work; forecasts and scenarios
  are labelled as projections; education results name the studied population
  and outcome before generalizing; vendor claims are separated from
  independent evaluations.
- Agents can only reach approved sources — no unrestricted browsing, no
  unapproved scraping, no social posting, no agent-triggered financial or
  educational decisions.
- Human approval before publication; automatic delivery only for low-risk
  personal briefings.
- **Finance is news and education only**: no buy/sell/hold calls, price
  targets, portfolio recommendations, personalized advice, trade execution, or
  brokerage connections; a notice on every finance page; every performance
  claim labelled backtest / simulated / live with period, benchmark, and
  source; forecasts never presented as facts; public data only, timestamped,
  with delay notes; human review for every finance story.

## Development

```bash
uv run pytest            # full suite (offline, no network)
uv run ruff check src tests && uv run ruff format --check src tests
```

Project layout:

```
src/newsren/
  config.py db.py models.py schemas.py textutil.py pipeline.py scheduler.py cli.py metrics.py
  llm/            provider-neutral tasks; Claude provider; offline provider; pricing
  agents/         orchestrator, brief, evidence, research (+ tools), learning, impact,
                  products, monitor, briefing
  ingest/         approved sources, fetchers, dedup, ingestion pipeline, demo corpus
  intelligence/   entities, events, clustering, timelines + story intelligence
  editorial/      guardrails, quality-gate checks, review workflow
  finance/        concepts, SEC EDGAR, funding, AI-in-finance claims, regulation, service
  users/          accounts, interests, follows, alerts
  delivery/       email (SMTP or outbox)
  web/            FastAPI app, routes, Jinja2 templates, CSS/JS (htmx)
tests/            pytest suite + saved source samples
docs/             architecture, contracts, API index, plan traceability
```

Docker: `docker build -t newsren . && docker run -p 8000:8000 -v newsren-data:/data newsren`.
