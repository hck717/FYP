# The Agentic Investment Analyst
### *A Multi-Agent, Self-Improving RAG System for Fundamental Equity Analysis*

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange)
![Neo4j](https://img.shields.io/badge/Neo4j-GraphDB-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-StructuredDB-blue)
![Ollama](https://img.shields.io/badge/Ollama-LocalLLM-purple)

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/hck717/FYP.git
cd FYP
cp .env .env.bak              # back up existing .env if present, or create .env from scratch
                               # (fill in all API keys — see Section 11)

# 2. Create and activate a Python 3.11+ virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Start Ollama and pull required models  (Ollama must be running before Docker starts)
ollama serve &                 # skip if Ollama is already running as a system service
ollama pull llama3.2:latest    # planner + quant narrative
ollama pull deepseek-r1:8b     # business analyst + summarizer + financial modelling

# 4. Start all Docker services (PostgreSQL, Neo4j, Airflow x3)
docker compose up --build -d
sleep 60                        # wait for all containers to initialise and Airflow to migrate DB

# 5. Verify all backends are healthy
python ingestion/inspect_data.py
# Expected:
#   ✅ Connected to localhost:5432/airflow
#   ✅ raw_timeseries     516,197 rows
#   ✅ :Company           5 nodes

# 6. Run the full analysis pipeline (Python API)
python - <<'EOF'
from orchestration.graph import run
result = run("What is Apple's competitive moat and current valuation?")
print(result["final_summary"])
EOF

# 7. Launch the Streamlit UI
streamlit run POC/streamlit/app.py
# Opens at http://localhost:8501
```

**Service endpoints once running:**

| Service | URL | Credentials |
|---|---|---|
| Airflow UI | http://localhost:8080 | admin / admin |
| Neo4j Browser | http://localhost:7474 | neo4j / SecureNeo4jPass2025! |
| PostgreSQL | localhost:5432 | airflow / airflow |
| Ollama API | http://localhost:11434 | — |
| Streamlit UI | http://localhost:8501 | — |

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Architecture Overview](#2-architecture-overview)
3. [The Four Agent System (Implemented)](#3-the-four-agent-system-implemented)
4. [Orchestration Layer](#4-orchestration-layer)
5. [Database Architecture](#5-database-architecture)
6. [Ingestion Pipeline](#6-ingestion-pipeline)
7. [Running Agents](#7-running-agents)
8. [Running the Orchestration](#8-running-the-orchestration)
9. [Testing](#9-testing)
10. [Streamlit UI](#10-streamlit-ui)
11. [Environment Variables](#11-environment-variables)
12. [Project Roadmap](#12-project-roadmap)
13. [Evaluation Framework](#13-evaluation-framework)

---

## 1. Abstract

Current AI solutions in finance suffer from **"black box" opacity**, **hallucinations**, and an inability to reconcile conflicting signals from multiple data sources. While Large Language Models can summarise text, they lack the structural reasoning to provide reliable investment analysis or justify their conclusions with auditable evidence.

**The Agentic Investment Analyst** is an autonomous, multi-agent platform designed to replicate the workflow of a senior buy-side research team. It deploys **4 specialised domain agents** coordinated by a **LangGraph orchestration graph** operating on a **Global Plan-and-Execute / Local ReAct** architecture. Every agent uses a purpose-built RAG or deterministic reasoning strategy matched to its data type. All numeric outputs are Python-computed — never LLM-generated. Every qualitative claim cites a specific retrieved chunk ID.

**Supported tickers (Phase 1–2):** AAPL, MSFT, GOOGL, TSLA, NVDA

---

## 2. Architecture Overview

The platform operates in three runtime layers:

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1: INGESTION  (Offline / Airflow Scheduled)                  │
│                                                                     │
│  EODHD DAG (01:00 UTC) ──► scrape ──► load_postgres.py             │
│  FMP DAG   (02:00 UTC)              ──► load_neo4j.py               │
│                                                                     │
│  Databases: PostgreSQL 15 | Neo4j 5.15  (all Docker)               │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │ data at rest
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 2: INFERENCE  (Real-Time / LangGraph)                        │
│                                                                     │
│  [user query]                                                       │
│       │                                                             │
│       ▼                                                             │
│  node_planner  (llama3.2:latest)                                    │
│  ├── classifies intent & complexity (1=simple, 2=moderate, 3=full) │
│  ├── resolves ticker symbol(s)                                      │
│  ├── selects which agents to invoke                                 │
│  └── runs data availability check (Neo4j + PG + Ollama)            │
│       │                                                             │
│       ▼                                                             │
│  node_parallel_agents  (ThreadPoolExecutor)                         │
│  ├── Business Analyst  (CRAG + hybrid retrieval)                    │
│  ├── Quant Fundamental (deterministic SQL → Python math)            │
│  ├── Financial Modelling (DCF + Comps + Technicals)                 │
│  └── Web Search        (Perplexity sonar-pro, live web)             │
│       │                                                             │
│       ▼                                                             │
│  node_react_check                                                   │
│  └── loops back to parallel_agents if gaps + iterations remaining   │
│       │                                                             │
│       ▼                                                             │
│  node_summarizer  (deepseek-r1:8b)                                  │
│  └── synthesises all outputs → structured research note            │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │ final_summary + citations
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3: SYNTHESIS & UI  (Streamlit)                               │
│                                                                     │
│  POC/streamlit/app.py                                               │
│  ├── streaming CoT display (plan → agents → synthesise)             │
│  ├── per-agent result cards                                         │
│  └── final research note with citations                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Orchestration Pattern

| Level | Pattern | Component | Description |
|:---|:---|:---|:---|
| **Global** | Plan-and-Execute | `node_planner` (llama3.2:latest) | Classifies intent, selects agents, caps ReAct iterations |
| **Local** | ReAct loop | `node_react_check` | Re-runs gap agents; stops at complexity-derived iteration cap |
| **Synthesis** | Prompt chaining | `node_summarizer` (deepseek-r1:8b) | Narrates all agent outputs into a 11-section research note |

---

## 3. The Four Agent System (Implemented)

### Agent 1: Business Analyst — "The Deep Reader"

**Status:** Complete | Live-tested, all 5 tickers validated

**Goal:** Extract strategy, risk, and moat from financial news/filings with zero hallucinations.

**Architecture:** Graph-Augmented Corrective RAG (CRAG) — 8-node LangGraph pipeline

```
Query + Ticker
    │
    ▼
fetch_sentiment_data   ←  PostgreSQL: sentiment_trends table (bullish/bearish/neutral %)
    │
    ▼
hybrid_retrieval       ←  Neo4j vector index (384-dim, all-MiniLM-L6-v2)  [primary when chunks exist]
                       ←  Web search fallback (Perplexity sonar-pro)       [current primary — no Chunk nodes yet]
                       ←  BM25 keyword scoring
    │
    ▼
hybrid_rerank          ←  30% BM25 + 70% Cross-Encoder (ms-marco-MiniLM-L-6-v2)
    │
    ▼
crag_evaluate          ←  Confidence score:  CORRECT (>0.55) / AMBIGUOUS (0.35–0.55) / INCORRECT (<0.35)
    │
    ├─ CORRECT    ──►  generate_analysis   (deepseek-r1:8b on retrieved context)
    ├─ AMBIGUOUS  ──►  rewrite_query  ──►  retry retrieval (max 1 loop)
    └─ INCORRECT  ──►  web_search_fallback (calls Web Search Agent)
    │
    ▼
format_json_output     ←  Structured JSON with chunk_id citations
```

**Key design choices:**
- All JSON fields cite `neo4j::TICKER::title-slug` chunk IDs — no unsourced claims
- `_strip_ungrounded_inline_citations()` post-processes output to remove hallucinated IDs
- `deepseek-r1:8b` with `think=False`; defensive `_strip_think_tags()` also applied
- Neo4j is the primary vector source (no Chunk nodes yet — vector search returns 0); web search fallback active for all qualitative queries

**CLI:**
```bash
# Default task: competitive moat
.venv/bin/python -m agents.business_analyst.agent --ticker AAPL --log-level WARNING

# Custom task
.venv/bin/python -m agents.business_analyst.agent \
  --ticker TSLA \
  --task "What are Tesla's key business risks and competitive vulnerabilities?" \
  --log-level WARNING
```

**Programmatic API:**
```python
from agents.business_analyst import run, run_full_analysis

result = run(task="What is Apple's competitive moat?", ticker="AAPL")
dossier = run_full_analysis(ticker="AAPL")   # all 5 pillars in one pass
```

---

### Agent 2: Quantitative Fundamental — "The Math Auditor"

**Status:** Complete | Live-tested, all 5 tickers validated

**Goal:** Compute fundamental factors deterministically — all numbers from Python/SQL, never LLM.

**Architecture:** 8-node deterministic LangGraph pipeline (Non-RAG)

```
ticker / natural-language prompt
    │
    ▼
fetch_financials         ←  PostgreSQL: raw_fundamentals
                               (ratios_ttm, key_metrics_ttm, financial_scores,
                                earnings_history, analyst_estimates_eodhd)
    │
    ▼
chain_of_table_reasoning ←  SELECT → FILTER → CALCULATE → RANK → IDENTIFY
    │
    ▼
data_quality_check       ←  Validates field presence + numeric ranges
    │
    ▼
calculate_value_factors  ←  P/E trailing, EV/EBITDA, P/FCF, EV/Revenue
    │
    ▼
calculate_quality_factors←  ROE, ROIC, Piotroski F-Score (9-pt), Beneish M-Score
    │
    ▼
calculate_momentum_risk  ←  Beta (60-day), Sharpe Ratio (12M), 12M return
    │
    ▼
flag_anomalies           ←  Z-score > 2 on gross_margin, ebit_margin, roe
    │
    ▼
format_json_output       ←  LLM writes quantitative_summary only (no arithmetic)
```

**CLI:**
```bash
# Direct ticker
.venv/bin/python -m agents.quant_fundamental.agent --ticker AAPL --log-level WARNING
.venv/bin/python -m agents.quant_fundamental.agent --ticker MSFT --log-level WARNING
.venv/bin/python -m agents.quant_fundamental.agent --ticker GOOGL --log-level WARNING
.venv/bin/python -m agents.quant_fundamental.agent --ticker TSLA --log-level WARNING
.venv/bin/python -m agents.quant_fundamental.agent --ticker NVDA --log-level WARNING

# Natural-language prompt (ticker extracted automatically)
.venv/bin/python -m agents.quant_fundamental.agent \
  --prompt "Analyze AAPL fundamentals" --log-level WARNING

# Override LLM model
LLM_MODEL_QUANTITATIVE=deepseek-r1:8b \
  .venv/bin/python -m agents.quant_fundamental.agent --ticker AAPL
```

**Programmatic API:**
```python
from agents.quant_fundamental.agent import run, run_full_analysis

result = run(ticker="AAPL")
result = run(prompt="Analyze AAPL fundamentals")
result["value_factors"]["pe_trailing"]         # → 33.08 (Python-computed)
result["quality_factors"]["piotroski_f_score"] # → 9
```

**Live results (validated):**

| Ticker | Data Quality | Piotroski | P/E | Sharpe 12M |
|---|---|---|---|---|
| AAPL | PASSED (9/9) | 9 | 33.08 | 0.69 |
| MSFT | PASSED (9/9) | 7 | 24.47 | -5.19 |
| GOOGL | PASSED (9/9) | 7 | 28.48 | -3.66 |
| TSLA | PASSED (9/9) | 6 | 342.78 | -2.49 |
| NVDA | PASSED (9/9) | 6 | 35.87 | -1.76 |

---

### Agent 3: Financial Modelling — "The Valuation Engine"

**Status:** Complete | Live-tested, all 5 tickers validated

**Goal:** Build rigorous, assumption-driven valuation models — all numbers are computed, never LLM-generated.

**Architecture:** 8-node deterministic computation engine (Non-RAG)

```
Query + Ticker
    │
    ▼
fetch_price_history      ←  PostgreSQL: raw_timeseries (1-year EOD, split-adjusted)
    │
    ▼
fetch_fundamentals       ←  PostgreSQL: raw_fundamentals (PE, EPS, EBITDA, FCF, Debt)
    │
    ▼
fetch_earnings_history   ←  PostgreSQL: raw_fundamentals (earnings_history)
    │
    ▼
calculate_technicals     ←  SMA 20/50/200, EMA 12/26, RSI(14), MACD, Bollinger Bands,
                             ATR(14), HV30, Stochastic Oscillator
    │
    ▼
run_dcf_model            ←  Python: 5-year FCF projection + WACC (CAPM) + Terminal Value
                             → Bear / Base / Bull scenarios
                             → Sensitivity matrix (WACC × Terminal Growth Rate)
    │
    ▼
run_comparable_analysis  ←  PostgreSQL: peer EV/EBITDA, P/E, EV/Revenue
                             Neo4j: COMPETES_WITH / BELONGS_TO edges for peer selection
    │
    ▼
assess_analyst_estimates ←  EPS/Revenue consensus vs actuals, surprise %, beat streak
    │
    ▼
format_json_output       ←  deepseek-r1:8b writes quantitative_summary narrative only
```

**Models implemented:**

| Model | Key Outputs |
|---|---|
| **DCF** | Intrinsic value (Bear/Base/Bull), WACC via CAPM, sensitivity matrix (WACC × g) |
| **Comps** | EV/EBITDA, P/E, P/S, EV/Revenue vs peer median; `vs_sector_avg` premium/discount |
| **Technicals** | RSI, MACD, Bollinger Bands, ATR, HV30, SMA 50/200, Golden/Death Cross |
| **Factor Scores** | Piotroski F-Score, Beneish M-Score, Altman Z-Score |

**CLI:**
```bash
# Full valuation dossier
python -m agents.financial_modelling.agent --ticker AAPL --log-level WARNING
python -m agents.financial_modelling.agent --ticker NVDA --log-level WARNING

# With explicit LLM model
LLM_MODEL_FINANCIAL_MODELING=deepseek-r1:8b \
  python -m agents.financial_modelling.agent --ticker MSFT --log-level WARNING
```

**Programmatic API:**
```python
from agents.financial_modelling import run, run_full_analysis

dossier = run_full_analysis(ticker="AAPL")
dossier["valuation"]["dcf"]["intrinsic_value_base"]  # → 195.20
dossier["technicals"]["rsi_14"]                       # → 58.3
dossier["earnings"]["beat_streak"]                    # → 6
```

---

### Agent 4: Web Search — "The News Desk"

**Status:** Complete | Live-tested | 6/6 tests passing

**Goal:** Surface "unknown unknowns" — breaking developments not yet in the local static knowledge base.

**Architecture:** HyDE + Step-Back Prompting → Perplexity sonar-pro (live web)

```
Query + Ticker
    │
    ▼
Step-Back Prompting   →  "What macro/sector/competitor events could affect this company?"
    │
    ▼
HyDE                  →  Generate hypothetical ideal news article → embed → guide search
    │
    ▼
Perplexity sonar-pro  →  Live web search with freshness reranking (<24h preferred)
    │
    ▼
Hallucination Guard   →  Every factual claim requires URL + date
                          Single-source claims labelled "UNCONFIRMED"
    │
    ▼
format_json_output    →  breaking_news[], sentiment_signal, unknown_risk_flags[],
                          competitor_signals[], supervisor_escalation
```

**CLI / Smoke test:**
```bash
# Verify API key loaded
python - <<'EOF'
from agents.web_search.tools import PERPLEXITY_API_KEY, DEFAULT_MODEL
print("Key loaded:", bool(PERPLEXITY_API_KEY))
print("Model:", DEFAULT_MODEL)
EOF

# Live smoke test
python - <<'EOF'
from agents.web_search.agent import run_web_search_agent
import json
result = run_web_search_agent({
    "query": "NVDA latest regulatory risk Q1 2026",
    "ticker": "NVDA",
    "recency_filter": "week",
    "model": "sonar-pro"
})
print(json.dumps(result, indent=2))
EOF
```

**Required `.env` key:**
```bash
PERPLEXITY_API_KEY=pplx-xxxxxxxxxxxxxxxx
```

---

## 4. Orchestration Layer

The orchestration layer (`/orchestration/`) coordinates the four agents via a LangGraph `StateGraph`.

### Graph Topology

```
[planner] ──► [parallel_agents] ──► [react_check] ──┐
                    ▲                                │ loop if gaps + iterations left
                    └────────────────────────────────┘
                                    │ all done
                                    ▼
                              [summarizer] ──► END
```

### Key Files

| File | Purpose |
|---|---|
| `graph.py` | `build_graph()` (parallel, default) + `build_sequential_graph()` (debug) + `run()` + `stream()` |
| `nodes.py` | All 8 node functions: planner, parallel_agents, 4 agent wrappers, react_check, summarizer |
| `state.py` | `OrchestrationState` TypedDict — the shared state schema |
| `llm.py` | `plan_query()` (llama3.2) + `summarise_results()` (deepseek-r1:8b) + prompts |
| `citations.py` | `build_citation_block()` + `inject_inline_numbers()` — `[N]` reference section |
| `data_availability.py` | `check_all()` — concurrent ping of Neo4j, PostgreSQL, Ollama |

### Programmatic Usage

```python
from orchestration.graph import run, stream

# Blocking call — returns full final state dict
result = run("What is Apple's competitive moat and current valuation?")
print(result["final_summary"])
print(result["ticker"])          # "AAPL"
print(result["plan"])            # {"agents": ["business_analyst", "quant_fundamental", ...]}

# Streaming — yields (node_name, partial_state) tuples as each node completes
for node_name, node_output in stream("Compare MSFT vs AAPL"):
    print(f"[{node_name}] completed")

# Force sequential mode (debug)
import os
os.environ["ORCHESTRATION_SEQUENTIAL"] = "1"
result = run("Simple AAPL P/E query")
```

### ReAct Loop Behaviour

| Complexity | Max Passes | Triggered By |
|---|---|---|
| 1 (simple look-up) | 1 | Single metric questions |
| 2 (moderate analysis) | 2 | Multi-factor analysis |
| 3 (full report) | 3 | Comparative / full fundamental analysis |

On each loop, only agents with **no output** or an **error** are re-run. Successful agents are never re-executed.

### Planner Models

| Node | Model | Justification |
|---|---|---|
| `node_planner` | `llama3.2:latest` | Fast (~3s); reliable JSON routing |
| `node_summarizer` | `deepseek-r1:8b` | Deep analytical prose for 11-section research note |

---

## 5. Database Architecture

All data runs **100% locally via Docker** — no cloud databases.

| Database | Container | Port | Role |
|---|---|---|---|
| PostgreSQL 15 | `fyp-postgres` | 5432 | Structured time-series + fundamentals |
| Neo4j 5.15 | `fyp-neo4j` | 7474/7687 | Graph relationships + company properties |

### PostgreSQL Schema

```sql
-- Time-series (prices, intraday, technicals, macro)
raw_timeseries   (id, agent_name, ticker_symbol, data_name, ts_date, payload JSONB, source, ingested_at)

-- Snapshot/fundamental data (ratios, estimates, filings)
raw_fundamentals (id, agent_name, ticker_symbol, data_name, as_of_date, payload JSONB, source, ingested_at)

-- Global datasets (ingested once per day)
market_eod_us            (id, ts_date, payload JSONB, source, ingested_at)
global_economic_calendar (id, ts_date, payload JSONB, source, ingested_at)
global_ipo_calendar      (id, ts_date, payload JSONB, source, ingested_at)

-- Agent query logs (Phase 3 — self-improvement)
query_logs       (query_id UUID, query_text, agents_invoked[], latency_ms, timestamp)
citation_tracking(query_id, chunk_id, cited BOOL, user_rating FLOAT)
sentiment_trends (ticker_symbol, bullish_pct, bearish_pct, neutral_pct, trend, updated_at)
```

**Inspect PostgreSQL:**
```bash
docker exec -it fyp-postgres psql -U airflow -d airflow
\dt                                               -- list all tables
SELECT ticker_symbol, COUNT(*) FROM raw_timeseries GROUP BY ticker_symbol;
SELECT MAX(ingested_at) FROM raw_fundamentals;
\q
```

### Neo4j Schema

```cypher
// Nodes (currently populated)
(:Company {ticker, Name, Sector, Industry, Exchange,
           Highlights_MarketCapitalization, Highlights_PERatio,
           Highlights_ProfitMargin, Valuation_TrailingPE,
           Valuation_ForwardPE, Valuation_EnterpriseValue, ...})  // 85+ properties

// Relationships (Phase 3 — not yet populated)
(:Company)-[:COMPETES_WITH]->(:Company)
(:Company)-[:BELONGS_TO]->(:Sector)
(:Company)-[:DEPENDS_ON]->(:Technology)
(:Company)-[:HAS_RISK]->(:RiskFactor)
```

**Inspect Neo4j:**
```bash
# Browser: http://localhost:7474
# Bolt:    bolt://localhost:7687  (neo4j / SecureNeo4jPass2025!)

# CLI
docker exec -it fyp-neo4j cypher-shell -u neo4j -p SecureNeo4jPass2025!
MATCH (c:Company) RETURN c.ticker, c.Name, c.Highlights_MarketCapitalization LIMIT 10;
```

---

## 6. Ingestion Pipeline

See [`ingestion/README.md`](ingestion/README.md) for detailed operations guide.

Two Airflow DAGs ingest data daily:

| DAG | Schedule | Purpose |
|---|---|---|
| `eodhd_complete_ingestion` | Daily 01:00 UTC | News, prices, fundamentals, technicals → PG + Neo4j |
| `fmp_complete_ingestion` | Daily 02:00 UTC | Analyst estimates, earnings, DCF inputs, insider trades → PG |

**Trigger DAGs manually:**
```bash
# Get scheduler container ID
SCHED=$(docker compose ps -q airflow-scheduler)

# List all DAGs
docker exec -it $SCHED airflow dags list

# Trigger EODHD ingestion
docker exec -it $SCHED airflow dags unpause eodhd_complete_ingestion
docker exec -it $SCHED airflow dags trigger eodhd_complete_ingestion

# Check run status
docker exec -it $SCHED airflow dags list-runs -d eodhd_complete_ingestion

# View task logs
docker exec -it $SCHED airflow tasks logs eodhd_complete_ingestion \
  eodhd_load_neo4j_business_analyst_AAPL 2026-02-24
```

**Health check:**
```bash
source .venv/bin/activate
python ingestion/inspect_data.py
# Expected output:
# ✅ Connected to localhost:5432/airflow
# ✅ raw_timeseries     516,197 rows
# ✅ :Company           5 nodes
```

---

## 7. Running Agents

All agents require the Docker stack to be running (`docker compose up -d`) and Ollama serving locally.

### Business Analyst Agent

```bash
# Suggested test queries (all validated with live data)
.venv/bin/python -m agents.business_analyst.agent --ticker AAPL --log-level WARNING
.venv/bin/python -m agents.business_analyst.agent \
  --ticker TSLA --task "What are Tesla's key business risks?" --log-level WARNING
.venv/bin/python -m agents.business_analyst.agent \
  --ticker NVDA --task "How defensible is NVIDIA's AI chip moat?" --log-level WARNING
.venv/bin/python -m agents.business_analyst.agent \
  --ticker MSFT --task "Assess Microsoft's cloud and AI services strategy." --log-level WARNING
.venv/bin/python -m agents.business_analyst.agent \
  --ticker GOOGL --task "What is Alphabet's advertising dependency risk?" --log-level INFO
```

### Quant Fundamental Agent

```bash
.venv/bin/python -m agents.quant_fundamental.agent --ticker AAPL --log-level WARNING
.venv/bin/python -m agents.quant_fundamental.agent --ticker MSFT --log-level WARNING
.venv/bin/python -m agents.quant_fundamental.agent --ticker GOOGL --log-level WARNING
.venv/bin/python -m agents.quant_fundamental.agent --ticker TSLA --log-level WARNING
.venv/bin/python -m agents.quant_fundamental.agent --ticker NVDA --log-level WARNING
.venv/bin/python -m agents.quant_fundamental.agent \
  --prompt "Run quant fundamental analysis for NVDA" --log-level WARNING
```

### Financial Modelling Agent

```bash
python -m agents.financial_modelling.agent --ticker AAPL --log-level WARNING
python -m agents.financial_modelling.agent --ticker TSLA --log-level WARNING
python -m agents.financial_modelling.agent --ticker NVDA --log-level WARNING
python -m agents.financial_modelling.agent --ticker MSFT --log-level WARNING
python -m agents.financial_modelling.agent --ticker GOOGL --log-level WARNING
```

### Web Search Agent

```bash
# Requires PERPLEXITY_API_KEY in .env
python - <<'EOF'
from agents.web_search.agent import run_web_search_agent
import json
for ticker in ["AAPL", "NVDA", "TSLA"]:
    result = run_web_search_agent({"query": f"{ticker} latest news Q1 2026",
                                   "ticker": ticker, "recency_filter": "week"})
    print(f"{ticker}: sentiment={result['sentiment_signal']}, "
          f"risks={len(result['unknown_risk_flags'])}, error={result['error']}")
EOF
```

---

## 8. Running the Orchestration

The orchestration layer requires all 4 agents' backends to be running.

```python
from orchestration.graph import run, stream

# Single ticker — full analysis
result = run("What is Apple's competitive moat and current valuation?")
print(result["final_summary"])

# Multi-ticker comparison
result = run("Compare MSFT vs AAPL cloud strategy and valuation")
print(result["tickers"])          # ["MSFT", "AAPL"]

# Simple metric look-up (complexity 1 — single pass, no retry)
result = run("What is NVDA's current P/E ratio?")

# Streaming (for UI integration)
for node_name, node_output in stream("Full fundamental analysis of TSLA"):
    if node_name == "planner":
        print("Plan:", node_output.get("plan"))
    elif node_name == "parallel_agents":
        print("Agents completed this pass")
    elif node_name == "summarizer":
        print("Summary ready")

# Debug mode — one agent at a time
import os; os.environ["ORCHESTRATION_SEQUENTIAL"] = "1"
result = run("AAPL P/E check")
```

**State dict keys returned:**
```python
result["final_summary"]                    # str — DeepSeek research note
result["ticker"]                           # str — primary ticker
result["tickers"]                          # list[str] — all tickers
result["plan"]                             # dict — planner's routing decision
result["business_analyst_output"]          # dict — BA agent result
result["quant_fundamental_output"]         # dict — QF agent result
result["financial_modelling_output"]       # dict — FM agent result
result["web_search_output"]               # dict — WS agent result
result["agent_errors"]                     # dict — any agent errors
result["react_steps"]                      # list — ReAct trace
```

---

## 9. Testing

```bash
# Run all tests
.venv/bin/python -m pytest agents/ -v

# Per-agent test suites
.venv/bin/python -m pytest agents/business_analyst/tests/     -v   # 39 tests
.venv/bin/python -m pytest agents/quant_fundamental/tests/    -v   # 44 tests
.venv/bin/python -m pytest agents/financial_modelling/tests/  -v
.venv/bin/python -m pytest agents/web_search/tests/           -v   # 6 tests

# Exclude integration tests (no live Docker required)
.venv/bin/python -m pytest agents/ -m "not integration" -v

# Fast smoke test for a single agent
.venv/bin/python -m pytest agents/quant_fundamental/tests/ -v -k "test_run"
```

All tests use mocked external services (Neo4j, PostgreSQL, Ollama). No live infrastructure required for the test suite.

---

## 10. Streamlit UI

The Streamlit POC is at `POC/streamlit/app.py`.

```bash
# Install dependencies (if running outside the main .venv)
pip install streamlit plotly

# Launch
streamlit run POC/streamlit/app.py
# Opens at http://localhost:8501
```

**Features:**
- Query input with example queries and ticker selector
- Real-time streaming progress display (planner → agents → summariser)
- Per-agent collapsible result cards
- DCF scenario table and implied price range
- Technical indicators summary
- Final research note rendering
- Agent error and data availability status

See [`POC/streamlit/`](POC/streamlit/) for the full implementation.

---

## 11. Environment Variables

Create `.env` at the repo root with the following variables:

```bash
# ── API Keys ───────────────────────────────────────────────────────────────
FMP_API_KEY=your_fmp_key
EODHD_API_KEY=your_eodhd_key
FRED_API_KEY=your_fred_key
PERPLEXITY_API_KEY=pplx-xxxxxxxxxxxxxxxx   # Web Search Agent (required)
OPENAI_API_KEY=sk-...                       # reserved for Phase 3 Critic Agent
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=...

# ── Databases ─────────────────────────────────────────────────────────────
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=airflow
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=SecureNeo4jPass2025!

# ── LLM (Ollama local) ────────────────────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434

# Orchestration
ORCHESTRATION_PLANNER_MODEL=llama3.2:latest
ORCHESTRATION_SUMMARIZER_MODEL=deepseek-r1:8b
ORCHESTRATION_SEQUENTIAL=0                  # set to 1 for sequential debug mode
# ORCHESTRATION_LLM_TIMEOUT=60             # planner timeout (s); unset = no cap
# ORCHESTRATION_SUMMARIZER_TIMEOUT=1200    # summarizer timeout (s); unset = no cap

# Agent LLMs
BUSINESS_ANALYST_MODEL=deepseek-r1:8b
LLM_MODEL_QUANTITATIVE=llama3.2:latest
LLM_MODEL_FINANCIAL_MODELING=deepseek-r1:8b

# Embeddings
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# ── Agent Tuning (optional) ───────────────────────────────────────────────
BUSINESS_ANALYST_TEMPERATURE=0.2
BUSINESS_ANALYST_MAX_TOKENS=3000
CRAG_CORRECT_THRESHOLD=0.55
CRAG_AMBIGUOUS_THRESHOLD=0.35
RAG_TOP_K=8
RAG_SCORE_THRESHOLD=0.6

QUANT_LLM_TEMPERATURE=0.1
QUANT_LLM_MAX_TOKENS=512
QUANT_REQUEST_TIMEOUT=120
ANOMALY_ZSCORE_THRESHOLD=2.0
BENEISH_THRESHOLD=-2.22
PIOTROSKI_STRONG_THRESHOLD=7

PRICE_HISTORY_DAYS=365
DCF_FORECAST_YEARS=5
DCF_TERMINAL_GROWTH_RATE=0.025
DCF_WACC=0.09

WEB_SEARCH_MODEL=sonar-pro
WEB_SEARCH_RECENCY_FILTER=week

# ── Ingestion ─────────────────────────────────────────────────────────────
TRACKED_TICKERS=AAPL,GOOGL,MSFT,NVDA,TSLA

# ── Airflow ───────────────────────────────────────────────────────────────
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
AIRFLOW__WEBSERVER__SECRET_KEY=your-secret-key
AIRFLOW_UID=50000
AIRFLOW_WWW_USER_USERNAME=admin
AIRFLOW_WWW_USER_PASSWORD=admin
```

---

## 12. Project Roadmap

**Duration:** February 1 – April 30, 2026 (13 weeks)

### Phase 1 — Foundation & Data Infrastructure (Weeks 1–4)  COMPLETE

- [x] Docker Compose: PostgreSQL, Neo4j, Airflow
- [x] EODHD + FMP API integration and Airflow DAGs
- [x] PostgreSQL schema: `raw_timeseries`, `raw_fundamentals`
- [x] ETL loaders: `load_postgres.py`, `load_neo4j.py`
- [x] Idempotent MD5 hash deduplication

### Phase 2 — Agent Layer (Weeks 3–8)  IN PROGRESS

- [x] Business Analyst Agent (CRAG + hybrid retrieval) — 39 tests passing
- [x] Quantitative Fundamental Agent (8-node deterministic) — 44 tests passing
- [x] Financial Modelling Agent (DCF + Comps + Technicals) — all 5 tickers live
- [x] Web Search Agent (Perplexity sonar-pro) — 6 tests passing
- [x] LangGraph orchestration graph (parallel + sequential)
- [ ] `agents/consensus_strategy/` — Contrastive RAG, Gap Analysis engine
- [ ] `agents/macro_economic/` — Text-to-SQL, economic cycle classifier
- [ ] `agents/insider_sentiment/` — Temporal Contrastive RAG, Form 4 reader

### Phase 3 — Trust, UI & Polish (Weeks 9–13)

- [x] Streamlit POC UI (`POC/streamlit/app.py`)
- [ ] Critic Agent (GPT-4o-mini NLI verification, ≥90% pass rate)
- [ ] Self-Improving RAG (`SelfImprovingRAG` class, Neo4j boost-factor updates)
- [ ] Weekly retraining DAG (`citation_tracking` → `boost_factor` update)
- [ ] Full Streamlit UI (`ui/app.py`): streaming CoT, DCF heatmap, clickable citations
- [ ] 50-query evaluation dataset (20 Q&A / 15 Comparison / 15 Full Analysis)
- [ ] 5 demo reports (AAPL, TSLA, NVDA, MSFT, GOOGL)

---

## 13. Evaluation Framework

| Metric | Definition | Target |
|:---|:---|:---|
| Citation Verification Rate (CVR) | % factual claims passing NLI check | ≥ 95% |
| Numerical Accuracy Score (NAS) | Python vs SQL dual-path agreement | 100% |
| Retrieval Precision@5 | Relevance of top-5 chunks (human annotated) | ≥ 0.85 |
| Citation Utilisation Rate (CUR) | % retrieved chunks cited in final answer | ≥ 60% |
| Query Resolution Rate (QRR) | % queries answered without "insufficient data" | ≥ 92% |
| Full Analysis Latency | End-to-end wall-clock for 4-agent parallel run | < 30s |
| Self-Improvement Rate | Citation precision gain after 1 week of usage | +5%/week |

---

## Directory Structure

```
FYP/
├── README.md                     ← This file
├── .env                          ← API keys and configuration (not committed — create manually)
├── docker-compose.yml            ← 5-service stack (PG, Neo4j, Airflow x3)
├── pyproject.toml                ← pytest configuration
├── conftest.py                   ← sys.path setup for pytest
│
├── orchestration/                ← LangGraph multi-agent pipeline
│   ├── README.md
│   ├── graph.py                  ← build_graph(), run(), stream()
│   ├── nodes.py                  ← all 8 LangGraph node functions
│   ├── state.py                  ← OrchestrationState TypedDict
│   ├── llm.py                    ← plan_query(), summarise_results()
│   ├── citations.py              ← build_citation_block(), inject_inline_numbers()
│   └── data_availability.py      ← check_all() — backend health check
│
├── agents/
│   ├── README.md                 ← Agent layer overview
│   ├── business_analyst/         ← CRAG pipeline (deepseek-r1:8b, Neo4j + BM25 + web search)
│   ├── quant_fundamental/        ← Deterministic factor engine (llama3.2, PostgreSQL only)
│   ├── financial_modelling/      ← DCF + Comps + Technicals (deepseek-r1:8b, PostgreSQL)
│   └── web_search/               ← Perplexity sonar-pro (live web)
│
├── ingestion/
│   ├── README.md                 ← Ingestion operations guide (English)
│   ├── inspect_data.py           ← Health check script
│   ├── dags/                     ← Airflow DAG definitions
│   └── etl/                      ← ETL loaders + agent_data/ cache
│
├── POC/
│   └── streamlit/
│       ├── app.py                ← Streamlit UI app
│       └── requirements.txt
│
├── ui/                           ← Full production UI (Phase 3 — not yet built)
├── rag/                          ← RAG utilities
├── skills/                       ← Standalone agent skills (prototypes)
├── data/                         ← Local data cache
└── docs/
    └── setup_guide.md
```

---

*Last updated: 2026-03-05 | Author: hck717*
