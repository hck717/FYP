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

# 3. Start Ollama and pull required models (Ollama must be running before Docker starts)
ollama serve &                 # skip if Ollama is already running as a system service
ollama pull llama3.2:latest    # planner + quant narrative
ollama pull deepseek-r1:8b     # business analyst + summarizer + financial modelling
ollama pull nomic-embed-text   # embedding model for vector search

# 4. Start all Docker services (PostgreSQL, Neo4j, Airflow x3)
docker compose up --build -d
sleep 60                        # wait for all containers to initialise and Airflow to migrate DB

# 5. Verify all backends are healthy
docker exec fyp-airflow-webserver python /opt/airflow/ingestion/etl/inspect_db.py
# Expected: All checks passed

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

Current AI solutions in finance suffer from **"black box"** opacity, **hallucinations**, and an inability to reconcile conflicting signals from multiple data sources. While Large Language Models can summarise text, they lack the structural reasoning to provide reliable investment analysis or justify their conclusions with auditable evidence.

**The Agentic Investment Analyst** is an autonomous, multi-agent platform designed to replicate the workflow of a senior buy-side research team. It deploys **4 specialised domain agents** coordinated by a **LangGraph orchestration graph** operating on a **Global Plan-and-Execute / Local ReAct** architecture. Every agent uses a purpose-built RAG or deterministic reasoning strategy matched to its data type. All numeric outputs are Python-computed — never LLM-generated. Every qualitative claim cites a specific retrieved chunk ID.

**Supported tickers (Phase 1–2):** AAPL, MSFT, GOOGL, TSLA, NVDA

---

## 2. Architecture Overview

The platform operates in three runtime layers:

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1: INGESTION  (Offline / Airflow Scheduled)                  │
│  EODHD API → PostgreSQL + Neo4j + Ollama Embeddings                 │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 2: AGENTS  (On-Demand / Parallel)                            │
│  • Business Analyst — Qualitative moat, sentiment, risk            │
│  • Quant Fundamental — Piotroski, Beneish, value/momentum           │
│  • Financial Modelling — DCF, comps, technicals                    │
│  • Web Search — Live news & events                                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3: ORCHESTRATION  (LangGraph Planner)                       │
│  • Query parsing → agent selection → parallel dispatch → synthesis  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Four Agent System (Implemented)

| Agent | Directory | Primary Responsibility | Key Data Sources |
|---|---|---|---|
| **Web Search** | `agents/web_search/` | Live breaking news, sentiment signals | Perplexity API |
| **Business Analyst** | `agents/business_analyst/` | Qualitative moat analysis, earnings calls, broker reports | Neo4j, PostgreSQL, Ollama |
| **Quant Fundamental** | `agents/quant_fundamental/` | Value/quality/momentum factors, Piotroski, Beneish | PostgreSQL |
| **Financial Modelling** | `agents/financial_modelling/` | DCF valuation, peer comps, technicals | PostgreSQL, Neo4j |

---

## 4. Orchestration Layer

See `orchestration/README.md` for detailed architecture.

**Key Features:**
- **Planner Node**: Uses `llama3.2:latest` for fast query parsing
- **Parallel Execution**: All agents run concurrently via ThreadPoolExecutor
- **Summarizer**: Uses `deepseek-r1:8b` for final narrative synthesis

---

## 5. Database Architecture

### PostgreSQL (Structured Data)
- `raw_timeseries` — Historical OHLCV data
- `sentiment_trends` — Bullish/bearish/neutral sentiment per ticker
- `financial_statements` — Balance sheet, income statement, cash flow
- `valuation_metrics` — PE, PB, PS ratios
- `insider_transactions` — Insider buying/selling
- `institutional_holders` — 13F holdings
- `earnings_surprises` — EPS actuals vs estimates
- `text_chunks` — pgvector embeddings for semantic search

### Neo4j (Graph + Documents)
- `:Company` nodes — 85+ properties per ticker (marketCap, PE, sector, etc.)
- `:Chunk` nodes — Text embeddings from:
  - Company profiles
  - Earnings call transcripts
  - Broker research reports
  - ESG, highlights, valuation sections

### Vector Search
- **Neo4j**: Native vector index (`chunk_embedding`) for semantic search
- **pgvector**: Alternative vector storage in PostgreSQL

---

## 6. Ingestion Pipeline

See `ingestion/README.md` for detailed documentation.

**DAG**: `eodhd_complete_ingestion` runs daily at 01:00 UTC

**Data Sources:**
- EODHD API (primary)
- PDF earnings calls (local files)
- PDF broker reports (local files)

**Textual Data Processing:**
- `ingest_earnings_calls.py` — Extracts text from PDF earnings calls → Neo4j
- `ingest_broker_reports.py` — Extracts text from PDF broker reports → Neo4j

---

## 7. Running Agents

```bash
# Business Analyst (inside Docker - recommended)
docker exec fyp-airflow-webserver python -m agents.business_analyst.agent --ticker AAPL --task "What is Apple's competitive moat?"

# Business Analyst (from host)
source .venv/bin/activate
python -m agents.business_analyst.agent --ticker AAPL --task "What is Apple's competitive moat?"

# Quant Fundamental
python -m agents.quant_fundamental.agent --ticker NVDA

# Financial Modelling
python -m agents.financial_modelling.agent --ticker TSLA
```

---

## 8. Running the Orchestration

```bash
python - <<'EOF'
from orchestration.graph import run
result = run("What is Apple's competitive moat and current valuation?")
print(result["final_summary"])
EOF
```

---

## 9. Testing

```bash
# Run all tests
pytest agents/ -v

# Run specific agent tests
pytest agents/business_analyst/tests/ -v
pytest agents/quant_fundamental/tests/ -v
pytest agents/financial_modelling/tests/ -v
```

---

## 10. Streamlit UI

```bash
streamlit run POC/streamlit/app.py
# Opens at http://localhost:8501
```

---

## 11. Environment Variables

See `.env` file for all required variables. Key variables:

| Variable | Description | Default |
|---|---|---|
| `EODHD_API_KEY` | EODHD.com API key | Required |
| `POSTGRES_HOST` | PostgreSQL host | localhost |
| `POSTGRES_PORT` | PostgreSQL port | 5432 |
| `NEO4J_URI` | Neo4j bolt URI | bolt://localhost:7687 |
| `OLLAMA_BASE_URL` | Ollama API URL | http://localhost:11434 |
| `BUSINESS_ANALYST_MODEL` | LLM for BA agent | deepseek-r1:8b |
| `TRACKED_TICKERS` | Comma-separated tickers | AAPL,TSLA,NVDA,MSFT,GOOGL |

**Note:** When running inside Docker, `OLLAMA_BASE_URL` defaults to `http://host.docker.internal:11434` for Mac/Windows compatibility.

---

## 12. Project Roadmap

- [x] Phase 1: Core infrastructure (PostgreSQL, Neo4j, Ollama)
- [x] Phase 2: Four-agent system
- [x] Phase 3: Earnings call & broker report ingestion
- [ ] Phase 4: Additional agents (macro, technical)
- [ ] Phase 5: Production deployment

---

## 13. Evaluation Framework

The system uses **CRAG (Corrective RAG)** for the Business Analyst agent:
- **CORRECT** (>0.55 confidence): Generate from retrieved context
- **AMBIGUOUS** (0.35–0.55): Rewrite query and retry
- **INCORRECT** (<0.35): Fall back to web search

Every qualitative claim cites a specific chunk_id from the retrieved documents.
