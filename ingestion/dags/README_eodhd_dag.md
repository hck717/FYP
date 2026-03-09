# EODHD Complete Ingestion DAG

`dag_eodhd_ingestion_unified.py` — Daily data ingestion pipeline powered by the [EODHD API](https://eodhd.com).

---

## Overview

| Property | Value |
|---|---|
| DAG ID | `eodhd_complete_ingestion` |
| Schedule | Daily at **01:00 UTC** (`0 1 * * *`) |
| Tickers | Configured via `TRACKED_TICKERS` env var (default: `AAPL, GOOGL, MSFT, NVDA, TSLA`) |
| Source | EODHD REST API (`https://eodhd.com/api`) |
| Retries | 2 retries, 5 min delay |
| Executor | LocalExecutor (runs inside scheduler container) |

---

## Pipeline Architecture

```
For each ticker:

  [scrape_*] ──┬──► [load_postgres_*]
               ├──► [load_neo4j_*]
               ├──► [textual_load_earnings_calls_*]
               └──► [textual_load_broker_reports_*] ──► [eodhd_generate_summary]
```

Scrape and load tasks run **per-ticker** in parallel. All load tasks fan into a single summary task at the end.

---

## Tasks

### 1. Scrape Tasks
- `eodhd_scrape_{TICKER}` — Fetches data from EODHD API and saves to `agent_data/`

### 2. Load Tasks
- `eodhd_load_postgres_{TICKER}` — Upserts data into PostgreSQL
- `eodhd_load_neo4j_{TICKER}` — Creates Company nodes and relationships in Neo4j
- `textual_load_earnings_calls_{TICKER}` — Ingests PDF earnings calls into Neo4j chunks
- `textual_load_broker_reports_{TICKER}` — Ingests PDF broker reports into Neo4j chunks

### 3. Macro Tasks
- `eodhd_load_postgres_macro` — Loads macro data (treasury, forex, GDP, etc.)
- `eodhd_load_neo4j_macro` — Creates macro relationships in Neo4j

### 4. Summary Task
- `eodhd_generate_summary` — Generates ingestion summary report

---

## Data Collected

### PostgreSQL Tables

| Dataset | Description | Records |
|---|---|---|
| `raw_timeseries` | Historical OHLCV data | ~870K rows |
| `financial_statements` | Balance sheet, income, cash flow | ~2,200 rows |
| `sentiment_trends` | Bullish/bearish/neutral % | 165 rows |
| `valuation_metrics` | PE, PB, PS ratios | 10 rows |
| `insider_transactions` | Insider buying/selling | ~1,700 rows |
| `institutional_holders` | 13F holdings | 243 rows |
| `earnings_surprises` | EPS actuals vs estimates | 521 rows |
| `text_chunks` | pgvector embeddings | 48 chunks |

### Neo4j Nodes

| Node Type | Description | Count |
|---|---|---|
| `:Company` | Company with 85+ properties | 51 nodes |
| `:Chunk` | Text chunks for RAG | 1,829 chunks |

### Chunk Breakdown

| Ticker | earnings_call | broker_report | other | Total |
|---|---|---|---|---|
| AAPL | 131 | 400 | 13 | 544 |
| TSLA | 130 | 158 | 13 | 301 |
| NVDA | 137 | 96 | 12 | 245 |
| MSFT | 138 | 73 | 13 | 224 |
| GOOGL | 142 | 361 | 12 | 515 |

---

## Running DAG Tasks Manually

```bash
# Test individual tasks
docker exec fyp-airflow-scheduler airflow tasks test eodhd_complete_ingestion eodhd_scrape_AAPL 2026-03-07
docker exec fyp-airflow-scheduler airflow tasks test eodhd_complete_ingestion eodhd_load_postgres_AAPL 2026-03-07
docker exec fyp-airflow-scheduler airflow tasks test eodhd_complete_ingestion eodhd_load_neo4j_AAPL 2026-03-07

# Test textual ingestion
docker exec fyp-airflow-scheduler airflow tasks test eodhd_complete_ingestion textual_load_earnings_calls_AAPL 2026-03-07
docker exec fyp-airflow-scheduler airflow tasks test eodhd_complete_ingestion textual_load_broker_reports_AAPL 2026-03-07
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `TRACKED_TICKERS` | Comma-separated tickers | `AAPL,GOOGL,MSFT,NVDA,TSLA` |
| `EODHD_API_KEY` | EODHD API key | Required |
| `OLLAMA_BASE_URL` | Ollama API URL | `http://host.docker.internal:11434` |

---

## Notes

- **Textual Data**: Earnings calls and broker reports are stored in `data/textual data/` and ingested separately
- **No Timeout**: Text embedding uses Ollama with no timeout
- **Cross-Platform**: Uses `host.docker.internal` for Docker containers to access host Ollama
