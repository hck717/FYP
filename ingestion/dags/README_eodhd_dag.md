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
For each agent × ticker:

  [scrape_*] ──┬──► [load_postgres_*]
               ├──► [load_neo4j_*]      ──► [eodhd_generate_summary]
               └──► [load_qdrant_*]
```

Scrape and load tasks run **per-agent per-ticker** in parallel. All load tasks fan into a single summary task at the end.

---

## Agents & Data Collected

### 1. `business_analyst`

Focused on qualitative and sentiment data for news-based analysis.

| Dataset | EODHD Endpoint | Records | Destination |
|---|---|---|---|
| `financial_news` | `/news` | 100 articles | Qdrant (vector) |
| `sentiment_trends` | `/sentiments` | 1 snapshot | Qdrant (vector) |
| `company_profile` | `/fundamentals/{ticker}` | 1 profile | Neo4j (graph) |

---

### 2. `quantitative_fundamental`

High-frequency price and technical data for quantitative models.

| Dataset | EODHD Endpoint | Frequency / Window | Destination |
|---|---|---|---|
| `realtime_quote` | `/real-time/{ticker}` | Latest snapshot | PostgreSQL |
| `live_stock_price` | `/real-time/{ticker}` | Latest snapshot | PostgreSQL |
| `historical_prices_eod` | `/eod/{ticker}` | Last 30 days (daily) | PostgreSQL |
| `intraday_1m` | `/intraday/{ticker}` | 1-minute bars (full history) | PostgreSQL |
| `intraday_5m` | `/intraday/{ticker}` | 5-minute bars | PostgreSQL |
| `intraday_15m` | `/intraday/{ticker}` | 15-minute bars | PostgreSQL |
| `intraday_1h` | `/intraday/{ticker}` | 1-hour bars | PostgreSQL |
| `fundamentals` | `/fundamentals/{ticker}` | Full nested JSON | PostgreSQL |
| `technical_sma` | `/technical/{ticker}` | SMA-50, full history | PostgreSQL |
| `technical_ema` | `/technical/{ticker}` | EMA-20, full history | PostgreSQL |

> **Note:** `fundamentals` is saved as JSON only — its deeply nested structure (`General`, `Financials` keys) cannot be auto-flattened to CSV.

---

### 3. `financial_modeling`

Long-term price history, corporate actions, and macro calendars for financial modeling.

| Dataset | EODHD Endpoint | Window | Destination |
|---|---|---|---|
| `historical_prices_weekly` | `/eod/{ticker}?period=w` | Last 365 days | PostgreSQL |
| `historical_prices_monthly` | `/eod/{ticker}?period=m` | Last 730 days | PostgreSQL |
| `dividends_history` | `/div/{ticker}` | Full history | PostgreSQL |
| `splits_history` | `/splits/{ticker}` | Full history | PostgreSQL |
| `earnings_history` | `/calendar/earnings` | Per ticker | PostgreSQL |
| `fundamentals_full` | `/fundamentals/{ticker}` | Full snapshot | PostgreSQL |
| `analyst_estimates_eodhd` | `/fundamentals/{ticker}` | Full snapshot | PostgreSQL |
| `economic_calendar` | `/economic-events` | Last 30 days | PostgreSQL (global table) |
| `ipo_calendar` | `/calendar/ipos` | Last 365 days | PostgreSQL (global table) |
| `bulk_eod_us` | `/eod-bulk-last-day/US` | All US equities, last day | PostgreSQL (global table) |

> **Global datasets** (`bulk_eod_us`, `economic_calendar`, `ipo_calendar`) are stored once per day in shared tables (`market_eod_us`, `global_economic_calendar`, `global_ipo_calendar`) — not duplicated per ticker.

---

## Storage Destinations

| Destination | Table(s) | Used For |
|---|---|---|
| **PostgreSQL** | `raw_timeseries`, `raw_fundamentals`, `market_eod_us`, `global_economic_calendar`, `global_ipo_calendar` | Quantitative & financial modeling data |
| **Neo4j** | Company nodes & relationships | Company profiles, graph relationships |
| **Qdrant** | Vector collections (prepared) | News embeddings, sentiment for RAG |

---

## Deduplication & Change Detection

- Each dataset is **MD5-hashed** on scrape. If the hash matches the previous run, the dataset is skipped (no re-write, no re-insert).
- PostgreSQL inserts use `ON CONFLICT ... DO UPDATE` — safe to re-run without duplicating rows.
- Unix integer timestamps from intraday/realtime endpoints are automatically converted to `TIMESTAMP` before insert.
- Global datasets check `ingested_at::date = CURRENT_DATE` before inserting — loaded only once per day.

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `EODHD_API_KEY` | EODHD API key | *(required)* |
| `TRACKED_TICKERS` | Comma-separated ticker list | `AAPL` |
| `EODHD_RATE_LIMIT` | API calls per minute | `1000` |
| `POSTGRES_HOST` | PostgreSQL host | `fyp-postgres` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_DB` | PostgreSQL database | `airflow` |
| `POSTGRES_USER` | PostgreSQL user | `airflow` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `airflow` |

---

## Triggering Manually

```bash
# Unpause and trigger
docker exec fyp-airflow-scheduler airflow dags unpause eodhd_complete_ingestion
docker exec fyp-airflow-scheduler airflow dags trigger eodhd_complete_ingestion

# Check task states
docker exec fyp-airflow-scheduler airflow tasks states-for-dag-run \
  eodhd_complete_ingestion <run_id>

# Test a single task
docker exec fyp-airflow-scheduler airflow tasks test \
  eodhd_complete_ingestion \
  eodhd_load_postgres_quantitative_fundamental_AAPL \
  2026-02-24
```
