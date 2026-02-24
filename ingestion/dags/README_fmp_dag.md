# FMP Complete Ingestion DAG

`dag_fmp_ingestion_unified.py` — Daily data ingestion pipeline powered by the [Financial Modeling Prep (FMP) API](https://financialmodelingprep.com).

---

## Overview

| Property | Value |
|---|---|
| DAG ID | `fmp_complete_ingestion` |
| Schedule | Daily at **02:00 UTC** (`0 2 * * *`) — 1 hour after EODHD |
| Tickers | Configured via `TRACKED_TICKERS` env var (default: `AAPL, GOOGL, MSFT, NVDA, TSLA`) |
| Source | FMP Stable API (`https://financialmodelingprep.com/stable`) |
| Retries | 2 retries, 5 min delay |
| Executor | LocalExecutor (runs inside scheduler container) |

---

## Pipeline Architecture

```
For each agent × ticker:

  [scrape_*] ──┬──► [load_postgres_*]
               ├──► [load_neo4j_*]      ──► [fmp_generate_summary]
               └──► [load_qdrant_*]
```

Scrape and load tasks run **per-agent per-ticker** in parallel. All load tasks fan into a single summary task at the end.

---

## Agents & Data Collected

### 1. `business_analyst`

Focused on qualitative filings, transcripts, and news for document-based RAG.

| Dataset | FMP Endpoint | Records | Destination |
|---|---|---|---|
| `company_profile` | `/profile` | 1 profile | Neo4j (graph) |
| `key_executives` | `/key-executives` | Latest list | Neo4j (graph) |
| `stock_quote` | `/quote` | Latest snapshot | Neo4j (graph) |
| `sec_filings_10k` | `/sec_filings?type=10-K` | Last 5 filings | Qdrant (vector) |
| `sec_filings_10q` | `/sec_filings?type=10-Q` | Last 10 filings | Qdrant (vector) |
| `sec_filings_8k` | `/sec_filings?type=8-K` | Last 20 filings | Qdrant (vector) |
| `earnings_call_transcripts` | `/earning_call_transcript` | Last 10 transcripts | Qdrant (vector) |
| `risk_factors` | `/risk-factors` | Latest | Neo4j (graph) |
| `company_notes` | `/company-notes` | Latest | Neo4j (graph) |
| `press_releases` | `/press-releases` | Last 50 | Qdrant (vector) |
| `stock_news` | `/news/stock` | Last 100 articles | Qdrant (vector) |

---

### 2. `quantitative_fundamental`

Financial statements, ratios, scores and market data for fundamental quantitative models.

| Dataset | FMP Endpoint | Records | Destination |
|---|---|---|---|
| `income_statement` | `/income-statement` | Last 40 quarters | PostgreSQL |
| `balance_sheet` | `/balance-sheet-statement` | Last 40 quarters | PostgreSQL |
| `cash_flow` | `/cash-flow-statement` | Last 40 quarters | PostgreSQL |
| `income_statement_as_reported` | `/income-statement-as-reported` | Last 10 | PostgreSQL |
| `balance_sheet_as_reported` | `/balance-sheet-as-reported` | Last 10 | PostgreSQL |
| `cash_flow_as_reported` | `/cash-flow-statement-as-reported` | Last 10 | PostgreSQL |
| `financial_ratios` | `/ratios` | Last 40 quarters | PostgreSQL |
| `ratios_ttm` | `/ratios-ttm` | TTM snapshot | PostgreSQL |
| `key_metrics` | `/key-metrics` | Last 40 quarters | PostgreSQL |
| `key_metrics_ttm` | `/key-metrics-ttm` | TTM snapshot | PostgreSQL |
| `financial_growth` | `/financial-growth` | Last 40 quarters | PostgreSQL |
| `enterprise_values` | `/enterprise-values` | Last 40 quarters | PostgreSQL |
| `financial_scores` | `/financial-scores` | Latest snapshot | PostgreSQL |
| `shares_float` | `/shares_float` | Latest snapshot | PostgreSQL |
| `historical_market_cap` | `/historical-market-capitalization` | Last 365 days | PostgreSQL |
| `company_core_info` | `/company-core-information` | Latest | PostgreSQL |
| `rating` | `/rating` | Latest | PostgreSQL |

---

### 3. `financial_modeling`

DCF inputs, segmentation, analyst forecasts, macro indicators and benchmarks for valuation models.

| Dataset | FMP Endpoint | Records | Destination |
|---|---|---|---|
| `income_statement` | `/income-statement` | Last 40 quarters | PostgreSQL |
| `balance_sheet` | `/balance-sheet-statement` | Last 40 quarters | PostgreSQL |
| `cash_flow` | `/cash-flow-statement` | Last 40 quarters | PostgreSQL |
| `dcf` | `/dcf` | Latest DCF value | PostgreSQL |
| `advanced_dcf` | `/advanced_dcf` | Latest | PostgreSQL |
| `levered_dcf` | `/levered_dcf` | Latest | PostgreSQL |
| `owner_earnings` | `/owner-earnings` | Last 40 quarters | PostgreSQL |
| `revenue_product_segmentation` | `/revenue-product-segmentation` | Latest | PostgreSQL |
| `revenue_geographic_segmentation` | `/revenue-geographic-segmentation` | Latest | PostgreSQL |
| `analyst_estimates` | `/analyst-estimates` | Latest | PostgreSQL |
| `analyst_estimates_eps` | `/analyst-estimates-eps` | Latest | PostgreSQL |
| `analyst_estimates_revenue` | `/analyst-estimates-revenue` | Latest | PostgreSQL |
| `price_target` | `/price-target` | Latest | PostgreSQL |
| `price_target_consensus` | `/price-target-consensus` | Latest | PostgreSQL |
| `historical_dividends` | `/historical-price-full/stock_dividend` | Full history | PostgreSQL |
| `stock_splits` | `/historical-price-full/stock_split` | Full history | PostgreSQL |
| `treasury_rates` | `/treasury` | Last 365 days | PostgreSQL |
| `economic_indicators_gdp` | `/economic?name=GDP` | Full history | PostgreSQL |
| `economic_indicators_cpi` | `/economic?name=CPI` | Full history | PostgreSQL |
| `economic_indicators_inflation` | `/economic?name=inflationRate` | Full history | PostgreSQL |
| `stock_peers` | `/stock_peers` | Latest | PostgreSQL |
| `historical_sectors_performance` | `/sectors-performance` | Latest | PostgreSQL |
| `company_notes` | `/company-notes` | Latest | PostgreSQL |
| `company_outlook` | `/company-outlook` | Latest | PostgreSQL |
| `enterprise_values` | `/enterprise-values` | Last 40 quarters | PostgreSQL |
| `market_cap_history` | `/historical-market-capitalization` | Last 365 days | PostgreSQL |

---

## Storage Destinations

| Destination | Table(s) | Used For |
|---|---|---|
| **PostgreSQL** | `raw_timeseries`, `raw_fundamentals` | All quantitative & financial modeling data |
| **Neo4j** | Company nodes & relationships | Company profiles, executives, risk factors |
| **Qdrant** | Vector collections (prepared) | SEC filings, transcripts, news for RAG |

---

## Deduplication & Change Detection

- Each dataset is **MD5-hashed** on scrape. If the hash matches the previous run, the dataset is skipped entirely.
- PostgreSQL inserts use `ON CONFLICT ... DO UPDATE` — idempotent and safe to re-run.
- Premium endpoints (403 Forbidden) are gracefully skipped with a warning — they do not fail the task.
- Empty API responses are skipped without error.

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `FMP_API_KEY` | FMP API key | *(required)* |
| `TRACKED_TICKERS` | Comma-separated ticker list | `AAPL` |
| `FMP_RATE_LIMIT` | API calls per minute | `300` |
| `POSTGRES_HOST` | PostgreSQL host | `fyp-postgres` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_DB` | PostgreSQL database | `airflow` |
| `POSTGRES_USER` | PostgreSQL user | `airflow` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `airflow` |

---

## Triggering Manually

```bash
# Unpause and trigger
docker exec fyp-airflow-scheduler airflow dags unpause fmp_complete_ingestion
docker exec fyp-airflow-scheduler airflow dags trigger fmp_complete_ingestion

# Check task states
docker exec fyp-airflow-scheduler airflow tasks states-for-dag-run \
  fmp_complete_ingestion <run_id>

# Test a single task
docker exec fyp-airflow-scheduler airflow tasks test \
  fmp_complete_ingestion \
  fmp_load_postgres_quantitative_fundamental_AAPL \
  2026-02-24
```

---

## FMP vs EODHD — Complementary Coverage

| Data Category | EODHD DAG | FMP DAG |
|---|---|---|
| Intraday prices (1m/5m/15m/1h) | ✅ | ❌ |
| EOD prices | ✅ | ❌ |
| Technical indicators (SMA/EMA) | ✅ | ❌ |
| Full fundamentals (nested JSON) | ✅ | ❌ |
| Financial statements (quarterly) | ❌ | ✅ |
| DCF valuation models | ❌ | ✅ |
| SEC filings (10-K/10-Q/8-K) | ❌ | ✅ |
| Earnings call transcripts | ❌ | ✅ |
| Analyst estimates & price targets | ✅ (via fundamentals) | ✅ (dedicated endpoints) |
| Macro indicators (GDP/CPI) | ✅ (economic calendar) | ✅ (dedicated endpoints) |
| Bulk US EOD (all equities) | ✅ | ❌ |
| News & sentiment | ✅ | ✅ |
