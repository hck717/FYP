# Ingestion Layer

The ingestion layer runs via **Apache Airflow** and populates two local databases (PostgreSQL and Neo4j) with fresh financial data daily. Agents never hit external APIs at query time — all structured data is pre-ingested and cached locally.

---

## Directory Structure

```
ingestion/
├── README.md                          ← This file
├── data_needed.txt                   ← Reference spec for all 24 data types
├── ingest_textual_metadata.py         ← One-shot: PDF metadata → PostgreSQL
│
├── dags/
│   └── dag_eodhd_ingestion_unified.py ← Airflow DAG (5 tickers + macro)
│
├── etl/
│   ├── load_postgres.py               ← CSV/JSON → PostgreSQL upsert loader
│   ├── load_neo4j.py                  ← CSV/JSON → Neo4j node/edge loader
│   ├── inspect_db.py                  ← Database health check CLI
│   ├── ingest_earnings_calls.py       ← PDF earnings calls → Neo4j chunks
│   ├── ingest_broker_reports.py       ← PDF broker reports → Neo4j chunks
│   └── agent_data/                    ← Local JSON + CSV cache (written by DAG)
│       ├── AAPL/
│       │   ├── metadata.json
│       │   ├── company_profile.json
│       │   ├── financial_news.json    ← Raw article list (list of dicts)
│       │   └── ...  (one file per endpoint)
│       ├── TSLA/ NVDA/ MSFT/ GOOGL/
│       └── _MACRO/
│           ├── metadata.json
│           ├── treasury_bill_rates.csv
│           ├── treasury_yield_curve.csv
│           └── ...  (global / macro endpoints)
```

---

## Architecture

```
EODHD API
    │
    ▼
dag_eodhd_ingestion_unified.py   (Airflow, daily 01:00 UTC)
    │
    ├── scrape_ticker()  ──writes──►  etl/agent_data/{TICKER}/
    │   └── _aggregate_news_sentiment()  ──derives──►  sentiment_trends.csv
    │                                    etl/agent_data/_MACRO/
    │
    ├── load_postgres_for_ticker()  ──upserts──►  PostgreSQL
    │   ├── financial_news.json  ──embeds+upserts──►  news_articles (pgvector)
    │   ├── news_word_weights.json  ──upserts──►  news_word_weights
    │   └── *.csv  ──upserts──►  typed specialty tables
    ├── load_neo4j_for_ticker()     ──merges──►  Neo4j
    ├── ingest_earnings_calls()     ──merges──►  Neo4j (Chunk nodes)
    ├── ingest_broker_reports()     ──merges──►  Neo4j (Chunk nodes)
    └── load_postgres_macro()       ──upserts──►  PostgreSQL (macro tables)
        ├── treasury_bill_rates  ──►  treasury_rates
        └── treasury_yield_curve ──►  treasury_rates
```

---

## PostgreSQL Tables

| Table | Source | Notes |
|---|---|---|
| `raw_timeseries` | DAG / generic | EOD prices, intraday, technicals, realtime news feed |
| `raw_fundamentals` | DAG / generic | Snapshot fundamentals (ratios, scores, statements) |
| `financial_statements` | DAG | Income statement, balance sheet, cash flow (quarterly + yearly) |
| `valuation_metrics` | DAG | P/E, EV/EBITDA, market cap, etc. per ticker |
| `sentiment_trends` | DAG (derived) | Daily pos/neg/neu aggregated from `financial_news` articles |
| `news_articles` | DAG | Full article content + 768-dim pgvector embedding per article |
| `news_word_weights` | DAG | Top keyword weights per ticker over rolling 30-day window |
| `insider_transactions` | DAG | SEC Form 4 insider buy/sell transactions |
| `institutional_holders` | DAG | 13F institutional holder snapshots |
| `short_interest` | DAG | Short interest, shares float, short ratio |
| `earnings_surprises` | DAG | EPS/revenue actuals vs estimates per quarter |
| `outstanding_shares` | DAG | Annual + quarterly shares outstanding history |
| `dividends_history` | DAG | Ex-date, payment date, amount |
| `splits_history` | DAG | Split ratio and ex-date |
| `financial_calendar` | DAG (macro) | Earnings dates, IPOs, splits, dividends calendar |
| `treasury_rates` | DAG (macro) | Bill rates + full yield curve (from UST API) |
| `forex_rates` | DAG (macro) | EUR/USD and other pairs |
| `corporate_bond_yields` | DAG (macro) | LQD/HYG proxy + EODHD bond fundamentals |
| `global_macro_indicators` | DAG (macro) | GDP, CPI, unemployment |
| `economic_events` | DAG (macro) | FOMC, CPI releases, NFP, etc. |
| `market_eod_us` | DAG (macro) | S&P 500 benchmark daily OHLCV |
| `market_screener` | DAG (macro) | EODHD screener bulk snapshot |
| `text_chunks` | DAG | Company profile text chunks with 768-dim pgvector embedding |
| `textual_documents` | One-shot script | PDF metadata (earnings calls, broker reports) |

---

## Quick Start

### Run Full DAG Test
```bash
docker exec fyp-airflow-scheduler airflow tasks test eodhd_complete_ingestion eodhd_scrape_AAPL 2026-03-07
docker exec fyp-airflow-scheduler airflow tasks test eodhd_complete_ingestion eodhd_load_postgres_AAPL 2026-03-07
docker exec fyp-airflow-scheduler airflow tasks test eodhd_complete_ingestion eodhd_load_neo4j_AAPL 2026-03-07
```

### Ingest Textual Data (Earnings Calls & Broker Reports)
```bash
# Inside Docker container (recommended)
docker exec fyp-airflow-webserver python /opt/airflow/ingestion/etl/ingest_earnings_calls.py --all
docker exec fyp-airflow-webserver python /opt/airflow/ingestion/etl/ingest_broker_reports.py --all

# From host
python ingestion/etl/ingest_earnings_calls.py --all
python ingestion/etl/ingest_broker_reports.py --all
```

---

## Data Validation Commands

### Check All Data Status (Inside Container - Recommended)
```bash
docker exec fyp-airflow-webserver python /opt/airflow/ingestion/etl/inspect_db.py
```

**Expected Output:**
```
============================================================
PostgreSQL checks
============================================================
  PASS  raw_timeseries: 878676 total rows, all 5 tickers covered
  PASS  financial_statements: 2229 total rows, all 5 tickers covered
  PASS  valuation_metrics: 10 total rows, all 5 tickers covered
  PASS  sentiment_trends: 170 total rows, all 5 tickers covered
  WARN  dividends_history: 394 rows, missing tickers (may not pay dividends): ['TSLA']
  PASS  market_eod_us: 502 rows (S&P 500 benchmark)
  PASS  insider_transactions: 2311 rows
  PASS  institutional_holders: 243 rows
  PASS  short_interest: 20 rows
  PASS  earnings_surprises: 521 rows
  PASS  raw_fundamentals: 2419 rows
  PASS  news_articles: 250 total rows, all 5 tickers covered, 250/250 embedded (100%)
  WARN  news_word_weights: 0 rows (DAG scrape not yet run for this endpoint)

--- pgvector text_chunks ---
  PASS  pgvector extension installed (version 0.8.2)
  PASS  text_chunks table exists
  PASS  text_chunks.embedding column: type=vector
  PASS  text_chunks [AAPL]: 10 chunks, all embedded
  ...

============================================================
Neo4j checks
============================================================
  PASS  Company nodes: 51
  PASS  Chunk nodes: 1829
  PASS  Chunk embedding dimension: 768 (nomic-embed-text ✓)
  PASS  Neo4j vector index 'chunk_embedding': ONLINE, dim=768

--- Textual Document Coverage ---
  PASS  earnings_call: 678 chunks across 5 tickers (AAPL, GOOGL, MSFT, NVDA, TSLA)
  PASS  broker_report: 1088 chunks across 5 tickers (AAPL, GOOGL, MSFT, NVDA, TSLA)
```

> **Note on `news_word_weights`:** This table is populated by the `news-word-weights` EODHD endpoint. 0 rows is expected until the DAG has run a full scrape cycle that includes this endpoint. It does not affect the overall PASS/FAIL status.

### Check PostgreSQL Only
```bash
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT data_name, period_type, COUNT(*)
  FROM raw_fundamentals
  GROUP BY data_name, period_type
  ORDER BY data_name;
"
```

### Check news_articles Embedding Coverage
```bash
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT ticker,
         COUNT(*) AS total,
         SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) AS embedded
  FROM news_articles
  GROUP BY ticker ORDER BY ticker;
"
```

### Check Neo4j Only
```bash
docker exec fyp-neo4j cypher-shell -u neo4j -p SecureNeo4jPass2025! "
  MATCH (c:Company)-[:HAS_CHUNK]->(ch:Chunk)
  RETURN c.ticker, ch.section, count(*) as cnt
  ORDER BY c.ticker, ch.section
"
```

---

## Sentiment Pipeline

Sentiment data flows from EODHD news articles through an aggregation step into PostgreSQL. The agent reads from `sentiment_trends` at query time.

### Step-by-step

1. **EODHD `/api/news`** — The DAG calls `GET /api/news?s={ticker}&limit=50` for each ticker. Each article in the response includes a `sentiment` sub-dict:
   ```json
   { "polarity": 0.98, "neg": 0.054, "neu": 0.851, "pos": 0.095 }
   ```
   - `pos` / `neg` / `neu` are normalised floats in `[0, 1]` representing the fraction of the article text that EODHD's NLP model scores as positive / negative / neutral tone. They sum to approximately 1.0.
   - `polarity` is a compound score from -1 (most negative) to +1 (most positive).

2. **`_aggregate_news_sentiment(ticker, articles, ticker_dir)`** — For each ticker, articles are **bucketed by calendar date** (extracted from the ISO `date` field). Within each date bucket:
   - `avg_pos = mean([a["pos"] for a in day_articles])`
   - `avg_neg = mean([a["neg"] for a in day_articles])`
   - `avg_neu = mean([a["neu"] for a in day_articles])`
   - These per-date averages are written to `sentiment_trends.json` and `sentiment_trends.csv` in `etl/agent_data/{TICKER}/`.

3. **`load_postgres_for_ticker()`** — The `sentiment_trends.csv` is read and upserted into the `sentiment_trends` PostgreSQL table with columns:
   ```
   ticker, bullish_pct, bearish_pct, neutral_pct, trend, as_of_date, ingested_at
   ```
   The mapping from EODHD fields to the agent-facing columns is:
   ```
   bullish_pct = avg_pos × 100
   bearish_pct = avg_neg × 100
   neutral_pct = avg_neu × 100
   trend       = "bullish"    if avg_pos > avg_neg + 0.05
               = "bearish"    if avg_neg > avg_pos + 0.05
               = "stable"     otherwise
   ```

4. **Agent query time** — `PostgresConnector.fetch_sentiment(ticker)` runs:
   ```sql
   SELECT bullish_pct, bearish_pct, neutral_pct, trend
   FROM sentiment_trends
   WHERE ticker = %s
   ORDER BY as_of_date DESC
   LIMIT 1
   ```
   A **7-day freshness gate** is applied: if the most recent row is older than 7 days, the agent falls back to a local VADER → TextBlob → keyword heuristic computed over recent Neo4j chunk text rather than using the stale DB value.

### sentiment_trends table schema

```sql
CREATE TABLE IF NOT EXISTS sentiment_trends (
    id              SERIAL PRIMARY KEY,
    ticker          VARCHAR(10) NOT NULL,
    bullish_pct     FLOAT,
    bearish_pct     FLOAT,
    neutral_pct     FLOAT,
    trend           VARCHAR(20),
    as_of_date      DATE,
    ingested_at     TIMESTAMP DEFAULT NOW()
);
```

### Example row

| ticker | bullish_pct | bearish_pct | neutral_pct | trend   | as_of_date |
|--------|-------------|-------------|-------------|---------|------------|
| AAPL   | 9.5         | 5.4         | 85.1        | stable  | 2026-03-08 |

> Note: EODHD `neu` scores are typically high (60–90%) because most financial news is neutral/factual in tone. `bullish_pct` and `bearish_pct` are therefore often low in absolute terms; what matters is their relative magnitude and trend direction.

---

## Textual Data Sources

### Earnings Calls
- **Location**: `/data/textual data/{TICKER}/earning_call/`
- **Format**: PDF files
- **Ingestion**: `ingest_earnings_calls.py` extracts text → splits into chunks → embeds with Ollama → stores in Neo4j
- **Exact Paths**:
  - `/Users/brianho/FYP/data/textual data/AAPL/earning_call/`
  - `/Users/brianho/FYP/data/textual data/MSFT/earning_call/`
  - `/Users/brianho/FYP/data/textual data/GOOGL/earning_call/`
  - `/Users/brianho/FYP/data/textual data/TSLA/earning_call/`
  - `/Users/brianho/FYP/data/textual data/NVDA/earning_call/`

### Broker Reports
- **Location**: `/Users/brianho/FYP/data/textual data/{TICKER}/broker/`
- **Format**: PDF files
- **Ingestion**: `ingest_broker_reports.py` extracts text → splits into chunks → embeds with Ollama → stores in Neo4j
- **Exact Paths**:
  - `/Users/brianho/FYP/data/textual data/AAPL/broker/`
  - `/Users/brianho/FYP/data/textual data/MSFT/broker/`
  - `/Users/brianho/FYP/data/textual data/GOOGL/broker/`
  - `/Users/brianho/FYP/data/textual data/TSLA/broker/`
  - `/Users/brianho/FYP/data/textual data/NVDA/broker/`

### Metadata Files
- **Location**: `/Users/brianho/FYP/data/textual data/{TICKER}/metadata.json`
- **Contains**: Ticker info, sector, industry, company description

---

## Complete Data Location Reference

### Local Data Directory Structure
```
/Users/brianho/FYP/data/
├── textual data/
│   ├── AAPL/
│   │   ├── broker/
│   │   ├── earning_call/
│   │   └── metadata.json
│   ├── MSFT/
│   │   ├── broker/
│   │   ├── earning_call/
│   │   └── metadata.json
│   ├── GOOGL/
│   │   ├── broker/
│   │   ├── earning_call/
│   │   └── metadata.json
│   ├── TSLA/
│   │   ├── broker/
│   │   ├── earning_call/
│   │   └── metadata.json
│   └── NVDA/
│       ├── broker/
│       ├── earning_call/
│       └── metadata.json
├── postgres_data/          # PostgreSQL data directory
├── neo4j_data/            # Neo4j graph database
├── neo4j_logs/            # Neo4j log files
└── logs/                  # Application logs
```

### PostgreSQL Database Tables (via Docker)
- **Host**: `localhost:5432` (external) or `postgres:5432` (Docker internal)
- **Database**: `airflow`
- **User**: `airflow`
- **Key Tables**:
  - `raw_timeseries` - OHLCV price data
  - `financial_statements` - Income statement, balance sheet, cash flow
  - `valuation_metrics` - PE, EV/EBITDA, market cap
  - `sentiment_trends` - Daily sentiment aggregations
  - `news_articles` - News with embeddings
  - `text_chunks` - Company profile chunks with embeddings
  - `earnings_surprises` - EPS actuals vs estimates
  - `insider_transactions` - SEC Form 4 transactions
  - `institutional_holders` - 13F holdings

### Neo4j Database (via Docker)
- **Host**: `localhost:7474` (HTTP) / `localhost:7687` (Bolt)
- **User**: `neo4j`
- **Password**: `SecureNeo4jPass2025!`
- **Key Nodes**:
  - `:Company` - 85+ properties per ticker
  - `:Chunk` - Text embeddings from earnings calls, broker reports

### Agent Data Cache (written by DAG)
- **Location**: `/Users/brianho/FYP/ingestion/etl/agent_data/{TICKER}/`
- **Contains**: Raw JSON/CSV files from EODHD API

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OLLAMA_BASE_URL` | Ollama API URL | `http://localhost:11434` (host) or `http://host.docker.internal:11434` (Docker) |
| `EMBEDDING_MODEL` | Ollama model for embeddings | `nomic-embed-text` (768-dim) |
| `TRACKED_TICKERS` | Comma-separated tickers | `AAPL,TSLA,NVDA,MSFT,GOOGL` |
| `EODHD_API_KEY` | EODHD API key | Required |
| `POSTGRES_HOST` | PostgreSQL host | `postgres` |
| `NEO4J_URI` | Neo4j URI | `bolt://neo4j:7687` |

---

## Notes

- **Upsert-safe**: All inserts use `ON CONFLICT … DO UPDATE` — safe to re-run the DAG
- **Graceful embedding**: If Ollama is unreachable during `news_articles` ingestion, rows are inserted with `NULL` embedding (pipeline does not fail)
- **Cross-Platform**: Scripts auto-detect Docker environment and use appropriate Ollama URL
- **pgvector**: Used for both `text_chunks` (company profile RAG) and `news_articles` (news semantic search)
- **Treasury rates**: New UST API endpoints (`treasury_bill_rates`, `treasury_yield_curve`) replace the old 8 GBOND entries; both feed into the single `treasury_rates` table via the multi-column legacy path in `_insert_treasury_rates()`
