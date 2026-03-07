# Ingestion Layer

The ingestion layer runs via **Apache Airflow** and populates two local databases (PostgreSQL and Neo4j) with fresh financial data daily. Agents never hit external APIs at query time — all structured data is pre-ingested and cached locally.

---

## Directory Structure

```
ingestion/
├── README.md                          ← This file
├── data_needed.txt                   ← Reference spec for all 24 data types
├── ingest_textual_metadata.py        ← One-shot: PDF metadata → PostgreSQL
│
├── dags/
│   └── dag_eodhd_ingestion_unified.py ← Airflow DAG (5 tickers + macro)
│
└── etl/
    ├── load_postgres.py              ← CSV/JSON → PostgreSQL upsert loader
    ├── load_neo4j.py                 ← CSV/JSON → Neo4j node/edge loader
    ├── inspect_db.py                 ← Database health check CLI
    └── agent_data/                   ← Local JSON + CSV cache (written by DAG)
        ├── AAPL/
        │   ├── metadata.json
        │   ├── company_profile.csv
        │   └── ...  (one file per endpoint)
        ├── TSLA/ NVDA/ MSFT/ GOOGL/
        └── _MACRO/
            ├── metadata.json
            ├── treasury_rates.csv
            └── ...  (global / macro endpoints)
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
    │                                 etl/agent_data/_MACRO/
    │
    ├── load_postgres_for_ticker()  ──upserts──►  PostgreSQL
    ├── load_neo4j_for_ticker()     ──merges───►  Neo4j
    └── load_postgres_macro()       ──upserts──►  PostgreSQL (macro tables)
```

---

## Quick Start

### Run Full DAG Test
```bash
docker exec fyp-airflow-scheduler airflow tasks test eodhd_complete_ingestion eodhd_scrape_AAPL 2026-03-07
docker exec fyp-airflow-scheduler airflow tasks test eodhd_complete_ingestion eodhd_load_postgres_AAPL 2026-03-07
docker exec fyp-airflow-scheduler airflow tasks test eodhd_complete_ingestion eodhd_load_neo4j_AAPL 2026-03-07
```

### Data Validation Commands

#### Check All Data Status (Local)
```bash
# From host machine - check all databases and coverage
python ingestion/db_inspect.py --coverage

# Check PostgreSQL only
python ingestion/db_inspect.py --pg

# Check Neo4j only
python ingestion/db_inspect.py --neo4j

# Check local files only
python ingestion/db_inspect.py --files

# Check specific ticker
python ingestion/db_inspect.py --coverage --ticker AAPL
```

#### Check Data Status (Inside Airflow Container)
```bash
# Full health check (PostgreSQL + Neo4j) - RECOMMENDED
docker exec fyp-airflow-scheduler python /opt/airflow/etl/inspect_db.py

# Check raw_fundamentals data by data_name and period_type
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT data_name, period_type, COUNT(*)
  FROM raw_fundamentals
  GROUP BY data_name, period_type
  ORDER BY data_name;
"

# Check insider_transactions by ticker
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT ticker, COUNT(*) as total, COUNT(DISTINCT insider_name) as insiders
  FROM insider_transactions
  GROUP BY ticker;
"

# Check all table row counts
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT tablename as table_name, n_live_tup as rows
  FROM pg_stat_user_tables
  ORDER BY n_live_tup DESC
  LIMIT 20;
"
```

#### Validate Against data_needed.txt
```bash
# Run coverage check
python ingestion/db_inspect.py --coverage

# Expected output shows each data type and whether it's present in files/DB
```

---

## Script Reference

### `dags/dag_eodhd_ingestion_unified.py`

**What it does:**

The main Airflow DAG (`eodhd_complete_ingestion`). Scheduled daily at **01:00 UTC**. For each of the 5 tracked tickers it:

1. **Scrapes** all EODHD endpoints (per-ticker + macro) via `scrape_ticker()`.
2. **Writes** raw JSON + CSV files to `etl/agent_data/{TICKER}/` and `etl/agent_data/_MACRO/`.
3. **Loads PostgreSQL** by calling `load_postgres_for_ticker()` (per-ticker specialty tables + generic timeseries).
4. **Loads Neo4j** by calling `load_neo4j_for_ticker()` (`:Company` nodes, profile properties + text chunks with embeddings).
5. **Loads macro data** via a dedicated `load_postgres_macro` task.
6. Prints a **summary** of updates via `report_summary()`.

**Key design decisions:**
- **Incremental loads** — each file is MD5-hashed; unchanged data is skipped.
- **Rate limiting** — sleeps `60 / EODHD_RATE_LIMIT` seconds after every HTTP call.
- **Macro deduplication** — global endpoints fetched only once by first ticker (AAPL).
- **Response caching** — `financial_news` ↔ `realtime_news_feed`, `realtime_quote` ↔ `live_stock_price` share HTTP calls.

### `etl/load_postgres.py`

**What it does:**

ETL loader that reads CSV/JSON from `agent_data/` and upserts into PostgreSQL.

**Key tables created:**
- `raw_timeseries` — Generic time-indexed data (prices, technicals, news)
- `raw_fundamentals` — Snapshot data (statements, ratios, scores) with `period_type` for quarterly/yearly
- `financial_statements` — Income Statement, Balance Sheet, Cash Flow
- `valuation_metrics` — Trailing/Forward PE, Price/Sales, EV metrics
- `sentiment_trends` — Bullish/bearish/neutral sentiment scores
- `insider_transactions` — SEC Form 4 insider buys/sells (composite key: ticker + insider + date + type + shares + price)
- `institutional_holders` — Major institutional holders
- `dividends_history` — Historical dividend payments
- `splits_history` — Historical stock splits
- `short_interest` — Short interest and float data
- `earnings_surprises` — EPS actuals vs estimates
- `outstanding_shares` — Historical shares outstanding
- `text_chunks` — pgvector table for semantic search
- `textual_documents` — PDF document metadata
- Macro tables: treasury_rates, economic_events, corporate_bond_yields, forex_rates, market_screener, market_eod_us

**2026-03 fixes:**
- Added `period_type` column to `raw_fundamentals` for quarterly/yearly financial statements
- Fixed duplicate key error in `insider_transactions` by including `shares` and `price` in unique constraint

### `etl/load_neo4j.py`

**What it does:**

ETL loader that reads CSV/JSON and creates Neo4j nodes/edges.

**Key features:**
- `:Company` nodes with profile properties (highlights, valuation, ESG, etc.)
- `:Chunk` nodes with 768-dim embeddings from Ollama (nomic-embed-text)
- `HAS_CHUNK` relationships for RAG retrieval
- `:DataRecord` generic nodes for any data type
- `[:CONTAINS]` edges for ETF constituents

### `etl/inspect_db.py`

**What it does:**

Database health check script. Verifies:
- PostgreSQL tables: row counts, freshness, per-ticker coverage
- pgvector text_chunks: row counts, embedding dimension check
- Neo4j: chunk count, vector index status and dimension

---

## Data Coverage (data_needed.txt)

| # | Data Type | Destination | Scope |
|---|-----------|-------------|-------|
| 1 | Company Profiles | Neo4j | Per ticker |
| 2 | Financial News | PostgreSQL | Per ticker |
| 3 | Insider Transactions | PostgreSQL | Per ticker |
| 4 | Institutional Holders | PostgreSQL | Per ticker |
| 5 | Historical Prices (EOD) | PostgreSQL | Per ticker |
| 6 | Intraday/Live Quotes | PostgreSQL | Per ticker |
| 7 | Technical Indicators | PostgreSQL | Per ticker |
| 8 | Screener (Bulk) | PostgreSQL | Macro |
| 9 | Basic Fundamentals | PostgreSQL | Per ticker |
| 10 | Dividends/Splits | PostgreSQL | Per ticker |
| 11 | Treasury Rates | PostgreSQL | Macro |
| 12 | Economic Events | PostgreSQL | Macro |
| 13 | Bonds Data | PostgreSQL | Macro |
| 14 | Forex Rates | PostgreSQL | Macro |
| 15 | ETF Holdings | Neo4j | Macro |
| 16 | Financial Calendar | PostgreSQL | Per ticker |
| 17 | Real-Time News | PostgreSQL | Per ticker |
| 18 | Financial Statements | PostgreSQL | Per ticker |
| 19 | Valuation Metrics | PostgreSQL | Per ticker |
| 20 | Short Interest | PostgreSQL | Per ticker |
| 21 | Earnings Surprises | PostgreSQL | Per ticker |
| 22 | Outstanding Shares | PostgreSQL | Per ticker |
| 23 | Sentiment Trends | PostgreSQL | Per ticker |
| 24 | Textual Documents | PostgreSQL | Per ticker |

---

## Troubleshooting

### Insider Transactions Error
If you see `duplicate key value violates unique constraint`, the table needs migration:
```bash
docker exec fyp-postgres psql -U airflow -d airflow -c "
  ALTER TABLE insider_transactions DROP CONSTRAINT IF EXISTS insider_transactions_ticker_insider_name_transaction_date_t_key;
  ALTER TABLE insider_transactions ADD UNIQUE (ticker, insider_name, transaction_date, transaction_type, shares, price);
"
```

### Raw Fundamentals Issues
If financial statements aren't populating, verify the table has `period_type` column:
```bash
docker exec fyp-postgres psql -U airflow -d airflow -c "\d raw_fundamentals"
```

### Vector Search Not Working
Check Neo4j vector index:
```bash
docker exec fyp-airflow-scheduler python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://neo4j:7687', auth=('neo4j', 'SecureNeo4jPass2025!'))
with driver.session() as s:
    result = s.run(\"SHOW INDEXES YIELD name, state, type WHERE name = 'chunk_embedding'\").data()
    print(result)
"
```
