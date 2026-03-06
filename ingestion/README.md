# Ingestion Layer

The ingestion layer runs via **Apache Airflow** and populates two local databases (PostgreSQL and Neo4j) with fresh financial data daily. Agents never hit external APIs at query time — all structured data is pre-ingested and cached locally.

---

## Directory Structure

```
ingestion/
├── README.md                          ← This file
├── db_inspect.py                      ← Health-check / coverage report CLI tool
├── ingest_textual_metadata.py         ← One-shot script: PDF metadata → PostgreSQL
├── data_needed.txt                    ← Reference spec for all 18 data types
│
├── dags/
│   └── dag_eodhd_ingestion_unified.py ← Airflow DAG (5 tickers + macro)
│
└── etl/
    ├── load_postgres.py               ← CSV/JSON → PostgreSQL upsert loader
    ├── load_neo4j.py                  ← CSV/JSON → Neo4j node/edge loader
    └── agent_data/                    ← Local JSON + CSV cache (written by DAG)
        ├── AAPL/
        │   ├── metadata.json          ← Per-ticker ingestion manifest
        │   ├── company_profile.csv
        │   ├── insider_transactions.csv
        │   └── ...  (one file per endpoint)
        ├── TSLA/ NVDA/ MSFT/ GOOGL/   ← Same layout for each ticker
        └── _MACRO/
            ├── metadata.json
            ├── treasury_rates.csv
            ├── forex_historical_rates.csv
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

## Script Reference

### `dags/dag_eodhd_ingestion_unified.py`

**What it does:**

The main Airflow DAG (`eodhd_complete_ingestion`). Scheduled daily at **01:00 UTC**. For each of the 5 tracked tickers it:

1. **Scrapes** all 22 EODHD endpoints (per-ticker + macro) via `scrape_ticker()`.
2. **Writes** raw JSON + CSV files to `etl/agent_data/{TICKER}/` and `etl/agent_data/_MACRO/`.
3. **Loads PostgreSQL** by calling `load_postgres_for_ticker()` (per-ticker specialty tables + generic timeseries).
4. **Loads Neo4j** by calling `load_neo4j_for_ticker()` (`:Company` nodes, profile properties).
5. **Loads macro data** via a dedicated `load_postgres_macro` task (treasury, forex, bonds, screener, GDP/CPI/UE, ETF constituents).
6. Prints a **summary** of updates via `report_summary()`.

**Key design decisions:**
- **Incremental loads** — each file is MD5-hashed; unchanged data is skipped and not re-written to disk or DB.
- **Rate limiting** — sleeps `60 / EODHD_RATE_LIMIT` seconds (≈ 0.06 s) after every real HTTP call only.
- **Macro deduplication** — global endpoints (treasury, forex, GDP etc.) are fetched only once by the first ticker (AAPL) and stored under `_MACRO/`; all other tickers skip them.
- **Response caching** — `financial_news` and `realtime_news_feed` share one HTTP call (same URL); same for `realtime_quote` and `live_stock_price`.

**Endpoints fetched per ticker:**

| Data Name | EODHD Endpoint | Destination |
|---|---|---|
| `company_profile` | `fundamentals/{ticker}` | Neo4j |
| `financial_news` | `news?s={ticker}` | PostgreSQL (`raw_timeseries`) |
| `realtime_news_feed` | `news?s={ticker}` (cached) | PostgreSQL |
| `sentiment_trends` | `sentiments?s={ticker}` | PostgreSQL (`sentiment_trends`) |
| `realtime_quote` | `real-time/{ticker}` | PostgreSQL |
| `live_stock_price` | `real-time/{ticker}` (cached) | PostgreSQL |
| `historical_prices_eod` | `eod/{ticker}` | PostgreSQL |
| `intraday_1m` | `intraday/{ticker}` | PostgreSQL |
| `technical_sma` | `technical/{ticker}` | PostgreSQL |
| `insider_transactions` | `insider-transactions?code={ticker}` | PostgreSQL (`insider_transactions`) |
| `institutional_holders` | `fundamentals/{ticker}?filter=Holders` | PostgreSQL (`institutional_holders`) |
| `dividends_history` | `div/{ticker}` | PostgreSQL (`dividends_history`) |
| `splits_history` | `splits/{ticker}` | PostgreSQL (`splits_history`) |
| `financial_calendar` | `calendar/earnings?symbols={ticker}` | PostgreSQL (`financial_calendar`) |

**Macro endpoints (fetched once, stored in `_MACRO/`):**

| Data Name | EODHD Endpoint | Destination |
|---|---|---|
| `etf_index_constituents` | `fundamentals/SPY.US?filter=ETF_Data::Holdings` | PostgreSQL (`raw_timeseries`) |
| `screener_bulk` | `screener?limit=100` | PostgreSQL (`market_screener`) |
| `treasury_rates` | `eod/US10Y.GBOND` | PostgreSQL (`treasury_rates`) |
| `economic_indicators_gdp` | `macro-indicator/USA?indicator=gdp_growth_annual` | PostgreSQL (`global_macro_indicators`) |
| `economic_indicators_cpi` | `macro-indicator/USA?indicator=consumer_price_index` | PostgreSQL (`global_macro_indicators`) |
| `economic_indicators_unemployment` | `macro-indicator/USA?indicator=unemployment_total_percent` | PostgreSQL (`global_macro_indicators`) |
| `corporate_bond_yields` | `eod/LQD.US` | PostgreSQL (`raw_timeseries`) |
| `forex_historical_rates` | `eod/EURUSD.FOREX` | PostgreSQL (`forex_rates`) |

**Task graph (per ticker):**

```
eodhd_scrape_{TICKER}
    ├──► eodhd_load_postgres_{TICKER}  ──►
    └──► eodhd_load_neo4j_{TICKER}    ──►  eodhd_generate_summary

eodhd_scrape_AAPL  ──►  eodhd_load_postgres_macro  ──►  eodhd_generate_summary
```

---

### `etl/load_postgres.py`

**What it does:**

ETL loader that reads CSV files from `agent_data/` and upserts rows into typed PostgreSQL tables. Called by the Airflow DAG automatically; can also be run manually.

**Functions:**
- `ensure_tables()` — creates all required tables if they don't exist (idempotent DDL).
- `load_postgres_for_ticker(ticker_symbol)` — reads `agent_data/{TICKER}/metadata.json`, dispatches each `data_name` to its typed insert function.
- `load_postgres_macro()` — reads `agent_data/_MACRO/metadata.json`, loads all macro tables.
- `_insert_insider_transactions()` — maps EODHD fields (`ownerName`, `transactionCode`, `transactionAmount`, `transactionPrice`, `transactionDate`) into the `insider_transactions` table.
- `_insert_institutional_holders()` — maps EODHD Holders response fields (`name`, `currentShares`, `change`, `totalAssets`, `date`) into `institutional_holders`.
- `_insert_financial_calendar()` — maps EODHD calendar fields (`code`, `report_date`, `estimate`) into `financial_calendar`.
- `_insert_dividends_history()`, `_insert_splits_history()` — typed inserts for corporate action tables.
- `_insert_treasury_rates()`, `_insert_macro_indicators()`, `_insert_screener_bulk()`, `_insert_forex_rates()` — macro table inserts.
- `_insert_textual_documents()` — upserts PDF metadata records into `textual_documents` (used by `ingest_textual_metadata.py`).
- `_insert_dataframe_generic()` — fallback: auto-detects date column and routes to `raw_timeseries` or `raw_fundamentals`.

All inserts use `ON CONFLICT DO UPDATE` (upsert), so re-running is always safe.

**When to run manually:** When you need to reload data for one ticker after fixing a bug in an insert function, without triggering a full DAG run:

```bash
# Activate venv and set host override
source .venv/bin/activate

# Reload one ticker
POSTGRES_HOST=localhost python ingestion/etl/load_postgres.py AAPL

# Reload all 5 tickers
POSTGRES_HOST=localhost python ingestion/etl/load_postgres.py --all

# Reload only macro data
POSTGRES_HOST=localhost python ingestion/etl/load_postgres.py --macro
```

---

### `etl/load_neo4j.py`

**What it does:**

ETL loader that reads CSV files from `agent_data/` and writes to Neo4j. Called by the Airflow DAG automatically; can also be run manually.

**Functions:**
- `load_neo4j_for_ticker(ticker_symbol)` — reads `metadata.json`, finds all entries with `storage_destination = "neo4j"`, loads each one.
- `_load_company_profile()` — `MERGE`s a `:Company` node and sets all scalar properties from the EODHD fundamentals response (85+ fields like `Highlights_MarketCapitalization`, `Valuation_TrailingPE`).
- `_load_etf_constituent()` — `MERGE`s `:Company` nodes for SPY and each constituent, creates `[:CONTAINS]` relationship with weight.
- `_load_generic_record()` — fallback: creates a `:DataRecord` node linked to the `:Company` node via `[:HAS_DATA]`.
- `_coerce_props()` — sanitises all values to Neo4j-safe types (handles `numpy` scalars, `NaN`, `Inf`, nested dicts).

All writes use `MERGE` (idempotent — safe to re-run).

**When to run manually:** After fixing a Neo4j loader bug or if the Neo4j database was wiped:

```bash
source .venv/bin/activate

# Reload one ticker
NEO4J_URI=bolt://localhost:7687 python ingestion/etl/load_neo4j.py AAPL

# Reload all 5 tickers
NEO4J_URI=bolt://localhost:7687 python ingestion/etl/load_neo4j.py --all
```

---

### `ingest_textual_metadata.py`

**What it does:**

A one-shot standalone script that reads the `metadata.json` files from each ticker's textual data directory (`/Users/brianho/FYP/data/textual data/{TICKER}/metadata.json`) and upserts PDF document metadata into the `textual_documents` PostgreSQL table.

**What it stores (per PDF):** `ticker`, `doc_type` (broker_report / earnings_call), `filename`, `filepath`, `institution`, `date_approx`, `file_size_bytes`, `md5_hash`, `ingested_into_qdrant` (default `false`).

**Important:** This script stores *metadata only* — no binary PDF content. The actual PDFs are ingested into Qdrant separately in a later phase.

**When to run:**
- **First time setup** — run once after cloning the repo to populate the `textual_documents` table.
- **After adding new PDFs** — if new PDF files are added to `data/textual data/` and their `metadata.json` is updated, re-run to upsert the new entries.
- **After wiping PostgreSQL** — re-run to restore the metadata.

```bash
source .venv/bin/activate
python ingestion/ingest_textual_metadata.py
```

Expected output:
```
=== Textual Document Metadata Ingestion ===
[TextualIngest] AAPL: 8 documents upserted
[TextualIngest] TSLA: 8 documents upserted
[TextualIngest] NVDA: 8 documents upserted
[TextualIngest] MSFT: 8 documents upserted
[TextualIngest] GOOGL: 8 documents upserted

Done. Total documents upserted: 40
```

---

### `db_inspect.py`

**What it does:**

A CLI tool to inspect the health and completeness of all ingested data. It checks three things:
1. **Local files** — which CSV/JSON files exist in `etl/agent_data/`.
2. **PostgreSQL tables** — row counts for every application table.
3. **Neo4j nodes** — node counts by label and Company node details.
4. **Coverage report** — checks all 18 data types from `data_needed.txt` and shows ✔/✘ per ticker and macro.

The `DATA_SPEC` in `db_inspect.py` has three scope types:
- `per_ticker` — checks `agent_data/{TICKER}/{data_name}.csv` for each of the 5 tickers.
- `macro` — checks `agent_data/_MACRO/{data_name}.csv` (global data, fetched once).
- `pg_table` — checks the PostgreSQL table directly for row count > 0.

**Commands:**

```bash
source .venv/bin/activate

# Full report: files + PostgreSQL + Neo4j + coverage
python ingestion/db_inspect.py

# PostgreSQL table row counts only
python ingestion/db_inspect.py --pg

# Neo4j node/relationship counts only
python ingestion/db_inspect.py --neo4j

# Local agent_data file listing only
python ingestion/db_inspect.py --files

# Data coverage report (checks all 18 data types from data_needed.txt)
python ingestion/db_inspect.py --coverage

# Filter any report to a single ticker
python ingestion/db_inspect.py --coverage --ticker AAPL
python ingestion/db_inspect.py --files --ticker NVDA
```

---

## Triggering the DAG

### Via Airflow UI

1. Open [http://localhost:8080](http://localhost:8080) — login: `airflow` / `airflow`
2. Find `eodhd_complete_ingestion` in the DAG list
3. Toggle it **on** (unpause) if paused
4. Click the **▶ Trigger DAG** button (play icon on the right)

### Via command line

```bash
# Trigger a manual run
docker exec fyp-airflow-webserver airflow dags trigger eodhd_complete_ingestion

# Unpause the DAG (enables scheduled daily runs)
docker exec fyp-airflow-webserver airflow dags unpause eodhd_complete_ingestion

# Pause the DAG (disables the daily schedule)
docker exec fyp-airflow-webserver airflow dags pause eodhd_complete_ingestion

# Check status of the last run
docker exec fyp-airflow-webserver airflow tasks states-for-dag-run \
  eodhd_complete_ingestion <run_id>

# List recent runs (get the run_id from here)
docker exec fyp-airflow-webserver airflow dags list-runs \
  -d eodhd_complete_ingestion --limit 5

# View logs for a specific task
docker exec fyp-airflow-webserver airflow tasks logs \
  eodhd_complete_ingestion eodhd_scrape_AAPL <run_id>

# Re-run a single failed task (without re-running the whole DAG)
docker exec fyp-airflow-webserver airflow tasks run \
  eodhd_complete_ingestion eodhd_load_postgres_AAPL <run_id>
```

---

## Inspecting PostgreSQL Data

### Connect directly

```bash
docker exec -it fyp-postgres psql -U airflow -d airflow
```

### Useful psql commands

```sql
-- List all tables
\dt

-- Row counts for all key tables
SELECT table_name,
       (SELECT COUNT(*) FROM information_schema.tables t2
        WHERE t2.table_name = t.table_name) AS rows
FROM information_schema.tables t
WHERE table_schema = 'public'
ORDER BY table_name;

-- Quick counts for main data tables
SELECT 'raw_timeseries'        AS tbl, COUNT(*) FROM raw_timeseries
UNION ALL
SELECT 'insider_transactions',          COUNT(*) FROM insider_transactions
UNION ALL
SELECT 'institutional_holders',         COUNT(*) FROM institutional_holders
UNION ALL
SELECT 'financial_calendar',            COUNT(*) FROM financial_calendar
UNION ALL
SELECT 'dividends_history',             COUNT(*) FROM dividends_history
UNION ALL
SELECT 'splits_history',                COUNT(*) FROM splits_history
UNION ALL
SELECT 'treasury_rates',                COUNT(*) FROM treasury_rates
UNION ALL
SELECT 'forex_rates',                   COUNT(*) FROM forex_rates
UNION ALL
SELECT 'market_screener',               COUNT(*) FROM market_screener
UNION ALL
SELECT 'global_macro_indicators',       COUNT(*) FROM global_macro_indicators
UNION ALL
SELECT 'sentiment_trends',              COUNT(*) FROM sentiment_trends
UNION ALL
SELECT 'textual_documents',             COUNT(*) FROM textual_documents;

-- Inspect insider transactions for a ticker
SELECT ticker, insider_name, transaction_type, shares, price, transaction_date
FROM insider_transactions
WHERE ticker = 'AAPL'
ORDER BY transaction_date DESC
LIMIT 10;

-- Inspect institutional holders
SELECT ticker, holder_name, shares, ownership_pct, as_of_date
FROM institutional_holders
WHERE ticker = 'NVDA'
ORDER BY shares DESC
LIMIT 10;

-- Check financial calendar (upcoming earnings)
SELECT ticker, event_type, event_date, eps_estimate
FROM financial_calendar
ORDER BY event_date
LIMIT 20;

-- Inspect textual documents metadata
SELECT ticker, doc_type, institution, filename, file_size_bytes, ingested_into_qdrant
FROM textual_documents
ORDER BY ticker, doc_type;

-- Check when data was last ingested
SELECT data_name, MAX(ingested_at) AS last_run
FROM raw_timeseries
GROUP BY data_name
ORDER BY last_run DESC;

-- Exit
\q
```

---

## Inspecting Neo4j Data

### Browser UI

Open [http://localhost:7474](http://localhost:7474) — login: `neo4j` / `SecureNeo4jPass2025!`

### Via cypher-shell

```bash
docker exec -it fyp-neo4j cypher-shell -u neo4j -p SecureNeo4jPass2025!
```

### Useful Cypher queries (run in browser or cypher-shell)

```cypher
// Count all nodes by label
CALL db.labels() YIELD label
CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) AS count', {})
YIELD value RETURN label, value.count AS count;

// List all Company nodes
MATCH (c:Company)
RETURN c.ticker, c.Name, c.Sector, c.Industry
ORDER BY c.ticker;

// Full profile of one company
MATCH (c:Company {ticker: 'AAPL'})
RETURN c;

// Key financial metrics for all companies
MATCH (c:Company)
RETURN c.ticker                          AS ticker,
       c.Name                            AS name,
       c.Highlights_MarketCapitalization AS market_cap,
       c.Highlights_EPS                  AS eps,
       c.Valuation_TrailingPE            AS pe_ratio,
       c.Highlights_DividendYield        AS div_yield
ORDER BY c.Highlights_MarketCapitalization DESC;

// Companies in a specific sector
MATCH (c:Company)
WHERE c.Sector = 'Technology'
RETURN c.ticker, c.Name, c.Highlights_MarketCapitalization
ORDER BY c.Highlights_MarketCapitalization DESC;

// SPY constituent relationships (if ETF data loaded)
MATCH (etf:Company {ticker: 'SPY'})-[r:CONTAINS]->(c:Company)
RETURN c.ticker, r.weight
ORDER BY r.weight DESC
LIMIT 20;

// Delete all data (full reset — use with caution)
MATCH (n) DETACH DELETE n;
```

---

## Service Endpoints

| Container | Service | URL / Credentials |
|---|---|---|
| `fyp-airflow-webserver` | Airflow UI | http://localhost:8080 — `airflow` / `airflow` |
| `fyp-postgres` | PostgreSQL | `localhost:5432` — db=`airflow`, user=`airflow`, pw=`airflow` |
| `fyp-neo4j` | Neo4j Browser | http://localhost:7474 — `neo4j` / `SecureNeo4jPass2025!` |
| `fyp-neo4j` | Neo4j Bolt | `bolt://localhost:7687` |

---

## Troubleshooting

### DAG not appearing in Airflow UI

```bash
# Check for syntax errors in the DAG file
docker exec fyp-airflow-webserver python /opt/airflow/dags/dag_eodhd_ingestion_unified.py

# Force Airflow to reparse the DAG file
docker exec fyp-airflow-webserver airflow dags reserialize
```

### Task failed — view logs

```bash
# Tail the scheduler logs
docker logs -f fyp-airflow-scheduler

# Or view a specific task log via CLI
docker exec fyp-airflow-webserver airflow tasks logs \
  eodhd_complete_ingestion eodhd_scrape_AAPL <run_id>
```

### After editing `load_postgres.py` or `load_neo4j.py`

You must copy the updated file into both Airflow containers before the next DAG run:

```bash
docker cp ingestion/dags/dag_eodhd_ingestion_unified.py fyp-airflow-webserver:/opt/airflow/dags/dag_eodhd_ingestion_unified.py
docker cp ingestion/dags/dag_eodhd_ingestion_unified.py fyp-airflow-scheduler:/opt/airflow/dags/dag_eodhd_ingestion_unified.py
docker cp ingestion/etl/load_postgres.py fyp-airflow-webserver:/opt/airflow/etl/load_postgres.py
docker cp ingestion/etl/load_postgres.py fyp-airflow-scheduler:/opt/airflow/etl/load_postgres.py
docker cp ingestion/etl/load_neo4j.py fyp-airflow-webserver:/opt/airflow/etl/load_neo4j.py
docker cp ingestion/etl/load_neo4j.py fyp-airflow-scheduler:/opt/airflow/etl/load_neo4j.py
```

### PostgreSQL connection refused

```bash
docker ps | grep postgres
docker exec fyp-postgres pg_isready -U airflow
docker logs fyp-postgres --tail=20
```

### Neo4j connection refused

```bash
docker ps | grep neo4j
docker logs fyp-neo4j --tail=30
docker exec fyp-neo4j cypher-shell -u neo4j -p SecureNeo4jPass2025! "RETURN 1;"
```

---

*Last updated: 2026-03-06*
