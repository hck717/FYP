# ETL Loaders — `load_postgres.py` · `load_neo4j.py` · `load_qdrant.py`

These three modules form the **Load** layer of the FYP ingestion pipeline. They are called as Airflow `PythonOperator` tasks after every scrape task completes, and are responsible for writing structured data from local `agent_data/` files into the three storage backends.

```
Scrape Task
    └─► load_postgres_for_agent_ticker(agent_name, ticker_symbol)
    └─► load_neo4j_for_agent_ticker(agent_name, ticker_symbol)
    └─► load_qdrant_for_agent_ticker(agent_name, ticker_symbol)
```

Each loader reads the `metadata.json` for a given `agent_name / ticker_symbol` pair, checks the `storage_destination` field of each dataset, and loads only the datasets marked for that backend.

---

## Storage Destination Routing

Every scraped dataset is tagged with a `storage_destination` in `metadata.json`:

| `storage_destination` | Handled By | Database |
|---|---|---|
| `postgresql` | `load_postgres.py` | PostgreSQL (timeseries + fundamentals) |
| `neo4j` | `load_neo4j.py` | Neo4j (graph database) |
| `qdrant_prep` | `load_qdrant.py` | Qdrant (vector database) |

Datasets tagged for one destination are silently skipped by the other two loaders.

---

## `load_postgres.py`

### What It Does

Loads all `postgresql`-tagged datasets into PostgreSQL as structured rows. Automatically routes data into the correct table based on whether the dataset has a detectable date column.

### Tables

| Table | Used For | Unique Key |
|---|---|---|
| `raw_timeseries` | Time-indexed data (prices, intraday, technicals) | `(agent_name, ticker_symbol, data_name, ts_date, source)` |
| `raw_fundamentals` | Non-time-indexed data (financials, ratios, scores) | `(agent_name, ticker_symbol, data_name, as_of_date, source)` |
| `market_eod_us` | Bulk US EOD (all equities, one row per day) | `(ts_date, source)` |
| `global_economic_calendar` | Macro economic events (global, shared) | `(ts_date, source)` |
| `global_ipo_calendar` | IPO calendar (global, shared) | `(ts_date, source)` |

### Routing Logic

```
Is data_name in {bulk_eod_us, economic_calendar, ipo_calendar}?
  ├── YES → _insert_global()  → shared global table, once per day
  └── NO  → _detect_date_col()
              ├── date col found → raw_timeseries
              └── no date col   → raw_fundamentals
```

### Date Column Detection

Searches for columns named `datetime`, `date`, `timestamp`, `reportedDate`, or `t` (in that priority order).

### Unix Timestamp Normalisation

Intraday and real-time endpoints return Unix integer timestamps. The loader automatically detects integer/float date columns and converts them to `TIMESTAMP` strings before insert:

```python
pd.to_datetime(df[date_col], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
```

### Deduplication

- All inserts use `ON CONFLICT ... DO UPDATE SET payload = EXCLUDED.payload, ingested_at = NOW()`
- Safe to re-run — no duplicate rows created
- Global datasets check `ingested_at::date = CURRENT_DATE` before inserting — loaded **only once per day** regardless of how many tickers trigger the task

### Environment Variables

| Variable | Default |
|---|---|
| `POSTGRES_HOST` | `fyp-postgres` |
| `POSTGRES_PORT` | `5432` |
| `POSTGRES_DB` | `airflow` |
| `POSTGRES_USER` | `airflow` |
| `POSTGRES_PASSWORD` | `airflow` |
| `BASE_ETL_DIR` | `/opt/airflow/etl/agent_data` |

### Run Manually

```bash
# Inside the Airflow container
python /opt/airflow/etl/load_postgres.py

# Or locally
python ingestion/etl/load_postgres.py
```

---

## `load_neo4j.py`

### What It Does

Loads all `neo4j`-tagged datasets into Neo4j as nodes and relationships. Routes each dataset into the appropriate graph structure based on `data_name` keywords.

### Graph Schema

```
(:Company {ticker})
    ├── (:Fact {fact_id})      -[:ABOUT]→        (:Company)    # generic data
    ├── (:Risk {risk_id})      -[:AFFECTS]→      (:Company)    # risk factors
    └── (:Strategy {id})       -[:APPLIES_TO]→   (:Company)    # strategies / MD&A
```

### Routing Logic by `data_name`

| `data_name` contains | Node Type | Relationship | Example datasets |
|---|---|---|---|
| `profile` | `Company` | `MERGE` on ticker | `company_profile` |
| `risk` | `Risk` | `AFFECTS` → Company | `risk_factors` |
| `strategy`, `narrative`, `mda`, `md&a` | `Strategy` | `APPLIES_TO` → Company | `company_notes` |
| *(anything else)* | `Fact` | `ABOUT` → Company | `key_executives`, `stock_quote` |

### Deduplication

- All writes use `MERGE` (not `CREATE`) — re-running never creates duplicate nodes
- `Fact` nodes use a deterministic `fact_id` computed as MD5 of `ticker + data_name + row content` — prevents duplicate facts across runs
- `Company` nodes are merged on `ticker` — upserted with latest properties each run via `SET c += $props`

### Environment Variables

| Variable | Default |
|---|---|
| `NEO4J_URI` | `bolt://neo4j:7687` |
| `NEO4J_USER` | `neo4j` |
| `NEO4J_PASSWORD` | `password` |
| `BASE_ETL_DIR` | `/opt/airflow/etl/agent_data` |

### Run Manually

```bash
python ingestion/etl/load_neo4j.py
```

---

## `load_qdrant.py`

### What It Does

Loads all `qdrant_prep`-tagged datasets into Qdrant as embedded vector points. Generates dense embeddings using a **local Ollama model** (no external API key required) and upserts them into a single Qdrant collection.

### Embedding Model

| Setting | Default | Notes |
|---|---|---|
| Model | `nomic-embed-text` | Run via Ollama locally |
| Dimensions | `768` | Must match collection config |
| Batch size | `20` | Keep small for local Ollama |
| Fallback | Zero vector `[0.0 × 768]` | Used if embedding fails — prevents task crash |

> If switching to `mxbai-embed-large`, set `EMBEDDING_DIMENSION=1024` in `.env`.

### Text Column Detection

Searches for a text column to embed in this priority order:
`text` → `content` → `body` → `headline` → `title`

If none are found, all columns are concatenated with ` | ` as the text to embed.

### Qdrant Point Payload

Each point stored in Qdrant includes:

```json
{
  "agent_name": "business_analyst",
  "ticker_symbol": "AAPL",
  "data_name": "financial_news",
  "source": "eodhd",
  "<all original CSV columns>": "..."
}
```

This allows agents to filter by `ticker_symbol`, `agent_name`, or `data_name` during vector search.

### Collection

All documents across all agents and tickers are upserted into a **single shared collection** (`agentic_analyst_docs` by default). Each point gets a random UUID as its ID.

### Deduplication

- Uses `upsert()` — safe to re-run, existing points are overwritten
- Each point gets a **new UUID per run** — Qdrant upsert on UUID means old points persist alongside new ones if data grows. This is acceptable for RAG (more context = better retrieval)

### Environment Variables

| Variable | Default |
|---|---|
| `QDRANT_HOST` | `fyp-qdrant` |
| `QDRANT_PORT` | `6333` |
| `QDRANT_COLLECTION_NAME` | `agentic_analyst_docs` |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` |
| `EMBEDDING_MODEL` | `nomic-embed-text` |
| `EMBEDDING_DIMENSION` | `768` |
| `BASE_ETL_DIR` | `/opt/airflow/etl/agent_data` |

### Run Manually

```bash
# Requires Ollama running locally and Qdrant container up
python ingestion/etl/load_qdrant.py
```

---

## Data Flow Summary

```
Agent Data Directory
 ingestion/etl/agent_data/
   └── {agent_name}/
         └── {ticker_symbol}/
               ├── metadata.json          ← routing config (storage_destination per dataset)
               ├── financial_news.csv     → Qdrant
               ├── company_profile.csv    → Neo4j
               ├── intraday_1m.csv        → PostgreSQL (raw_timeseries)
               ├── fundamentals.csv       → PostgreSQL (raw_fundamentals)
               └── bulk_eod_us.csv        → PostgreSQL (market_eod_us, global)
```

## Dependencies

```
load_postgres.py  →  psycopg2, pandas
load_neo4j.py     →  neo4j (Python driver), pandas
load_qdrant.py    →  qdrant-client, requests (Ollama), pandas
```

All dependencies are installed in the Airflow Docker image via `requirements.txt`.
