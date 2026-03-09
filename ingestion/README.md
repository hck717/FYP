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
│       │   ├── company_profile.csv
│       │   └── ...  (one file per endpoint)
│       ├── TSLA/ NVDA/ MSFT/ GOOGL/
│       └── _MACRO/
│           ├── metadata.json
│           ├── treasury_rates.csv
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
    │                                 etl/agent_data/_MACRO/
    │
    ├── load_postgres_for_ticker()  ──upserts──►  PostgreSQL
    ├── load_neo4j_for_ticker()     ──merges──►  Neo4j
    ├── ingest_earnings_calls()     ──merges──►  Neo4j (Chunk nodes)
    ├── ingest_broker_reports()      ──merges──►  Neo4j (Chunk nodes)
    └── load_postgres_macro()        ──upserts──►  PostgreSQL (macro tables)
```

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
  PASS  raw_timeseries: 873691 total rows
  PASS  financial_statements: 2229 total rows
  PASS  sentiment_trends: 165 total rows
  ...

============================================================
Neo4j checks
============================================================
  PASS  Company nodes: 51
  PASS  Chunk nodes: 1829
  PASS  Neo4j vector index 'chunk_embedding': ONLINE

--- Textual Document Coverage ---
  PASS  earnings_call: 678 chunks across 5 tickers
  PASS  broker_report: 1088 chunks across 5 tickers
```

### Check PostgreSQL Only
```bash
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT data_name, period_type, COUNT(*)
  FROM raw_fundamentals
  GROUP BY data_name, period_type
  ORDER BY data_name;
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

## Textual Data Sources

### Earnings Calls
- **Location**: `/data/textual data/{TICKER}/earning_call/`
- **Format**: PDF files
- **Ingestion**: `ingest_earnings_calls.py` extracts text → splits into chunks → embeds with Ollama → stores in Neo4j

### Broker Reports
- **Location**: `/data/textual data/{TICKER}/broker/`
- **Format**: PDF files
- **Ingestion**: `ingest_broker_reports.py` extracts text → splits into chunks → embeds with Ollama → stores in Neo4j

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OLLAMA_BASE_URL` | Ollama API URL | `http://localhost:11434` (host) or `http://host.docker.internal:11434` (Docker) |
| `TRACKED_TICKERS` | Comma-separated tickers | `AAPL,TSLA,NVDA,MSFT,GOOGL` |
| `EODHD_API_KEY` | EODHD API key | Required |
| `POSTGRES_HOST` | PostgreSQL host | `postgres` |
| `NEO4J_URI` | Neo4j URI | `bolt://neo4j:7687` |

---

## Notes

- **No Timeout**: Text embedding uses Ollama with no timeout for quality
- **Cross-Platform**: Scripts auto-detect Docker environment and use appropriate Ollama URL
- **pgvector**: PostgreSQL vector extension used for semantic search
- **Neo4j Vector**: Native vector index for graph-augmented retrieval
