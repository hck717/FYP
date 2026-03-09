# Common Issues & Fixes

This guide covers the most frequent problems when running the FYP agentic analyst stack.
Run all commands from the **repo root** unless otherwise stated.

---

## 1. Ollama Connection Issues

### Symptom
```
requests.exceptions.ConnectionError: http://ollama:11434/api/embed — Connection refused
```
or the agent returns `"ollama": false` in the healthcheck.

### Cause & Fix
Ollama now runs as a **Docker service** (`fyp-ollama`). It must be healthy before
Airflow/agent containers start.

```bash
# Check the Ollama container is running and healthy
docker compose ps ollama

# Verify the API is reachable from inside the network
docker compose exec airflow-scheduler curl -s http://ollama:11434/api/tags | python3 -m json.tool

# If the container is not running, start it:
docker compose up -d ollama

# Pull the required models (run once after first startup):
docker compose exec ollama ollama pull nomic-embed-text:v1.5
docker compose exec ollama ollama pull llama3.2:3b
```

### Local dev (outside Docker)
Set `OLLAMA_BASE_URL=http://localhost:11434` in your `.env` file and run
`ollama serve` separately.

---

## 2. Empty `sentiment_trends` Table

### Symptom
The BA agent returns `trend: null` or `"No sentiment data available"` in its output.
The `eodhd_validate_data` Airflow task may also fail with:
```
AirflowException: Data validation failed — sentiment_trends is EMPTY for ticker=AAPL
```

### Cause
The EODHD ingestion DAG has not run yet, the API key is invalid, or the
`sentiments` endpoint returned an empty response.

### Fix
```bash
# 1. Trigger the ingestion DAG manually
docker compose exec airflow-scheduler \
  airflow dags trigger eodhd_complete_ingestion

# 2. Check the task log for the sentiment task
docker compose exec airflow-scheduler \
  airflow tasks logs eodhd_complete_ingestion eodhd_scrape_AAPL <run_id>

# 3. Verify the EODHD API key is set
docker compose exec airflow-scheduler env | grep EODHD_API_KEY

# 4. Test the EODHD sentiment endpoint directly
curl "https://eodhd.com/api/sentiments?s=AAPL&api_token=YOUR_KEY&fmt=json"
```

### Fallback behaviour
The BA agent automatically falls back to **local VADER/TextBlob/keyword scoring**
over recent Neo4j text chunks when PostgreSQL sentiment is empty.
No action needed — the agent will still return a sentiment signal.

---

## 3. PDF Ingestion Not Running

### Symptom
Earnings call or broker report queries return no results from Neo4j chunks.

### Cause
PDF files are placed in `data/{TICKER}/earning_call/` or `data/{TICKER}/broker/`
but the ingestion task hasn't processed them.

### Fix
```bash
# Trigger the main DAG — PDF tasks run automatically at the end
docker compose exec airflow-scheduler \
  airflow dags trigger eodhd_complete_ingestion

# Or run ingestion scripts directly for a single ticker:
docker compose exec airflow-scheduler \
  python /opt/airflow/etl/ingest_earnings_calls.py AAPL

docker compose exec airflow-scheduler \
  python /opt/airflow/etl/ingest_broker_reports.py AAPL --only-new

# Check scan state (which PDFs were last processed):
cat data/AAPL/.pdf_scan_state_earning_call.json
cat data/AAPL/.pdf_scan_state_broker.json
```

### Notes
- The `--only-new` flag skips PDFs that haven't changed since last run.
- To force re-ingestion of all PDFs, delete the `.pdf_scan_state_*.json` files.

---

## 4. How to Verify DB Health

### Quick check with `inspect_db.py`
```bash
docker compose exec airflow-scheduler \
  python /opt/airflow/etl/inspect_db.py
```
This prints row counts, last-updated timestamps, and flags empty tables.

### Manual PostgreSQL queries
```bash
docker compose exec postgres psql -U airflow -d airflow -c "
  SELECT ticker, COUNT(*) AS rows, MAX(as_of_date) AS latest
  FROM sentiment_trends
  GROUP BY ticker
  ORDER BY ticker;
"

# Check all key tables at once
docker compose exec postgres psql -U airflow -d airflow -c "
  SELECT 'sentiment_trends' AS t, COUNT(*) FROM sentiment_trends
  UNION ALL SELECT 'historical_prices_eod', COUNT(*) FROM raw_timeseries
  UNION ALL SELECT 'financial_statements', COUNT(*) FROM financial_statements;
"
```

### Neo4j chunk count
```bash
# Open Neo4j browser: http://localhost:7474
# Run this Cypher:
# MATCH (c:Company)-[:HAS_CHUNK]->(ch:Chunk)
# RETURN c.ticker, count(ch) AS chunks
# ORDER BY chunks DESC

# Or via the CLI:
docker compose exec neo4j cypher-shell -u neo4j -p SecureNeo4jPass2025! \
  "MATCH (c:Company)-[:HAS_CHUNK]->(ch:Chunk) RETURN c.ticker, count(ch) AS chunks ORDER BY chunks DESC"
```

---

## 5. Airflow DAG Not Running

```bash
# Unpause the DAG
docker compose exec airflow-scheduler \
  airflow dags unpause eodhd_complete_ingestion

# Check for import errors
docker compose exec airflow-scheduler \
  airflow dags list-import-errors

# View recent task states
docker compose exec airflow-scheduler \
  airflow tasks states-for-dag-run eodhd_complete_ingestion <run_id>
```

---

## 6. CRAG Returning Too Many INCORRECT Verdicts

If the agent frequently falls back to web search even when local data exists:

```bash
# Lower the CORRECT threshold (add to .env or docker-compose.yml):
CRAG_CORRECT_THRESHOLD=0.45
CRAG_AMBIGUOUS_THRESHOLD=0.30
```

Monitor CRAG decisions in the Airflow task logs — every judgment is now logged:
```
[CRAG] score=0.412 → AMBIGUOUS (thresholds: correct>=0.60, ambiguous>=0.40)
```

---

## 7. GPU Support for Ollama (Optional)

If you have an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) installed:

1. Uncomment the `deploy` block in `docker-compose.yml` under the `ollama` service.
2. Restart: `docker compose up -d ollama`
3. Verify GPU is detected: `docker compose exec ollama ollama list`

---

## 8. Embedding Model Version Mismatch

If you upgrade the Ollama embedding model and vector search quality degrades:

```bash
# Check which embedding version is stored on chunks
docker compose exec neo4j cypher-shell -u neo4j -p SecureNeo4jPass2025! \
  "MATCH (ch:Chunk) RETURN ch.embedding_version, count(*) GROUP BY ch.embedding_version"

# Re-ingest all chunks with the new model:
docker compose exec airflow-scheduler \
  python /opt/airflow/etl/load_neo4j.py --all
```

To prevent accidental model upgrades, the embedding model is locked to
`nomic-embed-text:v1.5` in `docker-compose.yml`. Change `EMBEDDING_MODEL`
only when you intend to re-embed all data.

---

## 9. Data Freshness Warnings and the Pre-Check Layer

The Business Analyst agent runs a `precheck_data_coverage` node immediately after
`metadata_precheck`.  This node checks coverage and freshness **before** any
retrieval or LLM calls.  When something is wrong it sets `data_coverage_warning`
in the output JSON so you can act on it without trawling logs.

### What the warnings mean

| Warning text | Root cause | Fix |
|---|---|---|
| `Neo4j chunk count … is low (N < 20)` | The FMP ingestion DAG has not run or has not indexed documents for this ticker. | Run the `fmp_ingest` Airflow DAG; check Airflow logs for ingest errors. |
| `pgvector chunk count … is low (N < 5)` | The pgvector embedding step has not populated data for this ticker. | Run the `pgvec_embed` Airflow DAG or trigger `etl/load_pgvector.py --ticker <TICKER>`. |
| `Sentiment data … is stale (age=N days > 7)` | The `sentiment_trends` PostgreSQL table has not been refreshed recently. | Re-run the `eodhd_sentiment` DAG or manually call `etl/load_sentiment.py --ticker <TICKER>`. |
| `No sentiment data found … in PostgreSQL` | No rows exist for the ticker in `sentiment_trends`. | Run the sentiment ingestion pipeline for this ticker; check that the ticker symbol matches the EODHD convention. |

### Checking `data_coverage_warning` in the output

```python
import json, subprocess

result = subprocess.run(
    ["python", "-m", "agents.business_analyst.agent", "--ticker", "AAPL"],
    capture_output=True, text=True,
)
output = json.loads(result.stdout)
if output.get("data_coverage_warning"):
    print("WARNING:", output["data_coverage_warning"])
```

Or via CLI verbose mode (warnings are also printed to stderr with the `!?` symbol):

```bash
python -m agents.business_analyst.agent --ticker AAPL --verbose 2>&1 | grep "!?"
```

### Sentiment-only query short-circuit

When the query is primarily about market sentiment (contains keywords like
`sentiment`, `bullish`, `bearish`, `market view`, etc.) **and** the PostgreSQL
snapshot is fresh (≤ 7 days), the pre-check node pre-sets
`crag_status=CORRECT` and bypasses the full bi-encoder/cross-encoder retrieval
pipeline.  This reduces latency from ~10 s to ~2 s for those queries.

To force full retrieval even for sentiment queries (e.g. for debugging),
temporarily set:
```bash
CRAG_CORRECT_THRESHOLD=0.99  # prevents pre-check short-circuit from reaching CORRECT
```

### `use_sentiment_db` and `sentiment_is_fresh` state fields

These internal flags are visible in verbose logs but are **not** in the final
JSON output.  They control whether downstream nodes prefer the PostgreSQL
snapshot (`use_sentiment_db=True`) or the local NLP fallback over Neo4j chunks
(`use_sentiment_db=False`).  The `data_coverage_warning` field summarises the
outcome for operators.
