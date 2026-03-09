# Business Analyst Agent

> **Architecture:** Optimized Adaptive Agentic Graph RAG (v2)
> **Status:** Live-tested, all 5 tickers validated
> **LLM Backend:** DeepSeek API (`deepseek-reasoner`) — cloud API, no local GPU required

---

## Role

The Business Analyst Agent is the **qualitative intelligence layer** of the multi-agent equity research system. It answers structural questions about a company — competitive moat, business model, risk factors, strategic positioning, and earnings call highlights — by retrieving verified facts from local knowledge bases and synthesising them with an LLM.

It does **NOT** make buy/sell judgements or produce price targets. It surfaces grounded analysis and data gaps for the Supervisor and Synthesizer to interpret.

**Handled by this agent:**
- Competitive moat depth and trajectory
- Business model and revenue composition
- Risk factor synthesis (from filings, earnings calls, broker reports)
- Historical sentiment trends (bullish/bearish/neutral %)
- Management guidance and earnings call Q&A
- Broker research report synthesis
- Qualitative narrative with cited source IDs

**Not handled here (separate agents):**
- Breaking news and real-time events → Web Search Agent
- Financial ratios, DCF, valuation → Financial Modelling Agent
- Macro environment → Phase 3 (not yet implemented)

---

## Architecture: Optimized Adaptive Agentic Graph RAG

The pipeline classifies every query into one of three paths before retrieving, then evaluates retrieval quality (CRAG) on the complex path and adaptively rewrites or falls back as needed.

```
Query + Ticker
    │
    ▼
metadata_precheck         ←  Neo4j chunk counts, pgvector status, sentiment flag
    │
    ▼
precheck_data_coverage    ←  Coverage warnings, sentiment freshness gate, sentiment-query short-circuit
    │
    ▼
fetch_sentiment_data      ←  PostgreSQL: bullish/bearish/neutral % + Company node + graph community
    │
    ▼
classify_query            ←  Rule-based only: SIMPLE / NUMERICAL / COMPLEX (no LLM call)
    │
    ├─ SIMPLE     → fast_path_retrieval    (vector + BM25, small top_k, no rerank)
    │                    │
    ├─ NUMERICAL  → numerical_path         (fast retrieval + guaranteed sentiment fetch)
    │                    │
    └─ COMPLEX    → complex_retrieval      (multi-stage: bi-encoder recall → cross-encoder
                         │                  rerank → RRF fusion + section-diversity top-up
                         │                  + graph traversal)
                         ▼
                    crag_evaluate          ←  CORRECT (≥0.6) / AMBIGUOUS (0.4–0.6) / INCORRECT (<0.4)
                         │
                         ├─ CORRECT    → generate_analysis
                         ├─ AMBIGUOUS  → rewrite_query → retry complex_retrieval (max 2 loops)
                         └─ INCORRECT  → web_search_fallback (only if no local data at all)
                                              │
                    (All paths converge)      │
                         ▼  ←────────────────┘
                    semantic_cache_check   ←  pass-through (caching is at retrieval layer)
                         │
                         ▼
                    format_json_output     ←  citation grounding check, JSON assembly, inline sanitise
                         │
                        END → return to Supervisor
```

**SIMPLE and NUMERICAL paths bypass CRAG** — they pre-set `crag_status=CORRECT` and `confidence=1.0` and go straight to generation.

**COMPLEX path** runs full multi-stage retrieval followed by CRAG evaluation. INCORRECT is only routed to web fallback when there is genuinely no local data; otherwise the LLM generates from whatever was found and surfaces gaps in `missing_context`.

---

## Input-to-Output Walkthrough

### 1. Inputs
```python
task   = "What revenue guidance did NVIDIA give for next quarter?"
ticker = "NVDA"
```

### 2. `metadata_precheck`
Queries Neo4j and pgvector for chunk counts and index readiness. Caches the result in `state["metadata_profile"]` so later nodes don't re-query.

### 3. `precheck_data_coverage`
Reads `state["metadata_profile"]` and sets three diagnostic flags **before** any retrieval or LLM calls:

| State field | Type | Description |
|---|---|---|
| `data_coverage_warning` | `Optional[str]` | Human-readable warning when chunk coverage is thin (`< 20` Neo4j chunks or `< 5` pgvector chunks) or sentiment data is stale/missing. `None` when healthy. Included in the final JSON output. |
| `use_sentiment_db` | `bool` | `True` when the PostgreSQL sentiment snapshot is within the 7-day freshness window. |
| `sentiment_is_fresh` | `bool` | Mirrors `use_sentiment_db`; exposed separately for routing logic clarity. |

**Sentiment-query short-circuit:** When the query contains sentiment keywords (`bullish`, `bearish`, `sentiment`, `market view`, etc.) **and** the DB snapshot is fresh, `crag_status` is pre-set to `CORRECT` (`confidence=1.0`) — skipping multi-stage retrieval and CRAG evaluation entirely for those lightweight queries (~2 s vs ~10 s).

### 4. `fetch_sentiment_data`
- **Sentiment** — `SELECT bullish_pct, bearish_pct, neutral_pct, trend FROM sentiment_trends WHERE ticker=?` (PostgreSQL)
- **Company node** — `MATCH (c:Company {ticker: $t}) RETURN c` (Neo4j), used for the `company_overview` section (market cap, P/E, sector, etc.)
- **Graph community summary** — relationship-count centrality across `:RISK`, `:STRATEGY`, `:COMPETES_WITH` edges (2A: Graph RAG)

### 5. `classify_query`
Uses a **rule-based classifier** (`rule_based_classify` in `tools.py`) — no LLM call. Returns one of:
- `SIMPLE` — factual single-answer questions (e.g. "What sector is Apple in?")
- `NUMERICAL` — metric/ratio/guidance questions (e.g. "What revenue guidance did NVIDIA give?")
- `COMPLEX` — multi-faceted qualitative questions (e.g. "What are Tesla's key business risks?")

Always falls back to `COMPLEX` (the safe default).

### 6. Retrieval paths

#### SIMPLE — `fast_path_retrieval`
- Calls `toolkit.retrieve_fast(query, ticker)`
- Uses a smaller `fast_path_top_k` budget (default: 5)
- Neo4j vector similarity search + optional BM25 re-scoring
- No cross-encoder reranking
- Pre-sets `crag_status=CORRECT, confidence=1.0`

#### NUMERICAL — `numerical_path`
- Same `retrieve_fast()` call as SIMPLE
- Additionally ensures sentiment data is fetched if not already present
- Pre-sets `crag_status=CORRECT, confidence=1.0`

#### COMPLEX — `complex_retrieval`
Multi-stage pipeline:
1. **Bi-encoder recall** — Neo4j vector index (`chunk_embedding`) + pgvector cosine search, top-100 each
2. **Boilerplate filter** — garbled / cover-page chunks removed before reranking
3. **Cross-encoder rerank** — `ms-marco-MiniLM-L-6-v2` scores top candidates; when max CE score < 0.4 the blend falls back to 70% dense + 30% BM25
4. **RRF fusion** — Reciprocal Rank Fusion merges dense, BM25, cross-encoder, and graph results into a final ranked list
5. **Section-diversity top-up** — after fusion, ensures at least `min_chunks_per_section` (default: 3) chunks from each key section (`earnings_call`, `broker_report`) appear in the final result, even if their RRF scores are lower. This prevents vector-similarity skew from entirely omitting one document type.
6. **Graph traversal** — top-25 graph facts from `:RISK`, `:STRATEGY`, `:COMPETES_WITH`, `:MENTIONS` edges

Controlled by `multi_stage_recall_k` (default: 100 per source).

### 7. `crag_evaluate` (COMPLEX path only)
Scores the top retrieved chunks against the query using the cross-encoder. Classifies:
- **CORRECT** (confidence ≥ 0.6) → proceed to generation
- **AMBIGUOUS** (0.4 ≤ confidence < 0.6) → rewrite query and retry (up to `max_rewrite_loops`, default: 2)
- **INCORRECT** (confidence < 0.4) → web fallback only if no local data; otherwise generate and surface gaps

### 8. `generate_analysis`
Builds a rich context string:
- Analyst question (framed prominently)
- Sentiment block (bullish/bearish/neutral %)
- Graph facts (top-25 relationship facts)
- Graph community summary
- Top retrieved chunks — up to 7, each with `chunk_id`, `relevance`, `source_name` (human-readable institution/doc name), and `temporal_band`
- Temporal 2B tagging: RECENT vs. HISTORICAL chunks are separated so the LLM can analyse the *delta*

Sends to `llm.generate()` (model: `deepseek-reasoner` via DeepSeek API, `temperature=0.2`, single API call per run).

### 9. `format_json_output`
- Merges LLM output, sentiment block, company node, and web fallback results
- Priority for `company_overview`: Neo4j Company node > LLM-generated > graph_facts extraction
- Runs citation grounding check (`validate_citations()`)
- Strips ungrounded inline citations from prose (backstop sanitiser) — handles both old `[TICKER::section::hash]` and new `[source_name | chunk_id]` formats
- Normalises `key_risks` (nulls out invented source IDs, coerces `mitigation_observed: "null"`)
- Sanitises `competitive_moat.sources` (bare string arrays of hallucinated IDs)
- Falls back `qualitative_summary` from `management_guidance` when LLM left it null

### 10. Output
Returns a structured JSON dict (see Output Schema below).

---

## Data Sources

| Source | Content | Storage |
|--------|---------|---------|
| `sentiment_trends` | Bullish / bearish / neutral % per ticker + trend direction | PostgreSQL |
| `:Company` nodes | 85+ financial properties (market cap, P/E, sector, margins, …) | Neo4j |
| `:Chunk` nodes (earnings_call) | Earnings call transcript chunks | Neo4j vector index |
| `:Chunk` nodes (broker_report) | Broker research report chunks | Neo4j vector index |
| `:Chunk` nodes (description, highlights, …) | Company profile sections | Neo4j vector index |
| `text_chunks` | pgvector embeddings (same documents, alternative index) | PostgreSQL |
| `:RISK`, `:STRATEGY`, `:COMPETES_WITH` edges | Relationship graph for graph traversal | Neo4j |

### Current Chunk Coverage (Neo4j, post full re-ingestion)

| Ticker | earnings_call | broker_report | other | Total |
|--------|---------------|---------------|-------|-------|
| AAPL   | ~162          | ~878          | 13    | ~1053 |
| TSLA   | ~162          | ~391          | 13    | ~566  |
| NVDA   | ~162          | ~292          | 12    | ~466  |
| MSFT   | ~162          | ~182          | 13    | ~357  |
| GOOGL  | ~162          | ~795          | 12    | ~969  |

**Total: ~3,411 chunks across 5 tickers**

All chunks use content-hash MERGE in Neo4j — re-ingestion is idempotent (no duplicates).

---

## LLM & Models

| Purpose | Model | Backend | Notes |
|---------|-------|---------|-------|
| Primary generation | `deepseek-reasoner` | DeepSeek API (cloud) | `temperature=0.2`, one call per run; built-in chain-of-thought via `reasoning_content` |
| Query classification | rule-based only | n/a | No LLM call — keyword pattern matching in `tools.py` |
| Query rewrite | `deepseek-reasoner` | DeepSeek API (cloud) | Only fires on AMBIGUOUS CRAG path (rare) |
| Bi-encoder (embedding) | `nomic-embed-text:v1.5` | Ollama (local) | Neo4j + pgvector, 768-dim |
| Cross-encoder (rerank) | `ms-marco-MiniLM-L-6-v2` | HuggingFace (local) | CPU inference |

> **Ollama is kept for embeddings only.** All LLM generation uses the DeepSeek cloud API. No local GPU is required for generation.

---

## Chunk ID Formats & Citation Format

Chunks are stored in Neo4j with the format `TICKER::section::content_hash_prefix`, e.g.:
- `AAPL::broker_report::efd9122b713b00f6`
- `AAPL::earnings_call::3a7c1d9f2b0e5f8a`

### How source_name works

The `vector_search` Cypher query now fetches `node.institution` and `node.source_file` alongside the chunk text. A human-readable `source_name` is derived:

| Section | source_name derivation | Example |
|---------|----------------------|---------|
| `broker_report` | `institution` property on Chunk node (set during ingestion from PDF filename) | `"Wells Fargo"` |
| `earnings_call` | `{TICKER} Earnings Call {quarter}` derived from `source_file` or `filing_date` | `"AAPL Earnings Call Q4 2025"` |
| other | section name title-cased | `"Highlights"` |

### Citation format in LLM output

The LLM is instructed to use:
```
[<source_name> | <chunk_id>]
```
Example:
```
Apple's services growth is accelerating [Wells Fargo | AAPL::broker_report::efd9122b713b00f6].
```

The `_strip_ungrounded_inline_citations` post-processor recognises this format and validates the `chunk_id` part (right of `|`) against the retrieved set.

---

## Output Schema

```json
{
  "agent": "business_analyst",
  "ticker": "AAPL",
  "query_date": "2026-03-09",
  "company_overview": {
    "name": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "market_cap": 3.4e12,
    "pe_ratio": 29.5,
    "profit_margin": 0.253
  },
  "sentiment": {
    "bullish_pct": 90.88,
    "bearish_pct": 0.0,
    "neutral_pct": 9.12,
    "trend": "bullish",
    "source": "postgresql:sentiment_trends",
    "sentiment_interpretation": "..."
  },
  "competitive_moat": {
    "rating": "wide",
    "key_strengths": ["..."],
    "vulnerabilities": ["..."],
    "sources": ["AAPL::broker_report::efd9122b713b00f6"],
    "moat_trajectory": "..."
  },
  "qualitative_analysis": {
    "narrative": "...",
    "sentiment_signal": "...",
    "strategic_implication": "...",
    "data_quality_note": "..."
  },
  "management_guidance": {
    "most_recent_guidance": "...",
    "earnings_call_highlights": ["..."],
    "near_term_catalysts": [{"catalyst": "...", "direction": "POSITIVE", "timeline": "...", "magnitude": null, "source": null}],
    "forward_outlook_summary": "..."
  },
  "sentiment_verdict": {
    "signal": "CONSTRUCTIVE",
    "rationale": "...",
    "confidence": "HIGH"
  },
  "key_risks": [
    {
      "risk": "...",
      "severity": "HIGH",
      "mitigation_observed": "...",
      "source": "AAPL::broker_report::efd9122b713b00f6"
    }
  ],
  "missing_context": [
    { "gap": "...", "severity": "MEDIUM" }
  ],
  "crag_status": "CORRECT",
  "confidence": 0.85,
  "fallback_triggered": false,
  "qualitative_summary": "One-to-two sentence executive summary.",
  "data_coverage_warning": null
}
```

**`crag_status`** values: `CORRECT` | `AMBIGUOUS` | `INCORRECT`

**`confidence`**: 0.0–1.0 — 1.0 for SIMPLE/NUMERICAL (bypass CRAG), real cross-encoder score for COMPLEX.

**`data_coverage_warning`**: `null` when data is healthy; a human-readable string when chunk coverage is thin or sentiment data is stale/missing.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEEPSEEK_API_KEY` | **Required.** DeepSeek API key for LLM generation | — |
| `OLLAMA_BASE_URL` | Ollama API URL (embeddings only) | `http://localhost:11434` |
| `NEO4J_URI` | Neo4j bolt URI | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `SecureNeo4jPass2025!` |
| `BUSINESS_ANALYST_MODEL` | Primary LLM model name | `deepseek-reasoner` |
| `BUSINESS_ANALYST_LLM_PROVIDER` | LLM provider (`deepseek`) | `deepseek` |
| `BUSINESS_ANALYST_MAX_TOKENS` | Max output tokens for DeepSeek API | `8000` |
| `BUSINESS_ANALYST_REQUEST_TIMEOUT` | HTTP timeout for DeepSeek API (s) | `120` |
| `RAG_TOP_K` | Chunks returned to LLM | `15` |
| `RAG_SCORE_THRESHOLD` | Minimum relevance score | `0.6` |
| `BA_FAST_PATH_TOP_K` | Top-K for SIMPLE/NUMERICAL paths | `15` |
| `BA_MULTI_STAGE_RECALL_K` | Bi-encoder recall pool per source | `100` |
| `BA_MIN_CHUNKS_PER_SECTION` | Min chunks per section (diversity guarantee) | `3` |
| `BA_MAX_REWRITE_LOOPS` | Max query rewrites on AMBIGUOUS | `3` |
| `CRAG_CORRECT_THRESHOLD` | Confidence ≥ this → CORRECT | `0.6` |
| `CRAG_AMBIGUOUS_THRESHOLD` | Confidence ≥ this → AMBIGUOUS | `0.4` |

---

## Running the Agent

### CLI — basic

```bash
# From host (venv activated)
python -m agents.business_analyst.agent --ticker AAPL

# Inside Docker
docker exec fyp-airflow-webserver \
    python -m agents.business_analyst.agent --ticker AAPL \
    --task "What is Apple's competitive moat?"
```

### CLI — with verbose pipeline trace (stderr)

```bash
docker exec fyp-airflow-webserver \
    python -m agents.business_analyst.agent \
    --ticker TSLA \
    --task "What are Tesla's key business risks?" \
    --verbose
```

Example stderr output:
```
[10:15:01] [01] >> METADATA PRECHECK  |  ticker=TSLA
[10:15:01] [02]   OK METADATA PRECHECK done  |  neo4j_chunks=566  pgvec_chunks=10  sentiment=yes  index=ONLINE  (0.08s)
[10:15:01] [03] >> PRECHECK DATA COVERAGE  |  ticker=TSLA
[10:15:01] [04]   OK PRECHECK DATA COVERAGE done  |  sentiment_fresh=True  neo4j=566  pgvec=10  warning=no
[10:15:01] [05] >> FETCH SENTIMENT  |  ticker=TSLA
[10:15:01] [06]   OK FETCH SENTIMENT done  |  bullish=62.5%  bearish=15.0%  neutral=22.5%  trend=bullish  (0.08s)
[10:15:01] [07] >> CLASSIFY QUERY  |  query='What are Tesla's key business risks?'
[10:15:01] [08]   OK CLASSIFY QUERY done  |  class=COMPLEX  method=rule-based  (0.00s)
[10:15:01] [09] >> COMPLEX RETRIEVAL (multi-stage)  |  ticker=TSLA
[10:15:11] [10]   OK COMPLEX RETRIEVAL done  |  chunks=15  graph_facts=0  top_score=0.738  (9.90s)
[10:15:11] [11] >> CRAG EVALUATE  |  ticker=TSLA  chunks_to_score=15
[10:15:11] [12]   OK CRAG EVALUATE done  |  status=CORRECT  confidence=0.738  (0.00s)
[10:15:11] [13] >> GENERATE ANALYSIS (LLM)  |  ticker=TSLA  chunks=15  graph_facts=0
[10:19:01] [14]   OK GENERATE ANALYSIS done  |  (230s)
[10:19:01] [15]   -- SEMANTIC CACHE CHECK  |  cache operates at retrieval layer — pass-through
[10:19:01] [16] >> FORMAT JSON OUTPUT  |  ticker=TSLA  crag=CORRECT  confidence=0.738  fallback=False
[10:19:01] [17]   OK CITATION GROUNDING  |  cited=0  grounded=0  ungrounded=0  rate=100.0%
[10:19:01] [18]   OK FORMAT JSON OUTPUT done
```

> **Note on generation latency:** `deepseek-reasoner` performs a full chain-of-thought reasoning pass before producing output. Expect 2–6 minutes per generation call depending on context size. The reasoning trace is logged at DEBUG level (`reasoning_content`) and is not included in the output JSON.

### CLI — with citation validation report

```bash
docker exec fyp-airflow-webserver \
    python -m agents.business_analyst.agent --ticker AAPL \
    --task "What is Apple's competitive moat?" \
    --validate-citations
```

Appends to stderr after the run:
```
--- Citation Grounding Report ---
  Total cited IDs  : 5
  Grounded         : 5
  Ungrounded       : 0
  Grounding rate   : 100.0%
---------------------------------
```

### Programmatic

```python
from agents.business_analyst.agent import run, run_full_analysis

# Single question
result = run(task="What is Apple's competitive moat?", ticker="AAPL")

# Full institutional-grade dossier (all 7 pillars)
dossier = run_full_analysis(ticker="AAPL")
```

---

## Citation Grounding

Every factual claim in the output should cite a chunk_id from the retrieved set. The agent enforces this at two layers:

1. **Prompt** — the system prompt instructs the LLM to only cite verbatim chunk_ids from the `RETRIEVED DOCUMENT CHUNKS` section, using the format `[source_name | chunk_id]`. The `source_name` is shown in each chunk header (e.g. `"Wells Fargo"` for a broker report, `"AAPL Earnings Call Q4 2025"` for an earnings call).
2. **Post-processor** — `_strip_ungrounded_inline_citations()` scans all prose fields and removes any citation where the chunk_id doesn't match a real retrieved ID. Handles both formats: `[source_name | chunk_id]` and legacy `[TICKER::section::hash]`.

`validate_citations(output, retrieval)` provides a machine-readable grounding report:

```python
from agents.business_analyst.agent import validate_citations

report = validate_citations(output=result, retrieval=retrieval_result)
# {
#   "total_cited": 8,
#   "grounded": 8,
#   "ungrounded": 0,
#   "grounding_rate_pct": 100.0,
#   "ungrounded_ids": []
# }
```

---

## Section-Diversity Guarantee

After RRF fusion, the complex retrieval path applies a **section-diversity top-up**. This ensures that at least `min_chunks_per_section` (default: 3, configurable via `BA_MIN_CHUNKS_PER_SECTION`) chunks from `earnings_call` and `broker_report` are always present in the final result, even when vector-similarity scores skew toward one document type for a given query.

This prevents scenarios like "What is Apple's competitive moat?" retrieving exclusively broker report chunks and none from earnings calls.

---

## Infrastructure

| Service | Container | Status | Notes |
|---------|-----------|--------|-------|
| PostgreSQL | `fyp-postgres` | healthy | `sentiment_trends`, `financial_statements`, `text_chunks` (pgvector) |
| Neo4j | `fyp-neo4j` | healthy | 51 Company nodes, ~3,411 Chunk nodes, vector index ONLINE |
| Ollama | `fyp-ollama-1` | running | `nomic-embed-text:v1.5` for embeddings only |
| DeepSeek API | cloud | — | `deepseek-reasoner` for generation; requires `DEEPSEEK_API_KEY` |

---

## Testing

```bash
# Run all agent tests
pytest agents/business_analyst/tests/ -v

# Run specific test
pytest agents/business_analyst/tests/test_agent.py::test_full_pipeline -v

# Quick smoke-test three query classes from the CLI
docker exec fyp-airflow-webserver \
    python -m agents.business_analyst.agent --ticker AAPL \
    --task "What sector is Apple in?" --verbose          # → SIMPLE

docker exec fyp-airflow-webserver \
    python -m agents.business_analyst.agent --ticker AAPL \
    --task "What revenue guidance did NVIDIA give?" --verbose   # → NUMERICAL

docker exec fyp-airflow-webserver \
    python -m agents.business_analyst.agent --ticker AAPL \
    --task "What are Apple's key business risks and how does sentiment reflect them?" \
    --verbose                                            # → COMPLEX
```

## Role

The Business Analyst Agent is the **qualitative intelligence layer** of the multi-agent equity research system. It answers structural questions about a company — competitive moat, business model, risk factors, strategic positioning, and earnings call highlights — by retrieving verified facts from local knowledge bases and synthesising them with an LLM.

It does **NOT** make buy/sell judgements or produce price targets. It surfaces grounded analysis and data gaps for the Supervisor and Synthesizer to interpret.

**Handled by this agent:**
- Competitive moat depth and trajectory
- Business model and revenue composition
- Risk factor synthesis (from filings, earnings calls, broker reports)
- Historical sentiment trends (bullish/bearish/neutral %)
- Management guidance and earnings call Q&A
- Broker research report synthesis
- Qualitative narrative with cited source IDs

**Not handled here (separate agents):**
- Breaking news and real-time events → Web Search Agent
- Financial ratios, DCF, valuation → Financial Modelling Agent
- Macro environment → Phase 3 (not yet implemented)

---

## Architecture: Optimized Adaptive Agentic Graph RAG

The pipeline classifies every query into one of three paths before retrieving, then evaluates retrieval quality (CRAG) on the complex path and adaptively rewrites or falls back as needed.

```
Query + Ticker
    │
    ▼
metadata_precheck         ←  Neo4j chunk counts, pgvector status, sentiment flag
    │
    ▼
precheck_data_coverage    ←  Coverage warnings, sentiment freshness gate, sentiment-query short-circuit
    │
    ▼
fetch_sentiment_data      ←  PostgreSQL: bullish/bearish/neutral % + Company node + graph community
    │
    ▼
classify_query            ←  LLM (lightweight): SIMPLE / NUMERICAL / COMPLEX
    │
    ├─ SIMPLE     → fast_path_retrieval    (vector + BM25, small top_k, no rerank)
    │                    │
    ├─ NUMERICAL  → numerical_path         (fast retrieval + guaranteed sentiment fetch)
    │                    │
    └─ COMPLEX    → complex_retrieval      (multi-stage: bi-encoder recall → cross-encoder
                         │                  rerank → RRF fusion + graph traversal)
                         ▼
                    crag_evaluate          ←  CORRECT (≥0.6) / AMBIGUOUS (0.4–0.6) / INCORRECT (<0.4)
                         │
                         ├─ CORRECT    → generate_analysis
                         ├─ AMBIGUOUS  → rewrite_query → retry complex_retrieval (max 2 loops)
                         └─ INCORRECT  → web_search_fallback (only if no local data at all)
                                              │
                    (All paths converge)      │
                         ▼  ←────────────────┘
                    semantic_cache_check   ←  pass-through (caching is at retrieval layer)
                         │
                         ▼
                    format_json_output     ←  citation grounding check, JSON assembly, inline sanitise
                         │
                        END → return to Supervisor
```

**SIMPLE and NUMERICAL paths bypass CRAG** — they pre-set `crag_status=CORRECT` and `confidence=1.0` and go straight to generation.

**COMPLEX path** runs full multi-stage retrieval followed by CRAG evaluation. INCORRECT is only routed to web fallback when there is genuinely no local data; otherwise the LLM generates from whatever was found and surfaces gaps in `missing_context`.

---

## Input-to-Output Walkthrough

### 1. Inputs
```python
task   = "What revenue guidance did NVIDIA give for next quarter?"
ticker = "NVDA"
```

### 2. `metadata_precheck`
Queries Neo4j and pgvector for chunk counts and index readiness. Caches the result in `state["metadata_profile"]` so later nodes don't re-query.

### 3. `precheck_data_coverage`
Reads `state["metadata_profile"]` and sets three diagnostic flags **before** any retrieval or LLM calls:

| State field | Type | Description |
|---|---|---|
| `data_coverage_warning` | `Optional[str]` | Human-readable warning when chunk coverage is thin (`< 20` Neo4j chunks or `< 5` pgvector chunks) or sentiment data is stale/missing. `None` when healthy. Included in the final JSON output. |
| `use_sentiment_db` | `bool` | `True` when the PostgreSQL sentiment snapshot is within the 7-day freshness window. |
| `sentiment_is_fresh` | `bool` | Mirrors `use_sentiment_db`; exposed separately for routing logic clarity. |

**Sentiment-query short-circuit:** When the query contains sentiment keywords (`bullish`, `bearish`, `sentiment`, `market view`, etc.) **and** the DB snapshot is fresh, `crag_status` is pre-set to `CORRECT` (`confidence=1.0`) — skipping multi-stage retrieval and CRAG evaluation entirely for those lightweight queries (~2 s vs ~10 s).

### 4. `fetch_sentiment_data`
- **Sentiment** — `SELECT bullish_pct, bearish_pct, neutral_pct, trend FROM sentiment_trends WHERE ticker=?` (PostgreSQL)
- **Company node** — `MATCH (c:Company {ticker: $t}) RETURN c` (Neo4j), used for the `company_overview` section (market cap, P/E, sector, etc.)
- **Graph community summary** — relationship-count centrality across `:RISK`, `:STRATEGY`, `:COMPETES_WITH` edges (2A: Graph RAG)

### 5. `classify_query`
Uses a **rule-based classifier** (`rule_based_classify` in `tools.py`) — no LLM call. Returns one of:
- `SIMPLE` — factual single-answer questions (e.g. "What sector is Apple in?")
- `NUMERICAL` — metric/ratio/guidance questions (e.g. "What revenue guidance did NVIDIA give?")
- `COMPLEX` — multi-faceted qualitative questions (e.g. "What are Tesla's key business risks?")

Always falls back to `COMPLEX` (the safe default).

### 6. Retrieval paths

#### SIMPLE — `fast_path_retrieval`
- Calls `toolkit.retrieve_fast(query, ticker)`
- Uses a smaller `fast_path_top_k` budget (default: 5)
- Neo4j vector similarity search + optional BM25 re-scoring
- No cross-encoder reranking
- Pre-sets `crag_status=CORRECT, confidence=1.0`

#### NUMERICAL — `numerical_path`
- Same `retrieve_fast()` call as SIMPLE
- Additionally ensures sentiment data is fetched if not already present
- Pre-sets `crag_status=CORRECT, confidence=1.0`

#### COMPLEX — `complex_retrieval`
Multi-stage pipeline:
1. **Bi-encoder recall** — Neo4j vector index (`chunk_embedding`) + pgvector cosine search, top-100 each
2. **Cross-encoder rerank** — `ms-marco-MiniLM-L-6-v2` scores top candidates; when max CE score < 0.4 the blend falls back to 70% dense + 30% BM25
3. **Graph traversal** — follow `:RISK`, `:STRATEGY`, `:COMPETES_WITH`, `:MENTIONS` edges from matched Chunk nodes
4. **RRF fusion** — Reciprocal Rank Fusion merges dense, BM25, and graph results into a final ranked list

Controlled by `multi_stage_recall_k` (default: 100 per source).

### 7. `crag_evaluate` (COMPLEX path only)
Scores the top retrieved chunks against the query using the cross-encoder. Classifies:
- **CORRECT** (confidence ≥ 0.6) → proceed to generation
- **AMBIGUOUS** (0.4 ≤ confidence < 0.6) → rewrite query and retry (up to `max_rewrite_loops`, default: 2)
- **INCORRECT** (confidence < 0.4) → web fallback only if no local data; otherwise generate and surface gaps

### 8. `generate_analysis`
Builds a rich context string:
- Analyst question (framed prominently)
- Sentiment block (bullish/bearish/neutral %)
- Graph facts (top-25 relationship facts)
- Graph community summary
- Top retrieved chunks (up to 7, with chunk_id, score, source, temporal band)
- Temporal 2B tagging: RECENT vs. HISTORICAL chunks are separated so the LLM can analyse the *delta*

Sends to `llm.generate()` (model: `deepseek-reasoner` via DeepSeek API, `temperature=0.2`, single API call per run).

### 9. `format_json_output`
- Merges LLM output, sentiment block, company node, and web fallback results
- Priority for `company_overview`: Neo4j Company node > LLM-generated > graph_facts extraction
- Runs citation grounding check (`validate_citations()`)
- Strips ungrounded inline `[neo4j::...]` citations from prose (backstop sanitiser)
- Normalises `key_risks` (nulls out invented source IDs, coerces `mitigation_observed: "null"`)
- Sanitises `competitive_moat.sources` (bare string arrays of hallucinated IDs)
- Falls back `qualitative_summary` from `management_guidance` when LLM left it null

### 10. Output
Returns a structured JSON dict (see Output Schema below).

---

## Data Sources

| Source | Content | Storage |
|--------|---------|---------|
| `sentiment_trends` | Bullish / bearish / neutral % per ticker + trend direction | PostgreSQL |
| `:Company` nodes | 85+ financial properties (market cap, P/E, sector, margins, …) | Neo4j |
| `:Chunk` nodes (earnings_call) | Earnings call transcript chunks | Neo4j vector index |
| `:Chunk` nodes (broker_report) | Broker research report chunks | Neo4j vector index |
| `:Chunk` nodes (description, highlights, …) | Company profile sections | Neo4j vector index |
| `text_chunks` | pgvector embeddings (same documents, alternative index) | PostgreSQL |
| `:RISK`, `:STRATEGY`, `:COMPETES_WITH` edges | Relationship graph for graph traversal | Neo4j |

### Current Chunk Coverage (Neo4j)

| Ticker | earnings_call | broker_report | other | Total |
|--------|---------------|---------------|-------|-------|
| AAPL   | 131           | 400           | 13    | 544   |
| TSLA   | 130           | 158           | 13    | 301   |
| NVDA   | 137           | 96            | 12    | 245   |
| MSFT   | 138           | 73            | 13    | 224   |
| GOOGL  | 142           | 361           | 12    | 515   |

**Total: 1,829 chunks across 5 tickers**

---

## LLM & Models

| Purpose | Model | Backend | Notes |
|---------|-------|---------|-------|
| Primary generation | `deepseek-reasoner` | DeepSeek API (cloud) | `temperature=0.2`, one call per run; built-in chain-of-thought via `reasoning_content` |
| Query classification | rule-based only | n/a | No LLM call — keyword pattern matching in `tools.py` |
| Query rewrite | `deepseek-reasoner` | DeepSeek API (cloud) | Only fires on AMBIGUOUS CRAG path (rare) |
| Bi-encoder (embedding) | `nomic-embed-text:v1.5` | Ollama (local) | Neo4j + pgvector, 768-dim |
| Cross-encoder (rerank) | `ms-marco-MiniLM-L-6-v2` | HuggingFace (local) | CPU inference |

> **Ollama is kept for embeddings only.** All LLM generation uses the DeepSeek cloud API. No local GPU is required for generation.

---

## Chunk ID Formats

| Document Type | Format | Example |
|---------------|--------|---------|
| Neo4j Earnings Call | `neo4j::TICKER::earnings_call::<title>::<n>` | `neo4j::AAPL::earnings_call::Apple Inc Earnings Call 2026129::5` |
| Neo4j Broker Report | `neo4j::TICKER::broker_report::<institution>::<n>` | `neo4j::AAPL::broker_report::JP Morgan::2` |
| Neo4j Company Profile | `neo4j::TICKER::<section>::<n>` | `neo4j::AAPL::highlights::0` |
| pgvector chunk | `pgvec::TICKER::<uuid>` | `pgvec::NVDA::a3f1...` |

---

## Output Schema

```json
{
  "agent": "business_analyst",
  "ticker": "AAPL",
  "query_date": "2026-03-09",
  "company_overview": {
    "name": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "market_cap": 3.4e12,
    "pe_ratio": 29.5,
    "profit_margin": 0.253
  },
  "sentiment": {
    "bullish_pct": 90.88,
    "bearish_pct": 0.0,
    "neutral_pct": 9.12,
    "trend": "bullish",
    "source": "postgresql:sentiment_trends",
    "sentiment_interpretation": "..."
  },
  "competitive_moat": {
    "rating": "wide",
    "narrative": "...",
    "key_strengths": ["...", "..."],
    "threats": ["...", "..."],
    "trajectory": "stable",
    "sources": ["neo4j::AAPL::earnings_call::...::3"]
  },
  "qualitative_analysis": {
    "narrative": "...",
    "sentiment_signal": "...",
    "strategic_implication": "...",
    "data_quality_note": null
  },
  "management_guidance": {
    "forward_outlook_summary": "...",
    "earnings_call_highlights": ["...", "..."],
    "near_term_catalysts": ["...", "..."]
  },
  "sentiment_verdict": {
    "signal": "CONSTRUCTIVE",
    "rationale": "..."
  },
  "key_risks": [
    {
      "risk": "...",
      "severity": "HIGH",
      "mitigation_observed": "...",
      "source": "neo4j::AAPL::broker_report::JP Morgan::2"
    }
  ],
  "missing_context": [
    { "gap": "...", "severity": "MEDIUM" }
  ],
  "crag_status": "CORRECT",
  "confidence": 0.85,
  "fallback_triggered": false,
  "qualitative_summary": "One-to-two sentence executive summary.",
  "data_coverage_warning": null
}
```

**`crag_status`** values: `CORRECT` | `AMBIGUOUS` | `INCORRECT`

**`confidence`**: 0.0–1.0 — 1.0 for SIMPLE/NUMERICAL (bypass CRAG), real cross-encoder score for COMPLEX.

**`data_coverage_warning`**: `null` when data is healthy; a human-readable string when chunk coverage is thin or sentiment data is stale/missing. See [TROUBLESHOOTING.md §9](../../TROUBLESHOOTING.md#9-data-freshness-warnings-and-the-pre-check-layer) for remediation steps.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEEPSEEK_API_KEY` | **Required.** DeepSeek API key for LLM generation | — |
| `OLLAMA_BASE_URL` | Ollama API URL (embeddings only) | `http://localhost:11434` |
| `NEO4J_URI` | Neo4j bolt URI | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `password` |
| `BUSINESS_ANALYST_MODEL` | Primary LLM model name | `deepseek-reasoner` |
| `BUSINESS_ANALYST_LLM_PROVIDER` | LLM provider (`deepseek`) | `deepseek` |
| `BUSINESS_ANALYST_MAX_TOKENS` | Max output tokens for DeepSeek API | `8000` |
| `BUSINESS_ANALYST_REQUEST_TIMEOUT` | HTTP timeout for DeepSeek API (s) | `120` |
| `RAG_TOP_K` | Chunks returned to LLM | `15` |
| `RAG_SCORE_THRESHOLD` | Minimum relevance score | `0.6` |
| `BA_FAST_PATH_TOP_K` | Top-K for SIMPLE/NUMERICAL paths | `5` |
| `BA_MULTI_STAGE_RECALL_K` | Bi-encoder recall pool per source | `100` |
| `BA_MAX_REWRITE_LOOPS` | Max query rewrites on AMBIGUOUS | `3` |
| `CRAG_CORRECT_THRESHOLD` | Confidence ≥ this → CORRECT | `0.6` |
| `CRAG_AMBIGUOUS_THRESHOLD` | Confidence ≥ this → AMBIGUOUS | `0.4` |

---

## Running the Agent

### CLI — basic

```bash
# From host (venv activated)
python -m agents.business_analyst.agent --ticker AAPL

# Inside Docker
docker exec fyp-airflow-webserver \
    python -m agents.business_analyst.agent --ticker AAPL \
    --task "What is Apple's competitive moat?"
```

### CLI — with verbose pipeline trace (stderr)

```bash
docker exec fyp-airflow-webserver \
    python -m agents.business_analyst.agent --ticker AAPL \
    --ticker TSLA \
    --task "What are Tesla's key business risks?" \
    --verbose
```

Example stderr output:
```
[10:15:01] [01] >> METADATA PRECHECK  |  ticker=TSLA
[10:15:01] [02]   OK METADATA PRECHECK done  |  neo4j_chunks=301  pgvec_chunks=10  sentiment=yes  index=ONLINE  (0.08s)
[10:15:01] [03] >> PRECHECK DATA COVERAGE  |  ticker=TSLA
[10:15:01] [04]   OK PRECHECK DATA COVERAGE done  |  sentiment_fresh=True  neo4j=301  pgvec=10  warning=no
[10:15:01] [05] >> FETCH SENTIMENT  |  ticker=TSLA
[10:15:01] [06]   OK FETCH SENTIMENT done  |  bullish=62.5%  bearish=15.0%  neutral=22.5%  trend=bullish  (0.08s)
[10:15:01] [07] >> CLASSIFY QUERY  |  query='What are Tesla's key business risks?'
[10:15:01] [08]   OK CLASSIFY QUERY done  |  class=COMPLEX  method=rule-based  (0.00s)
[10:15:01] [09] >> COMPLEX RETRIEVAL (multi-stage)  |  ticker=TSLA
[10:15:11] [10]   OK COMPLEX RETRIEVAL done  |  chunks=15  graph_facts=0  top_score=0.738  (9.90s)
[10:15:11] [11] >> CRAG EVALUATE  |  ticker=TSLA  chunks_to_score=15
[10:15:11] [12]   OK CRAG EVALUATE done  |  status=CORRECT  confidence=0.738  (0.00s)
[10:15:11] [13] >> GENERATE ANALYSIS (LLM)  |  ticker=TSLA  chunks=15  graph_facts=0
[10:19:01] [14]   OK GENERATE ANALYSIS done  |  (230s)
[10:19:01] [15]   -- SEMANTIC CACHE CHECK  |  cache operates at retrieval layer — pass-through
[10:19:01] [16] >> FORMAT JSON OUTPUT  |  ticker=TSLA  crag=CORRECT  confidence=0.738  fallback=False
[10:19:01] [17]   OK CITATION GROUNDING  |  cited=0  grounded=0  ungrounded=0  rate=100.0%
[10:19:01] [18]   OK FORMAT JSON OUTPUT done
```

> **Note on generation latency:** `deepseek-reasoner` performs a full chain-of-thought reasoning pass before producing output. Expect 2–6 minutes per generation call depending on context size. The reasoning trace is logged at DEBUG level (`reasoning_content`) and is not included in the output JSON.

### CLI — with citation validation report

```bash
docker exec fyp-airflow-webserver \
    python -m agents.business_analyst.agent --ticker AAPL \
    --ticker NVDA \
    --task "What revenue guidance did NVIDIA give?" \
    --validate-citations
```

Appends to stderr after the run:
```
--- Citation Grounding Report ---
  Total cited IDs  : 5
  Grounded         : 5
  Ungrounded       : 0
  Grounding rate   : 100.0%
---------------------------------
```

### Programmatic

```python
from agents.business_analyst.agent import run, run_full_analysis

# Single question
result = run(task="What is Apple's competitive moat?", ticker="AAPL")

# Full institutional-grade dossier (all 7 pillars)
dossier = run_full_analysis(ticker="AAPL")
```

---

## Citation Grounding

Every factual claim in the output should cite a chunk_id from the retrieved set. The agent enforces this at two layers:

1. **Prompt (Option A)** — the system prompt instructs the LLM to only cite verbatim chunk_ids from the `RETRIEVED DOCUMENT CHUNKS` section.
2. **Post-processor (Option B)** — `_strip_ungrounded_inline_citations()` scans all prose fields and replaces any `[neo4j::...]` token that doesn't prefix-match a real retrieved ID.

`validate_citations(output, retrieval)` provides a machine-readable grounding report:

```python
from agents.business_analyst.agent import validate_citations

report = validate_citations(output=result, retrieval=retrieval_result)
# {
#   "total_cited": 8,
#   "grounded": 8,
#   "ungrounded": 0,
#   "grounding_rate_pct": 100.0,
#   "ungrounded_ids": []
# }
```

---

## Infrastructure

| Service | Container | Status | Notes |
|---------|-----------|--------|-------|
| PostgreSQL | `fyp-postgres` | healthy | `sentiment_trends`, `financial_statements`, `text_chunks` (pgvector) |
| Neo4j | `fyp-neo4j` | healthy | 51 Company nodes, 1,829 Chunk nodes, vector index ONLINE |
| Ollama | `fyp-ollama-1` | running | `nomic-embed-text:v1.5` for embeddings only |
| DeepSeek API | cloud | — | `deepseek-reasoner` for generation; requires `DEEPSEEK_API_KEY` |

---

## Testing

```bash
# Run all agent tests
pytest agents/business_analyst/tests/ -v

# Run specific test
pytest agents/business_analyst/tests/test_agent.py::test_full_pipeline -v

# Quick smoke-test three query classes from the CLI
docker exec fyp-airflow-webserver \
    python -m agents.business_analyst.agent --ticker AAPL \
    --task "What sector is Apple in?" --verbose          # → SIMPLE

docker exec fyp-airflow-webserver \
    python -m agents.business_analyst.agent --ticker AAPL \
    --task "What revenue guidance did NVIDIA give?" --verbose   # → NUMERICAL

docker exec fyp-airflow-webserver \
    python -m agents.business_analyst.agent --ticker AAPL \
    --task "What are Tesla's key business risks and how does sentiment reflect them?" \
    --verbose                                            # → COMPLEX
```
