# Business Analyst Agent

> **Architecture:** Optimized Adaptive Agentic Graph RAG (v2)
> **Status:** Live-tested, all 5 tickers validated

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

### 3. `fetch_sentiment_data`
- **Sentiment** — `SELECT bullish_pct, bearish_pct, neutral_pct, trend FROM sentiment_trends WHERE ticker=?` (PostgreSQL)
- **Company node** — `MATCH (c:Company {ticker: $t}) RETURN c` (Neo4j), used for the `company_overview` section (market cap, P/E, sector, etc.)
- **Graph community summary** — relationship-count centrality across `:RISK`, `:STRATEGY`, `:COMPETES_WITH` edges (2A: Graph RAG)

### 4. `classify_query`
Sends the query to a lightweight LLM call with `QUERY_CLASSIFICATION_PROMPT`. Returns one of:
- `SIMPLE` — factual single-answer questions (e.g. "What sector is Apple in?")
- `NUMERICAL` — metric/ratio/guidance questions (e.g. "What revenue guidance did NVIDIA give?")
- `COMPLEX` — multi-faceted qualitative questions (e.g. "What are Tesla's key business risks?")

Falls back to `COMPLEX` on any error.

### 5. Retrieval paths

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

### 6. `crag_evaluate` (COMPLEX path only)
Scores the top retrieved chunks against the query using the cross-encoder. Classifies:
- **CORRECT** (confidence ≥ 0.6) → proceed to generation
- **AMBIGUOUS** (0.4 ≤ confidence < 0.6) → rewrite query and retry (up to `max_rewrite_loops`, default: 2)
- **INCORRECT** (confidence < 0.4) → web fallback only if no local data; otherwise generate and surface gaps

### 7. `generate_analysis`
Builds a rich context string:
- Analyst question (framed prominently)
- Sentiment block (bullish/bearish/neutral %)
- Graph facts (top-25 relationship facts)
- Graph community summary
- Top retrieved chunks (up to 7, with chunk_id, score, source, temporal band)
- Temporal 2B tagging: RECENT vs. HISTORICAL chunks are separated so the LLM can analyse the *delta*

Sends to `llm.generate()` (model: `deepseek-r1:8b` by default, `temperature=0.2`).

### 8. `format_json_output`
- Merges LLM output, sentiment block, company node, and web fallback results
- Priority for `company_overview`: Neo4j Company node > LLM-generated > graph_facts extraction
- Runs citation grounding check (`validate_citations()`)
- Strips ungrounded inline `[neo4j::...]` citations from prose (backstop sanitiser)
- Normalises `key_risks` (nulls out invented source IDs, coerces `mitigation_observed: "null"`)
- Sanitises `competitive_moat.sources` (bare string arrays of hallucinated IDs)
- Falls back `qualitative_summary` from `management_guidance` when LLM left it null

### 9. Output
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

| Purpose | Model | Notes |
|---------|-------|-------|
| Primary generation | `deepseek-r1:8b` | via Ollama, `temperature=0.2`, no timeout |
| Query classification | `llama3.2:latest` (configurable) | Lightweight — single-word output |
| Query rewrite | same as primary | Used on AMBIGUOUS path |
| Bi-encoder (embedding) | `nomic-embed-text` (via Ollama) | Neo4j + pgvector |
| Cross-encoder (rerank) | `ms-marco-MiniLM-L-6-v2` | HuggingFace, local |

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
  "qualitative_summary": "One-to-two sentence executive summary."
}
```

**`crag_status`** values: `CORRECT` | `AMBIGUOUS` | `INCORRECT`

**`confidence`**: 0.0–1.0 — 1.0 for SIMPLE/NUMERICAL (bypass CRAG), real cross-encoder score for COMPLEX.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama API URL | `http://localhost:11434` |
| `NEO4J_URI` | Neo4j bolt URI | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `password` |
| `POSTGRES_DSN` | PostgreSQL connection string | — |
| `BUSINESS_ANALYST_MODEL` | Primary LLM model | `deepseek-r1:8b` |
| `BA_QUERY_CLASSIFIER_MODEL` | Query classification model | `llama3.2:latest` |
| `BUSINESS_ANALYST_REQUEST_TIMEOUT` | Ollama request timeout (s) | `None` (no timeout) |
| `RAG_TOP_K` | Chunks returned to LLM | `15` |
| `RAG_SCORE_THRESHOLD` | Minimum relevance score | `0.6` |
| `BA_FAST_PATH_TOP_K` | Top-K for SIMPLE/NUMERICAL paths | `5` |
| `BA_MULTI_STAGE_RECALL_K` | Bi-encoder recall pool per source | `100` |
| `BA_MAX_REWRITE_LOOPS` | Max query rewrites on AMBIGUOUS | `2` |
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
python -m agents.business_analyst.agent \
    --ticker TSLA \
    --task "What are Tesla's key business risks?" \
    --verbose
```

Example stderr output:
```
[10:15:01] [01] >> METADATA PRECHECK  |  ticker=TSLA
[10:15:01] [02]   OK METADATA PRECHECK done  |  neo4j_chunks=301  pgvec_chunks=0  sentiment=yes  index=ONLINE  (0.12s)
[10:15:01] [03] >> FETCH SENTIMENT  |  ticker=TSLA
[10:15:01] [04]   OK FETCH SENTIMENT done  |  bullish=62.5%  bearish=15.0%  neutral=22.5%  trend=bullish  (0.08s)
[10:15:01] [05] >> CLASSIFY QUERY  |  query='What are Tesla's key business risks?'
[10:15:03] [06]   OK CLASSIFY QUERY done  |  class=COMPLEX  (1.84s)
[10:15:03] [07] >> COMPLEX RETRIEVAL (multi-stage)  |  ticker=TSLA
[10:15:07] [08]   OK COMPLEX RETRIEVAL done  |  chunks=12  graph_facts=5  top_score=0.781  (3.92s)
[10:15:07] [09] >> CRAG EVALUATE  |  ticker=TSLA  chunks_to_score=12
[10:15:07] [10]   OK CRAG EVALUATE done  |  status=CORRECT  confidence=0.720  (0.41s)
[10:15:07] [11] >> GENERATE ANALYSIS (LLM)  |  ticker=TSLA  chunks=12  graph_facts=5
[10:15:45] [12]   OK GENERATE ANALYSIS done  |  (37.91s)
[10:15:45] [13] >> SEMANTIC CACHE CHECK  |  cache operates at retrieval layer — pass-through
[10:15:45] [14] >> FORMAT JSON OUTPUT  |  ticker=TSLA  crag=CORRECT  confidence=0.720  fallback=False
[10:15:45] [15]   OK CITATION GROUNDING  |  cited=8  grounded=8  ungrounded=0  rate=100.0%
[10:15:45] [16]   OK FORMAT JSON OUTPUT done
```

### CLI — with citation validation report

```bash
python -m agents.business_analyst.agent \
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
| Ollama | local | running | `deepseek-r1:8b`, `llama3.2:latest`, `nomic-embed-text` |

---

## Testing

```bash
# Run all agent tests
pytest agents/business_analyst/tests/ -v

# Run specific test
pytest agents/business_analyst/tests/test_agent.py::test_full_pipeline -v

# Quick smoke-test three query classes from the CLI
python -m agents.business_analyst.agent --ticker AAPL \
    --task "What sector is Apple in?" --verbose          # → SIMPLE

python -m agents.business_analyst.agent --ticker NVDA \
    --task "What revenue guidance did NVIDIA give?" --verbose   # → NUMERICAL

python -m agents.business_analyst.agent --ticker TSLA \
    --task "What are Tesla's key business risks and how does sentiment reflect them?" \
    --verbose                                            # → COMPLEX
```
