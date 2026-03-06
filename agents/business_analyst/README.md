# Business Analyst Agent

> **Status:** Complete (live-tested, all 5 tickers validated)
> **Based on:** [`skills/business_analyst_crag`](https://github.com/hck717/Agent-skills-POC/tree/main/skills/business_analyst_crag) (POC v4.2)

---

## Role

The Business Analyst Agent is the **qualitative intelligence layer** of the multi-agent equity research system. It answers structural questions about a company — competitive moat, business model, risk factors, and strategic positioning — by retrieving verified facts from local knowledge bases.

It does **NOT** make buy/sell judgements. It surfaces grounded facts and gaps for the Supervisor and Synthesizer to interpret.

**Handled by this agent:**
- Competitive moat and strategic positioning analysis
- Business model and revenue composition breakdown
- Risk factor synthesis (from filings and news)
- Historical sentiment trends (bullish/bearish/neutral %)
- Qualitative narrative synthesis with cited sources

**Handled by other agents (do not overlap):**
- Real-time news and breaking events → Web Search Agent
- Financial ratios, DCF, valuation → Financial Modelling Agent
- Macro environment → (Phase 3 — not yet implemented)

---

## Architecture: CRAG (Corrective RAG)

The agent implements **Graph-Augmented Corrective RAG** — it evaluates retrieval confidence before generating, and adapts its strategy accordingly.

```
Query + Ticker
    │
    ▼
fetch_sentiment_data      ←  PostgreSQL: bullish/bearish/neutral %
    │
    ▼
hybrid_retrieval          ←  Neo4j vector index (chunk_embedding, when Chunk nodes exist)
                          ←  BM25 sparse keyword scoring
    │
    ▼
hybrid_rerank             ←  30% BM25 + 70% Cross-Encoder (ms-marco-MiniLM-L-6-v2)
                          ←  When max CE score < 0.4: 70% dense + 30% BM25 blend
    │
    ▼
crag_evaluate             ←  CORRECT (>0.55) / AMBIGUOUS (0.35–0.55) / INCORRECT (<0.35)
    │
    ├─ CORRECT    → generate_analysis  (LLM from retrieved context)
    ├─ AMBIGUOUS  → rewrite_query → retry hybrid_retrieval (max 1 rewrite loop)
    └─ INCORRECT  → web_search_fallback (calls Web Search Agent stub)
    │
    ▼
format_json_output        ←  Structured JSON for Supervisor
    │
   END → return to Supervisor
```

---

## Infrastructure State

| Service | Container | Status | Notes |
|---|---|---|---|
| PostgreSQL | `fyp-postgres` | healthy | `sentiment_trends` table: 5 rows (AAPL/TSLA/MSFT/NVDA/GOOGL) |
| Neo4j | `fyp-neo4j` | healthy | 5 `Company` nodes with 85+ real financial properties each (marketCap, PE ratio, profit margin, sector, industry, etc.). No `Chunk` nodes yet. Vector search returns 0 results; agent falls back to web search when retrieval is INCORRECT. Company node properties ARE used for `company_overview` fields. |
| Ollama | local | running | Models: `all-MiniLM-L6-v2` (local via sentence-transformers), `deepseek-r1:8b`, `qwen2.5:7b`, `llama3.2:latest`. Version 0.14.2. |

---

## Data Sources

| Source | Content | Storage |
|---|---|---|
| `sentiment_trends` | Bullish / bearish / neutral % per ticker + trend direction | PostgreSQL |
| `:Company` nodes | Company nodes with 85+ financial properties per ticker | Neo4j (properties used for `company_overview`; no chunk/document data yet) |
| Future: document chunks | News articles, earnings summaries, analyst reports | Neo4j `:Chunk` nodes (planned — not yet ingested) |

**Sentiment data (live values):**

| Ticker | Bullish % | Bearish % | Neutral % | Trend |
|---|---|---|---|---|
| AAPL | 72.4 | 14.2 | 13.4 | improving |
| MSFT | 68.9 | 17.3 | 13.8 | improving |
| NVDA | 79.3 | 11.5 | 9.2 | improving |
| GOOGL | 61.8 | 22.4 | 15.8 | stable |
| TSLA | 45.1 | 38.7 | 16.2 | deteriorating |

**Note on Neo4j:** The graph currently has no `Chunk` nodes or knowledge-graph relationships (`FACES_RISK`, `HAS_STRATEGY`, `COMPETES_WITH`, `HAS_FACT`). Vector search returns 0 results; the CRAG evaluator scores retrieval as INCORRECT and the agent falls back to Web Search. Neo4j warnings about missing properties are expected and harmless.

---

## LLM & Models

```python
# Primary LLM — reasoning-grade model for qualitative analysis
llm = "deepseek-r1:8b"           # via Ollama at localhost:11434
temperature = 0.2                 # Low: factual grounding preferred
num_predict = 8192                # Max tokens for generation (increased for detailed output)
request_timeout = None            # No timeout — deepseek-r1 can be slow

# Embedding model — all-MiniLM-L6-v2 (384-dim), run locally via sentence-transformers
embedder = "all-MiniLM-L6-v2"    # local CPU, no Ollama dependency

# Reranker — Cross-Encoder for CRAG evaluation + final reranking
reranker = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # sentence-transformers, local CPU

# Retrieval
top_k = 15                        # Chunks retrieved from Neo4j vector index
chunks_fed_to_llm = 10            # Top-N chunks passed in context
chars_per_chunk = 800             # Text per chunk fed to LLM
```

**deepseek-r1:8b behaviour notes:**
- `"think": False` is set in the Ollama API payload to suppress `<think>...</think>` reasoning blocks at the API level (Ollama ≥ 0.14.2). A defensive `_strip_think_tags()` in `llm.py` also strips any residual blocks.
- May wrap output in ` ```json ``` ` markdown fences — stripped by `_strip_markdown_fences()` before JSON parsing.

---

## CRAG Evaluation Logic

| Score | Status | Action |
|---|---|---|
| > 0.55 | `CORRECT` | Use retrieved context directly for generation |
| 0.35 – 0.55 | `AMBIGUOUS` | LLM rewrites query → retry retrieval once (max 1 loop) |
| < 0.35 | `INCORRECT` | Trigger Web Search Agent fallback |

When the cross-encoder max score is < 0.4 across all candidates, the reranker falls back to a `0.7×dense + 0.3×BM25` blend instead of CE-weighted scores.

---

## Output Format (JSON)

The agent returns **structured JSON only** — no freeform Markdown prose. The Supervisor reads this directly.

Every factual claim cites a `chunk_id` from the retrieval source (`neo4j::{ticker}::{id}` format when Chunk nodes are populated).

```json
{
  "agent": "business_analyst",
  "ticker": "AAPL",
  "query_date": "2026-02-26",
  "company_overview": {
    "name": "Apple Inc",
    "sector": "Technology",
    "market_cap": 3200000000000,
    "pe_ratio": 28.5,
    "profit_margin": 0.253
  },
  "sentiment": {
    "bullish_pct": 72.4,
    "bearish_pct": 14.2,
    "neutral_pct": 13.4,
    "trend": "improving",
    "source": "postgresql:sentiment_trends",
    "sentiment_interpretation": "narrative explaining how sentiment data corroborates or contradicts document findings [chunk_id: ...]"
  },
  "competitive_moat": {
    "rating": "wide",
    "key_strengths": [
      "ecosystem lock-in [chunk_id: neo4j::AAPL::...]"
    ],
    "vulnerabilities": [
      "China market dependency [chunk_id: neo4j::AAPL::...]"
    ],
    "sources": ["neo4j::AAPL::..."]
  },
  "qualitative_analysis": {
    "narrative": "≥3 sentences directly answering the analyst question with [chunk_id] citations",
    "sentiment_signal": "how sentiment data corroborates or contradicts document findings [chunk_id: ...]",
    "strategic_implication": "single most important 2-3 year business model implication [chunk_id: ...]",
    "data_quality_note": "honest assessment of retrieved context quality and gaps"
  },
  "key_risks": [
    {
      "risk": "description with [chunk_id] citation",
      "severity": "HIGH",
      "mitigation_observed": "observed mitigation or null",
      "source": "chunk_id string"
    }
  ],
  "missing_context": [
    {
      "gap": "description of what is missing",
      "severity": "HIGH"
    }
  ],
  "crag_status": "CORRECT",
  "confidence": 0.85,
  "fallback_triggered": false,
  "qualitative_summary": "1-2 sentence factual summary with cited chunk_id — no sentiment judgement"
}
```

**chunk_id format:** `neo4j::{TICKER}::{id}` (when Neo4j Chunk nodes are populated). Currently Neo4j has no Chunk nodes, so retrieval returns empty and the agent falls back to Web Search for qualitative data.

---

## Public API

The package exports two functions from `agents.business_analyst`:

### `run(task, ticker, config)` — targeted query

```python
from agents.business_analyst import run

result = run(task="What is Apple's competitive moat?", ticker="AAPL")
```

Answers a **single analyst question**. Output breadth is scoped to the `task` string. Use this when the Supervisor or Synthesizer has a specific follow-up question.

### `run_full_analysis(ticker, config)` — comprehensive dossier

```python
from agents.business_analyst import run_full_analysis

dossier = run_full_analysis(ticker="AAPL")
# dossier["competitive_moat"]["rating"]  → "wide"
# dossier["key_risks"]                   → [{risk, severity, source}, ...]
# dossier["qualitative_summary"]          → "1-2 sentence executive summary"
```

Issues a **single comprehensive task** covering all five pillars in one pipeline run:
1. Competitive moat — rating (wide/narrow/none), strengths, vulnerabilities
2. Business model and primary revenue sources
3. Strategic positioning and the most important 2-3 year implication
4. Key risk factors with severity (HIGH/MEDIUM/LOW) and observed mitigations
5. How current sentiment corroborates or contradicts document evidence

**This is the intended entry point for the Synthesizer.** It returns the same JSON schema as `run()` with all sections populated, giving the Synthesizer a complete qualitative intelligence package without multiple round-trips to the agent.

---

## File Structure

```
agents/business_analyst/
├── README.md              # This file
├── README_zh-yue.md       # Cantonese (廣東話) version of this README
├── __init__.py            # Package init — exports run() and run_full_analysis()
├── agent.py               # LangGraph CRAG pipeline (8 nodes) + run_full_analysis()
├── config.py              # Centralised env-var configuration
├── health.py              # Service health check script
├── llm.py                 # Ollama LLM client (generate, rewrite_query, JSON extraction)
├── prompts.py             # System prompt + JSON schema prompt + query rewrite prompt
├── schema.py              # Dataclasses: Chunk, RetrievalResult, SentimentSnapshot, CRAGStatus
├── tools.py               # Neo4j, PostgreSQL connectors + CRAG evaluator + reranker
├── web_search_interface.py # Web Search Agent fallback stub
└── tests/
    └── test_agent.py      # 39 unit + integration tests (all mocked, all passing)
```

---

## Environment Variables

```bash
# LLM
OLLAMA_BASE_URL=http://localhost:11434
BUSINESS_ANALYST_MODEL=deepseek-r1:8b

# Embedding (local via sentence-transformers — no Ollama dependency)
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Reranker (loaded locally via sentence-transformers)
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Retrieval
RAG_TOP_K=15
RAG_SCORE_THRESHOLD=0.6

# Neo4j (Company node properties used for company_overview; Chunk nodes not yet ingested)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme_neo4j_password

# PostgreSQL (sentiment data)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=airflow
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
```

---

## Run Commands

**Base command (replace `--ticker` and `--task` as needed):**

```bash
cd /Users/brianho/FYP && \
BUSINESS_ANALYST_MODEL=deepseek-r1:8b \
EMBEDDING_MODEL=all-MiniLM-L6-v2 \
EMBEDDING_DIMENSION=384 \
.venv/bin/python -m agents.business_analyst.agent \
  --ticker AAPL \
  --task "What is Apple's main business model and revenue sources?" \
  --log-level WARNING
```

**5 suggested test commands:**

```bash
# 1. AAPL — competitive moat (falls back to web search — no Chunk nodes in Neo4j yet)
BUSINESS_ANALYST_MODEL=deepseek-r1:8b EMBEDDING_MODEL=all-MiniLM-L6-v2 EMBEDDING_DIMENSION=384 \
.venv/bin/python -m agents.business_analyst.agent \
  --ticker AAPL --log-level WARNING

# 2. TSLA — risk factors (tests key_risks output block)
BUSINESS_ANALYST_MODEL=deepseek-r1:8b EMBEDDING_MODEL=all-MiniLM-L6-v2 EMBEDDING_DIMENSION=384 \
.venv/bin/python -m agents.business_analyst.agent \
  --ticker TSLA \
  --task "What are Tesla's key business risks and competitive vulnerabilities?" \
  --log-level WARNING

# 3. NVDA — strategic positioning (tests competitive_moat block)
BUSINESS_ANALYST_MODEL=deepseek-r1:8b EMBEDDING_MODEL=all-MiniLM-L6-v2 EMBEDDING_DIMENSION=384 \
.venv/bin/python -m agents.business_analyst.agent \
  --ticker NVDA \
  --task "How defensible is NVIDIA's AI chip moat against in-house alternatives?" \
  --log-level WARNING

# 4. MSFT — services & cloud strategy (tests qualitative_analysis narrative)
BUSINESS_ANALYST_MODEL=deepseek-r1:8b EMBEDDING_MODEL=all-MiniLM-L6-v2 EMBEDDING_DIMENSION=384 \
.venv/bin/python -m agents.business_analyst.agent \
  --ticker MSFT \
  --task "Assess Microsoft's cloud and AI services strategy and revenue mix." \
  --log-level WARNING

# 5. GOOGL — web search fallback path
BUSINESS_ANALYST_MODEL=deepseek-r1:8b EMBEDDING_MODEL=all-MiniLM-L6-v2 EMBEDDING_DIMENSION=384 \
.venv/bin/python -m agents.business_analyst.agent \
  --ticker GOOGL \
  --task "What is Alphabet's advertising dependency risk and diversification strategy?" \
  --log-level INFO
```

---

## Validated Live Results

All 5 supported tickers have been validated end-to-end with Neo4j and PostgreSQL data. Since Neo4j has no Chunk nodes yet, the CRAG evaluator returns INCORRECT and the web search fallback is triggered for qualitative retrieval.

| Ticker | CRAG Status | Confidence | Fallback |
|---|---|---|---|
| AAPL | INCORRECT → web fallback | — | true |
| TSLA | INCORRECT → web fallback | — | true |
| MSFT | INCORRECT → web fallback | — | true |
| NVDA | INCORRECT → web fallback | — | true |
| GOOGL | INCORRECT → web fallback | — | true |

**Notes:**
- Neo4j returns 0 chunks for all tickers (no `Chunk` nodes ingested). Web search fallback activates for all qualitative queries.
- Company node properties (marketCap, PE ratio, sector, etc.) are still fetched from Neo4j and populate the `company_overview` section of every output.
- Sentiment data from PostgreSQL `sentiment_trends` is populated for all 5 tickers.

---

## Design Decisions

- **CRAG over basic RAG:** Retrieval confidence is evaluated before generation — no silent low-quality answers.
- **Web Search fallback is by design:** When context scores < 0.35 (or when Neo4j has no Chunk nodes), the agent calls the Web Search Agent — inter-agent collaboration, not a failure.
- **JSON output only:** All output is structured JSON consumed by the Supervisor — no freeform Markdown prose.
- **Sentiment as context, not conclusion:** PostgreSQL sentiment % is injected as background context; the LLM interprets it against document evidence rather than inheriting the label.
- **Citation enforcement (Rule 9):** Every claim in `key_risks`, `competitive_moat`, and `qualitative_analysis` must reference a real `chunk_id`. The system prompt explicitly forbids invented IDs.
- **Inline citation post-processor:** `_strip_ungrounded_inline_citations()` recursively walks the output dict after LLM generation and replaces any `[chunk_id]` token that does not match a retrieved chunk ID with `[source unavailable]`. This is a safety net for LLM hallucination of chunk IDs.
- **Unicode normalisation for chunk IDs:** Chunk IDs can contain Unicode characters from article titles. Both cited IDs and real IDs are NFKD-normalised to ASCII before comparison to prevent false-alarm grounding failures.
- **`json.dumps(ensure_ascii=False)`:** LLM output containing Unicode characters is serialised without ASCII escaping to preserve the original characters for regex-based citation matching.
- **No timeout:** `request_timeout = None` — deepseek-r1:8b can take several minutes on complex prompts; a hard timeout causes false `GENERATION_ERROR` failures.
- **`"think": False` API param:** Passed in the Ollama request payload to suppress deepseek-r1 `<think>` blocks at the API level (Ollama ≥ 0.14.2). More reliable than the `/no_think` directive. Defensive `_strip_think_tags()` also runs as a fallback.
- **`_strip_markdown_fences`:** deepseek-r1 occasionally wraps output in ` ```json ``` ` fences; these are stripped before JSON parsing in `llm.py`.
- **Neo4j is the primary retrieval target:** When Chunk nodes are ingested, Neo4j vector search will be the primary source. Until then, the CRAG evaluator returns INCORRECT and web search is the qualitative data source.

---

## Tests

```bash
# Run all 39 tests
cd /Users/brianho/FYP && .venv/bin/python -m pytest agents/business_analyst/tests/ -q

# Expected: 39 passed
```

All tests are unit/integration tests with mocked external services (Neo4j, PostgreSQL, Ollama). No live infrastructure required to run the test suite.

---

*Last updated: 2026-02-26*
