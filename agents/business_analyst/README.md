# Business Analyst Agent

> **Status:** 🔧 In Development — architecture finalised, implementation pending
> **Based on:** [`skills/business_analyst_crag`](https://github.com/hck717/Agent-skills-POC/tree/main/skills/business_analyst_crag) (POC v4.2, tested)

---

## Role

The Business Analyst Agent is the **qualitative intelligence layer** of the multi-agent equity research system. It answers structural questions about a company — competitive moat, business model, risk factors, and management quality — by retrieving verified facts from local knowledge bases.

It does **NOT** make buy/sell judgements. It surfaces grounded facts and gaps for the Supervisor and Synthesizer to interpret.

**Handled by this agent:**
- Competitive moat and strategic positioning analysis
- Business model and revenue composition breakdown
- Risk factor synthesis (from 10-K filings and news)
- Management quality assessment (from filings and news)
- Historical sentiment trends (bullish/bearish/neutral %)

**Handled by other agents (do not overlap):**
- Real-time news and breaking events → Web Search Agent
- Financial ratios, DCF, valuation → Financial Modelling Agent
- Macro environment → Macro Metrics Agent

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
hybrid_retrieval          ←  Neo4j vector index (384-dim)
                          ←  Neo4j Cypher graph traversal
                          ←  BM25 sparse keyword scoring
    │
    ▼
hybrid_rerank             ←  30% BM25 + 70% Cross-Encoder (ms-marco-MiniLM-L-6-v2)
    │
    ▼
crag_evaluate             ←  CORRECT (>0.7) / AMBIGUOUS (0.5-0.7) / INCORRECT (<0.5)
    │
    ├─ CORRECT    → generate_analysis  (LLM from graph context)
    ├─ AMBIGUOUS  → rewrite_query → retry hybrid_retrieval
    └─ INCORRECT  → web_search_fallback (calls Web Search Agent)
    │
    ▼
format_json_output        ←  Structured JSON for Supervisor
    │
   END → return to Supervisor
```

---

## Data Sources

| Source | Content | Storage |
|---|---|---|
| `financial_documents` | 10-K filings, earnings reports, news embeddings | Qdrant (news) + Neo4j vector index (filings) |
| `sentiment_trends` | Bullish / bearish / neutral % per ticker | PostgreSQL |
| `:Company` nodes | 46 company properties (PE, EBITDA, sector, description) | Neo4j |
| `:Chunk` nodes | Proposition-level chunked 10-K sections | Neo4j |
| `company_profile.json` | Full EODHD company description | Local file |

---

## LLM & Models

```python
# Primary LLM — reasoning-grade model for qualitative analysis
llm = "deepseek-v3.2-exp"        # via Ollama at localhost:11434
temperature = 0.2                 # Low: factual grounding preferred
num_predict = 1500

# Embedding model — must match Neo4j vector index dimensions
embedder = "all-MiniLM-L6-v2"    # SentenceTransformer, 384-dim, CPU-safe on M-series Mac

# Reranker — Cross-Encoder for CRAG evaluation + final reranking
reranker = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

> **Why DeepSeek-V3.2-Exp:** Stronger reasoning for multi-step qualitative analysis (moat, risks, management) vs. qwen2.5:7b. Runs locally via Ollama.

---

## CRAG Evaluation Logic

| Score | Status | Action |
|---|---|---|
| > 0.7 | `CORRECT` | Use retrieved context directly for generation |
| 0.5 – 0.7 | `AMBIGUOUS` | LLM rewrites query → retry retrieval once |
| < 0.5 | `INCORRECT` | Trigger Web Search Agent fallback |

---

## Output Format (JSON)

The agent returns a **structured JSON** — no freeform Markdown prose. The Supervisor reads this directly.

```json
{
  "agent": "business_analyst",
  "ticker": "AAPL",
  "query_date": "2026-02-24",
  "company_overview": {
    "name": "Apple Inc",
    "sector": "Technology",
    "market_cap": 3250000000000,
    "pe_ratio": 31.2,
    "profit_margin": 0.263
  },
  "sentiment": {
    "bullish_pct": 65,
    "bearish_pct": 20,
    "neutral_pct": 15,
    "trend": "improving",
    "source": "postgresql:sentiment_trends"
  },
  "competitive_moat": {
    "rating": "wide",
    "key_strengths": ["ecosystem lock-in", "brand premium", "services revenue growth"],
    "sources": ["chunk_id_001", "chunk_id_042"]
  },
  "key_risks": [
    {
      "risk": "China market revenue dependency (~18% of revenue)",
      "severity": "HIGH",
      "source": "chunk_id_018"
    }
  ],
  "missing_context": [
    {
      "gap": "No recent 10-K data for FY2026 — latest filing is FY2025",
      "severity": "MEDIUM"
    }
  ],
  "crag_status": "CORRECT",
  "confidence": 0.85,
  "fallback_triggered": false,
  "qualitative_summary": "1-2 sentence factual summary with no sentiment judgement"
}
```

---

## File Structure

```
agents/business_analyst/
├── README.md              # This file
├── agent.py               # LangGraph CRAG pipeline (to be built from POC v4.2)
├── tools.py               # Neo4j, Qdrant, PostgreSQL connectors
├── ingestion.py           # Proposition chunking + Neo4j ingestion (from POC)
├── semantic_chunker.py    # Embedding-based semantic chunker (from POC)
├── prompts.py             # System prompts for analysis + query rewrite
└── tests/
    └── test_agent.py
```

---

## Environment Variables

```bash
# LLM
OLLAMA_BASE_URL=http://localhost:11434
BUSINESS_ANALYST_MODEL=deepseek-v3.2-exp

# Embedding + Reranker (loaded via sentence-transformers, not Ollama)
EMBEDDING_MODEL=all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Qdrant (news embeddings)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=financial_documents
RAG_TOP_K=8
RAG_SCORE_THRESHOLD=0.6

# Neo4j (filings graph + vector index)
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

## Quick Test (once built)

```bash
source .venv/bin/activate

# Full CRAG pipeline test
python agents/business_analyst/agent.py --ticker AAPL

# Test Neo4j retrieval only
python agents/business_analyst/tools.py --ticker AAPL --query "Apple competitive moat"

# Run tests
python -m pytest agents/business_analyst/tests/ -v
```

---

## Design Decisions

- **CRAG over basic RAG:** Retrieval confidence is evaluated before generation — no silent low-quality answers
- **Web Search fallback is by design:** When graph context scores < 0.5, the agent calls the Web Search Agent automatically — this is inter-agent collaboration, not a failure
- **JSON output only:** All output is structured JSON consumed by the Supervisor — no freeform Markdown prose
- **Sentiment as context, not conclusion:** PostgreSQL sentiment % is injected as background context into the LLM prompt; the LLM does not inherit the sentiment label
- **Citation enforcement:** Every claim in `key_risks` and `competitive_moat` must reference a `chunk_id` from Neo4j — verified by the Critic Agent
- **DeepSeek-V3.2-Exp:** Chosen over qwen2.5:7b for stronger multi-step qualitative reasoning on complex moat/risk analysis tasks

---

## POC Reference

| POC File | FYP Target | Status |
|---|---|---|
| `skills/business_analyst_crag/agent.py` | `agents/business_analyst/agent.py` | 🔧 Migrate + add JSON output |
| `skills/business_analyst_crag/ingestion.py` | `agents/business_analyst/ingestion.py` | 🔧 Migrate as-is |
| `skills/business_analyst_crag/semantic_chunker.py` | `agents/business_analyst/semantic_chunker.py` | 🔧 Migrate as-is |
| PostgreSQL sentiment fetch | `agents/business_analyst/tools.py` | ❌ New — not in POC |
| JSON output wrapper | `agents/business_analyst/agent.py` | ❌ New — POC returns Markdown |

---

*Last updated: 2026-02-24 | Author: hck717*
