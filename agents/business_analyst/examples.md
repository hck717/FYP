# Business Analyst Agent — Examples

This agent is a **qualitative fact extractor**. It retrieves grounded insights from
local knowledge bases (Neo4j graph + Qdrant news + PostgreSQL sentiment).
It does NOT make buy/sell judgements. No BULLISH/BEARISH labels in output.
All interpretation is left to the Supervisor and Synthesizer.

---

## CRAG Status Quick Reference

| Score | Status | What happens |
|---|---|---|
| > 0.7 | `CORRECT` | Use retrieved graph context directly for generation |
| 0.5 – 0.7 | `AMBIGUOUS` | LLM rewrites query → retry retrieval once |
| < 0.5 | `INCORRECT` | Call Web Search Agent as fallback |

---

## Data Source Quick Reference

| Source | What it provides | Used when |
|---|---|---|
| Neo4j vector index | 10-K filing chunks (semantic search, 384-dim) | Always — primary retrieval |
| Neo4j Cypher traversal | Structured graph facts (risks, strategies, products) | Always — supplementary |
| BM25 (in-memory) | Keyword boosting over retrieved candidates | Always — part of hybrid ranking |
| PostgreSQL `sentiment_trends` | Bullish/bearish/neutral % per ticker | Always — injected as context |
| Qdrant `financial_documents` | Recent news embeddings | When Neo4j CRAG status = INCORRECT |
| Web Search Agent | Live news fallback | Only when CRAG status = INCORRECT |

---

## Example 1: Competitive Moat Analysis

**Input**
```json
{"task": "What is Apple's competitive moat and key strengths?", "ticker": "AAPL"}
```

**Retrieval angles (hybrid search covers all automatically)**
1. Neo4j vector: `Apple competitive advantage ecosystem lock-in`
2. Neo4j Cypher: `MATCH (c:Company {ticker:'AAPL'})-[:HAS_STRATEGY]->(s) RETURN s`
3. BM25 boost: keywords `moat`, `ecosystem`, `switching cost`, `brand`

**CRAG evaluation:** score = 0.86 → `CORRECT` → generate directly

**Expected output**
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
    "key_strengths": [
      "Ecosystem lock-in across iPhone, Mac, iPad, Watch, Services",
      "Brand premium enabling consistent ASP growth",
      "Services segment (App Store, iCloud, Apple TV+) expanding recurring revenue"
    ],
    "sources": ["chunk_id_023", "chunk_id_047"]
  },
  "key_risks": [],
  "missing_context": [],
  "crag_status": "CORRECT",
  "confidence": 0.86,
  "fallback_triggered": false,
  "qualitative_summary": "Apple maintains a wide competitive moat through deep ecosystem integration and a fast-growing services layer that reduces hardware revenue cyclicality."
}
```

**Notes**
- ✅ Every strength in `key_strengths` must cite a `chunk_id` from Neo4j
- ❌ Do NOT add sentiment judgement like "this is bullish for the stock"
- ❌ Do NOT invent metrics not found in the retrieved chunks

---

## Example 2: Risk Factor Analysis

**Input**
```json
{"task": "What are Tesla's key operational and financial risks?", "ticker": "TSLA"}
```

**Retrieval angles**
1. Neo4j vector: `Tesla production risk margin pressure competition EV`
2. Neo4j Cypher: `MATCH (c:Company {ticker:'TSLA'})-[:FACES_RISK]->(r) RETURN r`
3. BM25 boost: keywords `risk`, `margin`, `competition`, `recall`, `regulatory`

**CRAG evaluation:** score = 0.63 → `AMBIGUOUS` → rewrite query → retry

**Rewritten query:** `Tesla gross margin decline competition BYD 2025 operational risk factors`

**Retry CRAG evaluation:** score = 0.81 → `CORRECT` → generate

**Expected output**
```json
{
  "agent": "business_analyst",
  "ticker": "TSLA",
  "query_date": "2026-02-24",
  "competitive_moat": {
    "rating": "narrow",
    "key_strengths": ["Supercharger network scale", "FSD software optionality", "Gigafactory cost structure"],
    "sources": ["chunk_id_112"]
  },
  "key_risks": [
    {
      "risk": "Gross margin compression from price cuts to defend volume against BYD",
      "severity": "HIGH",
      "source": "chunk_id_098"
    },
    {
      "risk": "FSD regulatory approval uncertainty in EU and China markets",
      "severity": "MEDIUM",
      "source": "chunk_id_103"
    },
    {
      "risk": "Elon Musk key-person concentration risk",
      "severity": "MEDIUM",
      "source": "chunk_id_077"
    }
  ],
  "missing_context": [
    {
      "gap": "Q4 2025 10-K not yet ingested — risk section may have been updated",
      "severity": "MEDIUM"
    }
  ],
  "crag_status": "CORRECT",
  "confidence": 0.81,
  "fallback_triggered": false,
  "qualitative_summary": "Tesla faces significant margin pressure from Chinese EV competition and regulatory uncertainty around FSD, partially offset by its cost-efficient manufacturing base."
}
```

**Notes**
- ✅ AMBIGUOUS → rewrite → CORRECT is the normal path for vague queries
- ✅ `missing_context` flags stale data — Supervisor uses this to request Web Search Agent
- ❌ `severity` must be `HIGH / MEDIUM / LOW` only — no free text

---

## Example 3: Business Model / Revenue Composition

**Input**
```json
{"task": "Break down Microsoft's revenue composition and growth drivers", "ticker": "MSFT"}
```

**Retrieval angles**
1. Neo4j vector: `Microsoft revenue segments cloud Azure Office commercial`
2. Neo4j Cypher: `MATCH (c:Company {ticker:'MSFT'})-[:HAS_SEGMENT]->(s) RETURN s`
3. BM25 boost: keywords `revenue`, `Azure`, `cloud`, `Office 365`, `LinkedIn`, `growth`

**CRAG evaluation:** score = 0.91 → `CORRECT`

**Expected output**
```json
{
  "agent": "business_analyst",
  "ticker": "MSFT",
  "company_overview": {
    "name": "Microsoft Corporation",
    "sector": "Technology",
    "market_cap": 3100000000000,
    "pe_ratio": 34.1,
    "profit_margin": 0.358
  },
  "sentiment": {
    "bullish_pct": 72,
    "bearish_pct": 12,
    "neutral_pct": 16,
    "trend": "stable",
    "source": "postgresql:sentiment_trends"
  },
  "competitive_moat": {
    "rating": "wide",
    "key_strengths": [
      "Azure hyperscaler with enterprise cloud lock-in",
      "Office 365 recurring subscription base (~400M seats)",
      "GitHub + Copilot AI developer platform flywheel"
    ],
    "sources": ["chunk_id_201", "chunk_id_215", "chunk_id_233"]
  },
  "key_risks": [
    {
      "risk": "AWS and Google Cloud price competition compressing Azure margins",
      "severity": "MEDIUM",
      "source": "chunk_id_198"
    }
  ],
  "missing_context": [],
  "crag_status": "CORRECT",
  "confidence": 0.91,
  "fallback_triggered": false,
  "qualitative_summary": "Microsoft's three-segment model (Productivity, Cloud, Devices) is increasingly dominated by Azure and Office 365, creating durable recurring revenue with high switching costs."
}
```

---

## Example 4: Low-Confidence → Web Search Fallback

**Input**
```json
{"task": "What is Palantir's AI platform competitive positioning in 2026?", "ticker": "PLTR"}
```

**Retrieval angles**
1. Neo4j vector: `Palantir AIP platform enterprise AI competitive positioning`
2. Neo4j Cypher: `MATCH (c:Company {ticker:'PLTR'})-[:HAS_STRATEGY]->(s) RETURN s`
3. BM25 boost: keywords `AIP`, `government`, `commercial`, `ontology`

**CRAG evaluation:** score = 0.38 → `INCORRECT`
(Reason: PLTR not yet ingested into Neo4j graph — no Chunk nodes found)

**Action:** Call Web Search Agent with query `Palantir AIP competitive positioning enterprise AI 2026`

**Expected output**
```json
{
  "agent": "business_analyst",
  "ticker": "PLTR",
  "competitive_moat": null,
  "key_risks": [],
  "missing_context": [
    {
      "gap": "PLTR not yet ingested into Neo4j graph — no 10-K chunks available",
      "severity": "HIGH"
    }
  ],
  "crag_status": "INCORRECT",
  "confidence": 0.38,
  "fallback_triggered": true,
  "qualitative_summary": "INSUFFICIENT_DATA: Graph context unavailable for PLTR. Web Search Agent fallback triggered."
}
```

**Notes**
- ✅ `fallback_triggered: true` tells the Supervisor that output came from web, not graph
- ✅ `crag_status: "INCORRECT"` is not a failure — it's expected for un-ingested tickers
- ❌ Do NOT hallucinate Palantir facts from LLM training data — only use retrieved context

---

## Example 5: Multi-Company Supply Chain Query

**Input**
```json
{"task": "If TSMC production drops 30%, which companies in our coverage are most affected?", "ticker": null}
```

**Retrieval angles**
1. Neo4j vector: `TSMC supply chain dependency semiconductor manufacturing`
2. Neo4j Cypher multi-hop:
   ```cypher
   MATCH (tsmc:Company {ticker:'TSM'})<-[:SUPPLIED_BY]-(c:Company)
   RETURN c.ticker, c.name
   ```
3. BM25 boost: keywords `TSMC`, `foundry`, `supply`, `chip`, `outsourced manufacturing`

**CRAG evaluation:** score = 0.79 → `CORRECT`

**Expected output**
```json
{
  "agent": "business_analyst",
  "ticker": null,
  "competitive_moat": null,
  "key_risks": [
    {
      "risk": "NVDA: 100% fabless — all GPU production outsourced to TSMC N4/N3 nodes",
      "severity": "HIGH",
      "source": "chunk_id_334"
    },
    {
      "risk": "AAPL: ~90% of A-series and M-series chips manufactured at TSMC",
      "severity": "HIGH",
      "source": "chunk_id_289"
    },
    {
      "risk": "AMD: Zen 5 CPUs and MI300 GPUs on TSMC N4 — alternative sourcing limited",
      "severity": "MEDIUM",
      "source": "chunk_id_301"
    }
  ],
  "missing_context": [
    {
      "gap": "TSMC inventory buffer levels per customer not disclosed in public filings",
      "severity": "MEDIUM"
    }
  ],
  "crag_status": "CORRECT",
  "confidence": 0.79,
  "fallback_triggered": false,
  "qualitative_summary": "NVDA and AAPL face the highest supply chain concentration risk from a TSMC disruption due to their near-total reliance on TSMC advanced nodes with no short-term alternative foundry."
}
```

**Notes**
- ✅ Multi-hop Cypher traversal is the main advantage of graph over plain vector search
- ✅ `ticker: null` is valid for sector/cross-company queries
- ❌ Do NOT rank companies as investment recommendations — only report dependency facts

---

## Common Mistakes to Avoid

| ❌ Wrong | ✅ Correct |
|---|---|
| `"qualitative_summary": "This is bullish for AAPL"` | No sentiment — state facts only |
| `"rating": "strong buy"` in `competitive_moat` | `rating` must be `wide / narrow / none` only |
| `"source": "Apple 10-K 2025"` (vague) | `"source": "chunk_id_023"` (Neo4j chunk ID) |
| `"severity": "very high"` | Must be `HIGH / MEDIUM / LOW` exactly |
| Skipping `missing_context` when data is stale | Always flag stale or absent data explicitly |
| `confidence: 0.95` from a single Cypher result | Cap at 0.75 for single-source findings |
| Generating output when `crag_status: "INCORRECT"` | Set `fallback_triggered: true`, return minimal JSON |
| Including financial ratios not in Neo4j | Only include fields present in `company_overview` from Neo4j |

---

*Last updated: 2026-02-24 | Author: hck717*
