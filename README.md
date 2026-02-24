# 🎓 The Agentic Investment Analyst
### *A Multi-Agent, Self-Improving RAG System for Fundamental Equity Analysis*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-red)
![Neo4j](https://img.shields.io/badge/Neo4j-GraphDB-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-StructuredDB-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Quick Start (Infra)

```bash
git pull
cp .env.example .env
docker-compose up --build -d
```

- **Airflow UI:** http://localhost:8080 (admin/admin)
- **Qdrant Dashboard:** http://localhost:6333/dashboard
- **Neo4j Browser:** http://localhost:7474
- **PostgreSQL:** localhost:5432

### 🔍 Inspect Local Databases

**PostgreSQL (via psql):**
```bash
docker exec -it fyp-postgres psql -U airflow -d airflow
\dt          -- list tables
SELECT * FROM raw_timeseries LIMIT 10;
```

**Qdrant (via Dashboard):**
Navigate to http://localhost:6333/dashboard → Collections → `financial_documents`

**Neo4j (via Browser):**
Navigate to http://localhost:7474 → `MATCH (n) RETURN n LIMIT 25`

For full setup details, see `docs/setup_guide.md`.

### DAG Execution Commands

```bash
# Set scheduler container variable (docker compose v2)
SCHED=$(docker compose ps -q airflow-scheduler)

# List all DAGs
docker exec -it $SCHED airflow dags list

# Trigger a specific DAG (replace DAG_ID)
docker exec -it $SCHED airflow dags unpause <DAG_ID>
docker exec -it $SCHED airflow dags trigger  <DAG_ID>

# Check run status
docker exec -it $SCHED airflow dags list-runs -d <DAG_ID>

# View task logs
docker exec -it $SCHED airflow tasks list <DAG_ID>
docker exec -it $SCHED airflow tasks logs <DAG_ID> <TASK_ID> <RUN_ID>
```

---

## 📖 1. Abstract

Current AI solutions in finance suffer from **"black box" opacity**, **hallucinations**, and an inability to reconcile conflicting signals from multiple data sources. While Large Language Models can summarize text, they lack the structural reasoning to provide reliable investment analysis or justify their conclusions with auditable evidence.

**The Agentic Investment Analyst** is an autonomous, multi-agent platform designed to replicate the workflow of a senior buy-side research team. It deploys **9 specialized agents** — 7 domain experts, 1 Orchestrator (Supervisor), and 1 Synthesizer — operating on a **Global Plan-and-Execute / Local ReAct** architecture. Every agent uses a purpose-built RAG or reasoning strategy matched to its data type. The system handles queries from simple stock questions to full fundamental analyses, producing transparent, citation-verified reports with a full audit trail.

---

## 🎯 2. Project Objectives

- **Eliminate Hallucinations:** All numerical claims are computed deterministically via Python/SQL (dual-path verification). All qualitative claims must cite a specific retrieved chunk.
- **Visible Reasoning:** A full Chain-of-Thought trace — Plan → Execute → Conflict-Resolve → Synthesize — is displayed in the UI for every query.
- **Conflict-Aware Synthesis:** When agents disagree (e.g., Quant says "undervalued", Consensus says "earnings miss"), the Synthesizer explicitly resolves the conflict with a named winner and rationale.
- **All-Local Data Stack:** All databases (PostgreSQL, Qdrant, Neo4j) run in Docker on localhost — zero cloud dependency for data storage.
- **Cost-Effective LLM Strategy:** Local Ollama models (Qwen 2.5, Llama 3.2) handle high-volume offline tasks; cloud LLMs (GPT-4o, GPT-4o-mini) handle real-time reasoning only.

---

## 🏗️ 3. High-Level Architecture

The platform is structured in three runtime layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: INGESTION (Offline / Airflow Scheduled)               │
│  Airflow DAGs → ETL → PostgreSQL | Qdrant | Neo4j (All Local)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: INFERENCE (Real-Time / LangGraph)                     │
│  Supervisor (GPT-4o) → Parallel Agent Execution (7 Agents)      │
│  Global Plan-and-Execute  +  Per-Agent Local ReAct Loops        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: SYNTHESIS (Real-Time / GPT-4o)                        │
│  Conflict Resolution → Citation Verification → Report Assembly  │
│  Streamlit UI with streaming CoT + clickable citations          │
└─────────────────────────────────────────────────────────────────┘
```

### Orchestration Pattern: Global Plan-and-Execute + Local ReAct

| Level | Pattern | Agent | Description |
| :--- | :--- | :--- | :--- |
| **Global** | Plan-and-Execute | Supervisor (GPT-4o) | Classifies intent, selects agents, runs them in parallel, routes to Synthesizer |
| **Local** | ReAct | Each of the 7 Expert Agents | Each agent runs its own Thought → Action → Observation loop internally |

---

## 🤖 4. The Nine-Agent System

### Orchestration Agents

#### 🎯 Supervisor Agent (The "Senior PM")
- **Model:** GPT-4o
- **Role:** Global Plan-and-Execute orchestrator.
- **Responsibilities:**
  - Classify user intent (Specific Question / Comparative Analysis / Full Fundamental Analysis)
  - Select which of the 7 expert agents to invoke
  - Execute agents in parallel via LangGraph
  - Detect conflicts between agent outputs
  - Route consolidated outputs to the Synthesizer
- **Conflict Resolution Examples:**
  - *Analyst Optimism vs. Competitor Reality* → "⚠️ TRAP DETECTED. Street estimates ignore sector-wide warning signals from market leader."
  - *Insider Buying vs. Negative News* → "💡 CONTRARIAN OPPORTUNITY. Market overreacting to rumours. Insiders betting heavily on recovery."
  - *Cheap Valuation vs. Macro Headwind* → "⚠️ VALUE TRAP. FX headwinds will wipe out earnings gains. Wait for stabilization."

#### 📝 Synthesizer Agent (The "Editor-in-Chief")
- **Model:** GPT-4o
- **Role:** Post-execution consolidation, conflict narration, and final report formatting.
- **Responsibilities:**
  - Receive all 7 agent outputs as structured JSON
  - Resolve detected conflicts with explicit winner selection and rationale
  - Render one of three output templates (Q&A / Comparison / Full Analysis)
  - Enforce `[chunk:id]` citation protocol — every factual claim requires a source
  - Produce the Agent Consensus Matrix
- **Output Templates:**

```
# Q&A Template
## Answer: {direct_answer}
## Supporting Evidence: {cited_bullet_points}

# Comparison Template
## Executive Summary: {key_differences}
## Side-by-Side Table: {comparison_table_with_citations}
## Recommendation: {synthesized_insight}

# Full Analysis Template
## 1. Company Overview        → Business Analyst
## 2. Financial Health        → Quant Fundamental + Financial Modelling
## 3. Valuation               → Financial Modelling + Quant
## 4. Macro & FX Risk         → Macro Economic
## 5. Market Consensus Gaps   → Consensus & Strategy
## 6. Insider & Sentiment     → Insider & Sentiment
## 7. Breaking Developments   → Web Search
## 8. Conflict Resolution     → Synthesizer Narrative
## 9. Investment Thesis       → Conviction Level + Entry/Target/Stop
```

---

### 7 Expert Domain Agents

---

#### 1. 🧠 Business Analyst Agent (The "Deep Reader")
**Goal:** Extract strategy and risk from 10-K/10-Q filings with zero hallucinations.

**Architecture:** Graph-Augmented Corrective RAG (CRAG) with Proposition Chunking.

**Ingestion — Proposition-Based Chunking:**
- An LLM decomposes each 10-K/10-Q section into atomic, self-contained propositions.
- Example: `"Risk Factor 1: The company derives 65% of revenue from semiconductor customers, creating concentration risk."` → one complete Qdrant vector.
- Benefit: 25–30% better context preservation vs. naive character chunking.
- Storage: Qdrant with metadata `{filing_date, section, ticker, proposition_id}`.

**Retrieval — Hybrid Search (3 Paths):**
- **Dense (Vector):** Qdrant semantic search via `nomic-embed-text`.
- **Sparse (Keyword):** BM25 for exact financial terms (e.g., "goodwill impairment", "covenant breach").
- **Structural (Graph):** Neo4j Cypher traversal — `MATCH (:Strategy)-[:DEPENDS_ON]->(:Technology)`.

**Refinement — Corrective RAG (CRAG):**
- A BERT/small-LLM evaluator scores each retrieved document (0–1).
- Score < 0.5: Discard and trigger the Web Search Agent as fallback.
- Score 0.5–0.7: Supplement with web search for additional context.
- Score > 0.7: Use directly.

---

#### 2. 📊 Quantitative Fundamental Agent (The "Math Auditor")
**Goal:** Detect financial anomalies and compute fundamental factors with mathematical accuracy.

**Architecture:** Chain-of-Table with Dual-Path Code Verification (Non-RAG).

**Data Sources:** FMP Ultimate & EODHD All-In-One → PostgreSQL (local).

**Reasoning — Chain-of-Table:**
1. `SELECT` relevant columns (Revenue, EBIT, Assets, Liabilities…)
2. `FILTER` time range (TTM, FY2024, Last 4Q)
3. `CALCULATE` metrics (DSO, Gross Margin, FCF Conversion)
4. `RANK` by growth or quality score
5. `IDENTIFY` outliers (Z-score > 2 = anomaly flag)

**Execution — Dual-Path Verification:**
- **Path A:** Python (Pandas)
- **Path B:** SQL (DuckDB / PostgreSQL)
- `IF Path_A_result != Path_B_result THEN RAISE CalculationAlert` → prevents silent numeric errors.

**Factor Analysis (OLAP):**

| Category | Metrics |
| :--- | :--- |
| **Value** | P/E, EV/EBITDA, P/FCF, EV/Revenue |
| **Quality** | ROE, ROIC, Piotroski F-Score, Beneish M-Score |
| **Momentum/Risk** | Beta (60-day rolling), Sharpe Ratio, 12-Month Return |

---

#### 3. 💰 Financial Modelling Agent (The "Valuation Engine")
**Goal:** Build rigorous, assumption-driven valuation models and stress-test scenarios.

**Architecture:** Deterministic Python computation engine (Non-RAG). All outputs are calculated — never generated by LLM directly.

**Data Sources:** PostgreSQL (local) — historical financials, analyst estimates, WACC inputs.

**Models Implemented:**

**Discounted Cash Flow (DCF):**
- Inputs: Revenue growth assumptions (Base / Bull / Bear), EBIT margins, D&A, Capex, NWC changes.
- WACC calculation: `WACC = (E/V × Re) + (D/V × Rd × (1 - Tax))`
- Terminal Value: Gordon Growth Model — `TV = FCF_n × (1 + g) / (WACC - g)`
- Output: Intrinsic Value per share + Sensitivity Matrix (WACC × Terminal Growth Rate).

**Comparable Company Analysis (Comps):**
- Pulls peer group EV/EBITDA, P/E, EV/Revenue multiples from PostgreSQL.
- Applies median/mean peer multiples to target company metrics.
- Output: Implied valuation range across multiples.

**Precedent Transactions:**
- Historical M&A deal multiples (from FMP) for sector benchmarking.

**Scenario Analysis:**
```python
scenarios = {
    "Bear": {"revenue_growth": -0.05, "margin": 0.10, "wacc": 0.12},
    "Base": {"revenue_growth":  0.08, "margin": 0.18, "wacc": 0.10},
    "Bull": {"revenue_growth":  0.20, "margin": 0.25, "wacc": 0.09},
}
# Returns: {"Bear": $85, "Base": $130, "Bull": $195}
```

**LBO Analysis (Optional):**
- Models leveraged buyout entry/exit multiples and IRR for private-equity context.

**Output:** Intrinsic value range, upside/downside to current price, scenario probability weighting.

---

#### 4. 🤨 Market Consensus & Strategy Agent (The "Skeptic")
**Goal:** Understand "The Narrative" vs. "The Reality" — verify, don't trust.

**Architecture:** Contrastive RAG (Internal Company View vs. Street View vs. Competitor View).

**Data Sources:**
- **PostgreSQL (local):** FMP Analyst Estimates (EPS/Rev consensus), Price Targets, Ratings, Upgrades/Downgrades.
- **Qdrant (local):** Scraped competitor 10-K "Item 1A" (Risk) and "Item 7" (MD&A) sections from SEC EDGAR for top-5 peers; financial news summaries (Yahoo Finance / CNBC); FMP Earnings Transcripts.

**Process — Gap Analysis:**
1. Retrieve target company's stated strategy (from Business Analyst Agent).
2. Retrieve sell-side analyst thesis (from scraped summaries in Qdrant).
3. Retrieve competitor strategy (from peer filings in Qdrant).
4. Generate Gap Analysis:
   - Is the market ignoring a risk that a competitor highlighted?
   - Is the analyst's growth target higher than the company's own guidance?
   - Are consensus estimates above/below recent management tone?

**Output:** `Consensus Gap Score` (High gap = Opportunity or Trap, requires Synthesizer arbitration).

---

#### 5. 🌍 Macro Economic Agent (The "Economist")
**Goal:** Analyze economic cycles, interest rates, and FX impact on the target company.

**Architecture:** Text-to-SQL Time-Series Analysis + RAG on Central Bank Communications.

**Data Sources:**
- **PostgreSQL (local):** 30+ macro indicators — GDP, CPI, PPI, Unemployment, PMI, Yield Curve, Fed Funds Rate, 10Y/2Y Treasury Yields (EODHD & FMP).
- **Qdrant (local):** Fed/ECB meeting minutes, FOMC transcripts, economic news articles.

**Analysis:**
- **Economic Cycle Classification:** Expansion / Peak / Contraction / Trough (via Yield Curve inversion + PMI trend).
- **FX Impact:** Calculate interest rate differential; overlay with company's reported geographic revenue exposure from 10-K.
- **Rate Sensitivity:** Duration analysis for debt-heavy companies; discount rate impact on DCF valuations.

**Output:** `{cycle_stage, fx_risk_score, rate_sensitivity_rating, macro_narrative}`.

---

#### 6. 🐋 Insider & Sentiment Agent (The "Psychologist")
**Goal:** Detect divergence between what management says and what insiders actually do.

**Architecture:** Temporal Contrastive RAG (Time-Travel Comparison).

**Data Sources:**
- **Qdrant (local):** Earnings call transcripts (chunked by speaker turn); investor day presentation PDFs (scraped → text).
- **PostgreSQL (local):** SEC Form 4 insider trades; 13F institutional holdings.

**Retrieval — Time-Travel:**
- Compare current quarter's transcript embeddings vs. prior quarter.
- Cosine similarity between Q_n guidance language and Q_{n-1} guidance language.

**Analysis — Difference Engine:**
- **Semantic Drift:** Low cosine similarity score = narrative shift (e.g., CEO pivoting from "growth" language to "efficiency" language).
- **Visual vs. Verbal:** Compare slide deck promises (bullish visuals) vs. transcript Q&A (hesitant responses).
- **Action Check:** `IF Sentiment = Bullish AND Insiders = NET_SELLING → RED_FLAG`.
- **Institutional Crowding:** Flag if top-10 holders own >60% of float (liquidity risk).

**Output:** `{narrative_drift_score, insider_conviction_level, divergence_alerts[]}`.

---

#### 7. 🌐 Web Search Agent (The "News Desk")
**Goal:** Find "unknown unknowns" and surface breaking developments in real time.

**Model:** Qwen 2.5 (local Ollama) — fast, low-cost, sufficient for news triage.

**Architecture:** HyDE + Step-Back Prompting + Freshness Reranking.

**Workflow:**
1. **Step-Back Prompting:** "What recent macro or sector events could affect this company?"
2. **HyDE (Hypothetical Document Embeddings):** Generate a fake, ideal news article → embed → search Qdrant for real matching chunks.
3. **Reranking:** Filter results for Freshness (< 24h preferred) and Source Credibility (filings > wire news > social).
4. **Fallback Role:** Automatically triggered when Business Analyst's CRAG score < 0.5.

**Data Sources:** EODHD News Feed, FMP Press Releases, SEC EDGAR live filings, live web search.

**Output:** `{breaking_news[], sentiment_signal, unknown_risk_flags[]}`.

---

## 🗄️ 5. Local Database Architecture

All data storage runs **100% locally via Docker**. No cloud databases are used.

| Database | Container | Port | Role | What's Stored |
| :--- | :--- | :--- | :--- | :--- |
| **PostgreSQL 15** | `fyp-postgres` | 5432 | Structured time-series & fundamentals | OHLCV prices, ratios, macro indicators, insider trades, analyst estimates, earnings data |
| **Qdrant** | `fyp-qdrant` | 6333 | Vector embeddings for semantic search | 10-K/10-Q proposition chunks, news articles, earnings transcripts, central bank minutes |
| **Neo4j 5.15** | `fyp-neo4j` | 7474/7687 | Graph relationships | Company nodes, competitor edges, sector membership, supply chain links |

### PostgreSQL Schema Overview

```sql
-- Time-series data (prices, intraday, macro indicators)
raw_timeseries       (agent_name, ticker_symbol, data_name, ts_date, payload JSONB, source)

-- Fundamental/snapshot data (ratios, profile, estimates)
raw_fundamentals     (agent_name, ticker_symbol, data_name, as_of_date, payload JSONB, source)

-- Global shared datasets (once per day)
market_eod_us        (ts_date, payload JSONB, source)
global_economic_calendar (ts_date, payload JSONB, source)
global_ipo_calendar  (ts_date, payload JSONB, source)

-- Agent query logs (for self-improvement)
query_logs           (query_id UUID, query_text, agents_invoked[], latency_ms, timestamp)
citation_tracking    (query_id, chunk_id, cited BOOL, user_rating FLOAT)
```

### Qdrant Collection Schema

```
Collection: financial_documents
Vector size: 768 (nomic-embed-text via local Ollama)
Distance: Cosine

Payload metadata:
  - ticker_symbol    : str   (e.g. "AAPL")
  - agent_name       : str   (e.g. "business_analyst")
  - data_name        : str   (e.g. "10k_proposition")
  - filing_date      : str   (ISO date)
  - section          : str   (e.g. "Item 1A Risk Factors")
  - source           : str   (e.g. "eodhd", "sec_edgar", "fmp")
  - proposition_id   : str   (UUID)
  - boost_factor     : float (self-improving weight, default 1.0)
```

### Neo4j Graph Schema

```cypher
// Nodes
(:Company {ticker, name, sector, industry, country, marketCap})
(:Sector  {name})
(:Technology {name})

// Relationships
(:Company)-[:COMPETES_WITH]->(:Company)
(:Company)-[:BELONGS_TO]->(:Sector)
(:Company)-[:DEPENDS_ON]->(:Technology)
(:Company)-[:HAS_RISK]->(:RiskFactor {description, severity})
(:Company)-[:MENTIONED_IN]->(:Filing {date, type})
```

---

## ⚙️ 6. Data Pipelines (Airflow DAGs)

The system relies on **2 primary ingestion DAGs** (plus enrichment) to keep the local knowledge base fresh.

### DAG 1: EODHD Complete Ingestion (`eodhd_complete_ingestion`)
- **Schedule:** Daily at 01:00 UTC
- **Task Graph:** `scrape → [load_postgres ∥ load_neo4j ∥ load_qdrant] → summary`

| Agent Persona | Data Fetched | Destination |
| :--- | :--- | :--- |
| `business_analyst` | Financial news, sentiment trends, company profile | Qdrant (news) + PostgreSQL (sentiment) + Neo4j (profile) |
| `quantitative_fundamental` | Realtime quotes, OHLCV, intraday (1m/5m/15m/1h), SMA/EMA technicals, options | PostgreSQL |
| `financial_modeling` | Weekly/monthly prices, dividends, splits, earnings, IPO calendar, economic events, bulk EOD | PostgreSQL |

### DAG 2: FMP Complete Ingestion (`fmp_complete_ingestion`)
- **Schedule:** Daily at 02:00 UTC
- **Purpose:** Dual-path data verification — FMP data cross-validates EODHD figures.
- Fetches: Analyst estimates, price targets, earnings transcripts, insider trades (Form 4), institutional holdings (13F), DCF inputs.

### DAG 3: Sentiment Enrichment (`sentiment_processing`)
- **Schedule:** Triggered after DAG 1 completes
- **Action:** Runs local `FinTwitBERT` (Hugging Face) over raw news/social text to produce scalar sentiment scores (−1.0 to +1.0) stored in PostgreSQL.

### DAG 4: Weekly Self-Improvement (`weekly_model_retraining`)
- **Schedule:** Weekly (Sunday 03:00 UTC)
- **Action:**
  - Reads `citation_tracking` table → identifies high-rated chunks.
  - Updates `boost_factor` payload in Qdrant for cited chunks.
  - Triggers re-embedding of stale chunks with updated strategy.

### 🔮 Future DAGs
- **SEC EDGAR Parser:** Direct 10-K/10-Q HTML → proposition chunking pipeline.
- **Insider Trading Tracker:** Real-time Form 4 parser from EDGAR RSS feed.
- **Options Flow Monitor:** Unusual options activity detection from EODHD options data.

---

## 🤝 7. Multi-Agent Conflict Resolution

The Supervisor detects signal conflicts between agents and the Synthesizer resolves them explicitly.

### Conflict Resolution Examples

**Conflict A — Analyst Optimism vs. Competitor Reality**
> Consensus Agent: "Analysts project 20% growth, citing strong AI demand."
> Business Analyst (Competitor Data): "Top competitor 10-K warns of 'AI chip oversupply' and 'slowing enterprise capex'."
> **Synthesis:** `⚠️ TRAP DETECTED. Street estimates ignore sector-wide warning signals from the market leader. High probability of earnings miss. AVOID.`

**Conflict B — Insider Buying vs. Negative News**
> Web Search Agent: "Breaking: CEO investigation rumour causes stock to drop 10%."
> Insider Agent: "CFO purchased $2M shares yesterday — largest acquisition in 5 years."
> **Synthesis:** `💡 CONTRARIAN OPPORTUNITY. Market overreacting to unverified rumours. Insiders betting heavily on innocence/recovery. Speculative BUY with tight stop.`

**Conflict C — Cheap Valuation vs. Macro Headwind**
> Quant Agent: "P/E = 8x (historic low). FCF Yield = 12%."
> Macro Agent: "Company earns 60% of revenue in Europe. EUR/USD declining on ECB rate cuts."
> Financial Modelling Agent: "DCF Bear Case = $72 (current price $85) after 15% FX haircut on earnings."
> **Synthesis:** `⚠️ VALUE TRAP. Cheap for a reason. FX headwinds wipe out the apparent discount. Wait for currency stabilization before entry.`

---

## 📊 8. Final Output Structure

Every full analysis report includes:

### Agent Consensus Matrix

| Agent | Signal | Confidence | Key Finding |
| :--- | :--- | :--- | :--- |
| 🧠 Business Analyst | Neutral | 85% | Solid moat, but high regulatory risk in Item 1A |
| 📊 Quant Fundamental | Bullish | 92% | Undervalued; Piotroski F-Score = 8/9 |
| 💰 Financial Modelling | Bullish | 88% | DCF Base Case +35% upside; EV/EBITDA below peers |
| 🤨 Consensus & Strategy | Bearish | 78% | Street estimates 15% above peer guidance trend |
| 🌍 Macro Economic | Neutral | 70% | Late-cycle; FX exposure neutral |
| 🐋 Insider & Sentiment | Bullish | 88% | CFO buying; narrative tone stable QoQ |
| 🌐 Web Search | Neutral | 65% | No material breaking news |

### Executive Summary Block
```
Conviction Level: MODERATE BUY

Conflict Resolution: "Quant and Modelling see clear value; Consensus flags
an earnings miss risk from peer guidance. Insider conviction (CFO buying)
is the tie-breaker → Moderate Buy bias with reduced position sizing."

Actionable Recommendation:
  Entry Price:  $120
  Stop Loss:    $110  (-8%)
  Target:       $155  (+29%)
  Time Horizon: 12–18 months
```

---

## 🔄 9. Self-Improving RAG

The system learns continuously on three timescales:

**Per-Query (Real-Time):**
- Tracks which retrieved chunks were cited in the final answer.
- Boosts `boost_factor` in Qdrant payload for cited chunks (multiplicative weight applied at retrieval).

**Daily:**
- Identifies query types with low resolution rates → triggers targeted data acquisition.
- Flags retrieval gaps where no relevant document was found.

**Weekly:**
- Fine-tunes embedding strategy using successful query–chunk pairs.
- A/B tests chunking configurations (chunk size, overlap) on held-out query set.

```python
class SelfImprovingRAG:
    def retrieve(self, query: str) -> list[Chunk]:
        chunks = self.qdrant.search(query)
        for chunk in chunks:
            chunk.score *= self.get_boost_factor(chunk.id)  # Apply learned weights
        return sorted(chunks, key=lambda c: c.score, reverse=True)

    def post_query_feedback(self, query_id: str, cited_chunk_ids: list[str], rating: float):
        for chunk_id in cited_chunk_ids:
            self.increment_boost(chunk_id, weight=rating)
        self.log_to_postgres(query_id, cited_chunk_ids, rating)

    def weekly_retrain(self):
        pairs = self.get_high_rated_pairs(min_rating=4.0)
        self.embedding_model.fine_tune(pairs)
        self.reindex_qdrant()
```

---

## 🛡️ 10. Citation-Verification Protocol

### Three-Stage Verification

**Stage 1 — Grounded Generation:**
Every factual claim in the Synthesizer's output must reference a chunk ID:
> `"Apple's revenue grew 25% [chunk:149] driven by iPhone sales in China [chunk:203]."`
The system prompt explicitly forbids unsourced claims.

**Stage 2 — Critic Agent Audit (GPT-4o-mini):**
- Extracts factual assertions sentence by sentence.
- Checks each assertion against its cited chunk via NLI (Natural Language Inference).
- Entailment score > 0.85 required → `✅ VERIFIED`.
- Score 0.5–0.85 → `⚠️ PARTIAL` (flagged for human review).
- Score < 0.5 → `❌ REJECTED` (removed from report).

**Stage 3 — Interactive UI Verification:**
- Every `[chunk:id]` is a clickable hyperlink opening a source modal.
- Relevant sentences highlighted in yellow.
- Full provenance chain shown: `EODHD API → ETL → Qdrant → Chunk → Report`.

### Audit Trail (Logged Per Query)
```json
{
  "query_id": "uuid-12345",
  "timestamp": "2026-02-24T13:00:00Z",
  "agents_invoked": ["business_analyst", "quant_fundamental", "financial_modelling"],
  "chunks_retrieved": 31,
  "chunks_cited": 18,
  "critic_pass_rate": 0.94,
  "conflicts_detected": 1,
  "conflict_resolution": "insider_conviction_tie_breaker",
  "user_feedback": null
}
```

---

## 💻 11. Technical Specifications

### LLM Strategy

| Role | Model | Inference | Justification |
| :--- | :--- | :--- | :--- |
| **Supervisor** | GPT-4o | Cloud (runtime) | Complex multi-step planning, conflict detection |
| **Synthesizer** | GPT-4o | Cloud (runtime) | Coherent long-form narrative generation |
| **Critic** | GPT-4o-mini | Cloud (runtime) | Cost-effective NLI verification |
| **Business Analyst RAG** | GPT-4o-mini | Cloud (runtime) | CRAG evaluator scoring |
| **Ingestion ETL (News/Filings)** | Llama 3.2 (3B) | Local Ollama | High-volume offline extraction, zero cost |
| **Web Search Agent** | Qwen 2.5 | Local Ollama | Fast real-time news triage, low latency |
| **Embeddings** | nomic-embed-text | Local Ollama | 768-dim, zero API cost, fast |

### Data Sources & Budget (~$300 USD / 4 months)

| Data Type | Provider | Cost | Notes |
| :--- | :--- | :--- | :--- |
| Core Equity (Price/Fund/News) | EODHD All-In-One | ~$200 | Backbone: prices, fundamentals, news |
| Fundamentals + Estimates | FMP Ultimate | ~$0–50 | Analyst estimates, transcripts, Form 4 |
| Macro Indicators | EODHD / FRED | $0 | GDP, CPI, Yields, PMI |
| Insider/Institutional | FMP / SEC EDGAR | $0 | Form 4, 13F filings |
| Sentiment Model | Hugging Face | $0 | Local FinTwitBERT |
| Embeddings | Ollama (local) | $0 | nomic-embed-text |
| Vector DB | Qdrant (Docker) | $0 | Self-hosted |
| Graph DB | Neo4j Community (Docker) | $0 | Self-hosted |
| LLM API Buffer | OpenAI | ~$50–100 | Supervisor + Synthesizer + Critic |

---

## 📅 12. Implementation Roadmap

**Project Duration:** February 1 – April 30, 2026 (13 Weeks)

---

### 🔵 Phase 1: Foundation & Data Infrastructure (Weeks 1–4) ✅

#### Week 1 (Feb 1–7) — Environment Setup ✅
- [x] Project repo structure (`/ingestion`, `/agents`, `/skills`, `/rag`, `/ui`)
- [x] Docker Compose: Qdrant, PostgreSQL, Neo4j, Airflow
- [x] EODHD + FMP API integration scripts
- [x] PostgreSQL schema: `raw_timeseries`, `raw_fundamentals`
- [x] Qdrant collection initialized with `nomic-embed-text` (768-dim)

**Deliverable:** Docker stack running; price data ingested for 10 test stocks.

---

#### Week 2 (Feb 8–14) — Core Ingestion DAGs ✅
- [x] `dag_eodhd_ingestion_unified.py` — 3 agent personas, parallel fan-out
- [x] `dag_fmp_ingestion_unified.py` — dual-path data source
- [x] ETL loaders: `load_postgres.py`, `load_qdrant.py`, `load_neo4j.py`
- [x] Idempotent MD5 hash-based deduplication
- [x] Ollama warm-up probe in `load_qdrant.py`

**Deliverable:** Automated daily ingestion for S&P 100 stocks operational.

---

#### Week 3 (Feb 15–21) — Agent Skill: Quant Fundamental ✅
- [x] `skills/quant_fundamental/` — Chain-of-Table SQL/Python engine
- [x] Dual-path verification (Pandas vs DuckDB)
- [x] Factor calculations: ROE, ROIC, Piotroski F-Score, Beneish M-Score
- [x] Unit test suite for all calculations

**Deliverable:** Skill returns audited fundamental report for any ticker.

---

#### Week 4 (Feb 22–28) — Agent Skill: Financial Modelling
- [ ] `skills/financial_modelling/` — DCF engine
  - WACC calculation (Cost of Equity via CAPM + Cost of Debt)
  - 5-year FCF projection engine
  - Terminal Value (Gordon Growth Model)
  - Sensitivity matrix (WACC × Terminal Growth Rate)
- [ ] Comparable Company Analysis (Comps) from PostgreSQL peer data
- [ ] Scenario engine: Bear / Base / Bull with probability weighting
- [ ] LBO model (optional stretch goal)

**Deliverable:** CLI tool returning `{intrinsic_value, upside_pct, scenario_table}` for any ticker.

---

### 🟢 Phase 2: Agent Layer — RAG Specialists (Weeks 5–8)

#### Week 5 (Mar 1–7) — Business Analyst Agent
- [ ] `agents/business_analyst/` — CRAG + Proposition Chunking
  - SEC EDGAR 10-K/10-Q scraper → proposition decomposer (LLM)
  - Hybrid search: dense (Qdrant) + sparse (BM25) + graph (Neo4j Cypher)
  - CRAG evaluator with score threshold logic
  - Web Search fallback trigger (score < 0.5)
- [ ] Neo4j relationship schema: `COMPETES_WITH`, `DEPENDS_ON`, `HAS_RISK`

**Deliverable:** Agent extracts and cites specific risk factors from a 10-K filing.

---

#### Week 6 (Mar 8–14) — Consensus & Macro Agents
- [ ] `agents/consensus_strategy/` — Contrastive RAG
  - SEC EDGAR peer filing scraper (Item 1A + Item 7 for top-5 competitors)
  - Gap Analysis engine: company guidance vs. analyst consensus vs. peer warnings
  - Consensus Gap Score calculator
- [ ] `agents/macro_economic/` — Text-to-SQL + RAG
  - Economic cycle classifier (Yield Curve + PMI)
  - FX impact calculator (revenue exposure × rate differential)
  - Fed/ECB minutes RAG (Qdrant)

**Deliverable:** Both agents produce structured JSON output for AAPL and NVDA.

---

#### Week 7 (Mar 15–21) — Insider/Sentiment + Web Search Agents
- [ ] `agents/insider_sentiment/` — Temporal Contrastive RAG
  - Earnings transcript chunking by speaker turn
  - Semantic drift calculator (cosine similarity QoQ)
  - Form 4 + 13F data reader from PostgreSQL
  - Divergence alert logic
- [ ] `agents/web_search/` — HyDE + Step-Back + Reranking
  - Qwen 2.5 (local Ollama) integration
  - EODHD live news feed connector
  - Freshness + credibility reranker

**Deliverable:** Insider agent detects CFO sell-off in backdated test case. Web agent surfaces same-day news.

---

#### Week 8 (Mar 22–28) — LangGraph Supervisor + Synthesizer
- [ ] `agents/supervisor/` — Global Plan-and-Execute (GPT-4o)
  - Intent classifier (Q&A / Comparison / Full Analysis)
  - Parallel agent execution via LangGraph `send()` nodes
  - Conflict detection logic
- [ ] `agents/synthesizer/` — Conflict resolution + report assembly
  - Three output templates (Q&A / Comparison / Full Analysis)
  - Agent Consensus Matrix builder
  - `[chunk:id]` citation enforcement

**Deliverable:** Full 9-agent pipeline produces a report for NVDA end-to-end.

---

### 🟪 Phase 3: Trust, UI & Polish (Weeks 9–13)

#### Week 9 (Mar 29 – Apr 4) — Citation Verification + Critic Agent
- [ ] Critic Agent (GPT-4o-mini): NLI-based sentence-level verification
- [ ] Audit trail logging to PostgreSQL (`citation_tracking` table)
- [ ] Target: ≥ 90% critic pass rate on 25 test queries

#### Week 10 (Apr 5–11) — Self-Improving RAG
- [ ] `SelfImprovingRAG` class with Qdrant boost-factor updates
- [ ] Feedback collection API (cited chunks, user ratings)
- [ ] Weekly retraining DAG (`weekly_model_retraining`)

#### Week 11 (Apr 12–18) — Streamlit UI
- [ ] Real-time streaming CoT display (Plan → Execute → Synthesize)
- [ ] Agent Consensus Matrix widget
- [ ] Clickable `[chunk:id]` citations → source modal
- [ ] DCF sensitivity heatmap (Plotly)
- [ ] Price chart + sentiment timeline overlay

#### Week 12 (Apr 19–25) — Performance Optimization
- [ ] Latency profiling: target < 20s for full analysis
- [ ] A/B test chunking strategies (proposition vs. fixed-size)
- [ ] Optimize Qdrant payload indexing for filtered search

#### Week 13 (Apr 26–30) — Final Testing & Submission
- [ ] 50 diverse test queries (20 Q&A / 15 Comparison / 15 Full Analysis)
- [ ] 5 demo reports: AAPL, TSLA, NVDA, MSFT, GOOGL
- [ ] Architecture whitepaper
- [ ] Evaluation dataset with ground truth annotations

---

## 📊 13. Evaluation Framework

| Metric | Definition | Target |
| :--- | :--- | :--- |
| **Citation Verification Rate (CVR)** | % factual claims passing Critic Agent NLI check | ≥ 95% |
| **Numerical Accuracy Score (NAS)** | Dual-path Python vs SQL agreement rate | 100% |
| **Retrieval Precision@5** | Relevance of top-5 chunks (human annotated) | ≥ 0.85 |
| **Citation Utilization Rate (CUR)** | % retrieved chunks cited in final answer | ≥ 60% |
| **Query Resolution Rate (QRR)** | % queries answered without "insufficient data" | ≥ 92% |
| **Conflict Detection Accuracy** | % real conflicts correctly identified by Supervisor | ≥ 85% |
| **Full Analysis Latency** | End-to-end response time for 7-agent analysis | < 30s |
| **Self-Improvement Rate** | Citation precision gain after 1 week of usage | +5% / week |

---

## 📦 14. Deliverables

1. **Streamlit UI:** Real-time streaming app with CoT visualization, Agent Consensus Matrix, clickable citations, and DCF charts.
2. **Source Code:** Modular Python repo — `/ingestion`, `/agents`, `/skills`, `/rag`, `/ui`.
3. **Local Docker Stack:** One-command `docker-compose up` launches all 4 services (PostgreSQL, Qdrant, Neo4j, Airflow).
4. **Documentation:** Architecture whitepaper, agent design specs, evaluation report.
5. **Demo Reports:** 5 full-analysis reports (AAPL, TSLA, NVDA, MSFT, GOOGL) with audit trails.
6. **Evaluation Dataset:** 50-query test set with ground truth annotations.
