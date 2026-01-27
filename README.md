# 🎓 The Agentic Investment Analyst
### *An Autonomous, Self-Improving RAG System for Fundamental Equity Analysis*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 1. Abstract
Current AI solutions in finance often suffer from **"black box" opacity** and **hallucinations**. While Large Language Models (LLMs) can summarize text, they lack the structural understanding to provide reliable financial analysis or the rigor to justify their claims.

**The Agentic Investment Analyst** is an autonomous agent designed to mimic a human analyst's workflow. It utilizes a **Self-Improving Agentic RAG** architecture that learns from past queries and feedback, an **Iterative Chain-of-Thought (CoT)** engine to "plan, execute, and synthesize" investment insights, and a **Report Synthesis Layer** that consolidates multi-agent outputs into coherent, structured analyses. The system handles diverse user requests—from specific stock questions to multi-company comparisons to comprehensive fundamental analyses—producing transparent, evidence-backed reports with pixel-perfect citations that enable users to audit the AI's reasoning trace.

---

## 🎯 2. Project Objectives
*   **Solve the "Reliability Gap":** Deploy **Self-Improving Agentic RAG** that learns from successful and failed queries to continuously enhance retrieval accuracy and response quality.
*   **Eliminate the "Black Box":** Implement a **visible Chain-of-Thought** reasoning loop, allowing users to see *how* the agent reached each conclusion.
*   **Zero-Tolerance for Hallucination:** Enforce a strict **"Citation-Verification Protocol"** where every claim must be backed by a retrieved document snippet, audited by a Critic Agent.
*   **Flexible Query Handling:** Support diverse user intents including specific questions ("What is Apple's P/E ratio?"), comparative analyses ("Compare TSMC vs Samsung margins"), and comprehensive reports ("Give me a fundamental analysis of NVDA").
*   **Hybrid Architecture:** Demonstrate a cost-effective **"Cloud Supervisor / Local Worker"** model, using **GPT-4o** for high-level reasoning and **Llama 3.2 (3B)** for massive data processing.

---

## 🛠️ 3. Methodology

### A. High-Level Architecture
The platform follows an enhanced **Plan-Execute-Synthesize** pattern with three distinct layers:

#### 1. Ingestion Layer (The "Worker")
*   **Role:** Runs offline on a schedule (e.g., nightly) via **Airflow**.
*   **Task:** Ingests raw data (News, Filings, Tweets), processes it using a lightweight local LLM (**Llama 3.2**), and populates specialized databases.
*   **Tech Stack:**
    *   **Airflow:** Orchestration.
    *   **SQLite:** Structured numerical data (Prices, Ratios, Fundamentals).
    *   **Qdrant:** Semantic vector embeddings with metadata for self-improving retrieval.

#### 2. Inference Layer (The "Supervisor")
*   **Role:** Runs in real-time upon user query.
*   **Task:** Uses a high-intelligence Cloud LLM (**GPT-4o**) to:
    1. Understand user intent (Question / Comparison / Full Analysis)
    2. Plan tool calls across specialized Agent Skills
    3. Execute and monitor skill outputs
    4. Route results to the Synthesis Layer

#### 3. Synthesis Layer (The "Consolidator")
*   **Role:** Post-execution aggregation and formatting.
*   **Task:** A specialized **Summarizer Agent** that:
    1. Receives all skill outputs (financial metrics, news sentiment, valuation data, etc.)
    2. Consolidates information into a coherent narrative
    3. Structures the output based on user request type
    4. Formats final report with proper citations and markdown structure
    5. Ensures consistency and removes redundancies across agent outputs
*   **Implementation:** Follows the Agent Skills pattern with its own `SUMMARIZER_SKILL.md` manifest defining output templates for different report types (Q&A, Comparison, Full Analysis).

### B. Architectural Uniqueness

#### 1. Self-Improving Agentic RAG
Unlike traditional RAG systems that use static retrieval, our system continuously learns and adapts:
*   **Query Performance Tracking:** Logs every query with metrics (retrieval relevance, citation accuracy, user feedback)
*   **Feedback Loop Integration:** User corrections and ratings feed back into the system
*   **Retrieval Strategy Optimization:** Adjusts embedding weights, metadata filters, and chunk strategies based on historical performance
*   **Skill Performance Analysis:** Tracks which skills produce the most valuable insights for different query types
*   **Automated Retraining Triggers:** Periodically reindexes with improved chunking strategies when performance degrades

#### 2. Agentic Skills Architecture
Encapsulates logic into deterministic, modular tools:
*   Each skill has a `SKILL.md` manifest defining its purpose, inputs, outputs, and data dependencies
*   Python functions ensure mathematical accuracy (no LLM hallucination in calculations)
*   Skills can be independently tested, versioned, and improved

#### 3. Visible Chain-of-Thought
Displays the Agent's reasoning trace in real-time:
*   `Plan` → User intent classification and skill selection
*   `Execute` → Individual skill invocations with intermediate results
*   `Synthesize` → Consolidation logic and final report assembly
*   All steps visible in the UI for full transparency

### C. User Interface (Streamlit)
*   **Real-time Streaming:** Visualizes the Agent's thought process across all three layers.
*   **Interactive Charts:** Price trends, sentiment analysis, and valuation comparisons.
*   **Evidence Linking:** Side-by-side display of the final report and retrieved citations with clickable hyperlinks to source documents.
*   **Query Type Detection:** UI adapts to show different layouts for Q&A, comparisons, or full analyses.

### D. The Ingestion Logic: Building the Knowledge Base
We utilize an automated **"LLM-ETL"** pipeline:
1.  **Input:** Raw financial data (news articles, filings, price data).
2.  **Extraction:** **Llama 3.2 (3B)** extracts key information via structured JSON prompts.
3.  **Transformation:** Python scripts perform cleaning, normalization, and entity resolution (e.g., "Apple Inc." → `AAPL`).
4.  **Enrichment:** Generate embeddings and attach metadata (date, source, relevance scores).
5.  **Loading:** Insert into SQLite (structured) and Qdrant (semantic) with dual-index strategy for hybrid search.

---

## 🔄 4. Self-Improving RAG: How the System Learns

### A. The Learning Cycle
The system implements a continuous improvement loop that operates on three timescales:

#### Real-Time Learning (Per Query)
*   **Immediate Feedback:** Tracks which retrieved chunks were actually cited in the final answer
*   **Relevance Scoring:** Calculates citation rate per document chunk (cited chunks get boosted in future searches)
*   **Context Window Optimization:** Monitors token usage vs. information density to refine chunk sizes

#### Short-Term Learning (Daily)
*   **Query Pattern Analysis:** Identifies common question types and frequently requested companies
*   **Retrieval Gap Detection:** Flags queries where no relevant documents were found (triggers data acquisition)
*   **Skill Effectiveness Metrics:** Measures which skills contributed most to high-rated responses

#### Long-Term Learning (Weekly/Monthly)
*   **Embedding Model Fine-Tuning:** Uses successfully cited pairs (query → relevant chunks) as training data
*   **Chunking Strategy Refinement:** A/B tests different chunk sizes and overlap percentages
*   **Metadata Schema Evolution:** Adds new metadata fields discovered to improve filtering (e.g., "earnings_call" vs. "news_article")

### B. Implementation Details
```python
# Pseudo-code for self-improvement workflow
class SelfImprovingRAG:
    def retrieve(self, query):
        # Standard retrieval
        chunks = self.vector_db.search(query)
        
        # Apply learned boost factors
        for chunk in chunks:
            chunk.score *= self.get_boost_factor(chunk.id)
        
        return chunks
    
    def post_query_feedback(self, query_id, cited_chunks, user_rating):
        # Boost cited chunks
        for chunk_id in cited_chunks:
            self.increment_boost(chunk_id, weight=user_rating)
        
        # Log for batch retraining
        self.training_buffer.append({
            'query': query_id,
            'positive_chunks': cited_chunks,
            'rating': user_rating
        })
        
    def weekly_retrain(self):
        # Retrain embedding model on successful retrievals
        training_pairs = self.training_buffer.get_high_rated()
        self.embedding_model.fine_tune(training_pairs)
        self.reindex_knowledge_base()
```

### C. Performance Metrics Tracked
*   **Citation Rate:** % of retrieved chunks that appear in final answer
*   **Retrieval Precision:** Relevance of top-k results
*   **Query Resolution Rate:** % of queries successfully answered without "I don't know"
*   **User Satisfaction Score:** Explicit ratings + implicit signals (time on page, citation clicks)

---

## 🛡️ 5. Citation-Verification Loop: Ensuring Trust

### A. The Three-Stage Verification Process

#### Stage 1: Grounded Generation (During Synthesis)
*   **Constraint:** The Summarizer Agent receives retrieved context chunks as structured input with unique IDs
*   **Requirement:** Every factual claim must reference a chunk ID in square brackets: `[chunk:id]`
*   **Enforcement:** LLM system prompt explicitly forbids unsourced claims
*   **Example:** "Apple's revenue grew 25% [chunk:149] driven by iPhone sales in China [chunk:203]."

#### Stage 2: Critic Agent Audit (Post-Generation)
A specialized verification agent performs sentence-level validation:

**Input:** 
- Generated sentence: "Tesla's gross margin declined to 18.2% in Q3 2024."
- Retrieved chunks: [chunk:45, chunk:78, chunk:112]

**Process:**
1. **Claim Extraction:** Identifies factual assertions ("margin = 18.2%", "period = Q3 2024")
2. **Source Verification:** Checks if chunk:45 contains matching data
3. **Semantic Matching:** Uses NLI (Natural Language Inference) model to verify entailment
4. **Threshold:** Entailment score > 0.85 required for approval

**Output:**
- ✅ VERIFIED: Claim supported by source
- ⚠️ PARTIAL: Claim partially supported (flags for human review)
- ❌ REJECTED: No supporting evidence (removed from report)

#### Stage 3: Interactive User Verification (In UI)
*   **Clickable Citations:** Every `[chunk:id]` hyperlink opens a modal showing the source document
*   **Highlight Matching:** Relevant sentences highlighted in yellow
*   **Provenance Chain:** Shows data lineage (EODHD → Ingestion Pipeline → Chunk → Report)
*   **Feedback Mechanism:** Users can flag incorrect citations (feeds into self-improvement)

### B. Handling Edge Cases

#### Numerical Precision Conflicts
Problem: Source says "approximately 18%", model cites "18.2%"
Solution: Numerical claims extracted via Agent Skills (Python calculations) inherit source precision

#### Temporal Mismatches  
Problem: Source from "Q2 2024", claim about "Q3 2024"
Solution: Critic Agent flags temporal inconsistencies; Summarizer Agent uses most recent data

#### Contradictory Sources
Problem: Two chunks provide different values
Solution: System includes both with citations: "Revenue reported as $95B [chunk:12] (company filing) vs. $94.8B [chunk:34] (analyst estimate)"

### C. Audit Trail Logging
Every report includes a metadata appendix:
```json
{
  "query_id": "uuid-12345",
  "generation_timestamp": "2026-01-27T14:22:00Z",
  "chunks_retrieved": 24,
  "chunks_cited": 12,
  "critic_pass_rate": 0.94,
  "user_feedback": null,
  "regeneration_count": 0
}
```

---

## 🎭 6. Handling Diverse User Request Types

### A. Request Classification
The Supervisor Agent first classifies user intent into three categories:

| Request Type | Example Query | Response Structure |
| :--- | :--- | :--- |
| **Specific Question** | "What is Microsoft's current P/E ratio?" | Direct answer with 1-2 supporting facts |
| **Comparative Analysis** | "Compare Tesla vs. BYD margins and revenue growth" | Side-by-side table + narrative differences |
| **Fundamental Analysis** | "Give me a full fundamental analysis of NVDA" | Structured 5-section report (see below) |

### B. Skill Routing by Request Type

#### Specific Questions
- **Skills Invoked:** 1-2 targeted skills
- **Example:** P/E question → `Valuation Reality` skill only
- **Output Format:** 
  ```
  Microsoft's P/E ratio is 28.4 [chunk:78], below the sector median 
  of 32.1 [chunk:103] but above its 5-year average of 24.7 [chunk:156].
  ```

#### Comparative Analysis
- **Skills Invoked:** 3-4 relevant skills per company
- **Parallel Execution:** Skills run for both companies simultaneously
- **Output Format:** Markdown table + key differentiators
  ```markdown
  | Metric | Tesla | BYD |
  |--------|-------|-----|
  | Gross Margin | 18.2% [chunk:45] | 22.1% [chunk:89] |
  | Revenue Growth (YoY) | +25% [chunk:46] | +38% [chunk:90] |
  
  **Key Insight:** BYD demonstrates superior operational efficiency...
  ```

#### Fundamental Analysis
- **Skills Invoked:** All 5 core skills
- **Sequential Execution:** Skills run in dependency order
- **Output Format:** Structured 5-section report:
  1. **Company Overview** (from Knowledge Base skill)
  2. **Financial Health** (Fundamental Health skill)
  3. **Valuation Assessment** (Valuation Reality + Price Momentum skills)
  4. **Risk Factors** (Sentiment Analysis + Volatility skills)
  5. **Investment Thesis** (Synthesizer consolidation)

### C. Synthesizer Agent Templates
The Synthesizer maintains three report templates:

```python
# templates/qa_template.md
## Answer
{direct_answer}

## Supporting Context
{bullet_points_from_skills}

---

# templates/comparison_template.md
## Executive Summary
{key_differences}

## Detailed Comparison
{comparison_table}

## Recommendation
{synthesized_insight}

---

# templates/full_analysis_template.md
## 1. Company Overview
{business_description}

## 2. Financial Health Analysis
{fundamentals_output}

## 3. Valuation Assessment
{valuation_output}

## 4. Risk & Sentiment Analysis
{risk_output}

## 5. Investment Thesis
{final_synthesis}
```

---

## ⚙️ 7. Technical Specifications

### A. Data Sources & Budget (~$300 USD)
| Data Type | Provider | Cost (4 Mo) | Notes |
| :--- | :--- | :--- | :--- |
| **Core (Price/Fund/News)** | EODHD | $200 | Backbone of the system. |
| **Macro Economics** | EODHD / FRED | $0 | GDP, Inflation, Rates. |
| **Volatility / Risk** | yfinance | $0 | VIX (Fear Index). |
| **Social Sentiment** | Reddit (PRAW) | $0 | "Personal Use" script. |
| **Sentiment Model** | Hugging Face | $0 | Local FinTwitBERT. |
| **Vector DB** | Qdrant | $0 | Self-Hosted Docker. |
| **LLM Buffer** | OpenAI/Anthropic| ~$100 | Supervisor + Critic tokens. |

### B. AI Strategy: The "Hybrid Brain"
| Component | Model | Role | Justification |
| :--- | :--- | :--- | :--- |
| **Worker** | **Llama 3.2 (3B)** | Ingestion (Offline) | High volume, zero cost. |
| **Supervisor** | **GPT-4o** | Planner (Runtime) | Complex reasoning, intent classification. |
| **Summarizer** | **GPT-4o** | Synthesis (Runtime) | Coherent narrative generation. |
| **Critic** | **GPT-4o-mini** | Verification (Runtime) | Cost-effective validation. |

### C. The 5 Core Agentic Skills
| Skill | Purpose | Data Source | Logic / Output |
| :--- | :--- | :--- | :--- |
| **1. Fundamental Health** | Quality Check | SQLite | SQL: `ROE > 15%`, `Debt/EBITDA < 3x`, `FCF Margin`. |
| **2. Price & Momentum** | Timing | SQLite | `Price vs 200SMA` + `RSI` + `Volume Trends`. |
| **3. Valuation Reality** | Pricing | SQLite | `P/E` vs Peers/History + `DCF Sensitivity`. |
| **4. Sentiment Analysis** | Market Psychology | Qdrant + SQLite | News sentiment + Social media signals. |
| **5. Risk Assessment** | Protection | SQLite | `Beta`, `Volatility`, `Correlation` matrices. |

---

## 📅 8. Week-by-Week Implementation Roadmap
**Project Duration:** February 1 - April 30, 2026 (13 Weeks)

---

### 🔵 Phase 1: Foundation & Data Infrastructure (Weeks 1-4)

#### **Week 1: Feb 1-7 | Environment Setup & Data Pipeline Foundation**
**Milestone:** *Development environment operational with initial data ingestion*

**Tasks:**
- [ ] Set up project repository structure (`/ingestion`, `/skills`, `/agents`, `/rag`, `/ui`)
- [ ] Configure Docker Compose (Qdrant, Airflow, PostgreSQL)
- [ ] Create EODHD API integration scripts for price/fundamentals data
- [ ] Set up SQLite schema for structured financial data
- [ ] Initialize Qdrant collection with proper metadata fields

**Deliverable:** Docker stack running + ability to ingest price data for 10 test stocks

---

#### **Week 2: Feb 8-14 | Core Data Ingestion Pipeline**
**Milestone:** *Automated daily data ingestion operational*

**Tasks:**
- [ ] Build **Airflow DAG 1**: Daily price/fundamentals ingestion workflow
- [ ] Implement news article scraping (EODHD Financial News API)
- [ ] Create ETL scripts for data cleaning and normalization
- [ ] Set up entity resolution logic (company names → ticker symbols)
- [ ] Test end-to-end ingestion for 50 stocks

**Deliverable:** Automated pipeline ingesting daily data for S&P 100 stocks

---

#### **Week 3: Feb 15-21 | First Agentic Skill - Fundamental Health**
**Milestone:** *First deterministic skill functional with SQL queries*

**Tasks:**
- [ ] Create `skills/fundamental_health/` directory with `SKILL.md` manifest
- [ ] Implement SQL-based financial health checks:
  - ROE calculation and thresholds
  - Debt/EBITDA ratios
  - Free Cash Flow margin analysis
- [ ] Build skill testing framework with unit tests
- [ ] Document skill inputs/outputs/dependencies

**Deliverable:** Skill returns comprehensive health report for any ticker (e.g., AAPL, MSFT)

---

#### **Week 4: Feb 22-28 | Skills 2 & 3 - Valuation & Momentum**
**Milestone:** *Three core financial analysis skills operational*

**Tasks:**
- [ ] Implement **Skill 2: Price & Momentum**
  - Moving averages (50/200 SMA)
  - RSI calculations
  - Volume trend analysis
- [ ] Implement **Skill 3: Valuation Reality**
  - P/E ratio vs. sector median
  - Historical P/E comparison
  - PEG ratio analysis
- [ ] Create skill orchestration testing suite
- [ ] Build simple CLI tool to invoke skills manually

**Deliverable:** CLI that accepts ticker + skill name, returns structured JSON output

---

### 🟢 Phase 2: Intelligent Agent Layer (Weeks 5-8)

#### **Week 5: Mar 1-7 | LangGraph Supervisor Foundation**
**Milestone:** *Basic agentic orchestration working for single-skill queries*

**Tasks:**
- [ ] Set up LangGraph project structure
- [ ] Implement basic Supervisor agent:
  - Intent parser (identify which skill to call)
  - Single-skill execution flow
  - Output formatter
- [ ] Create prompt templates for GPT-4o Supervisor
- [ ] Test with 20 simple queries ("What is Tesla's P/E ratio?")

**Deliverable:** Agent correctly routes simple questions to appropriate skills

---

#### **Week 6: Mar 8-14 | Multi-Skill Orchestration & Request Classification**
**Milestone:** *Agent handles complex multi-step queries*

**Tasks:**
- [ ] Implement request type classifier (Q&A / Comparison / Full Analysis)
- [ ] Build skill dependency graph (which skills to call in what order)
- [ ] Implement parallel execution for comparison queries
- [ ] Create Plan-Execute workflow in LangGraph:
  - Planner: Generates execution plan
  - Executor: Runs skills and collects outputs
- [ ] Test with 15 comparison queries ("Compare NVDA vs AMD margins")

**Deliverable:** Agent executes multi-skill plans for all three query types

---

#### **Week 7: Mar 15-21 | Skills 4 & 5 + RAG Foundation**
**Milestone:** *All 5 skills operational + basic RAG retrieval working*

**Tasks:**
- [ ] Implement **Skill 4: Sentiment Analysis**
  - Integrate FinTwitBERT for news sentiment
  - Reddit sentiment scraping (PRAW)
  - Aggregate sentiment scores
- [ ] Implement **Skill 5: Risk Assessment**
  - Beta calculations
  - Volatility metrics (VIX correlation)
  - Portfolio correlation matrix
- [ ] Build basic RAG retrieval:
  - Embed news articles into Qdrant
  - Implement semantic search
  - Return top-k relevant chunks

**Deliverable:** Agent can answer "What's the market sentiment on Tesla?" with cited sources

---

#### **Week 8: Mar 22-28 | Synthesizer Agent & Report Templates**
**Milestone:** *Coherent multi-skill outputs consolidated into formatted reports*

**Tasks:**
- [ ] Create `agents/synthesizer/` with `SUMMARIZER_SKILL.md`
- [ ] Implement three report templates:
  - `qa_template.md` (direct answers)
  - `comparison_template.md` (side-by-side tables)
  - `full_analysis_template.md` (5-section report)
- [ ] Build template rendering engine (Jinja2 or custom)
- [ ] Implement Summarizer Agent that:
  - Receives all skill outputs
  - Consolidates into narrative
  - Applies appropriate template
- [ ] Test full analysis reports for 5 stocks

**Deliverable:** Agent generates publication-ready fundamental analysis report for NVDA

---

### 🟪 Phase 3: Trust, Self-Improvement & UI (Weeks 9-13)

#### **Week 9: Mar 29 - Apr 4 | Citation-Verification System**
**Milestone:** *Every claim in reports backed by verifiable sources*

**Tasks:**
- [ ] Implement Stage 1: Grounded Generation
  - Modify Summarizer to require `[chunk:id]` citations
  - Create system prompts that forbid unsourced claims
- [ ] Implement Stage 2: Critic Agent
  - Build NLI-based verification module
  - Sentence-level claim extraction
  - Automatic flagging/removal of unsupported claims
- [ ] Create audit trail logging (PostgreSQL)
- [ ] Test with 25 queries, measure citation accuracy

**Deliverable:** All generated reports have 90%+ verified citation rate

---

#### **Week 10: Apr 5-11 | Self-Improving RAG Infrastructure**
**Milestone:** *Feedback loop operational, system learns from usage*

**Tasks:**
- [ ] Implement `SelfImprovingRAG` class with boost logic
- [ ] Create feedback collection API:
  - Log cited chunks per query
  - Track user ratings
  - Store query-chunk relevance pairs
- [ ] Build PostgreSQL schema for:
  - Query logs
  - Citation tracking
  - Performance metrics
- [ ] Implement real-time boost factor updates
- [ ] Create analytics dashboard (Jupyter notebook)

**Deliverable:** System tracks and boosts high-performing chunks after 10 test queries

---

#### **Week 11: Apr 12-18 | Streamlit UI Development**
**Milestone:** *Functional web interface with reasoning visualization*

**Tasks:**
- [ ] Build Streamlit app structure:
  - Query input box
  - Query type selector (Q&A / Compare / Full Analysis)
  - Real-time streaming output display
- [ ] Implement Chain-of-Thought visualization:
  - Plan step (selected skills)
  - Execute step (skill outputs)
  - Synthesize step (final report)
- [ ] Create interactive citation modal:
  - Click `[chunk:id]` → show source document
  - Highlight matching sentences
  - Display provenance chain
- [ ] Add charting components (Plotly):
  - Price trends
  - Sentiment timelines

**Deliverable:** Deployed Streamlit app accessible via localhost

---

#### **Week 12: Apr 19-25 | Automated Retraining & Performance Optimization**
**Milestone:** *Weekly retraining pipeline operational*

**Tasks:**
- [ ] Implement weekly retraining workflow:
  - Extract high-rated query-chunk pairs
  - Fine-tune embedding model (sentence-transformers)
  - Reindex Qdrant with updated embeddings
- [ ] Build **Airflow DAG 2**: Weekly model retraining schedule
- [ ] Optimize chunk size/overlap based on citation rates
- [ ] Implement A/B testing framework for retrieval strategies
- [ ] Performance benchmarking suite:
  - Measure citation rate improvement
  - Track query resolution rate
  - Monitor latency

**Deliverable:** System demonstrates measurable improvement after 1 week of usage

---

#### **Week 13: Apr 26-30 | Final Testing, Documentation & Demo Prep**
**Milestone:** *Production-ready system with comprehensive documentation*

**Tasks:**
- [ ] Run 50 diverse test queries:
  - 20 specific questions
  - 15 comparisons
  - 15 full analyses
- [ ] Collect and analyze performance metrics:
  - Average citation rate
  - User satisfaction scores (simulated)
  - Query resolution rate
- [ ] Write comprehensive documentation:
  - Architecture whitepaper (Self-Improving RAG methodology)
  - API documentation
  - User guide
- [ ] Create 5 demo reports (AAPL, TSLA, NVDA, MSFT, GOOGL)
- [ ] Prepare final presentation materials
- [ ] Deploy to cloud (optional: Railway/Render for demo)

**Deliverable:** Complete system + documentation + demo reports ready for final submission

---

### 📋 Progress Tracking

**Key Metrics to Monitor Weekly:**
- [ ] Number of stocks in database
- [ ] Skills implemented (target: 5)
- [ ] Test queries passed
- [ ] Citation accuracy rate
- [ ] Average query latency
- [ ] Code coverage percentage

**Risk Mitigation:**
- **Week 1-2:** If Docker issues arise, use local installations initially
- **Week 5-6:** If LangGraph complexity is high, start with simpler ReAct pattern
- **Week 9:** If NLI verification is slow, use rule-based fallback initially
- **Week 12:** If fine-tuning is unstable, rely on boost factors only

---

## 📊 9. Evaluation Methodology: Measuring Accuracy & Quality

### A. Evaluation Framework Overview

To ensure the system produces reliable, high-quality financial analysis, we implement a **multi-dimensional evaluation framework** that assesses both technical performance and domain-specific accuracy.

---

### B. Quantitative Metrics

#### 1. Factual Accuracy Metrics

**Citation Verification Rate (CVR)**
- **Definition:** Percentage of factual claims that pass Critic Agent verification
- **Target:** ≥ 95%
- **Measurement:**
  ```python
  CVR = (Verified Claims / Total Claims) * 100
  ```
- **Evaluation Process:**
  1. Run 50 test queries across different types
  2. Extract all factual claims from generated reports
  3. Measure Critic Agent pass rate
  4. Manually validate 10% of claims against original sources

**Numerical Accuracy Score (NAS)**
- **Definition:** Precision of numerical data (P/E ratios, revenue figures, margins)
- **Target:** 100% for SQL-based calculations, ≥ 98% for extracted values
- **Measurement:**
  - Compare system outputs against ground truth from official filings
  - Test set: 100 numerical claims across 20 companies
  - Tolerance: ±0.1% for percentages, ±$1M for revenue figures

**Temporal Consistency Score (TCS)**
- **Definition:** Correctness of time-based references (Q3 2024, FY2023, etc.)
- **Target:** ≥ 99%
- **Measurement:** Automated check ensuring cited data matches claimed time period

---

#### 2. Retrieval Quality Metrics

**Retrieval Precision@K**
- **Definition:** Relevance of top-K retrieved chunks to the query
- **Target:** P@5 ≥ 0.85, P@10 ≥ 0.75
- **Measurement:**
  - Human annotators rate chunk relevance (Relevant / Partially Relevant / Irrelevant)
  - Calculate precision for top-5 and top-10 results
  - Test set: 30 diverse queries

**Citation Utilization Rate (CUR)**
- **Definition:** Percentage of retrieved chunks actually cited in final answer
- **Target:** ≥ 60% (indicates efficient retrieval, not over-fetching)
- **Measurement:**
  ```python
  CUR = (Cited Chunks / Retrieved Chunks) * 100
  ```

**Mean Reciprocal Rank (MRR)**
- **Definition:** Average rank position of the first relevant chunk
- **Target:** ≥ 0.80
- **Formula:** 
  \[
  MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
  \]

---

#### 3. System Performance Metrics

**Query Resolution Rate (QRR)**
- **Definition:** Percentage of queries successfully answered (vs. "insufficient data")
- **Target:** ≥ 92%
- **Breakdown by query type:**
  - Specific Questions: ≥ 98%
  - Comparisons: ≥ 90%
  - Full Analyses: ≥ 88%

**Average Response Latency**
- **Target:** 
  - Specific Questions: < 5 seconds
  - Comparisons: < 12 seconds
  - Full Analyses: < 20 seconds
- **Measurement:** End-to-end time from query submission to complete report generation

**Self-Improvement Rate**
- **Definition:** Measurable quality increase over time
- **Target:** +5% citation precision per week for first month
- **Measurement:**
  - Run identical 20-query benchmark at Week 10, 11, 12, 13
  - Track CVR, P@5, CUR improvements

---

### C. Qualitative Evaluation

#### 1. Financial Domain Accuracy Assessment

**Expert Review Panel**
- **Method:** Recruit 2-3 finance professionals (CFA holders or equity analysts)
- **Task:** Blind evaluation of 15 full analysis reports
- **Scoring Criteria (1-5 scale):**
  - **Analytical Depth:** Comprehensiveness of fundamental analysis
  - **Insight Quality:** Value-add beyond raw data retrieval
  - **Risk Identification:** Accuracy in flagging material risks
  - **Valuation Reasonableness:** Sound logic in valuation assessments
  - **Recommendation Validity:** Appropriateness of investment thesis

**Target:** Average score ≥ 4.0/5.0 across all criteria

---

#### 2. Report Quality Dimensions

**Coherence & Readability**
- **Method:** Use automated readability metrics + human evaluation
- **Metrics:**
  - Flesch Reading Ease Score: 50-70 (college-level)
  - Logical flow: No contradictions within report
  - Transition quality: Smooth section connections
- **Measurement:** 10 reports reviewed by 3 independent readers

**Completeness (Full Analysis Reports)**
- **Checklist:**
  - [ ] Company business model explained
  - [ ] At least 5 key financial metrics cited
  - [ ] Valuation vs. peers comparison included
  - [ ] Risk factors identified (min 3)
  - [ ] Clear investment thesis stated
- **Target:** 100% completeness rate

**Citation Quality**
- **Dimensions:**
  - **Density:** 0.5-1.5 citations per sentence (avoid under/over-citing)
  - **Relevance:** Citation directly supports the claim
  - **Recency:** Prefer sources < 6 months old (for news/sentiment)
  - **Authority:** Prioritize official filings > analyst reports > news
- **Measurement:** Manual audit of 50 random citations

---

### D. Comparative Benchmarking

#### Baseline Comparisons

To demonstrate system superiority, we compare against:

**1. Traditional RAG (Without Self-Improvement)**
- Same architecture minus feedback loops
- Comparison metrics: CVR, P@5, QRR
- Expected improvement: +8-12% across metrics

**2. GPT-4o Direct Prompting (No RAG)**
- Pure LLM with system prompt for financial analysis
- Measures hallucination rate difference
- Expected: 3-5x lower hallucination rate with our system

**3. Public Financial AI Tools (if accessible)**
- Bloomberg GPT, FinGPT outputs (where available)
- Focus on citation quality and factual accuracy

---

### E. Test Dataset Construction

#### Diverse Query Set (50 Queries)

**Specific Questions (20 queries)**
- 5 valuation queries ("What is [TICKER]'s P/E ratio?")
- 5 fundamental queries ("What is [TICKER]'s debt-to-equity ratio?")
- 5 sentiment queries ("What's the market sentiment on [TICKER]?")
- 5 risk queries ("What are the key risks for [TICKER]?")

**Comparative Analysis (15 queries)**
- 5 same-sector comparisons ("Compare NVDA vs AMD margins")
- 5 cross-sector comparisons ("Compare AAPL vs TSLA revenue growth")
- 5 metric-focused comparisons ("Which has better FCF: GOOGL or META?")

**Full Analysis (15 queries)**
- 5 large-cap tech (AAPL, MSFT, GOOGL, AMZN, META)
- 5 mid-cap diversified (TSM, BYD, NVDA, AMD, ASML)
- 5 challenging cases (recent IPOs, turnaround stories, controversial stocks)

---

### F. Evaluation Schedule

**Continuous Evaluation (During Development)**
- Week 9: Initial CVR measurement (target: 85%)
- Week 10: First self-improvement test (baseline)
- Week 11: Mid-iteration improvement check
- Week 12: Pre-deployment full evaluation

**Final Evaluation (Week 13)**
1. **Day 1-2:** Run 50-query benchmark, collect all metrics
2. **Day 3:** Expert panel review session
3. **Day 4:** Comparative benchmarking
4. **Day 5:** Compile evaluation report with visualizations

---

### G. Success Criteria Summary

| Metric | Target | Critical? |
|--------|--------|----------|
| Citation Verification Rate | ≥ 95% | ✅ Yes |
| Numerical Accuracy Score | ≥ 98% | ✅ Yes |
| Retrieval Precision@5 | ≥ 0.85 | ✅ Yes |
| Query Resolution Rate | ≥ 92% | ✅ Yes |
| Expert Review Score | ≥ 4.0/5.0 | ✅ Yes |
| Response Latency (Q&A) | < 5s | ❌ No |
| Self-Improvement Rate | +5%/week | ❌ No |
| Citation Utilization Rate | ≥ 60% | ❌ No |

**Passing Criteria:** All "Critical" metrics must meet targets for project success.

---

### H. Evaluation Tools & Infrastructure

**Automated Testing Suite**
```python
# evaluation/benchmark.py
class EvaluationFramework:
    def run_benchmark(self, query_set):
        results = []
        for query in query_set:
            response = self.agent.process(query)
            results.append({
                'query': query,
                'cvr': self.calculate_cvr(response),
                'latency': response.latency,
                'cited_chunks': len(response.citations),
                'critic_pass': response.critic_score
            })
        return self.aggregate_metrics(results)
```

**Human Evaluation Interface**
- Streamlit app for expert reviewers
- Side-by-side comparison of system output vs. ground truth
- Rating forms with 5-point Likert scales
- Citation verification checklist

---

## 📦 10. Deliverables
1.  **UI:** Functional Streamlit web app with three query modes (Q&A, Comparison, Full Analysis).
2.  **Source Code:** Modular Python repo organized by layers:
    - `/ingestion`: Airflow DAGs and ETL scripts
    - `/skills`: Individual agent skills with SKILL.md manifests
    - `/agents`: Supervisor, Summarizer, and Critic implementations
    - `/rag`: Self-improving retrieval logic
    - `/ui`: Streamlit interface
    - `/evaluation`: Benchmarking and evaluation scripts
3.  **Documentation:** 
    - Architecture whitepaper explaining Self-Improving RAG
    - Citation-Verification methodology report
    - Performance benchmarks showing improvement over time
    - **Evaluation Report:** Comprehensive metrics + expert review findings
4.  **Demo Reports:** 5 example outputs covering all three request types with audit trails.
5.  **Evaluation Dataset:** Curated 50-query test set with ground truth annotations.

---

# 📚 Project References & Learning Resources

A curated list of technical resources defining the architecture for **The Agentic Investment Analyst**.

---

## 🛠️ 1. Agent Skills (Modular Capabilities)
*Standardizing how the agent "loads" expertise.*

*   **[Introducing Agent Skills (Claude Blog)](https://www.anthropic.com/news/claude-3-5-sonnet)**  
    *Primary source for the "Skills" design pattern: standardized folders containing `SKILL.md` manifests and executable scripts.*
*   **[In-Depth Analysis of Agent Skills (CNBlogs)](https://www.cnblogs.com/sheng-jie/p/19381647)**  
    *Technical deep-dive on implementing the skills architecture.*
*   **[Agent Skills Guide (ExplainThis)](https://www.explainthis.io/en/ai/agent-skills)**  
    *Clear conceptual overview of why modular skills outperform monolithic prompts.*
*   **[Anthropic Skills Repository (GitHub)](https://github.com/anthropics/anthropic-cookbook/tree/main/skills)**  
    *Official code examples and directory structures to adapt for our project.*

---

## 🔄 2. Self-Improving RAG Systems
*Building RAG that learns from usage.*

*   **[Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511)**  
    *Academic foundation for retrieval-generation-reflection loops.*
*   **[Adaptive RAG with LangChain](https://blog.langchain.dev/adaptive-rag/)**  
    *Practical implementation of query-adaptive retrieval strategies.*
*   **[Building Production RAG Systems (Weaviate)](https://weaviate.io/blog/rag-evaluation)**  
    *Comprehensive guide on RAG evaluation metrics and feedback loops.*

---

## 🧠 3. Orchestration Pattern
*The "Brain" logic: Plan-Execute-Synthesize.*

*   **[LangGraph Tutorial: Plan-and-Execute](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/)**  
    *Step-by-step tutorial on building the specific "Planner → Executor → Replanner" loop.*
*   **[Planning Agents (LangChain Blog)](https://blog.langchain.com/planning-agents/)**  
    *Conceptual justification for why Planning agents are superior for complex, multi-step tasks like financial analysis.*
*   **[Multi-Agent Collaboration Patterns](https://langchain-ai.github.io/langgraph/how-tos/multi-agent/)**  
    *Patterns for coordinating multiple specialized agents.*

---

## 🛡️ 4. Verification & Trust
*Ensuring factual accuracy in LLM outputs.*

*   **[Chain-of-Verification (Meta AI Research)](https://arxiv.org/abs/2309.11495)**  
    *Academic paper on using LLMs to verify their own outputs.*
*   **[Citation Quality in RAG Systems (Anthropic)](https://www.anthropic.com/research/citations)**  
    *Best practices for grounding generation in retrieved sources.*
*   **[Natural Language Inference for Verification](https://huggingface.co/cross-encoder/nli-deberta-v3-base)**  
    *Pre-trained models for verifying claim-evidence alignment.*

---

## 💼 5. Financial Analysis with AI
*Domain-specific resources for equity research.*

*   **[FinGPT: Open-Source Financial LLMs](https://github.com/AI4Finance-Foundation/FinGPT)**  
    *Financial domain adaptation techniques and datasets.*
*   **[Financial Statement Analysis with LLMs (Papers with Code)](https://paperswithcode.com/task/financial-statement-analysis)**  
    *Latest research on automating fundamental analysis.*

---

## 📊 6. RAG Evaluation & Benchmarking
*Methodologies for measuring RAG system quality.*

*   **[RAGAS: RAG Assessment Framework](https://github.com/explodinggradients/ragas)**  
    *Automated metrics for RAG evaluation including faithfulness and answer relevancy.*
*   **[TruLens for LLM Evaluation](https://www.trulens.org/)**  
    *Observability and evaluation tools for LLM applications.*
*   **[LangChain Evaluators](https://python.langchain.com/docs/guides/evaluation/)**  
    *Built-in evaluation tools for assessing agent performance.*