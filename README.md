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

## 📅 8. Implementation Roadmap (12 Weeks)

### Phase 1: Infrastructure & Basic Skills (Weeks 1-4)
- [ ] Set up Docker (Qdrant, Airflow, PostgreSQL for feedback logging).
- [ ] Build **DAG 1**: Daily Price/News/Filing ingestion.
- [ ] Implement **Skills 1, 2, 3** (SQL-based fundamentals and valuation).
- [ ] Create basic LangGraph Supervisor for single-skill queries.

### Phase 2: Multi-Agent Orchestration (Weeks 5-8)
- [ ] Implement **Skills 4 & 5** (Sentiment and Risk).
- [ ] Build **Plan-Execute-Synthesize** workflow in LangGraph.
- [ ] Develop **Summarizer Agent** with three report templates.
- [ ] Implement request type classification (Q&A / Comparison / Full Analysis).

### Phase 3: Self-Improvement & Trust (Weeks 9-12)
- [ ] Implement **Citation-Verification Loop** with Critic Agent.
- [ ] Build **Self-Improving RAG** feedback collection and boosting logic.
- [ ] Develop **Streamlit UI** with reasoning visualization and interactive citations.
- [ ] Implement automated retraining pipeline for weekly embedding updates.
- [ ] Final Testing: Run 50 diverse queries, collect feedback, measure improvement.

---

## 📦 9. Deliverables
1.  **UI:** Functional Streamlit web app with three query modes (Q&A, Comparison, Full Analysis).
2.  **Source Code:** Modular Python repo organized by layers:
    - `/ingestion`: Airflow DAGs and ETL scripts
    - `/skills`: Individual agent skills with SKILL.md manifests
    - `/agents`: Supervisor, Summarizer, and Critic implementations
    - `/rag`: Self-improving retrieval logic
    - `/ui`: Streamlit interface
3.  **Documentation:** 
    - Architecture whitepaper explaining Self-Improving RAG
    - Citation-Verification methodology report
    - Performance benchmarks showing improvement over time
4.  **Demo Reports:** 5 example outputs covering all three request types with audit trails.

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