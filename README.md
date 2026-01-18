# 🎓 The Agentic Investment Analyst
### *An Autonomous, Graph-Augmented Reasoning Engine for Fundamental Equity Analysis*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange)
![Neo4j](https://img.shields.io/badge/Neo4j-GraphDB-blue)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 1. Abstract
Current AI solutions in finance often suffer from **"black box" opacity** and **hallucinations**. While Large Language Models (LLMs) can summarize text, they lack the structural understanding to identify hidden market risks (e.g., supplier dependencies) or the rigor to justify their claims.

**** is an autonomous agent designed to mimic a human analyst’s workflow. Unlike standard retrieval systems, it utilizes a **GraphRAG (Graph Retrieval-Augmented Generation)** architecture to map complex market relationships and an **Iterative Chain-of-Thought (CoT)** engine to "plan, execute, and verify" investment theses. The system produces transparent, evidence-backed reports with pixel-perfect citations, enabling users to audit the AI's reasoning trace.

---

## 🎯 2. Project Objectives
*   **Solve the "Hidden Risk" Problem:** Use **GraphRAG (Neo4j)** to detect indirect relationships (e.g., *Supplier Fire* $\rightarrow$ *Stock Drop*) that standard vector search misses.
*   **Eliminate the "Black Box":** Implement a **visible Chain-of-Thought** reasoning loop, allowing users to see *how* the agent reached a conclusion.
*   **Zero-Tolerance for Hallucination:** Enforce a strict **"Citation-Verification" protocol** where every claim must be backed by a retrieved document snippet.
*   **Hybrid Architecture:** Demonstrate a cost-effective **"Cloud Supervisor / Local Worker"** model, using **GPT-4o** for high-level reasoning and **Llama 3.2 (3B)** for massive data processing.

---

## 🛠️ 3. Methodology

### A. High-Level Architecture
The platform follows a **Plan-and-Execute** agent pattern, decoupled into two layers to balance cost and intelligence.

#### 1. Ingestion Layer (The "Worker")
*   **Role:** Runs offline on a schedule (e.g., nightly) via **Airflow**.
*   **Task:** Ingests raw data (News, Filings, Tweets), processes it using a lightweight local LLM (**Llama 3.2**), and populates specialized databases.
*   **Tech Stack:**
    *   **Airflow:** Orchestration.
    *   **SQLite:** Structured numerical data (Prices, Ratios).
    *   **Qdrant:** Semantic vector embeddings.
    *   **Neo4j:** Knowledge Graph of entity relationships.

#### 2. Inference Layer (The "Supervisor")
*   **Role:** Runs in real-time upon user query.
*   **Task:** Uses a high-intelligence Cloud LLM (**GPT-4o**) to understand intent, plan tool calls, and synthesize reports.

### B. Architectural Uniqueness
1.  **GraphRAG:** Constructs a Knowledge Graph to find second-order risks (e.g., supplier delays) that do not explicitly mention the target company.
2.  **Agentic Skills:** Encapsulates logic into deterministic tools (e.g., Python functions for P/E ratio), ensuring mathematical accuracy.
3.  **Visible Chain-of-Thought:** Displays the Agent's reasoning trace (`Plan` $\rightarrow$ `Action` $\rightarrow$ `Observation`) in the UI.

### C. User Interface (Streamlit)
*   **Real-time Streaming:** Visualizes the Agent's thought process.
*   **Interactive Charts:** Price and Sentiment visualization.
*   **Evidence Linking:** Side-by-side display of the final report and retrieved citations.

### D. The Ingestion Logic: How Llama 3.2 Builds the Graph
We utilize an automated **"LLM-ETL"** pipeline:
1.  **Input:** Raw news article (e.g., *"TSMC announced a delay in Arizona..."*).
2.  **Extraction:** **Llama 3.2 (3B)** extracts entities via a strict JSON prompt.
3.  **Transformation:** Python script performs Entity Resolution (e.g., "Apple Inc." $\rightarrow$ `AAPL`).
4.  **Loading:** Generates **Cypher** queries to insert nodes/edges into Neo4j.

---

## 🛡️ 4. Hallucination Control
We implement a **Citation-Verification Loop** to ensure trust:
*   **Grounding:** The Supervisor is constrained to generate claims *only* supported by retrieved context.
*   **The "Critic" Agent:** A secondary model audits every sentence against the source text. Unsupported claims are flagged/removed.
*   **UI Linking:** All citations are clickable hyperlinks, displaying the raw source text for instant verification.

---

## ⚙️ 5. Technical Specifications

### A. Data Sources & Budget (~$300 USD)
| Data Type | Provider | Cost (4 Mo) | Notes |
| :--- | :--- | :--- | :--- |
| **Core (Price/Fund/News)** | EODHD | $200 | Backbone of the system. |
| **Macro Economics** | EODHD / FRED | $0 | GDP, Inflation, Rates. |
| **Volatility / Risk** | yfinance | $0 | VIX (Fear Index). |
| **Social Sentiment** | Reddit (PRAW) | $0 | "Personal Use" script. |
| **Sentiment Model** | Hugging Face | $0 | Local FinTwitBERT. |
| **Vector/Graph DB** | Qdrant / Neo4j | $0 | Self-Hosted Docker. |
| **LLM Buffer** | OpenAI/Anthropic| ~$100 | Supervisor tokens. |

### B. AI Strategy: The "Hybrid Brain"
| Component | Model | Role | Justification |
| :--- | :--- | :--- | :--- |
| **Worker** | **Llama 3.2 (3B)** | Ingestion (Offline) | High volume, zero cost. |
| **Supervisor** | **GPT-4o** | Planner (Runtime) | Complex reasoning, high quality. |

### C. The 5 Core Agentic Skills
| Skill | Purpose | Data Source | Logic / Output |
| :--- | :--- | :--- | :--- |
| **1. Fundamental Health** | Quality Check | SQLite | SQL: `ROE > 15%`, `Debt/EBITDA < 3x`. |
| **2. Price & Momentum** | Timing | SQLite | `Price vs 200SMA` + `RSI`. |
| **3. Valuation Reality** | Pricing | SQLite | `P/E` vs Peers/History. |
| **4. Graph Catalyst** | **The Alpha** | Neo4j | Traversal: Find 2-hop hidden risks. |
| **5. Hedge Optimizer** | Protection | SQLite | Math: `Beta` & `Correlation` matrices. |

---

## 📅 6. Implementation Roadmap (12 Weeks)

### Phase 1: Infrastructure (Weeks 1-4)
- [ ] Set up Docker (Neo4j, Qdrant, Airflow).
- [ ] Build **DAG 1**: Daily Price/News ingestion.
- [ ] Implement **Skills 1 & 2** (SQL-based fundamentals).

### Phase 2: The "Brain" (Weeks 5-8)
- [ ] Build **DAG 2 (Graph Ingestion)**: Llama 3.2 entity extraction.
- [ ] Implement **Skill 4 (GraphRAG)**: Cypher queries for traversal.
- [ ] Develop **LangGraph Supervisor** logic.

### Phase 3: Trust & UI (Weeks 9-12)
- [ ] Implement **"Critic" Agent** for verification.
- [ ] Build **Streamlit UI** (Reasoning Log + Final Report).
- [ ] Final Testing & Documentation.

---

## 📦 7. Deliverables
1.  ** Dashboard:** Functional Streamlit web app.
2.  **Source Code:** Modular Python repo (Ingestion, Agent, UI).
3.  **Final Thesis:** Report on GraphRAG & Citation-Verification methodology.
4.  **Investment Theses:** 2-4 distinct, AI-generated reports.

---
*Created by Brian - Final Year Project 2026*
