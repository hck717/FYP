# Data Destinations - Complete Reference

This document lists all data types, their sources, and destinations in the FYP pipeline.

---

## Section 1: EODHD API Data

| # | Data Type | Source API Endpoint | Destination | Destination Type | Table/Node |
|---|-----------|---------------------|-------------|------------------|-------------|
| 1 | Company Profiles | `/api/profile/{ticker}` | PostgreSQL | Table | `textual_documents` |
| 2 | Financial News | `/api/news` | PostgreSQL | Table + pgvector | `news_articles` |
| 3 | Insider Transactions | `/api/insider-transactions` | PostgreSQL | Table | `insider_transactions` |
| 4 | Institutional Holders | `/api/institutional-holder` | PostgreSQL | Table | `institutional_holders` |
| 5 | Historical Stock Prices | `/api/eod` | PostgreSQL | Table | `raw_timeseries` |
| 6 | Intraday Quotes | `/api/intraday` | PostgreSQL | Table | `raw_timeseries` |
| 7 | Beta & Volatility | `/api/technical/{ticker}` | PostgreSQL | Table | `raw_timeseries` |
| 8 | Screener API | `/api/screener` | PostgreSQL | Table | `market_screener` |
| 9 | Basic Fundamentals | `/api fundamentals/{ticker}` | PostgreSQL | Table | `raw_fundamentals` |
| 10 | Dividend History | `/api dividends-history` | PostgreSQL | Table | `dividends_history` |
| 11 | Stock Splits | `/api/splits` | PostgreSQL | Table | `splits_history` |
| 12 | Treasury Rates | `/api/ treasury-bill-rates`, `/api/treasury-yield-curve` | PostgreSQL | Table | `treasury_rates` |
| 13 | Economic Events | `/api/economic-events` | PostgreSQL | Table | `economic_events` |
| 14 | Bonds Data | `/api/bonds` | PostgreSQL | Table | `corporate_bond_yields` |
| 15 | Forex Rates | `/api/forex` | PostgreSQL | Table | `forex_rates` |
| 16 | ETF Constituents | `/api/eft/{ticker}` | PostgreSQL | Table | `market_screener` |
| 17 | Financial Calendar | `/api-financial-scalendar` | PostgreSQL | Table | `financial_calendar` |
| 18 | Financial Statements | `/api-income-statement`, `/api-balance-sheet`, `/api-cash-flow` | PostgreSQL | Table | `financial_statements` |
| 19 | Valuation Metrics | `/api/valuation/{ticker}` | PostgreSQL | Table | `valuation_metrics` |
| 20 | Short Interest | `/api/short-interest` | PostgreSQL | Table | `short_interest` |
| 21 | Earnings Surprises | `/api/earnings` | PostgreSQL | Table | `earnings_surprises` |
| 22 | Outstanding Shares | `/api/shares-outstanding` | PostgreSQL | Table | `outstanding_shares` |
| 23 | Analyst Ratings | `/api/rating` | PostgreSQL + pgvector | Table | `text_chunks` |
| 24 | Market EOD US | `/api/eod` | PostgreSQL | Table | `market_eod_us` |

---

## Section 2: Textual Data (PDF Sources)

| # | Data Type | Source | Destination | Destination Type | Table/Node |
|---|-----------|--------|-------------|------------------|-------------|
| 25 | Earnings Call Transcripts | PDF files in `/data/textual data/{TICKER}/earning_call/` | Neo4j | Nodes | `:Chunk` (section: earnings_call) |
| 26 | Broker Reports | PDF files in `/data/textual data/{TICKER}/broker/` | Neo4j | Nodes | `:Chunk` (section: broker_report) |
| 27 | Text Chunks (Company Profile) | Extracted from PDF + EODHD | PostgreSQL | pgvector | `text_chunks` |
| 28 | Textual Documents Metadata | Ingestion script | PostgreSQL | Table | `textual_documents` |

---

## Section 3: Derived/Aggregated Data

| # | Data Type | Source | Destination | Destination Type | Table/Node |
|---|-----------|--------|-------------|------------------|-------------|
| 29 | Sentiment Trends | Derived from `news_articles` aggregation | PostgreSQL | Table | `sentiment_trends` |
| 30 | News Word Weights | Derived from `news_articles` | PostgreSQL | Table | `news_word_weights` |
| 31 | Text Embeddings | Ollama (nomic-embed-text) | PostgreSQL + Neo4j | pgvector | `text_chunks`, `news_articles`, `:Chunk` |

---

## Section 4: Agent Runtime Data

| # | Data Type | Source | Destination | Destination Type | Table/Node |
|---|-----------|--------|-------------|------------------|-------------|
| 32 | Agent Run Telemetry | Orchestration pipeline | PostgreSQL | Table | `agent_run_telemetry` |
| 33 | Episodic Memory | Orchestration pipeline | PostgreSQL | Table | `agent_episodic_memory` |
| 34 | Query Logs | Orchestration pipeline | PostgreSQL | Table | `query_logs` |
| 35 | Critic Run Log | Orchestration pipeline | PostgreSQL | Table | `critic_run_log` |

---

## Section 5: Feedback Data (RLAIF + User)

| # | Data Type | Source | Destination | Destination Type | Table/Node |
|---|-----------|--------|-------------|------------------|-------------|
| 36 | RLAIF Feedback | AI Judge (DeepSeek) | PostgreSQL | Table | `rl_feedback` |
| 37 | User Feedback | Streamlit UI | PostgreSQL | Table | `user_feedback` |
| 38 | Prompt Versions | Orchestration pipeline | PostgreSQL | Table | `prompt_versions` |

---

## Complete Destination Summary

### PostgreSQL Tables

| Table Name | Data Type(s) | Source |
|------------|--------------|--------|
| `raw_timeseries` | Stock prices, intraday, technicals | EODHD API |
| `raw_fundamentals` | EPS, margins, ratios | EODHD API |
| `financial_statements` | Income Statement, Balance Sheet, Cash Flow | EODHD API |
| `valuation_metrics` | P/E, EV/EBITDA, Market Cap | EODHD API |
| `sentiment_trends` | Daily pos/neg/neu from news | Derived |
| `news_articles` | Full article content + embeddings | EODHD API |
| `news_word_weights` | Keyword weights per ticker | Derived |
| `insider_transactions` | SEC Form 4 transactions | EODHD API |
| `institutional_holders` | 13F holder snapshots | EODHD API |
| `short_interest` | Short squeeze metrics | EODHD API |
| `earnings_surprises` | EPS beats/misses | EODHD API |
| `dividends_history` | Dividend history | EODHD API |
| `splits_history` | Stock splits | EODHD API |
| `outstanding_shares` | Shares outstanding history | EODHD API |
| `financial_calendar` | Earnings dates, IPOs, dividends | EODHD API |
| `treasury_rates` | Bill rates + yield curve | EODHD API |
| `forex_rates` | Currency rates | EODHD API |
| `corporate_bond_yields` | Bond yields | EODHD API |
| `economic_events` | FOMC, CPI, NFP | EODHD API |
| `global_macro_indicators` | GDP, CPI, unemployment | EODHD API |
| `market_eod_us` | S&P 500 benchmark | EODHD API |
| `market_screener` | Cross-sectional snapshot | EODHD API |
| `text_chunks` | Company profile chunks + embeddings | EODHD API + Ollama |
| `textual_documents` | PDF metadata | Ingestion script |
| `agent_run_telemetry` | Agent execution logs | Orchestration |
| `agent_episodic_memory` | Query failure patterns | Orchestration |
| `query_logs` | Query history | Orchestration |
| `critic_run_log` | Critic evaluation results | Orchestration |
| `rl_feedback` | AI judge scores | RLAIF scorer |
| `user_feedback` | User ratings/comments | Streamlit UI |
| `prompt_versions` | Prompt version tracking | Orchestration |

### Neo4j Nodes & Relationships

| Node/Relationship | Data Type | Source |
|--------------------|-----------|--------|
| `:Company` | Company profiles | EODHD API |
| `:Chunk` (section: earnings_call) | Earnings call transcripts | PDF files |
| `:Chunk` (section: broker_report) | Broker reports | PDF files |
| `:HAS_CHUNK` | Company → Chunk relationship | Ingestion |

### Vector Stores

| Store | Data Type | Dimension | Source |
|-------|-----------|----------|--------|
| PostgreSQL `text_chunks.embedding` | Company profile text | 768 | Ollama (nomic-embed-text) |
| PostgreSQL `news_articles.embedding` | News articles | 768 | Ollama (nomic-embed-text) |
| Neo4j `:Chunk.embedding` | Earnings calls + broker reports | 768 | Ollama (nomic-embed-text) |

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SOURCES                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  EODHD API                    PDF Files              Orchestration           │
│  ──────────                   ─────────              ──────────────           │
│  • Company Profiles           • Earnings Calls       • Agent telemetry        │
│  • Financial News            • Broker Reports      • Query logs             │
│  • Insider Transactions                             • RLAIF scores          │
│  • Institutional Holders                            • User feedback         │
│  • Stock Prices (EOD/Intraday)                                             │
│  • Technical Indicators                                                    │
│  • Fundamentals              USER INPUT            STORMLIT UI             │
│  • Financial Statements      ──────────            ────────────            │
│  • Valuation Metrics         • User queries        • User feedback         │
│  • Dividends/Splits          • Language preference • Ratings/comments      │
│  • Treasury Rates                                                         │
│  • Economic Events                                                         │
│  • And 15+ more...                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Airflow DAG (dag_eodhd_ingestion_unified.py)                              │
│  ├── scrape_ticker() → etl/agent_data/{TICKER}/                           │
│  ├── load_postgres_for_ticker() → PostgreSQL                              │
│  ├── load_neo4j_for_ticker() → Neo4j                                      │
│  ├── ingest_earnings_calls.py → Neo4j (Chunk nodes)                       │
│  └── ingest_broker_reports.py → Neo4j (Chunk nodes)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
┌──────────────────────────┐ ┌──────────────────┐ ┌──────────────────────────┐
│     POSTGRESQL           │ │      NEO4J       │ │      QDRANT             │
├──────────────────────────┤ ├──────────────────┤ ├──────────────────────────┤
│ 24+ Tables               │ │  :Company nodes  │ │  (not currently used)   │
│ • raw_timeseries         │ │  :Chunk nodes    │ │                          │
│ • financial_statements   │ │  (earnings_call)│ │                          │
│ • valuation_metrics      │ │  (broker_report)│ │                          │
│ • news_articles (+emb)  │ │                  │ │                          │
│ • text_chunks (+emb)    │ │  HAS_CHUNK       │ │                          │
│ • rl_feedback ← NEW     │ │  relationships   │ │                          │
│ • user_feedback ← NEW   │ │                  │ │                          │
│ • prompt_versions ← NEW │ │                  │ │                          │
└──────────────────────────┘ └──────────────────┘ └──────────────────────────┘
                    │                 │                 │
                    ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AGENTS                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Business Analyst    ←──  Neo4j + PostgreSQL (sentiment_trends)            │
│  Quant Fundamental  ←──  PostgreSQL (all financial tables)                 │
│  Financial Modeller ←──  PostgreSQL (financial_statements, valuation)      │
│  Web Search         ←──  Perplexity API (real-time)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  LangGraph StateGraph                                                      │
│  ├── planner → parallel_agents → react_check → summarizer → rlaif_scorer  │
│  └── → translator → memory_update → END                                   │
│                                                                           │
│  Outputs:                                                                 │
│  • final_summary (research note)                                           │
│  • rl_feedback_scores (AI judge)                                          │
│  • Per-agent outputs                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            UI LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Streamlit UI                                                             │
│  ├── Display research note                                                 │
│  ├── Show RLAIF quality scores                                             │
│  └── Collect user feedback (thumbs up/down, comments, issue tags)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Tables by Category

### Core Financial Data
- `raw_timeseries`, `financial_statements`, `valuation_metrics`, `raw_fundamentals`

### Market Data
- `sentiment_trends`, `news_articles`, `news_word_weights`, `market_eod_us`

### Corporate Actions
- `dividends_history`, `splits_history`, `outstanding_shares`

### Ownership
- `insider_transactions`, `institutional_holders`, `short_interest`

### Earnings
- `earnings_surprises`, `financial_calendar`

### Macro
- `treasury_rates`, `forex_rates`, `corporate_bond_yields`, `economic_events`, `global_macro_indicators`, `market_screener`

### Textual
- `text_chunks`, `textual_documents`

### Agent Runtime
- `agent_run_telemetry`, `agent_episodic_memory`, `query_logs`, `critic_run_log`

### Feedback (NEW)
- `rl_feedback`, `user_feedback`, `prompt_versions`

---

## Python Code Examples: How Agents Connect to Data

### 1. PostgreSQL Connection (using `agents/database/connector.py`)

```python
import os
import psycopg2
from psycopg2.extras import RealDictCursor

# Connection setup
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB = os.getenv("POSTGRES_DB", "airflow")
PG_USER = os.getenv("POSTGRES_USER", "airflow")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")

def get_psql_connection():
    """Create a new PostgreSQL connection."""
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD,
    )

# ============================================================
# EXAMPLE: Fetch Financial Statements
# ============================================================

def fetch_financial_statements(ticker: str, period_type: str = "quarterly"):
    """Fetch income statement, balance sheet, cash flow for a ticker."""
    conn = get_psql_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT ticker, statement_type, period_type, period_date, data
                FROM financial_statements
                WHERE ticker = %s AND period_type = %s
                ORDER BY period_date DESC
                LIMIT 10
            """, (ticker, period_type))
            return cur.fetchall()
    finally:
        conn.close()

# Usage
statements = fetch_financial_statements("AAPL", "quarterly")
for row in statements:
    print(f"{row['statement_type']}: {row['period_date']}")

# ============================================================
# EXAMPLE: Fetch Valuation Metrics
# ============================================================

def fetch_valuation_metrics(ticker: str):
    """Fetch P/E, EV/EBITDA, market cap for a ticker."""
    conn = get_psql_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM valuation_metrics
                WHERE ticker = %s
                ORDER BY latest_date DESC
                LIMIT 1
            """, (ticker,))
            return cur.fetchone()
    finally:
        conn.close()

# Usage
valuation = fetch_valuation_metrics("NVDA")
print(f"Trailing P/E: {valuation.get('trailing_pe')}")
print(f"Forward P/E: {valuation.get('forward_pe')}")
print(f"Market Cap: {valuation.get('market_capitalization')}")

# ============================================================
# EXAMPLE: Fetch Sentiment Trends
# ============================================================

def fetch_sentiment(ticker: str):
    """Fetch latest sentiment trends for a ticker."""
    conn = get_psql_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT bullish_pct, bearish_pct, neutral_pct, trend, as_of_date
                FROM sentiment_trends
                WHERE ticker = %s
                ORDER BY as_of_date DESC
                LIMIT 1
            """, (ticker,))
            return cur.fetchone()
    finally:
        conn.close()

# Usage
sentiment = fetch_sentiment("TSLA")
print(f"Bullish: {sentiment['bullish_pct']}%")
print(f"Bearish: {sentiment['bearish_pct']}%")
print(f"Trend: {sentiment['trend']}")

# ============================================================
# EXAMPLE: Semantic Search with pgvector (text_chunks)
# ============================================================

def semantic_search_chunks(query: str, ticker: str, top_k: int = 5):
    """Search company profile chunks using vector similarity."""
    from agents.embedding import get_embedding
    
    # Get query embedding
    query_embedding = get_embedding(query)
    
    conn = get_psql_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT chunk_text, chunk_id, section,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM text_chunks
                WHERE ticker = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, ticker, query_embedding, top_k))
            return cur.fetchall()
    finally:
        conn.close()

# Usage
chunks = semantic_search_chunks("competitive moat and market share", "AAPL")
for chunk in chunks:
    print(f"[{chunk['similarity']:.3f}] {chunk['section']}: {chunk['chunk_text'][:100]}...")

# ============================================================
# EXAMPLE: Fetch News Articles
# ============================================================

def fetch_recent_news(ticker: str, limit: int = 10):
    """Fetch recent news articles for a ticker."""
    conn = get_psql_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT title, content, url, published_at, sentiment
                FROM news_articles
                WHERE ticker = %s
                ORDER BY published_at DESC
                LIMIT %s
            """, (ticker, limit))
            return cur.fetchall()
    finally:
        conn.close()

# Usage
news = fetch_recent_news("MSFT")
for article in news:
    print(f"{article['published_at']}: {article['title']}")
```

---

### 2. Neo4j Connection (using `agents/graph/connector.py`)

```python
import os
from neo4j import GraphDatabase

# Connection setup
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "SecureNeo4jPass2025!")

def get_neo4j_driver():
    """Create a new Neo4j driver."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ============================================================
# EXAMPLE: Fetch Company Profile
# ============================================================

def fetch_company_profile(ticker: str):
    """Fetch company node with all properties."""
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Company {ticker: $ticker})
                RETURN c
            """, ticker=ticker)
            record = result.single()
            if record:
                return dict(record["c"])
            return None
    finally:
        driver.close()

# Usage
company = fetch_company_profile("AAPL")
print(f"Company: {company.get('name')}")
print(f"Sector: {company.get('sector')}")
print(f"Industry: {company.get('industry')}")
print(f"Description: {company.get('description', '')[:200]}...")

# ============================================================
# EXAMPLE: Semantic Search Earnings Calls (RAG)
# ============================================================

def search_earnings_calls(ticker: str, query: str, top_k: int = 5):
    """Search earnings call transcripts using vector similarity."""
    from agents.embedding import get_embedding
    
    query_embedding = get_embedding(query)
    
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Company {ticker: $ticker})-[:HAS_CHUNK]->(ch:Chunk {section: 'earnings_call'})
                WHERE ch.embedding IS NOT NULL
                WITH ch, 
                     1 - (ch.embedding <=> $embedding AS similarity)
                ORDER BY similarity DESC
                LIMIT $top_k
                RETURN ch.text AS text, 
                       ch.chunk_id AS chunk_id, 
                       similarity
            """, ticker=ticker, embedding=query_embedding, top_k=top_k)
            return [dict(record) for record in result]
    finally:
        driver.close()

# Usage
earnings = search_earnings_calls("AAPL", "management guidance revenue forecast")
for chunk in earnings:
    print(f"[{chunk['similarity']:.3f}] {chunk['text'][:200]}...")

# ============================================================
# EXAMPLE: Search Broker Reports (RAG)
# ============================================================

def search_broker_reports(ticker: str, query: str, top_k: int = 5):
    """Search broker report chunks using vector similarity."""
    from agents.embedding import get_embedding
    
    query_embedding = get_embedding(query)
    
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Company {ticker: $ticker})-[:HAS_CHUNK]->(ch:Chunk {section: 'broker_report'})
                WHERE ch.embedding IS NOT NULL
                WITH ch, 
                     1 - (ch.embedding <=> $embedding AS similarity)
                ORDER BY similarity DESC
                LIMIT $top_k
                RETURN ch.text AS text, 
                       ch.chunk_id AS chunk_id, 
                       similarity
            """, ticker=ticker, embedding=query_embedding, top_k=top_k)
            return [dict(record) for record in result]
    finally:
        driver.close()

# Usage
reports = search_broker_reports("NVDA", "AI chip market share competitive advantage")
for report in reports:
    print(f"[{report['similarity']:.3f}] {report['text'][:200]}...")

# ============================================================
# EXAMPLE: Get All Chunks for a Company
# ============================================================

def get_all_chunks(ticker: str):
    """Get all text chunks for a company, grouped by section."""
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Company {ticker: $ticker})-[:HAS_CHUNK]->(ch:Chunk)
                RETURN ch.section AS section, count(ch) AS count
                ORDER BY section
            """, ticker=ticker)
            return [dict(record) for record in result]
    finally:
        driver.close()

# Usage
chunks = get_all_chunks("GOOGL")
for item in chunks:
    print(f"{item['section']}: {item['count']} chunks")

# ============================================================
# EXAMPLE: Vector Search with Filter
# ============================================================

def search_with_filter(ticker: str, query: str, section: str, top_k: int = 3):
    """Search chunks with section filter."""
    from agents.embedding import get_embedding
    
    query_embedding = get_embedding(query)
    
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Company {ticker: $ticker})-[:HAS_CHUNK]->(ch:Chunk {section: $section})
                WHERE ch.embedding IS NOT NULL
                WITH ch, 
                     1 - (ch.embedding <=> $embedding AS similarity)
                ORDER BY similarity DESC
                LIMIT $top_k
                RETURN ch.text AS text, similarity
            """, ticker=ticker, embedding=query_embedding, section=section, top_k=top_k)
            return [dict(record) for record in result]
    finally:
        driver.close()

# Usage
# Only search in earnings calls
results = search_with_filter("TSLA", "battery production capacity", "earnings_call")
```

---

### 3. Using Agent Connectors (Recommended)

```python
# ============================================================
# EXAMPLE: Using Database Connectors (Preferred Method)
# ============================================================

from agents.database.connector import PostgresConnector
from agents.graph.connector import Neo4jConnector

# Initialize connectors
pg = PostgresConnector()
neo4j = Neo4jConnector()

# ============================================================
# PostgreSQL Examples via Connector
# ============================================================

# Get financial statements
statements = pg.fetch_financial_statements("AAPL", period_type="quarterly")

# Get valuation metrics  
valuation = pg.fetch_valuation_metrics("NVDA")

# Get sentiment
sentiment = pg.fetch_sentiment("MSFT")

# Get insider transactions
insiders = pg.fetch_insider_transactions("GOOGL", limit=20)

# Get institutional holders
holders = pg.fetch_institutional_holders("TSLA")

# Get earnings surprises
surprises = pg.fetch_earnings_surprises("AAPL")

# Vector search in text_chunks
chunks = pg.semantic_search("competitive moat", ticker="AAPL", top_k=5)

# ============================================================
# Neo4j Examples via Connector
# ============================================================

# Get company profile
company = neo4j.get_company("AAPL")

# Search earnings calls (RAG)
earnings = neo4j.search_earnings_calls("AAPL", "revenue guidance", top_k=5)

# Search broker reports (RAG)
reports = neo4j.search_broker_reports("NVDA", "market share", top_k=5)

# Get chunk counts by section
chunk_counts = neo4j.get_chunk_counts("GOOGL")

# Get all textual data for a company
all_chunks = neo4j.get_all_text_chunks("TSLA")

# ============================================================
# Combined Example: Full Company Analysis
# ============================================================

def get_full_company_data(ticker: str):
    """Get all relevant data for a company from both databases."""
    
    # PostgreSQL data
    valuation = pg.fetch_valuation_metrics(ticker)
    sentiment = pg.fetch_sentiment(ticker)
    financial_statements = pg.fetch_financial_statements(ticker)
    news = pg.fetch_recent_news(ticker, limit=5)
    
    # Neo4j data
    company = neo4j.get_company(ticker)
    earnings_chunks = neo4j.search_earnings_calls(ticker, "guidance outlook", top_k=3)
    broker_chunks = neo4j.search_broker_reports(ticker, "competitive analysis", top_k=3)
    
    return {
        "valuation": valuation,
        "sentiment": sentiment,
        "financial_statements": financial_statements,
        "news": news,
        "company_profile": company,
        "earnings_calls": earnings_chunks,
        "broker_reports": broker_chunks,
    }

# Usage
data = get_full_company_data("AAPL")
print(f"Company: {data['company_profile']['name']}")
print(f"P/E: {data['valuation']['trailing_pe']}")
print(f"Sentiment: {data['sentiment']['trend']}")
```

---

### 4. Environment Variables Required

```bash
# PostgreSQL
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=airflow
export POSTGRES_USER=airflow
export POSTGRES_PASSWORD=airflow

# Neo4j
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=SecureNeo4jPass2025!

# Ollama (for embeddings)
export OLLAMA_BASE_URL=http://localhost:11434
export EMBEDDING_MODEL=nomic-embed-text
```

---

*Last updated: 2026-03-14*
