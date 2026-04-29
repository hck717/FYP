# Data Check Commands - Complete Reference

This file contains all commands to check data status in PostgreSQL and Neo4j databases.

---

## Section 1: PostgreSQL Tables (All Data Types)

### Core Financial Tables

```bash
# 1. Raw Timeseries (OHLCV prices)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT ticker_symbol, COUNT(*) as cnt, MIN(date) as earliest, MAX(date) as latest
  FROM raw_timeseries
  GROUP BY ticker_symbol ORDER BY ticker_symbol;
"

# 2. Financial Statements (Income Statement, Balance Sheet, Cash Flow)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT ticker, COUNT(*) as cnt, period_type
  FROM financial_statements
  GROUP BY ticker, period_type ORDER BY ticker;
"

# 3. Valuation Metrics (P/E, EV/EBITDA, Market Cap)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT ticker, COUNT(*) as cnt, latest_date
  FROM valuation_metrics
  GROUP BY ticker ORDER BY ticker;
"

# 4. Raw Fundamentals (EPS, margins, ratios)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT ticker_symbol, data_name, COUNT(*) as cnt
  FROM raw_fundamentals
  GROUP BY ticker_symbol, data_name
  ORDER BY ticker_symbol, data_name;
"
```

### Market Data Tables

```bash
# 5. Sentiment Trends (Daily pos/neg/neu aggregated from news)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT ticker, COUNT(*) as days, MIN(as_of_date) as earliest, MAX(as_of_date) as latest
  FROM sentiment_trends
  GROUP BY ticker ORDER BY ticker;
"

# 6. News Articles (Full content + embeddings)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT ticker,
         COUNT(*) AS total,
         SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) AS embedded
  FROM news_articles
  GROUP BY ticker ORDER BY ticker;
"

# 7. News Word Weights (Keyword weights per ticker)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT ticker, COUNT(*) as cnt
  FROM news_word_weights
  GROUP BY ticker ORDER BY ticker;
"

# 8. Market EOD US (S&P 500 benchmark)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT COUNT(*) as total, MIN(date) as earliest, MAX(date) as latest
  FROM market_eod_us;
"
```

### Corporate Actions Tables

```bash
# 9. Dividends History
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT ticker, COUNT(*) as cnt, MIN(ex_dividend_date) as earliest
  FROM dividends_history
  GROUP BY ticker ORDER BY ticker;
"

# 10. Splits History
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT ticker, COUNT(*) as cnt
  FROM splits_history
  GROUP BY ticker ORDER BY ticker;
"

# 11. Outstanding Shares History
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT ticker, COUNT(*) as cnt, MIN(date) as earliest, MAX(date) as latest
  FROM outstanding_shares
  GROUP BY ticker ORDER BY ticker;
"
```

### Ownership & Transactions Tables

```bash
# 12. Insider Transactions (SEC Form 4)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT ticker, COUNT(*) as transactions, COUNT(DISTINCT owner_name) as insiders
  FROM insider_transactions
  GROUP BY ticker ORDER BY ticker;
"

# 13. Institutional Holders (13F snapshots)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT ticker, COUNT(*) as holders, MAX(period_end) as latest
  FROM institutional_holders
  GROUP BY ticker ORDER BY ticker;
"

# 14. Short Interest (Short squeeze potential)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT ticker, COUNT(*) as records, MAX(settlement_date) as latest
  FROM short_interest
  GROUP BY ticker ORDER BY ticker;
"
```

### Earnings & Estimates Tables

```bash
# 15. Earnings Surprises (EPS beats/misses)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT ticker, COUNT(*) as quarters, 
         SUM(CASE WHEN surprise_percent > 0 THEN 1 ELSE 0 END) as beats,
         SUM(CASE WHEN surprise_percent < 0 THEN 1 ELSE 0 END) as misses
  FROM earnings_surprises
  GROUP BY ticker ORDER BY ticker;
"

# 16. Financial Calendar (Earnings dates, IPOs, dividends)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT event_type, COUNT(*) as events, MIN(event_date) as earliest, MAX(event_date) as latest
  FROM financial_calendar
  GROUP BY event_type ORDER BY event_type;
"
```

### Macro Data Tables

```bash
# 17. Treasury Rates (Bill rates + yield curve)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT COUNT(*) as rates, MIN(rate_date) as earliest, MAX(rate_date) as latest
  FROM treasury_rates;
"

# 18. Forex Rates (EUR/USD and other pairs)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT currency_pair, COUNT(*) as rates
  FROM forex_rates
  GROUP BY currency_pair ORDER BY currency_pair;
"

# 19. Corporate Bond Yields (LQD/HYG proxy)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT COUNT(*) as bonds, MIN(maturity_date) as earliest
  FROM corporate_bond_yields;
"

# 20. Economic Events (FOMC, CPI, NFP)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT event_name, COUNT(*) as events, COUNT(DISTINCT country_code) as countries
  FROM economic_events
  GROUP BY event_name ORDER BY event_name;
"

# 21. Global Macro Indicators
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT indicator_name, COUNT(*) as records, MIN(date) as earliest, MAX(date) as latest
  FROM global_macro_indicators
  GROUP BY indicator_name ORDER BY indicator_name;
"

# 22. Market Screener (Bulk cross-sectional snapshot)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT COUNT(*) as companies, MAX(as_of_date) as latest
  FROM market_screener;
"
```

### Textual Data Tables

```bash
# 23. Text Chunks (company profile + textual PDFs with embeddings)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT ticker, COUNT(*) as chunks,
         SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) as embedded
  FROM text_chunks
  GROUP BY ticker ORDER BY ticker;
"

# 24. Textual Documents Metadata (PDF earnings calls, broker reports, macro reports)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT ticker, doc_type, COUNT(*) as docs
  FROM textual_documents
  GROUP BY ticker, doc_type ORDER BY ticker, doc_type;
"

# 24a. Textual metadata coverage by doc_type (must all be > 0)
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT doc_type, COUNT(*) as docs
  FROM textual_documents
  WHERE doc_type IN ('earnings_call', 'broker_report', 'macro_report')
  GROUP BY doc_type
  ORDER BY doc_type;
"
```

### pgvector Extension Check

```bash
# Check pgvector extension
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT extversion FROM pg_extension WHERE extname = 'vector';
"
```

---

## Section 2: Feedback Tables (RLAIF + User Feedback)

### RLAIF Feedback (AI-generated scores)

```bash
# 25. RLAIF Feedback - Overall counts
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT COUNT(*) as total,
         AVG(overall_score) as avg_score,
         MIN(overall_score) as min_score,
         MAX(overall_score) as max_score
  FROM rl_feedback;
"

# 25a. RLAIF Feedback - Scores by agent blamed
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT agent_blamed, COUNT(*) as cnt, AVG(overall_score) as avg_score
  FROM rl_feedback
  WHERE agent_blamed IS NOT NULL
  GROUP BY agent_blamed
  ORDER BY cnt DESC;
"

# 25b. RLAIF Feedback - Recent low-scoring reports
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT run_id, ticker, overall_score, agent_blamed, timestamp
  FROM rl_feedback
  WHERE overall_score < 7.0
  ORDER BY timestamp DESC
  LIMIT 20;
"

# 25c. RLAIF Feedback - Score breakdown by dimension
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT 
    AVG(factual_accuracy) as avg_factual,
    AVG(citation_completeness) as avg_citation,
    AVG(analysis_depth) as avg_analysis,
    AVG(structure_compliance) as avg_structure,
    AVG(language_quality) as avg_language
  FROM rl_feedback;
"
```

### User Feedback (Explicit ratings from UI)

```bash
# 26. User Feedback - Overall counts
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT COUNT(*) as total,
         SUM(CASE WHEN helpful = true THEN 1 ELSE 0 END) as positive,
         SUM(CASE WHEN helpful = false THEN 1 ELSE 0 END) as negative
  FROM user_feedback;
"

# 26a. User Feedback - Issue tag distribution
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT issue_tag, COUNT(*) as cnt
  FROM user_feedback, jsonb_array_elements_text(issue_tags) as issue_tag
  GROUP BY issue_tag
  ORDER BY cnt DESC;
"

# 26b. User Feedback - Recent feedback
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT run_id, helpful, comment, timestamp
  FROM user_feedback
  ORDER BY timestamp DESC
  LIMIT 20;
"
```

### Prompt Versions (A/B testing)

```bash
# 27. Prompt Versions - Tracking prompt changes
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT agent_name, version, deployed_at, deployed_to, avg_score_before, avg_score_after
  FROM prompt_versions
  ORDER BY agent_name, deployed_at DESC;
"
```

---

## Section 3: Neo4j Database

### Company and Chunk Nodes

```bash
# Check Company nodes
docker exec fyp-neo4j cypher-shell -u neo4j -p SecureNeo4jPass2025! "
  MATCH (c:Company) RETURN c.ticker as ticker, count(*) as chunks
  ORDER BY ticker;
"

# Check Chunk nodes with embeddings
docker exec fyp-neo4j cypher-shell -u neo4j -p SecureNeo4jPass2025! "
  MATCH (ch:Chunk) 
  RETURN count(ch) as total,
         count(CASE WHEN ch.embedding IS NOT NULL THEN 1 END) as embedded;
"

# Check embedding dimension
docker exec fyp-neo4j cypher-shell -u neo4j -p SecureNeo4jPass2025! "
  MATCH (ch:Chunk) WHERE ch.embedding IS NOT NULL 
  RETURN size(ch.embedding) as dimension LIMIT 1;
"

# Check Chunk by section
docker exec fyp-neo4j cypher-shell -u neo4j -p SecureNeo4jPass2025! "
  MATCH (c:Company)-[:HAS_CHUNK]->(ch:Chunk)
  RETURN c.ticker as ticker, ch.section as section, count(ch) as cnt
  ORDER BY ticker, section;
"

# Check financial news mirrored into Neo4j
docker exec fyp-neo4j cypher-shell -u neo4j -p SecureNeo4jPass2025! "
  MATCH (n:NewsArticle)
  RETURN count(n) as news_nodes;
"

# Check HAS_NEWS relationships
docker exec fyp-neo4j cypher-shell -u neo4j -p SecureNeo4jPass2025! "
  MATCH (:Company)-[r:HAS_NEWS]->(:NewsArticle)
  RETURN count(r) as has_news_rels;
"

# Check vector index status
docker exec fyp-neo4j cypher-shell -u neo4j -p SecureNeo4jPass2025! "
  SHOW INDEXES YIELD name, state, type, labelsOrTypes, properties, options
  WHERE name = 'chunk_embedding';
"
```

---

## Section 4: Complete Database Health Check (All-in-One)

### Run Full Inspection (Recommended)

```bash
# Inside container
docker exec fyp-airflow-webserver python /opt/airflow/ingestion/etl/inspect_db.py

# Or from host
python ingestion/etl/inspect_db.py
```

This command checks:
- All PostgreSQL tables (row counts, coverage)
- pgvector extension and text_chunks
- Macro data tables (`treasury_rates`, `global_macro_indicators`, `economic_events`, `market_screener`, `forex_rates`, `market_eod_us`)
- `textual_documents` coverage by `doc_type` (`earnings_call`, `broker_report`, `macro_report`)
- All feedback tables (rl_feedback, user_feedback, prompt_versions)
- Neo4j nodes and embeddings (`:Chunk`, `:NewsArticle`)
- Vector index status

---

## Section 5: Agent Run Telemetry

```bash
# Check agent run telemetry
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT agent_name, event_type, COUNT(*) as cnt
  FROM agent_run_telemetry
  WHERE recorded_at > NOW() - INTERVAL '24 hours'
  GROUP BY agent_name, event_type
  ORDER BY agent_name, cnt DESC;
"

# Check episodic memory
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT failure_agent, failure_reason, COUNT(*) as cnt
  FROM agent_episodic_memory
  GROUP BY failure_agent, failure_reason
  ORDER BY cnt DESC;
"
```

---

## Quick Summary Commands

### Row Count Summary (All Tables)

```bash
docker exec fyp-postgres psql -U airflow -d airflow -c "
  SELECT 'raw_timeseries' as table_name, COUNT(*) as cnt FROM raw_timeseries
  UNION ALL SELECT 'financial_statements', COUNT(*) FROM financial_statements
  UNION ALL SELECT 'valuation_metrics', COUNT(*) FROM valuation_metrics
  UNION ALL SELECT 'sentiment_trends', COUNT(*) FROM sentiment_trends
  UNION ALL SELECT 'news_articles', COUNT(*) FROM news_articles
  UNION ALL SELECT 'dividends_history', COUNT(*) FROM dividends_history
  UNION ALL SELECT 'splits_history', COUNT(*) FROM splits_history
  UNION ALL SELECT 'outstanding_shares', COUNT(*) FROM outstanding_shares
  UNION ALL SELECT 'insider_transactions', COUNT(*) FROM insider_transactions
  UNION ALL SELECT 'institutional_holders', COUNT(*) FROM institutional_holders
  UNION ALL SELECT 'short_interest', COUNT(*) FROM short_interest
  UNION ALL SELECT 'earnings_surprises', COUNT(*) FROM earnings_surprises
  UNION ALL SELECT 'financial_calendar', COUNT(*) FROM financial_calendar
  UNION ALL SELECT 'treasury_rates', COUNT(*) FROM treasury_rates
  UNION ALL SELECT 'forex_rates', COUNT(*) FROM forex_rates
  UNION ALL SELECT 'corporate_bond_yields', COUNT(*) FROM corporate_bond_yields
  UNION ALL SELECT 'economic_events', COUNT(*) FROM economic_events
  UNION ALL SELECT 'text_chunks', COUNT(*) FROM text_chunks
  UNION ALL SELECT 'textual_documents', COUNT(*) FROM textual_documents
  UNION ALL SELECT 'rl_feedback', COUNT(*) FROM rl_feedback
  UNION ALL SELECT 'user_feedback', COUNT(*) FROM user_feedback
  UNION ALL SELECT 'prompt_versions', COUNT(*) FROM prompt_versions
  ORDER BY cnt DESC;
"
```

---

## Data Types Reference

| # | Data Type | Table(s) | Source |
|---|-----------|----------|--------|
| 1 | Company Profiles / Tickers | textual_documents | EODHD |
| 2 | Financial News | news_articles | EODHD |
| 3 | Insider Transactions | insider_transactions | EODHD |
| 4 | Institutional Holders | institutional_holders | EODHD |
| 5 | Historical Stock Prices | raw_timeseries | EODHD |
| 6 | Intraday Quotes | raw_timeseries | EODHD |
| 7 | Beta & Volatility | raw_timeseries | EODHD |
| 8 | Screener API | market_screener | EODHD |
| 9 | Basic Fundamentals | raw_fundamentals | EODHD |
| 10 | Dividend History | dividends_history | EODHD |
| 11 | Treasury Rates | treasury_rates | EODHD |
| 12 | Economic Events | economic_events | EODHD |
| 13 | Bonds Data | corporate_bond_yields | EODHD |
| 14 | Forex Rates | forex_rates | EODHD |
| 15 | ETF Constituents | market_screener | EODHD |
| 16 | Financial Calendar | financial_calendar | EODHD |
| 17 | Financial Statements | financial_statements | EODHD |
| 18 | Valuation Metrics | valuation_metrics | EODHD |
| 19 | Short Interest | short_interest | EODHD |
| 20 | Earnings Surprises | earnings_surprises | EODHD |
| 21 | Outstanding Shares | outstanding_shares | EODHD |
| 22 | Analyst Ratings | textual_documents | EODHD |
| 23 | Broker Reports | text_chunks + textual_documents + Neo4j Chunks | PDF |
| 24 | Earnings Call Transcripts | text_chunks + textual_documents + Neo4j Chunks | PDF |
| 25 | Macro Reports | text_chunks + textual_documents + Neo4j Chunks | PDF |
| 25 | **RLAIF Feedback** | rl_feedback | AI Judge |
| 26 | **User Feedback** | user_feedback | UI |
| 27 | **Prompt Versions** | prompt_versions | A/B Testing |
