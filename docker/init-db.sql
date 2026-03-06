-- init-db.sql
-- Runs once at PostgreSQL container first-start
-- (mounted at /docker-entrypoint-initdb.d/init-db.sql)
-- Creates all application tables so they exist before Airflow DAGs run.

-- Enable pgvector extension (requires pgvector/pgvector:pg15 image)
CREATE EXTENSION IF NOT EXISTS vector;

-- Raw time-series data (price, volume, intraday, etc.)
CREATE TABLE IF NOT EXISTS raw_timeseries (
    id            SERIAL PRIMARY KEY,
    agent_name    TEXT      NOT NULL,
    ticker_symbol TEXT      NOT NULL,
    data_name     TEXT      NOT NULL,
    ts_date       TIMESTAMP,
    payload       JSONB     NOT NULL,
    source        TEXT      NOT NULL,
    ingested_at   TIMESTAMP DEFAULT NOW(),
    UNIQUE (agent_name, ticker_symbol, data_name, ts_date, source)
);

-- Point-in-time fundamentals (balance sheet, income statement, etc.)
CREATE TABLE IF NOT EXISTS raw_fundamentals (
    id            SERIAL PRIMARY KEY,
    agent_name    TEXT      NOT NULL,
    ticker_symbol TEXT      NOT NULL,
    data_name     TEXT      NOT NULL,
    as_of_date    DATE      NOT NULL,
    payload       JSONB     NOT NULL,
    source        TEXT      NOT NULL,
    ingested_at   TIMESTAMP DEFAULT NOW(),
    UNIQUE (agent_name, ticker_symbol, data_name, as_of_date, source)
);

-- Global US end-of-day prices (stored once per day, not per ticker)
CREATE TABLE IF NOT EXISTS market_eod_us (
    id          SERIAL PRIMARY KEY,
    ts_date     TIMESTAMP NOT NULL,
    payload     JSONB     NOT NULL,
    source      TEXT      NOT NULL,
    ingested_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (ts_date, source)
);

-- Macro / economic calendar events
CREATE TABLE IF NOT EXISTS global_economic_calendar (
    id          SERIAL PRIMARY KEY,
    ts_date     TIMESTAMP,
    payload     JSONB     NOT NULL,
    source      TEXT      NOT NULL,
    ingested_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (ts_date, source)
);

-- IPO calendar events
CREATE TABLE IF NOT EXISTS global_ipo_calendar (
    id          SERIAL PRIMARY KEY,
    ts_date     TIMESTAMP,
    payload     JSONB     NOT NULL,
    source      TEXT      NOT NULL,
    ingested_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (ts_date, source)
);

-- Sentiment trends per ticker (queried by Business Analyst agent)
-- bullish_pct / bearish_pct / neutral_pct are 0–100 percentages
-- trend is one of: improving | deteriorating | stable | unknown
CREATE TABLE IF NOT EXISTS sentiment_trends (
    id           SERIAL PRIMARY KEY,
    ticker       VARCHAR(10)  NOT NULL,
    bullish_pct  NUMERIC(6, 4),
    bearish_pct  NUMERIC(6, 4),
    neutral_pct  NUMERIC(6, 4),
    trend        VARCHAR(20)  DEFAULT 'unknown',
    as_of_date   DATE         NOT NULL,
    ingested_at  TIMESTAMP    DEFAULT NOW(),
    UNIQUE (ticker, as_of_date)
);

-- Index for fast per-ticker sentiment lookups
CREATE INDEX IF NOT EXISTS idx_sentiment_trends_ticker_date
    ON sentiment_trends (ticker, as_of_date DESC);

-- Episodic memory: records past query failures so the planner can pre-empt
-- known failure patterns on future runs (see orchestration/episodic_memory.py)
CREATE TABLE IF NOT EXISTS agent_episodic_memory (
    id                    SERIAL PRIMARY KEY,
    query_signature       TEXT        NOT NULL,
    ticker                TEXT        NOT NULL,
    failure_agent         TEXT        NOT NULL,
    failure_reason        TEXT        NOT NULL DEFAULT 'UNKNOWN',
    react_iterations_used INT         NOT NULL DEFAULT 1,
    query_embedding       JSONB,
    recorded_at           TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_episodic_memory_ticker
    ON agent_episodic_memory (ticker);

CREATE INDEX IF NOT EXISTS idx_episodic_memory_agent
    ON agent_episodic_memory (failure_agent);

-- ---------------------------------------------------------------------------
-- 3A: Materialized view — pre-computed daily factor scores per ticker
--     Refreshed by a nightly cron/Airflow job via:
--         REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_factor_scores;
--     The view pulls the most recent raw_fundamentals payload for each ticker
--     and JSON-extracts the pre-computed financial-score fields written by the
--     FMP ingestion DAG (piotroskiScore, altmanZScore, beneishMScore, etc.)
-- ---------------------------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_factor_scores AS
SELECT
    rf.ticker_symbol                                              AS ticker,
    MAX(rf.as_of_date)                                            AS as_of_date,
    MAX((rf.payload ->> 'piotroskiScore')::NUMERIC)               AS piotroski_score,
    MAX((rf.payload ->> 'altmanZScore')::NUMERIC)                 AS altman_z_score,
    MAX((rf.payload ->> 'beneishMScore')::NUMERIC)                AS beneish_m_score,
    MAX((rf.payload ->> 'returnOnEquityTTM')::NUMERIC)            AS roe_ttm,
    MAX((rf.payload ->> 'returnOnAssetsTTM')::NUMERIC)            AS roa_ttm,
    MAX((rf.payload ->> 'returnOnInvestedCapitalTTM')::NUMERIC)   AS roic_ttm,
    MAX((rf.payload ->> 'grossProfitMarginTTM')::NUMERIC)         AS gross_margin_ttm,
    MAX((rf.payload ->> 'netProfitMarginTTM')::NUMERIC)           AS net_margin_ttm,
    MAX((rf.payload ->> 'debtToEquityRatioTTM')::NUMERIC)         AS debt_to_equity_ttm,
    MAX((rf.payload ->> 'currentRatioTTM')::NUMERIC)              AS current_ratio_ttm,
    NOW()                                                         AS refreshed_at
FROM raw_fundamentals rf
WHERE rf.data_name IN ('financial_scores', 'key_metrics_ttm', 'ratios_ttm')
GROUP BY rf.ticker_symbol
WITH NO DATA;

-- Unique index required for CONCURRENTLY refresh
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_daily_factor_scores_ticker
    ON mv_daily_factor_scores (ticker);

CREATE INDEX IF NOT EXISTS idx_mv_daily_factor_scores_date
    ON mv_daily_factor_scores (as_of_date DESC);

-- ---------------------------------------------------------------------------
-- Feedback loop tables (A1, A2)
-- ---------------------------------------------------------------------------

-- User explicit feedback (thumbs up/down + text) from Streamlit UI
CREATE TABLE IF NOT EXISTS citation_tracking (
    id              SERIAL PRIMARY KEY,
    run_id          TEXT        NOT NULL,
    ticker          TEXT,
    query_text      TEXT,
    chunk_id        TEXT,
    agent_name      TEXT,
    was_cited       BOOLEAN     DEFAULT FALSE,
    feedback_score  SMALLINT    CHECK (feedback_score IN (-1, 0, 1)),
    feedback_text   TEXT,
    recorded_at     TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_citation_tracking_chunk
    ON citation_tracking (chunk_id);
CREATE INDEX IF NOT EXISTS idx_citation_tracking_run
    ON citation_tracking (run_id);

-- Per-run query logs (explicit feedback + metadata from UI)
CREATE TABLE IF NOT EXISTS query_logs (
    id              SERIAL PRIMARY KEY,
    run_id          TEXT        NOT NULL UNIQUE,
    ticker          TEXT,
    query_text      TEXT,
    overall_rating  SMALLINT    CHECK (overall_rating IN (-1, 0, 1)),
    feedback_text   TEXT,
    complexity      SMALLINT,
    agents_used     JSONB,
    final_note      TEXT,
    recorded_at     TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_logs_ticker
    ON query_logs (ticker);
CREATE INDEX IF NOT EXISTS idx_query_logs_recorded_at
    ON query_logs (recorded_at DESC);

-- Implicit telemetry: per-agent-run latency and loop events (A2)
CREATE TABLE IF NOT EXISTS agent_run_telemetry (
    id                   SERIAL PRIMARY KEY,
    run_id               TEXT        NOT NULL,
    agent_name           TEXT        NOT NULL,
    event_type           TEXT        NOT NULL,  -- 'latency' | 'complexity_mismatch' | 'crag_fallback' | 'timeout'
    latency_ms           INT,
    complexity_declared  SMALLINT,
    react_loops_used     SMALLINT,
    notes                TEXT,
    recorded_at          TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_telemetry_run
    ON agent_run_telemetry (run_id);
CREATE INDEX IF NOT EXISTS idx_telemetry_agent
    ON agent_run_telemetry (agent_name, event_type);

-- ── B: Critic Agent run log ────────────────────────────────────────────────────
-- Stores NLI hallucination check results and Citation Utilisation Rate (CUR)
-- written by agents/critic/agent.py, populated nightly by airflow/dags/critic_dag.py

CREATE TABLE IF NOT EXISTS critic_run_log (
    id                    SERIAL PRIMARY KEY,
    run_id                TEXT        NOT NULL,
    agent_name            TEXT        NOT NULL,
    check_type            TEXT        NOT NULL,  -- 'nli_hallucination' | 'citation_utilisation_rate'
    claim                 TEXT,                  -- the factual claim being checked (NLI only)
    score                 FLOAT,                 -- entailment score (NLI) or CUR value
    verified              BOOLEAN,               -- TRUE = claim verified / CUR acceptable
    best_source_chunk_id  TEXT,                  -- chunk with highest entailment score
    notes                 TEXT,                  -- JSON metadata (e.g. {total_retrieved, total_cited})
    checked_at            TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_critic_run
    ON critic_run_log (run_id);
CREATE INDEX IF NOT EXISTS idx_critic_check_type
    ON critic_run_log (check_type, verified);

-- ---------------------------------------------------------------------------
-- pgvector: text_chunks — embedded text segments for semantic search
-- Stores chunks from company profiles + news, embedded with nomic-embed-text
-- (768-dim via Ollama).  Used by the Business Analyst agent for hybrid RAG.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS text_chunks (
    id          SERIAL PRIMARY KEY,
    ticker      TEXT NOT NULL,
    chunk_id    TEXT NOT NULL UNIQUE,
    text        TEXT NOT NULL,
    section     TEXT,
    filing_date TEXT,
    embedding   VECTOR(768),
    source      TEXT DEFAULT 'eodhd',
    ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_text_chunks_ticker
    ON text_chunks (ticker);

-- HNSW index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS text_chunks_embedding_idx
    ON text_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

