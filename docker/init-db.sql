-- init-db.sql
-- Runs once at PostgreSQL container first-start
-- (mounted at /docker-entrypoint-initdb.d/init-db.sql)
-- Creates all application tables so they exist before Airflow DAGs run.

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
