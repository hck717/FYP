"""
load_postgres.py  —  ETL loader: agent_data CSV/JSON  →  PostgreSQL
====================================================================

Called by the Airflow DAG after each ticker's scrape task.
Also called directly:

    python load_postgres.py            # loads AAPL (default)
    python load_postgres.py TSLA       # loads one ticker
    python load_postgres.py --macro    # loads _MACRO data

Key design decisions
--------------------
* All inserts use ON CONFLICT … DO UPDATE (upsert) — safe to re-run.
* NaN / NULL / "nan" strings in date columns are converted to None.
* Macro data (_MACRO folder: treasury, forex, bonds, screener, GDP/CPI/UE,
  economic_events, financial_calendar) is loaded via load_postgres_macro().
* Each specialty table (dividends, splits, sentiment, financial_statements,
  valuation_metrics, short_interest, earnings_surprises, outstanding_shares,
  economic_events, etc.) has its own typed insert function for correctness.
* Raw timeseries / fundamentals go to generic tables raw_timeseries /
  raw_fundamentals for anything that doesn't match a specialty table.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import requests

# ── Path resolution (works in Airflow container and local venv) ───────────────
# In the Airflow container __file__ is /opt/airflow/etl/load_postgres.py,
# so parent == /opt/airflow/etl and agent_data lives at parent / "agent_data".
# For local dev, __file__ is .../ingestion/etl/load_postgres.py — same layout.
_THIS_ETL_DIR    = Path(__file__).resolve().parent          # .../etl/
_DEFAULT_ETL_DIR = _THIS_ETL_DIR / "agent_data"
BASE_ETL_DIR = Path(os.getenv("BASE_ETL_DIR", str(_DEFAULT_ETL_DIR)))

# ── PostgreSQL connection ─────────────────────────────────────────────────────
PG_HOST     = os.getenv("POSTGRES_HOST",     "postgres")
PG_PORT     = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB       = os.getenv("POSTGRES_DB",       "airflow")
PG_USER     = os.getenv("POSTGRES_USER",     "airflow")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")

# Ollama embedding config (for pgvector text_chunks)
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("EMBEDDING_MODEL",  "nomic-embed-text")

_MACRO_TICKER = "_MACRO"


# ── Connection ────────────────────────────────────────────────────────────────

def get_pg_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        dbname=PG_DB, user=PG_USER, password=PG_PASSWORD,
    )


# ── Schema DDL ────────────────────────────────────────────────────────────────

def ensure_tables() -> None:
    ddl = """
    -- Generic timeseries store (news, EOD prices, intraday, technicals, quotes)
    CREATE TABLE IF NOT EXISTS raw_timeseries (
        id            SERIAL PRIMARY KEY,
        ticker_symbol TEXT      NOT NULL,
        data_name     TEXT      NOT NULL,
        ts_date       TIMESTAMP,
        payload       JSONB     NOT NULL,
        source        TEXT      NOT NULL,
        ingested_at   TIMESTAMP DEFAULT NOW(),
        UNIQUE (ticker_symbol, data_name, ts_date, source)
    );
    CREATE INDEX IF NOT EXISTS idx_raw_ts_ticker_name_date
        ON raw_timeseries (ticker_symbol, data_name, ts_date DESC);

    -- Generic fundamentals store (single-row snapshots without a time axis)
    CREATE TABLE IF NOT EXISTS raw_fundamentals (
        id            SERIAL PRIMARY KEY,
        ticker_symbol TEXT      NOT NULL,
        data_name     TEXT      NOT NULL,
        period_type   TEXT,
        as_of_date    DATE      NOT NULL,
        payload       JSONB     NOT NULL,
        source        TEXT      NOT NULL,
        ingested_at   TIMESTAMP DEFAULT NOW(),
        UNIQUE (ticker_symbol, data_name, period_type, as_of_date, source)
    );

    -- Macro: bulk screener snapshot
    CREATE TABLE IF NOT EXISTS market_screener (
        id          SERIAL PRIMARY KEY,
        ts_date     TIMESTAMP NOT NULL,
        ticker_code TEXT      NOT NULL DEFAULT '',
        payload     JSONB     NOT NULL,
        source      TEXT      NOT NULL,
        ingested_at TIMESTAMP DEFAULT NOW(),
        UNIQUE (ts_date, ticker_code, source)
    );

    -- Macro: GDP / CPI / unemployment
    CREATE TABLE IF NOT EXISTS global_macro_indicators (
        id          SERIAL PRIMARY KEY,
        indicator   TEXT      NOT NULL,
        ts_date     TIMESTAMP,
        payload     JSONB     NOT NULL,
        source      TEXT      NOT NULL,
        ingested_at TIMESTAMP DEFAULT NOW(),
        UNIQUE (indicator, ts_date, source)
    );
    CREATE INDEX IF NOT EXISTS idx_macro_indicators_date
        ON global_macro_indicators (indicator, ts_date DESC);

    -- Macro: Economic Events (row 12 of data_needed.txt)
    -- EODHD economic-events endpoint: CPI releases, FOMC, NFP, etc.
    CREATE TABLE IF NOT EXISTS economic_events (
        id               SERIAL PRIMARY KEY,
        event_date       DATE      NOT NULL,
        country          VARCHAR(10),
        event_name       TEXT,
        actual           NUMERIC,
        forecast         NUMERIC,
        previous         NUMERIC,
        impact           TEXT,
        currency         VARCHAR(10),
        comparison       TEXT,
        unit             TEXT,
        source_url       TEXT,
        ingested_at      TIMESTAMP DEFAULT NOW(),
        UNIQUE (event_date, country, event_name)
    );
    CREATE INDEX IF NOT EXISTS idx_economic_events_date
        ON economic_events (event_date DESC);
    CREATE INDEX IF NOT EXISTS idx_economic_events_country
        ON economic_events (country, event_date DESC);

    -- Sentiment
    CREATE TABLE IF NOT EXISTS sentiment_trends (
        id           SERIAL PRIMARY KEY,
        ticker       VARCHAR(20)  NOT NULL,
        bullish_pct  NUMERIC,
        bearish_pct  NUMERIC,
        neutral_pct  NUMERIC,
        trend        VARCHAR(20)  DEFAULT 'unknown',
        as_of_date   DATE         NOT NULL,
        ingested_at  TIMESTAMP    DEFAULT NOW(),
        UNIQUE (ticker, as_of_date)
    );
    CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date
        ON sentiment_trends (ticker, as_of_date DESC);

    -- Insider transactions
    CREATE TABLE IF NOT EXISTS insider_transactions (
        id               SERIAL PRIMARY KEY,
        ticker           TEXT      NOT NULL,
        insider_name     TEXT,
        transaction_type TEXT,
        shares           BIGINT,
        price            NUMERIC,
        transaction_date DATE,
        ingested_at      TIMESTAMP DEFAULT NOW(),
        UNIQUE (ticker, insider_name, transaction_date, transaction_type, shares, price)
    );
    CREATE INDEX IF NOT EXISTS idx_insider_ticker_date
        ON insider_transactions (ticker, transaction_date DESC);

    -- Institutional holders
    CREATE TABLE IF NOT EXISTS institutional_holders (
        id            SERIAL PRIMARY KEY,
        ticker        TEXT      NOT NULL,
        holder_name   TEXT,
        shares        BIGINT,
        shares_change NUMERIC,
        ownership_pct NUMERIC,
        as_of_date    DATE,
        ingested_at   TIMESTAMP DEFAULT NOW(),
        UNIQUE (ticker, holder_name, as_of_date)
    );
    CREATE INDEX IF NOT EXISTS idx_institutional_ticker_date
        ON institutional_holders (ticker, as_of_date DESC);

    -- Earnings / IPO financial calendar
    CREATE TABLE IF NOT EXISTS financial_calendar (
        id               SERIAL PRIMARY KEY,
        ticker           TEXT      NOT NULL,
        event_type       TEXT      NOT NULL,
        event_date       DATE      NOT NULL,
        eps_estimate     NUMERIC,
        revenue_estimate NUMERIC,
        ingested_at      TIMESTAMP DEFAULT NOW(),
        UNIQUE (ticker, event_type, event_date)
    );
    CREATE INDEX IF NOT EXISTS idx_calendar_ticker_date
        ON financial_calendar (ticker, event_date DESC);

    -- Dividend history
    CREATE TABLE IF NOT EXISTS dividends_history (
        id          SERIAL PRIMARY KEY,
        ticker      TEXT      NOT NULL,
        amount      NUMERIC,
        ex_date     DATE      NOT NULL,
        pay_date    DATE,
        record_date DATE,
        ingested_at TIMESTAMP DEFAULT NOW(),
        UNIQUE (ticker, ex_date)
    );
    CREATE INDEX IF NOT EXISTS idx_dividends_ticker_date
        ON dividends_history (ticker, ex_date DESC);

    -- Stock splits
    CREATE TABLE IF NOT EXISTS splits_history (
        id            SERIAL PRIMARY KEY,
        ticker        TEXT      NOT NULL,
        split_ratio   TEXT,
        announce_date DATE,
        ex_date       DATE      NOT NULL,
        ingested_at   TIMESTAMP DEFAULT NOW(),
        UNIQUE (ticker, ex_date)
    );
    CREATE INDEX IF NOT EXISTS idx_splits_ticker_date
        ON splits_history (ticker, ex_date DESC);

    -- Treasury yield curve
    CREATE TABLE IF NOT EXISTS treasury_rates (
        id          SERIAL PRIMARY KEY,
        indicator   TEXT      NOT NULL,
        rate        NUMERIC,
        ts_date     DATE      NOT NULL,
        ingested_at TIMESTAMP DEFAULT NOW(),
        UNIQUE (indicator, ts_date)
    );
    CREATE INDEX IF NOT EXISTS idx_treasury_date
        ON treasury_rates (ts_date DESC);

    -- Corporate bond yields / bond fundamentals (Row 13)
    -- Stores both EOD price-series rows (LQD/HYG proxy) and bond-fundamentals detail rows.
    CREATE TABLE IF NOT EXISTS corporate_bond_yields (
        id                  SERIAL PRIMARY KEY,
        isin                TEXT,
        ticker              TEXT,
        issuer_name         TEXT,
        coupon_rate         NUMERIC,
        maturity_date       DATE,
        yield_to_maturity   NUMERIC,
        current_price       NUMERIC,
        accrued_interest    NUMERIC,
        duration            NUMERIC,
        modified_duration   NUMERIC,
        convexity           NUMERIC,
        issue_date          DATE,
        credit_rating       TEXT,
        currency            TEXT,
        ts_date             DATE,
        ingested_at         TIMESTAMP DEFAULT NOW(),
        UNIQUE (isin, ts_date),
        UNIQUE (ticker, maturity_date, ts_date)
    );
    CREATE INDEX IF NOT EXISTS idx_bond_date
        ON corporate_bond_yields (ts_date DESC);

    -- Forex rates
    CREATE TABLE IF NOT EXISTS forex_rates (
        id          SERIAL PRIMARY KEY,
        forex_pair  TEXT      NOT NULL,
        rate        NUMERIC,
        ts_date     DATE      NOT NULL,
        ingested_at TIMESTAMP DEFAULT NOW(),
        UNIQUE (forex_pair, ts_date)
    );
    CREATE INDEX IF NOT EXISTS idx_forex_date
        ON forex_rates (ts_date DESC);

    -- IPO calendar (global) — DEPRECATED / ORPHANED.
    -- IPO data from calendar/ipos is now stored in financial_calendar
    -- with event_type = 'ipo'.  This table is kept for schema compatibility
    -- but is no longer populated by the pipeline.
    CREATE TABLE IF NOT EXISTS global_ipo_calendar (
        id          SERIAL PRIMARY KEY,
        ticker      TEXT,
        name        TEXT,
        exchange    TEXT,
        ipo_date    DATE,
        price       NUMERIC,
        ingested_at TIMESTAMP DEFAULT NOW(),
        UNIQUE (ticker, ipo_date)
    );

    -- Row 18: Financial Statements (Income Statement, Balance Sheet, Cash Flow)
    CREATE TABLE IF NOT EXISTS financial_statements (
        id             SERIAL PRIMARY KEY,
        ticker         TEXT      NOT NULL,
        statement_type TEXT      NOT NULL,  -- Income_Statement | Balance_Sheet | Cash_Flow
        period_type    TEXT      NOT NULL,  -- quarterly | yearly
        report_date    DATE      NOT NULL,
        payload        JSONB     NOT NULL,
        source         TEXT      NOT NULL DEFAULT 'eodhd',
        ingested_at    TIMESTAMP DEFAULT NOW(),
        UNIQUE (ticker, statement_type, period_type, report_date)
    );
    CREATE INDEX IF NOT EXISTS idx_fin_stmt_ticker_date
        ON financial_statements (ticker, statement_type, report_date DESC);

    -- Row 19: Valuation Metrics
    CREATE TABLE IF NOT EXISTS valuation_metrics (
        id                     SERIAL PRIMARY KEY,
        ticker                 TEXT      NOT NULL,
        as_of_date             DATE      NOT NULL,
        trailing_pe            NUMERIC,
        forward_pe             NUMERIC,
        price_sales_ttm        NUMERIC,
        price_book_mrq         NUMERIC,
        enterprise_value       NUMERIC,
        ev_revenue             NUMERIC,
        ev_ebitda              NUMERIC,
        market_cap             NUMERIC,
        ebitda                 NUMERIC,
        pe_ratio               NUMERIC,
        peg_ratio              NUMERIC,
        wall_street_target     NUMERIC,
        book_value             NUMERIC,
        dividend_share         NUMERIC,
        dividend_yield         NUMERIC,
        eps                    NUMERIC,
        eps_est_current_year   NUMERIC,
        eps_est_next_year      NUMERIC,
        profit_margin          NUMERIC,
        operating_margin       NUMERIC,
        roa                    NUMERIC,
        roe                    NUMERIC,
        revenue_ttm            NUMERIC,
        quarterly_rev_growth   NUMERIC,
        quarterly_earn_growth  NUMERIC,
        ingested_at            TIMESTAMP DEFAULT NOW(),
        UNIQUE (ticker, as_of_date)
    );
    CREATE INDEX IF NOT EXISTS idx_valuation_ticker_date
        ON valuation_metrics (ticker, as_of_date DESC);

    -- Row 20: Short Interest & Shares Stats
    CREATE TABLE IF NOT EXISTS short_interest (
        id                         SERIAL PRIMARY KEY,
        ticker                     TEXT      NOT NULL,
        as_of_date                 DATE      NOT NULL,
        shares_outstanding         BIGINT,
        shares_float               BIGINT,
        percent_insiders           NUMERIC,
        percent_institutions       NUMERIC,
        shares_short               BIGINT,
        shares_short_prior_month   BIGINT,
        short_ratio                NUMERIC,
        short_percent_outstanding  NUMERIC,
        short_percent_float        NUMERIC,
        ingested_at                TIMESTAMP DEFAULT NOW(),
        UNIQUE (ticker, as_of_date)
    );
    CREATE INDEX IF NOT EXISTS idx_short_interest_ticker_date
        ON short_interest (ticker, as_of_date DESC);

    -- Row 21: Earnings History & Surprises
    CREATE TABLE IF NOT EXISTS earnings_surprises (
        id                      SERIAL PRIMARY KEY,
        ticker                  TEXT      NOT NULL,
        period_date             DATE      NOT NULL,  -- quarter end date
        eps_actual              NUMERIC,
        eps_estimate            NUMERIC,
        eps_surprise_pct        NUMERIC,
        revenue_actual          NUMERIC,
        revenue_estimate        NUMERIC,
        revenue_surprise_pct    NUMERIC,
        before_after_market     TEXT,
        currency                TEXT,
        ingested_at             TIMESTAMP DEFAULT NOW(),
        UNIQUE (ticker, period_date)
    );
    CREATE INDEX IF NOT EXISTS idx_earnings_surprises_ticker_date
        ON earnings_surprises (ticker, period_date DESC);

    -- Row 22: Outstanding Shares History
    CREATE TABLE IF NOT EXISTS outstanding_shares (
        id                       SERIAL PRIMARY KEY,
        ticker                   TEXT      NOT NULL,
        period_type              TEXT      NOT NULL,  -- annual | quarterly
        shares_date              DATE      NOT NULL,
        shares_outstanding       BIGINT,
        ingested_at              TIMESTAMP DEFAULT NOW(),
        UNIQUE (ticker, period_type, shares_date)
    );
    CREATE INDEX IF NOT EXISTS idx_outstanding_shares_ticker_date
        ON outstanding_shares (ticker, shares_date DESC);

    -- Agent run telemetry / episodic memory (used by agents, not ingestion)
    CREATE TABLE IF NOT EXISTS agent_run_telemetry (
        id          SERIAL PRIMARY KEY,
        run_id      TEXT      UNIQUE,
        agent_name  TEXT,
        started_at  TIMESTAMP DEFAULT NOW(),
        finished_at TIMESTAMP,
        status      TEXT,
        payload     JSONB
    );

    CREATE TABLE IF NOT EXISTS agent_episodic_memory (
        id          SERIAL PRIMARY KEY,
        agent_name  TEXT,
        memory_key  TEXT,
        content     JSONB,
        created_at  TIMESTAMP DEFAULT NOW(),
        UNIQUE (agent_name, memory_key)
    );

    CREATE TABLE IF NOT EXISTS critic_run_log (
        id           SERIAL PRIMARY KEY,
        run_id       TEXT,
        critic_name  TEXT,
        verdict      TEXT,
        score        NUMERIC,
        notes        TEXT,
        created_at   TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS query_logs (
        id          SERIAL PRIMARY KEY,
        query_text  TEXT,
        agent_name  TEXT,
        result_rows INTEGER,
        duration_ms NUMERIC,
        logged_at   TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS citation_tracking (
        id          SERIAL PRIMARY KEY,
        source_id   TEXT,
        source_type TEXT,
        cited_in    TEXT,
        cited_at    TIMESTAMP DEFAULT NOW()
    );

    -- Textual documents metadata (PDF broker reports + earnings call transcripts)
    -- Binary content is NOT stored here; PDFs are ingested into Qdrant separately.
    CREATE TABLE IF NOT EXISTS textual_documents (
        id                  SERIAL PRIMARY KEY,
        ticker              TEXT        NOT NULL,
        doc_type            TEXT        NOT NULL,  -- broker_report | earnings_call
        filename            TEXT        NOT NULL,
        filepath            TEXT,
        institution         TEXT,
        date_approx         TEXT,
        file_size_bytes     BIGINT,
        md5_hash            TEXT,
        ingested_into_qdrant BOOLEAN    DEFAULT FALSE,
        ingested_at         TIMESTAMP   DEFAULT NOW(),
        UNIQUE (ticker, filename)
    );
    CREATE INDEX IF NOT EXISTS idx_textual_docs_ticker
        ON textual_documents (ticker);

    -- S&P 500 / market benchmark daily prices (for beta / benchmark calculations)
    CREATE TABLE IF NOT EXISTS market_eod_us (
        id          SERIAL PRIMARY KEY,
        ts_date     TIMESTAMP NOT NULL,
        payload     JSONB     NOT NULL,
        source      TEXT      NOT NULL,
        ingested_at TIMESTAMP DEFAULT NOW(),
        UNIQUE (ts_date, source)
    );
    CREATE INDEX IF NOT EXISTS idx_market_eod_us_date
        ON market_eod_us (ts_date DESC);

    -- pgvector: text chunks for semantic search (Business Analyst RAG)
    -- Requires pgvector extension (pgvector/pgvector:pg15 image).
    -- embedding column will be NULL until the extension is available.
    CREATE TABLE IF NOT EXISTS text_chunks (
        id          SERIAL PRIMARY KEY,
        ticker      TEXT NOT NULL,
        chunk_id    TEXT NOT NULL UNIQUE,
        text        TEXT NOT NULL,
        section     TEXT,
        filing_date TEXT,
        source      TEXT DEFAULT 'eodhd',
        ingested_at TIMESTAMP DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_text_chunks_ticker
        ON text_chunks (ticker);
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()

    # ── Schema migrations (idempotent ALTER TABLE / CREATE statements) ─────────
    # Run after the main DDL block so CREATE TABLE IF NOT EXISTS doesn't interfere.
    _run_migrations()


def _run_migrations() -> None:
    """Apply incremental schema migrations that are safe to re-run."""
    migrations = [
        # 2026-03: raw_fundamentals - add period_type column for financial statements
        # This allows quarterly and yearly data for the same date
        "ALTER TABLE raw_fundamentals ADD COLUMN IF NOT EXISTS period_type TEXT",
        # Drop old constraint and add new one with period_type
        "ALTER TABLE raw_fundamentals DROP CONSTRAINT IF EXISTS raw_fundamentals_ticker_symbol_data_name_as_of_date_source_key",
        "ALTER TABLE raw_fundamentals ADD UNIQUE (ticker_symbol, data_name, period_type, as_of_date, source)",
        # 2026-03: insider_transactions - add shares and price to unique constraint
        # This handles multiple transactions for same insider/date/type with different amounts
        "ALTER TABLE insider_transactions DROP CONSTRAINT IF EXISTS insider_transactions_ticker_insider_name_transaction_date_transaction_type_key",
        "ALTER TABLE insider_transactions ADD UNIQUE (ticker, insider_name, transaction_date, transaction_type, shares, price)",
        # 2025-Q4: corporate_bond_yields gained isin + extra fundamental columns.
        # Old schema only had: id, ticker, yield, maturity_date, ts_date, ingested_at
        "ALTER TABLE corporate_bond_yields ADD COLUMN IF NOT EXISTS isin TEXT",
        "ALTER TABLE corporate_bond_yields ADD COLUMN IF NOT EXISTS issuer_name TEXT",
        "ALTER TABLE corporate_bond_yields ADD COLUMN IF NOT EXISTS coupon_rate NUMERIC",
        "ALTER TABLE corporate_bond_yields ADD COLUMN IF NOT EXISTS yield_to_maturity NUMERIC",
        "ALTER TABLE corporate_bond_yields ADD COLUMN IF NOT EXISTS current_price NUMERIC",
        "ALTER TABLE corporate_bond_yields ADD COLUMN IF NOT EXISTS accrued_interest NUMERIC",
        "ALTER TABLE corporate_bond_yields ADD COLUMN IF NOT EXISTS duration NUMERIC",
        "ALTER TABLE corporate_bond_yields ADD COLUMN IF NOT EXISTS modified_duration NUMERIC",
        "ALTER TABLE corporate_bond_yields ADD COLUMN IF NOT EXISTS convexity NUMERIC",
        "ALTER TABLE corporate_bond_yields ADD COLUMN IF NOT EXISTS issue_date DATE",
        "ALTER TABLE corporate_bond_yields ADD COLUMN IF NOT EXISTS credit_rating TEXT",
        "ALTER TABLE corporate_bond_yields ADD COLUMN IF NOT EXISTS currency TEXT",
        # 2025-Q4: mv_daily_factor_scores materialized view for pre-computed factor scores.
        # Reads from financial_statements (always populated) + raw_fundamentals (FMP, optional).
        """
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_factor_scores AS
        SELECT
            fs.ticker                                                          AS ticker,
            MAX(fs.report_date)                                                AS as_of_date,
            MAX((rf.payload ->> 'piotroskiScore')::NUMERIC)                    AS piotroski_score,
            MAX((rf.payload ->> 'altmanZScore')::NUMERIC)                      AS altman_z_score,
            MAX((rf.payload ->> 'beneishMScore')::NUMERIC)                     AS beneish_m_score,
            MAX((rf.payload ->> 'returnOnEquityTTM')::NUMERIC)                 AS roe_ttm,
            MAX((rf.payload ->> 'returnOnAssetsTTM')::NUMERIC)                 AS roa_ttm,
            MAX((rf.payload ->> 'returnOnInvestedCapitalTTM')::NUMERIC)        AS roic_ttm,
            MAX((rf.payload ->> 'grossProfitMarginTTM')::NUMERIC)              AS gross_margin_ttm,
            MAX((rf.payload ->> 'netProfitMarginTTM')::NUMERIC)                AS net_margin_ttm,
            MAX((rf.payload ->> 'debtToEquityRatioTTM')::NUMERIC)              AS debt_to_equity_ttm,
            MAX((rf.payload ->> 'currentRatioTTM')::NUMERIC)                   AS current_ratio_ttm,
            NOW()                                                               AS refreshed_at
        FROM (
            SELECT DISTINCT ticker, MAX(report_date) AS report_date
            FROM financial_statements
            WHERE statement_type = 'Income_Statement'
            GROUP BY ticker
        ) fs
        LEFT JOIN raw_fundamentals rf
            ON rf.ticker_symbol = fs.ticker
            AND rf.data_name IN ('financial_scores', 'key_metrics_ttm', 'ratios_ttm')
        GROUP BY fs.ticker
        WITH NO DATA
        """,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_daily_factor_scores_ticker ON mv_daily_factor_scores (ticker)",
        "CREATE INDEX IF NOT EXISTS idx_mv_daily_factor_scores_date ON mv_daily_factor_scores (as_of_date DESC)",
        # pgvector: add embedding column to text_chunks if extension is available
        # (will silently fail if vector extension is not installed yet)
        "ALTER TABLE text_chunks ADD COLUMN IF NOT EXISTS embedding vector(768)",
        "CREATE INDEX IF NOT EXISTS text_chunks_embedding_idx ON text_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
    ]
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            for sql in migrations:
                try:
                    cur.execute(sql.strip())
                except Exception as exc:
                    # Ignore "already exists" errors; re-raise unexpected ones
                    msg = str(exc).lower()
                    if "already exists" not in msg and "duplicate" not in msg:
                        import logging
                        logging.getLogger(__name__).warning("Migration warning: %s", exc)
                    conn.rollback()
                    conn = get_pg_conn()
                    cur = conn.cursor()
                    continue
        conn.commit()


# ── Date helper ───────────────────────────────────────────────────────────────

def _safe_date(val) -> str | None:
    """Convert any value to an ISO-8601 date string, or None if not valid."""
    if val is None:
        return None
    try:
        if isinstance(val, float) and math.isnan(val):
            return None
    except Exception:
        pass
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "nat", "null", "", "0000-00-00"):
        return None
    return s[:10]  # take YYYY-MM-DD portion


def _safe_numeric(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> int | None:
    n = _safe_numeric(val)
    return None if n is None else int(n)


# ── Date-column detection ─────────────────────────────────────────────────────

def _detect_date_col(df: pd.DataFrame) -> str | None:
    for candidate in ["datetime", "date", "timestamp", "reportedDate", "t", "Date"]:
        if candidate in df.columns:
            return candidate
    return None


def _normalise_date_col(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Convert epoch-integer date columns to 'YYYY-MM-DD HH:MM:SS' strings."""
    col = df[date_col]
    if pd.api.types.is_integer_dtype(col) or pd.api.types.is_float_dtype(col):
        df = df.copy()
        df[date_col] = (
            pd.to_datetime(col, unit="s", errors="coerce")
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )
    return df


# ── Specialty insert functions ────────────────────────────────────────────────

def _insert_sentiment(ticker_symbol: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = []
    today = date.today().isoformat()

    for _, row in df.iterrows():
        d = row.to_dict()
        row_date = _safe_date(d.get("date") or d.get("datetime") or d.get("timestamp")) or today

        if "normalized" in d and "pos" not in d:
            try:
                norm = float(d["normalized"])
            except (TypeError, ValueError):
                continue
            bullish = round(max(0.0, norm - 0.5) * 200, 4)
            bearish = round(max(0.0, 0.5 - norm) * 200, 4)
            neutral = round(100.0 - bullish - bearish, 4)
        else:
            def _pct(key: str) -> float:
                raw = d.get(key) or d.get(f"sentiment_{key}") or d.get(f"sentiment.{key}")
                try:
                    v = float(raw)
                    return round(v * 100, 4) if v <= 1.0 else round(v, 4)
                except (TypeError, ValueError):
                    return 0.0
            bullish = _pct("pos")
            bearish = _pct("neg")
            neutral = _pct("neu")

        if bullish == 0.0 and bearish == 0.0 and neutral == 0.0:
            continue
        trend = (
            "improving"    if bullish > bearish + 10 else
            "deteriorating" if bearish > bullish + 10 else
            "stable"
        )
        rows.append((ticker_symbol, bullish, bearish, neutral, trend, row_date))

    if not rows:
        return 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO sentiment_trends
                    (ticker, bullish_pct, bearish_pct, neutral_pct, trend, as_of_date)
                VALUES %s
                ON CONFLICT (ticker, as_of_date) DO UPDATE SET
                    bullish_pct = EXCLUDED.bullish_pct,
                    bearish_pct = EXCLUDED.bearish_pct,
                    neutral_pct = EXCLUDED.neutral_pct,
                    trend       = EXCLUDED.trend,
                    ingested_at = NOW()
            """, rows)
        conn.commit()
    return len(rows)


def _insert_insider_transactions(ticker_symbol: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        # EODHD insider-transactions fields: ownerName, transactionCode,
        # transactionAmount (shares), transactionPrice, transactionDate
        insider_name = str(
            d.get("ownerName") or d.get("insider") or d.get("insider_name") or d.get("name") or "unknown"
        )
        tx_type = str(
            d.get("transactionCode") or d.get("transactionType") or d.get("type") or "unknown"
        )
        shares = _safe_int(
            d.get("transactionAmount") or d.get("shares") or d.get("sharesTraded")
        )
        price = _safe_numeric(
            d.get("transactionPrice") or d.get("price") or d.get("sharePrice")
        )
        tx_date = _safe_date(
            d.get("transactionDate") or d.get("date")
        )
        rows.append((ticker_symbol, insider_name, tx_type, shares, price, tx_date))
    if not rows:
        return 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO insider_transactions
                    (ticker, insider_name, transaction_type, shares, price, transaction_date)
                VALUES %s
                ON CONFLICT (ticker, insider_name, transaction_date, transaction_type, shares, price)
                DO UPDATE SET ingested_at = NOW()
            """, rows)
        conn.commit()
    return len(rows)


def _insert_institutional_holders(ticker_symbol: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = []
    today = date.today().isoformat()
    for _, row in df.iterrows():
        d = row.to_dict()
        # EODHD fields: holder_type, name, date, totalShares, totalAssets,
        #               currentShares, change, change_p
        rows.append((
            ticker_symbol,
            str(d.get("name") or d.get("holder") or d.get("holder_name") or "unknown"),
            _safe_int(d.get("currentShares") or d.get("shares") or d.get("sharesHeld")),
            _safe_numeric(d.get("change") or d.get("sharesChange")),
            _safe_numeric(d.get("totalAssets") or d.get("ownership") or d.get("ownershipPct")),
            _safe_date(d.get("date") or d.get("asOfDate")) or today,
        ))
    if not rows:
        return 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO institutional_holders
                    (ticker, holder_name, shares, shares_change, ownership_pct, as_of_date)
                VALUES %s
                ON CONFLICT (ticker, holder_name, as_of_date) DO UPDATE SET
                    shares        = EXCLUDED.shares,
                    shares_change = EXCLUDED.shares_change,
                    ownership_pct = EXCLUDED.ownership_pct,
                    ingested_at   = NOW()
            """, rows)
        conn.commit()
    return len(rows)


def _insert_financial_calendar(df: pd.DataFrame, event_type_override: str | None = None) -> int:
    """Financial calendar — ticker comes from the data itself (per-ticker or macro).

    Parameters
    ----------
    event_type_override : str | None
        When set, forces every row to use this event_type string instead of
        reading it from the row data.  Use "ipo", "split", "dividend" for the
        respective calendar/ipos, calendar/splits, calendar/dividends feeds.
    """
    if df.empty:
        return 0
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        # EODHD calendar/earnings fields: code, report_date, date (fiscal period end),
        # estimate (EPS estimate), actual, before_after_market, currency, difference, percent
        # IPO fields: code/name, exchange, date, price
        # Split fields: code, splitDate, optionalDate, split (ratio string)
        # Dividend fields: code, ex_date, paymentDate, declarationDate, record_date, amount, currency
        raw_ticker = (d.get("code") or d.get("ticker") or d.get("symbol") or "UNKNOWN")
        # Strip exchange suffix e.g. "AAPL.US" → "AAPL"
        ticker = str(raw_ticker).split(".")[0]
        if event_type_override:
            ev_type = event_type_override
        else:
            ev_type = str(d.get("event_type") or "earnings")
        # Prefer report_date (actual filing date) over date (fiscal period end).
        # IPO: "date" field; Split: "splitDate"; Dividend: "ex_date" / "declarationDate"
        ev_date = _safe_date(
            d.get("report_date")
            or d.get("splitDate")
            or d.get("ex_date")
            or d.get("declarationDate")
            or d.get("date")
            or d.get("event_date")
        )
        if not ev_date:
            continue
        rows.append((
            ticker, ev_type, ev_date,
            _safe_numeric(d.get("estimate") or d.get("epsEstimate") or d.get("eps_estimate")),
            _safe_numeric(d.get("revenue_estimate") or d.get("revenueEstimate")),
        ))
    if not rows:
        return 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO financial_calendar
                    (ticker, event_type, event_date, eps_estimate, revenue_estimate)
                VALUES %s
                ON CONFLICT (ticker, event_type, event_date) DO UPDATE SET
                    eps_estimate     = EXCLUDED.eps_estimate,
                    revenue_estimate = EXCLUDED.revenue_estimate,
                    ingested_at      = NOW()
            """, rows)
        conn.commit()
    return len(rows)


def _insert_dividends_history(ticker_symbol: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        ex_date = _safe_date(d.get("exDate") or d.get("ex_date") or d.get("Date") or d.get("date"))
        if not ex_date:
            continue
        rows.append((
            ticker_symbol,
            _safe_numeric(d.get("amount") or d.get("dividend") or d.get("value")),
            ex_date,
            _safe_date(d.get("paymentDate") or d.get("pay_date")),
            _safe_date(d.get("recordDate") or d.get("record_date")),
        ))
    if not rows:
        return 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO dividends_history
                    (ticker, amount, ex_date, pay_date, record_date)
                VALUES %s
                ON CONFLICT (ticker, ex_date) DO UPDATE SET
                    amount      = EXCLUDED.amount,
                    pay_date    = EXCLUDED.pay_date,
                    record_date = EXCLUDED.record_date,
                    ingested_at = NOW()
            """, rows)
        conn.commit()
    return len(rows)


def _insert_splits_history(ticker_symbol: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        ex_date = _safe_date(d.get("date") or d.get("exDate") or d.get("ex_date"))
        if not ex_date:
            continue
        rows.append((
            ticker_symbol,
            str(d.get("splitRatio") or d.get("ratio") or d.get("split_ratio") or ""),
            _safe_date(d.get("announcedDate") or d.get("announce_date")),
            ex_date,
        ))
    if not rows:
        return 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO splits_history (ticker, split_ratio, announce_date, ex_date)
                VALUES %s
                ON CONFLICT (ticker, ex_date) DO UPDATE SET
                    split_ratio   = EXCLUDED.split_ratio,
                    announce_date = EXCLUDED.announce_date,
                    ingested_at   = NOW()
            """, rows)
        conn.commit()
    return len(rows)


def _insert_treasury_rates(df: pd.DataFrame, indicator_override: str | None = None) -> int:
    if df.empty:
        return 0
    # New format: EOD response {date, open, high, low, close, adjusted_close, volume}
    # We store the 'close' column as the yield rate.
    # indicator_override lets the caller specify the tenor name (e.g. "US3M", "US30Y").
    date_col = _detect_date_col(df) or "date"
    rows = []
    if "close" in df.columns or "adjusted_close" in df.columns:
        # EOD GBOND format — store close as the indicator
        indicator = indicator_override or "US10Y"
        for _, row in df.iterrows():
            d = row.to_dict()
            ts = _safe_date(d.get(date_col))
            if not ts:
                continue
            rate = _safe_numeric(d.get("close") or d.get("adjusted_close"))
            if rate is not None:
                rows.append((indicator, rate, ts))
    else:
        # Legacy multi-column format {date, y1, y2, y5, y10, y30, ...}
        rate_cols = [c for c in df.columns if c != date_col]
        for _, row in df.iterrows():
            d = row.to_dict()
            ts = _safe_date(d.get(date_col))
            if not ts:
                continue
            for col in rate_cols:
                val = _safe_numeric(d.get(col))
                if val is not None:
                    rows.append((col, val, ts))
    if not rows:
        return 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO treasury_rates (indicator, rate, ts_date)
                VALUES %s
                ON CONFLICT (indicator, ts_date) DO UPDATE SET
                    rate        = EXCLUDED.rate,
                    ingested_at = NOW()
            """, rows)
        conn.commit()
    return len(rows)


def _insert_macro_indicators(indicator_name: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    date_col = _detect_date_col(df) or "date"
    rows = []
    payload_cols = [c for c in df.columns if c != date_col]
    for _, row in df.iterrows():
        d = row.to_dict()
        ts = d.get(date_col)
        if ts is None:
            continue
        payload = {c: (None if (isinstance(d[c], float) and math.isnan(d[c])) else d[c])
                   for c in payload_cols}
        rows.append((indicator_name, str(ts)[:10] if ts else None, json.dumps(payload), "eodhd"))
    if not rows:
        return 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO global_macro_indicators (indicator, ts_date, payload, source)
                VALUES %s
                ON CONFLICT (indicator, ts_date, source) DO UPDATE SET
                    payload     = EXCLUDED.payload,
                    ingested_at = NOW()
            """, rows)
        conn.commit()
    return len(rows)


def _insert_screener_bulk(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    now_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        payload = {k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                   for k, v in d.items()}
        # Use ticker code (screener API uses 'code' field) as row identifier
        ticker_code = str(d.get("code") or d.get("ticker") or d.get("Code") or "")
        rows.append((now_ts, ticker_code, json.dumps(payload), "eodhd"))
    if not rows:
        return 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO market_screener (ts_date, ticker_code, payload, source)
                VALUES %s
                ON CONFLICT (ts_date, ticker_code, source) DO UPDATE SET
                    payload     = EXCLUDED.payload,
                    ingested_at = NOW()
            """, rows)
        conn.commit()
    return len(rows)


def _insert_forex_rates(df: pd.DataFrame, pair_name: str = "EURUSD") -> int:
    """
    Handle EODHD forex data in EOD format {date, open, high, low, close, ...}
    (from eod/EURUSD.FOREX endpoint) or legacy metadata-list format.
    pair_name should be passed by the caller (derived from data_name or endpoint).
    """
    if df.empty:
        return 0

    has_rate = "close" in df.columns or "Close" in df.columns or "adjusted_close" in df.columns
    date_col = _detect_date_col(df) or "date"

    if has_rate and date_col in df.columns:
        # EOD format: {date, open, high, low, close, adjusted_close, volume}
        rows = []
        for _, row in df.iterrows():
            d = row.to_dict()
            ts   = _safe_date(d.get(date_col))
            rate = _safe_numeric(d.get("close") or d.get("adjusted_close") or d.get("Close"))
            if not ts or rate is None:
                continue
            rows.append((pair_name, rate, ts))
        if rows:
            with get_pg_conn() as conn:
                with conn.cursor() as cur:
                    execute_values(cur, """
                        INSERT INTO forex_rates (forex_pair, rate, ts_date)
                        VALUES %s
                        ON CONFLICT (forex_pair, ts_date) DO UPDATE SET
                            rate        = EXCLUDED.rate,
                            ingested_at = NOW()
                    """, rows)
                conn.commit()
        return len(rows)

    # Fallback: store as raw_timeseries rows using per-row date from any date column.
    # Use a unique ts per row to avoid the UNIQUE constraint collapsing all rows.
    rows_ts = []
    for idx, (_, row) in enumerate(df.iterrows()):
        d = row.to_dict()
        payload = {k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                   for k, v in d.items()}
        # Try to find a date in the row; fall back to today + row-index offset to keep unique
        row_date = _safe_date(
            d.get("date") or d.get("Date") or d.get("datetime") or d.get("timestamp")
        )
        if not row_date:
            row_date = (
                datetime.utcnow().replace(hour=0, minute=0, second=idx, microsecond=0)
                .strftime("%Y-%m-%d %H:%M:%S")
            )
        rows_ts.append(("_GLOBAL", "forex_historical_rates", row_date, json.dumps(payload), "eodhd", row_date))
    if not rows_ts:
        return 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO raw_timeseries
                    (ticker_symbol, data_name, ts_date, payload, source, ingested_at)
                VALUES %s
                ON CONFLICT (ticker_symbol, data_name, ts_date, source) DO UPDATE SET
                    payload     = EXCLUDED.payload,
                    ingested_at = NOW()
            """, rows_ts)
        conn.commit()
    return len(rows_ts)


def _insert_bond_fundamentals(df: pd.DataFrame, ticker: str = "") -> int:
    """
    Insert EODHD bond-fundamentals API response into corporate_bond_yields.
    EODHD bond-fundamentals fields (top-level dict or list of dicts):
    Isin, ShortName/Name, CouponRate, MaturityDate, Yield, Price,
    Duration, ModifiedDuration, Convexity, IssueDate, CreditRating, Currency.
    """
    if df.empty:
        return 0
    today = date.today().isoformat()
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        isin         = str(d.get("ISIN") or d.get("Isin") or d.get("isin") or "")
        issuer_name  = str(d.get("Name") or d.get("ShortName") or d.get("issuer_name") or d.get("name") or "")
        coupon       = _safe_numeric(d.get("Coupon") or d.get("CouponRate") or d.get("coupon_rate") or d.get("coupon"))
        maturity     = _safe_date(d.get("Maturity_Date") or d.get("MaturityDate") or d.get("maturity_date") or d.get("maturity"))
        ytm          = _safe_numeric(d.get("YieldToMaturity") or d.get("Yield") or d.get("yield_to_maturity") or d.get("ytm"))
        price        = _safe_numeric(d.get("Price") or d.get("current_price") or d.get("price"))
        accrued      = _safe_numeric(d.get("AccruedInterest") or d.get("accrued_interest"))
        duration     = _safe_numeric(d.get("Duration") or d.get("duration"))
        mod_dur      = _safe_numeric(d.get("ModifiedDuration") or d.get("modified_duration"))
        convexity    = _safe_numeric(d.get("Convexity") or d.get("convexity"))
        issue_date   = _safe_date(d.get("IssueDate") or d.get("issue_date"))
        credit_rating = str(d.get("CreditRating") or d.get("credit_rating") or d.get("rating") or "")
        currency     = str(d.get("Currency") or d.get("currency") or "")
        ts           = _safe_date(d.get("date") or d.get("ts_date") or d.get("Date")) or today
        rows.append((
            isin or None,
            ticker or str(d.get("code") or d.get("ticker") or ""),
            issuer_name,
            coupon, maturity, ytm, price, accrued,
            duration, mod_dur, convexity,
            issue_date, credit_rating, currency, ts,
        ))
    if not rows:
        return 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO corporate_bond_yields
                    (isin, ticker, issuer_name, coupon_rate, maturity_date,
                     yield_to_maturity, current_price, accrued_interest,
                     duration, modified_duration, convexity,
                     issue_date, credit_rating, currency, ts_date)
                VALUES %s
                ON CONFLICT (ticker, maturity_date, ts_date) DO UPDATE SET
                    isin              = COALESCE(EXCLUDED.isin, corporate_bond_yields.isin),
                    issuer_name       = EXCLUDED.issuer_name,
                    coupon_rate       = EXCLUDED.coupon_rate,
                    yield_to_maturity = EXCLUDED.yield_to_maturity,
                    current_price     = EXCLUDED.current_price,
                    accrued_interest  = EXCLUDED.accrued_interest,
                    duration          = EXCLUDED.duration,
                    modified_duration = EXCLUDED.modified_duration,
                    convexity         = EXCLUDED.convexity,
                    credit_rating     = EXCLUDED.credit_rating,
                    ingested_at       = NOW()
            """, rows)
        conn.commit()
    return len(rows)


def _insert_corporate_bond_eod(df: pd.DataFrame, ticker: str = "LQD") -> int:
    """
    Insert LQD/HYG ETF EOD price-series into corporate_bond_yields as a proxy row.
    Stores close price as current_price; ticker is the ETF code.
    """
    if df.empty:
        return 0
    date_col = _detect_date_col(df) or "date"
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        ts    = _safe_date(d.get(date_col))
        price = _safe_numeric(d.get("close") or d.get("adjusted_close") or d.get("Close"))
        if not ts or price is None:
            continue
        # Use a far-future placeholder maturity for the UNIQUE constraint
        rows.append((None, ticker, ticker, None, None, None, price, None, None, None, None, None, "", "USD", ts))
    if not rows:
        return 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO corporate_bond_yields
                    (isin, ticker, issuer_name, coupon_rate, maturity_date,
                     yield_to_maturity, current_price, accrued_interest,
                     duration, modified_duration, convexity,
                     issue_date, credit_rating, currency, ts_date)
                VALUES %s
                ON CONFLICT (ticker, maturity_date, ts_date) DO UPDATE SET
                    current_price = EXCLUDED.current_price,
                    ingested_at   = NOW()
            """, rows)
        conn.commit()
    return len(rows)


def _insert_dataframe_generic(
    df: pd.DataFrame,
    ticker_symbol: str,
    data_name: str,
    source: str,
) -> int:
    """
    Generic upsert into raw_timeseries (if a date column is found) or
    raw_fundamentals (snapshot with today's date).
    """
    if df.empty:
        return 0

    df = df.copy()
    date_col = _detect_date_col(df)

    if date_col:
        df = _normalise_date_col(df, date_col)
        df = df.drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True)
    else:
        df = df.drop_duplicates().reset_index(drop=True)

    payload_cols = [c for c in df.columns if c != date_col]
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Build separate row lists for timeseries vs fundamentals
    rows_ts = []
    rows_fund = []
    for _, row in df.iterrows():
        ts_val = str(row[date_col]) if date_col else now[:10]
        payload = {}
        for c in payload_cols:
            v = row[c]
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                v = None
            payload[c] = v
        # timeseries: 6 columns
        rows_ts.append((ticker_symbol, data_name, ts_val, json.dumps(payload), source, now))
        # fundamentals: 7 columns (with period_type = None)
        rows_fund.append((ticker_symbol, data_name, None, ts_val, json.dumps(payload), source, now))

    # Deduplicate within each batch
    def deduplicate(rows):
        seen = {}
        for r in rows:
            key = tuple(r[:4]) + (r[5],)  # exclude payload and timestamp from key
            if key not in seen:
                seen[key] = r
        return list(seen.values())

    rows_ts = deduplicate(rows_ts)
    rows_fund = deduplicate(rows_fund)

    if not rows_ts and not rows_fund:
        return 0

    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            if date_col and rows_ts:
                execute_values(cur, """
                    INSERT INTO raw_timeseries
                        (ticker_symbol, data_name, ts_date, payload, source, ingested_at)
                    VALUES %s
                    ON CONFLICT (ticker_symbol, data_name, ts_date, source) DO UPDATE SET
                        payload     = EXCLUDED.payload,
                        ingested_at = EXCLUDED.ingested_at
                """, rows_ts)
            if not date_col and rows_fund:
                execute_values(cur, """
                    INSERT INTO raw_fundamentals
                        (ticker_symbol, data_name, period_type, as_of_date, payload, source, ingested_at)
                    VALUES %s
                    ON CONFLICT (ticker_symbol, data_name, period_type, as_of_date, source) DO UPDATE SET
                        payload     = EXCLUDED.payload,
                        ingested_at = EXCLUDED.ingested_at
                """, rows_fund)
        conn.commit()
    return len(rows_ts) + len(rows_fund)


def _insert_textual_documents(docs: list[dict]) -> int:
    """
    Upsert textual document metadata records into the textual_documents table.
    Each dict should have: ticker, doc_type, filename, filepath, institution,
    date_approx, file_size_bytes, md5_hash, ingested_into_qdrant (optional).
    """
    if not docs:
        return 0
    rows = []
    for d in docs:
        rows.append((
            str(d.get("ticker") or ""),
            str(d.get("doc_type") or d.get("document_type") or ""),
            str(d.get("filename") or ""),
            str(d.get("filepath") or ""),
            str(d.get("institution") or ""),
            str(d.get("date_approx") or "") or None,
            int(d["file_size_bytes"]) if d.get("file_size_bytes") is not None else None,
            str(d.get("md5_hash") or "") or None,
            bool(d.get("ingested_into_qdrant", False)),
        ))
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO textual_documents
                    (ticker, doc_type, filename, filepath, institution,
                     date_approx, file_size_bytes, md5_hash, ingested_into_qdrant)
                VALUES %s
                ON CONFLICT (ticker, filename) DO UPDATE SET
                    filepath             = EXCLUDED.filepath,
                    institution          = EXCLUDED.institution,
                    date_approx          = EXCLUDED.date_approx,
                    file_size_bytes      = EXCLUDED.file_size_bytes,
                    md5_hash             = EXCLUDED.md5_hash,
                    ingested_into_qdrant = EXCLUDED.ingested_into_qdrant,
                    ingested_at          = NOW()
            """, rows)
        conn.commit()
    return len(rows)


# ── New specialty insert functions (rows 12, 18–22 of data_needed.txt) ────────

def _insert_economic_events(df: pd.DataFrame) -> int:
    """
    Insert EODHD economic-events API results into the economic_events table.
    EODHD response fields: date, country, event, actual, forecast, previous,
    impact, currency, comparison, unit, sourceURL (or source_url).
    """
    if df.empty:
        return 0
    rows = []
    seen = set()
    for _, row in df.iterrows():
        d = row.to_dict()
        ev_date = _safe_date(d.get("date") or d.get("event_date"))
        if not ev_date:
            continue
        country = str(d.get("country") or "")[:10]
        event = str(d.get("event") or d.get("event_name") or d.get("name") or "")
        key = (ev_date, country, event)
        if key in seen:
            continue
        seen.add(key)
        rows.append((
            ev_date,
            country,
            event,
            _safe_numeric(d.get("actual")),
            _safe_numeric(d.get("forecast")),
            _safe_numeric(d.get("previous")),
            str(d.get("impact") or ""),
            str(d.get("currency") or "")[:10],
            str(d.get("comparison") or d.get("comparison_period") or ""),
            str(d.get("unit") or ""),
            str(d.get("sourceURL") or d.get("source_url") or ""),
        ))
    if not rows:
        return 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO economic_events
                    (event_date, country, event_name, actual, forecast, previous,
                     impact, currency, comparison, unit, source_url)
                VALUES %s
                ON CONFLICT (event_date, country, event_name) DO UPDATE SET
                    actual      = EXCLUDED.actual,
                    forecast    = EXCLUDED.forecast,
                    previous    = EXCLUDED.previous,
                    impact      = EXCLUDED.impact,
                    ingested_at = NOW()
            """, rows)
        conn.commit()
    return len(rows)


def _insert_financial_statements(ticker_symbol: str, df: pd.DataFrame) -> int:
    """
    Insert financial statements (Income, Balance Sheet, Cash Flow) into
    financial_statements table.  The CSV has columns:
      statement_type, period_type, report_date, + all financial line-items.
    """
    if df.empty:
        return 0
    rows = []
    date_col = "report_date"
    payload_cols = [c for c in df.columns if c not in ("statement_type", "period_type", "report_date")]
    for _, row in df.iterrows():
        d = row.to_dict()
        rpt_date = _safe_date(d.get(date_col) or d.get("date") or d.get("Date"))
        if not rpt_date:
            continue
        stmt_type   = str(d.get("statement_type") or "unknown")
        period_type = str(d.get("period_type") or "unknown")
        payload = {}
        for c in payload_cols:
            v = d[c]
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                v = None
            payload[c] = v
        rows.append((ticker_symbol, stmt_type, period_type, rpt_date, json.dumps(payload), "eodhd"))
    if not rows:
        return 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO financial_statements
                    (ticker, statement_type, period_type, report_date, payload, source)
                VALUES %s
                ON CONFLICT (ticker, statement_type, period_type, report_date) DO UPDATE SET
                    payload     = EXCLUDED.payload,
                    ingested_at = NOW()
            """, rows)
        conn.commit()
    return len(rows)


def _insert_valuation_metrics(ticker_symbol: str, df: pd.DataFrame) -> int:
    """
    Insert valuation metrics snapshot into the valuation_metrics table.
    The CSV comes from the fundamentals company_profile row that includes
    Highlights_* and Valuation_* prefixed columns.
    """
    if df.empty:
        return 0
    today = date.today().isoformat()
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        as_of = _safe_date(d.get("UpdatedAt") or d.get("as_of_date")) or today
        rows.append((
            ticker_symbol,
            as_of,
            _safe_numeric(d.get("Valuation_TrailingPE")   or d.get("TrailingPE")),
            _safe_numeric(d.get("Valuation_ForwardPE")    or d.get("ForwardPE")),
            _safe_numeric(d.get("Valuation_PriceSalesTTM") or d.get("PriceSalesTTM")),
            _safe_numeric(d.get("Valuation_PriceBookMRQ")  or d.get("PriceBookMRQ")),
            _safe_numeric(d.get("Valuation_EnterpriseValue") or d.get("EnterpriseValue")),
            _safe_numeric(d.get("Valuation_EnterpriseValueRevenue") or d.get("EnterpriseValueRevenue")),
            _safe_numeric(d.get("Valuation_EnterpriseValueEbitda")  or d.get("EnterpriseValueEbitda")),
            _safe_numeric(d.get("Highlights_MarketCapitalization") or d.get("MarketCapitalization")),
            _safe_numeric(d.get("Highlights_EBITDA")                or d.get("EBITDA")),
            _safe_numeric(d.get("Highlights_PERatio")               or d.get("PERatio")),
            _safe_numeric(d.get("Highlights_PEGRatio")              or d.get("PEGRatio")),
            _safe_numeric(d.get("Highlights_WallStreetTargetPrice") or d.get("WallStreetTargetPrice")),
            _safe_numeric(d.get("Highlights_BookValue")             or d.get("BookValue")),
            _safe_numeric(d.get("Highlights_DividendShare")         or d.get("DividendShare")),
            _safe_numeric(d.get("Highlights_DividendYield")         or d.get("DividendYield")),
            _safe_numeric(d.get("Highlights_EarningsShare")         or d.get("EarningsShare")),
            _safe_numeric(d.get("Highlights_EPSEstimateCurrentYear") or d.get("EPSEstimateCurrentYear")),
            _safe_numeric(d.get("Highlights_EPSEstimateNextYear")    or d.get("EPSEstimateNextYear")),
            _safe_numeric(d.get("Highlights_ProfitMargin")           or d.get("ProfitMargin")),
            _safe_numeric(d.get("Highlights_OperatingMarginTTM")     or d.get("OperatingMarginTTM")),
            _safe_numeric(d.get("Highlights_ReturnOnAssetsTTM")      or d.get("ReturnOnAssetsTTM")),
            _safe_numeric(d.get("Highlights_ReturnOnEquityTTM")      or d.get("ReturnOnEquityTTM")),
            _safe_numeric(d.get("Highlights_RevenueTTM")             or d.get("RevenueTTM")),
            _safe_numeric(d.get("Highlights_QuarterlyRevenueGrowthYOY") or d.get("QuarterlyRevenueGrowthYOY")),
            _safe_numeric(d.get("Highlights_QuarterlyEarningsGrowthYOY") or d.get("QuarterlyEarningsGrowthYOY")),
        ))
    if not rows:
        return 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO valuation_metrics
                    (ticker, as_of_date,
                     trailing_pe, forward_pe, price_sales_ttm, price_book_mrq,
                     enterprise_value, ev_revenue, ev_ebitda,
                     market_cap, ebitda, pe_ratio, peg_ratio, wall_street_target,
                     book_value, dividend_share, dividend_yield, eps,
                     eps_est_current_year, eps_est_next_year,
                     profit_margin, operating_margin, roa, roe, revenue_ttm,
                     quarterly_rev_growth, quarterly_earn_growth)
                VALUES %s
                ON CONFLICT (ticker, as_of_date) DO UPDATE SET
                    trailing_pe          = EXCLUDED.trailing_pe,
                    forward_pe           = EXCLUDED.forward_pe,
                    price_sales_ttm      = EXCLUDED.price_sales_ttm,
                    price_book_mrq       = EXCLUDED.price_book_mrq,
                    enterprise_value     = EXCLUDED.enterprise_value,
                    ev_revenue           = EXCLUDED.ev_revenue,
                    ev_ebitda            = EXCLUDED.ev_ebitda,
                    market_cap           = EXCLUDED.market_cap,
                    ebitda               = EXCLUDED.ebitda,
                    pe_ratio             = EXCLUDED.pe_ratio,
                    peg_ratio            = EXCLUDED.peg_ratio,
                    wall_street_target   = EXCLUDED.wall_street_target,
                    book_value           = EXCLUDED.book_value,
                    dividend_share       = EXCLUDED.dividend_share,
                    dividend_yield       = EXCLUDED.dividend_yield,
                    eps                  = EXCLUDED.eps,
                    eps_est_current_year = EXCLUDED.eps_est_current_year,
                    eps_est_next_year    = EXCLUDED.eps_est_next_year,
                    profit_margin        = EXCLUDED.profit_margin,
                    operating_margin     = EXCLUDED.operating_margin,
                    roa                  = EXCLUDED.roa,
                    roe                  = EXCLUDED.roe,
                    revenue_ttm          = EXCLUDED.revenue_ttm,
                    quarterly_rev_growth = EXCLUDED.quarterly_rev_growth,
                    quarterly_earn_growth = EXCLUDED.quarterly_earn_growth,
                    ingested_at          = NOW()
            """, rows)
        conn.commit()
    return len(rows)


def _insert_short_interest(ticker_symbol: str, df: pd.DataFrame) -> int:
    """
    Insert short interest / shares stats from fundamentals SharesStats filter.
    The CSV is a single-row snapshot with SharesOutstanding, SharesFloat, etc.
    """
    if df.empty:
        return 0
    today = date.today().isoformat()
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        as_of = _safe_date(d.get("ShortInterestDate") or d.get("as_of_date") or d.get("date")) or today
        rows.append((
            ticker_symbol,
            as_of,
            _safe_int(d.get("SharesOutstanding")),
            _safe_int(d.get("SharesFloat")),
            _safe_numeric(d.get("PercentInsiders")),
            _safe_numeric(d.get("PercentInstitutions")),
            _safe_int(d.get("SharesShort")),
            _safe_int(d.get("SharesShortPriorMonth")),
            _safe_numeric(d.get("ShortRatio")),
            _safe_numeric(d.get("ShortPercentOutstanding")),
            _safe_numeric(d.get("ShortPercentFloat")),
        ))
    if not rows:
        return 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO short_interest
                    (ticker, as_of_date,
                     shares_outstanding, shares_float,
                     percent_insiders, percent_institutions,
                     shares_short, shares_short_prior_month,
                     short_ratio, short_percent_outstanding, short_percent_float)
                VALUES %s
                ON CONFLICT (ticker, as_of_date) DO UPDATE SET
                    shares_outstanding        = EXCLUDED.shares_outstanding,
                    shares_float              = EXCLUDED.shares_float,
                    percent_insiders          = EXCLUDED.percent_insiders,
                    percent_institutions      = EXCLUDED.percent_institutions,
                    shares_short              = EXCLUDED.shares_short,
                    shares_short_prior_month  = EXCLUDED.shares_short_prior_month,
                    short_ratio               = EXCLUDED.short_ratio,
                    short_percent_outstanding = EXCLUDED.short_percent_outstanding,
                    short_percent_float       = EXCLUDED.short_percent_float,
                    ingested_at               = NOW()
            """, rows)
        conn.commit()
    return len(rows)


def _insert_earnings_surprises(ticker_symbol: str, df: pd.DataFrame) -> int:
    """
    Insert earnings history & surprises from fundamentals Earnings::History filter.
    CSV columns: period_date, epsActual, epsEstimate, epsDifference, surprisePercent,
    revenueActual, revenueEstimate, beforeAfterMarket, currency.
    """
    if df.empty:
        return 0
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        period_date = _safe_date(
            d.get("period_date") or d.get("reportDate") or d.get("date") or d.get("Date")
        )
        if not period_date:
            continue
        # Revenue surprise % — compute if not directly provided
        rev_actual   = _safe_numeric(d.get("revenueActual"))
        rev_estimate = _safe_numeric(d.get("revenueEstimate"))
        if rev_actual is not None and rev_estimate and rev_estimate != 0:
            rev_surp = round((rev_actual - rev_estimate) / abs(rev_estimate) * 100, 4)
        else:
            rev_surp = _safe_numeric(d.get("revenueSurprise") or d.get("revenue_surprise_pct"))
        rows.append((
            ticker_symbol,
            period_date,
            _safe_numeric(d.get("epsActual")        or d.get("eps_actual")),
            _safe_numeric(d.get("epsEstimate")       or d.get("eps_estimate")),
            _safe_numeric(d.get("surprisePercent")   or d.get("epsSurprisePct") or d.get("eps_surprise_pct")),
            rev_actual,
            rev_estimate,
            rev_surp,
            str(d.get("beforeAfterMarket") or d.get("before_after_market") or ""),
            str(d.get("currency") or ""),
        ))
    if not rows:
        return 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO earnings_surprises
                    (ticker, period_date,
                     eps_actual, eps_estimate, eps_surprise_pct,
                     revenue_actual, revenue_estimate, revenue_surprise_pct,
                     before_after_market, currency)
                VALUES %s
                ON CONFLICT (ticker, period_date) DO UPDATE SET
                    eps_actual           = EXCLUDED.eps_actual,
                    eps_estimate         = EXCLUDED.eps_estimate,
                    eps_surprise_pct     = EXCLUDED.eps_surprise_pct,
                    revenue_actual       = EXCLUDED.revenue_actual,
                    revenue_estimate     = EXCLUDED.revenue_estimate,
                    revenue_surprise_pct = EXCLUDED.revenue_surprise_pct,
                    before_after_market  = EXCLUDED.before_after_market,
                    currency             = EXCLUDED.currency,
                    ingested_at          = NOW()
            """, rows)
        conn.commit()
    return len(rows)


def _insert_outstanding_shares(ticker_symbol: str, df: pd.DataFrame) -> int:
    """
    Insert outstanding shares history from fundamentals outstandingShares filter.
    CSV columns: period_type (annual|quarterly), date (or shares_date),
    sharesMln (or numberOfShares / shares_outstanding).
    """
    if df.empty:
        return 0
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        shares_date = _safe_date(d.get("date") or d.get("shares_date") or d.get("Date"))
        if not shares_date:
            continue
        period_type = str(d.get("period_type") or "unknown")
        # EODHD returns shares in millions (sharesMln) or as numberOfShares
        raw_shares = d.get("numberOfShares") or d.get("shares_outstanding") or d.get("SharesOutstanding")
        if raw_shares is None:
            shares_mln = _safe_numeric(d.get("sharesMln"))
            raw_shares = int(shares_mln * 1_000_000) if shares_mln else None
        shares = _safe_int(raw_shares)
        rows.append((ticker_symbol, period_type, shares_date, shares))
    if not rows:
        return 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO outstanding_shares
                    (ticker, period_type, shares_date, shares_outstanding)
                VALUES %s
                ON CONFLICT (ticker, period_type, shares_date) DO UPDATE SET
                    shares_outstanding = EXCLUDED.shares_outstanding,
                    ingested_at        = NOW()
            """, rows)
        conn.commit()
    return len(rows)


# ── Per-ticker loader ─────────────────────────────────────────────────────────

def load_valuation_metrics_from_profile(ticker_symbol: str) -> int:
    """
    Extract valuation / highlights data from company_profile.json and upsert
    into the valuation_metrics table.

    This is a separate step because company_profile has storage_destination
    "neo4j", so the main CSV loop skips it.  We call this from
    load_postgres_for_ticker automatically, and it can also be run standalone.
    """
    profile_path = BASE_ETL_DIR / ticker_symbol / "company_profile.json"
    if not profile_path.exists():
        # Fall back to CSV
        csv_path = BASE_ETL_DIR / ticker_symbol / "company_profile.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                return _insert_valuation_metrics(ticker_symbol, df)
            except Exception as exc:
                print(f"[PG Loader] {ticker_symbol}/valuation_from_profile CSV fallback: {exc}")
        return 0

    try:
        with open(profile_path) as f:
            profile: dict = json.load(f)
    except Exception as exc:
        print(f"[PG Loader] {ticker_symbol}/company_profile.json read error: {exc}")
        return 0

    hl  = profile.get("Highlights", {}) or {}
    val = profile.get("Valuation",  {}) or {}
    gen = profile.get("General",    {}) or {}

    row: dict = {
        "UpdatedAt":                      gen.get("UpdatedAt"),
        # Valuation section
        "Valuation_TrailingPE":           val.get("TrailingPE"),
        "Valuation_ForwardPE":            val.get("ForwardPE"),
        "Valuation_PriceSalesTTM":        val.get("PriceSalesTTM"),
        "Valuation_PriceBookMRQ":         val.get("PriceBookMRQ"),
        "Valuation_EnterpriseValue":      val.get("EnterpriseValue"),
        "Valuation_EnterpriseValueRevenue": val.get("EnterpriseValueRevenue"),
        "Valuation_EnterpriseValueEbitda": val.get("EnterpriseValueEbitda"),
        # Highlights section
        "Highlights_MarketCapitalization": hl.get("MarketCapitalization"),
        "Highlights_EBITDA":              hl.get("EBITDA"),
        "Highlights_PERatio":             hl.get("PERatio"),
        "Highlights_PEGRatio":            hl.get("PEGRatio"),
        "Highlights_WallStreetTargetPrice": hl.get("WallStreetTargetPrice"),
        "Highlights_BookValue":           hl.get("BookValue"),
        "Highlights_DividendShare":       hl.get("DividendShare"),
        "Highlights_DividendYield":       hl.get("DividendYield"),
        "Highlights_EarningsShare":       hl.get("EarningsShare"),
        "Highlights_EPSEstimateCurrentYear": hl.get("EPSEstimateCurrentYear"),
        "Highlights_EPSEstimateNextYear": hl.get("EPSEstimateNextYear"),
        "Highlights_ProfitMargin":        hl.get("ProfitMargin"),
        "Highlights_OperatingMarginTTM":  hl.get("OperatingMarginTTM"),
        "Highlights_ReturnOnAssetsTTM":   hl.get("ReturnOnAssetsTTM"),
        "Highlights_ReturnOnEquityTTM":   hl.get("ReturnOnEquityTTM"),
        "Highlights_RevenueTTM":          hl.get("RevenueTTM"),
        "Highlights_QuarterlyRevenueGrowthYOY":  hl.get("QuarterlyRevenueGrowthYOY"),
        "Highlights_QuarterlyEarningsGrowthYOY": hl.get("QuarterlyEarningsGrowthYOY"),
    }

    df = pd.DataFrame([row])
    n = _insert_valuation_metrics(ticker_symbol, df)
    print(f"[PG Loader] {ticker_symbol}/valuation_from_profile: {n} rows upserted")
    return n


# ── pgvector text chunk ingestion ────────────────────────────────────────────

def _ollama_embed_batch(texts: list[str], model: str, base_url: str) -> list[list[float]]:
    """Embed a list of texts via Ollama /api/embed. Returns one vector per text."""
    url = f"{base_url.rstrip('/')}/api/embed"
    embeddings: list[list[float]] = []
    batch_size = 16
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = requests.post(
            url,
            json={"model": model, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        batch_embeddings = data.get("embeddings")
        if not batch_embeddings:
            raise RuntimeError(
                f"Ollama /api/embed returned empty embeddings for model '{model}'. "
                "Ensure the model is pulled: `ollama pull {model}`"
            )
        embeddings.extend(batch_embeddings)
    return embeddings


def _split_text_pg(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-boundary chunks."""
    if not text or not text.strip():
        return []
    words = text.split()
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for word in words:
        wl = len(word) + 1
        if buf_len + wl > chunk_size and buf:
            chunks.append(" ".join(buf))
            carry: list[str] = []
            carry_len = 0
            for w in reversed(buf):
                if carry_len + len(w) + 1 > overlap:
                    break
                carry.insert(0, w)
                carry_len += len(w) + 1
            buf = carry
            buf_len = carry_len
        buf.append(word)
        buf_len += wl
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _insert_text_chunks_pg(
    ticker_symbol: str,
    profile: dict,
    ollama_url: str,
    model: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> int:
    """
    Build text chunks from a company_profile dict, embed them with Ollama
    (nomic-embed-text, 768-dim), and upsert into the text_chunks table.

    Returns the number of chunks upserted.
    """
    from datetime import datetime as _dt
    now_str = _dt.utcnow().strftime("%Y-%m-%d")

    chunk_dicts: list[dict] = []

    def _add(text: str, section: str) -> None:
        for i, seg in enumerate(_split_text_pg(text, chunk_size, overlap)):
            chunk_dicts.append({
                "chunk_id": f"{ticker_symbol}::{section}::{i}",
                "text": seg,
                "section": section,
                "filing_date": now_str,
            })

    gen = profile.get("General", {})
    desc = str(gen.get("Description") or "").strip()
    if desc:
        _add(desc, "description")

    officers = gen.get("Officers") or {}
    if isinstance(officers, dict):
        lines = [
            f"{v.get('Name','?')} – {v.get('Title','?')}"
            for v in officers.values() if isinstance(v, dict)
        ]
        if lines:
            _add(
                f"{gen.get('Name', ticker_symbol)} leadership: " + "; ".join(lines),
                "officers",
            )

    hl = profile.get("Highlights", {})
    if hl:
        _add(
            f"{gen.get('Name', ticker_symbol)} ({ticker_symbol}) financial highlights: "
            + " | ".join(f"{k}={v}" for k, v in hl.items() if v not in (None, "", "None", "0", 0)),
            "highlights",
        )

    val = profile.get("Valuation", {})
    if val:
        _add(
            f"{ticker_symbol} valuation metrics: "
            + " | ".join(f"{k}={v}" for k, v in val.items() if v not in (None, "", "None", "0", 0)),
            "valuation",
        )

    ar = profile.get("AnalystRatings", {})
    if ar:
        _add(
            f"{ticker_symbol} analyst ratings: "
            + " | ".join(f"{k}={v}" for k, v in ar.items() if v not in (None, "")),
            "analyst_ratings",
        )

    if not chunk_dicts:
        print(f"[PG text_chunks] No text found for {ticker_symbol} — skipping")
        return 0

    print(f"[PG text_chunks] {ticker_symbol}: embedding {len(chunk_dicts)} chunks via Ollama …")
    texts = [c["text"] for c in chunk_dicts]
    embeddings = _ollama_embed_batch(texts, model, ollama_url)

    # Build rows for execute_values: (ticker, chunk_id, text, section, filing_date, embedding::text, source)
    # pgvector accepts the vector as a string like '[0.1, 0.2, ...]'
    rows = []
    for c, emb in zip(chunk_dicts, embeddings):
        emb_str = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"
        rows.append((
            ticker_symbol,
            c["chunk_id"],
            c["text"],
            c.get("section"),
            c.get("filing_date"),
            emb_str,
            "eodhd",
        ))

    sql = """
        INSERT INTO text_chunks (ticker, chunk_id, text, section, filing_date, embedding, source)
        VALUES %s
        ON CONFLICT (chunk_id) DO UPDATE SET
            text        = EXCLUDED.text,
            section     = EXCLUDED.section,
            filing_date = EXCLUDED.filing_date,
            embedding   = EXCLUDED.embedding,
            ingested_at = NOW()
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur, sql, rows,
                template="(%s, %s, %s, %s, %s, %s::vector, %s)",
            )
        conn.commit()
    print(f"[PG text_chunks] {ticker_symbol}: upserted {len(rows)} chunks")
    return len(rows)


def _populate_raw_fundamentals_from_statements(ticker_symbol: str) -> int:
    ticker_dir = BASE_ETL_DIR / ticker_symbol
    total = 0
    today = date.today().isoformat()

    # ── 1. Extract from financial_statements.json ──
    fin_stmt_path = ticker_dir / "financial_statements.json"
    if fin_stmt_path.exists():
        try:
            with open(fin_stmt_path) as f:
                fin_data = json.load(f)
            
            financials = fin_data
            
            # Use a dict to deduplicate by (data_name, as_of_date)
            # Key format: (data_name, period_date, period_type)
            rows_dict = {}
            
            # Income Statement
            income_stmt_q = financials.get("Income_Statement", {}).get("quarterly", {})
            income_stmt_y = financials.get("Income_Statement", {}).get("yearly", {})
            
            for period_date, stmt in income_stmt_q.items():
                if stmt:
                    key = ("income_statement", "quarterly", period_date)
                    rows_dict[key] = (
                        ticker_symbol,
                        "income_statement",
                        "quarterly",
                        _safe_date(period_date) or today,
                        json.dumps({"period": period_date, "period_type": "quarterly", **stmt}),
                        "eodhd",
                    )
            
            for period_date, stmt in income_stmt_y.items():
                if stmt:
                    key = ("income_statement", "yearly", period_date)
                    rows_dict[key] = (
                        ticker_symbol,
                        "income_statement",
                        "yearly",
                        _safe_date(period_date) or today,
                        json.dumps({"period": period_date, "period_type": "yearly", **stmt}),
                        "eodhd",
                    )
            
            # Balance Sheet
            balance_q = financials.get("Balance_Sheet", {}).get("quarterly", {})
            balance_y = financials.get("Balance_Sheet", {}).get("yearly", {})
            
            for period_date, stmt in balance_q.items():
                if stmt:
                    key = ("balance_sheet", "quarterly", period_date)
                    rows_dict[key] = (
                        ticker_symbol,
                        "balance_sheet",
                        "quarterly",
                        _safe_date(period_date) or today,
                        json.dumps({"period": period_date, "period_type": "quarterly", **stmt}),
                        "eodhd",
                    )
            
            for period_date, stmt in balance_y.items():
                if stmt:
                    key = ("balance_sheet", "yearly", period_date)
                    rows_dict[key] = (
                        ticker_symbol,
                        "balance_sheet",
                        "yearly",
                        _safe_date(period_date) or today,
                        json.dumps({"period": period_date, "period_type": "yearly", **stmt}),
                        "eodhd",
                    )
            
            # Cash Flow
            cashflow_q = financials.get("Cash_Flow", {}).get("quarterly", {})
            cashflow_y = financials.get("Cash_Flow", {}).get("yearly", {})
            
            for period_date, stmt in cashflow_q.items():
                if stmt:
                    key = ("cash_flow", "quarterly", period_date)
                    rows_dict[key] = (
                        ticker_symbol,
                        "cash_flow",
                        "quarterly",
                        _safe_date(period_date) or today,
                        json.dumps({"period": period_date, "period_type": "quarterly", **stmt}),
                        "eodhd",
                    )
            
            for period_date, stmt in cashflow_y.items():
                if stmt:
                    key = ("cash_flow", "yearly", period_date)
                    rows_dict[key] = (
                        ticker_symbol,
                        "cash_flow",
                        "yearly",
                        _safe_date(period_date) or today,
                        json.dumps({"period": period_date, "period_type": "yearly", **stmt}),
                        "eodhd",
                    )
            
            # Use individual inserts instead of execute_values due to JSON handling issues
            inserted = 0
            if rows_dict:
                try:
                    with get_pg_conn() as conn:
                        with conn.cursor() as cur:
                            for k, v in rows_dict.items():
                                ticker = v[0]
                                data_name = v[1]
                                period_type = v[2]
                                as_of_date = v[3]
                                payload = v[4]
                                source = v[5]
                                cur.execute("""
                                    INSERT INTO raw_fundamentals
                                        (ticker_symbol, data_name, period_type, as_of_date, payload, source)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (ticker_symbol, data_name, period_type, as_of_date, source) DO UPDATE SET
                                        payload = EXCLUDED.payload,
                                        ingested_at = NOW()
                                """, (ticker, data_name, period_type, as_of_date, payload, source))
                                inserted += 1
                        conn.commit()
                    total += inserted
                    print(f"[PG raw_fundamentals] {ticker_symbol}: {inserted} statement rows upserted")
                except Exception as stmt_exc:
                    print(f"[PG raw_fundamentals] {ticker_symbol}/financial_statements: ERROR — {stmt_exc}")
                    raise
        
        except Exception as exc:
            print(f"[PG raw_fundamentals] {ticker_symbol}/financial_statements: ERROR — {exc}")
    
    # ── 2. Extract ratios/metrics from company_profile.json (unchanged) ──
    ticker_dir = BASE_ETL_DIR / ticker_symbol
    profile_path = ticker_dir / "company_profile.json"
    if profile_path.exists():
        try:
            with open(profile_path) as f:
                profile = json.load(f)
            
            highlights = profile.get("Highlights", {}) or {}
            valuation = profile.get("Valuation", {}) or {}
            general = profile.get("General", {}) or {}
            
            rows_metrics = []
            as_of = _safe_date(general.get("UpdatedAt")) or today
            
            # financial_ratios: subset of Highlights
            if highlights:
                financial_ratios = {
                    "PERatio": highlights.get("PERatio"),
                    "PEGRatio": highlights.get("PEGRatio"),
                    "ProfitMargin": highlights.get("ProfitMargin"),
                    "OperatingMarginTTM": highlights.get("OperatingMarginTTM"),
                    "ReturnOnAssetsTTM": highlights.get("ReturnOnAssetsTTM"),
                    "ReturnOnEquityTTM": highlights.get("ReturnOnEquityTTM"),
                    "DividendYield": highlights.get("DividendYield"),
                }
                rows_metrics.append((
                    ticker_symbol,
                    "financial_ratios",
                    None,  # period_type - null for snapshot metrics
                    as_of,
                    json.dumps({k: v for k, v in financial_ratios.items() if v is not None}),
                    "eodhd",
                ))
            
            # ratios_ttm: merge Highlights + Valuation TTM fields
            if highlights or valuation:
                ratios_ttm = {
                    **{k: v for k, v in highlights.items() if "TTM" in k or "Ratio" in k},
                    **{k: v for k, v in valuation.items() if "TTM" in k or k.startswith("Price")},
                }
                if ratios_ttm:
                    rows_metrics.append((
                        ticker_symbol,
                        "ratios_ttm",
                        None,  # period_type
                        as_of,
                        json.dumps(ratios_ttm),
                        "eodhd",
                    ))
            
            # key_metrics_ttm: full Highlights section
            if highlights:
                rows_metrics.append((
                    ticker_symbol,
                    "key_metrics_ttm",
                    None,  # period_type
                    as_of,
                    json.dumps(highlights),
                    "eodhd",
                ))
            
            # enterprise_values: Valuation section
            if valuation:
                rows_metrics.append((
                    ticker_symbol,
                    "enterprise_values",
                    None,  # period_type
                    as_of,
                    json.dumps(valuation),
                    "eodhd",
                ))
            
            # financial_scores: placeholder
            rows_metrics.append((
                ticker_symbol,
                "financial_scores",
                None,  # period_type
                as_of,
                json.dumps({
                    "piotroskiScore": None,
                    "altmanZScore": None,
                    "beneishMScore": None,
                    "note": "Not available from EODHD"
                }),
                "eodhd",
            ))
            
            if rows_metrics:
                inserted = 0
                try:
                    with get_pg_conn() as conn:
                        with conn.cursor() as cur:
                            for row in rows_metrics:
                                cur.execute("""
                                    INSERT INTO raw_fundamentals
                                        (ticker_symbol, data_name, period_type, as_of_date, payload, source)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (ticker_symbol, data_name, period_type, as_of_date, source) DO UPDATE SET
                                        payload = EXCLUDED.payload,
                                        ingested_at = NOW()
                                """, row)
                                inserted += 1
                        conn.commit()
                    total += inserted
                    print(f"[PG raw_fundamentals] {ticker_symbol}: {inserted} metrics rows upserted")
                except Exception as exc:
                    print(f"[PG raw_fundamentals] {ticker_symbol}/company_profile: ERROR — {exc}")
                    raise
        
        except Exception as exc:
            print(f"[PG raw_fundamentals] {ticker_symbol}/company_profile: ERROR — {exc}")
    
    return total



def load_postgres_for_ticker(ticker_symbol: str) -> int:
    ticker_dir    = BASE_ETL_DIR / ticker_symbol
    metadata_path = ticker_dir / "metadata.json"

    if not metadata_path.exists():
        print(f"[PG Loader] No metadata.json for {ticker_symbol} — skipping")
        return 0

    with open(metadata_path) as f:
        metadata: dict = json.load(f)

    ensure_tables()
    total = 0

    for data_name, info in metadata.items():
        if info.get("storage_destination") != "postgresql":
            continue

        csv_path = ticker_dir / f"{data_name}.csv"
        if not csv_path.exists():
            print(f"[PG Loader] {ticker_symbol}/{data_name}: no CSV — skipping")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[PG Loader] {ticker_symbol}/{data_name}: read error — {exc}")
            continue

        source = info.get("source", "eodhd")

        try:
            if data_name == "sentiment_trends":
                n = _insert_sentiment(ticker_symbol, df)
            elif data_name == "insider_transactions":
                n = _insert_insider_transactions(ticker_symbol, df)
            elif data_name == "institutional_holders":
                n = _insert_institutional_holders(ticker_symbol, df)
            elif data_name == "financial_calendar":
                n = _insert_financial_calendar(df)
            elif data_name == "dividends_history":
                n = _insert_dividends_history(ticker_symbol, df)
            elif data_name == "splits_history":
                n = _insert_splits_history(ticker_symbol, df)
            elif data_name == "financial_statements":
                n = _insert_financial_statements(ticker_symbol, df)
            elif data_name == "valuation_metrics":
                n = _insert_valuation_metrics(ticker_symbol, df)
            elif data_name == "company_profile":
                # company_profile CSV has flattened Highlights + Valuation columns:
                # insert into both valuation_metrics (typed) and raw_fundamentals (generic)
                n = _insert_valuation_metrics(ticker_symbol, df)
                n += _insert_dataframe_generic(df, ticker_symbol, data_name, source)
            elif data_name == "short_interest":
                n = _insert_short_interest(ticker_symbol, df)
            elif data_name == "earnings_surprises":
                n = _insert_earnings_surprises(ticker_symbol, df)
            elif data_name == "outstanding_shares":
                n = _insert_outstanding_shares(ticker_symbol, df)
            else:
                n = _insert_dataframe_generic(df, ticker_symbol, data_name, source)

            total += n
            print(f"[PG Loader] {ticker_symbol}/{data_name}: {n} rows upserted")
        except Exception as exc:
            print(f"[PG Loader] {ticker_symbol}/{data_name}: ERROR — {exc}")

    # Always attempt to populate valuation_metrics from company_profile.json
    # (company_profile is neo4j-destined, so the loop above skips it)
    try:
        vm_n = load_valuation_metrics_from_profile(ticker_symbol)
        total += vm_n
    except Exception as exc:
        print(f"[PG Loader] {ticker_symbol}/valuation_from_profile: ERROR — {exc}")
    # Populate raw_fundamentals with decomposed statement views
    try:
        rf_n = _populate_raw_fundamentals_from_statements(ticker_symbol)
        total += rf_n
    except Exception as exc:
        print(f"[PG Loader] {ticker_symbol}/raw_fundamentals: ERROR — {exc}")

        
    # Embed and upsert text chunks into pgvector text_chunks table
    profile_path = BASE_ETL_DIR / ticker_symbol / "company_profile.json"
    if profile_path.exists():
        try:
            with open(profile_path) as f:
                profile_json: dict = json.load(f)
            chunk_n = _insert_text_chunks_pg(
                ticker_symbol,
                profile_json,
                ollama_url=OLLAMA_BASE_URL,
                model=OLLAMA_EMBED_MODEL,
            )
            total += chunk_n
        except Exception as exc:
            print(f"[PG Loader] {ticker_symbol}/text_chunks: WARNING — {exc}")
    return total



# ── Macro loader ──────────────────────────────────────────────────────────────

def _insert_market_eod_us(df: "pd.DataFrame") -> int:
    """Insert S&P 500 EOD rows (from market_sp500_eod CSV) into market_eod_us.

    Expected CSV columns from EODHD EOD endpoint:
        date, open, high, low, close, adjusted_close, volume
    Each row becomes one JSONB payload keyed on the date.
    """
    rows = []
    for _, row in df.iterrows():
        d = row.to_dict()
        # Normalise NaN → None for JSON serialisation
        payload = {k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                   for k, v in d.items()}
        ts_date = d.get("date") or d.get("Date") or d.get("ts_date")
        if not ts_date:
            continue
        rows.append((str(ts_date), json.dumps(payload), "eodhd"))

    if not rows:
        return 0

    sql = """
        INSERT INTO market_eod_us (ts_date, payload, source)
        VALUES %s
        ON CONFLICT (ts_date, source) DO UPDATE SET
            payload     = EXCLUDED.payload,
            ingested_at = NOW()
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows)
        conn.commit()
    return len(rows)


def load_postgres_macro() -> int:
    """
    Load all macro / global data from agent_data/_MACRO/ into PostgreSQL.
    Called by the dedicated eodhd_load_postgres_macro Airflow task.
    """
    macro_dir     = BASE_ETL_DIR / _MACRO_TICKER
    metadata_path = macro_dir / "metadata.json"

    if not metadata_path.exists():
        print(f"[PG Loader Macro] No metadata.json in _MACRO — skipping")
        return 0

    with open(metadata_path) as f:
        metadata: dict = json.load(f)

    ensure_tables()
    total = 0

    for data_name, info in metadata.items():
        if info.get("storage_destination") != "postgresql":
            continue

        csv_path = macro_dir / f"{data_name}.csv"
        if not csv_path.exists():
            print(f"[PG Loader Macro] {data_name}: no CSV — skipping")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[PG Loader Macro] {data_name}: read error — {exc}")
            continue

        try:
            if data_name == "financial_calendar":
                n = _insert_financial_calendar(df)
            elif data_name == "calendar_ipo":
                n = _insert_financial_calendar(df, event_type_override="ipo")
            elif data_name == "calendar_splits":
                n = _insert_financial_calendar(df, event_type_override="split")
            elif data_name == "calendar_dividends":
                n = _insert_financial_calendar(df, event_type_override="dividend")
            elif data_name in ("treasury_rates",
                               "treasury_rates_3m", "treasury_rates_6m",
                               "treasury_rates_1y", "treasury_rates_2y",
                               "treasury_rates_5y", "treasury_rates_20y",
                               "treasury_rates_30y"):
                # Derive indicator name from data_name (e.g. "treasury_rates_5y" → "US5Y")
                _tenor_map = {
                    "treasury_rates":     "US10Y",
                    "treasury_rates_3m":  "US3M",
                    "treasury_rates_6m":  "US6M",
                    "treasury_rates_1y":  "US1Y",
                    "treasury_rates_2y":  "US2Y",
                    "treasury_rates_5y":  "US5Y",
                    "treasury_rates_20y": "US20Y",
                    "treasury_rates_30y": "US30Y",
                }
                n = _insert_treasury_rates(df, indicator_override=_tenor_map.get(data_name, "US10Y"))
            elif data_name in ("economic_indicators_gdp",
                               "economic_indicators_cpi",
                               "economic_indicators_unemployment"):
                indicator = data_name.replace("economic_indicators_", "")
                n = _insert_macro_indicators(indicator, df)
            elif data_name == "economic_events":
                n = _insert_economic_events(df)
            elif data_name == "screener_bulk":
                n = _insert_screener_bulk(df)
            elif data_name == "forex_historical_rates":
                # Derive pair from data_name (e.g. "forex_historical_rates" → EURUSD by convention)
                # Additional forex pairs can follow naming "forex_historical_rates_GBPUSD" etc.
                _pair = "EURUSD"
                if data_name.startswith("forex_historical_rates_"):
                    _pair = data_name.replace("forex_historical_rates_", "").upper()
                n = _insert_forex_rates(df, pair_name=_pair)
            elif data_name == "etf_index_constituents":
                # Store ETF/index constituents as raw_timeseries rows keyed by SPY (the ETF ticker)
                # Deduplicate by code to avoid duplicate key errors
                rows_etf = []
                seen = set()
                base_ts = datetime.utcnow()
                for idx, (_, row) in enumerate(df.iterrows()):
                    d = row.to_dict()
                    code = d.get("key") or d.get("Code") or d.get("ticker") or d.get("symbol") or ""
                    if code in seen:
                        continue
                    seen.add(code)
                    payload = {k: (None if (isinstance(v, float) and math.isnan(v)) else v)
                               for k, v in d.items()}
                    payload['_constituent_code'] = code
                    # Use base timestamp + microsecond offset to ensure uniqueness
                    ts = base_ts.replace(microsecond=base_ts.microsecond + idx * 1000)
                    rows_etf.append(("SPY", "etf_index_constituents", ts, json.dumps(payload), "eodhd", ts))
                if rows_etf:
                    with get_pg_conn() as conn:
                        with conn.cursor() as cur:
                            execute_values(cur, """
                                INSERT INTO raw_timeseries
                                    (ticker_symbol, data_name, ts_date, payload, source, ingested_at)
                                VALUES %s
                                ON CONFLICT (ticker_symbol, data_name, ts_date, source) DO UPDATE SET
                                    payload = EXCLUDED.payload,
                                    ingested_at = NOW()
                            """, rows_etf)
                        conn.commit()
                n = len(rows_etf)
            elif data_name == "corporate_bond_yields":
                n = _insert_corporate_bond_eod(df, ticker="LQD")
            elif data_name in ("bond_aapl_fundamentals", "bond_amzn_fundamentals"):
                _ticker = "US037833AK68" if data_name == "bond_aapl_fundamentals" else "US023135BX34"
                n = _insert_bond_fundamentals(df, ticker=_ticker)
            elif data_name == "market_sp500_eod":
                n = _insert_market_eod_us(df)
            else:
                n = _insert_dataframe_generic(df, "_GLOBAL", data_name, "eodhd")

            total += n
            print(f"[PG Loader Macro] {data_name}: {n} rows upserted")
        except Exception as exc:
            print(f"[PG Loader Macro] {data_name}: ERROR — {exc}")

    return total


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    # Load .env for local runs
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as ef:
            for line in ef:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())

    # Re-read PG env after loading .env
    PG_HOST     = os.getenv("POSTGRES_HOST", "localhost")
    PG_PORT     = int(os.getenv("POSTGRES_PORT", "5432"))
    PG_DB       = os.getenv("POSTGRES_DB", "airflow")
    PG_USER     = os.getenv("POSTGRES_USER", "airflow")
    PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")

    parser = argparse.ArgumentParser(description="Load EODHD agent data into PostgreSQL")
    parser.add_argument("ticker", nargs="?", default="AAPL",
                        help="Ticker symbol to load (default: AAPL)")
    parser.add_argument("--macro", action="store_true",
                        help="Load _MACRO data instead of a ticker")
    parser.add_argument("--all", action="store_true",
                        help="Load all tickers + macro")
    args = parser.parse_args()

    if args.all:
        tickers = os.getenv("TRACKED_TICKERS", "AAPL,TSLA,NVDA,MSFT,GOOGL").split(",")
        for t in tickers:
            print(f"\n=== Loading {t.strip()} ===")
            print(f"  rows: {load_postgres_for_ticker(t.strip())}")
        print("\n=== Loading _MACRO ===")
        print(f"  rows: {load_postgres_macro()}")
    elif args.macro:
        print(load_postgres_macro())
    else:
        print(load_postgres_for_ticker(args.ticker))
