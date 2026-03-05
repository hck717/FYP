# ingestion/etl/load_postgres.py
import os
import json
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

BASE_ETL_DIR = Path(os.getenv("BASE_ETL_DIR", "/opt/airflow/etl/agent_data"))

PG_HOST     = os.getenv("POSTGRES_HOST",     "postgres")
PG_PORT     = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB       = os.getenv("POSTGRES_DB",       "airflow")
PG_USER     = os.getenv("POSTGRES_USER",     "airflow")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")

# Global datasets — store once per day in shared tables, not per ticker
GLOBAL_ONCE_PER_DAY = {"bulk_eod_us", "economic_calendar", "ipo_calendar"}

GLOBAL_TABLES = {
    "bulk_eod_us":       "market_eod_us",
    "economic_calendar": "global_economic_calendar",
    "ipo_calendar":      "global_ipo_calendar",
}

# Macro indicator data_names — stored under _MACRO pseudo-ticker by EODHD scraper
_MACRO_TICKER = "_MACRO"
_MACRO_DATA_NAMES = {
    "economic_indicators_gdp",
    "economic_indicators_cpi",
    "economic_indicators_unemployment",
}
# Only load macro data once — when processing this anchor ticker
_MACRO_ANCHOR_TICKER = "AAPL"

# EODHD /sentiments field mapping → sentiment_trends columns
# EODHD returns: {"sentiment": {"polarity": 0.12, "neg": 0.15, "neu": 0.67, "pos": 0.18}}
# tools.py expects: bullish_pct, bearish_pct, neutral_pct
_SENTIMENT_FIELD_MAP = {
    "pos": "bullish_pct",
    "neg": "bearish_pct",
    "neu": "neutral_pct",
}


def get_pg_conn():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD,
    )


def ensure_tables():
    """
    FIX 3: Added sentiment_trends table.
    BEFORE: missing — tools.py SELECT from sentiment_trends always raised
            'relation sentiment_trends does not exist'
    AFTER:  table created with (ticker, date) unique constraint matching
            the query in tools.py fetch_sentiment().
    """
    ddl = """
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

    CREATE TABLE IF NOT EXISTS market_eod_us (
        id          SERIAL PRIMARY KEY,
        ts_date     TIMESTAMP NOT NULL,
        payload     JSONB     NOT NULL,
        source      TEXT      NOT NULL,
        ingested_at TIMESTAMP DEFAULT NOW(),
        UNIQUE (ts_date, source)
    );

    CREATE TABLE IF NOT EXISTS global_economic_calendar (
        id          SERIAL PRIMARY KEY,
        ts_date     TIMESTAMP,
        payload     JSONB     NOT NULL,
        source      TEXT      NOT NULL,
        ingested_at TIMESTAMP DEFAULT NOW(),
        UNIQUE (ts_date, source)
    );

    CREATE TABLE IF NOT EXISTS global_ipo_calendar (
        id          SERIAL PRIMARY KEY,
        ts_date     TIMESTAMP,
        payload     JSONB     NOT NULL,
        source      TEXT      NOT NULL,
        ingested_at TIMESTAMP DEFAULT NOW(),
        UNIQUE (ts_date, source)
    );

    CREATE TABLE IF NOT EXISTS global_macro_indicators (
        id          SERIAL PRIMARY KEY,
        indicator   TEXT      NOT NULL,
        ts_date     TIMESTAMP,
        payload     JSONB     NOT NULL,
        source      TEXT      NOT NULL,
        ingested_at TIMESTAMP DEFAULT NOW(),
        UNIQUE (indicator, ts_date, source)
    );
    CREATE INDEX IF NOT EXISTS idx_macro_indicators_indicator_date
        ON global_macro_indicators (indicator, ts_date DESC);

    CREATE TABLE IF NOT EXISTS sentiment_trends (
        id           SERIAL PRIMARY KEY,
        ticker       VARCHAR(10)  NOT NULL,
        bullish_pct  NUMERIC,
        bearish_pct  NUMERIC,
        neutral_pct  NUMERIC,
        trend        VARCHAR(20)  DEFAULT 'unknown',
        as_of_date   DATE         NOT NULL,
        ingested_at  TIMESTAMP    DEFAULT NOW(),
        UNIQUE (ticker, as_of_date)
    );
    CREATE INDEX IF NOT EXISTS idx_sentiment_trends_ticker_date
        ON sentiment_trends (ticker, as_of_date DESC);

    -- query_logs: stores full agent session outputs (used by critic_dag.py)
    -- CREATE TABLE creates it fresh on new installs; ALTER TABLE adds missing
    -- columns to any pre-existing version of this table.
    CREATE TABLE IF NOT EXISTS query_logs (
        id            SERIAL PRIMARY KEY,
        recorded_at   TIMESTAMP    DEFAULT NOW()
    );
    ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS session_id    TEXT;
    ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS agent_name    TEXT;
    ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS ticker_symbol TEXT;
    ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS query_text    TEXT;
    ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS agent_outputs JSONB;
    CREATE INDEX IF NOT EXISTS idx_query_logs_session
        ON query_logs (session_id, recorded_at DESC);

    -- citation_tracking: tracks which Qdrant chunks were cited per run
    -- (used by critic_dag.py NLI check and qdrant_boost_dag.py)
    CREATE TABLE IF NOT EXISTS citation_tracking (
        id            SERIAL PRIMARY KEY,
        run_id        TEXT         NOT NULL,
        was_cited     BOOLEAN      DEFAULT FALSE,
        recorded_at   TIMESTAMP    DEFAULT NOW()
    );
    ALTER TABLE citation_tracking ADD COLUMN IF NOT EXISTS session_id  TEXT;
    ALTER TABLE citation_tracking ADD COLUMN IF NOT EXISTS agent_name  TEXT;
    ALTER TABLE citation_tracking ADD COLUMN IF NOT EXISTS chunk_id    TEXT;
    ALTER TABLE citation_tracking ADD COLUMN IF NOT EXISTS chunk_text  TEXT;
    CREATE INDEX IF NOT EXISTS idx_citation_tracking_run_id
        ON citation_tracking (run_id, recorded_at DESC);
    CREATE INDEX IF NOT EXISTS idx_citation_tracking_chunk_id
        ON citation_tracking (chunk_id);

    -- agent_run_telemetry: stores per-run metadata for the critic_dag
    CREATE TABLE IF NOT EXISTS agent_run_telemetry (
        id            SERIAL PRIMARY KEY,
        run_id        TEXT         NOT NULL,
        recorded_at   TIMESTAMP    DEFAULT NOW()
    );
    ALTER TABLE agent_run_telemetry ADD COLUMN IF NOT EXISTS session_id    TEXT;
    ALTER TABLE agent_run_telemetry ADD COLUMN IF NOT EXISTS agent_name    TEXT;
    ALTER TABLE agent_run_telemetry ADD COLUMN IF NOT EXISTS ticker_symbol TEXT;
    ALTER TABLE agent_run_telemetry ADD COLUMN IF NOT EXISTS status        TEXT;
    ALTER TABLE agent_run_telemetry ADD COLUMN IF NOT EXISTS duration_sec  NUMERIC;
    CREATE INDEX IF NOT EXISTS idx_agent_run_telemetry_run_id
        ON agent_run_telemetry (run_id, recorded_at DESC);
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()

    # Create materialized view for pre-computed factor scores
    # (used by quant_fundamental/tools.py fetch_factor_scores_from_mv)
    # The MV joins raw_fundamentals for piotroski/altman/beneish scores
    # from financial_scores + key_metrics_ttm + ratios_ttm.
    _ensure_mv_daily_factor_scores()


def _ensure_mv_daily_factor_scores():
    """Create (or refresh) the mv_daily_factor_scores materialized view.

    Schema matches exactly what quant_fundamental/tools.py expects:
      ticker, as_of_date, piotroski_score, altman_z_score, beneish_m_score,
      roe_ttm, roa_ttm, roic_ttm, gross_margin_ttm, net_margin_ttm,
      debt_to_equity_ttm, current_ratio_ttm, refreshed_at
    """
    create_mv_sql = """
    CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_factor_scores AS
    SELECT
        fs.ticker_symbol                                        AS ticker,
        fs.as_of_date,
        -- Piotroski F-Score from financial_scores payload
        (fs.payload->>'piotroskiScore')::NUMERIC                AS piotroski_score,
        -- Altman Z-Score from financial_scores payload
        (fs.payload->>'altmanZScore')::NUMERIC                  AS altman_z_score,
        -- Beneish M-Score: not stored separately, use NULL placeholder
        NULL::NUMERIC                                           AS beneish_m_score,
        -- TTM profitability ratios from ratios_ttm
        (rt.payload->>'returnOnEquityTTM')::NUMERIC             AS roe_ttm,
        (rt.payload->>'returnOnAssetsTTM')::NUMERIC             AS roa_ttm,
        (rt.payload->>'returnOnCapitalEmployedTTM')::NUMERIC    AS roic_ttm,
        (rt.payload->>'grossProfitMarginTTM')::NUMERIC          AS gross_margin_ttm,
        (rt.payload->>'netProfitMarginTTM')::NUMERIC            AS net_margin_ttm,
        (rt.payload->>'debtEquityRatioTTM')::NUMERIC            AS debt_to_equity_ttm,
        (rt.payload->>'currentRatioTTM')::NUMERIC               AS current_ratio_ttm,
        NOW()                                                   AS refreshed_at
    FROM raw_fundamentals fs
    LEFT JOIN raw_fundamentals rt
           ON rt.ticker_symbol = fs.ticker_symbol
          AND rt.data_name     = 'ratios_ttm'
          AND rt.as_of_date    = (
                SELECT MAX(as_of_date)
                FROM raw_fundamentals
                WHERE ticker_symbol = fs.ticker_symbol
                  AND data_name = 'ratios_ttm'
              )
    WHERE fs.data_name = 'financial_scores'
    WITH DATA;
    """
    index_sql = """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_factor_scores_ticker_date
        ON mv_daily_factor_scores (ticker, as_of_date DESC);
    """
    refresh_sql = "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_factor_scores;"

    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(create_mv_sql)
                conn.commit()
                print("[Postgres Loader] Created mv_daily_factor_scores")
            except Exception as e:
                conn.rollback()
                if "already exists" in str(e):
                    print("[Postgres Loader] mv_daily_factor_scores already exists — will refresh")
                    # Do NOT return — fall through to refresh so scores stay current
                else:
                    print(f"[Postgres Loader] Warning creating MV: {e}")
                    return  # Unexpected error — skip refresh

    # Create unique index (needed for CONCURRENTLY refresh) — separate transaction
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(index_sql)
                conn.commit()
            except Exception:
                conn.rollback()

    # Refresh to populate with current data
    with get_pg_conn() as conn:
        conn.set_isolation_level(0)  # autocommit — REFRESH CONCURRENTLY cannot run in txn
        with conn.cursor() as cur:
            try:
                cur.execute(refresh_sql)
                print("[Postgres Loader] Refreshed mv_daily_factor_scores")
            except Exception as e:
                print(f"[Postgres Loader] Warning refreshing MV (may be empty): {e}")


def _already_loaded_today(data_name: str, table: str) -> bool:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM {table} WHERE ingested_at::date = CURRENT_DATE"
            )
            return cur.fetchone()[0] > 0


def _detect_date_col(df: pd.DataFrame):
    for candidate in ["datetime", "date", "timestamp", "reportedDate", "t"]:
        if candidate in df.columns:
            return candidate
    return None


def _normalise_date_col(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Convert Unix int/float timestamps to ISO string."""
    if pd.api.types.is_integer_dtype(df[date_col]) or pd.api.types.is_float_dtype(df[date_col]):
        df[date_col] = (
            pd.to_datetime(df[date_col], unit='s')
            .dt.strftime('%Y-%m-%d %H:%M:%S')
        )
    return df


def _insert_global(df: pd.DataFrame, data_name: str, source: str) -> int:
    """Insert global datasets into shared tables — once per day, deduped."""
    if df.empty:
        return 0

    table = GLOBAL_TABLES[data_name]

    if _already_loaded_today(data_name, table):
        print(f"[Postgres Loader] {table} already loaded today — skipping")
        return 0

    df = df.copy()
    date_col = _detect_date_col(df)
    if date_col:
        df = _normalise_date_col(df, date_col)

    payload_cols = [c for c in df.columns if c != date_col]

    rows_dict = {}
    for _, row in df.iterrows():
        ts_val  = row[date_col] if date_col else None
        payload = {c: (None if pd.isna(row[c]) else row[c]) for c in payload_cols}
        key = (str(ts_val), source)
        rows_dict[key] = (ts_val, json.dumps(payload), source)

    rows = list(rows_dict.values())
    if not rows:
        return 0

    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            sql = f"""
                INSERT INTO {table} (ts_date, payload, source)
                VALUES %s
                ON CONFLICT (ts_date, source)
                DO UPDATE SET payload     = EXCLUDED.payload,
                              ingested_at = NOW();
            """
            execute_values(cur, sql, rows)
        conn.commit()

    print(f"[Postgres Loader] {table}: inserted {len(rows)} rows (global, deduped)")
    return len(rows)


def _insert_sentiment(ticker_symbol: str, df: pd.DataFrame) -> int:
    """
    FIX 3: Dedicated upsert path for sentiment_trends.

    EODHD /sentiments endpoint returns either:
      A) A dict: {"sentiment": {"polarity": 0.12, "neg": 0.15, "neu": 0.67, "pos": 0.18}}
         → saved as a single-row CSV with columns: polarity, neg, neu, pos
      B) A list of dicts (date-keyed) with same structure per row.

    Maps EODHD field names → sentiment_trends column names:
      pos → bullish_pct  (multiply by 100 for percentage)
      neg → bearish_pct
      neu → neutral_pct

    tools.py fetch_sentiment() queries:
      SELECT bullish_pct, bearish_pct, neutral_pct, as_of_date
      FROM sentiment_trends WHERE ticker = %s ORDER BY as_of_date DESC LIMIT 1
    """
    if df.empty:
        return 0

    rows = []
    today = date.today().isoformat()

    # Handle both flat and nested column layouts:
    #   Schema A (FMP/legacy): columns include pos, neg, neu  (0–1 fractions)
    #   Schema B (EODHD /sentiments): columns are date, count, normalized  (0–1 composite)
    #     normalized ≈ 0.5 → neutral; >0.6 → bullish; <0.4 → bearish
    for _, row in df.iterrows():
        row_dict = row.to_dict()

        # Try to extract date from row, fall back to today
        row_date = (
            row_dict.get("date")
            or row_dict.get("datetime")
            or row_dict.get("timestamp")
            or today
        )
        # Normalise date to just YYYY-MM-DD string
        try:
            row_date = str(row_date)[:10]
        except Exception:
            row_date = today

        # ----------------------------------------------------------------
        # Schema B — EODHD /sentiments: single 'normalized' composite score
        # normalized is 0–1 where 0.5 = perfectly neutral.
        # Decompose into bullish/neutral/bearish bands:
        #   bullish  = max(0, normalized - 0.5) * 200   → 0–100
        #   bearish  = max(0, 0.5 - normalized) * 200   → 0–100
        #   neutral  = 100 - bullish - bearish
        # ----------------------------------------------------------------
        if "normalized" in row_dict and "pos" not in row_dict and "neg" not in row_dict:
            try:
                norm = float(row_dict["normalized"])
            except (TypeError, ValueError):
                continue
            bullish = round(max(0.0, norm - 0.5) * 200, 4)
            bearish = round(max(0.0, 0.5 - norm) * 200, 4)
            neutral = round(100.0 - bullish - bearish, 4)
        else:
            # ----------------------------------------------------------------
            # Schema A — FMP / legacy: explicit pos, neg, neu columns
            # ----------------------------------------------------------------
            # Map EODHD field names to column names
            # Support both 'pos' and 'sentiment_pos' prefixed versions
            def _get_pct(key: str) -> float:
                """Try both raw key and 'sentiment_' prefixed key; multiply by 100."""
                raw = (
                    row_dict.get(key)
                    or row_dict.get(f"sentiment_{key}")
                    or row_dict.get(f"sentiment.{key}")
                )
                try:
                    val = float(raw)
                    # EODHD returns 0.0–1.0 fractions — convert to 0–100 percentage
                    return round(val * 100, 4) if val <= 1.0 else round(val, 4)
                except (TypeError, ValueError):
                    return 0.0

            bullish = _get_pct("pos")
            bearish = _get_pct("neg")
            neutral = _get_pct("neu")

        # Skip rows where all values are zero (likely header/empty rows)
        if bullish == 0.0 and bearish == 0.0 and neutral == 0.0:
            continue

        # Derive trend from bullish/bearish spread
        if bullish > bearish + 10:
            trend = "improving"
        elif bearish > bullish + 10:
            trend = "deteriorating"
        else:
            trend = "stable"

        rows.append((ticker_symbol, bullish, bearish, neutral, trend, row_date))

    if not rows:
        print(f"[Postgres Loader] sentiment_trends: no valid rows for {ticker_symbol}")
        return 0

    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            sql = """
                INSERT INTO sentiment_trends
                    (ticker, bullish_pct, bearish_pct, neutral_pct, trend, as_of_date)
                VALUES %s
                ON CONFLICT (ticker, as_of_date)
                DO UPDATE SET
                    bullish_pct = EXCLUDED.bullish_pct,
                    bearish_pct = EXCLUDED.bearish_pct,
                    neutral_pct = EXCLUDED.neutral_pct,
                    trend       = EXCLUDED.trend,
                    ingested_at = NOW();
            """
            execute_values(cur, sql, rows)
        conn.commit()

    print(
        f"[Postgres Loader] sentiment_trends: upserted {len(rows)} rows "
        f"for {ticker_symbol}"
    )
    return len(rows)


def _insert_dataframe(df, agent_name, ticker_symbol, data_name, source):
    if df.empty:
        return 0

    df = df.copy()
    date_col = _detect_date_col(df)

    if date_col:
        df = _normalise_date_col(df, date_col)
        df = df.drop_duplicates(subset=[date_col], keep='last').reset_index(drop=True)
    else:
        df = df.drop_duplicates().reset_index(drop=True)

    payload_cols = [c for c in df.columns if c != date_col]

    rows = []
    for _, row in df.iterrows():
        ts_val  = row[date_col] if date_col else datetime.utcnow().strftime('%Y-%m-%d')
        payload = {c: (None if pd.isna(row[c]) else row[c]) for c in payload_cols}
        rows.append((agent_name, ticker_symbol, data_name, ts_val, json.dumps(payload), source))

    # Deduplicate on constraint key (col 0-3 + 5) to avoid CardinalityViolation
    # when multiple rows share the same (agent_name, ticker_symbol, data_name, ts_date, source)
    seen: dict = {}
    for r in rows:
        key = (r[0], r[1], r[2], r[3], r[5])  # agent, ticker, data_name, ts_val, source
        seen[key] = r  # last row wins
    rows = list(seen.values())

    if not rows:
        return 0

    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            if date_col:
                sql = """
                    INSERT INTO raw_timeseries (
                        agent_name, ticker_symbol, data_name, ts_date, payload, source
                    ) VALUES %s
                    ON CONFLICT (agent_name, ticker_symbol, data_name, ts_date, source)
                    DO UPDATE SET payload     = EXCLUDED.payload,
                                  ingested_at = NOW();
                """
            else:
                sql = """
                    INSERT INTO raw_fundamentals (
                        agent_name, ticker_symbol, data_name, as_of_date, payload, source
                    ) VALUES %s
                    ON CONFLICT (agent_name, ticker_symbol, data_name, as_of_date, source)
                    DO UPDATE SET payload     = EXCLUDED.payload,
                                  ingested_at = NOW();
                """
            execute_values(cur, sql, rows)
        conn.commit()

    return len(rows)


def load_postgres_for_agent_ticker(agent_name: str, ticker_symbol: str) -> int:
    agent_dir     = BASE_ETL_DIR / agent_name / ticker_symbol
    metadata_path = agent_dir / "metadata.json"

    if not metadata_path.exists():
        print(f"[Postgres Loader] No metadata.json for {agent_name}/{ticker_symbol}")
        return 0

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    total_rows = 0
    ensure_tables()

    for data_name, info in metadata.items():
        if info.get("storage_destination") != "postgresql":
            continue

        csv_path = agent_dir / f"{data_name}.csv"
        if not csv_path.exists():
            print(f"[Postgres Loader] Missing CSV for {data_name} at {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[Postgres Loader] Failed to read {csv_path}: {e}")
            continue

        source = info.get("source", "unknown")

        # FIX 3: Route sentiment_trends to its dedicated table with field mapping
        # BEFORE: fell through to _insert_dataframe() → raw_fundamentals as JSONB
        #         tools.py SELECT FROM sentiment_trends → relation does not exist
        # AFTER:  dedicated _insert_sentiment() upserts into sentiment_trends table
        #         with correct bullish_pct/bearish_pct/neutral_pct column names
        if data_name == "sentiment_trends":
            rows_inserted = _insert_sentiment(ticker_symbol, df)

        elif data_name in GLOBAL_ONCE_PER_DAY:
            rows_inserted = _insert_global(df, data_name, source)

        else:
            rows_inserted = _insert_dataframe(
                df=df,
                agent_name=agent_name,
                ticker_symbol=ticker_symbol,
                data_name=data_name,
                source=source,
            )

        total_rows += rows_inserted
        print(
            f"[Postgres Loader] {agent_name}/{ticker_symbol}/{data_name}: "
            f"{rows_inserted} rows"
        )

    # Load macro indicators once — only when processing the anchor ticker
    # (EODHD scraper stores them under financial_modeling/_MACRO/, not per-equity ticker)
    if agent_name == "financial_modeling" and ticker_symbol == _MACRO_ANCHOR_TICKER:
        macro_rows = _insert_macro_indicators(agent_name)
        total_rows += macro_rows

    return total_rows


def _insert_macro_indicators(agent_name: str) -> int:
    """Load macro indicator CSVs from the _MACRO pseudo-ticker folder into global_macro_indicators.

    Called once per DAG run (guarded by _MACRO_ANCHOR_TICKER) so that the three
    economic_indicators_* data_names written by the EODHD scraper under
    financial_modeling/_MACRO/ land in a queryable table.

    The 'indicator' column is derived from the data_name suffix
    (e.g. 'economic_indicators_gdp' → 'gdp_growth_rate').
    """
    macro_dir = BASE_ETL_DIR / agent_name / _MACRO_TICKER
    meta_path = macro_dir / "metadata.json"

    if not meta_path.exists():
        print(f"[Postgres Loader] No _MACRO metadata.json for agent '{agent_name}' — skipping macro load")
        return 0

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    total_rows = 0

    for data_name, info in metadata.items():
        if data_name not in _MACRO_DATA_NAMES:
            continue
        if info.get("storage_destination") != "postgresql":
            continue

        csv_path = macro_dir / f"{data_name}.csv"
        if not csv_path.exists():
            print(f"[Postgres Loader] Missing macro CSV: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[Postgres Loader] Failed to read macro CSV {csv_path}: {e}")
            continue

        if df.empty:
            continue

        source = info.get("source", "eodhd")
        # Derive a short indicator name from the data_name
        # e.g. economic_indicators_gdp → gdp_growth_rate (matches EODHD indicator param)
        _INDICATOR_NAMES = {
            "economic_indicators_gdp":          "gdp_growth_rate",
            "economic_indicators_cpi":          "inflation_cpi",
            "economic_indicators_unemployment": "unemployment_rate",
        }
        indicator = _INDICATOR_NAMES.get(data_name, data_name)

        df = df.copy()
        date_col = _detect_date_col(df)
        if date_col:
            df = _normalise_date_col(df, date_col)

        payload_cols = [c for c in df.columns if c != date_col]

        rows_dict = {}
        for _, row in df.iterrows():
            ts_val  = row[date_col] if date_col else None
            payload = {c: (None if pd.isna(row[c]) else row[c]) for c in payload_cols}
            key = (indicator, str(ts_val), source)
            rows_dict[key] = (indicator, ts_val, json.dumps(payload), source)

        rows = list(rows_dict.values())
        if not rows:
            continue

        with get_pg_conn() as conn:
            with conn.cursor() as cur:
                sql = """
                    INSERT INTO global_macro_indicators (indicator, ts_date, payload, source)
                    VALUES %s
                    ON CONFLICT (indicator, ts_date, source)
                    DO UPDATE SET payload     = EXCLUDED.payload,
                                  ingested_at = NOW();
                """
                execute_values(cur, sql, rows)
            conn.commit()

        print(f"[Postgres Loader] global_macro_indicators ({indicator}): inserted {len(rows)} rows")
        total_rows += len(rows)

    return total_rows


if __name__ == "__main__":
    print(load_postgres_for_agent_ticker("quantitative_fundamental", "AAPL"))
