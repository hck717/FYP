# ingestion/etl/load_postgres.py
import os
import json
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

BASE_ETL_DIR = Path(os.getenv("BASE_ETL_DIR", "/opt/airflow/etl/agent_data"))

PG_HOST     = os.getenv("POSTGRES_HOST",     "fyp-postgres")
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

    CREATE TABLE IF NOT EXISTS sentiment_trends (
        id           SERIAL PRIMARY KEY,
        ticker       VARCHAR(10)  NOT NULL,
        bullish_pct  NUMERIC,
        bearish_pct  NUMERIC,
        neutral_pct  NUMERIC,
        trend        VARCHAR(20)  DEFAULT 'unknown',
        date         DATE         NOT NULL,
        ingested_at  TIMESTAMP    DEFAULT NOW(),
        UNIQUE (ticker, date)
    );
    CREATE INDEX IF NOT EXISTS idx_sentiment_trends_ticker_date
        ON sentiment_trends (ticker, date DESC);
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


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
      SELECT bullish_pct, bearish_pct, neutral_pct, date
      FROM sentiment_trends WHERE ticker = %s ORDER BY date DESC LIMIT 2
    """
    if df.empty:
        return 0

    rows = []
    today = date.today().isoformat()

    # Handle both flat and nested column layouts
    # If EODHD returned a nested dict saved as single-row CSV:
    # columns may be: polarity, neg, neu, pos  (flat after _flatten_eodhd_fundamentals)
    # OR columns may be: date, sentiment_neg, sentiment_neu, sentiment_pos etc.
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
                    (ticker, bullish_pct, bearish_pct, neutral_pct, trend, date)
                VALUES %s
                ON CONFLICT (ticker, date)
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

    return total_rows


if __name__ == "__main__":
    print(load_postgres_for_agent_ticker("quantitative_fundamental", "AAPL"))
