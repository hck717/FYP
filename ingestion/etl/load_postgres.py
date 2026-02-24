import os
import json
from pathlib import Path
from datetime import datetime
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


def get_pg_conn():
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD,
    )


def ensure_tables():
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

        if data_name in GLOBAL_ONCE_PER_DAY:
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
