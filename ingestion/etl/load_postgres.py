import os
import json
from pathlib import Path
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

BASE_ETL_DIR = Path(os.getenv("BASE_ETL_DIR", "/opt/airflow/etl/agent_data"))

PG_HOST = os.getenv("POSTGRES_HOST", "fyp-postgres")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB = os.getenv("POSTGRES_DB", "fyp")
PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")


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
        id SERIAL PRIMARY KEY,
        agent_name TEXT NOT NULL,
        ticker_symbol TEXT NOT NULL,
        data_name TEXT NOT NULL,
        ts_date TIMESTAMP,
        payload JSONB NOT NULL,
        source TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE (agent_name, ticker_symbol, data_name, ts_date, source)
    );

    CREATE TABLE IF NOT EXISTS raw_fundamentals (
        id SERIAL PRIMARY KEY,
        agent_name TEXT NOT NULL,
        ticker_symbol TEXT NOT NULL,
        data_name TEXT NOT NULL,
        as_of_date TIMESTAMP,
        payload JSONB NOT NULL,
        source TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE (agent_name, ticker_symbol, data_name, as_of_date, source)
    );
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


def _insert_dataframe(df, agent_name, ticker_symbol, data_name, source):
    if df.empty:
        return 0

    df = df.copy()
    date_col = None
    for candidate in ["date", "timestamp", "datetime", "reportedDate"]:
        if candidate in df.columns:
            date_col = candidate
            break

    payload_cols = [c for c in df.columns if c != date_col]

    rows = []
    for _, row in df.iterrows():
        ts_val = row[date_col] if date_col else None
        payload = {c: row[c] for c in payload_cols}
        rows.append((agent_name, ticker_symbol, data_name, ts_val, json.dumps(payload), source))

    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            if date_col:
                sql = """
                    INSERT INTO raw_timeseries (
                        agent_name, ticker_symbol, data_name, ts_date, payload, source
                    ) VALUES %s
                    ON CONFLICT (agent_name, ticker_symbol, data_name, ts_date, source)
                    DO UPDATE SET payload = EXCLUDED.payload;
                """
            else:
                sql = """
                    INSERT INTO raw_fundamentals (
                        agent_name, ticker_symbol, data_name, as_of_date, payload, source
                    ) VALUES %s
                    ON CONFLICT (agent_name, ticker_symbol, data_name, as_of_date, source)
                    DO UPDATE SET payload = EXCLUDED.payload;
                """
            execute_values(cur, sql, rows)
        conn.commit()
    return len(rows)


def load_postgres_for_agent_ticker(agent_name: str, ticker_symbol: str) -> int:
    agent_dir = BASE_ETL_DIR / agent_name / ticker_symbol
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

        rows_inserted = _insert_dataframe(
            df=df,
            agent_name=agent_name,
            ticker_symbol=ticker_symbol,
            data_name=data_name,
            source=info.get("source", "unknown"),
        )
        total_rows += rows_inserted
        print(f"[Postgres Loader] {agent_name}/{ticker_symbol}/{data_name}: {rows_inserted} rows")

    return total_rows


if __name__ == "__main__":
    print(load_postgres_for_agent_ticker("quantitative_fundamental", "AAPL"))
