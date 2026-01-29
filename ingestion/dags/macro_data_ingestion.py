from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sqlite3
import pandas as pd

DB_PATH = "/opt/airflow/data/finance.db"

def fetch_macro_indicators():
    """
    Placeholder: Fetch macro data from FRED / EODHD.
    Indicators: GDP, CPI, Fed Funds Rate, 10Y Treasury, Unemployment.
    """
    print("Fetching GDP Growth Rate...")
    print("Fetching CPI / Inflation Data...")
    print("Fetching Fed Funds Rate & Treasury Yields...")
    print("Fetching Unemployment Rate...")
    # Logic: requests.get(FRED_API_URL...) -> clean -> save to DB table 'macro_economic_data'
    return "Macro data ingested successfully."


def etl_clean_macro():
    """Clean and normalize macro economic data after ingestion."""
    conn = sqlite3.connect(DB_PATH)

    # 1) Ensure clean table exists
    conn.execute("""
    CREATE TABLE IF NOT EXISTS macro_economic_data_clean (
      indicator TEXT,
      date TEXT,
      value REAL,
      PRIMARY KEY (indicator, date)
    )
    """)

    # 2) Read from raw table (assuming it exists from ingestion)
    try:
        # Check if raw table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='macro_economic_data'")
        if not cursor.fetchone():
            print("No macro_economic_data table found yet. Skipping cleaning.")
            conn.close()
            return

        df = pd.read_sql("SELECT * FROM macro_economic_data", conn)

        if df.empty:
            print("No macro data to clean.")
            conn.close()
            return

        # 3) Clean / normalize
        df["indicator"] = df["indicator"].astype(str).str.upper().str.strip()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Drop nulls and dedupe
        df = df.dropna(subset=["indicator", "date", "value"])
        df = df.drop_duplicates(subset=["indicator", "date"], keep="last")

        # 4) Load to clean table (replace all)
        conn.execute("DELETE FROM macro_economic_data_clean")
        df[["indicator", "date", "value"]].to_sql(
            "macro_economic_data_clean", conn, if_exists="append", index=False
        )

        conn.commit()
        print(f"Cleaned macro_economic_data -> macro_economic_data_clean: {len(df)} rows")
    except Exception as e:
        print(f"Error cleaning macro data: {e}")
    finally:
        conn.close()


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=30),
}

with DAG(
    'macro_data_ingestion',
    default_args=default_args,
    description='Fetches GDP, CPI, Rates, and other macro indicators',
    schedule_interval='0 8 * * 5',  # Weekly on Fridays at 8 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['macro', 'weekly']
) as dag:

    fetch_task = PythonOperator(
        task_id='fetch_macro_data',
        python_callable=fetch_macro_indicators,
    )

    clean_task = PythonOperator(
        task_id='etl_clean_macro',
        python_callable=etl_clean_macro,
    )

    fetch_task >> clean_task
