from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sqlite3
import pandas as pd

DB_PATH = "/opt/airflow/data/finance.db"

def calculate_market_risk():
    """
    Placeholder: Fetch VIX and calculate Beta/Sector Rotation.
    Source: yfinance or EODHD.
    """
    print("Fetching VIX (Fear Index)...")
    
    # 1. Fetch Sector ETF data (XLK, XLU, etc.) to track rotation
    print("Analyzing Sector Performance...")
    
    # 2. Calculate Rolling Beta for watched stocks
    # Load stock prices from DB -> Compute covariance with SPY -> Save Beta to DB
    print("Calculating rolling 60-day Beta for all tracked stocks...")
    
    return "Risk metrics updated."


def etl_clean_risk():
    """Clean and normalize risk metrics data after ingestion."""
    conn = sqlite3.connect(DB_PATH)

    # 1) Ensure clean table exists
    conn.execute("""
    CREATE TABLE IF NOT EXISTS risk_metrics_clean (
      metric TEXT,
      symbol TEXT,
      date TEXT,
      value REAL,
      PRIMARY KEY (metric, symbol, date)
    )
    """)

    # 2) Read from raw table (assuming it exists from ingestion)
    try:
        # Check if raw table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='risk_metrics'")
        if not cursor.fetchone():
            print("No risk_metrics table found yet. Skipping cleaning.")
            conn.close()
            return

        df = pd.read_sql("SELECT * FROM risk_metrics", conn)

        if df.empty:
            print("No risk metrics data to clean.")
            conn.close()
            return

        # 3) Clean / normalize
        df["metric"] = df["metric"].astype(str).str.upper().str.strip()
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Drop nulls and dedupe
        df = df.dropna(subset=["metric", "symbol", "date", "value"])
        df = df.drop_duplicates(subset=["metric", "symbol", "date"], keep="last")

        # Enforce naming conventions (e.g., VIX, BETA_60D)
        metric_map = {
            'vix': 'VIX',
            'beta': 'BETA_60D',
            'beta_60d': 'BETA_60D',
            'volatility': 'VOLATILITY'
        }
        df["metric"] = df["metric"].str.lower().map(metric_map).fillna(df["metric"])

        # 4) Load to clean table (replace all)
        conn.execute("DELETE FROM risk_metrics_clean")
        df[["metric", "symbol", "date", "value"]].to_sql(
            "risk_metrics_clean", conn, if_exists="append", index=False
        )

        conn.commit()
        print(f"Cleaned risk_metrics -> risk_metrics_clean: {len(df)} rows")
    except Exception as e:
        print(f"Error cleaning risk metrics: {e}")
    finally:
        conn.close()


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

with DAG(
    'risk_metrics_ingestion',
    default_args=default_args,
    description='Calculates VIX, Beta, and Sector Rotation metrics',
    schedule_interval='0 22 * * 1-5',  # Daily at 10 PM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['risk', 'daily']
) as dag:

    risk_task = PythonOperator(
        task_id='calc_risk_metrics',
        python_callable=calculate_market_risk,
    )

    clean_task = PythonOperator(
        task_id='etl_clean_risk',
        python_callable=etl_clean_risk,
    )

    risk_task >> clean_task
