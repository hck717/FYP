from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add the /opt/airflow path so we can import 'etl' modules
sys.path.append('/opt/airflow')

from etl.extraction import fetch_stock_prices, fetch_company_news
from etl.transformation import transform_prices, transform_news
from etl.loading import load_prices_to_db, load_news_to_db
from etl.entity_resolution import normalize_symbol

# Default S&P 50 list (can be expanded to 500)
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK.B", "TSM", "UNH",
    "JNJ", "JPM", "XOM", "V", "PG", "MA", "HD", "CVX", "ABBV", "PEP",
    "KO", "LLY", "BAC", "AVGO", "COST", "PFE", "TMO", "CSCO", "WMT", "MCD",
    "DIS", "ADBE", "ACN", "NFLX", "LIN", "NKE", "ABT", "DHR", "NEE", "TXN",
    "VZ", "CRM", "PM", "BMY", "UPS", "RTX", "HON", "MS", "INTC", "AMD"
]

def run_pipeline_for_ticker(ticker):
    """Runs E-T-L for a single ticker."""
    symbol = normalize_symbol(ticker)
    print(f"Processing {symbol}...")
    
    # 1. Prices
    raw_prices = fetch_stock_prices(symbol)
    df_prices = transform_prices(raw_prices, symbol)
    load_prices_to_db(df_prices)
    
    # 2. News
    raw_news = fetch_company_news(symbol)
    df_news = transform_news(raw_news, symbol)
    load_news_to_db(df_news)

def run_batch_ingestion():
    """Wrapper to loop through all tickers."""
    for ticker in TICKERS:
        try:
            run_pipeline_for_ticker(ticker)
        except Exception as e:
            print(f"Failed processing {ticker}: {e}")

# Define DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'daily_financial_ingestion',
    default_args=default_args,
    description='Ingests Prices and News for S&P 50 Stocks',
    schedule_interval='@daily',
    catchup=False,
) as dag:

    ingest_task = PythonOperator(
        task_id='ingest_sp50_data',
        python_callable=run_batch_ingestion,
    )

    ingest_task
