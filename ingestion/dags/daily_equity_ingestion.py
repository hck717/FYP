from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add path for ETL modules
sys.path.append('/opt/airflow')

# Import existing ETL functions (ensure these exist or use placeholders if extending)
# From Phase 1 we have extraction/transformation/loading for prices/news
from etl.extraction import fetch_stock_prices, fetch_company_news
from etl.transformation import transform_prices, transform_news
from etl.loading import load_prices_to_db, load_news_to_db
from etl.entity_resolution import normalize_symbol

# Default tickers (S&P 50 subset for demo)
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK.B", "TSM", "UNH",
    "JNJ", "JPM", "XOM", "V", "PG", "MA", "HD", "CVX", "ABBV", "PEP"
]

def ingest_ticker_data(ticker):
    """Orchestrates E-T-L for a single ticker."""
    symbol = normalize_symbol(ticker)
    print(f"Processing {symbol}...")
    
    # 1. Prices (OHLCV)
    try:
        raw_prices = fetch_stock_prices(symbol)
        df_prices = transform_prices(raw_prices, symbol)
        load_prices_to_db(df_prices)
        print(f"Prices loaded for {symbol}")
    except Exception as e:
        print(f"Error fetching prices for {symbol}: {e}")

    # 2. News (Qualitative)
    try:
        raw_news = fetch_company_news(symbol)
        df_news = transform_news(raw_news, symbol)
        load_news_to_db(df_news)
        print(f"News loaded for {symbol}")
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")

    # 3. Fundamentals (Placeholder for future expansion)
    # fetch_fundamentals(symbol) -> transform -> load
    pass

def run_equity_batch():
    """Wrapper to loop through tickers."""
    print(f"Starting batch ingestion for {len(TICKERS)} tickers.")
    for ticker in TICKERS:
        ingest_ticker_data(ticker)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'daily_equity_ingestion',
    default_args=default_args,
    description='Ingests Prices, News, and Fundamentals for Target Stocks',
    schedule_interval='0 21 * * 1-5',  # Daily at 9 PM (after market close)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['core', 'finance', 'daily']
) as dag:

    ingest_task = PythonOperator(
        task_id='ingest_equity_data',
        python_callable=run_equity_batch,
    )

    ingest_task
