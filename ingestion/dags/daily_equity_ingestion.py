from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import sqlite3
import pandas as pd

# Add path for ETL modules
sys.path.append('/opt/airflow')

# Import existing ETL functions (ensure these exist or use placeholders if extending)
# From Phase 1 we have extraction/transformation/loading for prices/news
from etl.extraction import fetch_stock_prices, fetch_company_news
from etl.transformation import transform_prices, transform_news
from etl.loading import load_prices_to_db, load_news_to_db
from etl.entity_resolution import normalize_symbol

DB_PATH = "/opt/airflow/data/finance.db"

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


def etl_clean_equity():
    """Clean and normalize equity data after ingestion."""
    conn = sqlite3.connect(DB_PATH)

    # 1) Ensure clean tables exist
    conn.execute("""
    CREATE TABLE IF NOT EXISTS stock_prices_clean (
      symbol TEXT,
      date   TEXT,
      open   REAL,
      high   REAL,
      low    REAL,
      close  REAL,
      volume INTEGER,
      PRIMARY KEY (symbol, date)
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS financial_news_clean (
      symbol TEXT,
      date TEXT,
      title TEXT,
      content TEXT,
      source TEXT,
      url TEXT UNIQUE,
      sentiment_score REAL
    )
    """)

    # 2) Clean stock_prices
    try:
        df_prices = pd.read_sql("SELECT * FROM stock_prices", conn)

        if not df_prices.empty:
            # Normalize data types
            df_prices["symbol"] = df_prices["symbol"].astype(str).str.upper().str.strip()
            df_prices["date"] = pd.to_datetime(df_prices["date"], errors="coerce").dt.date.astype(str)
            
            for c in ["open","high","low","close"]:
                df_prices[c] = pd.to_numeric(df_prices[c], errors="coerce")

            # Handle volume (sometimes stored as bytes in SQLite)
            df_prices["volume"] = df_prices["volume"].apply(
                lambda x: int.from_bytes(x, "little") if isinstance(x, (bytes, bytearray)) else x
            )
            df_prices["volume"] = pd.to_numeric(df_prices["volume"], errors="coerce").fillna(0).astype(int)

            # Drop bad rows + dedupe
            df_prices = df_prices.dropna(subset=["symbol","date","close"])
            df_prices = df_prices.drop_duplicates(subset=["symbol","date"], keep="last")

            # Load to clean table (replace all)
            conn.execute("DELETE FROM stock_prices_clean")
            df_prices[["symbol","date","open","high","low","close","volume"]].to_sql(
                "stock_prices_clean", conn, if_exists="append", index=False
            )

            print(f"Cleaned stock_prices -> stock_prices_clean: {len(df_prices)} rows")
    except Exception as e:
        print(f"Error cleaning stock_prices: {e}")

    # 3) Clean financial_news
    try:
        df_news = pd.read_sql("SELECT * FROM financial_news", conn)

        if not df_news.empty:
            # Normalize strings
            df_news["symbol"] = df_news["symbol"].astype(str).str.upper().str.strip()
            df_news["title"] = df_news["title"].astype(str).str.strip()
            df_news["content"] = df_news["content"].astype(str).str.strip()
            df_news["source"] = df_news["source"].astype(str).str.strip()
            df_news["url"] = df_news["url"].astype(str).str.strip()
            
            # Normalize date
            df_news["date"] = pd.to_datetime(df_news["date"], errors="coerce").dt.date.astype(str)
            
            # Normalize sentiment score
            df_news["sentiment_score"] = pd.to_numeric(df_news["sentiment_score"], errors="coerce").fillna(0)
            
            # Drop empty titles/content
            df_news = df_news.dropna(subset=["title", "url"])
            df_news = df_news[df_news["title"].str.len() > 0]
            
            # Dedupe by URL
            df_news = df_news.drop_duplicates(subset=["url"], keep="last")

            # Load to clean table (replace all)
            conn.execute("DELETE FROM financial_news_clean")
            df_news[["symbol","date","title","content","source","url","sentiment_score"]].to_sql(
                "financial_news_clean", conn, if_exists="append", index=False
            )

            print(f"Cleaned financial_news -> financial_news_clean: {len(df_news)} rows")
    except Exception as e:
        print(f"Error cleaning financial_news: {e}")

    conn.commit()
    conn.close()


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

    clean_task = PythonOperator(
        task_id='etl_clean_equity',
        python_callable=etl_clean_equity,
    )

    ingest_task >> clean_task
