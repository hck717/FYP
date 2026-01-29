from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sqlite3
import pandas as pd
import re

DB_PATH = "/opt/airflow/data/finance.db"

def scrape_reddit_sentiment():
    """
    Placeholder: Use PRAW to fetch discussions from financial subreddits.
    Targets: r/investing, r/stocks, r/wallstreetbets.
    """
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    
    if not client_id or client_id == 'demo':
        print("Skipping Reddit scrape: No valid credentials provided.")
        return

    print("Connecting to Reddit API...")
    print("Scraping top daily posts from r/investing, r/wallstreetbets...")
    # Logic: praw.Reddit(...) -> subreddit.top('day') -> extract title/body/comments
    # Save to table 'social_posts'
    return "Social sentiment raw data ingested."


def etl_clean_social():
    """Clean and normalize social media posts after ingestion."""
    conn = sqlite3.connect(DB_PATH)

    # 1) Ensure clean table exists
    conn.execute("""
    CREATE TABLE IF NOT EXISTS social_posts_clean (
      post_id TEXT PRIMARY KEY,
      subreddit TEXT,
      title TEXT,
      body TEXT,
      url TEXT UNIQUE,
      score INTEGER,
      created_utc TEXT,
      tickers_mentioned TEXT
    )
    """)

    # 2) Read from raw table (assuming it exists from ingestion)
    try:
        # Check if raw table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='social_posts'")
        if not cursor.fetchone():
            print("No social_posts table found yet. Skipping cleaning.")
            conn.close()
            return

        df = pd.read_sql("SELECT * FROM social_posts", conn)

        if df.empty:
            print("No social posts to clean.")
            conn.close()
            return

        # 3) Clean / normalize
        df["subreddit"] = df["subreddit"].astype(str).str.lower().str.strip()
        df["title"] = df["title"].astype(str).str.strip()
        df["body"] = df["body"].astype(str).str.strip()
        df["url"] = df["url"].astype(str).str.strip()
        
        # Normalize created_utc to date string
        if "created_utc" in df.columns:
            df["created_utc"] = pd.to_datetime(df["created_utc"], unit='s', errors="coerce").dt.date.astype(str)
        
        # Drop empty posts
        df = df.dropna(subset=["post_id", "title"])
        df = df[df["title"].str.len() > 0]
        
        # Dedupe by post_id or url
        df = df.drop_duplicates(subset=["post_id"], keep="last")

        # Optional: Extract tickers via regex (e.g., $AAPL, TSLA)
        def extract_tickers(text):
            if pd.isna(text):
                return ""
            # Find patterns like $AAPL or standalone AAPL (1-5 uppercase letters)
            tickers = re.findall(r'\$([A-Z]{1,5})\b|\b([A-Z]{1,5})\b', str(text))
            # Flatten and dedupe
            tickers = list(set([t[0] if t[0] else t[1] for t in tickers]))
            return ",".join(tickers) if tickers else ""
        
        df["tickers_mentioned"] = df.apply(
            lambda row: extract_tickers(str(row.get("title", "")) + " " + str(row.get("body", ""))),
            axis=1
        )

        # 4) Load to clean table (replace all)
        conn.execute("DELETE FROM social_posts_clean")
        
        # Select only the columns that exist and match the clean table schema
        cols_to_save = [c for c in ["post_id", "subreddit", "title", "body", "url", "score", "created_utc", "tickers_mentioned"] if c in df.columns]
        df[cols_to_save].to_sql(
            "social_posts_clean", conn, if_exists="append", index=False
        )

        conn.commit()
        print(f"Cleaned social_posts -> social_posts_clean: {len(df)} rows")
    except Exception as e:
        print(f"Error cleaning social posts: {e}")
    finally:
        conn.close()


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=15),
}

with DAG(
    'social_media_ingestion',
    default_args=default_args,
    description='Ingests discussions from Reddit/Twitter for retail sentiment',
    schedule_interval='0 23 * * 1-5',  # Daily at 11 PM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['social', 'daily']
) as dag:

    scrape_task = PythonOperator(
        task_id='scrape_reddit',
        python_callable=scrape_reddit_sentiment,
    )

    clean_task = PythonOperator(
        task_id='etl_clean_social',
        python_callable=etl_clean_social,
    )

    scrape_task >> clean_task
