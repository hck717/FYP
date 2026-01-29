from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

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

    scrape_task
