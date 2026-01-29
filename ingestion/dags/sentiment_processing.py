from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
import json

DB_PATH = "/opt/airflow/data/finance.db"

def process_nlp_enrichment():
    """
    Placeholder: NLP pipeline to process raw text into structured data.
    """
    # 1. Fetch unprocessed records from 'financial_news' and 'social_posts'
    print("Fetching new text records from DB...")
    
    # 2. Run Sentiment Analysis (FinTwitBERT or similar)
    print("Running local LLM/BERT inference for sentiment scores...")
    
    # 3. Extract Named Entities (NER)
    print("Extracting Companies, CEOs, and Products...")
    
    # 4. Update DB records with 'sentiment_score' and 'entities_json'
    print("Enrichment complete. DB updated.")


def etl_clean_sentiment():
    """Clean and normalize sentiment data after processing."""
    conn = sqlite3.connect(DB_PATH)
    
    # 1) Ensure clean table exists
    conn.execute("""
    CREATE TABLE IF NOT EXISTS sentiment_data_clean (
      source_type TEXT,
      source_id TEXT,
      sentiment_score REAL,
      entities_json TEXT,
      processed_date TEXT,
      PRIMARY KEY (source_type, source_id)
    )
    """)
    
    # 2) Process financial_news sentiment (if sentiment column exists)
    try:
        cursor = conn.cursor()
        
        # Check if financial_news_clean exists and has sentiment data
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='financial_news_clean'")
        if cursor.fetchone():
            df_news = pd.read_sql(
                "SELECT url, sentiment_score FROM financial_news_clean WHERE sentiment_score IS NOT NULL",
                conn
            )
            
            if not df_news.empty:
                df_news["source_type"] = "news"
                df_news["source_id"] = df_news["url"]
                df_news["processed_date"] = datetime.now().date().isoformat()
                df_news["entities_json"] = "{}"  # Placeholder for NER results
                
                # Clamp sentiment to [-1, 1]
                df_news["sentiment_score"] = pd.to_numeric(df_news["sentiment_score"], errors="coerce").fillna(0)
                df_news["sentiment_score"] = df_news["sentiment_score"].clip(-1, 1)
                
                # Insert/update sentiment data
                for _, row in df_news.iterrows():
                    conn.execute("""
                        INSERT OR REPLACE INTO sentiment_data_clean 
                        (source_type, source_id, sentiment_score, entities_json, processed_date)
                        VALUES (?, ?, ?, ?, ?)
                    """, (row["source_type"], row["source_id"], row["sentiment_score"], 
                          row["entities_json"], row["processed_date"]))
                
                print(f"Processed sentiment for {len(df_news)} news articles")
        
        # 3) Process social_posts sentiment (if exists)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='social_posts_clean'")
        if cursor.fetchone():
            # Check if sentiment column exists in social_posts
            cursor.execute("PRAGMA table_info(social_posts_clean)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if "sentiment_score" in columns:
                df_social = pd.read_sql(
                    "SELECT post_id, sentiment_score, tickers_mentioned FROM social_posts_clean WHERE sentiment_score IS NOT NULL",
                    conn
                )
                
                if not df_social.empty:
                    df_social["source_type"] = "social"
                    df_social["source_id"] = df_social["post_id"]
                    df_social["processed_date"] = datetime.now().date().isoformat()
                    
                    # Create entities JSON from tickers
                    df_social["entities_json"] = df_social["tickers_mentioned"].apply(
                        lambda x: json.dumps({"tickers": x.split(",") if x else []}) if pd.notna(x) else "{}"
                    )
                    
                    # Clamp sentiment to [-1, 1]
                    df_social["sentiment_score"] = pd.to_numeric(df_social["sentiment_score"], errors="coerce").fillna(0)
                    df_social["sentiment_score"] = df_social["sentiment_score"].clip(-1, 1)
                    
                    # Insert/update sentiment data
                    for _, row in df_social.iterrows():
                        conn.execute("""
                            INSERT OR REPLACE INTO sentiment_data_clean 
                            (source_type, source_id, sentiment_score, entities_json, processed_date)
                            VALUES (?, ?, ?, ?, ?)
                        """, (row["source_type"], row["source_id"], row["sentiment_score"], 
                              row["entities_json"], row["processed_date"]))
                    
                    print(f"Processed sentiment for {len(df_social)} social posts")
        
        # 4) Validate JSON fields
        cursor.execute("SELECT source_type, source_id, entities_json FROM sentiment_data_clean")
        for row in cursor.fetchall():
            try:
                # Validate JSON is parseable
                json.loads(row[2])
            except json.JSONDecodeError:
                # Fix invalid JSON
                conn.execute(
                    "UPDATE sentiment_data_clean SET entities_json = '{}' WHERE source_type = ? AND source_id = ?",
                    (row[0], row[1])
                )
        
        conn.commit()
        print("Sentiment data cleaning complete.")
        
    except Exception as e:
        print(f"Error cleaning sentiment data: {e}")
    finally:
        conn.close()


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=15),
}

with DAG(
    'sentiment_processing',
    default_args=default_args,
    description='Runs NLP models on ingested text to generate sentiment scores',
    schedule_interval='30 23 * * 1-5',  # Daily at 11:30 PM (after ingestion DAGs)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['nlp', 'enrichment', 'gpu']
) as dag:

    nlp_task = PythonOperator(
        task_id='run_nlp_enrichment',
        python_callable=process_nlp_enrichment,
    )

    clean_task = PythonOperator(
        task_id='etl_clean_sentiment',
        python_callable=etl_clean_sentiment,
    )

    nlp_task >> clean_task
