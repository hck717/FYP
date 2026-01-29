from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sqlite3
import pandas as pd

DB_PATH = "/opt/airflow/data/finance.db"

def run_system_improvement():
    """
    Placeholder: Logic for self-improving RAG.
    """
    # 1. Analyze query logs
    print("Analyzing query logs for successful citations...")
    
    # 2. Identify high-value chunks
    print("Identifying most cited document chunks...")
    
    # 3. Fine-tune embeddings or boost weights
    print("Adjusting Qdrant retrieval weights based on user feedback...")
    
    return "System retraining/optimization complete."


def etl_cleanup_training_logs():
    """Compact training logs and remove stale data after retraining."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # 1) Create training metadata table if it doesn't exist
        conn.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            run_id TEXT PRIMARY KEY,
            run_date TEXT,
            metrics_json TEXT,
            model_version TEXT
        )
        """)
        
        # 2) Create query logs table if it doesn't exist (for future use)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            query_id TEXT PRIMARY KEY,
            query_text TEXT,
            created_at TEXT,
            cited_chunks TEXT,
            user_rating REAL
        )
        """)
        
        # 3) Remove old query logs (keep last 90 days)
        cursor.execute("""
            DELETE FROM query_logs 
            WHERE created_at < date('now', '-90 days')
        """)
        deleted_queries = cursor.rowcount
        print(f"Deleted {deleted_queries} query logs older than 90 days.")
        
        # 4) Keep only last 12 weeks of training runs (maintain reasonable history)
        cursor.execute("""
            DELETE FROM training_runs 
            WHERE run_date < date('now', '-84 days')
        """)
        deleted_runs = cursor.rowcount
        print(f"Deleted {deleted_runs} training runs older than 12 weeks.")
        
        # 5) Log current training run
        current_run_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        current_date = datetime.now().date().isoformat()
        
        cursor.execute("""
            INSERT OR REPLACE INTO training_runs (run_id, run_date, metrics_json, model_version)
            VALUES (?, ?, ?, ?)
        """, (current_run_id, current_date, '{}', 'v1.0'))
        
        print(f"Logged training run: {current_run_id}")
        
        # 6) Optimize database (reclaim space from deletions)
        print("Running VACUUM to optimize database...")
        conn.execute("VACUUM")
        
        # 7) Generate cleanup summary
        cursor.execute("SELECT COUNT(*) FROM query_logs")
        total_queries = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM training_runs")
        total_runs = cursor.fetchone()[0]
        
        print(f"\n=== Cleanup Summary ===")
        print(f"Active query logs: {total_queries}")
        print(f"Training run history: {total_runs}")
        print(f"Database optimized successfully.")
        
        conn.commit()
        
    except Exception as e:
        print(f"Error during training log cleanup: {e}")
        conn.rollback()
    finally:
        conn.close()


default_args = {
    'owner': 'airflow',
    'retries': 0,
    'retry_delay': timedelta(minutes=60),
}

with DAG(
    'weekly_model_retraining',
    default_args=default_args,
    description='Analyzes system performance and retrains/optimizes retrieval',
    schedule_interval='0 2 * * 0',  # Weekly on Sunday at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['maintenance', 'ml-ops', 'weekly']
) as dag:

    train_task = PythonOperator(
        task_id='optimize_rag_system',
        python_callable=run_system_improvement,
    )

    cleanup_task = PythonOperator(
        task_id='etl_cleanup_logs',
        python_callable=etl_cleanup_training_logs,
    )

    train_task >> cleanup_task
