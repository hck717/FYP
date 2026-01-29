import os
import sqlite3
from pathlib import Path

# Paths inside Docker container
DB_PATH = Path("/opt/airflow/data/finance.db")
EODHD_API_KEY = os.getenv("EODHD_API_KEY", "demo") # Use 'demo' for testing if no key

def get_db_connection():
    """Returns a connection to the SQLite database, creating tables if needed."""
    # Ensure directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create Tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            symbol TEXT,
            date DATE,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (symbol, date)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS financial_news (
            symbol TEXT,
            date DATETIME,
            title TEXT,
            content TEXT,
            source TEXT,
            url TEXT,
            sentiment_score REAL,
            UNIQUE(url)
        )
    ''')
    
    conn.commit()
    return conn
