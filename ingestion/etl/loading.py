from .config import get_db_connection
import pandas as pd

def load_prices_to_db(df: pd.DataFrame):
    """Saves price data to SQLite."""
    if df.empty:
        return
    
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Convert DataFrame to list of tuples (standard Python types)
        # using to_records() passes numpy types which can be stored as BLOBs in SQLite
        data = df.values.tolist()
        
        cursor.executemany('''
            INSERT OR IGNORE INTO stock_prices (symbol, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data)
        conn.commit()
        print(f"Loaded {len(df)} price records.")
    finally:
        conn.close()

def load_news_to_db(df: pd.DataFrame):
    """Saves news data to SQLite."""
    if df.empty:
        return
        
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Convert to list of python types to avoid numpy/adapter issues
        data = df.values.tolist()
        
        cursor.executemany('''
            INSERT OR IGNORE INTO financial_news (symbol, date, title, content, source, url, sentiment_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data)
        conn.commit()
        print(f"Loaded {len(df)} news articles.")
    finally:
        conn.close()
