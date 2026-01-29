from .config import get_db_connection
import pandas as pd

def load_prices_to_db(df: pd.DataFrame):
    """Saves price data to SQLite."""
    if df.empty:
        return
    
    conn = get_db_connection()
    try:
        # Append data, ignore duplicates handled by DB constraint if possible
        # Or simpler: Iterative insert with 'OR IGNORE'
        cursor = conn.cursor()
        data = df.to_records(index=False)
        
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
        data = df.to_records(index=False)
        
        cursor.executemany('''
            INSERT OR IGNORE INTO financial_news (symbol, date, title, content, source, url, sentiment_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data)
        conn.commit()
        print(f"Loaded {len(df)} news articles.")
    finally:
        conn.close()
