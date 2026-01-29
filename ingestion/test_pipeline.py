import sys
import os

# Ensure we can import modules
# Since this script will be run from /opt/airflow/, we ensure that path is in sys.path
sys.path.append('/opt/airflow')

from etl.extraction import fetch_stock_prices, fetch_company_news
from etl.transformation import transform_prices, transform_news
from etl.loading import load_prices_to_db, load_news_to_db, get_db_connection
from etl.entity_resolution import normalize_symbol

def test_pipeline():
    print("🧪 Starting End-to-End Test for 1 Stock (AAPL)...")
    
    # 1. Define Test List (Single Ticker)
    test_tickers = ["AAPL"]
    
    success_count = 0
    
    for ticker in test_tickers:
        try:
            symbol = normalize_symbol(ticker)
            print(f"   Processing {symbol}...", end=" ")
            
            # Extract
            prices = fetch_stock_prices(symbol, days=5)
            news = fetch_company_news(symbol, days=5)
            
            # Transform
            df_prices = transform_prices(prices, symbol)
            df_news = transform_news(news, symbol)
            
            # Load
            load_prices_to_db(df_prices)
            load_news_to_db(df_news)
            
            success_count += 1
            print("✅ OK")
            
        except Exception as e:
            print(f"❌ FAILED: {e}")

    print(f"\n🎉 Test Complete. Successfully processed {success_count}/{len(test_tickers)} stocks.")
    
    # Verify Data in DB
    conn = get_db_connection()
    cursor = conn.cursor()
    price_count = cursor.execute("SELECT COUNT(*) FROM stock_prices").fetchone()[0]
    news_count = cursor.execute("SELECT COUNT(*) FROM financial_news").fetchone()[0]
    conn.close()
    
    print(f"\n📊 Database Stats:")
    print(f"   - Total Price Records: {price_count}")
    print(f"   - Total News Articles: {news_count}")

if __name__ == "__main__":
    test_pipeline()
