import requests
import pandas as pd
from datetime import datetime, timedelta
from .config import EODHD_API_KEY

BASE_URL = "https://eodhd.com/api"

def fetch_stock_prices(symbol: str, days: int = 1):
    """Fetches OHLCV data for the last N days."""
    url = f"{BASE_URL}/eod/{symbol}"
    
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    params = {
        "api_token": EODHD_API_KEY,
        "fmt": "json",
        "from": start_date,
        "period": "d"
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return []

def fetch_company_news(symbol: str, days: int = 1):
    """Fetches financial news for a specific stock."""
    url = f"{BASE_URL}/news"
    
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    params = {
        "api_token": EODHD_API_KEY,
        "s": symbol,
        "from": start_date,
        "limit": 50,
        "offset": 0,
        "fmt": "json"
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return []
