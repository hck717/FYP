import requests
from .config import EODHD_API_KEY

def resolve_ticker(company_name: str) -> str:
    """
    Resolves a company name to a ticker symbol using EODHD Search API.
    Example: "Apple Inc." -> "AAPL.US"
    """
    url = f"https://eodhd.com/api/search/{company_name}"
    params = {
        "api_token": EODHD_API_KEY,
        "fmt": "json",
        "limit": 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data and len(data) > 0:
            return f"{data[0]['Code']}.{data[0]['Exchange']}" # e.g., AAPL.US
        return None
    except Exception as e:
        print(f"Error resolving {company_name}: {e}")
        return None

def normalize_symbol(symbol: str) -> str:
    """Ensures consistent format (e.g., AAPL -> AAPL.US)"""
    if "." not in symbol:
        return f"{symbol}.US"
    return symbol
