import pandas as pd
import re

def clean_html(raw_html):
    """Removes HTML tags from news content."""
    if not raw_html:
        return ""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext.strip()

def transform_prices(raw_data, symbol):
    """Normalizes price data into a consistent DataFrame."""
    if not raw_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(raw_data)
    df['symbol'] = symbol
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Ensure standard columns
    required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    return df[required_cols]

def transform_news(raw_data, symbol):
    """Cleans and structures news data."""
    if not raw_data:
        return pd.DataFrame()
        
    cleaned_news = []
    for item in raw_data:
        cleaned_news.append({
            'symbol': symbol,
            'date': item.get('date'),
            'title': item.get('title'),
            'content': clean_html(item.get('content', '')),
            'source': item.get('source'),
            'url': item.get('link'),
            'sentiment_score': item.get('sentiment', {}).get('polarity', 0) # EODHD sometimes provides this
        })
        
    return pd.DataFrame(cleaned_news)
