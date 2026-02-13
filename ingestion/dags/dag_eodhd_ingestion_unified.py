
"""
EODHD Complete Data Ingestion DAG - 100% Requirements Match

Implements ALL EODHD data types from architecture requirements:
- Business Analyst: News, sentiment (for Qdrant embeddings)
- Quantitative: Real-time quotes, intraday, fundamentals, historical prices
- Financial Modeling: Weekly/monthly prices, dividends, splits, earnings

Complements FMP data for dual-path verification
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import requests
import json
import pandas as pd
import os
from pathlib import Path
import hashlib
import time

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Configuration
EODHD_API_KEY = os.getenv('EODHD_API_KEY')
BASE_URL = "https://eodhd.com/api"
BASE_OUTPUT_DIR = "/opt/airflow/etl/agent_data"

# Get tickers
TRACKED_TICKERS_RAW = os.getenv('TRACKED_TICKERS', 'AAPL').split(',')
TICKERS = [f"{ticker.strip()}.US" for ticker in TRACKED_TICKERS_RAW]
TICKER_SYMBOLS = [ticker.strip() for ticker in TRACKED_TICKERS_RAW]

# Rate limiting
EODHD_RATE_LIMIT = int(os.getenv('EODHD_RATE_LIMIT', '1000'))
RATE_LIMIT_DELAY = 60.0 / EODHD_RATE_LIMIT

# COMPLETE AGENT CONFIGURATIONS - 100% Requirements Match
AGENT_CONFIGS = {
    "business_analyst": {
        "endpoints": [
            # News (for Qdrant semantic search)
            ("financial_news", "news", {"s": "{ticker_symbol}", "limit": 100}, "qdrant_prep"),

            # Sentiment trends
            ("sentiment_trends", "sentiments", {"s": "{ticker_symbol}"}, "qdrant_prep"),

            # Company profile (dual source with FMP)
            ("company_profile", "fundamentals/{ticker}", {}, "neo4j"),
        ]
    },

    "quantitative_fundamental": {
        "endpoints": [
            # Real-time quote (dual-path with FMP)
            ("realtime_quote", "real-time/{ticker}", {}, "postgresql"),

            # Historical EOD prices (dual-path with FMP)
            ("historical_prices_eod", "eod/{ticker}", {
                "from": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            }, "postgresql"),

            # Intraday data (high-frequency for quantitative analysis)
            ("intraday_1m", "intraday/{ticker}", {
                "interval": "1m",
                "to": int(time.time())
            }, "postgresql"),
            ("intraday_5m", "intraday/{ticker}", {
                "interval": "5m", 
                "to": int(time.time())
            }, "postgresql"),
            ("intraday_15m", "intraday/{ticker}", {
                "interval": "15m",
                "to": int(time.time())
            }, "postgresql"),
            ("intraday_1h", "intraday/{ticker}", {
                "interval": "1h",
                "to": int(time.time())
            }, "postgresql"),

            # Fundamentals (dual-path verification)
            ("fundamentals", "fundamentals/{ticker}", {}, "postgresql"),

            # Options data (for volatility calculations)
            ("options_data", "options/{ticker}", {}, "postgresql"),

            # Technical indicators
            ("technical_sma", "technical/{ticker}", {"function": "sma", "period": 50}, "postgresql"),
            ("technical_ema", "technical/{ticker}", {"function": "ema", "period": 20}, "postgresql"),

            # Live stock price
            ("live_stock_price", "real-time/{ticker}", {}, "postgresql"),
        ]
    },

    "financial_modeling": {
        "endpoints": [
            # Weekly prices (medium-term trends)
            ("historical_prices_weekly", "eod/{ticker}", {
                "from": (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                "period": "w"
            }, "postgresql"),

            # Monthly prices (long-term trends)
            ("historical_prices_monthly", "eod/{ticker}", {
                "from": (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'),
                "period": "m"
            }, "postgresql"),

            # Dividends history (for DDM)
            ("dividends_history", "div/{ticker}", {}, "postgresql"),

            # Stock splits (for price adjustments)
            ("splits_history", "splits/{ticker}", {}, "postgresql"),

            # Earnings history (analyst comparison)
            ("earnings_history", "calendar/earnings", {"symbols": "{ticker_symbol}"}, "postgresql"),

            # IPO calendar (for peer analysis)
            ("ipo_calendar", "calendar/ipos", {"from": (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')}, "postgresql"),

            # Fundamentals (balance sheet, income, cash flow)
            ("fundamentals_full", "fundamentals/{ticker}", {}, "postgresql"),

            # Analyst estimates from fundamentals
            ("analyst_estimates_eodhd", "fundamentals/{ticker}", {}, "postgresql"),

            # Economic calendar (macro factors)
            ("economic_calendar", "economic-events", {"from": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')}, "postgresql"),

            # Exchange traded data
            ("exchange_details", "exchange-details/{ticker}", {}, "postgresql"),

            # Bulk EOD (for peer comparison)
            ("bulk_eod_us", "eod-bulk-last-day/US", {}, "postgresql"),
        ]
    }
}

def get_data_hash(data):
    """Calculate MD5 hash for change detection"""
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()

def load_metadata(agent_name, ticker_symbol):
    """Load metadata for incremental updates"""
    metadata_path = Path(BASE_OUTPUT_DIR) / agent_name / ticker_symbol / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}

def save_metadata(agent_name, ticker_symbol, metadata):
    """Save metadata with storage destination info"""
    metadata_path = Path(BASE_OUTPUT_DIR) / agent_name / ticker_symbol / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def fetch_data(endpoint, params=None):
    """Generic fetch function with error handling"""
    url = f"{BASE_URL}/{endpoint}"
    if params is None:
        params = {}
    params['api_token'] = EODHD_API_KEY
    params['fmt'] = 'json'

    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"  URL: {endpoint}")
        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            # Check for empty responses
            if isinstance(data, list) and len(data) == 0:
                print(f"  ⊘ Empty response (no data available)")
                return None

            return data
        else:
            print(f"  Error {response.status_code}: {response.text[:200]}")
            return None

    except Exception as e:
        print(f"  Exception: {e}")
        return None

def save_data(agent_name, ticker_symbol, data_name, data, metadata, storage_dest):
    """Save data with storage destination metadata"""
    if not data:
        print(f"  ⊘ Skipped (no data): {data_name}")
        return False

    agent_dir = Path(BASE_OUTPUT_DIR) / agent_name / ticker_symbol
    agent_dir.mkdir(parents=True, exist_ok=True)

    data_hash = get_data_hash(data)
    last_hash = metadata.get(data_name, {}).get('hash')

    if last_hash == data_hash:
        print(f"  = Skipped (no changes): {data_name}")
        return False

    # Save JSON
    json_path = agent_dir / f"{data_name}.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    # Save CSV
    csv_path = agent_dir / f"{data_name}.csv"
    try:
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)
        elif isinstance(data, dict):
            # For fundamentals, flatten structure
            if 'General' in data or 'Financials' in data:
                # Complex nested structure - save as is
                pass
            else:
                df = pd.DataFrame([data])
                df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"  Warning: Could not save CSV: {e}")

    # Update metadata
    metadata[data_name] = {
        'hash': data_hash,
        'last_updated': datetime.now().isoformat(),
        'record_count': len(data) if isinstance(data, list) else 1,
        'source': 'eodhd',
        'storage_destination': storage_dest
    }

    print(f"  ✓ Updated: {data_name} ({metadata[data_name]['record_count']} records) → {storage_dest}")
    return True

def scrape_agent_ticker(agent_name, ticker, ticker_symbol, **context):
    """Scrape data for specific agent and ticker"""
    print(f"\n{'='*70}")
    print(f"[EODHD Complete] Agent: {agent_name} | Ticker: {ticker}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    config = AGENT_CONFIGS[agent_name]
    metadata = load_metadata(agent_name, ticker_symbol)

    updates_made = 0
    errors = 0

    for idx, endpoint_config in enumerate(config['endpoints'], 1):
        data_name, endpoint_template, params, storage_dest = endpoint_config

        print(f"\n[{idx}/{len(config['endpoints'])}] Fetching: {data_name}...")

        # Replace placeholders
        endpoint = endpoint_template.format(ticker=ticker)
        params_copy = {k: v.format(ticker=ticker, ticker_symbol=ticker_symbol) if isinstance(v, str) else v 
                      for k, v in params.items()}

        try:
            data = fetch_data(endpoint, params_copy)
            if data and save_data(agent_name, ticker_symbol, data_name, data, metadata, storage_dest):
                updates_made += 1
            elif not data:
                errors += 1
        except Exception as e:
            print(f"  Failed: {e}")
            errors += 1

        time.sleep(RATE_LIMIT_DELAY)

    save_metadata(agent_name, ticker_symbol, metadata)

    print(f"\n{'─'*70}")
    print(f"[EODHD Complete] {agent_name}/{ticker_symbol}:")
    print(f"  Updates: {updates_made}")
    print(f"  Errors: {errors}")
    print(f"  Success rate: {updates_made}/{len(config['endpoints'])}")
    print(f"{'─'*70}")

    context['task_instance'].xcom_push(
        key=f'{agent_name}_{ticker_symbol}_updates',
        value=updates_made
    )

    return updates_made

def create_scrape_task(agent_name, ticker, ticker_symbol):
    """Factory function to create scraping tasks"""
    return PythonOperator(
        task_id=f'eodhd_scrape_{agent_name}_{ticker_symbol}',
        python_callable=scrape_agent_ticker,
        op_kwargs={
            'agent_name': agent_name,
            'ticker': ticker,
            'ticker_symbol': ticker_symbol
        },
        provide_context=True,
    )

def report_summary(**context):
    """Generate comprehensive summary report"""
    ti = context['task_instance']

    summary = {}
    total_updates = 0
    storage_stats = {'neo4j': 0, 'postgresql': 0, 'qdrant_prep': 0}

    for agent_name in AGENT_CONFIGS.keys():
        for ticker_symbol in TICKER_SYMBOLS:
            key = f'{agent_name}_{ticker_symbol}_updates'
            task_id = f'eodhd_scrape_{agent_name}_{ticker_symbol}'
            updates = ti.xcom_pull(task_ids=task_id, key=key)
            summary[f'{agent_name}/{ticker_symbol}'] = updates or 0
            total_updates += (updates or 0)

            # Count storage destinations
            for endpoint_config in AGENT_CONFIGS[agent_name]['endpoints']:
                if updates and len(endpoint_config) >= 4:
                    storage_dest = endpoint_config[3]
                    storage_stats[storage_dest] = storage_stats.get(storage_dest, 0) + 1

    print(f"\n{'='*70}")
    print(f"EODHD COMPLETE INGESTION SUMMARY")
    print(f"{'='*70}")
    for key, value in summary.items():
        print(f"{key}: {value} files updated")
    print(f"{'='*70}")
    print(f"Total updates: {total_updates}")
    print(f"Tracked tickers: {', '.join(TICKER_SYMBOLS)}")
    print(f"\nStorage Destinations:")
    print(f"  Neo4j:       {storage_stats.get('neo4j', 0)} endpoints")
    print(f"  PostgreSQL:  {storage_stats.get('postgresql', 0)} endpoints")
    print(f"  Qdrant prep: {storage_stats.get('qdrant_prep', 0)} endpoints")
    print(f"{'='*70}")

    return summary

# Define the DAG
with DAG(
    'eodhd_complete_ingestion',
    default_args=default_args,
    description='EODHD Complete ingestion - 100% requirements match',
    schedule_interval='0 * * * *',  # Every hour
    start_date=days_ago(1),
    catchup=False,
    tags=['eodhd', 'complete', 'all-data', 'production'],
) as dag:

    scrape_tasks = {}

    for agent_name in AGENT_CONFIGS.keys():
        for ticker, ticker_symbol in zip(TICKERS, TICKER_SYMBOLS):
            task = create_scrape_task(agent_name, ticker, ticker_symbol)
            scrape_tasks[f'{agent_name}_{ticker_symbol}'] = task

    summary_task = PythonOperator(
        task_id='eodhd_generate_summary',
        python_callable=report_summary,
        provide_context=True,
    )

    # Set dependencies
    for task in scrape_tasks.values():
        task >> summary_task