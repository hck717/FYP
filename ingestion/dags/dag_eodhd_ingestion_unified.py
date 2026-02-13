"""
EODHD All-in-One Data Ingestion DAG
Uses environment variables from unified .env file
No overlap with FMP ingestion
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

# Configuration from .env
EODHD_API_KEY = os.getenv('EODHD_API_KEY')
BASE_URL = "https://eodhd.com/api"
BASE_OUTPUT_DIR = "/opt/airflow/etl/agent_data"

# Get tickers from environment and convert to EODHD format (SYMBOL.US)
TRACKED_TICKERS_RAW = os.getenv('TRACKED_TICKERS', 'AAPL').split(',')
TICKERS = [f"{ticker.strip()}.US" for ticker in TRACKED_TICKERS_RAW]
TICKER_SYMBOLS = [ticker.strip() for ticker in TRACKED_TICKERS_RAW]

# Rate limiting from environment
EODHD_RATE_LIMIT = int(os.getenv('EODHD_RATE_LIMIT', '1000'))
RATE_LIMIT_DELAY = 60.0 / EODHD_RATE_LIMIT

# Agent configurations (EODHD only - NO OVERLAP with FMP)
AGENT_CONFIGS = {
    "quantitative_fundamental": {
        "endpoints": [
            ("realtime_quote", "real-time/{ticker}", {}),
            ("historical_prices_eod", "eod/{ticker}", {"from": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')}),
            ("intraday_1m", "intraday/{ticker}", {"interval": "1m", "to": int(time.time())}),
            ("intraday_5m", "intraday/{ticker}", {"interval": "5m", "to": int(time.time())}),
        ]
    },
    "financial_modeling": {
        "endpoints": [
            ("historical_prices_weekly", "eod/{ticker}", {
                "from": (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), 
                "period": "w"
            }),
            ("historical_prices_monthly", "eod/{ticker}", {
                "from": (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'), 
                "period": "m"
            }),
            ("dividends_history", "div/{ticker}", {}),
            ("splits_history", "splits/{ticker}", {}),
        ]
    },
    "business_analyst": {
        "endpoints": [
            ("financial_news", "news", {"s": "{ticker}", "limit": 50}),
            ("sentiment_trends", "sentiments", {"s": "{ticker}"}),
        ]
    }
}


def get_data_hash(data):
    """Calculate MD5 hash of data"""
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load_metadata(agent_name, ticker_symbol):
    """Load metadata for incremental updates"""
    metadata_path = Path(BASE_OUTPUT_DIR) / agent_name / ticker_symbol / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}


def save_metadata(agent_name, ticker_symbol, metadata):
    """Save metadata for incremental updates"""
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
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        print(f"Error fetching {endpoint}: {e}")
        raise


def save_data(agent_name, ticker_symbol, data_name, data, metadata):
    """Save data with incremental update logic"""
    agent_dir = Path(BASE_OUTPUT_DIR) / agent_name / ticker_symbol
    agent_dir.mkdir(parents=True, exist_ok=True)

    data_hash = get_data_hash(data)
    last_hash = metadata.get(data_name, {}).get('hash')

    if last_hash == data_hash:
        print(f"Skipped (no changes): {data_name}")
        return False

    # Save JSON
    json_path = agent_dir / f"{data_name}.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    # Save CSV
    csv_path = agent_dir / f"{data_name}.csv"
    if isinstance(data, list) and len(data) > 0:
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
        df.to_csv(csv_path, index=False)

    metadata[data_name] = {
        'hash': data_hash,
        'last_updated': datetime.now().isoformat(),
        'record_count': len(data) if isinstance(data, list) else 1,
        'source': 'eodhd'
    }

    print(f"Updated: {data_name}")
    return True


def scrape_agent_ticker(agent_name, ticker, ticker_symbol, **context):
    """Scrape data for a specific agent and ticker"""
    print(f"\n{'='*60}")
    print(f"[EODHD] Agent: {agent_name} | Ticker: {ticker}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    config = AGENT_CONFIGS[agent_name]
    metadata = load_metadata(agent_name, ticker_symbol)
    updates_made = 0

    for idx, (data_name, endpoint_template, params) in enumerate(config['endpoints'], 1):
        print(f"\n[{idx}/{len(config['endpoints'])}] Fetching: {data_name}...")

        endpoint = endpoint_template.format(ticker=ticker)
        params_copy = {k: v.format(ticker=ticker) if isinstance(v, str) else v 
                      for k, v in params.items()}

        try:
            data = fetch_data(endpoint, params_copy)
            if data:
                if save_data(agent_name, ticker_symbol, data_name, data, metadata):
                    updates_made += 1
        except Exception as e:
            print(f"Failed to fetch {data_name}: {e}")

        time.sleep(RATE_LIMIT_DELAY)

    save_metadata(agent_name, ticker_symbol, metadata)

    print(f"\n{'─'*60}")
    print(f"[EODHD] {agent_name}/{ticker_symbol}: {updates_made} files updated")
    print(f"{'─'*60}")

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
    """Generate summary report of all updates"""
    ti = context['task_instance']

    summary = {}
    total_updates = 0

    for agent_name in AGENT_CONFIGS.keys():
        for ticker_symbol in TICKER_SYMBOLS:
            key = f'{agent_name}_{ticker_symbol}_updates'
            updates = ti.xcom_pull(task_ids=f'eodhd_scrape_{agent_name}_{ticker_symbol}', key=key)
            summary[f'{agent_name}/{ticker_symbol}'] = updates or 0
            total_updates += (updates or 0)

    print(f"\n{'='*70}")
    print(f"EODHD INGESTION SUMMARY")
    print(f"{'='*70}")
    for key, value in summary.items():
        print(f"{key}: {value} files updated")
    print(f"{'='*70}")
    print(f"Total updates: {total_updates}")
    print(f"Tracked tickers from .env: {', '.join(TICKER_SYMBOLS)}")
    print(f"{'='*70}")

    return summary


# Define the DAG
with DAG(
    'eodhd_incremental_ingestion',
    default_args=default_args,
    description='EODHD All-in-One incremental data ingestion (uses .env config)',
    schedule_interval='0 * * * *',
    start_date=days_ago(1),
    catchup=False,
    tags=['eodhd', 'realtime', 'incremental', 'agents'],
) as dag:

    scrape_tasks = {}

    for agent_name in AGENT_CONFIGS.keys():
        for ticker, ticker_symbol in zip(TICKERS, TICKER_SYMBOLS):
            task = create_scrape_task(agent_name, ticker, ticker_symbol)
            scrape_tasks[f'{agent_name}_{ticker_symbol}'] = task

    summary_task = PythonOperator(
        task_id='eodhd_generate_summary_report',
        python_callable=report_summary,
        provide_context=True,
    )

    for task in scrape_tasks.values():
        task >> summary_task