"""
FMP Stable API Data Ingestion DAG
Uses FMP's new /stable/ endpoint structure with free-tier endpoints
Compatible with basic FMP API keys
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
FMP_API_KEY = os.getenv('FMP_API_KEY')
BASE_URL = "https://financialmodelingprep.com/stable"
BASE_OUTPUT_DIR = "/opt/airflow/etl/agent_data"

TICKERS = os.getenv('TRACKED_TICKERS', 'AAPL').split(',')
FMP_RATE_LIMIT = int(os.getenv('FMP_RATE_LIMIT', '300'))
RATE_LIMIT_DELAY = 60.0 / FMP_RATE_LIMIT

# FREE ENDPOINTS from /stable/ API
AGENT_CONFIGS = {
    "business_analyst": {
        "endpoints": [
            # Company information (FREE)
            ("company_profile", "profile", {"symbol": "{ticker}"}),
            ("stock_quote", "quote", {"symbol": "{ticker}"}),
            ("key_executives", "key-executives", {"symbol": "{ticker}"}),
            # News (FREE)
            ("stock_news", "news/stock", {"symbols": "{ticker}"}),
        ]
    },
    "quantitative_fundamental": {
        "endpoints": [
            # Financial metrics (FREE)
            ("key_metrics", "key-metrics", {"symbol": "{ticker}"}),
            ("financial_ratios", "ratios", {"symbol": "{ticker}"}),
            ("key_metrics_ttm", "key-metrics-ttm", {"symbol": "{ticker}"}),
            ("ratios_ttm", "ratios-ttm", {"symbol": "{ticker}"}),
            ("financial_scores", "financial-scores", {"symbol": "{ticker}"}),
        ]
    },
    "financial_modeling": {
        "endpoints": [
            # Financial statements (FREE)
            ("income_statement", "income-statement", {"symbol": "{ticker}", "limit": 5}),
            ("balance_sheet", "balance-sheet-statement", {"symbol": "{ticker}", "limit": 5}),
            ("cash_flow", "cash-flow-statement", {"symbol": "{ticker}", "limit": 5}),
            # Growth and valuation (FREE)
            ("financial_growth", "financial-growth", {"symbol": "{ticker}"}),
            ("enterprise_values", "enterprise-values", {"symbol": "{ticker}"}),
            ("owner_earnings", "owner-earnings", {"symbol": "{ticker}"}),
            # Segmentation (may be FREE)
            ("revenue_product_segmentation", "revenue-product-segmentation", {"symbol": "{ticker}"}),
            ("revenue_geographic_segmentation", "revenue-geographic-segmentation", {"symbol": "{ticker}"}),
        ]
    }
}

def get_data_hash(data):
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()

def load_metadata(agent_name, ticker):
    metadata_path = Path(BASE_OUTPUT_DIR) / agent_name / ticker / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}

def save_metadata(agent_name, ticker, metadata):
    metadata_path = Path(BASE_OUTPUT_DIR) / agent_name / ticker / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def fetch_data(endpoint, params=None):
    """Fetch from FMP /stable/ API"""
    url = f"{BASE_URL}/{endpoint}"
    if params is None:
        params = {}
    params['apikey'] = FMP_API_KEY

    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"  URL: {endpoint}")
        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            # Check for error message in response
            if isinstance(data, dict) and 'Error Message' in data:
                print(f"  API Error: {data['Error Message'][:100]}")
                return None
            if isinstance(data, dict) and 'error' in data:
                print(f"  API Error: {data['error'][:100]}")
                return None
            return data
        elif response.status_code == 403:
            print(f"  403 Forbidden - Premium endpoint or invalid key")
            return None
        else:
            print(f"  Error {response.status_code}: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"  Exception: {e}")
        return None

def save_data(agent_name, ticker, data_name, data, metadata):
    if not data:
        print(f"  ⊘ Skipped (no data): {data_name}")
        return False

    agent_dir = Path(BASE_OUTPUT_DIR) / agent_name / ticker
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

    # Save CSV if data is a list
    if isinstance(data, list) and len(data) > 0:
        csv_path = agent_dir / f"{data_name}.csv"
        try:
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"  Warning: Could not save CSV: {e}")

    metadata[data_name] = {
        'hash': data_hash,
        'last_updated': datetime.now().isoformat(),
        'record_count': len(data) if isinstance(data, list) else 1,
        'source': 'fmp_stable'
    }

    print(f"  ✓ Updated: {data_name} ({metadata[data_name]['record_count']} records)")
    return True

def scrape_agent_ticker(agent_name, ticker, **context):
    print(f"\n{'='*70}")
    print(f"[FMP Stable] Agent: {agent_name} | Ticker: {ticker}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    config = AGENT_CONFIGS[agent_name]
    metadata = load_metadata(agent_name, ticker)
    updates_made = 0
    errors = 0

    for idx, (data_name, endpoint, params) in enumerate(config['endpoints'], 1):
        print(f"\n[{idx}/{len(config['endpoints'])}] Fetching: {data_name}...")

        # Replace {ticker} placeholder
        params_copy = {k: v.format(ticker=ticker) if isinstance(v, str) else v 
                      for k, v in params.items()}

        try:
            data = fetch_data(endpoint, params_copy)
            if data and save_data(agent_name, ticker, data_name, data, metadata):
                updates_made += 1
            elif not data:
                errors += 1
        except Exception as e:
            print(f"  Failed: {e}")
            errors += 1

        time.sleep(RATE_LIMIT_DELAY)

    save_metadata(agent_name, ticker, metadata)

    print(f"\n{'─'*70}")
    print(f"[FMP Stable] {agent_name}/{ticker}:")
    print(f"  Updates: {updates_made}")
    print(f"  Errors: {errors}")
    print(f"  Success rate: {updates_made}/{len(config['endpoints'])}")
    print(f"{'─'*70}")

    context['task_instance'].xcom_push(
        key=f'{agent_name}_{ticker}_updates',
        value=updates_made
    )

    return updates_made

def create_scrape_task(agent_name, ticker):
    return PythonOperator(
        task_id=f'fmp_stable_scrape_{agent_name}_{ticker}',
        python_callable=scrape_agent_ticker,
        op_kwargs={'agent_name': agent_name, 'ticker': ticker},
        provide_context=True,
    )

def report_summary(**context):
    ti = context['task_instance']

    summary = {}
    total_updates = 0

    for agent_name in AGENT_CONFIGS.keys():
        for ticker in TICKERS:
            key = f'{agent_name}_{ticker}_updates'
            task_id = f'fmp_stable_scrape_{agent_name}_{ticker}'
            updates = ti.xcom_pull(task_ids=task_id, key=key)
            summary[f'{agent_name}/{ticker}'] = updates or 0
            total_updates += (updates or 0)

    print(f"\n{'='*70}")
    print(f"FMP STABLE API INGESTION SUMMARY")
    print(f"{'='*70}")
    for key, value in summary.items():
        print(f"{key}: {value} files updated")
    print(f"{'='*70}")
    print(f"Total updates: {total_updates}")
    print(f"Tracked tickers: {', '.join(TICKERS)}")
    print(f"{'='*70}")

    return summary

# Define the DAG
with DAG(
    'fmp_stable_ingestion',
    default_args=default_args,
    description='FMP /stable/ API data ingestion (free endpoints)',
    schedule_interval='0 */6 * * *',
    start_date=days_ago(1),
    catchup=False,
    tags=['fmp', 'stable', 'financial', 'agents', 'free'],
) as dag:

    scrape_tasks = {}

    for agent_name in AGENT_CONFIGS.keys():
        for ticker in TICKERS:
            task = create_scrape_task(agent_name, ticker)
            scrape_tasks[f'{agent_name}_{ticker}'] = task

    summary_task = PythonOperator(
        task_id='fmp_stable_generate_summary',
        python_callable=report_summary,
        provide_context=True,
    )

    for task in scrape_tasks.values():
        task >> summary_task