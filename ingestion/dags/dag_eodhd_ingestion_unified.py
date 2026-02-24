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
import sys
from pathlib import Path
import hashlib
import time

# ETL loaders
sys.path.insert(0, "/opt/airflow/etl")
from load_postgres import load_postgres_for_agent_ticker
from load_neo4j import load_neo4j_for_agent_ticker
from load_qdrant import load_qdrant_for_agent_ticker

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

TRACKED_TICKERS_RAW = os.getenv('TRACKED_TICKERS', 'AAPL').split(',')
TICKERS = [f"{ticker.strip()}.US" for ticker in TRACKED_TICKERS_RAW]
TICKER_SYMBOLS = [ticker.strip() for ticker in TRACKED_TICKERS_RAW]

EODHD_RATE_LIMIT = int(os.getenv('EODHD_RATE_LIMIT', '1000'))
RATE_LIMIT_DELAY = 60.0 / EODHD_RATE_LIMIT

AGENT_CONFIGS = {
    "business_analyst": {
        "endpoints": [
            # financial_news → Qdrant (news embeddings for RAG)
            ("financial_news",   "news",                  {"s": "{ticker_symbol}", "limit": 100}, "qdrant_prep"),
            # sentiment_trends → PostgreSQL (structured scores: bearish/bullish %)
            # NOT qdrant_prep — single-row dict has no embeddable text
            ("sentiment_trends", "sentiments",            {"s": "{ticker_symbol}"},               "postgresql"),
            # company_profile → Neo4j (company node with 40+ properties)
            ("company_profile",  "fundamentals/{ticker}", {},                                    "neo4j"),
        ]
    },
    "quantitative_fundamental": {
        "endpoints": [
            ("realtime_quote",          "real-time/{ticker}",    {},                                                                    "postgresql"),
            ("historical_prices_eod",   "eod/{ticker}",          {"from": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')},  "postgresql"),
            ("intraday_1m",             "intraday/{ticker}",     {"interval": "1m",  "to": int(time.time())},                          "postgresql"),
            ("intraday_5m",             "intraday/{ticker}",     {"interval": "5m",  "to": int(time.time())},                          "postgresql"),
            ("intraday_15m",            "intraday/{ticker}",     {"interval": "15m", "to": int(time.time())},                          "postgresql"),
            ("intraday_1h",             "intraday/{ticker}",     {"interval": "1h",  "to": int(time.time())},                          "postgresql"),
            ("fundamentals",            "fundamentals/{ticker}", {},                                                                    "postgresql"),
            ("options_data",            "options/{ticker}",      {},                                                                    "postgresql"),
            ("technical_sma",           "technical/{ticker}",    {"function": "sma", "period": 50},                                    "postgresql"),
            ("technical_ema",           "technical/{ticker}",    {"function": "ema", "period": 20},                                    "postgresql"),
            ("live_stock_price",        "real-time/{ticker}",    {},                                                                    "postgresql"),
        ]
    },
    "financial_modeling": {
        "endpoints": [
            ("historical_prices_weekly",  "eod/{ticker}",              {"from": (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), "period": "w"}, "postgresql"),
            ("historical_prices_monthly", "eod/{ticker}",              {"from": (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'), "period": "m"}, "postgresql"),
            ("dividends_history",         "div/{ticker}",              {},                                                                                    "postgresql"),
            ("splits_history",            "splits/{ticker}",           {},                                                                                    "postgresql"),
            ("earnings_history",          "calendar/earnings",         {"symbols": "{ticker_symbol}"},                                                        "postgresql"),
            ("ipo_calendar",              "calendar/ipos",             {"from": (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')},                 "postgresql"),
            ("fundamentals_full",         "fundamentals/{ticker}",     {},                                                                                    "postgresql"),
            ("analyst_estimates_eodhd",   "fundamentals/{ticker}",     {},                                                                                    "postgresql"),
            ("economic_calendar",         "economic-events",           {"from": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')},                  "postgresql"),
            ("exchange_details",          "exchange-details/{ticker}", {},                                                                                    "postgresql"),
            ("bulk_eod_us",               "eod-bulk-last-day/US",      {},                                                                                    "postgresql"),
        ]
    }
}


def get_data_hash(data):
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load_metadata(agent_name, ticker_symbol):
    metadata_path = Path(BASE_OUTPUT_DIR) / agent_name / ticker_symbol / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}


def save_metadata(agent_name, ticker_symbol, metadata):
    metadata_path = Path(BASE_OUTPUT_DIR) / agent_name / ticker_symbol / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def fetch_data(endpoint, params=None):
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
            if isinstance(data, list) and len(data) == 0:
                print(f"  ⊘ Empty response")
                return None
            return data
        else:
            print(f"  Error {response.status_code}: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"  Exception: {e}")
        return None


def _flatten_eodhd_fundamentals(data: dict) -> dict:
    """
    EODHD fundamentals/{ticker} returns a deeply nested dict.
    Strategy:
      1. Pull the 'General' section (company metadata) as the primary flat row.
      2. Merge in select scalar fields from Highlights + Valuation sections.
      3. Drop any remaining nested dicts/lists so the result is always flat.
    """
    general = data.get("General", {})
    flat = {k: v for k, v in general.items() if not isinstance(v, (dict, list))}

    highlights = data.get("Highlights", {})
    for key in ["MarketCapitalization", "EBITDA", "PERatio", "EPS",
                "DividendYield", "ProfitMargin", "RevenueGrowthYoY",
                "EPSEstimateCurrentYear", "WallStreetTargetPrice"]:
        if key in highlights and not isinstance(highlights[key], (dict, list)):
            flat[f"Highlights_{key}"] = highlights[key]

    valuation = data.get("Valuation", {})
    for key in ["TrailingPE", "ForwardPE", "PriceSalesTTM",
                "PriceBookMRQ", "EnterpriseValue", "EnterpriseValueRevenue",
                "EnterpriseValueEbitda"]:
        if key in valuation and not isinstance(valuation[key], (dict, list)):
            flat[f"Valuation_{key}"] = valuation[key]

    return flat


def save_data(agent_name, ticker_symbol, data_name, data, metadata, storage_dest):
    if not data:
        print(f"  ⊘ Skipped (no data): {data_name}")
        return False

    agent_dir = Path(BASE_OUTPUT_DIR) / agent_name / ticker_symbol
    agent_dir.mkdir(parents=True, exist_ok=True)

    data_hash = get_data_hash(data)
    if metadata.get(data_name, {}).get('hash') == data_hash:
        print(f"  = Skipped (no changes): {data_name}")
        return False

    # Always write JSON
    with open(agent_dir / f"{data_name}.json", 'w') as f:
        json.dump(data, f, indent=2)

    # Write CSV
    try:
        if isinstance(data, list) and len(data) > 0:
            pd.DataFrame(data).to_csv(agent_dir / f"{data_name}.csv", index=False)

        elif isinstance(data, dict):
            if "General" in data:
                flat_row = _flatten_eodhd_fundamentals(data)
            else:
                list_val = next(
                    (v for v in data.values() if isinstance(v, list) and len(v) > 0),
                    None
                )
                if list_val:
                    pd.DataFrame(list_val).to_csv(agent_dir / f"{data_name}.csv", index=False)
                    flat_row = None
                else:
                    flat_row = {k: str(v) for k, v in data.items()
                                if not isinstance(v, (dict, list))}

            if flat_row is not None:
                if flat_row:
                    pd.DataFrame([flat_row]).to_csv(agent_dir / f"{data_name}.csv", index=False)
                else:
                    print(f"  Warning: flat_row empty for {data_name}, skipping CSV")

    except Exception as e:
        print(f"  Warning: Could not save CSV for {data_name}: {e}")

    metadata[data_name] = {
        'hash': data_hash,
        'last_updated': datetime.now().isoformat(),
        'record_count': len(data) if isinstance(data, list) else 1,
        'source': 'eodhd',
        'storage_destination': storage_dest,
    }
    print(f"  ✓ Updated: {data_name} ({metadata[data_name]['record_count']} records) → {storage_dest}")
    return True


def scrape_agent_ticker(agent_name, ticker, ticker_symbol, **context):
    print(f"\n{'='*70}")
    print(f"[EODHD] Agent: {agent_name} | Ticker: {ticker}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    config = AGENT_CONFIGS[agent_name]
    metadata = load_metadata(agent_name, ticker_symbol)
    updates_made = 0
    errors = 0

    for idx, (data_name, endpoint_template, params, storage_dest) in enumerate(config['endpoints'], 1):
        print(f"\n[{idx}/{len(config['endpoints'])}] Fetching: {data_name}...")
        endpoint = endpoint_template.format(ticker=ticker)
        params_copy = {
            k: v.format(ticker=ticker, ticker_symbol=ticker_symbol) if isinstance(v, str) else v
            for k, v in params.items()
        }
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
    print(f"[EODHD] {agent_name}/{ticker_symbol}: {updates_made} updates, {errors} errors")
    print(f"{'─'*70}")

    context['task_instance'].xcom_push(
        key=f'{agent_name}_{ticker_symbol}_updates',
        value=updates_made
    )
    return updates_made


def report_summary(**context):
    ti = context['task_instance']
    summary = {}
    total_updates = 0

    for agent_name in AGENT_CONFIGS.keys():
        for ticker_symbol in TICKER_SYMBOLS:
            task_id = f'eodhd_scrape_{agent_name}_{ticker_symbol}'
            updates = ti.xcom_pull(task_ids=task_id, key=f'{agent_name}_{ticker_symbol}_updates')
            summary[f'{agent_name}/{ticker_symbol}'] = updates or 0
            total_updates += (updates or 0)

    print(f"\n{'='*70}")
    print(f"EODHD INGESTION SUMMARY — Total updates: {total_updates}")
    print(f"{'='*70}")
    for k, v in summary.items():
        print(f"  {k}: {v} updates")
    print(f"{'='*70}")
    return summary


# ── DAG ──────────────────────────────────────────────────────────────────────
with DAG(
    'eodhd_complete_ingestion',
    default_args=default_args,
    description='EODHD Complete ingestion - all agents, all tickers',
    schedule_interval='0 1 * * *',
    start_date=days_ago(1),
    catchup=False,
    tags=['eodhd', 'complete', 'production'],
) as dag:

    scrape_tasks      = {}
    load_pg_tasks     = {}
    load_neo4j_tasks  = {}
    load_qdrant_tasks = {}

    for agent_name in AGENT_CONFIGS.keys():
        for ticker, ticker_symbol in zip(TICKERS, TICKER_SYMBOLS):
            key = f'{agent_name}_{ticker_symbol}'

            scrape_tasks[key] = PythonOperator(
                task_id=f'eodhd_scrape_{agent_name}_{ticker_symbol}',
                python_callable=scrape_agent_ticker,
                op_kwargs={'agent_name': agent_name, 'ticker': ticker, 'ticker_symbol': ticker_symbol},
                provide_context=True,
            )
            load_pg_tasks[key] = PythonOperator(
                task_id=f'eodhd_load_postgres_{agent_name}_{ticker_symbol}',
                python_callable=load_postgres_for_agent_ticker,
                op_kwargs={'agent_name': agent_name, 'ticker_symbol': ticker_symbol},
            )
            load_neo4j_tasks[key] = PythonOperator(
                task_id=f'eodhd_load_neo4j_{agent_name}_{ticker_symbol}',
                python_callable=load_neo4j_for_agent_ticker,
                op_kwargs={'agent_name': agent_name, 'ticker_symbol': ticker_symbol},
            )
            load_qdrant_tasks[key] = PythonOperator(
                task_id=f'eodhd_load_qdrant_{agent_name}_{ticker_symbol}',
                python_callable=load_qdrant_for_agent_ticker,
                op_kwargs={'agent_name': agent_name, 'ticker_symbol': ticker_symbol},
                # Qdrant tasks get extra retries since Ollama may need warm-up time
                retries=3,
                retry_delay=timedelta(minutes=2),
            )

    summary_task = PythonOperator(
        task_id='eodhd_generate_summary',
        python_callable=report_summary,
        provide_context=True,
    )

    # scrape → [postgres ∥ neo4j ∥ qdrant] → summary
    for key in scrape_tasks:
        scrape_tasks[key] >> load_pg_tasks[key]
        scrape_tasks[key] >> load_neo4j_tasks[key]
        scrape_tasks[key] >> load_qdrant_tasks[key]
        load_pg_tasks[key]     >> summary_task
        load_neo4j_tasks[key]  >> summary_task
        load_qdrant_tasks[key] >> summary_task
