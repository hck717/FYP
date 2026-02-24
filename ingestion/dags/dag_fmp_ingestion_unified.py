"""
FMP Complete Data Ingestion DAG - 100% Requirements Match

Implements ALL data types from architecture requirements:
- Business Analyst: SEC filings, transcripts, risk factors, strategies
- Quantitative: Financial statements, ratios, scores, historical data
- Financial Modeling: DCF inputs, segments, analyst data, benchmarks

Handles both free and premium FMP endpoints with graceful fallback
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
FMP_API_KEY = os.getenv('FMP_API_KEY')
BASE_URL = "https://financialmodelingprep.com/stable"
BASE_OUTPUT_DIR = "/opt/airflow/etl/agent_data"

TICKERS = [t.strip() for t in os.getenv('TRACKED_TICKERS', 'AAPL').split(',')]
FMP_RATE_LIMIT = int(os.getenv('FMP_RATE_LIMIT', '300'))
RATE_LIMIT_DELAY = 60.0 / FMP_RATE_LIMIT

AGENT_CONFIGS = {
    "business_analyst": {
        "endpoints": [
            ("company_profile",           "profile",               {"symbol": "{ticker}"},                                   "neo4j"),
            ("key_executives",            "key-executives",        {"symbol": "{ticker}"},                                   "neo4j"),
            ("stock_quote",               "quote",                 {"symbol": "{ticker}"},                                   "neo4j"),
            ("sec_filings_10k",           "sec_filings",           {"symbol": "{ticker}", "type": "10-K", "limit": 5},       "qdrant_prep"),
            ("sec_filings_10q",           "sec_filings",           {"symbol": "{ticker}", "type": "10-Q", "limit": 10},      "qdrant_prep"),
            ("sec_filings_8k",            "sec_filings",           {"symbol": "{ticker}", "type": "8-K",  "limit": 20},      "qdrant_prep"),
            ("earnings_call_transcripts", "earning_call_transcript",{"symbol": "{ticker}", "limit": 10},                     "qdrant_prep"),
            ("risk_factors",              "risk-factors",          {"symbol": "{ticker}"},                                   "neo4j"),
            ("company_notes",             "company-notes",         {"symbol": "{ticker}"},                                   "neo4j"),
            ("press_releases",            "press-releases",        {"symbol": "{ticker}", "limit": 50},                      "qdrant_prep"),
            ("stock_news",                "news/stock",            {"symbols": "{ticker}", "limit": 100},                    "qdrant_prep"),
        ]
    },
    "quantitative_fundamental": {
        "endpoints": [
            ("income_statement",              "income-statement",               {"symbol": "{ticker}", "limit": 40}, "postgresql"),
            ("balance_sheet",                 "balance-sheet-statement",        {"symbol": "{ticker}", "limit": 40}, "postgresql"),
            ("cash_flow",                     "cash-flow-statement",            {"symbol": "{ticker}", "limit": 40}, "postgresql"),
            ("income_statement_as_reported",  "income-statement-as-reported",   {"symbol": "{ticker}", "limit": 10}, "postgresql"),
            ("balance_sheet_as_reported",     "balance-sheet-as-reported",      {"symbol": "{ticker}", "limit": 10}, "postgresql"),
            ("cash_flow_as_reported",         "cash-flow-statement-as-reported",{"symbol": "{ticker}", "limit": 10}, "postgresql"),
            ("financial_ratios",              "ratios",                         {"symbol": "{ticker}", "limit": 40}, "postgresql"),
            ("ratios_ttm",                    "ratios-ttm",                     {"symbol": "{ticker}"},              "postgresql"),
            ("key_metrics",                   "key-metrics",                    {"symbol": "{ticker}", "limit": 40}, "postgresql"),
            ("key_metrics_ttm",               "key-metrics-ttm",                {"symbol": "{ticker}"},              "postgresql"),
            ("financial_growth",              "financial-growth",               {"symbol": "{ticker}", "limit": 40}, "postgresql"),
            ("enterprise_values",             "enterprise-values",              {"symbol": "{ticker}", "limit": 40}, "postgresql"),
            ("financial_scores",              "financial-scores",               {"symbol": "{ticker}"},              "postgresql"),
            ("shares_float",                  "shares_float",                   {"symbol": "{ticker}"},              "postgresql"),
            ("historical_market_cap",         "historical-market-capitalization",{"symbol": "{ticker}", "limit": 365},"postgresql"),
            ("company_core_info",             "company-core-information",       {"symbol": "{ticker}"},              "postgresql"),
            ("rating",                        "rating",                         {"symbol": "{ticker}"},              "postgresql"),
        ]
    },
    "financial_modeling": {
        "endpoints": [
            ("income_statement",                   "income-statement",                                     {"symbol": "{ticker}", "limit": 40},                                                                                           "postgresql"),
            ("balance_sheet",                      "balance-sheet-statement",                              {"symbol": "{ticker}", "limit": 40},                                                                                           "postgresql"),
            ("cash_flow",                          "cash-flow-statement",                                  {"symbol": "{ticker}", "limit": 40},                                                                                           "postgresql"),
            ("dcf",                                "dcf",                                                  {"symbol": "{ticker}"},                                                                                                        "postgresql"),
            ("advanced_dcf",                       "advanced_dcf",                                         {"symbol": "{ticker}"},                                                                                                        "postgresql"),
            ("levered_dcf",                        "levered_dcf",                                          {"symbol": "{ticker}"},                                                                                                        "postgresql"),
            ("owner_earnings",                     "owner-earnings",                                       {"symbol": "{ticker}", "limit": 40},                                                                                           "postgresql"),
            ("revenue_product_segmentation",       "revenue-product-segmentation",                         {"symbol": "{ticker}"},                                                                                                        "postgresql"),
            ("revenue_geographic_segmentation",    "revenue-geographic-segmentation",                      {"symbol": "{ticker}"},                                                                                                        "postgresql"),
            ("analyst_estimates",                  "analyst-estimates",                                    {"symbol": "{ticker}"},                                                                                                        "postgresql"),
            ("analyst_estimates_eps",              "analyst-estimates-eps",                                {"symbol": "{ticker}"},                                                                                                        "postgresql"),
            ("analyst_estimates_revenue",          "analyst-estimates-revenue",                            {"symbol": "{ticker}"},                                                                                                        "postgresql"),
            ("price_target",                       "price-target",                                         {"symbol": "{ticker}"},                                                                                                        "postgresql"),
            ("price_target_consensus",             "price-target-consensus",                               {"symbol": "{ticker}"},                                                                                                        "postgresql"),
            ("historical_dividends",               "historical-price-full/stock_dividend/{ticker}",        {},                                                                                                                            "postgresql"),
            ("stock_splits",                       "historical-price-full/stock_split/{ticker}",           {},                                                                                                                            "postgresql"),
            ("treasury_rates",                     "treasury",                                             {"from": (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), "to": datetime.now().strftime('%Y-%m-%d')},              "postgresql"),
            ("economic_indicators_gdp",            "economic",                                             {"name": "GDP"},                                                                                                               "postgresql"),
            ("economic_indicators_cpi",            "economic",                                             {"name": "CPI"},                                                                                                               "postgresql"),
            ("economic_indicators_inflation",      "economic",                                             {"name": "inflationRate"},                                                                                                     "postgresql"),
            ("stock_peers",                        "stock_peers",                                          {"symbol": "{ticker}"},                                                                                                        "postgresql"),
            ("historical_sectors_performance",     "sectors-performance",                                  {},                                                                                                                            "postgresql"),
            ("company_notes",                      "company-notes",                                        {"symbol": "{ticker}"},                                                                                                        "postgresql"),
            ("company_outlook",                    "company-outlook",                                      {"symbol": "{ticker}"},                                                                                                        "postgresql"),
            ("enterprise_values",                  "enterprise-values",                                    {"symbol": "{ticker}", "limit": 40},                                                                                           "postgresql"),
            ("market_cap_history",                 "historical-market-capitalization",                     {"symbol": "{ticker}", "limit": 365},                                                                                          "postgresql"),
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
            if isinstance(data, dict) and 'Error Message' in data:
                print(f"  API Error: {data['Error Message'][:100]}")
                return None
            if isinstance(data, dict) and 'error' in data:
                print(f"  API Error: {data['error'][:100]}")
                return None
            if isinstance(data, list) and len(data) == 0:
                print(f"  ⊘ Empty response")
                return None
            return data
        elif response.status_code == 403:
            print(f"  403 Forbidden — Premium endpoint or invalid key")
            return None
        elif response.status_code == 404:
            print(f"  404 Not Found")
            return None
        else:
            print(f"  Error {response.status_code}: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"  Exception: {e}")
        return None


def save_data(agent_name, ticker, data_name, data, metadata, storage_dest):
    if not data:
        print(f"  ⊘ Skipped (no data): {data_name}")
        return False

    agent_dir = Path(BASE_OUTPUT_DIR) / agent_name / ticker
    agent_dir.mkdir(parents=True, exist_ok=True)

    data_hash = get_data_hash(data)
    if metadata.get(data_name, {}).get('hash') == data_hash:
        print(f"  = Skipped (no changes): {data_name}")
        return False

    with open(agent_dir / f"{data_name}.json", 'w') as f:
        json.dump(data, f, indent=2)

    # FIX: write CSV for both list and dict responses
    try:
        if isinstance(data, list) and len(data) > 0:
            pd.DataFrame(data).to_csv(agent_dir / f"{data_name}.csv", index=False)
        elif isinstance(data, dict):
            # Unwrap single-item lists wrapped in a dict key (e.g. FMP profile endpoint)
            # Try to find a list value first
            list_val = next((v for v in data.values() if isinstance(v, list) and len(v) > 0), None)
            if list_val:
                pd.DataFrame(list_val).to_csv(agent_dir / f"{data_name}.csv", index=False)
            else:
                # Flatten top-level dict — drop nested dicts/lists
                flat_row = {k: str(v) for k, v in data.items() if not isinstance(v, (dict, list))}
                if flat_row:
                    pd.DataFrame([flat_row]).to_csv(agent_dir / f"{data_name}.csv", index=False)
    except Exception as e:
        print(f"  Warning: Could not save CSV for {data_name}: {e}")

    metadata[data_name] = {
        'hash': data_hash,
        'last_updated': datetime.now().isoformat(),
        'record_count': len(data) if isinstance(data, list) else 1,
        'source': 'fmp_stable',
        'storage_destination': storage_dest,
    }
    print(f"  ✓ Updated: {data_name} ({metadata[data_name]['record_count']} records) → {storage_dest}")
    return True


def scrape_agent_ticker(agent_name, ticker, **context):
    print(f"\n{'='*70}")
    print(f"[FMP] Agent: {agent_name} | Ticker: {ticker}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    config = AGENT_CONFIGS[agent_name]
    metadata = load_metadata(agent_name, ticker)
    updates_made = 0
    errors = 0

    for idx, (data_name, endpoint, params, storage_dest) in enumerate(config['endpoints'], 1):
        print(f"\n[{idx}/{len(config['endpoints'])}] Fetching: {data_name}...")
        endpoint_formatted = endpoint.format(ticker=ticker)
        params_copy = {
            k: v.format(ticker=ticker) if isinstance(v, str) else v
            for k, v in params.items()
        }
        try:
            data = fetch_data(endpoint_formatted, params_copy)
            if data and save_data(agent_name, ticker, data_name, data, metadata, storage_dest):
                updates_made += 1
            elif not data:
                errors += 1
        except Exception as e:
            print(f"  Failed: {e}")
            errors += 1
        time.sleep(RATE_LIMIT_DELAY)

    save_metadata(agent_name, ticker, metadata)
    print(f"\n{'─'*70}")
    print(f"[FMP] {agent_name}/{ticker}: {updates_made} updates, {errors} errors")
    print(f"{'─'*70}")

    context['task_instance'].xcom_push(
        key=f'{agent_name}_{ticker}_updates',
        value=updates_made
    )
    return updates_made


def report_summary(**context):
    ti = context['task_instance']
    summary = {}
    total_updates = 0

    for agent_name in AGENT_CONFIGS.keys():
        for ticker in TICKERS:
            task_id = f'fmp_scrape_{agent_name}_{ticker}'
            updates = ti.xcom_pull(task_ids=task_id, key=f'{agent_name}_{ticker}_updates')
            summary[f'{agent_name}/{ticker}'] = updates or 0
            total_updates += (updates or 0)

    print(f"\n{'='*70}")
    print(f"FMP INGESTION SUMMARY — Total updates: {total_updates}")
    print(f"{'='*70}")
    for k, v in summary.items():
        print(f"  {k}: {v} updates")
    print(f"{'='*70}")
    return summary


# ── DAG ──────────────────────────────────────────────────────────────────────
with DAG(
    'fmp_complete_ingestion',
    default_args=default_args,
    description='FMP Complete API ingestion - all agents, all tickers',
    schedule_interval='0 2 * * *',   # Daily 2am — after EODHD at 1am
    start_date=days_ago(1),
    catchup=False,
    tags=['fmp', 'complete', 'production'],
) as dag:

    scrape_tasks      = {}
    load_pg_tasks     = {}
    load_neo4j_tasks  = {}
    load_qdrant_tasks = {}

    for agent_name in AGENT_CONFIGS.keys():
        for ticker in TICKERS:
            key = f'{agent_name}_{ticker}'

            scrape_tasks[key] = PythonOperator(
                task_id=f'fmp_scrape_{agent_name}_{ticker}',
                python_callable=scrape_agent_ticker,
                op_kwargs={'agent_name': agent_name, 'ticker': ticker},
                provide_context=True,
            )
            load_pg_tasks[key] = PythonOperator(
                task_id=f'fmp_load_postgres_{agent_name}_{ticker}',
                python_callable=load_postgres_for_agent_ticker,
                op_kwargs={'agent_name': agent_name, 'ticker_symbol': ticker},
            )
            load_neo4j_tasks[key] = PythonOperator(
                task_id=f'fmp_load_neo4j_{agent_name}_{ticker}',
                python_callable=load_neo4j_for_agent_ticker,
                op_kwargs={'agent_name': agent_name, 'ticker_symbol': ticker},
            )
            load_qdrant_tasks[key] = PythonOperator(
                task_id=f'fmp_load_qdrant_{agent_name}_{ticker}',
                python_callable=load_qdrant_for_agent_ticker,
                op_kwargs={'agent_name': agent_name, 'ticker_symbol': ticker},
            )

    summary_task = PythonOperator(
        task_id='fmp_generate_summary',
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