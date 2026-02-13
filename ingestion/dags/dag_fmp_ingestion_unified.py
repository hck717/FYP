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
FMP_API_KEY = os.getenv('FMP_API_KEY')
BASE_URL = "https://financialmodelingprep.com/stable"
BASE_OUTPUT_DIR = "/opt/airflow/etl/agent_data"

TICKERS = [t.strip() for t in os.getenv('TRACKED_TICKERS', 'AAPL').split(',')]
FMP_RATE_LIMIT = int(os.getenv('FMP_RATE_LIMIT', '300'))
RATE_LIMIT_DELAY = 60.0 / FMP_RATE_LIMIT

# COMPLETE AGENT CONFIGURATIONS - 100% Requirements Match
AGENT_CONFIGS = {
    "business_analyst": {
        "endpoints": [
            # Company Profiles (Neo4j entity nodes)
            ("company_profile", "profile", {"symbol": "{ticker}"}, "neo4j"),
            ("key_executives", "key-executives", {"symbol": "{ticker}"}, "neo4j"),
            ("stock_quote", "quote", {"symbol": "{ticker}"}, "neo4j"),

            # SEC Filings (Qdrant + Neo4j for CRAG)
            ("sec_filings_10k", "sec_filings", {"symbol": "{ticker}", "type": "10-K", "limit": 5}, "qdrant_prep"),
            ("sec_filings_10q", "sec_filings", {"symbol": "{ticker}", "type": "10-Q", "limit": 10}, "qdrant_prep"),
            ("sec_filings_8k", "sec_filings", {"symbol": "{ticker}", "type": "8-K", "limit": 20}, "qdrant_prep"),

            # Earnings Call Transcripts (Qdrant + Neo4j)
            ("earnings_call_transcripts", "earning_call_transcript", {"symbol": "{ticker}", "limit": 10}, "qdrant_prep"),

            # Risk Factors (Neo4j risk nodes)
            ("risk_factors", "risk-factors", {"symbol": "{ticker}"}, "neo4j"),

            # Business Strategy & MD&A (from 10-K/10-Q sections)
            ("company_notes", "company-notes", {"symbol": "{ticker}"}, "neo4j"),

            # Press releases (additional narrative source)
            ("press_releases", "press-releases", {"symbol": "{ticker}", "limit": 50}, "qdrant_prep"),

            # News for semantic search
            ("stock_news", "news/stock", {"symbols": "{ticker}", "limit": 100}, "qdrant_prep"),
        ]
    },

    "quantitative_fundamental": {
        "endpoints": [
            # Financial Statements (dual-path verification)
            ("income_statement", "income-statement", {"symbol": "{ticker}", "limit": 40}, "postgresql"),
            ("balance_sheet", "balance-sheet-statement", {"symbol": "{ticker}", "limit": 40}, "postgresql"),
            ("cash_flow", "cash-flow-statement", {"symbol": "{ticker}", "limit": 40}, "postgresql"),

            # As-reported financials (GAAP/IFRS)
            ("income_statement_as_reported", "income-statement-as-reported", {"symbol": "{ticker}", "limit": 10}, "postgresql"),
            ("balance_sheet_as_reported", "balance-sheet-as-reported", {"symbol": "{ticker}", "limit": 10}, "postgresql"),
            ("cash_flow_as_reported", "cash-flow-statement-as-reported", {"symbol": "{ticker}", "limit": 10}, "postgresql"),

            # Financial Ratios (dual-path)
            ("financial_ratios", "ratios", {"symbol": "{ticker}", "limit": 40}, "postgresql"),
            ("ratios_ttm", "ratios-ttm", {"symbol": "{ticker}"}, "postgresql"),

            # Key Metrics (dual-path)
            ("key_metrics", "key-metrics", {"symbol": "{ticker}", "limit": 40}, "postgresql"),
            ("key_metrics_ttm", "key-metrics-ttm", {"symbol": "{ticker}"}, "postgresql"),

            # Growth Metrics
            ("financial_growth", "financial-growth", {"symbol": "{ticker}", "limit": 40}, "postgresql"),

            # Enterprise Values
            ("enterprise_values", "enterprise-values", {"symbol": "{ticker}", "limit": 40}, "postgresql"),

            # Financial Scores (Piotroski, Beneish)
            ("financial_scores", "financial-scores", {"symbol": "{ticker}"}, "postgresql"),

            # Share Float Data
            ("shares_float", "shares_float", {"symbol": "{ticker}"}, "postgresql"),

            # Market Cap (historical via prices)
            ("historical_market_cap", "historical-market-capitalization", {"symbol": "{ticker}", "limit": 365}, "postgresql"),

            # Company core info
            ("company_core_info", "company-core-information", {"symbol": "{ticker}"}, "postgresql"),

            # Rating (includes Piotroski in detail)
            ("rating", "rating", {"symbol": "{ticker}"}, "postgresql"),
        ]
    },

    "financial_modeling": {
        "endpoints": [
            # Financial Statements (for DCF)
            ("income_statement", "income-statement", {"symbol": "{ticker}", "limit": 40}, "postgresql"),
            ("balance_sheet", "balance-sheet-statement", {"symbol": "{ticker}", "limit": 40}, "postgresql"),
            ("cash_flow", "cash-flow-statement", {"symbol": "{ticker}", "limit": 40}, "postgresql"),

            # DCF Valuation Models
            ("dcf", "dcf", {"symbol": "{ticker}"}, "postgresql"),
            ("advanced_dcf", "advanced_dcf", {"symbol": "{ticker}"}, "postgresql"),
            ("levered_dcf", "levered_dcf", {"symbol": "{ticker}"}, "postgresql"),

            # Owner Earnings
            ("owner_earnings", "owner-earnings", {"symbol": "{ticker}", "limit": 40}, "postgresql"),

            # Revenue Segmentation (critical for DCF)
            ("revenue_product_segmentation", "revenue-product-segmentation", {"symbol": "{ticker}"}, "postgresql"),
            ("revenue_geographic_segmentation", "revenue-geographic-segmentation", {"symbol": "{ticker}"}, "postgresql"),

            # Analyst Estimates (consensus tracking)
            ("analyst_estimates", "analyst-estimates", {"symbol": "{ticker}"}, "postgresql"),
            ("analyst_estimates_eps", "analyst-estimates-eps", {"symbol": "{ticker}"}, "postgresql"),
            ("analyst_estimates_revenue", "analyst-estimates-revenue", {"symbol": "{ticker}"}, "postgresql"),
            ("price_target", "price-target", {"symbol": "{ticker}"}, "postgresql"),
            ("price_target_consensus", "price-target-consensus", {"symbol": "{ticker}"}, "postgresql"),

            # Dividend History
            ("historical_dividends", "historical-price-full/stock_dividend/{ticker}", {}, "postgresql"),

            # Stock splits
            ("stock_splits", "historical-price-full/stock_split/{ticker}", {}, "postgresql"),

            # Treasury Rates (risk-free rate)
            ("treasury_rates", "treasury", {"from": (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), "to": datetime.now().strftime('%Y-%m-%d')}, "postgresql"),

            # Economic Indicators
            ("economic_indicators_gdp", "economic", {"name": "GDP"}, "postgresql"),
            ("economic_indicators_cpi", "economic", {"name": "CPI"}, "postgresql"),
            ("economic_indicators_inflation", "economic", {"name": "inflationRate"}, "postgresql"),

            # Peer Comparison
            ("stock_peers", "stock_peers", {"symbol": "{ticker}"}, "postgresql"),
            ("historical_sectors_performance", "sectors-performance", {}, "postgresql"),

            # Company notes and outlook
            ("company_notes", "company-notes", {"symbol": "{ticker}"}, "postgresql"),
            ("company_outlook", "company-outlook", {"symbol": "{ticker}"}, "postgresql"),

            # Valuation metrics
            ("enterprise_values", "enterprise-values", {"symbol": "{ticker}", "limit": 40}, "postgresql"),
            ("market_cap_history", "historical-market-capitalization", {"symbol": "{ticker}", "limit": 365}, "postgresql"),
        ]
    }
}

def get_data_hash(data):
    """Calculate MD5 hash for change detection"""
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()

def load_metadata(agent_name, ticker):
    """Load metadata for incremental updates"""
    metadata_path = Path(BASE_OUTPUT_DIR) / agent_name / ticker / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}

def save_metadata(agent_name, ticker, metadata):
    """Save metadata with storage destination info"""
    metadata_path = Path(BASE_OUTPUT_DIR) / agent_name / ticker / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def fetch_data(endpoint, params=None):
    """Fetch from FMP /stable/ API with error handling"""
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

            # Check for error messages
            if isinstance(data, dict) and 'Error Message' in data:
                error_msg = data['Error Message'][:100]
                print(f"  API Error: {error_msg}")
                # Check if it's a legacy endpoint error
                if "Legacy" in error_msg or "no longer supported" in error_msg:
                    print(f"  ⚠️  Legacy endpoint - skipping")
                return None
            if isinstance(data, dict) and 'error' in data:
                print(f"  API Error: {data['error'][:100]}")
                return None

            # Check for empty responses
            if isinstance(data, list) and len(data) == 0:
                print(f"  ⊘ Empty response (no data available)")
                return None

            return data
        elif response.status_code == 403:
            print(f"  403 Forbidden - Premium endpoint or invalid key")
            return None
        elif response.status_code == 404:
            print(f"  404 Not Found - Endpoint may not exist")
            return None
        else:
            print(f"  Error {response.status_code}: {response.text[:200]}")
            return None

    except Exception as e:
        print(f"  Exception: {e}")
        return None

def save_data(agent_name, ticker, data_name, data, metadata, storage_dest):
    """Save data with storage destination metadata"""
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

    # Update metadata
    metadata[data_name] = {
        'hash': data_hash,
        'last_updated': datetime.now().isoformat(),
        'record_count': len(data) if isinstance(data, list) else 1,
        'source': 'fmp_stable',
        'storage_destination': storage_dest
    }

    print(f"  ✓ Updated: {data_name} ({metadata[data_name]['record_count']} records) → {storage_dest}")
    return True

def scrape_agent_ticker(agent_name, ticker, **context):
    """Scrape data for specific agent and ticker"""
    print(f"\n{'='*70}")
    print(f"[FMP Complete] Agent: {agent_name} | Ticker: {ticker}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    config = AGENT_CONFIGS[agent_name]
    metadata = load_metadata(agent_name, ticker)

    updates_made = 0
    errors = 0
    premium_blocked = 0

    for idx, endpoint_config in enumerate(config['endpoints'], 1):
        data_name, endpoint, params, storage_dest = endpoint_config

        print(f"\n[{idx}/{len(config['endpoints'])}] Fetching: {data_name}...")

        # Replace {ticker} placeholder
        endpoint_formatted = endpoint.format(ticker=ticker)
        params_copy = {k: v.format(ticker=ticker) if isinstance(v, str) else v 
                      for k, v in params.items()}

        try:
            data = fetch_data(endpoint_formatted, params_copy)
            if data:
                if save_data(agent_name, ticker, data_name, data, metadata, storage_dest):
                    updates_made += 1
            else:
                # Check if it was 403 (premium)
                if "403" in str(data):
                    premium_blocked += 1
                errors += 1
        except Exception as e:
            print(f"  Failed: {e}")
            errors += 1

        time.sleep(RATE_LIMIT_DELAY)

    save_metadata(agent_name, ticker, metadata)

    print(f"\n{'─'*70}")
    print(f"[FMP Complete] {agent_name}/{ticker}:")
    print(f"  Updates: {updates_made}")
    print(f"  Errors: {errors}")
    print(f"  Premium blocked: {premium_blocked}")
    print(f"  Success rate: {updates_made}/{len(config['endpoints'])}")
    print(f"{'─'*70}")

    context['task_instance'].xcom_push(
        key=f'{agent_name}_{ticker}_updates',
        value=updates_made
    )

    return updates_made

def create_scrape_task(agent_name, ticker):
    """Factory function to create scraping tasks"""
    return PythonOperator(
        task_id=f'fmp_scrape_{agent_name}_{ticker}',
        python_callable=scrape_agent_ticker,
        op_kwargs={'agent_name': agent_name, 'ticker': ticker},
        provide_context=True,
    )

def report_summary(**context):
    """Generate comprehensive summary report"""
    ti = context['task_instance']

    summary = {}
    total_updates = 0
    storage_stats = {'neo4j': 0, 'postgresql': 0, 'qdrant_prep': 0}

    for agent_name in AGENT_CONFIGS.keys():
        for ticker in TICKERS:
            key = f'{agent_name}_{ticker}_updates'
            task_id = f'fmp_scrape_{agent_name}_{ticker}'
            updates = ti.xcom_pull(task_ids=task_id, key=key)
            summary[f'{agent_name}/{ticker}'] = updates or 0
            total_updates += (updates or 0)

            # Count storage destinations
            for endpoint_config in AGENT_CONFIGS[agent_name]['endpoints']:
                if updates and len(endpoint_config) >= 4:
                    storage_dest = endpoint_config[3]
                    storage_stats[storage_dest] = storage_stats.get(storage_dest, 0) + 1

    print(f"\n{'='*70}")
    print(f"FMP COMPLETE API INGESTION SUMMARY")
    print(f"{'='*70}")
    for key, value in summary.items():
        print(f"{key}: {value} files updated")
    print(f"{'='*70}")
    print(f"Total updates: {total_updates}")
    print(f"Tracked tickers: {', '.join(TICKERS)}")
    print(f"\nStorage Destinations:")
    print(f"  Neo4j:       {storage_stats.get('neo4j', 0)} endpoints")
    print(f"  PostgreSQL:  {storage_stats.get('postgresql', 0)} endpoints")
    print(f"  Qdrant prep: {storage_stats.get('qdrant_prep', 0)} endpoints")
    print(f"{'='*70}")

    return summary

# Define the DAG
with DAG(
    'fmp_complete_ingestion',
    default_args=default_args,
    description='FMP Complete API ingestion - 100% requirements match',
    schedule_interval='0 */6 * * *',  # Every 6 hours
    start_date=days_ago(1),
    catchup=False,
    tags=['fmp', 'complete', 'all-data', 'production'],
) as dag:

    scrape_tasks = {}

    for agent_name in AGENT_CONFIGS.keys():
        for ticker in TICKERS:
            task = create_scrape_task(agent_name, ticker)
            scrape_tasks[f'{agent_name}_{ticker}'] = task

    summary_task = PythonOperator(
        task_id='fmp_generate_summary',
        python_callable=report_summary,
        provide_context=True,
    )

    # Set dependencies
    for task in scrape_tasks.values():
        task >> summary_task