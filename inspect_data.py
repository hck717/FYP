#!/usr/bin/env python3
"""
Quick data inspection script
Run after DAGs complete to verify data quality
"""

import json
import os
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("ingestion/etl/agent_data")

def inspect_agent_data():
    """Inspect data for all agents and tickers"""

    print("="*70)
    print("DATA INSPECTION REPORT")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)

    agents = ["business_analyst", "quantitative_fundamental", "financial_modeling"]

    for agent in agents:
        agent_dir = BASE_DIR / agent

        if not agent_dir.exists():
            print(f"\n❌ {agent}: Directory not found")
            continue

        tickers = [d.name for d in agent_dir.iterdir() if d.is_dir()]

        print(f"\n{'='*70}")
        print(f"AGENT: {agent.upper().replace('_', ' ')}")
        print(f"{'='*70}")
        print(f"Tickers tracked: {len(tickers)}")
        print(f"Tickers: {', '.join(tickers)}")

        for ticker in tickers:
            ticker_dir = agent_dir / ticker
            metadata_file = ticker_dir / "metadata.json"

            # Count files
            json_files = list(ticker_dir.glob("*.json"))
            csv_files = list(ticker_dir.glob("*.csv"))

            print(f"\n  📊 {ticker}:")
            print(f"    JSON files: {len(json_files)}")
            print(f"    CSV files:  {len(csv_files)}")

            # Check metadata
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)

                print(f"    Data sources:")
                for data_name, info in metadata.items():
                    source = info.get('source', 'unknown')
                    record_count = info.get('record_count', 0)
                    last_updated = info.get('last_updated', 'unknown')

                    # Parse timestamp
                    try:
                        dt = datetime.fromisoformat(last_updated)
                        time_str = dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        time_str = last_updated

                    print(f"      ✓ {data_name[:30]:<30} [{source:6}] {record_count:>4} records | {time_str}")
            else:
                print(f"    ⚠️  No metadata.json found")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Total file count
    total_json = len(list(BASE_DIR.glob("**/*.json")))
    total_csv = len(list(BASE_DIR.glob("**/*.csv")))
    total_size = sum(f.stat().st_size for f in BASE_DIR.glob("**/*") if f.is_file())

    print(f"Total JSON files: {total_json}")
    print(f"Total CSV files:  {total_csv}")
    print(f"Total disk usage: {total_size / (1024*1024):.2f} MB")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    if not BASE_DIR.exists():
        print(f"ERROR: {BASE_DIR} not found")
        print("Make sure you're running this from /Users/brianho/FYP/")
        exit(1)

    inspect_agent_data()
