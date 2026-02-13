#!/usr/bin/env python3

# clean ALL DB before ingestion : 
# rm -rf ingestion/etl/agent_data/* && \ docker exec neo4j cypher-shell -u neo4j -p password "MATCH (n) DETACH DELETE n" 2>/dev/null && \ docker exec -i postgres psql -U postgres -d financial_db -c "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;" 2>/dev/null && \ curl -s http://localhost:6333/collections | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | xargs -I {} curl -s -X DELETE "http://localhost:6333/collections/{}" && \ echo '✅ All data cleaned!'



"""
Enhanced Data Inspection Script
Inspects ingested data and shows storage destination mapping
Tracks data readiness for Neo4j, PostgreSQL, and Qdrant ETL pipelines
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys

# Configuration
BASE_DIR = Path("ingestion/etl/agent_data")
AGENTS = ["business_analyst", "quantitative_fundamental", "financial_modeling"]

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def format_bytes(bytes_size):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def get_directory_size(path):
    """Calculate total size of directory"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_directory_size(entry.path)
    except Exception:
        pass
    return total

def load_metadata(agent_dir):
    """Load metadata.json from agent directory"""
    metadata_path = agent_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}

def get_ticker_stats(ticker_dir):
    """Get file counts and metadata for a ticker"""
    if not ticker_dir.exists():
        return None

    json_files = list(ticker_dir.glob("*.json"))
    csv_files = list(ticker_dir.glob("*.csv"))

    # Exclude metadata.json from count
    json_files = [f for f in json_files if f.name != "metadata.json"]

    metadata = load_metadata(ticker_dir)

    return {
        'json_count': len(json_files),
        'csv_count': len(csv_files),
        'metadata': metadata,
        'size': get_directory_size(ticker_dir)
    }

def print_header(text, char='='):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{char * 78}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{char * 78}{Colors.ENDC}")

def print_section(text, char='─'):
    """Print formatted section"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{char * 78}{Colors.ENDC}")
    print(f"{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{char * 78}{Colors.ENDC}")

def main():
    print_header(f"DATA INGESTION INSPECTION REPORT")
    print(f"{Colors.CYAN}Timestamp: {datetime.now().isoformat()}{Colors.ENDC}")
    print(f"{Colors.CYAN}Base Directory: {BASE_DIR}{Colors.ENDC}")

    if not BASE_DIR.exists():
        print(f"\n{Colors.RED}❌ ERROR: Base directory does not exist: {BASE_DIR}{Colors.ENDC}")
        sys.exit(1)

    # Global statistics
    total_json = 0
    total_csv = 0
    total_size = 0
    total_endpoints = 0

    # Storage destination tracking
    storage_stats = defaultdict(lambda: {'count': 0, 'endpoints': set(), 'tickers': set()})
    source_stats = defaultdict(int)

    # Get list of tickers
    ticker_dirs = set()
    for agent in AGENTS:
        agent_dir = BASE_DIR / agent
        if agent_dir.exists():
            for ticker_dir in agent_dir.iterdir():
                if ticker_dir.is_dir():
                    ticker_dirs.add(ticker_dir.name)

    all_tickers = sorted(ticker_dirs)

    # Inspect each agent
    for agent in AGENTS:
        agent_dir = BASE_DIR / agent

        if not agent_dir.exists():
            print(f"\n{Colors.YELLOW}⚠️  WARNING: Agent directory not found: {agent}{Colors.ENDC}")
            continue

        print_section(f"AGENT: {agent.upper().replace('_', ' ')}")

        print(f"\n{Colors.BOLD}Tickers tracked: {len(all_tickers)}{Colors.ENDC}")
        print(f"{Colors.CYAN}Tickers: {', '.join(all_tickers)}{Colors.ENDC}")

        # Inspect each ticker
        for ticker in all_tickers:
            ticker_dir = agent_dir / ticker
            stats = get_ticker_stats(ticker_dir)

            if not stats:
                print(f"\n  {Colors.YELLOW}📊 {ticker}: No data{Colors.ENDC}")
                continue

            print(f"\n  {Colors.BOLD}{Colors.GREEN}📊 {ticker}:{Colors.ENDC}")
            print(f"    {Colors.CYAN}JSON files: {stats['json_count']}{Colors.ENDC}")
            print(f"    {Colors.CYAN}CSV files:  {stats['csv_count']}{Colors.ENDC}")
            print(f"    {Colors.CYAN}Size:       {format_bytes(stats['size'])}{Colors.ENDC}")

            total_json += stats['json_count']
            total_csv += stats['csv_count']
            total_size += stats['size']

            # Parse metadata for detailed info
            metadata = stats['metadata']
            if metadata:
                print(f"    {Colors.BOLD}Data sources:{Colors.ENDC}")

                # Group by storage destination
                by_destination = defaultdict(list)

                for data_name, data_info in metadata.items():
                    source = data_info.get('source', 'unknown')
                    storage_dest = data_info.get('storage_destination', 'unknown')
                    record_count = data_info.get('record_count', 0)
                    last_updated = data_info.get('last_updated', 'N/A')

                    # Update global stats
                    source_stats[source] += 1
                    storage_stats[storage_dest]['count'] += 1
                    storage_stats[storage_dest]['endpoints'].add(data_name)
                    storage_stats[storage_dest]['tickers'].add(ticker)
                    total_endpoints += 1

                    # Format date
                    if last_updated != 'N/A':
                        try:
                            dt = datetime.fromisoformat(last_updated)
                            date_str = dt.strftime('%Y-%m-%d %H:%M')
                        except:
                            date_str = last_updated[:16]
                    else:
                        date_str = 'N/A'

                    by_destination[storage_dest].append({
                        'name': data_name,
                        'source': source,
                        'records': record_count,
                        'date': date_str
                    })

                # Print grouped by storage destination
                dest_order = ['neo4j', 'postgresql', 'qdrant_prep', 'unknown']
                for dest in dest_order:
                    if dest in by_destination:
                        items = by_destination[dest]

                        # Color code by destination
                        if dest == 'neo4j':
                            dest_color = Colors.GREEN
                            dest_icon = '🗄️'
                        elif dest == 'postgresql':
                            dest_color = Colors.BLUE
                            dest_icon = '📊'
                        elif dest == 'qdrant_prep':
                            dest_color = Colors.YELLOW
                            dest_icon = '🔍'
                        else:
                            dest_color = Colors.RED
                            dest_icon = '❓'

                        print(f"\n      {dest_color}{Colors.BOLD}→ {dest.upper()}{Colors.ENDC}")
                        for item in sorted(items, key=lambda x: x['name']):
                            source_tag = f"[{item['source']:^10}]"
                            name_short = item['name'][:30].ljust(30)
                            print(f"      {dest_icon} {Colors.CYAN}{name_short}{Colors.ENDC} "
                                  f"{source_tag} {item['records']:>6} records | {item['date']}")
            else:
                print(f"    {Colors.YELLOW}⚠️  No metadata found{Colors.ENDC}")

    # Storage Destination Summary
    print_section("STORAGE DESTINATION SUMMARY")

    for dest in ['neo4j', 'postgresql', 'qdrant_prep', 'unknown']:
        if dest in storage_stats:
            stats = storage_stats[dest]

            if dest == 'neo4j':
                icon = '🗄️'
                color = Colors.GREEN
                desc = "Graph Database"
            elif dest == 'postgresql':
                icon = '📊'
                color = Colors.BLUE
                desc = "Relational Database"
            elif dest == 'qdrant_prep':
                icon = '🔍'
                color = Colors.YELLOW
                desc = "Vector Database (Preparation)"
            else:
                icon = '❓'
                color = Colors.RED
                desc = "Unknown Destination"

            print(f"\n{color}{Colors.BOLD}{icon} {dest.upper()} - {desc}{Colors.ENDC}")
            print(f"  {Colors.CYAN}Total data points: {stats['count']}{Colors.ENDC}")
            print(f"  {Colors.CYAN}Unique endpoints:  {len(stats['endpoints'])}{Colors.ENDC}")
            print(f"  {Colors.CYAN}Tickers covered:   {len(stats['tickers'])}{Colors.ENDC}")

            # List endpoints
            if stats['endpoints']:
                print(f"  {Colors.BOLD}Endpoints:{Colors.ENDC}")
                for endpoint in sorted(stats['endpoints'])[:10]:  # Show first 10
                    print(f"    • {endpoint}")
                if len(stats['endpoints']) > 10:
                    print(f"    {Colors.YELLOW}... and {len(stats['endpoints']) - 10} more{Colors.ENDC}")

    # Data Source Summary
    print_section("DATA SOURCE SUMMARY")

    total_from_sources = sum(source_stats.values())
    for source, count in sorted(source_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_from_sources * 100) if total_from_sources > 0 else 0

        if source == 'fmp_stable':
            icon = '💰'
            color = Colors.GREEN
        elif source == 'eodhd':
            icon = '🌐'
            color = Colors.BLUE
        else:
            icon = '❓'
            color = Colors.YELLOW

        print(f"\n{color}{Colors.BOLD}{icon} {source.upper()}{Colors.ENDC}")
        print(f"  {Colors.CYAN}Data points: {count} ({percentage:.1f}%){Colors.ENDC}")

    # ETL Readiness Check
    print_section("ETL PIPELINE READINESS")

    # Check readiness for each database
    print(f"\n{Colors.BOLD}Neo4j ETL Pipeline:{Colors.ENDC}")
    neo4j_ready = storage_stats['neo4j']['count'] > 0
    if neo4j_ready:
        print(f"  {Colors.GREEN}✅ Ready - {storage_stats['neo4j']['count']} data points available{Colors.ENDC}")
        print(f"  {Colors.CYAN}   Covers {len(storage_stats['neo4j']['tickers'])} tickers{Colors.ENDC}")
        print(f"  {Colors.CYAN}   Next: Run etl/company_profiles_to_neo4j.py{Colors.ENDC}")
    else:
        print(f"  {Colors.RED}❌ Not Ready - No data available{Colors.ENDC}")

    print(f"\n{Colors.BOLD}PostgreSQL ETL Pipeline:{Colors.ENDC}")
    postgres_ready = storage_stats['postgresql']['count'] > 0
    if postgres_ready:
        print(f"  {Colors.GREEN}✅ Ready - {storage_stats['postgresql']['count']} data points available{Colors.ENDC}")
        print(f"  {Colors.CYAN}   Covers {len(storage_stats['postgresql']['tickers'])} tickers{Colors.ENDC}")
        print(f"  {Colors.CYAN}   Next: Run etl/financial_data_to_postgres.py{Colors.ENDC}")
    else:
        print(f"  {Colors.RED}❌ Not Ready - No data available{Colors.ENDC}")

    print(f"\n{Colors.BOLD}Qdrant ETL Pipeline:{Colors.ENDC}")
    qdrant_ready = storage_stats['qdrant_prep']['count'] > 0
    if qdrant_ready:
        print(f"  {Colors.GREEN}✅ Ready - {storage_stats['qdrant_prep']['count']} data points available{Colors.ENDC}")
        print(f"  {Colors.CYAN}   Covers {len(storage_stats['qdrant_prep']['tickers'])} tickers{Colors.ENDC}")
        print(f"  {Colors.CYAN}   Next: Run etl/news_to_qdrant.py (embeddings){Colors.ENDC}")
    else:
        print(f"  {Colors.YELLOW}⚠️  Limited Data - Only {storage_stats['qdrant_prep']['count']} points{Colors.ENDC}")
        print(f"  {Colors.CYAN}   Note: SEC filings (10-K, 10-Q) not yet available from FMP{Colors.ENDC}")

    # Overall Summary
    print_section("OVERALL SUMMARY")

    print(f"\n{Colors.BOLD}Data Files:{Colors.ENDC}")
    print(f"  {Colors.CYAN}Total JSON files: {total_json}{Colors.ENDC}")
    print(f"  {Colors.CYAN}Total CSV files:  {total_csv}{Colors.ENDC}")
    print(f"  {Colors.CYAN}Total endpoints:  {total_endpoints}{Colors.ENDC}")
    print(f"  {Colors.CYAN}Total disk usage: {format_bytes(total_size)}{Colors.ENDC}")

    print(f"\n{Colors.BOLD}Coverage:{Colors.ENDC}")
    print(f"  {Colors.CYAN}Tickers tracked:  {len(all_tickers)}{Colors.ENDC}")
    print(f"  {Colors.CYAN}Agents active:    {len([a for a in AGENTS if (BASE_DIR / a).exists()])}/{len(AGENTS)}{Colors.ENDC}")

    # Health indicators
    print(f"\n{Colors.BOLD}System Health:{Colors.ENDC}")

    health_score = 0
    max_score = 4

    if total_endpoints > 0:
        print(f"  {Colors.GREEN}✅ Data ingestion working{Colors.ENDC}")
        health_score += 1
    else:
        print(f"  {Colors.RED}❌ No data ingested{Colors.ENDC}")

    if len(all_tickers) >= 5:
        print(f"  {Colors.GREEN}✅ Multiple tickers tracked ({len(all_tickers)}){Colors.ENDC}")
        health_score += 1
    else:
        print(f"  {Colors.YELLOW}⚠️  Limited ticker coverage ({len(all_tickers)}){Colors.ENDC}")

    if storage_stats['neo4j']['count'] > 0 and storage_stats['postgresql']['count'] > 0:
        print(f"  {Colors.GREEN}✅ All storage destinations have data{Colors.ENDC}")
        health_score += 1
    else:
        print(f"  {Colors.YELLOW}⚠️  Some storage destinations empty{Colors.ENDC}")

    if len(source_stats) >= 2:
        print(f"  {Colors.GREEN}✅ Multiple data sources active{Colors.ENDC}")
        health_score += 1
    else:
        print(f"  {Colors.YELLOW}⚠️  Single data source{Colors.ENDC}")

    health_percentage = (health_score / max_score) * 100

    print(f"\n{Colors.BOLD}Overall Health: ", end="")
    if health_percentage >= 75:
        print(f"{Colors.GREEN}{health_percentage:.0f}% - Excellent{Colors.ENDC}")
    elif health_percentage >= 50:
        print(f"{Colors.YELLOW}{health_percentage:.0f}% - Good{Colors.ENDC}")
    else:
        print(f"{Colors.RED}{health_percentage:.0f}% - Needs Attention{Colors.ENDC}")

    print_header("END OF REPORT")

    # Return exit code based on health
    return 0 if health_score >= 2 else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Colors.RED}ERROR: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)