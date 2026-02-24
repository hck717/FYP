#!/usr/bin/env python3
"""
FYP — Full System Health Check & Data Inspection

Checks:
  1. Local agent_data/ file system (JSON/CSV counts per agent/ticker)
  2. PostgreSQL — row counts per table, sample tickers
  3. Neo4j      — node counts per label, relationship counts, company list
  4. Qdrant     — collection info, vector count

Usage:
  python inspect_data.py          # full report
  python inspect_data.py --db     # DB checks only (skip file system)
  python inspect_data.py --files  # file system only (skip DB)

Clean ALL data (use with caution):
  rm -rf ingestion/etl/agent_data/* && \\
  docker exec fyp-neo4j cypher-shell -u neo4j -p changeme_neo4j_password "MATCH (n) DETACH DELETE n" && \\
  docker exec -i fyp-postgres psql -U airflow -d airflow -c "TRUNCATE raw_timeseries, raw_fundamentals, market_eod_us, global_economic_calendar, global_ipo_calendar;" && \\
  curl -s -X DELETE http://localhost:6333/collections/agentic_analyst_docs && \\
  echo '✅ All data cleaned!'
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── Optional DB drivers (graceful skip if not installed) ──────────────────────
try:
    import psycopg2
    HAS_PG = True
except ImportError:
    HAS_PG = False

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path("ingestion/etl/agent_data")
AGENTS   = ["business_analyst", "quantitative_fundamental", "financial_modeling"]

PG_HOST     = os.getenv("POSTGRES_HOST",     "localhost")
PG_PORT     = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB       = os.getenv("POSTGRES_DB",       "airflow")
PG_USER     = os.getenv("POSTGRES_USER",     "airflow")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme_neo4j_password")

QDRANT_HOST       = os.getenv("QDRANT_HOST",            "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT",        "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "agentic_analyst_docs")


# ── ANSI colours ──────────────────────────────────────────────────────────────
class C:
    BOLD   = '\033[1m'
    CYAN   = '\033[96m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    BLUE   = '\033[94m'
    RESET  = '\033[0m'

def ok(msg):   print(f"  {C.GREEN}✅ {msg}{C.RESET}")
def warn(msg): print(f"  {C.YELLOW}⚠️  {msg}{C.RESET}")
def err(msg):  print(f"  {C.RED}❌ {msg}{C.RESET}")
def info(msg): print(f"  {C.CYAN}   {msg}{C.RESET}")

def header(text, char='═'):
    print(f"\n{C.BOLD}{char*76}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {text}{C.RESET}")
    print(f"{C.BOLD}{char*76}{C.RESET}")

def section(text):
    print(f"\n{C.BOLD}{C.BLUE}── {text} {'─'*(70-len(text))}{C.RESET}")

def fmt_bytes(n):
    for unit in ['B','KB','MB','GB']:
        if n < 1024: return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ══════════════════════════════════════════════════════════════════════════════
# 1. FILE SYSTEM CHECK
# ══════════════════════════════════════════════════════════════════════════════
def check_files():
    header("FILE SYSTEM — ingestion/etl/agent_data/")

    if not BASE_DIR.exists():
        err(f"Base directory not found: {BASE_DIR}")
        return

    total_json = total_csv = total_size = total_endpoints = 0
    storage_stats  = defaultdict(lambda: {'count': 0, 'endpoints': set(), 'tickers': set()})
    source_stats   = defaultdict(int)
    ticker_dirs    = set()

    for agent in AGENTS:
        agent_dir = BASE_DIR / agent
        if agent_dir.exists():
            for td in agent_dir.iterdir():
                if td.is_dir():
                    ticker_dirs.add(td.name)

    all_tickers = sorted(ticker_dirs)

    for agent in AGENTS:
        agent_dir = BASE_DIR / agent
        if not agent_dir.exists():
            warn(f"Agent dir missing: {agent}")
            continue

        section(f"Agent: {agent}")

        for ticker in all_tickers:
            td = agent_dir / ticker
            if not td.exists():
                continue

            jsons   = [f for f in td.glob("*.json") if f.name != "metadata.json"]
            csvs    = list(td.glob("*.csv"))
            size    = sum(f.stat().st_size for f in td.iterdir() if f.is_file())
            meta    = json.loads((td/"metadata.json").read_text()) if (td/"metadata.json").exists() else {}

            total_json += len(jsons)
            total_csv  += len(csvs)
            total_size += size

            # Check CSV coverage — which json datasets are missing a CSV?
            json_names = {f.stem for f in jsons}
            csv_names  = {f.stem for f in csvs}
            missing_csv = json_names - csv_names

            print(f"\n  {C.BOLD}{C.GREEN}📦 {ticker}{C.RESET}  "
                  f"JSON:{len(jsons)}  CSV:{len(csvs)}  {fmt_bytes(size)}")

            if missing_csv:
                warn(f"No CSV for: {', '.join(sorted(missing_csv))}  ← neo4j/qdrant may skip these")

            for dname, dinfo in meta.items():
                dest    = dinfo.get('storage_destination', 'unknown')
                source  = dinfo.get('source', 'unknown')
                records = dinfo.get('record_count', 0)
                storage_stats[dest]['count'] += 1
                storage_stats[dest]['endpoints'].add(dname)
                storage_stats[dest]['tickers'].add(ticker)
                source_stats[source] += 1
                total_endpoints += 1

    section("Storage Destination Summary")
    dest_meta = {
        'neo4j':      ('🗄 ', C.GREEN,  'Graph DB'),
        'postgresql': ('📊', C.BLUE,   'PostgreSQL'),
        'qdrant_prep':('🔍', C.YELLOW, 'Qdrant Vector DB'),
        'unknown':    ('❓', C.RED,    'Unknown'),
    }
    for dest, (icon, color, label) in dest_meta.items():
        s = storage_stats.get(dest)
        if not s: continue
        print(f"  {color}{C.BOLD}{icon} {label:20}{C.RESET}  "
              f"datasets:{s['count']:>4}  "
              f"endpoints:{len(s['endpoints']):>3}  "
              f"tickers:{len(s['tickers']):>2}")

    section("Overall File Stats")
    info(f"Total JSON files    : {total_json}")
    info(f"Total CSV files     : {total_csv}")
    info(f"Total disk usage    : {fmt_bytes(total_size)}")
    info(f"Tickers tracked     : {len(all_tickers)} — {', '.join(all_tickers)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. POSTGRESQL HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════
def check_postgres():
    header("POSTGRESQL HEALTH CHECK")

    if not HAS_PG:
        warn("psycopg2 not installed — skipping. pip install psycopg2-binary")
        return

    try:
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT,
            dbname=PG_DB, user=PG_USER, password=PG_PASSWORD
        )
        ok(f"Connected to {PG_HOST}:{PG_PORT}/{PG_DB}")
    except Exception as e:
        err(f"Cannot connect to PostgreSQL: {e}")
        return

    tables = [
        "raw_timeseries",
        "raw_fundamentals",
        "market_eod_us",
        "global_economic_calendar",
        "global_ipo_calendar",
    ]

    section("Table Row Counts")
    with conn.cursor() as cur:
        for table in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                status = ok if count > 0 else warn
                status(f"{table:<35} {count:>10,} rows")
            except Exception as e:
                err(f"{table}: {e}")

    section("Tickers in raw_timeseries")
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ticker_symbol, COUNT(*) AS rows, MIN(ts_date), MAX(ts_date)
                FROM raw_timeseries
                GROUP BY ticker_symbol
                ORDER BY ticker_symbol
            """)
            rows = cur.fetchall()
            if rows:
                for r in rows:
                    info(f"{r[0]:<8} {r[1]:>10,} rows  {str(r[2])[:10]} → {str(r[3])[:10]}")
            else:
                warn("raw_timeseries is empty")
    except Exception as e:
        err(f"Query failed: {e}")

    section("Tickers in raw_fundamentals")
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ticker_symbol, COUNT(*) AS rows
                FROM raw_fundamentals
                GROUP BY ticker_symbol
                ORDER BY ticker_symbol
            """)
            rows = cur.fetchall()
            if rows:
                for r in rows:
                    info(f"{r[0]:<8} {r[1]:>10,} rows")
            else:
                warn("raw_fundamentals is empty")
    except Exception as e:
        err(f"Query failed: {e}")

    section("Latest Ingestion Times")
    try:
        with conn.cursor() as cur:
            for table in ["raw_timeseries", "raw_fundamentals"]:
                cur.execute(f"SELECT MAX(ingested_at) FROM {table}")
                ts = cur.fetchone()[0]
                if ts:
                    ok(f"{table}: last ingested at {ts}")
                else:
                    warn(f"{table}: no data yet")
    except Exception as e:
        err(f"Query failed: {e}")

    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# 3. NEO4J HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════
def check_neo4j():
    header("NEO4J HEALTH CHECK")

    if not HAS_NEO4J:
        warn("neo4j driver not installed — skipping. pip install neo4j")
        return

    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        ok(f"Connected to {NEO4J_URI}")
    except Exception as e:
        err(f"Cannot connect to Neo4j: {e}")
        return

    with driver.session() as session:

        section("Node Counts by Label")
        try:
            result = session.run("""
                CALL db.labels() YIELD label
                CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) AS cnt', {})
                YIELD value
                RETURN label, value.cnt AS count
                ORDER BY count DESC
            """)
            rows = result.data()
            if rows:
                for r in rows:
                    status = ok if r['count'] > 0 else warn
                    status(f":{r['label']:<20} {r['count']:>6} nodes")
            else:
                raise Exception("APOC not available")
        except Exception:
            # Fallback without APOC
            try:
                for label in ["Company", "Fact", "Risk", "Strategy"]:
                    result = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
                    count = result.single()["cnt"]
                    status = ok if count > 0 else warn
                    status(f":{label:<20} {count:>6} nodes")
            except Exception as e2:
                err(f"Node count query failed: {e2}")

        section("Relationship Counts")
        try:
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS rel, count(r) AS count
                ORDER BY count DESC
            """)
            rows = result.data()
            if rows:
                for r in rows:
                    ok(f"[:{r['rel']:<20}]  {r['count']:>6}")
            else:
                warn("No relationships found — nodes may be isolated")
        except Exception as e:
            err(f"Relationship query failed: {e}")

        section("Company Nodes")
        try:
            result = session.run("""
                MATCH (c:Company)
                RETURN c.ticker AS ticker,
                       c.companyName AS name,
                       c.sector AS sector,
                       c.exchange AS exchange
                ORDER BY c.ticker
            """)
            rows = result.data()
            if rows:
                ok(f"{len(rows)} Company node(s) found")
                for r in rows:
                    info(f"{r.get('ticker','?'):<8}  {str(r.get('name',''))[:35]:<35}  "
                         f"{str(r.get('sector',''))[:20]:<20}  {r.get('exchange','')}")
            else:
                err("No :Company nodes found — check save_data() CSV fix and re-run DAG")
        except Exception as e:
            err(f"Company query failed: {e}")

        section("Facts per Company (top 5)")
        try:
            result = session.run("""
                MATCH (f:Fact)-[:ABOUT]->(c:Company)
                RETURN c.ticker AS ticker, f.data_name AS dataset, count(f) AS cnt
                ORDER BY cnt DESC
                LIMIT 20
            """)
            rows = result.data()
            if rows:
                for r in rows:
                    info(f"{r.get('ticker','?'):<8}  {str(r.get('dataset','')):<30}  {r.get('cnt',0)} facts")
            else:
                warn("No :Fact nodes linked to companies yet")
        except Exception as e:
            err(f"Facts query failed: {e}")

        section("Risk & Strategy Nodes")
        try:
            for label, rel in [("Risk", "AFFECTS"), ("Strategy", "APPLIES_TO")]:
                result = session.run(f"""
                    MATCH (n:{label})-[:{rel}]->(c:Company)
                    RETURN c.ticker AS ticker, count(n) AS cnt
                    ORDER BY ticker
                """)
                rows = result.data()
                if rows:
                    for r in rows:
                        ok(f":{label} → {r['ticker']:<8}  {r['cnt']} nodes")
                else:
                    warn(f"No :{label} nodes found")
        except Exception as e:
            err(f"Risk/Strategy query failed: {e}")

    driver.close()


# ══════════════════════════════════════════════════════════════════════════════
# 4. QDRANT HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════
def check_qdrant():
    header("QDRANT HEALTH CHECK")

    if not HAS_REQUESTS:
        warn("requests not installed — skipping")
        return

    base_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

    # Connectivity
    try:
        resp = requests.get(f"{base_url}/readyz", timeout=5)
        if resp.status_code == 200:
            ok(f"Connected to {base_url}")
        else:
            err(f"Qdrant not ready: {resp.status_code}")
            return
    except Exception as e:
        err(f"Cannot connect to Qdrant: {e}")
        return

    # List all collections
    section("Collections")
    try:
        resp = requests.get(f"{base_url}/collections", timeout=5)
        collections = resp.json().get("result", {}).get("collections", [])
        if collections:
            ok(f"{len(collections)} collection(s) found")
            for c in collections:
                info(f"• {c['name']}")
        else:
            warn("No collections found — Qdrant is empty")
            return
    except Exception as e:
        err(f"Collections query failed: {e}")
        return

    # Target collection details
    section(f"Collection: {QDRANT_COLLECTION}")
    try:
        resp = requests.get(f"{base_url}/collections/{QDRANT_COLLECTION}", timeout=5)
        if resp.status_code == 404:
            err(f"Collection '{QDRANT_COLLECTION}' not found")
            return

        info_data = resp.json().get("result", {})
        vectors_count = info_data.get("vectors_count", 0)
        points_count  = info_data.get("points_count",  0)
        status        = info_data.get("status", "unknown")
        dim           = info_data.get("config", {}).get("params", {}).get("vectors", {}).get("size", "?")

        fn = ok if points_count > 0 else err
        fn(f"Status: {status}  |  Points: {points_count:,}  |  Vectors: {vectors_count:,}  |  Dim: {dim}")

        if points_count == 0:
            warn("Qdrant collection empty — Ollama must be running for embeddings")
            info("Check: curl http://localhost:11434/api/tags")

    except Exception as e:
        err(f"Collection detail query failed: {e}")

    # Sample payload breakdown
    section("Sample Point Payload")
    try:
        resp = requests.post(
            f"{base_url}/collections/{QDRANT_COLLECTION}/points/scroll",
            json={"limit": 3, "with_payload": True, "with_vector": False},
            timeout=10
        )
        points = resp.json().get("result", {}).get("points", [])
        if points:
            for p in points:
                payload = p.get("payload", {})
                info(f"id:{p['id']}  ticker:{payload.get('ticker_symbol','?')}  "
                     f"agent:{payload.get('agent_name','?')}  "
                     f"dataset:{payload.get('data_name','?')}")
        else:
            warn("No points returned from scroll")
    except Exception as e:
        err(f"Scroll query failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="FYP System Health Check")
    parser.add_argument("--db",    action="store_true", help="DB checks only")
    parser.add_argument("--files", action="store_true", help="File system check only")
    parser.add_argument("--neo4j", action="store_true", help="Neo4j check only")
    parser.add_argument("--pg",    action="store_true", help="PostgreSQL check only")
    parser.add_argument("--qdrant",action="store_true", help="Qdrant check only")
    args = parser.parse_args()

    run_all = not any([args.db, args.files, args.neo4j, args.pg, args.qdrant])

    header(f"FYP SYSTEM HEALTH CHECK  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if run_all or args.files:
        check_files()

    if run_all or args.db or args.pg:
        check_postgres()

    if run_all or args.db or args.neo4j:
        check_neo4j()

    if run_all or args.db or args.qdrant:
        check_qdrant()

    header("END OF HEALTH CHECK")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Interrupted{C.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{C.RED}FATAL: {e}{C.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
