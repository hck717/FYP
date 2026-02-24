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
  python inspect_data.py --neo4j  # Neo4j only
  python inspect_data.py --pg     # PostgreSQL only
  python inspect_data.py --qdrant # Qdrant only

Clean ALL data (use with caution):
  rm -rf ingestion/etl/agent_data/* && \\
  docker exec fyp-neo4j cypher-shell -u neo4j -p changeme_neo4j_password "MATCH (n) DETACH DELETE n" && \\
  docker exec -i fyp-postgres psql -U airflow -d airflow -c "TRUNCATE raw_timeseries, raw_fundamentals, market_eod_us, global_economic_calendar, global_ipo_calendar;" && \\
  curl -s -X DELETE http://localhost:6333/collections/financial_documents && \\
  echo '✅ All data cleaned!'
"""

import json
import os
import sys
import warnings
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Suppress neo4j GQL notification warnings (label-not-found etc.)
logging.getLogger("neo4j").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ── Optional DB drivers (graceful skip if not installed) ──────────────────────
try:
    import psycopg2
    HAS_PG = True
except ImportError:
    HAS_PG = False

try:
    from neo4j import GraphDatabase
    import neo4j.warnings as neo4j_warnings
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

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
# Leave blank to auto-detect from actual collections
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "")


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
    storage_stats = defaultdict(lambda: {'count': 0, 'endpoints': set(), 'tickers': set()})
    source_stats  = defaultdict(int)
    ticker_dirs   = set()

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

            jsons  = [f for f in td.glob("*.json") if f.name != "metadata.json"]
            csvs   = list(td.glob("*.csv"))
            size   = sum(f.stat().st_size for f in td.iterdir() if f.is_file())
            meta   = json.loads((td/"metadata.json").read_text()) if (td/"metadata.json").exists() else {}

            total_json += len(jsons)
            total_csv  += len(csvs)
            total_size += size

            json_names  = {f.stem for f in jsons}
            csv_names   = {f.stem for f in csvs}
            missing_csv = json_names - csv_names

            # Only flag missing CSVs that are tagged for neo4j (others are OK to skip)
            neo4j_missing = [
                n for n in missing_csv
                if meta.get(n, {}).get('storage_destination') == 'neo4j'
            ]

            print(f"\n  {C.BOLD}{C.GREEN}📦 {ticker}{C.RESET}  "
                  f"JSON:{len(jsons)}  CSV:{len(csvs)}  {fmt_bytes(size)}")

            if neo4j_missing:
                err(f"Neo4j CSV MISSING: {', '.join(sorted(neo4j_missing))}  ← will NOT load to graph")
            elif missing_csv:
                warn(f"No CSV (non-critical): {', '.join(sorted(missing_csv))}")

            for dname, dinfo in meta.items():
                dest   = dinfo.get('storage_destination', 'unknown')
                source = dinfo.get('source', 'unknown')
                storage_stats[dest]['count'] += 1
                storage_stats[dest]['endpoints'].add(dname)
                storage_stats[dest]['tickers'].add(ticker)
                source_stats[source] += 1
                total_endpoints += 1

    section("Storage Destination Summary")
    dest_meta = {
        'neo4j':       ('🗄 ', C.GREEN,  'Graph DB'),
        'postgresql':  ('📊', C.BLUE,   'PostgreSQL'),
        'qdrant_prep': ('🔍', C.YELLOW, 'Qdrant Vector DB'),
        'unknown':     ('❓', C.RED,    'Unknown'),
    }
    for dest, (icon, color, label) in dest_meta.items():
        s = storage_stats.get(dest)
        if not s:
            continue
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
            dbname=PG_DB, user=PG_USER, password=PG_PASSWORD,
            connect_timeout=5
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
                (ok if count > 0 else warn)(f"{table:<35} {count:>10,} rows")
            except Exception as e:
                err(f"{table}: {e}")
                conn.rollback()

    section("Tickers in raw_timeseries")
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ticker_symbol, COUNT(*) AS rows, MIN(ts_date), MAX(ts_date)
                FROM raw_timeseries
                GROUP BY ticker_symbol ORDER BY ticker_symbol
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
                GROUP BY ticker_symbol ORDER BY ticker_symbol
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
                (ok if ts else warn)(f"{table}: {ts if ts else 'no data yet'}")
    except Exception as e:
        err(f"Query failed: {e}")

    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# 3. NEO4J HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════
def _neo4j_count(session, label):
    """Return node count for a label, suppressing not-found warnings."""
    try:
        result = session.run(f"MATCH (n:`{label}`) RETURN count(n) AS cnt")
        return result.single()["cnt"]
    except Exception:
        return 0


def check_neo4j():
    header("NEO4J HEALTH CHECK")

    if not HAS_NEO4J:
        warn("neo4j driver not installed — skipping. pip install neo4j")
        return

    try:
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            notifications_min_severity="OFF",   # silence GQL label-not-found etc.
        )
        driver.verify_connectivity()
        ok(f"Connected to {NEO4J_URI}")
    except TypeError:
        # older neo4j driver doesn't support notifications_min_severity
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            driver.verify_connectivity()
            ok(f"Connected to {NEO4J_URI}")
        except Exception as e:
            err(f"Cannot connect to Neo4j: {e}")
            return
    except Exception as e:
        err(f"Cannot connect to Neo4j: {e}")
        return

    with driver.session() as session:

        # ── Node Counts ──────────────────────────────────────────────────────
        section("Node Counts by Label")

        # Get all labels actually in the DB first
        try:
            labels_result = session.run("CALL db.labels() YIELD label RETURN label")
            db_labels = [r["label"] for r in labels_result]
        except Exception:
            db_labels = []

        # Always check these expected labels
        expected = ["Company", "Fact", "Risk", "Strategy"]
        all_labels = list(dict.fromkeys(expected + db_labels))  # expected first, no dups

        for label in all_labels:
            count = _neo4j_count(session, label)
            (ok if count > 0 else warn)(f":{label:<20} {count:>6} nodes")

        # ── Relationship Counts ──────────────────────────────────────────────
        section("Relationship Counts")
        try:
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS rel, count(r) AS cnt
                ORDER BY cnt DESC
            """)
            rows = result.data()
            if rows:
                for r in rows:
                    ok(f"[:{r['rel']:<25}]  {r['cnt']:>6}")
            else:
                warn("No relationships found — nodes may be isolated")
        except Exception as e:
            err(f"Relationship query failed: {e}")

        # ── Company Nodes — dynamically read all properties ──────────────────
        section("Company Nodes")
        try:
            # Fetch all properties dynamically — works regardless of source (FMP vs EODHD)
            result = session.run("""
                MATCH (c:Company)
                RETURN c
                ORDER BY c.ticker
            """)
            rows = result.data()
            if rows:
                ok(f"{len(rows)} Company node(s) found")
                # Determine display fields — try common names from both FMP & EODHD
                for r in rows:
                    node = r['c']
                    ticker   = node.get('ticker', '?')
                    # FMP: companyName / EODHD: Name
                    name     = node.get('companyName') or node.get('Name') or node.get('name') or '—'
                    # FMP: sector / EODHD: Sector or GicSector
                    sector   = node.get('sector') or node.get('Sector') or node.get('GicSector') or '—'
                    # FMP: exchangeShortName / EODHD: Exchange
                    exchange = node.get('exchangeShortName') or node.get('Exchange') or node.get('exchange') or '—'
                    info(f"{ticker:<8}  {str(name)[:35]:<35}  {str(sector)[:22]:<22}  {exchange}")
            else:
                err("No :Company nodes found — CSV may be missing, re-run DAG")
        except Exception as e:
            err(f"Company query failed: {e}")

        # ── Facts per Company ────────────────────────────────────────────────
        section("Facts per Company")
        try:
            result = session.run("""
                MATCH (f:Fact)-[:ABOUT]->(c:Company)
                RETURN c.ticker AS ticker, f.data_name AS dataset, count(f) AS cnt
                ORDER BY cnt DESC LIMIT 20
            """)
            rows = result.data()
            if rows:
                for r in rows:
                    info(f"{r.get('ticker','?'):<8}  {str(r.get('dataset','')):<30}  {r.get('cnt',0)} facts")
            else:
                warn("No :Fact nodes linked to companies yet")
        except Exception as e:
            err(f"Facts query failed: {e}")

        # ── Risk & Strategy ──────────────────────────────────────────────────
        section("Risk & Strategy Nodes")
        for label, rel in [("Risk", "AFFECTS"), ("Strategy", "APPLIES_TO")]:
            count = _neo4j_count(session, label)
            if count == 0:
                warn(f"No :{label} nodes (0 total)")
                continue
            try:
                result = session.run(f"""
                    MATCH (n:`{label}`)-[:`{rel}`]->(c:Company)
                    RETURN c.ticker AS ticker, count(n) AS cnt
                    ORDER BY ticker
                """)
                rows = result.data()
                if rows:
                    for r in rows:
                        ok(f":{label} → {r['ticker']:<8}  {r['cnt']} nodes")
                else:
                    warn(f":{label} nodes exist but no [{rel}] relationship")
            except Exception as e:
                err(f":{label} query failed: {e}")

        # ── Raw property dump for first Company (debug aid) ──────────────────
        section("Sample Company Node Properties")
        try:
            result = session.run("MATCH (c:Company) RETURN c LIMIT 1")
            row = result.single()
            if row:
                node = row['c']
                info(f"Keys present: {', '.join(sorted(node.keys()))}")
        except Exception as e:
            err(f"Property dump failed: {e}")

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

    # ── List all collections ─────────────────────────────────────────────────
    section("Collections")
    collections = []
    try:
        resp = requests.get(f"{base_url}/collections", timeout=5)
        collections = resp.json().get("result", {}).get("collections", [])
        if collections:
            ok(f"{len(collections)} collection(s) found")
            for c in collections:
                info(f"• {c['name']}")
        else:
            warn("No collections — Qdrant is empty")
            return
    except Exception as e:
        err(f"Collections query failed: {e}")
        return

    # ── Auto-detect target collection if not set ─────────────────────────────
    target = QDRANT_COLLECTION
    if not target:
        # Prefer known names, else use first collection
        known = ["financial_documents", "agentic_analyst_docs", "documents"]
        names = [c['name'] for c in collections]
        target = next((n for n in known if n in names), names[0] if names else None)
        if target:
            info(f"Auto-detected collection: {target}")

    if not target:
        warn("No collection to inspect")
        return

    # ── Collection detail ────────────────────────────────────────────────────
    section(f"Collection: {target}")
    try:
        resp = requests.get(f"{base_url}/collections/{target}", timeout=5)
        if resp.status_code == 404:
            err(f"Collection '{target}' not found")
            return

        data          = resp.json().get("result", {})
        points_count  = data.get("points_count",  0)
        vectors_count = data.get("vectors_count", 0)
        status        = data.get("status", "unknown")
        dim           = (data.get("config", {})
                             .get("params", {})
                             .get("vectors", {})
                             .get("size", "?"))

        (ok if points_count > 0 else err)(
            f"Status: {status}  |  Points: {points_count:,}  "
            f"|  Vectors: {vectors_count:,}  |  Dim: {dim}"
        )

        if points_count == 0:
            warn("Collection empty — Ollama must be running to embed data")
            info("Check: curl http://localhost:11434/api/tags")

    except Exception as e:
        err(f"Collection detail failed: {e}")

    # ── Sample payloads ──────────────────────────────────────────────────────
    section("Sample Point Payloads")
    try:
        resp = requests.post(
            f"{base_url}/collections/{target}/points/scroll",
            json={"limit": 3, "with_payload": True, "with_vector": False},
            timeout=10
        )
        points = resp.json().get("result", {}).get("points", [])
        if points:
            for p in points:
                pl = p.get("payload", {})
                info(f"id:{p['id']}  ticker:{pl.get('ticker_symbol','?')}  "
                     f"agent:{pl.get('agent_name','?')}  "
                     f"dataset:{pl.get('data_name','?')}")
        else:
            warn("No points to sample")
    except Exception as e:
        err(f"Scroll failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="FYP System Health Check")
    parser.add_argument("--db",     action="store_true", help="DB checks only")
    parser.add_argument("--files",  action="store_true", help="File system check only")
    parser.add_argument("--neo4j",  action="store_true", help="Neo4j check only")
    parser.add_argument("--pg",     action="store_true", help="PostgreSQL check only")
    parser.add_argument("--qdrant", action="store_true", help="Qdrant check only")
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
