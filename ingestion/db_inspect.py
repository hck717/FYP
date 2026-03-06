#!/usr/bin/env python3
"""
db_inspect.py — FYP Database Content Inspector (data_needed.txt only)

New folder structure: agent_data/{ticker}/{data_name}.csv|json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

logging.getLogger("neo4j").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

try:
    import psycopg2
    import psycopg2.extras
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

PG_HOST     = "localhost"
PG_PORT     = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB       = os.getenv("POSTGRES_DB",       "airflow")
PG_USER     = os.getenv("POSTGRES_USER",     "airflow")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "SecureNeo4jPass2025!")

QDRANT_HOST = "localhost"
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Use local path by default, fall back to Airflow container path
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ETL_DIR = _REPO_ROOT / "ingestion" / "etl" / "agent_data"
BASE_ETL_DIR = Path(os.getenv("BASE_ETL_DIR", str(_DEFAULT_ETL_DIR)))
TRACKED_TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"]


class C:
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    WHITE  = "\033[97m"
    RESET  = "\033[0m"


def _banner(title: str) -> None:
    w = 80
    print(f"\n{C.BOLD}{C.CYAN}{'═' * w}{C.RESET}")
    pad = (w - len(title) - 2) // 2
    print(f"{C.BOLD}{C.CYAN}{'═' * pad}  {title}  {'═' * (w - pad - len(title) - 4)}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'═' * w}{C.RESET}")


def _section(title: str) -> None:
    print(f"\n{C.BOLD}{C.BLUE}  ┌─ {title} {C.DIM}{'─' * max(0, 70 - len(title))}{C.RESET}")


def _table_header(*cols: tuple) -> None:
    line = "  " + "  ".join(f"{C.BOLD}{C.CYAN}{h:<{w}}{C.RESET}" for h, w in cols)
    sep  = "  " + "  ".join("─" * w for _, w in cols)
    print(f"\n{line}")
    print(f"{C.DIM}{sep}{C.RESET}")


def _table_row(*cells: tuple) -> None:
    print("  " + "  ".join(f"{color}{str(text):<{w}}{C.RESET}" for text, w, color in cells))


def _ok(msg: str)   -> None: print(f"    {C.GREEN}✔  {msg}{C.RESET}")
def _warn(msg: str) -> None: print(f"    {C.YELLOW}⚠  {msg}{C.RESET}")
def _err(msg: str)  -> None: print(f"    {C.RED}✘  {msg}{C.RESET}")
def _info(msg: str) -> None: print(f"    {C.DIM}   {msg}{C.RESET}")


DATA_SPEC = [
    # (row, description, dest, data_names, scope)
    # scope: "per_ticker" = check BASE_ETL_DIR/{ticker}/
    #        "macro"      = check BASE_ETL_DIR/_MACRO/
    #        "pg_table"   = check PostgreSQL table row count > 0
    #
    # Rows align 1-to-1 with data_needed.txt (24 rows).
    {"row": 1,  "description": "Company Profiles / Tickers",                "dest": "neo4j",       "data_names": ["company_profile"],                                                              "scope": "per_ticker"},
    {"row": 2,  "description": "Financial News (Ticker-Tagged)",             "dest": "postgresql",  "data_names": ["financial_news"],                                                               "scope": "per_ticker"},
    {"row": 3,  "description": "Insider Transactions",                       "dest": "postgresql",  "data_names": ["insider_transactions"],                                                         "scope": "per_ticker"},
    {"row": 4,  "description": "Major Institutional Holders",                "dest": "postgresql",  "data_names": ["institutional_holders"],                                                        "scope": "per_ticker"},
    {"row": 5,  "description": "Historical Stock Prices (EOD)",              "dest": "postgresql",  "data_names": ["historical_prices_eod"],                                                        "scope": "per_ticker"},
    {"row": 6,  "description": "Intraday / Delayed Live Quotes",             "dest": "postgresql",  "data_names": ["realtime_quote", "live_stock_price", "intraday_1m"],                            "scope": "per_ticker"},
    {"row": 7,  "description": "Beta & Volatility / Technicals",             "dest": "postgresql",  "data_names": ["technical_rsi", "technical_macd", "technical_sma20", "technical_sma", "technical_ema", "technical_bbands", "technical_atr", "technical_stochrsi", "technical_adx", "technical_cci", "technical_slope", "technical_roc", "technical_stddev"],    "scope": "per_ticker"},
    {"row": 8,  "description": "Screener API (Bulk)",                        "dest": "postgresql",  "data_names": ["screener_bulk"],                                                                "scope": "macro"},
    {"row": 9,  "description": "Basic Fundamentals (EPS, key metrics)",      "dest": "postgresql",  "data_names": ["company_profile"],                                                              "scope": "per_ticker"},
    {"row": 10, "description": "Dividend History / Stock Splits",            "dest": "postgresql",  "data_names": ["dividends_history", "splits_history"],                                          "scope": "per_ticker"},
    {"row": 11, "description": "Treasury Rates / Macro Indicators",          "dest": "postgresql",  "data_names": ["treasury_rates", "treasury_rates_3m", "treasury_rates_6m", "treasury_rates_1y", "treasury_rates_2y", "treasury_rates_5y", "treasury_rates_20y", "treasury_rates_30y", "economic_indicators_gdp", "economic_indicators_cpi", "economic_indicators_unemployment"], "scope": "macro"},
    {"row": 12, "description": "Economic Events Data API",                   "dest": "postgresql",  "data_names": ["economic_events"],                                                              "scope": "macro"},
    {"row": 13, "description": "Bonds Data (Yields, Pricing)",               "dest": "postgresql",  "data_names": ["bond_aapl_fundamentals", "bond_amzn_fundamentals", "corporate_bond_yields"],   "scope": "macro"},
    {"row": 14, "description": "Forex Historical Rates (EOD)",               "dest": "postgresql",  "data_names": ["forex_historical_rates"],                                                       "scope": "macro"},
    {"row": 15, "description": "ETF & Index Constituent Holdings",           "dest": "neo4j",       "data_names": ["etf_index_constituents"],                                                       "scope": "macro"},
    {"row": 16, "description": "Financial Calendar",                         "dest": "postgresql",  "data_names": ["financial_calendar", "calendar_ipo", "calendar_splits", "calendar_dividends"],  "scope": "per_ticker"},
    {"row": 17, "description": "Real-Time News Feed",                        "dest": "postgresql",  "data_names": ["realtime_news_feed"],                                                           "scope": "per_ticker"},
    {"row": 18, "description": "Financial Statements (IS/BS/CF)",            "dest": "postgresql",  "data_names": ["financial_statements"],                                                         "scope": "per_ticker"},
    {"row": 19, "description": "Valuation Metrics",                          "dest": "postgresql",  "data_names": ["company_profile"],                                                              "scope": "per_ticker"},
    {"row": 20, "description": "Short Interest & Shares Stats",              "dest": "postgresql",  "data_names": ["short_interest"],                                                               "scope": "per_ticker"},
    {"row": 21, "description": "Earnings History & Surprises",               "dest": "postgresql",  "data_names": ["earnings_surprises"],                                                           "scope": "per_ticker"},
    {"row": 22, "description": "Outstanding Shares History",                 "dest": "postgresql",  "data_names": ["outstanding_shares"],                                                           "scope": "per_ticker"},
    # Row 23 (Sentiment Trends) — supplementary per data_needed.txt ordering
    {"row": 23, "description": "Sentiment Trends",                           "dest": "postgresql",  "data_names": ["sentiment_trends"],                                                             "scope": "per_ticker"},
    # Row 24 — Textual Documents (loaded separately via ingest_textual_metadata.py)
    {"row": 24, "description": "Textual Documents Metadata",                 "dest": "postgresql",  "data_names": ["textual_documents"],                                                            "scope": "pg_table"},
]


def _pg_connect():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        dbname=PG_DB, user=PG_USER, password=PG_PASSWORD,
        connect_timeout=5,
    )


def inspect_postgres(ticker_filter: str | None = None) -> None:
    _banner("POSTGRESQL")

    if not HAS_PG:
        _err("psycopg2 not installed")
        return

    try:
        conn = _pg_connect()
    except Exception as exc:
        _err(f"Cannot connect to PostgreSQL at {PG_HOST}:{PG_PORT}/{PG_DB}  →  {exc}")
        return

    _ok(f"Connected  {PG_HOST}:{PG_PORT}/{PG_DB}")

    with conn.cursor() as cur:
        _section("Application Tables")
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        all_tables = [r[0] for r in cur.fetchall()]

        _table_header(("Table", 35), ("Rows", 10))
        for tname in all_tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {tname}")
                row_count = cur.fetchone()[0]
                color = C.GREEN if row_count > 0 else C.YELLOW
                _table_row((tname, 35, C.WHITE), (f"{row_count:,}", 10, color))
            except Exception:
                pass

    conn.close()


def inspect_neo4j(ticker_filter: str | None = None) -> None:
    _banner("NEO4J")

    if not HAS_NEO4J:
        _err("neo4j driver not installed")
        return

    try:
        driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD),
            notifications_min_severity="OFF",
        )
        driver.verify_connectivity()
        _ok(f"Connected  {NEO4J_URI}")
    except Exception as exc:
        _err(f"Cannot connect to Neo4j at {NEO4J_URI}  →  {exc}")
        return

    with driver.session() as s:
        _section("Node Counts by Label")
        try:
            labels = [r["label"] for r in s.run("CALL db.labels() YIELD label RETURN label")]
        except Exception:
            labels = []

        for lbl in ["Company", "Chunk", "Fact"]:
            if lbl not in labels:
                labels.insert(0, lbl)
        labels = list(dict.fromkeys(labels))

        _table_header(("Label", 20), ("Nodes", 10))
        for lbl in labels:
            try:
                cnt = s.run(f"MATCH (n:`{lbl}`) RETURN count(n) AS c").single()["c"]
            except Exception:
                cnt = 0
            color = C.GREEN if cnt > 0 else C.YELLOW
            _table_row((lbl, 20, C.WHITE), (f"{cnt:,}", 10, color))

        _section("Company Nodes")
        try:
            q = "MATCH (c:Company) RETURN c.ticker AS t, c.companyName AS n ORDER BY c.ticker"
            companies = list(s.run(q))
            if not companies:
                _warn("No :Company nodes found")
            else:
                _table_header(("Ticker", 8), ("Name", 40))
                for rec in companies:
                    ticker = rec["t"] or "?"
                    name = (rec["n"] or "—")[:39]
                    _table_row((ticker, 8, C.CYAN), (name, 40, C.WHITE))
        except Exception as exc:
            _err(f"Company query failed: {exc}")

    driver.close()


def inspect_files(ticker_filter: str | None = None) -> None:
    _banner("LOCAL FILES")
    _ok(f"Base directory: {BASE_ETL_DIR}")

    tickers = [ticker_filter] if ticker_filter else TRACKED_TICKERS

    _section("Data Files by Ticker")
    _table_header(("Ticker", 10), ("Files", 10), ("Sample data_names", 50))

    for ticker in tickers:
        ticker_dir = BASE_ETL_DIR / ticker
        if not ticker_dir.exists():
            _table_row((ticker, 10, C.YELLOW), ("missing", 10, C.RED), ("", 50, C.DIM))
            continue

        files = list(ticker_dir.glob("*.csv"))
        files += list(ticker_dir.glob("*.json"))
        files = [f for f in files if f.name != "metadata.json"]

        if files:
            sample_dns = ", ".join([f.stem for f in files[:3]])
            _table_row((ticker, 10, C.CYAN), (f"{len(files)}", 10, C.GREEN), (sample_dns[:49], 50, C.DIM))
        else:
            _table_row((ticker, 10, C.CYAN), ("0", 10, C.YELLOW), ("no data files", 50, C.DIM))


def inspect_coverage(ticker_filter: str | None = None) -> None:
    _banner("DATA COVERAGE — data_needed.txt")

    tickers = [ticker_filter] if ticker_filter else TRACKED_TICKERS
    macro_dir = BASE_ETL_DIR / "_MACRO"

    # Pre-connect to PostgreSQL for pg_table scope checks
    pg_conn = None
    if HAS_PG:
        try:
            pg_conn = _pg_connect()
        except Exception:
            pass

    _section("Coverage by Data Type")
    _table_header(("Row", 4), ("Description", 40), ("Scope/Ticker", 12), ("Status", 12))

    for spec in DATA_SPEC:
        row_id     = spec["row"]
        desc       = spec["description"][:39]
        data_names = spec["data_names"]
        scope      = spec.get("scope", "per_ticker")

        if scope == "macro":
            # Check _MACRO/ folder for files
            found = []
            for dn in data_names:
                if (macro_dir / f"{dn}.csv").exists() or (macro_dir / f"{dn}.json").exists():
                    found.append(dn)
            label = "_MACRO"
            if found:
                _table_row((str(row_id), 4, C.DIM), (desc, 40, C.DIM), (label, 12, C.CYAN), (f"✔ {len(found)}", 12, C.GREEN))
            else:
                _table_row((str(row_id), 4, C.DIM), (desc, 40, C.DIM), (label, 12, C.CYAN), ("✘ missing", 12, C.RED))

        elif scope == "pg_table":
            # Check PostgreSQL table has rows
            table_name = data_names[0]
            label = "pg:" + table_name[:9]
            if pg_conn is None:
                _table_row((str(row_id), 4, C.DIM), (desc, 40, C.DIM), (label, 12, C.YELLOW), ("no pg conn", 12, C.RED))
            else:
                try:
                    with pg_conn.cursor() as cur:
                        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                        cnt = cur.fetchone()[0]
                    if cnt > 0:
                        _table_row((str(row_id), 4, C.DIM), (desc, 40, C.DIM), (label, 12, C.CYAN), (f"✔ {cnt:,}", 12, C.GREEN))
                    else:
                        _table_row((str(row_id), 4, C.DIM), (desc, 40, C.DIM), (label, 12, C.CYAN), ("✘ 0 rows", 12, C.RED))
                except Exception as exc:
                    _table_row((str(row_id), 4, C.DIM), (desc, 40, C.DIM), (label, 12, C.YELLOW), (f"err: {str(exc)[:8]}", 12, C.RED))

        else:
            # per_ticker: check each ticker's folder
            for ticker in tickers:
                ticker_dir = BASE_ETL_DIR / ticker
                if not ticker_dir.exists():
                    _table_row((str(row_id), 4, C.DIM), (desc, 40, C.DIM), (ticker, 12, C.YELLOW), ("no folder", 12, C.RED))
                    continue

                found = []
                for dn in data_names:
                    if (ticker_dir / f"{dn}.csv").exists() or (ticker_dir / f"{dn}.json").exists():
                        found.append(dn)

                if found:
                    _table_row((str(row_id), 4, C.DIM), (desc, 40, C.DIM), (ticker, 12, C.CYAN), (f"✔ {len(found)}", 12, C.GREEN))
                else:
                    _table_row((str(row_id), 4, C.DIM), (desc, 40, C.DIM), (ticker, 12, C.CYAN), ("✘ missing", 12, C.RED))

    if pg_conn:
        pg_conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="FYP Database Content Inspector — data_needed.txt only")
    parser.add_argument("--pg", action="store_true", help="PostgreSQL only")
    parser.add_argument("--neo4j", action="store_true", help="Neo4j only")
    parser.add_argument("--files", action="store_true", help="Local files only")
    parser.add_argument("--coverage", action="store_true", help="data_needed.txt coverage report")
    parser.add_argument("--ticker", default=None, metavar="TICKER", help="Filter to this ticker")
    args = parser.parse_args()

    if args.pg:
        inspect_postgres(args.ticker)
    elif args.neo4j:
        inspect_neo4j(args.ticker)
    elif args.files:
        inspect_files(args.ticker)
    elif args.coverage:
        inspect_coverage(args.ticker)
    else:
        inspect_files(args.ticker)
        inspect_postgres(args.ticker)
        inspect_neo4j(args.ticker)
        inspect_coverage(args.ticker)


if __name__ == "__main__":
    main()
