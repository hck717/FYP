#!/usr/bin/env python3
"""
db_inspect.py — FYP Database Content Inspector

Prints a clean, table-formatted summary of every database used by the system,
cross-referenced against the full data_needed.txt single-source spec (19 rows).

Databases inspected:
  • PostgreSQL  — per-agent/ticker coverage of every expected data_name
  • Neo4j       — node/relationship counts, Company nodes, graph structure
  • Qdrant      — collection stats, per-ticker vector counts, data_name breakdown

Ingestion coverage is checked against the two DAGs:
  EODHD DAG  (eodhd_complete_ingestion)  — company profiles, prices, technicals,
             intraday, dividends, splits, macro indicators, sentiment, news
  FMP DAG    (fmp_complete_ingestion, currently PAUSED) — SEC filings, transcripts,
             financials, ratios, scores, estimates, DCF inputs, peers

Usage
-----
    python ingestion/db_inspect.py                   # all three databases
    python ingestion/db_inspect.py --pg              # PostgreSQL only
    python ingestion/db_inspect.py --neo4j           # Neo4j only
    python ingestion/db_inspect.py --qdrant          # Qdrant only
    python ingestion/db_inspect.py --coverage        # data_needed.txt coverage report
    python ingestion/db_inspect.py --ticker AAPL     # filter per-ticker views
    python ingestion/db_inspect.py --pg --ticker TSLA
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Load .env from repo root so host-side credentials are picked up automatically
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# Silence driver-level noise
logging.getLogger("neo4j").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ── Optional DB drivers ───────────────────────────────────────────────────────
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

# ── Connection config (host-side — always localhost) ─────────────────────────
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

TRACKED_TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"]


# ── ANSI colours ──────────────────────────────────────────────────────────────
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


# ── Print helpers ─────────────────────────────────────────────────────────────
def _banner(title: str) -> None:
    w = 80
    print(f"\n{C.BOLD}{C.CYAN}{'═' * w}{C.RESET}")
    pad = (w - len(title) - 2) // 2
    print(f"{C.BOLD}{C.CYAN}{'═' * pad}  {title}  {'═' * (w - pad - len(title) - 4)}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'═' * w}{C.RESET}")


def _section(title: str) -> None:
    print(f"\n{C.BOLD}{C.BLUE}  ┌─ {title} {C.DIM}{'─' * max(0, 70 - len(title))}{C.RESET}")


def _row(label: str, value: str, indent: int = 4, color: str = C.WHITE) -> None:
    print(f"{' ' * indent}{C.DIM}│{C.RESET}  {C.BOLD}{label:<28}{C.RESET}  {color}{value}{C.RESET}")


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


# ══════════════════════════════════════════════════════════════════════════════
# Data-needed spec — ground truth for coverage checks
# Derived directly from data_needed.txt (19 rows) + both DAG AGENT_CONFIGS
# ══════════════════════════════════════════════════════════════════════════════

# Each entry: (agent, description, source, destination, [data_name, ...])
# data_names are the actual keys used in metadata.json / DB queries.
DATA_SPEC: list[dict] = [
    # ── Business Analyst (EODHD) ──────────────────────────────────────────────
    {
        "row": 4,  "agent": "Business Analyst",
        "description": "Company Profiles / Tickers",
        "source": "EODHD", "dest": "neo4j",
        "data_names": ["company_profile"],
        "dag": "eodhd",
    },
    {
        "row": "4b", "agent": "Business Analyst",
        "description": "Financial News (EODHD)",
        "source": "EODHD", "dest": "qdrant",
        "data_names": ["financial_news"],
        "dag": "eodhd",
    },
    {
        "row": "4c", "agent": "Business Analyst",
        "description": "Sentiment Trends",
        "source": "EODHD", "dest": "postgresql",
        "data_names": ["sentiment_trends"],
        "dag": "eodhd",
    },
    # ── Business Analyst (FMP) ────────────────────────────────────────────────
    {
        "row": 1,  "agent": "Business Analyst",
        "description": "SEC Filings (10-K, 10-Q, 8-K)",
        "source": "FMP", "dest": "qdrant",
        "data_names": ["sec_filings_10k", "sec_filings_10q", "sec_filings_8k"],
        "dag": "fmp",
    },
    {
        "row": 2,  "agent": "Business Analyst",
        "description": "Earnings Call Transcripts",
        "source": "FMP", "dest": "qdrant",
        "data_names": ["earnings_call_transcripts"],
        "dag": "fmp",
    },
    {
        "row": 3,  "agent": "Business Analyst",
        "description": "Risk Factors / MD&A Narratives",
        "source": "FMP", "dest": "neo4j",
        "data_names": ["risk_factors", "company_notes", "press_releases", "stock_news"],
        "dag": "fmp",
    },
    {
        "row": "3b", "agent": "Business Analyst",
        "description": "Company Profile (FMP)",
        "source": "FMP", "dest": "neo4j",
        "data_names": ["company_profile", "key_executives", "stock_quote"],
        "dag": "fmp",
    },
    # ── Quantitative (FMP) ────────────────────────────────────────────────────
    {
        "row": 5,  "agent": "Quantitative",
        "description": "Financial Statements (Normalized)",
        "source": "FMP", "dest": "postgresql",
        "data_names": ["income_statement", "balance_sheet", "cash_flow"],
        "dag": "fmp",
    },
    {
        "row": 6,  "agent": "Quantitative",
        "description": "Financial Ratios / Growth Metrics",
        "source": "FMP", "dest": "postgresql",
        "data_names": ["financial_ratios", "ratios_ttm", "key_metrics", "key_metrics_ttm",
                       "financial_growth", "enterprise_values"],
        "dag": "fmp",
    },
    {
        "row": 7,  "agent": "Quantitative",
        "description": "Financial Scores (Piotroski, Beneish)",
        "source": "FMP", "dest": "postgresql",
        "data_names": ["financial_scores", "shares_float", "rating", "historical_market_cap",
                       "company_core_info",
                       "income_statement_as_reported", "balance_sheet_as_reported",
                       "cash_flow_as_reported"],
        "dag": "fmp",
    },
    # ── Quantitative (EODHD) ─────────────────────────────────────────────────
    {
        "row": 8,  "agent": "Quantitative",
        "description": "Historical Stock Prices (EOD)",
        "source": "EODHD", "dest": "postgresql",
        "data_names": ["historical_prices_eod"],
        "dag": "eodhd",
    },
    {
        "row": 9,  "agent": "Quantitative",
        "description": "Intraday / Live Quotes",
        "source": "EODHD", "dest": "postgresql",
        "data_names": ["realtime_quote", "live_stock_price",
                       "intraday_1m", "intraday_5m", "intraday_15m", "intraday_1h"],
        "dag": "eodhd",
    },
    {
        "row": 10, "agent": "Quantitative",
        "description": "Beta & Volatility / Technicals",
        "source": "EODHD", "dest": "postgresql",
        "data_names": ["technical_sma", "technical_ema", "options_data"],
        "dag": "eodhd",
    },
    # ── Financial Modeling (FMP) ──────────────────────────────────────────────
    {
        "row": 11, "agent": "Financial Modeling",
        "description": "As-Reported Financials (GAAP/IFRS)",
        "source": "FMP", "dest": "postgresql",
        "data_names": ["income_statement", "balance_sheet", "cash_flow",
                       "dcf", "advanced_dcf", "levered_dcf", "owner_earnings",
                       "enterprise_values", "market_cap_history"],
        "dag": "fmp",
    },
    {
        "row": 12, "agent": "Financial Modeling",
        "description": "Revenue Segmentation (Product/Geo)",
        "source": "FMP", "dest": "postgresql",
        "data_names": ["revenue_product_segmentation", "revenue_geographic_segmentation"],
        "dag": "fmp",
    },
    {
        "row": 13, "agent": "Financial Modeling",
        "description": "Analyst Estimates / Price Targets",
        "source": "FMP", "dest": "postgresql",
        "data_names": ["analyst_estimates", "analyst_estimates_eps", "analyst_estimates_revenue",
                       "price_target", "price_target_consensus"],
        "dag": "fmp",
    },
    {
        "row": 15, "agent": "Financial Modeling",
        "description": "Treasury Rates / WACC Inputs",
        "source": "FMP", "dest": "postgresql",
        "data_names": ["treasury_rates"],
        "dag": "fmp",
    },
    {
        "row": 17, "agent": "Financial Modeling",
        "description": "Industry Benchmarks / Peers",
        "source": "FMP", "dest": "postgresql",
        "data_names": ["stock_peers", "historical_sectors_performance",
                       "company_outlook", "company_notes"],
        "dag": "fmp",
    },
    # ── Financial Modeling (EODHD) ────────────────────────────────────────────
    {
        "row": 14, "agent": "Financial Modeling",
        "description": "Dividend History / Stock Splits",
        "source": "EODHD", "dest": "postgresql",
        "data_names": ["dividends_history", "splits_history",
                       "historical_prices_weekly", "historical_prices_monthly"],
        "dag": "eodhd",
    },
    {
        "row": 16, "agent": "Financial Modeling",
        "description": "Economic Indicators (Macro)",
        "source": "EODHD", "dest": "postgresql",
        "data_names": ["economic_indicators_gdp", "economic_indicators_cpi",
                       "economic_indicators_unemployment"],
        "dag": "eodhd",
        "ticker": "_MACRO",  # stored under _MACRO pseudo-ticker
    },
]

# ── Airflow system tables to skip ─────────────────────────────────────────────
_AIRFLOW_PREFIXES = (
    "alembic_", "dag", "job", "log", "slot_pool", "task_", "trigger",
    "xcom", "ab_", "callback_", "import_error", "serialized_", "rendered_",
    "connection", "variable", "celery_", "dataset_",
)


def _is_airflow_table(name: str) -> bool:
    return any(name.startswith(p) for p in _AIRFLOW_PREFIXES)


# ── Agent name mapping (metadata.json uses these exact folder names) ──────────
_AGENT_FOLDER = {
    "Business Analyst":   "business_analyst",
    "Quantitative":       "quantitative_fundamental",
    "Financial Modeling": "financial_modeling",
}

# Maps DAG data_name → which raw_ table stores it in PostgreSQL
# time-series data (has a date column) → raw_timeseries
# snapshot/fundamental data (no rolling date) → raw_fundamentals
_TIMESERIES_DATA_NAMES = {
    # EODHD
    "historical_prices_eod", "historical_prices_weekly", "historical_prices_monthly",
    "intraday_1m", "intraday_5m", "intraday_15m", "intraday_1h",
    "realtime_quote", "live_stock_price",
    "technical_sma", "technical_ema",
    "options_data",
    "dividends_history", "splits_history",
    # FMP time-series
    "income_statement", "balance_sheet", "cash_flow",
    "income_statement_as_reported", "balance_sheet_as_reported", "cash_flow_as_reported",
    "financial_ratios", "key_metrics", "financial_growth", "enterprise_values",
    "historical_market_cap", "market_cap_history",
    "analyst_estimates", "analyst_estimates_eps", "analyst_estimates_revenue",
    "price_target",
    "treasury_rates",
    "revenue_product_segmentation", "revenue_geographic_segmentation",
    "historical_sectors_performance",
    "owner_earnings",
    # macro
    "economic_indicators_gdp", "economic_indicators_cpi", "economic_indicators_unemployment",
}


# ══════════════════════════════════════════════════════════════════════════════
# PostgreSQL
# ══════════════════════════════════════════════════════════════════════════════

def _pg_connect():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        dbname=PG_DB, user=PG_USER, password=PG_PASSWORD,
        connect_timeout=5,
    )


def inspect_postgres(ticker_filter: str | None = None) -> None:
    _banner("POSTGRESQL")

    if not HAS_PG:
        _err("psycopg2 not installed — run: pip install psycopg2-binary")
        return

    try:
        conn = _pg_connect()
    except Exception as exc:
        _err(f"Cannot connect to PostgreSQL at {PG_HOST}:{PG_PORT}/{PG_DB}  →  {exc}")
        return

    _ok(f"Connected  {PG_HOST}:{PG_PORT}/{PG_DB}  (user={PG_USER})")

    with conn.cursor() as cur:

        # ── 1. All user tables ─────────────────────────────────────────────────
        _section("All Application Tables")
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        all_tables = [r[0] for r in cur.fetchall()]
        app_tables = [t for t in all_tables if not _is_airflow_table(t)]
        skipped    = len(all_tables) - len(app_tables)

        _table_header(("Table", 35), ("Rows", 10), ("Columns", 8))
        for tname in app_tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {tname}")
                row_count = cur.fetchone()[0]
                cur.execute("""
                    SELECT COUNT(*) FROM information_schema.columns
                    WHERE table_schema='public' AND table_name=%s
                """, (tname,))
                col_count = cur.fetchone()[0]
                color = C.GREEN if row_count > 0 else C.YELLOW
                _table_row(
                    (tname,           35, C.WHITE),
                    (f"{row_count:,}", 10, color),
                    (str(col_count),   8, C.DIM),
                )
            except Exception as exc:
                conn.rollback()
                _table_row((tname, 35, C.WHITE), (f"ERROR: {exc}", 30, C.RED), ("", 8, ""))

        _info(f"({skipped} Airflow system tables hidden)")

        # ── 2. Core table breakdowns ────────────────────────────────────────────
        _section("raw_timeseries  —  per-agent / per-ticker breakdown")
        _pg_breakdown_table(cur, conn, "raw_timeseries",
                            group_cols=["agent_name", "ticker_symbol", "data_name"],
                            ticker_filter=ticker_filter)

        _section("raw_fundamentals  —  per-agent / per-ticker breakdown")
        _pg_breakdown_table(cur, conn, "raw_fundamentals",
                            group_cols=["agent_name", "ticker_symbol", "data_name"],
                            ticker_filter=ticker_filter)

        # ── 3. Sentiment trends ────────────────────────────────────────────────
        _section("sentiment_trends  —  EODHD bullish/bearish/neutral")
        _pg_sentiment(cur, conn, ticker_filter)

        # ── 4. Global tables ──────────────────────────────────────────────────
        _section("Global / shared tables")
        for tname, desc in [
            ("market_eod_us",            "US EOD benchmark prices (bulk_eod_us)"),
            ("global_economic_calendar", "EODHD macro event calendar"),
            ("global_ipo_calendar",      "EODHD IPO calendar"),
            ("global_macro_indicators",  "EODHD GDP / CPI / unemployment (stored under _MACRO)"),
        ]:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {tname}")
                cnt = cur.fetchone()[0]
                color = C.GREEN if cnt > 0 else C.YELLOW
                mark  = "✔" if cnt > 0 else "⚠"
                print(f"    {color}{mark}  {tname:<35}  {cnt:>8,} rows   {C.DIM}({desc}){C.RESET}")
                if tname == "global_macro_indicators" and cnt > 0:
                    cur.execute(
                        "SELECT indicator, COUNT(*) FROM global_macro_indicators "
                        "GROUP BY indicator ORDER BY indicator"
                    )
                    for row in cur.fetchall():
                        print(f"         {C.DIM}└─ {row[0]:<35} {row[1]:>6,} rows{C.RESET}")
            except Exception:
                conn.rollback()
                print(f"    {C.DIM}   {tname:<35}  (table not yet created){C.RESET}")

        # ── 5. Most-recent ingestion timestamps ────────────────────────────────
        _section("Most-recent ingestion timestamps (raw_timeseries + raw_fundamentals)")
        for tname, date_col in [("raw_timeseries", "ingested_at"), ("raw_fundamentals", "ingested_at")]:
            try:
                cur.execute(
                    f"SELECT agent_name, ticker_symbol, data_name, MAX({date_col}) "
                    f"FROM {tname} "
                    + (f"WHERE ticker_symbol=%s " if ticker_filter else "")
                    + f"GROUP BY agent_name, ticker_symbol, data_name "
                    f"ORDER BY MAX({date_col}) DESC LIMIT 10",
                    (ticker_filter,) if ticker_filter else None,
                )
                rows = cur.fetchall()
                if rows:
                    print(f"\n    {C.BOLD}{tname}{C.RESET}  (top 10 most recent)")
                    _table_header(("Agent", 26), ("Ticker", 8), ("data_name", 36), ("Last ingested", 22))
                    for r in rows:
                        _table_row(
                            (r[0], 26, C.DIM),
                            (r[1],  8, C.CYAN),
                            (r[2], 36, C.WHITE),
                            (str(r[3])[:21], 22, C.DIM),
                        )
            except Exception as exc:
                conn.rollback()
                _warn(f"Timestamp query on {tname} failed: {exc}")

    conn.close()


def _pg_breakdown_table(
    cur,
    conn,
    tname: str,
    group_cols: list[str],
    ticker_filter: str | None,
) -> None:
    """Print agent / ticker / data_name row-count breakdown for a raw_ table."""
    try:
        cur.execute(f"SELECT COUNT(*) FROM {tname}")
        total = cur.fetchone()[0]
    except Exception:
        conn.rollback()
        _warn(f"{tname} does not exist yet")
        return

    if total == 0:
        _warn(f"{tname} is empty")
        return

    _ok(f"{tname}: {total:,} rows total")

    try:
        where = "WHERE ticker_symbol=%s " if ticker_filter else ""
        params = (ticker_filter,) if ticker_filter else None
        cur.execute(
            f"SELECT {', '.join(group_cols)}, COUNT(*) "
            f"FROM {tname} {where}"
            f"GROUP BY {', '.join(group_cols)} "
            f"ORDER BY {group_cols[0]}, {group_cols[1]}, {group_cols[2]}",
            params,
        )
        rows = cur.fetchall()
        if rows:
            widths = [22, 8, 38, 10]
            _table_header(
                ("Agent", widths[0]), ("Ticker", widths[1]),
                ("data_name", widths[2]), ("Rows", widths[3]),
            )
            last_agent = last_ticker = None
            for r in rows:
                agent  = r[0] if r[0] != last_agent  else "↳"
                ticker = r[1] if r[1] != last_ticker or r[0] != last_agent else "↳"
                last_agent  = r[0]
                last_ticker = r[1]
                color = C.GREEN if r[3] > 0 else C.YELLOW
                _table_row(
                    (agent[:21],  widths[0], C.DIM),
                    (ticker[:7],  widths[1], C.CYAN),
                    (r[2][:37],   widths[2], C.WHITE),
                    (f"{r[3]:,}", widths[3], color),
                )
    except Exception as exc:
        conn.rollback()
        _warn(f"Breakdown query failed: {exc}")


def _pg_sentiment(cur, conn, ticker_filter: str | None) -> None:
    try:
        cur.execute("SELECT COUNT(*) FROM sentiment_trends")
        total = cur.fetchone()[0]
    except Exception:
        conn.rollback()
        _warn("sentiment_trends does not exist yet")
        return

    if total == 0:
        _warn("sentiment_trends is empty — EODHD sentiment not yet ingested")
        return

    _ok(f"sentiment_trends: {total:,} rows")
    try:
        where = "WHERE ticker=%s " if ticker_filter else ""
        params = (ticker_filter,) if ticker_filter else None
        cur.execute(
            "SELECT ticker, as_of_date, bullish_pct, bearish_pct, neutral_pct, trend "
            f"FROM sentiment_trends {where}ORDER BY ticker, as_of_date DESC LIMIT 10",
            params,
        )
        rows = cur.fetchall()
        if rows:
            _table_header(
                ("Ticker", 8), ("Date", 12), ("Bullish%", 10),
                ("Bearish%", 10), ("Neutral%", 10), ("Trend", 14),
            )
            for r in rows:
                _table_row(
                    (str(r[0]),           8,  C.CYAN),
                    (str(r[1])[:11],      12, C.DIM),
                    (f"{r[2] or 0:.1f}", 10,  C.GREEN),
                    (f"{r[3] or 0:.1f}", 10,  C.RED),
                    (f"{r[4] or 0:.1f}", 10,  C.YELLOW),
                    (str(r[5] or "?"),    14, C.WHITE),
                )
    except Exception as exc:
        conn.rollback()
        _warn(f"Sentiment sample failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Neo4j
# ══════════════════════════════════════════════════════════════════════════════

def inspect_neo4j(ticker_filter: str | None = None) -> None:
    _banner("NEO4J")

    if not HAS_NEO4J:
        _err("neo4j driver not installed — run: pip install neo4j")
        return

    try:
        driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD),
            notifications_min_severity="OFF",
        )
        driver.verify_connectivity()
        _ok(f"Connected  {NEO4J_URI}  (user={NEO4J_USER})")
    except TypeError:
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            driver.verify_connectivity()
            _ok(f"Connected  {NEO4J_URI}  (user={NEO4J_USER})")
        except Exception as exc:
            _err(f"Cannot connect to Neo4j at {NEO4J_URI}  →  {exc}")
            return
    except Exception as exc:
        _err(f"Cannot connect to Neo4j at {NEO4J_URI}  →  {exc}")
        return

    with driver.session() as s:

        # ── 1. Node counts ─────────────────────────────────────────────────────
        _section("Node Counts by Label")
        try:
            labels = [r["label"] for r in s.run("CALL db.labels() YIELD label RETURN label")]
        except Exception:
            labels = []

        # Always show expected labels first
        for lbl in ["Company", "Chunk", "Fact", "Risk", "Strategy"]:
            if lbl not in labels:
                labels.insert(0, lbl)
        labels = list(dict.fromkeys(labels))

        _table_header(("Label", 20), ("Nodes", 10), ("Status", 30))
        for lbl in labels:
            try:
                cnt = s.run(f"MATCH (n:`{lbl}`) RETURN count(n) AS c").single()["c"]
            except Exception:
                cnt = 0
            status = "populated" if cnt > 0 else "empty — not yet ingested"
            color  = C.GREEN if cnt > 0 else C.YELLOW
            _table_row((lbl, 20, C.WHITE), (f"{cnt:,}", 10, color), (status, 30, C.DIM))

        # ── 2. Relationship counts ─────────────────────────────────────────────
        _section("Relationship Counts")
        try:
            rels = s.run(
                "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY cnt DESC"
            ).data()
            if rels:
                _table_header(("Relationship Type", 28), ("Count", 10))
                for r in rels:
                    _table_row((r["rel"], 28, C.CYAN), (f"{r['cnt']:,}", 10, C.WHITE))
            else:
                _warn("No relationships found — graph not yet populated")
        except Exception as exc:
            _warn(f"Relationship query failed: {exc}")

        # ── 3. Company nodes ───────────────────────────────────────────────────
        _section("Company Nodes  (EODHD company_profile → Neo4j)")
        try:
            q = (
                "MATCH (c:Company) WHERE c.ticker=$t RETURN c ORDER BY c.ticker"
                if ticker_filter else
                "MATCH (c:Company) RETURN c ORDER BY c.ticker"
            )
            params = {"t": ticker_filter} if ticker_filter else {}
            companies = [r["c"] for r in s.run(q, **params)]

            if not companies:
                _warn("No :Company nodes found — EODHD company_profile not yet loaded")
            else:
                _table_header(
                    ("Ticker", 8), ("Name", 34), ("Sector", 24), ("Exchange", 10), ("Props", 5)
                )
                for node in companies:
                    ticker   = node.get("ticker", "?")
                    name     = (node.get("companyName") or node.get("Name") or "—")[:33]
                    sector   = (node.get("sector") or node.get("Sector") or node.get("GicSector") or "—")[:23]
                    exchange = (node.get("exchangeShortName") or node.get("Exchange") or "—")[:9]
                    n_props  = len(dict(node))
                    _table_row(
                        (ticker,      8,  C.CYAN),
                        (name,        34, C.WHITE),
                        (sector,      24, C.DIM),
                        (exchange,    10, C.DIM),
                        (str(n_props), 5, C.DIM),
                    )
                if companies:
                    keys = sorted(dict(companies[0]).keys())
                    _info(f"Property keys on first node: {', '.join(keys)}")
        except Exception as exc:
            _err(f"Company query failed: {exc}")

        # ── 4. Risk nodes ──────────────────────────────────────────────────────
        _section("Risk Nodes  (FMP risk_factors → Neo4j  [row 3])")
        try:
            q = (
                "MATCH (c:Company {ticker: $t})-[:FACES_RISK]->(r:Risk) "
                "RETURN c.ticker AS ticker, count(r) AS cnt"
                if ticker_filter else
                "MATCH (c:Company)-[:FACES_RISK]->(r:Risk) "
                "RETURN c.ticker AS ticker, count(r) AS cnt ORDER BY ticker"
            )
            params = {"t": ticker_filter} if ticker_filter else {}
            risk_rows = s.run(q, **params).data()
            if not risk_rows:
                _warn("No :Risk nodes — FMP risk_factors not yet loaded (FMP DAG paused)")
            else:
                _table_header(("Ticker", 8), ("Risk Nodes", 12))
                for r in risk_rows:
                    _table_row((r["ticker"], 8, C.CYAN), (f"{r['cnt']:,}", 12, C.WHITE))
        except Exception as exc:
            _warn(f"Risk query failed: {exc}")

        # ── 5. Strategy nodes ─────────────────────────────────────────────────
        _section("Strategy Nodes  (FMP company_outlook / company_notes → Neo4j)")
        try:
            q = (
                "MATCH (c:Company {ticker: $t})-[:HAS_STRATEGY]->(s:Strategy) "
                "RETURN c.ticker AS ticker, count(s) AS cnt"
                if ticker_filter else
                "MATCH (c:Company)-[:HAS_STRATEGY]->(s:Strategy) "
                "RETURN c.ticker AS ticker, count(s) AS cnt ORDER BY ticker"
            )
            params = {"t": ticker_filter} if ticker_filter else {}
            strat_rows = s.run(q, **params).data()
            if not strat_rows:
                _warn("No :Strategy nodes — FMP company_outlook not yet loaded (FMP DAG paused)")
            else:
                _table_header(("Ticker", 8), ("Strategy Nodes", 16))
                for r in strat_rows:
                    _table_row((r["ticker"], 8, C.CYAN), (f"{r['cnt']:,}", 16, C.WHITE))
        except Exception as exc:
            _warn(f"Strategy query failed: {exc}")

        # ── 6. Fact nodes (key_executives, stock_quote, etc.) ─────────────────
        _section("Fact Nodes  (per data_name from FMP BA agent)")
        try:
            q = (
                "MATCH (c:Company {ticker: $t})-[:HAS_FACT]->(f:Fact) "
                "RETURN c.ticker AS ticker, f.data_name AS dn, count(f) AS cnt "
                "ORDER BY ticker, dn"
                if ticker_filter else
                "MATCH (c:Company)-[:HAS_FACT]->(f:Fact) "
                "RETURN c.ticker AS ticker, f.data_name AS dn, count(f) AS cnt "
                "ORDER BY ticker, dn"
            )
            params = {"t": ticker_filter} if ticker_filter else {}
            fact_rows = s.run(q, **params).data()
            if not fact_rows:
                _warn("No :Fact nodes")
            else:
                _table_header(("Ticker", 8), ("data_name", 34), ("Facts", 8))
                for r in fact_rows:
                    _table_row(
                        (r["ticker"],     8,  C.CYAN),
                        (str(r["dn"])[:33], 34, C.WHITE),
                        (f"{r['cnt']:,}", 8,  C.DIM),
                    )
        except Exception as exc:
            _warn(f"Fact query failed: {exc}")

        # ── 7. Chunk nodes (LLM synthesis) ─────────────────────────────────────
        _section("Chunk Nodes  (LLM synthesis via ingest_neo4j_chunks)")
        try:
            total_chunks = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
            if total_chunks == 0:
                _warn("No :Chunk nodes — ingest_neo4j_chunks not yet run")
            else:
                _ok(f"{total_chunks:,} Chunk nodes total")
                q_chunks = (
                    "MATCH (c:Chunk) WHERE c.ticker=$t "
                    "RETURN c.ticker AS ticker, count(c) AS cnt, "
                    "min(c.filing_date) AS earliest, max(c.filing_date) AS latest"
                    if ticker_filter else
                    "MATCH (c:Chunk) RETURN c.ticker AS ticker, count(c) AS cnt, "
                    "min(c.filing_date) AS earliest, max(c.filing_date) AS latest "
                    "ORDER BY ticker"
                )
                params_c = {"t": ticker_filter} if ticker_filter else {}
                rows = s.run(q_chunks, **params_c).data()
                if rows:
                    _table_header(
                        ("Ticker", 8), ("Chunks", 8),
                        ("Earliest filing", 16), ("Latest filing", 16),
                    )
                    for r in rows:
                        _table_row(
                            (r.get("ticker") or "?",         8,  C.CYAN),
                            (f"{r['cnt']:,}",                 8,  C.WHITE),
                            (str(r.get("earliest") or "—")[:15], 16, C.DIM),
                            (str(r.get("latest") or "—")[:15],   16, C.DIM),
                        )
        except Exception as exc:
            _warn(f"Chunk query failed: {exc}")

        # ── 8. Peer edges ──────────────────────────────────────────────────────
        _section("COMPETES_WITH Peer Edges  (FMP stock_peers → Neo4j)")
        try:
            q = (
                "MATCH (c:Company {ticker: $t})-[:COMPETES_WITH]->(p:Company) "
                "RETURN c.ticker AS src, collect(p.ticker) AS peers"
                if ticker_filter else
                "MATCH (c:Company)-[:COMPETES_WITH]->(p:Company) "
                "RETURN c.ticker AS src, count(p) AS n ORDER BY src"
            )
            params = {"t": ticker_filter} if ticker_filter else {}
            peer_rows = s.run(q, **params).data()
            if not peer_rows:
                _warn("No COMPETES_WITH edges — FMP stock_peers not yet loaded")
            else:
                if ticker_filter:
                    for r in peer_rows:
                        peers = r.get("peers", [])
                        _ok(f"{r['src']} competes with: {', '.join(peers)}")
                else:
                    _table_header(("Ticker", 8), ("Peer Count", 12))
                    for r in peer_rows:
                        _table_row((r["src"], 8, C.CYAN), (f"{r['n']:,}", 12, C.WHITE))
        except Exception as exc:
            _warn(f"Peer query failed: {exc}")

        # ── 9. Vector indexes ─────────────────────────────────────────────────
        _section("Vector Indexes  (all-MiniLM-L6-v2 / 384-dim via sentence_transformers)")
        try:
            indexes = s.run("SHOW INDEXES WHERE type = 'VECTOR'").data()
            if not indexes:
                _warn("No VECTOR indexes found")
            else:
                _table_header(
                    ("Name", 24), ("State", 10), ("Label", 12),
                    ("Property", 16), ("Dimension", 10),
                )
                for idx in indexes:
                    cfg   = idx.get("options", {}).get("indexConfig", {})
                    dim   = cfg.get("vector.dimensions", "?")
                    state = idx.get("state", "?")
                    color = C.GREEN if state == "ONLINE" else C.YELLOW
                    _table_row(
                        (idx.get("name", "?"),                    24, C.WHITE),
                        (state,                                   10, color),
                        (str(idx.get("labelsOrTypes", ["?"])[0]), 12, C.DIM),
                        (str(idx.get("properties", ["?"])[0]),    16, C.DIM),
                        (str(dim),                                10, C.DIM),
                    )
        except Exception as exc:
            _warn(f"SHOW INDEXES failed: {exc}")

    driver.close()


# ══════════════════════════════════════════════════════════════════════════════
# Qdrant
# ══════════════════════════════════════════════════════════════════════════════

def inspect_qdrant(ticker_filter: str | None = None) -> None:
    _banner("QDRANT")

    if not HAS_REQUESTS:
        _err("requests not installed — run: pip install requests")
        return

    base = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

    try:
        r = requests.get(f"{base}/healthz", timeout=5)
        if r.status_code != 200:
            r = requests.get(f"{base}/readyz", timeout=5)
        if r.status_code == 200:
            _ok(f"Connected  {base}")
        else:
            _warn(f"Qdrant responded with HTTP {r.status_code}")
    except Exception as exc:
        _err(f"Cannot connect to Qdrant at {base}  →  {exc}")
        return

    # ── Collection list ────────────────────────────────────────────────────────
    _section("Collections Overview")
    try:
        resp = requests.get(f"{base}/collections", timeout=5)
        collections = resp.json().get("result", {}).get("collections", [])
    except Exception as exc:
        _err(f"Cannot list collections: {exc}")
        return

    if not collections:
        _warn("No collections found — Qdrant is empty")
        return

    _table_header(
        ("Collection", 28), ("Points", 10), ("Vectors", 10),
        ("Dim", 6), ("Distance", 10), ("Status", 10),
    )
    for col in collections:
        cname = col["name"]
        try:
            det  = requests.get(f"{base}/collections/{cname}", timeout=5).json().get("result", {})
            pts  = det.get("points_count", det.get("vectors_count", 0))
            vecs = det.get("vectors_count", pts)
            stat = det.get("status", "?")
            cfg  = det.get("config", {}).get("params", {}).get("vectors", {})
            if isinstance(cfg, dict) and "size" in cfg:
                dim = cfg.get("size", "?"); dist = cfg.get("distance", "?")
            elif isinstance(cfg, dict) and cfg:
                first = next(iter(cfg.values()), {})
                dim = first.get("size", "?"); dist = first.get("distance", "?")
            else:
                dim = dist = "?"
            color = C.GREEN if pts > 0 else C.YELLOW
            _table_row(
                (cname,       28, C.WHITE),
                (f"{pts:,}",  10, color),
                (f"{vecs:,}", 10, C.DIM),
                (str(dim),     6, C.DIM),
                (str(dist),   10, C.DIM),
                (stat,        10, C.DIM),
            )
        except Exception as exc:
            _table_row((cname, 28, C.WHITE), (f"ERROR: {exc}", 50, C.RED),
                       ("", 0, ""), ("", 0, ""), ("", 0, ""), ("", 0, ""))

    # ── Per-collection detail ──────────────────────────────────────────────────
    for col in collections:
        cname = col["name"]
        _section(f"Collection: {cname}")

        try:
            det = requests.get(f"{base}/collections/{cname}", timeout=5).json().get("result", {})
            pts = det.get("points_count", 0)
            cfg = det.get("config", {}).get("params", {}).get("vectors", {})
            if isinstance(cfg, dict) and "size" in cfg:
                dim = cfg.get("size", "?"); dist = cfg.get("distance", "?")
            elif isinstance(cfg, dict) and cfg:
                first = next(iter(cfg.values()), {})
                dim = first.get("size", "?"); dist = first.get("distance", "?")
            else:
                dim = dist = "?"
            _row("Points (vectors)", f"{pts:,}")
            _row("Dimensions",       str(dim))
            _row("Distance metric",  str(dist))
            _row("Status",           det.get("status", "?"))
        except Exception as exc:
            _warn(f"Detail fetch failed: {exc}")
            continue

        if pts == 0:
            _warn("Collection is empty")
            continue

        # Payload field discovery
        try:
            scroll = requests.post(
                f"{base}/collections/{cname}/points/scroll",
                json={"limit": 5, "with_payload": True, "with_vector": False},
                timeout=10,
            ).json().get("result", {}).get("points", [])
            if scroll:
                all_keys: set = set()
                for p in scroll:
                    all_keys.update(p.get("payload", {}).keys())
                _row("Payload fields", ", ".join(sorted(all_keys)), color=C.DIM)
        except Exception:
            pass

        # ── Per-ticker vector counts ───────────────────────────────────────────
        ticker_field = _qdrant_find_ticker_field(base, cname)
        if ticker_field:
            _section(f"  Per-Ticker Counts  (field: {ticker_field})")
            tickers_to_check = [ticker_filter] if ticker_filter else TRACKED_TICKERS
            _table_header(("Ticker", 10), ("Vectors", 10), ("Sample title / source", 48))
            for t in tickers_to_check:
                try:
                    cnt_resp = requests.post(
                        f"{base}/collections/{cname}/points/count",
                        json={"filter": {"must": [{"key": ticker_field, "match": {"value": t}}]}},
                        timeout=5,
                    ).json()
                    count = cnt_resp.get("result", {}).get("count", 0)

                    sample_title = ""
                    if count > 0:
                        sr = requests.post(
                            f"{base}/collections/{cname}/points/scroll",
                            json={
                                "limit": 1, "with_payload": True, "with_vector": False,
                                "filter": {"must": [{"key": ticker_field, "match": {"value": t}}]},
                            },
                            timeout=5,
                        ).json()
                        pts_list = sr.get("result", {}).get("points", [])
                        if pts_list:
                            pl = pts_list[0].get("payload", {})
                            sample_title = (
                                pl.get("title") or pl.get("data_name")
                                or pl.get("source") or ""
                            )[:47]

                    if count > 0:
                        _table_row(
                            (t,             10, C.CYAN),
                            (f"{count:,}",  10, C.GREEN),
                            (sample_title,  48, C.DIM),
                        )
                    elif ticker_filter:
                        _table_row(
                            (t,              10, C.YELLOW),
                            ("0",            10, C.YELLOW),
                            ("no vectors",   48, C.DIM),
                        )
                except Exception:
                    pass

            # ── Per-data_name breakdown ────────────────────────────────────────
            _section(f"  Per-data_name Counts  (field: data_name)")
            _qdrant_data_name_breakdown(base, cname, ticker_filter, ticker_field)

        # ── Recent ingestion ───────────────────────────────────────────────────
        _section(f"  Recent Ingestion  (last 5 points by ingested_at)")
        try:
            recent = requests.post(
                f"{base}/collections/{cname}/points/scroll",
                json={
                    "limit": 5, "with_payload": True, "with_vector": False,
                    "order_by": {"key": "ingested_at", "direction": "desc"},
                },
                timeout=10,
            ).json().get("result", {}).get("points", [])
            if recent:
                _table_header(
                    ("ID", 8), ("Ticker", 8), ("data_name", 28),
                    ("Ingested at", 22), ("Source / Title", 28),
                )
                for p in recent:
                    pl = p.get("payload", {})
                    _table_row(
                        (str(p["id"])[:7],                                          8,  C.DIM),
                        (pl.get("ticker_symbol") or pl.get("ticker") or "?",        8,  C.CYAN),
                        (str(pl.get("data_name") or "?")[:27],                      28, C.WHITE),
                        (str(pl.get("ingested_at") or "?")[:21],                    22, C.DIM),
                        ((pl.get("title") or pl.get("source") or "?")[:27],         28, C.DIM),
                    )
            else:
                _info("order_by not supported on this Qdrant version — skipping")
        except Exception:
            _info("Recency view unavailable")


def _qdrant_find_ticker_field(base: str, cname: str) -> str | None:
    """Return 'ticker_symbol' or 'ticker' if either exists in this collection."""
    for candidate in ("ticker_symbol", "ticker"):
        try:
            test = requests.post(
                f"{base}/collections/{cname}/points/scroll",
                json={
                    "limit": 1, "with_payload": True, "with_vector": False,
                    "filter": {"must": [{"key": candidate, "match": {"value": "AAPL"}}]},
                },
                timeout=5,
            ).json()
            if "error" not in str(test.get("status", "")):
                return candidate
        except Exception:
            pass
    return None


def _qdrant_data_name_breakdown(
    base: str, cname: str, ticker_filter: str | None, ticker_field: str
) -> None:
    """Count vectors per data_name (optionally filtered to one ticker)."""
    # Fetch up to 5000 points to build a data_name histogram
    # (Qdrant doesn't support GROUP BY natively)
    try:
        filt = {}
        if ticker_filter:
            filt = {"filter": {"must": [{"key": ticker_field, "match": {"value": ticker_filter}}]}}
        # Use scroll with offset pagination — cap at 2000 points for speed
        counts: dict[str, int] = {}
        next_offset = None
        total_scanned = 0
        while total_scanned < 2000:
            req: dict = {"limit": 250, "with_payload": ["data_name", "source"], "with_vector": False}
            req.update(filt)
            if next_offset:
                req["offset"] = next_offset
            resp = requests.post(
                f"{base}/collections/{cname}/points/scroll",
                json=req, timeout=10,
            ).json().get("result", {})
            pts = resp.get("points", [])
            if not pts:
                break
            for p in pts:
                dn = p.get("payload", {}).get("data_name", "unknown")
                counts[dn] = counts.get(dn, 0) + 1
            total_scanned += len(pts)
            next_offset = resp.get("next_page_offset")
            if next_offset is None:
                break

        if counts:
            _table_header(("data_name", 38), ("Vectors (sample)", 18))
            for dn, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                _table_row((dn[:37], 38, C.WHITE), (f"{cnt:,}+", 18, C.DIM))
        else:
            _info("No data_name field found in payloads")
    except Exception as exc:
        _warn(f"data_name breakdown failed: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# Coverage report — data_needed.txt vs. what's actually in the DB
# ══════════════════════════════════════════════════════════════════════════════

def inspect_coverage(ticker_filter: str | None = None) -> None:
    _banner("DATA COVERAGE  —  data_needed.txt vs. DB")

    tickers = [ticker_filter] if ticker_filter else TRACKED_TICKERS

    # ── Gather PostgreSQL presence ─────────────────────────────────────────────
    pg_present: dict[tuple, int] = {}  # (agent_folder, ticker, data_name) → row_count
    if HAS_PG:
        try:
            conn = _pg_connect()
            with conn.cursor() as cur:
                for tname in ("raw_timeseries", "raw_fundamentals"):
                    date_col = "ts_date" if tname == "raw_timeseries" else "as_of_date"
                    try:
                        cur.execute(
                            f"SELECT agent_name, ticker_symbol, data_name, COUNT(*) "
                            f"FROM {tname} GROUP BY agent_name, ticker_symbol, data_name"
                        )
                        for r in cur.fetchall():
                            pg_present[(r[0], r[1], r[2])] = r[3]
                    except Exception:
                        conn.rollback()

                # sentiment_trends
                try:
                    cur.execute("SELECT ticker, COUNT(*) FROM sentiment_trends GROUP BY ticker")
                    for r in cur.fetchall():
                        pg_present[("business_analyst", r[0], "sentiment_trends")] = r[1]
                except Exception:
                    conn.rollback()

                # global_macro_indicators
                try:
                    cur.execute(
                        "SELECT indicator, COUNT(*) FROM global_macro_indicators GROUP BY indicator"
                    )
                    for r in cur.fetchall():
                        dn_map = {
                            "gdp_growth_rate": "economic_indicators_gdp",
                            "inflation_cpi":   "economic_indicators_cpi",
                            "unemployment_rate": "economic_indicators_unemployment",
                        }
                        dn = dn_map.get(r[0], r[0])
                        pg_present[("financial_modeling", "_MACRO", dn)] = r[1]
                except Exception:
                    conn.rollback()

            conn.close()
        except Exception:
            pass

    # ── Gather Neo4j presence ──────────────────────────────────────────────────
    neo4j_present: dict[str, int] = {}  # ticker → node count
    if HAS_NEO4J:
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with driver.session() as s:
                rows = s.run(
                    "MATCH (c:Company) RETURN c.ticker AS t, 1 AS n"
                ).data()
                for r in rows:
                    neo4j_present[r["t"]] = 1
            driver.close()
        except Exception:
            pass

    # ── Gather Qdrant presence ─────────────────────────────────────────────────
    qdrant_present: dict[tuple, int] = {}  # (ticker, data_name) → count
    if HAS_REQUESTS:
        base = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
        try:
            resp = requests.get(f"{base}/collections", timeout=5)
            collections = resp.json().get("result", {}).get("collections", [])
            for col in collections:
                cname = col["name"]
                # Sample up to 2000 points to build a histogram
                next_off = None
                total = 0
                while total < 2000:
                    req = {"limit": 250, "with_payload": ["ticker_symbol", "data_name"],
                           "with_vector": False}
                    if next_off:
                        req["offset"] = next_off
                    res = requests.post(
                        f"{base}/collections/{cname}/points/scroll",
                        json=req, timeout=10,
                    ).json().get("result", {})
                    pts = res.get("points", [])
                    if not pts:
                        break
                    for p in pts:
                        pl  = p.get("payload", {})
                        t   = pl.get("ticker_symbol") or pl.get("ticker") or "?"
                        dn  = pl.get("data_name", "unknown")
                        key = (t, dn)
                        qdrant_present[key] = qdrant_present.get(key, 0) + 1
                    total += len(pts)
                    next_off = res.get("next_page_offset")
                    if next_off is None:
                        break
        except Exception:
            pass

    # ── Print coverage table ───────────────────────────────────────────────────
    _section("Coverage Matrix  (✔ = data found, ✘ = missing, ~ = FMP DAG paused)")

    _table_header(
        ("Row", 4), ("Agent", 18), ("Description", 36), ("Src", 5),
        ("Dest", 5), ("Status", 10),
    )

    for spec in DATA_SPEC:
        row_id = spec["row"]
        agent  = spec["agent"][:17]
        desc   = spec["description"][:35]
        src    = spec["source"][:4]
        dest   = spec["dest"][:4]
        dag    = spec.get("dag", "")
        macro_ticker = spec.get("ticker")  # "_MACRO" for macro rows
        data_names = spec["data_names"]
        agent_folder = _AGENT_FOLDER.get(spec["agent"], spec["agent"].lower())

        found = False

        if spec["dest"] == "postgresql":
            if macro_ticker == "_MACRO":
                # check global_macro_indicators via pg_present
                found = any(
                    pg_present.get((agent_folder, "_MACRO", dn), 0) > 0
                    for dn in data_names
                )
            else:
                # check raw_timeseries or raw_fundamentals
                for t in tickers:
                    if any(
                        pg_present.get((agent_folder, t, dn), 0) > 0
                        for dn in data_names
                    ):
                        found = True
                        break

        elif spec["dest"] == "neo4j":
            found = any(neo4j_present.get(t, 0) > 0 for t in tickers)

        elif spec["dest"] == "qdrant":
            for t in tickers:
                if any(qdrant_present.get((t, dn), 0) > 0 for dn in data_names):
                    found = True
                    break

        if found:
            status_str = "✔ ingested"
            s_color = C.GREEN
        elif dag == "fmp":
            status_str = "~ FMP paused"
            s_color = C.YELLOW
        else:
            status_str = "✘ missing"
            s_color = C.RED

        _table_row(
            (str(row_id),  4,  C.DIM),
            (agent,        18, C.WHITE),
            (desc,         36, C.DIM),
            (src,          5,  C.CYAN),
            (dest,         5,  C.CYAN),
            (status_str,   10, s_color),
        )

    # ── Per-ticker PostgreSQL completeness ────────────────────────────────────
    _section("Per-Ticker PostgreSQL completeness  (data_names present in raw_ tables)")
    print(f"\n  {C.BOLD}EODHD data_names  (raw_timeseries){C.RESET}")
    eodhd_dns = [
        "historical_prices_eod", "historical_prices_weekly", "historical_prices_monthly",
        "realtime_quote", "live_stock_price",
        "intraday_1m", "intraday_5m", "intraday_15m", "intraday_1h",
        "technical_sma", "technical_ema", "options_data",
        "dividends_history", "splits_history",
        "sentiment_trends",
    ]
    fmp_ts_dns = [
        "income_statement", "balance_sheet", "cash_flow",
        "income_statement_as_reported", "balance_sheet_as_reported", "cash_flow_as_reported",
        "financial_ratios", "ratios_ttm", "key_metrics", "key_metrics_ttm",
        "financial_growth", "enterprise_values", "financial_scores",
        "shares_float", "historical_market_cap", "rating", "company_core_info",
        "analyst_estimates", "price_target", "revenue_product_segmentation",
        "revenue_geographic_segmentation", "treasury_rates",
        "dcf", "owner_earnings", "stock_peers",
    ]

    for label, dns in [("EODHD", eodhd_dns), ("FMP", fmp_ts_dns)]:
        print(f"\n  {C.BOLD}{C.CYAN}{label}{C.RESET}")
        header_cols = [("data_name", 36)] + [(t, 6) for t in tickers]
        _table_header(*header_cols)
        for dn in dns:
            cells = [(dn[:35], 36, C.WHITE)]
            for t in tickers:
                # Check across all agent folders
                has = any(
                    pg_present.get((af, t, dn), 0) > 0
                    for af in _AGENT_FOLDER.values()
                )
                if not has:
                    # also check sentiment_trends which uses "business_analyst"
                    has = pg_present.get(("business_analyst", t, dn), 0) > 0
                mark  = "✔" if has else ("~" if label == "FMP" else "✘")
                color = C.GREEN if has else (C.YELLOW if label == "FMP" else C.RED)
                cells.append((mark, 6, color))
            _table_row(*cells)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FYP Database Content Inspector — tabular summary of all three databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingestion/db_inspect.py                   # all databases
  python ingestion/db_inspect.py --pg              # PostgreSQL only
  python ingestion/db_inspect.py --neo4j           # Neo4j only
  python ingestion/db_inspect.py --qdrant          # Qdrant only
  python ingestion/db_inspect.py --coverage        # data_needed.txt coverage report
  python ingestion/db_inspect.py --ticker AAPL     # filter all views to AAPL
  python ingestion/db_inspect.py --pg --ticker TSLA
        """,
    )
    parser.add_argument("--pg",       action="store_true", help="PostgreSQL only")
    parser.add_argument("--neo4j",    action="store_true", help="Neo4j only")
    parser.add_argument("--qdrant",   action="store_true", help="Qdrant only")
    parser.add_argument("--coverage", action="store_true", help="data_needed.txt coverage report")
    parser.add_argument(
        "--ticker", default=None, metavar="TICKER",
        help="Filter per-ticker views to this ticker (e.g. AAPL)",
    )
    args = parser.parse_args()

    run_all = not any([args.pg, args.neo4j, args.qdrant, args.coverage])

    print(f"\n{C.BOLD}{C.WHITE}FYP Database Inspector  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}")
    if args.ticker:
        print(f"{C.CYAN}  Ticker filter: {args.ticker}{C.RESET}")

    if run_all or args.pg:
        inspect_postgres(args.ticker)
    if run_all or args.neo4j:
        inspect_neo4j(args.ticker)
    if run_all or args.qdrant:
        inspect_qdrant(args.ticker)
    if run_all or args.coverage:
        inspect_coverage(args.ticker)

    print(f"\n{C.BOLD}{C.DIM}{'─' * 80}{C.RESET}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Interrupted.{C.RESET}")
        sys.exit(0)
    except Exception as exc:
        print(f"\n{C.RED}FATAL: {exc}{C.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
