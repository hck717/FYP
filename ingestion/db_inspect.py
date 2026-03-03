#!/usr/bin/env python3
"""
db_inspect.py — FYP Database Content Inspector

Prints a clean, table-formatted summary of every database used by the system:
  • PostgreSQL  — all non-Airflow tables, row counts, column descriptions, sample tickers
  • Neo4j       — node/relationship counts, Company node details, graph structure
  • Qdrant      — collection stats, per-ticker vector counts, sample payload fields

Usage
-----
    python ingestion/db_inspect.py              # all three databases
    python ingestion/db_inspect.py --pg         # PostgreSQL only
    python ingestion/db_inspect.py --neo4j      # Neo4j only
    python ingestion/db_inspect.py --qdrant     # Qdrant only
    python ingestion/db_inspect.py --ticker AAPL  # filter per-ticker views
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

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

# ── Connection config (env-var overridable) ───────────────────────────────────
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


def _table_header(*cols: tuple[str, int]) -> None:
    """cols = list of (header_text, width)"""
    line = "  " + "  ".join(f"{C.BOLD}{C.CYAN}{h:<{w}}{C.RESET}" for h, w in cols)
    sep  = "  " + "  ".join("─" * w for _, w in cols)
    print(f"\n{line}")
    print(f"{C.DIM}{sep}{C.RESET}")


def _table_row(*cells: tuple[str, int, str]) -> None:
    """cells = list of (text, width, color)"""
    print("  " + "  ".join(f"{color}{str(text):<{w}}{C.RESET}" for text, w, color in cells))


def _ok(msg: str)   -> None: print(f"    {C.GREEN}✔  {msg}{C.RESET}")
def _warn(msg: str) -> None: print(f"    {C.YELLOW}⚠  {msg}{C.RESET}")
def _err(msg: str)  -> None: print(f"    {C.RED}✘  {msg}{C.RESET}")
def _info(msg: str) -> None: print(f"    {C.DIM}   {msg}{C.RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# PostgreSQL
# ══════════════════════════════════════════════════════════════════════════════

# Tables we care about — name, short description, ticker_col (None if no per-ticker breakdown)
_PG_TABLES: list[tuple[str, str, str | None]] = [
    ("sentiment_trends",        "Market sentiment % per ticker (bullish/bearish/neutral + trend)", "ticker"),
    ("market_eod_us",           "US equity end-of-day OHLCV prices",                              None),
    ("global_economic_calendar","Macro economic event calendar",                                   None),
    ("global_ipo_calendar",     "Upcoming / recent IPO listings",                                 None),
    ("raw_timeseries",          "Raw time-series data ingested by Airflow DAGs",                   "ticker_symbol"),
    ("raw_fundamentals",        "Raw fundamental data ingested by Airflow DAGs",                   "ticker_symbol"),
]

# Airflow system tables to skip in the "all tables" scan
_AIRFLOW_PREFIXES = (
    "alembic_", "dag", "job", "log", "slot_pool", "task_", "trigger",
    "xcom", "ab_", "callback_", "import_error", "serialized_", "rendered_",
    "connection", "variable", "celery_", "dataset_",
)


def _is_airflow_table(name: str) -> bool:
    return any(name.startswith(p) for p in _AIRFLOW_PREFIXES)


def inspect_postgres(ticker_filter: str | None = None) -> None:
    _banner("POSTGRESQL")

    if not HAS_PG:
        _err("psycopg2 not installed — run: pip install psycopg2-binary")
        return

    try:
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT,
            dbname=PG_DB, user=PG_USER, password=PG_PASSWORD,
            connect_timeout=5,
        )
    except Exception as exc:
        _err(f"Cannot connect to PostgreSQL at {PG_HOST}:{PG_PORT}/{PG_DB}  →  {exc}")
        return

    _ok(f"Connected  {PG_HOST}:{PG_PORT}/{PG_DB}  (user={PG_USER})")

    with conn.cursor() as cur:

        # ── All user tables (excluding Airflow system tables) ─────────────────
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
                    (tname,    35, C.WHITE),
                    (f"{row_count:,}", 10, color),
                    (str(col_count), 8, C.DIM),
                )
            except Exception as exc:
                conn.rollback()
                _table_row((tname, 35, C.WHITE), (f"ERROR: {exc}", 20, C.RED), ("", 8, ""))

        _info(f"({skipped} Airflow system tables hidden)")

        # ── Detail view for known FYP tables ─────────────────────────────────
        _section("FYP Table Details")
        for tname, description, ticker_col in _PG_TABLES:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {tname}")
                row_count = cur.fetchone()[0]
            except Exception:
                conn.rollback()
                continue

            # Column names
            cur.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema='public' AND table_name=%s
                ORDER BY ordinal_position
            """, (tname,))
            cols = cur.fetchall()
            col_str = ", ".join(f"{c[0]}({c[1][:8]})" for c in cols[:8])
            if len(cols) > 8:
                col_str += f" … +{len(cols)-8}"

            print()
            _row("Table",       tname,       color=C.BOLD + C.WHITE)
            _row("Description", description, color=C.DIM)
            _row("Row count",   f"{row_count:,}")
            _row("Columns",     col_str,     color=C.DIM)

            if row_count == 0:
                _warn(f"  {tname} is empty")
                continue

            # Per-ticker breakdown if applicable
            if ticker_col:
                try:
                    if ticker_filter:
                        cur.execute(
                            f"SELECT {ticker_col}, COUNT(*) FROM {tname} "
                            f"WHERE {ticker_col}=%s GROUP BY {ticker_col}",
                            (ticker_filter,),
                        )
                    else:
                        cur.execute(
                            f"SELECT {ticker_col}, COUNT(*) FROM {tname} "
                            f"GROUP BY {ticker_col} ORDER BY {ticker_col}"
                        )
                    ticker_rows = cur.fetchall()
                    if ticker_rows:
                        print()
                        _table_header(("  Ticker", 12), ("Rows", 10))
                        for tr in ticker_rows:
                            _table_row(
                                (f"  {tr[0]}", 12, C.CYAN),
                                (f"{tr[1]:,}", 10, C.WHITE),
                            )
                except Exception as exc:
                    conn.rollback()
                    _warn(f"Ticker breakdown failed: {exc}")

            # Show sample row for sentiment_trends (small, fully informative)
            if tname == "sentiment_trends":
                try:
                    q = (
                        f"SELECT * FROM {tname} WHERE ticker=%s LIMIT 1"
                        if ticker_filter else
                        f"SELECT * FROM {tname} LIMIT 3"
                    )
                    params = (ticker_filter,) if ticker_filter else None
                    cur.execute(q, params)
                    sample_rows = cur.fetchall()
                    col_names = [d[0] for d in cur.description]
                    if sample_rows:
                        print()
                        widths = [max(len(c), 10) for c in col_names]
                        _table_header(*zip(col_names, widths))
                        for sr in sample_rows:
                            cells = [(str(v)[:w], w, C.WHITE) for v, w in zip(sr, widths)]
                            _table_row(*cells)
                except Exception as exc:
                    conn.rollback()
                    _warn(f"Sample query failed: {exc}")

    conn.close()


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
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            notifications_min_severity="OFF",
        )
        driver.verify_connectivity()
        _ok(f"Connected  {NEO4J_URI}  (user={NEO4J_USER})")
    except TypeError:
        # Older driver version — no notifications_min_severity param
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

        # ── Node counts ───────────────────────────────────────────────────────
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
            status = "populated" if cnt > 0 else "empty — data not yet ingested"
            color  = C.GREEN if cnt > 0 else C.YELLOW
            _table_row((lbl, 20, C.WHITE), (f"{cnt:,}", 10, color), (status, 30, C.DIM))

        # ── Relationship counts ───────────────────────────────────────────────
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
                _warn("No relationships found — nodes are isolated (graph not yet populated)")
        except Exception as exc:
            _warn(f"Relationship query failed: {exc}")

        # ── Company nodes ─────────────────────────────────────────────────────
        _section("Company Nodes")
        try:
            q = (
                "MATCH (c:Company) WHERE c.ticker=$t RETURN c ORDER BY c.ticker"
                if ticker_filter else
                "MATCH (c:Company) RETURN c ORDER BY c.ticker"
            )
            params = {"t": ticker_filter} if ticker_filter else {}
            companies = [r["c"] for r in s.run(q, **params)]

            if not companies:
                _warn("No :Company nodes found")
            else:
                _table_header(
                    ("Ticker", 8), ("Name", 36), ("Sector", 24), ("Exchange", 10), ("Properties", 5)
                )
                for node in companies:
                    ticker   = node.get("ticker", "?")
                    name     = (node.get("companyName") or node.get("Name") or node.get("name") or "—")[:35]
                    sector   = (node.get("sector") or node.get("Sector") or node.get("GicSector") or "—")[:23]
                    exchange = (node.get("exchangeShortName") or node.get("Exchange") or "—")[:9]
                    n_props  = len(dict(node))
                    _table_row(
                        (ticker,   8,  C.CYAN),
                        (name,     36, C.WHITE),
                        (sector,   24, C.DIM),
                        (exchange, 10, C.DIM),
                        (str(n_props), 5, C.DIM),
                    )

                # Show all property keys for first company as reference
                if companies:
                    keys = sorted(dict(companies[0]).keys())
                    _info(f"Property keys on first node: {', '.join(keys)}")
        except Exception as exc:
            _err(f"Company query failed: {exc}")

        # ── Chunk nodes (if any) ──────────────────────────────────────────────
        _section("Chunk Nodes (filing data)")
        try:
            total_chunks = s.run("MATCH (c:Chunk) RETURN count(c) AS n").single()["n"]
            if total_chunks == 0:
                _warn("No :Chunk nodes — filing ingestion has not been run yet")
                _info("To ingest: python agents/business_analyst/ingestion.py --help")
            else:
                _ok(f"{total_chunks:,} Chunk nodes total")
                # Per-ticker breakdown
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
                    _table_header(("Ticker", 8), ("Chunks", 8), ("Earliest filing", 16), ("Latest filing", 16))
                    for r in rows:
                        _table_row(
                            (r.get("ticker") or "?", 8, C.CYAN),
                            (f"{r['cnt']:,}", 8, C.WHITE),
                            (str(r.get("earliest") or "—")[:15], 16, C.DIM),
                            (str(r.get("latest") or "—")[:15], 16, C.DIM),
                        )
        except Exception as exc:
            _warn(f"Chunk query failed: {exc}")

        # ── Vector index status ───────────────────────────────────────────────
        _section("Vector Indexes")
        try:
            indexes = s.run(
                "SHOW INDEXES WHERE type = 'VECTOR'"
            ).data()
            if not indexes:
                _warn("No VECTOR indexes found")
                _info("To create: python agents/business_analyst/setup_neo4j_index.py")
            else:
                _table_header(("Name", 24), ("State", 10), ("Label", 12), ("Property", 16), ("Dimension", 10))
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

    # ── Connectivity ──────────────────────────────────────────────────────────
    try:
        r = requests.get(f"{base}/healthz", timeout=5)
        if r.status_code == 200:
            _ok(f"Connected  {base}")
        else:
            # /readyz is the older endpoint
            r2 = requests.get(f"{base}/readyz", timeout=5)
            if r2.status_code == 200:
                _ok(f"Connected  {base}  (via /readyz)")
            else:
                _warn(f"Qdrant responded with HTTP {r.status_code} — may still work")
    except Exception as exc:
        _err(f"Cannot connect to Qdrant at {base}  →  {exc}")
        return

    # ── List collections ──────────────────────────────────────────────────────
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

    _table_header(("Collection", 28), ("Points", 10), ("Vectors", 10), ("Dim", 6), ("Distance", 10), ("Status", 10))
    for col in collections:
        cname = col["name"]
        try:
            det  = requests.get(f"{base}/collections/{cname}", timeout=5).json().get("result", {})
            pts  = det.get("points_count", det.get("vectors_count", 0))
            vecs = det.get("vectors_count", pts)
            stat = det.get("status", "?")

            cfg    = det.get("config", {}).get("params", {}).get("vectors", {})
            # cfg may be a dict of named vectors or a direct config dict
            if isinstance(cfg, dict) and "size" in cfg:
                dim  = cfg.get("size", "?")
                dist = cfg.get("distance", "?")
            elif isinstance(cfg, dict):
                # Named vectors — grab first entry
                first = next(iter(cfg.values()), {}) if cfg else {}
                dim  = first.get("size", "?")
                dist = first.get("distance", "?")
            else:
                dim = dist = "?"

            color = C.GREEN if pts > 0 else C.YELLOW
            _table_row(
                (cname, 28, C.WHITE),
                (f"{pts:,}",  10, color),
                (f"{vecs:,}", 10, C.DIM),
                (str(dim),     6, C.DIM),
                (str(dist),   10, C.DIM),
                (stat,        10, C.DIM),
            )
        except Exception as exc:
            _table_row((cname, 28, C.WHITE), (f"ERROR: {exc}", 40, C.RED), ("","",""), ("","",""), ("","",""), ("","",""))

    # ── Per-collection details ────────────────────────────────────────────────
    for col in collections:
        cname = col["name"]
        _section(f"Collection: {cname}")

        # Collection-level info
        try:
            det = requests.get(f"{base}/collections/{cname}", timeout=5).json().get("result", {})
            pts = det.get("points_count", 0)

            cfg  = det.get("config", {}).get("params", {}).get("vectors", {})
            if isinstance(cfg, dict) and "size" in cfg:
                dim  = cfg.get("size", "?")
                dist = cfg.get("distance", "?")
            elif isinstance(cfg, dict) and cfg:
                first = next(iter(cfg.values()), {})
                dim  = first.get("size", "?")
                dist = first.get("distance", "?")
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

        # ── Payload field discovery (sample 5 points) ─────────────────────────
        try:
            scroll = requests.post(
                f"{base}/collections/{cname}/points/scroll",
                json={"limit": 5, "with_payload": True, "with_vector": False},
                timeout=10,
            ).json().get("result", {}).get("points", [])

            if scroll:
                # Discover all payload keys from the sample
                all_keys: set[str] = set()
                for p in scroll:
                    all_keys.update(p.get("payload", {}).keys())
                _row("Payload fields", ", ".join(sorted(all_keys)), color=C.DIM)
        except Exception as exc:
            _warn(f"Payload discovery failed: {exc}")

        # ── Per-ticker vector counts ───────────────────────────────────────────
        # Try ticker_symbol field first, fall back to ticker
        ticker_field = None
        for candidate in ("ticker_symbol", "ticker"):
            try:
                test = requests.post(
                    f"{base}/collections/{cname}/points/scroll",
                    json={
                        "limit": 1,
                        "with_payload": True,
                        "with_vector": False,
                        "filter": {"must": [{"key": candidate, "match": {"value": "AAPL"}}]},
                    },
                    timeout=5,
                ).json()
                # If no error key in response, field probably exists
                if "error" not in test.get("status", ""):
                    ticker_field = candidate
                    break
            except Exception:
                pass

        if ticker_field:
            _section(f"  Per-Ticker Counts  (field: {ticker_field})")
            tickers_to_check = [ticker_filter] if ticker_filter else [
                "AAPL", "TSLA", "MSFT", "NVDA", "GOOGL",
                "AMZN", "META", "BRK", "JPM", "V",
            ]
            _table_header(("Ticker", 10), ("Vectors", 10), ("Sample title", 50))
            for t in tickers_to_check:
                try:
                    # Count using scroll (Qdrant doesn't have a count-by-filter in all versions)
                    cnt_resp = requests.post(
                        f"{base}/collections/{cname}/points/count",
                        json={"filter": {"must": [{"key": ticker_field, "match": {"value": t}}]}},
                        timeout=5,
                    ).json()
                    count = cnt_resp.get("result", {}).get("count", 0)

                    # Grab one sample title
                    sample_resp = requests.post(
                        f"{base}/collections/{cname}/points/scroll",
                        json={
                            "limit": 1,
                            "with_payload": True,
                            "with_vector": False,
                            "filter": {"must": [{"key": ticker_field, "match": {"value": t}}]},
                        },
                        timeout=5,
                    ).json()
                    pts_list = sample_resp.get("result", {}).get("points", [])
                    sample_title = ""
                    if pts_list:
                        pl = pts_list[0].get("payload", {})
                        sample_title = pl.get("title") or pl.get("data_name") or pl.get("source") or ""
                        sample_title = str(sample_title)[:49]

                    if count > 0:
                        _table_row(
                            (t,            10, C.CYAN),
                            (f"{count:,}", 10, C.GREEN),
                            (sample_title, 50, C.DIM),
                        )
                    # skip tickers with 0 vectors unless explicitly filtered
                    elif ticker_filter:
                        _table_row(
                            (t,  10, C.YELLOW),
                            ("0", 10, C.YELLOW),
                            ("no vectors found", 50, C.DIM),
                        )
                except Exception:
                    pass

        # ── Recent ingestion timestamps ───────────────────────────────────────
        _section(f"  Recent Ingestion  (last 5 points by ingested_at)")
        try:
            recent = requests.post(
                f"{base}/collections/{cname}/points/scroll",
                json={
                    "limit": 5,
                    "with_payload": True,
                    "with_vector": False,
                    "order_by": {"key": "ingested_at", "direction": "desc"},
                },
                timeout=10,
            ).json().get("result", {}).get("points", [])

            if recent:
                _table_header(("ID", 8), ("Ticker", 8), ("Ingested at", 22), ("Source / Title", 36))
                for p in recent:
                    pl = p.get("payload", {})
                    _table_row(
                        (str(p["id"])[:7],                                             8,  C.DIM),
                        (pl.get("ticker_symbol") or pl.get("ticker") or "?",          8,  C.CYAN),
                        (str(pl.get("ingested_at") or "?")[:21],                      22, C.DIM),
                        ((pl.get("title") or pl.get("source") or "?")[:35],           36, C.WHITE),
                    )
            else:
                _info("order_by not supported on this Qdrant version — skipping recency view")
        except Exception:
            _info("Recency view unavailable (order_by may not be supported)")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FYP Database Content Inspector — clean tabular summary of all three databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingestion/db_inspect.py                  # all databases
  python ingestion/db_inspect.py --pg             # PostgreSQL only
  python ingestion/db_inspect.py --neo4j          # Neo4j only
  python ingestion/db_inspect.py --qdrant         # Qdrant only
  python ingestion/db_inspect.py --ticker AAPL    # filter all views to AAPL
  python ingestion/db_inspect.py --qdrant --ticker NVDA
        """,
    )
    parser.add_argument("--pg",     action="store_true", help="PostgreSQL only")
    parser.add_argument("--neo4j",  action="store_true", help="Neo4j only")
    parser.add_argument("--qdrant", action="store_true", help="Qdrant only")
    parser.add_argument("--ticker", default=None, metavar="TICKER",
                        help="Filter per-ticker views to this ticker (e.g. AAPL)")
    args = parser.parse_args()

    run_all = not any([args.pg, args.neo4j, args.qdrant])

    print(f"\n{C.BOLD}{C.WHITE}FYP Database Inspector  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.RESET}")
    if args.ticker:
        print(f"{C.CYAN}  Ticker filter: {args.ticker}{C.RESET}")

    if run_all or args.pg:
        inspect_postgres(args.ticker)
    if run_all or args.neo4j:
        inspect_neo4j(args.ticker)
    if run_all or args.qdrant:
        inspect_qdrant(args.ticker)

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
