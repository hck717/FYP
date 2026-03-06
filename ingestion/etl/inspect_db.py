#!/usr/bin/env python3
"""
inspect_db.py — Comprehensive database health-check for the FYP pipeline.

Verifies:
  1. PostgreSQL tables — row counts, freshness, per-ticker coverage
  2. pgvector text_chunks — row counts, embedding dimension check
  3. Neo4j — chunk count, vector index status and dimension

Usage (local):
    POSTGRES_HOST=localhost python ingestion/etl/inspect_db.py

Usage (in Airflow container):
    python /opt/airflow/etl/inspect_db.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ── Load .env for local runs ──────────────────────────────────────────────────
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ── Config ─────────────────────────────────────────────────────────────────────
PG_HOST     = os.getenv("POSTGRES_HOST",     "localhost")
PG_PORT     = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB       = os.getenv("POSTGRES_DB",       "airflow")
PG_USER     = os.getenv("POSTGRES_USER",     "airflow")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "SecureNeo4jPass2025!")

TRACKED_TICKERS = [t.strip() for t in os.getenv("TRACKED_TICKERS", "AAPL,TSLA,NVDA,MSFT,GOOGL").split(",")]

# ── Colour helpers ─────────────────────────────────────────────────────────────
_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"

def _ok(msg: str)   -> str: return f"{_GREEN}  PASS{_RESET}  {msg}"
def _fail(msg: str) -> str: return f"{_RED}  FAIL{_RESET}  {msg}"
def _warn(msg: str) -> str: return f"{_YELLOW}  WARN{_RESET}  {msg}"


# ── PostgreSQL checks ──────────────────────────────────────────────────────────

def _pg_connect():
    import psycopg2
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        dbname=PG_DB, user=PG_USER, password=PG_PASSWORD,
    )


def _pg_count(cur, table: str, where: str = "") -> int:
    sql = f"SELECT COUNT(*) FROM {table}"
    if where:
        sql += f" WHERE {where}"
    cur.execute(sql)
    return cur.fetchone()[0]


def _pg_ticker_counts(cur, table: str, ticker_col: str = "ticker") -> dict[str, int]:
    cur.execute(
        f"SELECT {ticker_col}, COUNT(*) FROM {table} GROUP BY {ticker_col} ORDER BY {ticker_col}"
    )
    return {row[0]: row[1] for row in cur.fetchall()}


def check_postgres() -> bool:
    print("\n" + "=" * 60)
    print("PostgreSQL checks")
    print("=" * 60)
    all_pass = True
    try:
        conn = _pg_connect()
    except Exception as exc:
        print(_fail(f"Cannot connect to PostgreSQL: {exc}"))
        return False

    with conn, conn.cursor() as cur:
        # ── Core timeseries tables ───────────────────────────────────────────
        for table, ticker_col, min_rows in [
            ("raw_timeseries",         "ticker_symbol", 1000),
            ("financial_statements",   "ticker",        100),
            ("valuation_metrics",      "ticker",        5),
            ("sentiment_trends",       "ticker",        1),
        ]:
            try:
                total = _pg_count(cur, table)
                counts = _pg_ticker_counts(cur, table, ticker_col)
                covered = [t for t in TRACKED_TICKERS if counts.get(t, 0) > 0]
                if total >= min_rows and len(covered) == len(TRACKED_TICKERS):
                    print(_ok(f"{table}: {total} total rows, all {len(TRACKED_TICKERS)} tickers covered"))
                elif total >= min_rows:
                    missing = [t for t in TRACKED_TICKERS if t not in covered]
                    print(_warn(f"{table}: {total} rows, but missing tickers: {missing}"))
                    all_pass = False
                else:
                    print(_fail(f"{table}: only {total} rows (expected >= {min_rows})"))
                    all_pass = False
            except Exception as exc:
                print(_fail(f"{table}: query error — {exc}"))
                all_pass = False

        # dividends_history: warn-only if some tickers missing (e.g. TSLA pays no dividends)
        try:
            total = _pg_count(cur, "dividends_history")
            counts = _pg_ticker_counts(cur, "dividends_history", "ticker")
            covered = [t for t in TRACKED_TICKERS if counts.get(t, 0) > 0]
            missing = [t for t in TRACKED_TICKERS if t not in covered]
            if total >= 10 and not missing:
                print(_ok(f"dividends_history: {total} total rows, all {len(TRACKED_TICKERS)} tickers covered"))
            elif total >= 10:
                print(_warn(f"dividends_history: {total} rows, missing tickers (may not pay dividends): {missing}"))
            else:
                print(_warn(f"dividends_history: only {total} rows"))
        except Exception as exc:
            print(_warn(f"dividends_history: {exc}"))

        # ── market_eod_us (global table, no per-ticker) ──────────────────────
        try:
            n = _pg_count(cur, "market_eod_us")
            if n >= 100:
                print(_ok(f"market_eod_us: {n} rows (S&P 500 benchmark)"))
            else:
                print(_warn(f"market_eod_us: only {n} rows (expected >= 100)"))
                all_pass = False
        except Exception as exc:
            print(_fail(f"market_eod_us: {exc}"))
            all_pass = False

        # ── Optional / expected-empty tables ────────────────────────────────
        for table in ["insider_transactions", "institutional_holders",
                      "short_interest", "earnings_surprises"]:
            try:
                n = _pg_count(cur, table)
                if n > 0:
                    print(_ok(f"{table}: {n} rows"))
                else:
                    print(_warn(f"{table}: 0 rows (optional — may be empty if not scraped)"))
            except Exception as exc:
                print(_warn(f"{table}: {exc}"))

        # ── raw_fundamentals (FMP-sourced, may be empty) ─────────────────────
        try:
            n = _pg_count(cur, "raw_fundamentals")
            if n > 0:
                print(_ok(f"raw_fundamentals: {n} rows"))
            else:
                print(_warn("raw_fundamentals: 0 rows (FMP DAG not ingested — acceptable)"))
        except Exception as exc:
            print(_warn(f"raw_fundamentals: {exc}"))

        # ── text_chunks (pgvector) ────────────────────────────────────────────
        check_pgvector(cur)

    conn.close()
    return all_pass


def check_pgvector(cur) -> bool:
    print("\n--- pgvector text_chunks ---")
    all_pass = True

    # Check extension
    try:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
        if row:
            print(_ok(f"pgvector extension installed (version {row[0]})"))
        else:
            print(_fail("pgvector extension NOT installed — run: CREATE EXTENSION vector"))
            return False
    except Exception as exc:
        print(_fail(f"pgvector extension check failed: {exc}"))
        return False

    # Check table exists
    try:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'text_chunks')"
        )
        exists = cur.fetchone()[0]
        if not exists:
            print(_fail("text_chunks table does not exist"))
            return False
        print(_ok("text_chunks table exists"))
    except Exception as exc:
        print(_fail(f"text_chunks table check: {exc}"))
        return False

    # Check embedding column
    try:
        cur.execute(
            "SELECT udt_name FROM information_schema.columns "
            "WHERE table_name = 'text_chunks' AND column_name = 'embedding'"
        )
        col = cur.fetchone()
        if col:
            print(_ok(f"text_chunks.embedding column: type={col[0]}"))
        else:
            print(_warn("text_chunks.embedding column not found (migration not run yet)"))
    except Exception as exc:
        print(_warn(f"embedding column check: {exc}"))

    # Row counts per ticker
    try:
        cur.execute(
            "SELECT ticker, COUNT(*), "
            "SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) AS embedded "
            "FROM text_chunks GROUP BY ticker ORDER BY ticker"
        )
        rows = cur.fetchall()
        if rows:
            for ticker, total, embedded in rows:
                if embedded == total:
                    print(_ok(f"text_chunks [{ticker}]: {total} chunks, all embedded"))
                else:
                    print(_warn(f"text_chunks [{ticker}]: {total} chunks, {embedded} embedded ({total - embedded} missing)"))
        else:
            print(_warn("text_chunks: 0 rows — run DAG or load_postgres.py to populate"))
            all_pass = False
    except Exception as exc:
        print(_fail(f"text_chunks row count: {exc}"))
        all_pass = False

    # HNSW index
    try:
        cur.execute(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE tablename = 'text_chunks' AND indexdef LIKE '%hnsw%'"
        )
        idx = cur.fetchone()
        if idx:
            print(_ok(f"text_chunks HNSW index: {idx[0]}"))
        else:
            print(_warn("text_chunks HNSW index not found"))
    except Exception as exc:
        print(_warn(f"HNSW index check: {exc}"))

    return all_pass


# ── Neo4j checks ───────────────────────────────────────────────────────────────

def check_neo4j() -> bool:
    print("\n" + "=" * 60)
    print("Neo4j checks")
    print("=" * 60)
    all_pass = True
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except Exception as exc:
        print(_fail(f"Cannot import neo4j or create driver: {exc}"))
        return False

    try:
        with driver.session() as session:
            # Company nodes
            result = session.run("MATCH (c:Company) RETURN count(c) AS n").single()
            n_companies = result["n"] if result else 0
            if n_companies >= len(TRACKED_TICKERS):
                print(_ok(f"Company nodes: {n_companies}"))
            else:
                print(_fail(f"Company nodes: {n_companies} (expected >= {len(TRACKED_TICKERS)})"))
                all_pass = False

            # Chunk nodes
            result = session.run("MATCH (ch:Chunk) RETURN count(ch) AS n").single()
            n_chunks = result["n"] if result else 0
            if n_chunks > 0:
                print(_ok(f"Chunk nodes: {n_chunks}"))
            else:
                print(_warn("Chunk nodes: 0 — run load_neo4j.py to populate"))

            # Chunk embedding dimension
            result = session.run(
                "MATCH (ch:Chunk) WHERE ch.embedding IS NOT NULL "
                "RETURN size(ch.embedding) AS dim LIMIT 1"
            ).single()
            if result:
                dim = result["dim"]
                if dim == 768:
                    print(_ok(f"Chunk embedding dimension: {dim} (nomic-embed-text ✓)"))
                elif dim == 384:
                    print(_fail(f"Chunk embedding dimension: {dim} (WRONG — still 384-dim, need re-embed at 768)"))
                    all_pass = False
                else:
                    print(_warn(f"Chunk embedding dimension: {dim} (unexpected)"))
            else:
                print(_warn("No embedded chunks found in Neo4j"))

            # Vector index
            result = session.run(
                "SHOW INDEXES YIELD name, state, type, labelsOrTypes, properties, options "
                "WHERE name = 'chunk_embedding'"
            ).data()
            if result:
                idx = result[0]
                state = idx.get("state", "UNKNOWN")
                opts = idx.get("options", {}) or {}
                config = opts.get("indexConfig", {}) or {}
                dim_cfg = config.get("vector.dimensions", "?")
                if state == "ONLINE" and dim_cfg == 768:
                    print(_ok(f"Neo4j vector index 'chunk_embedding': ONLINE, dim={dim_cfg}"))
                elif state == "ONLINE" and dim_cfg != 768:
                    print(_fail(f"Neo4j vector index 'chunk_embedding': ONLINE but dim={dim_cfg} (need 768)"))
                    all_pass = False
                else:
                    print(_warn(f"Neo4j vector index 'chunk_embedding': state={state}, dim={dim_cfg}"))
            else:
                print(_fail("Neo4j vector index 'chunk_embedding' NOT FOUND"))
                all_pass = False

            # Per-ticker chunk coverage
            result = session.run(
                "MATCH (c:Company)-[:HAS_CHUNK]->(ch:Chunk) "
                "RETURN c.ticker AS ticker, count(ch) AS n ORDER BY ticker"
            ).data()
            if result:
                for row in result:
                    print(_ok(f"  Neo4j chunks [{row['ticker']}]: {row['n']}"))
            else:
                print(_warn("No Company->Chunk relationships found"))

    except Exception as exc:
        print(_fail(f"Neo4j query failed: {exc}"))
        all_pass = False
    finally:
        driver.close()

    return all_pass


# ── Summary ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("FYP Database Health Check")
    print(f"Tickers: {TRACKED_TICKERS}")
    print("=" * 60)

    pg_ok    = check_postgres()
    neo4j_ok = check_neo4j()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print((_ok if pg_ok    else _fail)("PostgreSQL (including pgvector)"))
    print((_ok if neo4j_ok else _fail)("Neo4j"))

    if pg_ok and neo4j_ok:
        print(f"\n{_GREEN}All checks passed.{_RESET}")
        sys.exit(0)
    else:
        print(f"\n{_RED}Some checks failed — see above for details.{_RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
