#!/usr/bin/env python3
"""
inspect_db.py — Comprehensive database health-check for the FYP pipeline.

Verifies:
  1. PostgreSQL tables — row counts, freshness, per-ticker coverage
  2. pgvector text_chunks — row counts, embedding dimension check
  3. Neo4j — chunk count, vector index status and dimension

Automatically detects whether it runs inside Docker (uses the compose
service hostnames) or on the host (defaults to localhost). Override the
context with `INSPECT_DB_CONTEXT=host|docker` if necessary.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

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
def _inspect_db_context() -> str:
    override = os.getenv("INSPECT_DB_CONTEXT")
    if override in ("host", "docker"):
        return override
    if Path("/.dockerenv").exists():
        return "docker"
    return "host"


INSPECT_DB_CONTEXT = _inspect_db_context()
IN_DOCKER = INSPECT_DB_CONTEXT == "docker"


def _resolve_env_for_context(
    key: str,
    host_default: str,
    docker_default: str,
    docker_aliases: tuple[str, ...] = (",",)
) -> str:
    value = os.getenv(key)
    if IN_DOCKER:
        return value or docker_default
    if value and value not in docker_aliases:
        return value
    return host_default


PG_HOST     = _resolve_env_for_context("POSTGRES_HOST", "localhost", "postgres", ("postgres",))
PG_PORT     = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB       = os.getenv("POSTGRES_DB",       "airflow")
PG_USER     = os.getenv("POSTGRES_USER",     "airflow")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")

NEO4J_URI      = _resolve_env_for_context("NEO4J_URI", "bolt://localhost:7687", "bolt://neo4j:7687", ("bolt://neo4j:7687",))
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


def _pg_column_exists(cur, table: str, column: str) -> bool:
    cur.execute(
        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
        "WHERE table_name = %s AND column_name = %s)",
        (table, column),
    )
    row = cur.fetchone()
    return bool(row[0]) if row else False


def _pg_sample_rows(cur, table: str, where: str = "", limit: int = 2) -> list[tuple]:
    """Fetch up to `limit` sample rows from a table, optionally filtered."""
    sql = f"SELECT * FROM {table}"
    if where:
        sql += f" WHERE {where}"
    sql += f" ORDER BY 1 DESC LIMIT {limit}"
    try:
        cur.execute(sql)
        return cur.fetchall()
    except Exception:
        return []


def _print_sample_rows(cur, table: str, where: str = "", label: str = "") -> None:
    """Print 2 sample rows from a table if any exist."""
    rows = _pg_sample_rows(cur, table, where=where, limit=2)
    if not rows:
        return
    tag = label or table
    for i, row in enumerate(rows, 1):
        # Truncate long values for readability
        display = tuple(
            str(v)[:80] + "…" if isinstance(v, str) and len(str(v)) > 80 else v
            for v in row
        )
        print(f"    sample {i}: {display}")


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
        # sentiment_trends is now populated from real per-article news aggregation
        # (pos/neg/neu averaged per day), NOT from the old /api/sentiments endpoint.
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
                    _print_sample_rows(cur, table)
                elif total >= min_rows:
                    missing = [t for t in TRACKED_TICKERS if t not in covered]
                    print(_warn(f"{table}: {total} rows, but missing tickers: {missing}"))
                    _print_sample_rows(cur, table)
                    all_pass = False
                else:
                    print(_fail(f"{table}: only {total} rows (expected >= {min_rows})"))
                    all_pass = False
            except Exception as exc:
                conn.rollback()
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
                _print_sample_rows(cur, "dividends_history")
            elif total >= 10:
                print(_warn(f"dividends_history: {total} rows, missing tickers (may not pay dividends): {missing}"))
                _print_sample_rows(cur, "dividends_history")
            else:
                print(_warn(f"dividends_history: only {total} rows"))
        except Exception as exc:
            conn.rollback()
            print(_warn(f"dividends_history: {exc}"))

        # ── market_eod_us (global table, no per-ticker) ──────────────────────
        try:
            n = _pg_count(cur, "market_eod_us")
            if n >= 100:
                print(_ok(f"market_eod_us: {n} rows (S&P 500 benchmark)"))
                _print_sample_rows(cur, "market_eod_us")
            else:
                print(_warn(f"market_eod_us: only {n} rows (expected >= 100)"))
                all_pass = False
        except Exception as exc:
            conn.rollback()
            print(_fail(f"market_eod_us: {exc}"))
            all_pass = False

        # ── Optional / expected-empty tables ────────────────────────────────
        for table in ["insider_transactions", "institutional_holders",
                      "short_interest", "earnings_surprises"]:
            try:
                n = _pg_count(cur, table)
                if n > 0:
                    print(_ok(f"{table}: {n} rows"))
                    _print_sample_rows(cur, table)
                else:
                    print(_warn(f"{table}: 0 rows (optional — may be empty if not scraped)"))
            except Exception as exc:
                conn.rollback()
                print(_warn(f"{table}: {exc}"))

        # ── raw_fundamentals (FMP-sourced, may be empty) ─────────────────────
        try:
            n = _pg_count(cur, "raw_fundamentals")
            if n > 0:
                print(_ok(f"raw_fundamentals: {n} rows"))
                _print_sample_rows(cur, "raw_fundamentals")
            else:
                print(_warn("raw_fundamentals: 0 rows (FMP DAG not ingested — acceptable)"))
        except Exception as exc:
            conn.rollback()
            print(_warn(f"raw_fundamentals: {exc}"))

        # ── news_articles (EODHD /api/news, full content + pgvector embedding) ──────
        try:
            total = _pg_count(cur, "news_articles")
            counts = _pg_ticker_counts(cur, "news_articles", "ticker")
            covered = [t for t in TRACKED_TICKERS if counts.get(t, 0) > 0]
            missing = [t for t in TRACKED_TICKERS if t not in covered]
            embedded_count = None
            if _pg_column_exists(cur, "news_articles", "embedding"):
                # Check embedding coverage only when the schema has this column.
                cur.execute(
                    "SELECT COUNT(*) FROM news_articles WHERE embedding IS NOT NULL"
                )
                _emb_row = cur.fetchone()
                embedded_count = _emb_row[0] if _emb_row else 0
            if total > 0 and not missing:
                embed_note = ""
                if embedded_count is not None:
                    embed_pct = round(100 * embedded_count / total) if total else 0
                    embed_note = f", {embedded_count}/{total} embedded ({embed_pct}%)"
                print(_ok(f"news_articles: {total} total rows, all {len(TRACKED_TICKERS)} tickers covered{embed_note}"))
                _print_sample_rows(cur, "news_articles")
            elif total > 0:
                print(_warn(f"news_articles: {total} rows, missing tickers: {missing}"))
                _print_sample_rows(cur, "news_articles")
                all_pass = False
            else:
                print(_fail("news_articles: 0 rows (expected >= 1 per ticker)"))
                all_pass = False
        except Exception as exc:
            conn.rollback()
            print(_fail(f"news_articles: query error — {exc}"))
            all_pass = False

        # ── news_word_weights (EODHD /api/news-word-weights, keyed by ticker+date+word) ──
        # 0 rows is acceptable if the DAG hasn't run the news-word-weights scrape yet.
        try:
            total = _pg_count(cur, "news_word_weights")
            counts = _pg_ticker_counts(cur, "news_word_weights", "ticker")
            covered = [t for t in TRACKED_TICKERS if counts.get(t, 0) > 0]
            if total > 0 and len(covered) == len(TRACKED_TICKERS):
                print(_ok(f"news_word_weights: {total} total rows, all {len(TRACKED_TICKERS)} tickers covered"))
                _print_sample_rows(cur, "news_word_weights")
            elif total > 0:
                missing = [t for t in TRACKED_TICKERS if t not in covered]
                print(_warn(f"news_word_weights: {total} rows, missing tickers: {missing}"))
                _print_sample_rows(cur, "news_word_weights")
            else:
                print(_warn("news_word_weights: 0 rows (DAG scrape not yet run for this endpoint)"))
        except Exception as exc:
            conn.rollback()
            print(_warn(f"news_word_weights: query error — {exc}"))

        # ── text_chunks (pgvector) ────────────────────────────────────────────
        check_pgvector(cur)

        # ── Macro tables coverage ──────────────────────────────────────────────
        print("\n--- Macro Data ---")
        macro_tables = [
            ("treasury_rates", 1),
            ("global_macro_indicators", 1),
            ("economic_events", 1),
            ("market_screener", 1),
            ("forex_rates", 1),
            ("market_eod_us", 1),
        ]
        for table, min_rows in macro_tables:
            try:
                n = _pg_count(cur, table)
                if n >= min_rows:
                    print(_ok(f"{table}: {n} rows"))
                    _print_sample_rows(cur, table)
                else:
                    print(_warn(f"{table}: {n} rows (expected >= {min_rows})"))
                    all_pass = False
            except Exception as exc:
                conn.rollback()
                print(_warn(f"{table}: {exc}"))
                all_pass = False

        # ── Textual document metadata coverage ─────────────────────────────────
        print("\n--- Textual Documents ---")
        textual_doc_types = ["earnings_call", "broker_report", "macro_report"]
        for doc_type in textual_doc_types:
            try:
                cur.execute(
                    "SELECT COUNT(*) FROM textual_documents WHERE doc_type = %s",
                    (doc_type,),
                )
                row = cur.fetchone()
                count = int(row[0]) if row else 0
                if count > 0:
                    print(_ok(f"textual_documents[{doc_type}]: {count} rows"))
                else:
                    print(_warn(f"textual_documents[{doc_type}]: 0 rows"))
                    all_pass = False
            except Exception as exc:
                conn.rollback()
                print(_warn(f"textual_documents[{doc_type}]: {exc}"))
                all_pass = False

        # ── Feedback tables (RLAIF + User Feedback) ─────────────────────────
        check_feedback_tables(cur)

    conn.close()
    return all_pass


def check_feedback_tables(cur) -> bool:
    """Check RLAIF and User Feedback tables."""
    print("\n--- Feedback Tables ---")
    all_pass = True

    feedback_tables = [
        ("rl_feedback", "RLAIF scores from AI judge"),
        ("user_feedback", "Explicit user ratings from UI"),
        ("prompt_versions", "Prompt version tracking for A/B testing"),
    ]

    for table, description in feedback_tables:
        try:
            # Check if table exists
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_name = %s)",
                (table,)
            )
            exists = cur.fetchone()[0]

            if not exists:
                print(_warn(f"{table}: table does not exist (will be created on first use)"))
                continue

            # Get row count
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]

            if count > 0:
                print(_ok(f"{table}: {count} rows ({description})"))
                _print_sample_rows(cur, table)
            else:
                print(_warn(f"{table}: 0 rows (no feedback recorded yet)"))

            # Check specific columns exist
            expected_cols = {
                "rl_feedback": ["run_id", "user_query", "overall_score", "agent_blamed", "factual_accuracy"],
                "user_feedback": ["run_id", "helpful", "comment", "issue_tags"],
                "prompt_versions": ["agent_name", "version", "prompt_text", "deployed_at"],
            }

            if table in expected_cols:
                for col in expected_cols[table]:
                    cur.execute(
                        "SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                        "WHERE table_name = %s AND column_name = %s)",
                        (table, col)
                    )
                    col_exists = cur.fetchone()[0]
                    if not col_exists:
                        print(_fail(f"{table}: missing column {col}"))
                        all_pass = False

        except Exception as exc:
            print(_warn(f"{table}: {exc}"))

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
                _print_sample_rows(cur, "text_chunks", where=f"ticker = '{ticker}'")
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

            # Per-ticker chunk coverage - breakdown by section
            result = session.run(
                "MATCH (c:Company)-[:HAS_CHUNK]->(ch:Chunk) "
                "RETURN c.ticker AS ticker, ch.section AS section, count(ch) AS n "
                "ORDER BY ticker, section"
            ).data()
            if result:
                # Group by ticker
                ticker_chunks = {}
                for row in result:
                    t = row['ticker']
                    s = row['section']
                    n = row['n']
                    if t not in ticker_chunks:
                        ticker_chunks[t] = {}
                    ticker_chunks[t][s] = n
                
                for ticker in sorted(ticker_chunks.keys()):
                    sections = ticker_chunks[ticker]
                    total = sum(sections.values())
                    # Show breakdown
                    breakdown = ", ".join([f"{k}:{v}" for k, v in sections.items()])
                    print(_ok(f"  Neo4j chunks [{ticker}]: {total} total ({breakdown})"))
            else:
                print(_warn("No Company->Chunk relationships found"))

            # Check for textual document types specifically
            print("\n--- Textual Document Coverage ---")
            for section in ['earnings_call', 'broker_report', 'macro_report']:
                query: Any = (
                    f"MATCH (c:Company)-[:HAS_CHUNK]->(ch:Chunk {{section: '{section}'}}) "
                    "RETURN c.ticker AS ticker, count(ch) AS n ORDER BY ticker"
                )
                result = session.run(query).data()
                if result and sum(r['n'] for r in result) > 0:
                    total = sum(r['n'] for r in result)
                    tickers_covered = [r['ticker'] for r in result if r['n'] > 0]
                    print(_ok(f"{section}: {total} chunks across {len(tickers_covered)} tickers ({', '.join(tickers_covered)})"))
                else:
                    print(_warn(f"{section}: 0 chunks (run textual ingestion scripts)"))

            print("\n--- Neo4j News Coverage ---")
            result = session.run("MATCH (n:NewsArticle) RETURN count(n) AS n").single()
            n_news = result["n"] if result else 0
            if n_news > 0:
                print(_ok(f"NewsArticle nodes: {n_news}"))
            else:
                print(_warn("NewsArticle nodes: 0 (run load_neo4j_for_ticker with financial_news.json present)"))

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
