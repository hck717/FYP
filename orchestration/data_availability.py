"""Dynamic data availability checker for the multi-agent equity research pipeline.

This module is the single source of truth for what data is actually present in every
backend at query time.  It is called:
  - by the orchestration planner (node_planner) to annotate the plan with availability
  - by the BA / QF nodes to skip dead code-paths rather than trying and failing noisily
  - by the summarizer to add a DATA AVAILABILITY NOTICE when any tier is degraded

Design goals
────────────
1. **Fast** — all checks are a single lightweight query per backend (no full table scans).
2. **Non-blocking** — every check is individually try/excepted; one backend being down
   never prevents the others from being reported.
3. **Cacheable** — results are cached per request (not globally) via the OrchestrationState.
4. **Actionable** — the returned dict uses human-readable tier names and boolean flags so
   each agent node can branch on them without string parsing.

Returned structure
──────────────────
{
  "neo4j": {
    "reachable": bool,
    "company_nodes": int,                # total :Company nodes
    "chunk_nodes": int,                  # total :Chunk nodes
    "chunk_index_exists": bool,          # chunk_embedding vector index present
    "tickers_with_chunks": [str, ...],   # tickers that have >=1 :Chunk node
  },
  "postgres": {
    "reachable": bool,
    "sentiment_rows": int,               # rows in sentiment_trends
    "fundamentals_tickers": [str, ...],  # distinct ticker_symbols in raw_fundamentals
    "timeseries_tickers": [str, ...],    # distinct ticker_symbols in raw_timeseries
    "market_eod_rows": int,              # rows in market_eod_us
  },
  "deepseek": {
    "reachable": bool,
    "available_models": [str, ...],
    "embedding_model_ready": bool,       # sentence-transformers (local)
    "llm_model_ready": bool,             # deepseek-reasoner or configured model
  },
  "tickers_fully_ready": [str, ...],     # tickers with neo4j chunks + pg fundamentals
  "tickers_partially_ready": [str, ...], # tickers with >=1 data source
  "degraded_tiers": [str, ...],          # list of tier names that are down/empty
  "summary": str,                        # human-readable one-liner for logs / UI
}
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Load .env so env vars are available even when not set in the shell (e.g.
# when this module is imported before any agent has had a chance to call
# load_dotenv itself).
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed — rely on env vars being set externally

# Determine if running inside Docker
_IN_DOCKER = Path("/.dockerenv").exists()

# ── defaults (mirror agents/business_analyst/config.py) ────────────────────
_NEO4J_URI      = os.getenv("NEO4J_URI",            "bolt://localhost:7687" if not _IN_DOCKER else "bolt://neo4j:7687")
_NEO4J_USER     = os.getenv("NEO4J_USER",            "neo4j")
_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD",        "SecureNeo4jPass2025!")
_PG_HOST        = os.getenv("POSTGRES_HOST",         "localhost" if not _IN_DOCKER else "postgres")
_PG_PORT        = int(os.getenv("POSTGRES_PORT",     "5432"))
_PG_DB          = os.getenv("POSTGRES_DB",           "airflow")
_PG_USER        = os.getenv("POSTGRES_USER",         "airflow")
_PG_PASS        = os.getenv("POSTGRES_PASSWORD",     "airflow")
_DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
_LLM_MODEL      = os.getenv("LLM_MODEL_BUSINESS_ANALYST", "deepseek-v4-pro")
_EMBED_MODEL    = os.getenv("EMBEDDING_MODEL",       "nomic-embed-text:v1.5")

# DeepSeek model mapping for compatibility
_DEEPSEEK_MODELS = {
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-chat",  # fallback if reasoner not available
    "deepseek-v4-pro": "deepseek-chat",    # fallback if v4-pro not available
}

_SUPPORTED_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]


# ── Individual tier checkers ─────────────────────────────────────────────────

def _check_neo4j() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "reachable": False,
        "company_nodes": 0,
        "chunk_nodes": 0,
        "chunk_index_exists": False,
        "tickers_with_chunks": [],
        "error": None,
    }
    try:
        from neo4j import GraphDatabase, basic_auth  # type: ignore[import]
        driver = GraphDatabase.driver(
            _NEO4J_URI,
            auth=basic_auth(_NEO4J_USER, _NEO4J_PASSWORD),
            encrypted=False,
            connection_timeout=5,
            max_connection_lifetime=10,
        )
        with driver.session(database=None) as session:
            # Company count
            rec = session.run("MATCH (c:Company) RETURN count(c) AS n").single()
            result["company_nodes"] = rec["n"] if rec else 0

            # Chunk count
            rec = session.run("MATCH (ch:Chunk) RETURN count(ch) AS n").single()
            result["chunk_nodes"] = rec["n"] if rec else 0

            # Tickers that have chunks
            rows = session.run(
                "MATCH (c:Company)-[:HAS_CHUNK]->(ch:Chunk) "
                "RETURN DISTINCT c.ticker AS t ORDER BY t"
            ).data()
            result["tickers_with_chunks"] = [r["t"] for r in rows if r.get("t")]

            # Vector index presence
            idx_rows = session.run(
                "SHOW INDEXES YIELD name, type WHERE name = 'chunk_embedding' RETURN name"
            ).data()
            result["chunk_index_exists"] = bool(idx_rows)

        driver.close()
        result["reachable"] = True
    except Exception as exc:
        result["error"] = str(exc)
        logger.debug("Neo4j availability check failed: %s", exc)
    return result


def _check_postgres() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "reachable": False,
        "sentiment_rows": 0,
        "fundamentals_tickers": [],
        "timeseries_tickers": [],
        "market_eod_rows": 0,
        "error": None,
    }
    try:
        import psycopg2  # type: ignore[import]
        conn = psycopg2.connect(
            host=_PG_HOST, port=_PG_PORT, dbname=_PG_DB,
            user=_PG_USER, password=_PG_PASS, connect_timeout=5,
        )
        result["reachable"] = True

        with conn.cursor() as cur:
            # sentiment_trends
            try:
                cur.execute("SELECT COUNT(*) FROM sentiment_trends")
                row = cur.fetchone()
                result["sentiment_rows"] = row[0] if row else 0
            except Exception:
                pass

            # financial_statements — which tickers are present? (used as "fundamentals" proxy)
            try:
                cur.execute(
                    "SELECT DISTINCT ticker FROM financial_statements "
                    "WHERE ticker = ANY(%s)",
                    (_SUPPORTED_TICKERS,)
                )
                result["fundamentals_tickers"] = [r[0] for r in cur.fetchall()]
            except Exception:
                pass

            # raw_timeseries — which tickers are present?
            try:
                cur.execute(
                    "SELECT DISTINCT ticker_symbol FROM raw_timeseries "
                    "WHERE ticker_symbol = ANY(%s)",
                    (_SUPPORTED_TICKERS,)
                )
                result["timeseries_tickers"] = [r[0] for r in cur.fetchall()]
            except Exception:
                pass

            # market_eod_us
            try:
                cur.execute("SELECT COUNT(*) FROM market_eod_us")
                row = cur.fetchone()
                result["market_eod_rows"] = row[0] if row else 0
            except Exception:
                pass

        conn.close()
    except Exception as exc:
        result["error"] = str(exc)
        logger.debug("PostgreSQL availability check failed: %s", exc)
    return result


def _check_deepseek() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "reachable": False,
        "available_models": [],
        "embedding_model_ready": False,
        "llm_model_ready": False,
        "error": None,
    }
    # Re-read at call time so .env loaded after module import is respected
    _DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    if not _DEEPSEEK_API_KEY:
        result["error"] = "DEEPSEEK_API_KEY not set"
        logger.debug("DeepSeek API key not configured")
        return result
    
    try:
        headers = {
            "Authorization": f"Bearer {_DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        resp = requests.get(
            f"{_DEEPSEEK_BASE_URL}/v1/models",
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        result["reachable"] = True
        result["available_models"] = [m["id"] for m in resp.json().get("data", [])]
        
        # Check if configured model or its fallback is available
        check_model = _DEEPSEEK_MODELS.get(_LLM_MODEL, _LLM_MODEL)
        result["llm_model_ready"] = any(
            check_model in m for m in result["available_models"]
        ) or any(
            _LLM_MODEL in m for m in result["available_models"]
        )
    except Exception as exc:
        result["error"] = str(exc)
        logger.debug("DeepSeek availability check failed: %s", exc)

    try:
        from sentence_transformers import SentenceTransformer
        result["embedding_model_ready"] = True
    except ImportError:
        result["embedding_model_ready"] = False

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def check_all(tickers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run all tier checks and return the consolidated availability report.

    All three backend checks (Neo4j, PostgreSQL, DeepSeek) are executed
    concurrently via a ThreadPoolExecutor so one slow backend does not block
    the others.  Total wall-clock time is bounded by the slowest single check
    rather than their sum.

    Parameters
    ----------
    tickers:
        Specific tickers to assess readiness for.  Defaults to all 5 supported tickers.
    """
    tickers = tickers or _SUPPORTED_TICKERS

    # Timeout for each backend check.  Each individual connector already has a
    # 5-second socket connect_timeout, but DNS stalls or OS TCP retries can
    # still cause the thread to hang well beyond that.  This hard ceiling
    # ensures check_all() always returns within ~15 seconds even when a backend
    # is completely unreachable.
    _CHECK_TIMEOUT = 10  # seconds per future

    # Run all three checks concurrently
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="avail_check") as pool:
        fut_neo4j    = pool.submit(_check_neo4j)
        fut_postgres = pool.submit(_check_postgres)
        fut_deepseek = pool.submit(_check_deepseek)

        def _safe_result(fut, name: str) -> Dict[str, Any]:
            """Return future result or a safe error dict if it times out."""
            from concurrent.futures import TimeoutError as FutTimeoutError
            try:
                return fut.result(timeout=_CHECK_TIMEOUT)
            except FutTimeoutError:
                logger.warning(
                    "[data_availability] %s check timed out after %ds — treating as unreachable.",
                    name, _CHECK_TIMEOUT,
                )
                return {"reachable": False, "error": f"timeout after {_CHECK_TIMEOUT}s"}
            except Exception as exc:
                logger.warning("[data_availability] %s check raised: %s", name, exc)
                return {"reachable": False, "error": str(exc)}

        neo4j    = _safe_result(fut_neo4j,    "neo4j")
        postgres = _safe_result(fut_postgres, "postgres")
        deepseek = _safe_result(fut_deepseek, "deepseek")

    # Derive per-ticker readiness
    tickers_with_chunks  = set(neo4j.get("tickers_with_chunks") or [])
    tickers_with_pg_fund = set(postgres.get("fundamentals_tickers") or [])

    fully_ready:   List[str] = []
    partially_ready: List[str] = []
    for t in tickers:
        has_ba = t in tickers_with_chunks
        has_qf = t in tickers_with_pg_fund
        if has_ba and has_qf:
            fully_ready.append(t)
        elif has_ba or has_qf:
            partially_ready.append(t)

    # Degrade tier list
    degraded: List[str] = []
    if not neo4j["reachable"]:
        degraded.append("neo4j_unreachable")
    elif neo4j["chunk_nodes"] == 0:
        degraded.append("neo4j_no_chunks")

    if not postgres["reachable"]:
        degraded.append("postgres_unreachable")
    elif not postgres["fundamentals_tickers"]:
        degraded.append("postgres_no_fundamentals")

    if not deepseek["reachable"]:
        degraded.append("deepseek_unreachable")
    elif not deepseek["llm_model_ready"]:
        degraded.append(f"deepseek_missing_llm_{_LLM_MODEL}")

    # Human summary
    if not degraded:
        summary = (
            f"All tiers healthy. "
            f"Fully ready: {fully_ready}. "
            f"Neo4j chunks: {neo4j['chunk_nodes']}, "
            f"PG fundamentals tickers: {len(tickers_with_pg_fund)}."
        )
    else:
        summary = (
            f"Degraded tiers: {degraded}. "
            f"Fully ready tickers: {fully_ready}. "
            f"Partially ready: {partially_ready}."
        )

    return {
        "neo4j":    neo4j,
        "postgres": postgres,
        "deepseek": deepseek,
        "tickers_fully_ready":    fully_ready,
        "tickers_partially_ready": partially_ready,
        "degraded_tiers":         degraded,
        "summary":                summary,
    }


def availability_notice(avail: Dict[str, Any], tickers: List[str]) -> str:
    """Return a short plain-text notice for the summarizer prompt when data is degraded.

    Returns an empty string when everything is healthy.
    """
    degraded = avail.get("degraded_tiers") or []
    if not degraded:
        return ""

    lines = ["⚠ DATA AVAILABILITY NOTICE (auto-generated)"]

    neo4j = avail.get("neo4j") or {}
    if "neo4j_unreachable" in degraded:
        lines.append("  • Neo4j knowledge graph is UNREACHABLE — no qualitative chunks available.")
    elif "neo4j_no_chunks" in degraded:
        chunks_with = neo4j.get("tickers_with_chunks") or []
        missing = [t for t in tickers if t not in chunks_with]
        lines.append(
            f"  • Neo4j Chunk nodes missing for: {missing or 'all requested tickers'}. "
            "Qualitative CRAG retrieval will score 0 for these tickers."
        )

    postgres = avail.get("postgres") or {}
    if "postgres_unreachable" in degraded:
        lines.append("  • PostgreSQL is UNREACHABLE — no quantitative fundamentals available.")
    elif "postgres_no_fundamentals" in degraded:
        lines.append(
            "  • PostgreSQL raw_fundamentals table is EMPTY — "
            "all P/E, ROE, and factor computations will return null."
        )

    deepseek = avail.get("deepseek") or {}
    if "deepseek_unreachable" in degraded:
        lines.append(
            f"  • DeepSeek API is UNREACHABLE at {_DEEPSEEK_BASE_URL} — "
            "LLM summarization and query planning cannot run in this pass."
        )
    elif any("deepseek_missing_llm" in d for d in degraded):
        lines.append(
            f"  • DeepSeek LLM model '{_LLM_MODEL}' is NOT available. "
            "Check DEEPSEEK_API_KEY and model configuration."
        )

    fully_ready = avail.get("tickers_fully_ready") or []
    missing_tickers = [t for t in tickers if t not in fully_ready]
    if missing_tickers:
        lines.append(
            f"  • Tickers with incomplete data: {missing_tickers}. "
            "Summaries for these will rely on whatever partial data is available."
        )

    lines.append(
        "  → The analyst MUST acknowledge these gaps explicitly in the research note "
        "and MUST NOT fabricate metrics that are not present in source data."
    )

    return "\n".join(lines)


def ticker_data_profile(avail: Dict[str, Any], ticker: str) -> Dict[str, bool]:
    """Return a simple boolean capability profile for a single ticker.

    Used by agent nodes to decide which code paths to attempt.

    Returns
    -------
    {
      "has_neo4j_chunks": bool,
      "has_pg_fundamentals": bool,
      "has_pg_timeseries": bool,
      "has_pg_sentiment": bool,
      "has_any_qualitative": bool,   # neo4j chunks
      "has_any_quantitative": bool,  # pg fundamentals or timeseries
      "has_llm": bool,
    }
    """
    neo4j    = avail.get("neo4j") or {}
    postgres = avail.get("postgres") or {}
    deepseek = avail.get("deepseek") or {}

    has_neo4j   = ticker in (neo4j.get("tickers_with_chunks") or [])
    has_pg_fund = ticker in (postgres.get("fundamentals_tickers") or [])
    has_pg_ts   = ticker in (postgres.get("timeseries_tickers") or [])
    has_pg_sent = (postgres.get("sentiment_rows") or 0) > 0

    return {
        "has_neo4j_chunks":    has_neo4j,
        "has_pg_fundamentals": has_pg_fund,
        "has_pg_timeseries":   has_pg_ts,
        "has_pg_sentiment":    has_pg_sent,
        "has_any_qualitative": has_neo4j,
        "has_any_quantitative": has_pg_fund or has_pg_ts,
        "has_llm":             deepseek.get("llm_model_ready", False),
    }


__all__ = [
    "check_all",
    "availability_notice",
    "ticker_data_profile",
]
