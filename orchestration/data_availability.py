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
  "qdrant": {
    "reachable": bool,
    "collection_exists": bool,
    "point_count": int,
    "tickers_with_vectors": [str, ...],  # tickers that have >=1 vector
  },
  "postgres": {
    "reachable": bool,
    "sentiment_rows": int,               # rows in sentiment_trends
    "fundamentals_tickers": [str, ...],  # distinct ticker_symbols in raw_fundamentals
    "timeseries_tickers": [str, ...],    # distinct ticker_symbols in raw_timeseries
    "market_eod_rows": int,              # rows in market_eod_us
  },
  "ollama": {
    "reachable": bool,
    "available_models": [str, ...],
    "embedding_model_ready": bool,       # all-MiniLM-L6-v2 (local ST, always True if installed)
    "llm_model_ready": bool,             # deepseek-r1:8b or configured model
  },
  "tickers_fully_ready": [str, ...],     # tickers with neo4j chunks + qdrant vectors + pg fundamentals
  "tickers_partially_ready": [str, ...], # tickers with >=1 data source
  "degraded_tiers": [str, ...],          # list of tier names that are down/empty
  "summary": str,                        # human-readable one-liner for logs / UI
}
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ── defaults (mirror agents/business_analyst/config.py) ────────────────────
_NEO4J_URI      = os.getenv("NEO4J_URI",            "bolt://localhost:7687")
_NEO4J_USER     = os.getenv("NEO4J_USER",            "neo4j")
_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD",        "changeme_neo4j_password")
_QDRANT_HOST    = os.getenv("QDRANT_HOST",           "localhost")
_QDRANT_PORT    = int(os.getenv("QDRANT_PORT",       "6333"))
_QDRANT_COLL    = os.getenv("QDRANT_COLLECTION_NAME","financial_documents")
_PG_HOST        = os.getenv("POSTGRES_HOST",         "localhost")
_PG_PORT        = int(os.getenv("POSTGRES_PORT",     "5432"))
_PG_DB          = os.getenv("POSTGRES_DB",           "airflow")
_PG_USER        = os.getenv("POSTGRES_USER",         "airflow")
_PG_PASS        = os.getenv("POSTGRES_PASSWORD",     "airflow")
_OLLAMA_URL     = os.getenv("OLLAMA_BASE_URL",       "http://localhost:11434")
_LLM_MODEL      = os.getenv("LLM_MODEL_BUSINESS_ANALYST", "deepseek-r1:8b")
_EMBED_MODEL    = os.getenv("EMBEDDING_MODEL",       "all-MiniLM-L6-v2")

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


def _check_qdrant() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "reachable": False,
        "collection_exists": False,
        "point_count": 0,
        "tickers_with_vectors": [],
        "error": None,
    }
    try:
        from qdrant_client import QdrantClient  # type: ignore[import]
        from qdrant_client.http import models as qmodels  # type: ignore[import]

        client = QdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT, timeout=5)
        collections = [c.name for c in client.get_collections().collections]
        result["reachable"] = True

        if _QDRANT_COLL not in collections:
            return result

        result["collection_exists"] = True
        info = client.get_collection(_QDRANT_COLL)
        result["point_count"] = info.points_count or 0

        # Which tickers have vectors?
        if result["point_count"] > 0:
            found: List[str] = []
            for ticker in _SUPPORTED_TICKERS:
                try:
                    resp = client.scroll(
                        collection_name=_QDRANT_COLL,
                        scroll_filter=qmodels.Filter(
                            must=[qmodels.FieldCondition(
                                key="ticker_symbol",
                                match=qmodels.MatchValue(value=ticker),
                            )]
                        ),
                        limit=1,
                        with_payload=False,
                        with_vectors=False,
                    )
                    if resp[0]:
                        found.append(ticker)
                except Exception:
                    pass
            result["tickers_with_vectors"] = found

    except Exception as exc:
        result["error"] = str(exc)
        logger.debug("Qdrant availability check failed: %s", exc)
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

            # raw_fundamentals — which tickers are present?
            try:
                cur.execute(
                    "SELECT DISTINCT ticker_symbol FROM raw_fundamentals "
                    "WHERE ticker_symbol = ANY(%s)",
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


def _check_ollama() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "reachable": False,
        "available_models": [],
        "embedding_model_ready": False,  # sentence-transformers (local, not Ollama)
        "llm_model_ready": False,
        "error": None,
    }
    try:
        resp = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        result["reachable"] = True
        result["available_models"] = [m["name"] for m in resp.json().get("models", [])]
        result["llm_model_ready"] = any(
            _LLM_MODEL in m for m in result["available_models"]
        )
    except Exception as exc:
        result["error"] = str(exc)
        logger.debug("Ollama availability check failed: %s", exc)

    # Check sentence-transformers locally (never requires Ollama)
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        result["embedding_model_ready"] = True
    except ImportError:
        result["embedding_model_ready"] = False

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def check_all(tickers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run all tier checks and return the consolidated availability report.

    All four backend checks (Neo4j, Qdrant, PostgreSQL, Ollama) are executed
    concurrently via a ThreadPoolExecutor so one slow backend does not block
    the others.  Total wall-clock time is bounded by the slowest single check
    rather than their sum.

    Parameters
    ----------
    tickers:
        Specific tickers to assess readiness for.  Defaults to all 5 supported tickers.
    """
    tickers = tickers or _SUPPORTED_TICKERS

    # Run all four checks concurrently
    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="avail_check") as pool:
        fut_neo4j    = pool.submit(_check_neo4j)
        fut_qdrant   = pool.submit(_check_qdrant)
        fut_postgres = pool.submit(_check_postgres)
        fut_ollama   = pool.submit(_check_ollama)

        neo4j    = fut_neo4j.result()
        qdrant   = fut_qdrant.result()
        postgres = fut_postgres.result()
        ollama   = fut_ollama.result()

    # Derive per-ticker readiness
    tickers_with_chunks   = set(neo4j.get("tickers_with_chunks") or [])
    tickers_with_vectors  = set(qdrant.get("tickers_with_vectors") or [])
    tickers_with_pg_fund  = set(postgres.get("fundamentals_tickers") or [])

    fully_ready:   List[str] = []
    partially_ready: List[str] = []
    for t in tickers:
        has_ba   = t in tickers_with_chunks or t in tickers_with_vectors
        has_qf   = t in tickers_with_pg_fund
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

    if not qdrant["reachable"]:
        degraded.append("qdrant_unreachable")
    elif not qdrant["collection_exists"] or qdrant["point_count"] == 0:
        degraded.append("qdrant_empty")

    if not postgres["reachable"]:
        degraded.append("postgres_unreachable")
    elif not postgres["fundamentals_tickers"]:
        degraded.append("postgres_no_fundamentals")

    if not ollama["reachable"]:
        degraded.append("ollama_unreachable")
    elif not ollama["llm_model_ready"]:
        degraded.append(f"ollama_missing_llm_{_LLM_MODEL}")

    # Human summary
    if not degraded:
        summary = (
            f"All tiers healthy. "
            f"Fully ready: {fully_ready}. "
            f"Neo4j chunks: {neo4j['chunk_nodes']}, "
            f"Qdrant vectors: {qdrant['point_count']}, "
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
        "qdrant":   qdrant,
        "postgres": postgres,
        "ollama":   ollama,
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

    qdrant = avail.get("qdrant") or {}
    if "qdrant_unreachable" in degraded:
        lines.append("  • Qdrant vector store is UNREACHABLE — news article retrieval unavailable.")
    elif "qdrant_empty" in degraded:
        lines.append(
            f"  • Qdrant collection '{_QDRANT_COLL}' is EMPTY — "
            "no news article vectors indexed yet."
        )

    postgres = avail.get("postgres") or {}
    if "postgres_unreachable" in degraded:
        lines.append("  • PostgreSQL is UNREACHABLE — no quantitative fundamentals available.")
    elif "postgres_no_fundamentals" in degraded:
        lines.append(
            "  • PostgreSQL raw_fundamentals table is EMPTY — "
            "all P/E, ROE, and factor computations will return null."
        )

    ollama = avail.get("ollama") or {}
    if "ollama_unreachable" in degraded:
        lines.append(
            f"  • Ollama is UNREACHABLE at {_OLLAMA_URL} — "
            "LLM summarization and query planning unavailable."
        )
    elif any("ollama_missing_llm" in d for d in degraded):
        lines.append(
            f"  • Ollama LLM model '{_LLM_MODEL}' is NOT loaded. "
            "Run: ollama pull " + _LLM_MODEL
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
        "and MUST NOT fabricate metrics that are unavailable."
    )

    return "\n".join(lines)


def ticker_data_profile(avail: Dict[str, Any], ticker: str) -> Dict[str, bool]:
    """Return a simple boolean capability profile for a single ticker.

    Used by agent nodes to decide which code paths to attempt.

    Returns
    -------
    {
      "has_neo4j_chunks": bool,
      "has_qdrant_vectors": bool,
      "has_pg_fundamentals": bool,
      "has_pg_timeseries": bool,
      "has_pg_sentiment": bool,
      "has_any_qualitative": bool,   # neo4j or qdrant
      "has_any_quantitative": bool,  # pg fundamentals or timeseries
      "has_llm": bool,
    }
    """
    neo4j   = avail.get("neo4j") or {}
    qdrant  = avail.get("qdrant") or {}
    postgres = avail.get("postgres") or {}
    ollama  = avail.get("ollama") or {}

    has_neo4j   = ticker in (neo4j.get("tickers_with_chunks") or [])
    has_qdrant  = ticker in (qdrant.get("tickers_with_vectors") or [])
    has_pg_fund = ticker in (postgres.get("fundamentals_tickers") or [])
    has_pg_ts   = ticker in (postgres.get("timeseries_tickers") or [])
    has_pg_sent = (postgres.get("sentiment_rows") or 0) > 0

    return {
        "has_neo4j_chunks":    has_neo4j,
        "has_qdrant_vectors":  has_qdrant,
        "has_pg_fundamentals": has_pg_fund,
        "has_pg_timeseries":   has_pg_ts,
        "has_pg_sentiment":    has_pg_sent,
        "has_any_qualitative": has_neo4j or has_qdrant,
        "has_any_quantitative": has_pg_fund or has_pg_ts,
        "has_llm":             ollama.get("llm_model_ready", False),
    }


__all__ = [
    "check_all",
    "availability_notice",
    "ticker_data_profile",
]
