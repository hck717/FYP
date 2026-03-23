"""
Orchestration wrapper for the Stock-Research-Agent.

This module exposes ``run_full_analysis(ticker, availability_profile)`` — the
standard entry-point convention used by every agent in the orchestration system
(BA, QF, FM, WS).  It adapts the standalone pipeline in ``agent_step7_synthesis``
so that ``node_parallel_agents`` in ``orchestration/nodes.py`` can call it exactly
like the other four agents.

Data source strategy
--------------------
The agent tries PostgreSQL/pgvector first (data ingested by the Airflow pipeline)
and falls back to local PDF files if PostgreSQL is unavailable or has no data
for the ticker.

**PG mode** (preferred):
  Chunks already extracted and embedded by
  ``ingestion/etl/ingest_earnings_calls.py`` / ``ingest_broker_reports.py``
  are fetched from PostgreSQL text_chunks via pgvector HNSW index.
  No PDF reading or local embedding model required.
  PostgreSQL connection uses env vars POSTGRES_HOST/PORT/DB/USER/PASSWORD.

**PDF mode** (fallback):
  PDF files are expected at:
      <data_dir>/<TICKER>/broker/
      <data_dir>/<TICKER>/earnings*/
  where <data_dir> defaults to ``data_reports/`` inside this directory.
  Override with ``STOCK_RESEARCH_DATA_DIR`` env var.

Output format
-------------
Returns a dict compatible with ``OrchestrationState`` consumption:

    {
        "agent":                  "stock_research",
        "ticker":                 str,
        "latest_transcript":      str,          # filename or source name
        "previous_transcript":    str,          # filename or source name
        "transcript_comparison":  str,          # LLM analysis text (Task A)
        "qa_behavior":            str,          # LLM analysis text (Task B)
        "broker_consensus":       str,          # LLM analysis text (Task C)
        "broker_labels":          dict,         # {doc_name: {"rating": ...}}
        "broker_parsed":          list[dict],   # per-broker extracted metrics
        "features": {
            "latest":   dict,                   # deterministic NLP features
            "previous": dict,
            "kpi_diff": dict,
        },
        "citations":              list[dict],   # all [doc_name p.N] citations
        "thinking_trace":         list[str],    # empty — no chain-of-thought
        "error":                  str | None,   # set if pipeline failed
        "data_source":            str,          # "pg" | "pdf"
    }
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Agent directory ──────────────────────────────────────────────────────────
_AGENT_DIR = Path(__file__).resolve().parent

# ── Data directory (PDF fallback) ─────────────────────────────────────────────
_DEFAULT_DATA_DIR = _AGENT_DIR / "data_reports"

# ── Load .env for local Streamlit/CLI runs ───────────────────────────────────
_REPO_ROOT = _AGENT_DIR.parents[1]
_ENV_PATH = _REPO_ROOT / ".env"
if _ENV_PATH.exists():
    try:
        with open(_ENV_PATH) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())
    except Exception:
        pass


def _resolve_data_dir() -> Path:
    override = os.getenv("STOCK_RESEARCH_DATA_DIR", "").strip()
    if override:
        p = Path(override)
        if p.is_dir():
            return p
        logger.warning(
            "[stock_research] STOCK_RESEARCH_DATA_DIR=%r does not exist — "
            "falling back to default %s",
            override, _DEFAULT_DATA_DIR,
        )
    return _DEFAULT_DATA_DIR


def _pg_has_data(ticker: str) -> bool:
    """Return True if PostgreSQL has at least 2 earnings-call docs for ticker."""
    try:
        import psycopg2
        PG_HOST     = os.getenv("POSTGRES_HOST",     "localhost")
        PG_PORT     = int(os.getenv("POSTGRES_PORT", "5432"))
        PG_DB       = os.getenv("POSTGRES_DB",       "airflow")
        PG_USER     = os.getenv("POSTGRES_USER",     "airflow")
        PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT,
            dbname=PG_DB, user=PG_USER, password=PG_PASSWORD,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(DISTINCT filename)
                    FROM textual_documents
                    WHERE ticker = %s
                      AND doc_type = 'earnings_call'
                    """,
                    (ticker,),
                )
                row = cur.fetchone()
                cnt = row[0] if row else 0
                return cnt >= 2
        finally:
            conn.close()
    except Exception as exc:
        logger.debug("[stock_research] PostgreSQL check failed: %s", exc)
        return False


# ── Main entry point ──────────────────────────────────────────────────────────

def run_full_analysis(
    ticker: str,
    availability_profile: Optional[Dict[str, Any]] = None,  # noqa: ARG001 — reserved
) -> Dict[str, Any]:
    """Run the full stock-research pipeline for *ticker* and return a result dict.

    Tries Neo4j first; falls back to local PDFs if Neo4j data is unavailable.

    Parameters
    ----------
    ticker:
        Uppercase ticker symbol, e.g. ``"AAPL"``.
    availability_profile:
        Reserved for future use.

    Returns
    -------
    dict
        Orchestration-compatible result dict (see module docstring).
    """
    ticker = ticker.strip().upper()
    base_dir = _resolve_data_dir()

    logger.info("[stock_research] Starting full analysis for ticker=%s", ticker)

    # Import here (not at module level) to avoid startup cost.
    # Use package-qualified imports to prevent cross-agent module collisions.
    from agents.stock_research_agent.agent_step7_synthesis import run_full_analysis as _run  # type: ignore[import]

    # Initialise to satisfy static analysis — will be set in one of the branches below
    raw: Dict[str, Any] = {}
    data_source: str = "none"

    # ── Try PG mode first ──────────────────────────────────────────────────────
    use_neo4j = _pg_has_data(ticker)

    if use_neo4j:
        logger.info("[stock_research] PostgreSQL/pgvector data found for ticker=%s — using PG mode", ticker)
        try:
            from agents.stock_research_agent.agent_step1_neo4j import load_neo4j_pages  # type: ignore[import]
            transcript_pages, broker_pages, latest_name, previous_name = load_neo4j_pages(ticker)

            raw: Dict[str, Any] = _run(
                ticker=ticker,
                base_dir=base_dir,
                use_neo4j=True,
                transcript_pages=transcript_pages,
                broker_pages=broker_pages,
                latest_name=latest_name,
                previous_name=previous_name,
            )
            data_source = "pg"
        except Exception as exc:
            logger.warning(
                "[stock_research] PG mode failed for ticker=%s (%s) — "
                "falling back to PDF mode",
                ticker, exc,
            )
            use_neo4j = False

    if not use_neo4j:
        # ── PDF fallback ──────────────────────────────────────────────────────
        ticker_dir = base_dir / ticker
        if not ticker_dir.is_dir():
            msg = (
                f"No data found for ticker {ticker!r}. "
                f"PostgreSQL has no chunks and local PDF directory {ticker_dir} does not exist. "
                f"Run `ingestion/etl/ingest_earnings_calls.py {ticker}` to populate PostgreSQL, "
                f"or place PDF files under {ticker_dir}/broker/ and {ticker_dir}/earnings*/."
            )
            logger.error("[stock_research] %s", msg)
            return _error_output(ticker, msg)

        logger.info("[stock_research] Using PDF mode from %s", base_dir)
        try:
            raw = _run(ticker=ticker, base_dir=base_dir, use_neo4j=False)
            data_source = "pdf"
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            logger.error("[stock_research] PDF pipeline failed for ticker=%s: %s", ticker, msg)
            return _error_output(ticker, msg)

    # ── Reshape into orchestration-compatible format ───────────────────────────
    tasks: List[Dict[str, Any]] = raw.get("tasks") or []

    def _task(name: str) -> str:
        for t in tasks:
            if t.get("task") == name:
                return t.get("analysis") or ""
        return ""

    all_citations: List[Dict[str, Any]] = []
    for t in tasks:
        all_citations.extend(t.get("citations_found") or [])

    result: Dict[str, Any] = {
        "agent":               "stock_research",
        "ticker":              raw.get("ticker", ticker),
        "latest_transcript":   raw.get("latest_transcript", ""),
        "previous_transcript": raw.get("previous_transcript", ""),
        "transcript_comparison": _task("transcript_comparison"),
        "qa_behavior":           _task("qa_behavior"),
        "broker_consensus":      _task("broker_consensus"),
        "broker_labels":         raw.get("broker_labels") or {},
        "broker_parsed":         raw.get("broker_parsed") or [],
        "features":              raw.get("features") or {},
        "citations":             all_citations,
        "thinking_trace":        [],
        "error":                 None,
        "data_source":           data_source,
    }

    logger.info(
        "[stock_research] Done for ticker=%s  source=%s  transcripts=%s/%s  brokers=%d  citations=%d",
        ticker,
        data_source,
        result["latest_transcript"],
        result["previous_transcript"],
        len(result["broker_parsed"]),
        len(all_citations),
    )
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _error_output(ticker: str, error_msg: str) -> Dict[str, Any]:
    """Return a minimal error-flagged result dict that the orchestrator can handle."""
    return {
        "agent":               "stock_research",
        "ticker":              ticker,
        "latest_transcript":   "",
        "previous_transcript": "",
        "transcript_comparison": "",
        "qa_behavior":           "",
        "broker_consensus":      "",
        "broker_labels":         {},
        "broker_parsed":         [],
        "features":              {},
        "citations":             [],
        "thinking_trace":        [],
        "error":                 error_msg,
        "data_source":           "none",
    }


__all__ = ["run_full_analysis"]
