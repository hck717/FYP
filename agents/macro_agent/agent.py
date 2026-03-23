"""
Orchestration wrapper for the Macro-Agent.

This module exposes ``run_full_analysis(ticker, availability_profile)`` — the
standard entry-point convention used by every agent in the orchestration system
(BA, QF, FM, WS, stock_research).  It adapts the standalone pipeline in 
``agent_step7_synthesis`` so that ``node_parallel_agents`` in ``orchestration/nodes.py`` 
can call it exactly like the other agents.

Data source strategy
--------------------
The agent fetches macro analysis data from Neo4j/PostgreSQL and links it to
a target ticker by analyzing the latest earnings call for that ticker.

**PG/Neo4j mode** (preferred):
  Macro report chunks (section='macro_report') are fetched from Neo4j/PostgreSQL.
  Latest earnings call for target ticker provides context on what the company does.
  No PDF fallback for macro_agent (macro data is always in the DB).

Output format
-------------
Returns a dict compatible with ``OrchestrationState`` consumption:

    {
        "agent":                  "macro",
        "ticker":                 str,
        "regime":                 str,           # "risk-off" / "growth-at-risk" / etc.
        "macro_themes": [
            {
                "theme": str,
                "direction": str,                # "bullish" | "bearish" | "neutral"
                "confidence": float,             # 0.0-1.0
                "transmission_channel": str,
                "impact_magnitude": str,         # "high" | "medium" | "low"
                "time_horizon": str,             # "immediate" | "medium-term" | "long-term"
            },
            ...
        ],
        "per_report_summaries": [
            {
                "report_name": str,
                "summary": str,                  # 2-3 sentence thesis
                "stock_relevance": str,          # transmission channel to target ticker
                "report_date": str,
            },
            ...
        ],
        "top_macro_drivers": list[str],         # top 2-3 factors
        "top_risk": str,
        "risk_scenario": str,
        "citations":              list[dict],    # all [doc_name p.N] citations
        "thinking_trace":         list[str],     # empty — no chain-of-thought
        "error":                  str | None,    # set if pipeline failed
        "data_source":            str,           # "pg" | "neo4j"
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


def _pg_has_macro_data() -> bool:
    """Return True if PostgreSQL has macro report chunks."""
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
                    SELECT COUNT(*) FROM text_chunks
                    WHERE ticker = '_MACRO' AND section = 'macro_report'
                    """,
                )
                row = cur.fetchone()
                cnt = row[0] if row else 0
                return cnt > 0
        finally:
            conn.close()
    except Exception as exc:
        logger.debug("[macro] PostgreSQL check failed: %s", exc)
        return False


# ── Main entry point ──────────────────────────────────────────────────────────

def run_full_analysis(
    ticker: str,
    availability_profile: Optional[Dict[str, Any]] = None,  # noqa: ARG001 — reserved
) -> Dict[str, Any]:
    """Run the full macro-analysis pipeline for *ticker* and return a result dict.

    Fetches all macro report chunks from Neo4j/PostgreSQL and links them to
    the target ticker by analyzing the latest earnings call.

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

    logger.info("[macro] Starting full analysis for ticker=%s", ticker)

    # Import here (not at module level) to avoid startup cost.
    # Use package-qualified imports to prevent cross-agent module collisions.
    from agents.macro_agent.agent_step7_synthesis import run_full_analysis as _run  # type: ignore[import]

    # Initialise to satisfy static analysis — will be set below
    raw: Dict[str, Any] = {}
    data_source: str = "none"

    # ── Try Neo4j/PG mode ─────────────────────────────────────────────────────
    has_macro_data = _pg_has_macro_data()

    if has_macro_data:
        logger.info("[macro] PostgreSQL/Neo4j macro data found — using DB mode")
        try:
            from agents.macro_agent.agent_step1_neo4j import load_macro_and_earnings  # type: ignore[import]
            macro_pages, earnings_pages, macro_doc_names = load_macro_and_earnings(ticker)

            raw = _run(
                ticker=ticker,
                macro_pages=macro_pages,
                earnings_pages=earnings_pages,
                macro_doc_names=macro_doc_names,
            )
            data_source = "neo4j"
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            logger.error("[macro] Neo4j pipeline failed for ticker=%s: %s", ticker, msg)
            return _error_output(ticker, msg)
    else:
        msg = (
            f"No macro data found. PostgreSQL has no '_MACRO' chunks with section='macro_report'. "
            f"Run `ingestion/etl/ingest_macro_reports.py` to populate PostgreSQL."
        )
        logger.error("[macro] %s", msg)
        return _error_output(ticker, msg)

    # ── Reshape into orchestration-compatible format ───────────────────────────
    result: Dict[str, Any] = {
        "agent":               "macro",
        "ticker":              raw.get("ticker", ticker),
        "regime":              raw.get("regime", ""),
        "macro_themes":        raw.get("macro_themes") or [],
        "per_report_summaries": raw.get("per_report_summaries") or [],
        "top_macro_drivers":   raw.get("top_macro_drivers") or [],
        "top_risk":            raw.get("top_risk", ""),
        "risk_scenario":       raw.get("risk_scenario", ""),
        "citations":           raw.get("citations") or [],
        "thinking_trace":      [],
        "error":               None,
        "data_source":         data_source,
    }

    logger.info(
        "[macro] Done for ticker=%s  source=%s  macro_reports=%d  themes=%d  citations=%d",
        ticker,
        data_source,
        len(result["per_report_summaries"]),
        len(result["macro_themes"]),
        len(result["citations"]),
    )
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _error_output(ticker: str, error_msg: str) -> Dict[str, Any]:
    """Return a minimal error-flagged result dict that the orchestrator can handle."""
    return {
        "agent":               "macro",
        "ticker":              ticker,
        "regime":              "",
        "macro_themes":        [],
        "per_report_summaries": [],
        "top_macro_drivers":   [],
        "top_risk":            "",
        "risk_scenario":       "",
        "citations":           [],
        "thinking_trace":      [],
        "error":               error_msg,
        "data_source":         "none",
    }


__all__ = ["run_full_analysis"]
