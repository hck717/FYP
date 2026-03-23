"""
Orchestration wrapper for the Insider-News-Agent.

This module exposes ``run_full_analysis(ticker, availability_profile)`` — the
standard entry-point convention used by every agent in the orchestration system.
It adapts the standalone pipeline in ``agent_step1_pg`` and ``agent_step7_synthesis``
so that ``node_parallel_agents`` in ``orchestration/nodes.py`` can call it exactly
like the other agents.

Data source strategy
--------------------
The agent fetches insider trading transactions and news articles from PostgreSQL.

**PostgreSQL mode** (primary):
  Insider transactions and news articles already extracted by the Airflow pipeline
  are fetched from PostgreSQL tables.
  Uses env vars: POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

**Fallback**: If PostgreSQL unavailable, tries Neo4j relationships.

Output format
-------------
Returns a dict compatible with ``OrchestrationState`` consumption:

    {
        "agent": "insider_news",
        "ticker": str,
        "insider_analysis": {
            "activity_summary": str,
            "buy_sell_ratio": float,
            "net_position": str,
            "conviction": str,
            "notable_insiders": list[dict],
            "insider_sentiment": str,
            "red_flags": list[str],
        },
        "news_analysis": {
            "sentiment_summary": str,
            "avg_sentiment_score": float,
            "sentiment_trend": str,
            "positive_catalysts": list[str],
            "negative_catalysts": list[str],
            "key_themes": list[str],
            "credibility": str,
        },
        "investment_thesis": {
            "combined_thesis": str,
            "signal_alignment": str,
            "bull_case": str,
            "bear_case": str,
            "key_risks": list[str],
            "key_opportunities": list[str],
            "recommendation": str,
            "conviction": str,
        },
        "citations": list[dict],
        "thinking_trace": list[str],
        "error": str | None,
        "data_source": str,
        "data_coverage": {
            "insider_transactions_count": int,
            "news_articles_count": int,
            "date_range": str,
        },
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


def _pg_has_insider_news_data(ticker: str) -> bool:
    """Return True if PostgreSQL has insider/news data for ticker."""
    try:
        import psycopg2

        PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
        PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
        PG_DB = os.getenv("POSTGRES_DB", "airflow")
        PG_USER = os.getenv("POSTGRES_USER", "airflow")
        PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")

        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD,
        )
        cursor = conn.cursor()

        # Check if insider_transactions table exists and has data
        cursor.execute(
            "SELECT COUNT(*) FROM insider_transactions WHERE ticker = %s",
            (ticker,),
        )
        insider_row = cursor.fetchone()
        insider_count = int(insider_row[0]) if insider_row else 0

        # Check if news_articles table exists and has data
        cursor.execute(
            "SELECT COUNT(*) FROM news_articles WHERE ticker = %s",
            (ticker,),
        )
        news_row = cursor.fetchone()
        news_count = int(news_row[0]) if news_row else 0

        cursor.close()
        conn.close()

        has_data = insider_count > 0 or news_count > 0
        logger.info(f"PostgreSQL check: {ticker} has {insider_count} insider + {news_count} news records")
        return has_data

    except Exception as e:
        logger.warning(f"PostgreSQL availability check failed: {e}")
        return False


def run_full_analysis(
    ticker: str,
    availability_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for insider+news analysis.

    Parameters
    ----------
    ticker : str
        Stock ticker (e.g., "AAPL")
    availability_profile : Dict | None
        Optional agent availability hints (unused, for compatibility)

    Returns
    -------
    Dict[str, Any]
        Output matching OrchestrationState schema
    """
    ticker = ticker.upper()
    print(f"\n{'='*60}")
    print(f"  Insider+News analysis: {ticker}")
    print(f"{'='*60}\n")

    try:
        # Check if data is available
        if not _pg_has_insider_news_data(ticker):
            logger.warning(f"No insider/news data available for {ticker}")
            return {
                "agent": "insider_news",
                "ticker": ticker,
                "insider_analysis": {},
                "news_analysis": {},
                "investment_thesis": {},
                "citations": [],
                "thinking_trace": [],
                "error": f"No insider/news data available for {ticker}",
                "data_source": "pg",
                "data_coverage": {
                    "insider_transactions_count": 0,
                    "news_articles_count": 0,
                    "date_range": "N/A",
                },
            }

        # Step 1: Fetch data from PostgreSQL
        logger.info(f"[step1_pg] Fetching insider & news data for {ticker}...")
        from agents.insider_news_agent.agent_step1_pg import fetch_insider_and_news_data

        insider_docs, news_docs, metadata = fetch_insider_and_news_data(ticker)

        # Step 7: Run LLM synthesis
        logger.info(f"[step7_synthesis] Running 3 concurrent LLM tasks...")
        from agents.insider_news_agent.agent_step7_synthesis import run_synthesis

        synthesis_result = run_synthesis(insider_docs, news_docs, ticker)

        # Aggregate results
        return {
            "agent": "insider_news",
            "ticker": ticker,
            "insider_analysis": synthesis_result.get("insider_analysis", {}),
            "news_analysis": synthesis_result.get("news_analysis", {}),
            "investment_thesis": synthesis_result.get("combined_thesis", {}),
            "citations": synthesis_result.get("all_citations", []),
            "thinking_trace": [],
            "error": None,
            "data_source": metadata.get("data_source", "pg"),
            "data_coverage": {
                "insider_transactions_count": metadata.get("insider_count", 0),
                "news_articles_count": metadata.get("news_count", 0),
                "date_range": metadata.get("date_range", "N/A"),
            },
        }

    except Exception as e:
        logger.error(f"Error in insider_news agent: {e}", exc_info=True)
        return {
            "agent": "insider_news",
            "ticker": ticker,
            "insider_analysis": {},
            "news_analysis": {},
            "investment_thesis": {},
            "citations": [],
            "thinking_trace": [],
            "error": f"Agent failed: {str(e)}",
            "data_source": "pg",
            "data_coverage": {
                "insider_transactions_count": 0,
                "news_articles_count": 0,
                "date_range": "N/A",
            },
        }


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = run_full_analysis(ticker)

    import json

    print("\n" + json.dumps(result, indent=2))
