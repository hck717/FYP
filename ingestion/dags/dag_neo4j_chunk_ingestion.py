"""Airflow DAG — Neo4j Chunk Ingestion (LLM synthesis + vector embedding).

Schedule: weekly on Saturday at 01:00 UTC, after overnight data ingestion.

Purpose
-------
For each tracked ticker (AAPL, MSFT, GOOGL, TSLA, NVDA) this DAG:
  1. Reads company_profile.csv and financial_news.csv from
     ingestion/etl/agent_data/business_analyst/{TICKER}/.
  2. Calls a local Ollama LLM (qwen2.5:7b, fallback deepseek-r1:8b) to
     synthesise 5 analytical text chunks per ticker:
       - competitive_moat, risk_factors, mda, earnings, news
  3. Embeds each chunk with all-MiniLM-L6-v2 (384-dim, sentence-transformers).
  4. Upserts :Chunk nodes into Neo4j linked via (Company)-[:HAS_CHUNK]->(Chunk).

This DAG can also be triggered manually after a Neo4j schema reset or when new
company profile / news data has been ingested.

Dependencies
------------
- Ollama running at OLLAMA_BASE_URL with qwen2.5:7b or deepseek-r1:8b pulled.
- sentence-transformers installed in the Airflow Python environment.
- Neo4j vector index `chunk_embedding` (384-dim) — created by the DAG if absent.
- FYP_REPO_ROOT env var pointing to the repo root (default: /opt/airflow/repo).
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Airflow imports ────────────────────────────────────────────────────────────
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False
    logger.warning("[neo4j_chunk_dag] Airflow not installed — DAG will not register.")

# ── default args ───────────────────────────────────────────────────────────────

_DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": None,  # no cap — LLM synthesis can take a while
}

# ── configuration ─────────────────────────────────────────────────────────────

_REPO_ROOT = os.getenv("FYP_REPO_ROOT", "/opt/airflow/repo")
_TICKERS: List[str] = [
    t.strip()
    for t in os.getenv("TRACKED_TICKERS", "AAPL,MSFT,GOOGL,TSLA,NVDA").split(",")
    if t.strip()
]


# ── task helpers ──────────────────────────────────────────────────────────────

def _ensure_repo_on_path() -> None:
    """Add repo root to sys.path so ingestion.etl is importable."""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)


def _ingest_chunks_for_ticker(ticker: str, **_context) -> dict:
    """Synthesise and embed Neo4j Chunk nodes for a single ticker.

    Returns a summary dict with keys: ticker, chunks_ingested, status.
    """
    _ensure_repo_on_path()
    try:
        from ingestion.etl.ingest_neo4j_chunks import main as _ingest_main  # type: ignore[import]
    except ImportError as exc:
        logger.error(
            "[neo4j_chunk_dag] Cannot import ingest_neo4j_chunks: %s  "
            "(FYP_REPO_ROOT=%s — is it set correctly?)",
            exc, _REPO_ROOT,
        )
        raise

    logger.info("[neo4j_chunk_dag] Starting chunk ingestion for ticker: %s", ticker)
    try:
        _ingest_main(tickers=[ticker])
        logger.info("[neo4j_chunk_dag] Chunk ingestion complete for %s", ticker)
        return {"ticker": ticker, "status": "ok"}
    except Exception as exc:
        logger.error("[neo4j_chunk_dag] Chunk ingestion failed for %s: %s", ticker, exc)
        raise


def _log_summary(**context) -> None:
    """Pull XCom results from all ticker tasks and emit a summary log."""
    summary_lines = []
    for ticker in _TICKERS:
        result = context["task_instance"].xcom_pull(task_ids=f"ingest_{ticker.lower()}")
        status = (result or {}).get("status", "unknown")
        summary_lines.append(f"  {ticker}: {status}")
    logger.info(
        "[neo4j_chunk_dag] ══ NEO4J CHUNK INGESTION SUMMARY ══\n%s",
        "\n".join(summary_lines),
    )


# ── DAG definition ────────────────────────────────────────────────────────────

if _AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="neo4j_chunk_ingestion_weekly",
        description=(
            "Weekly Neo4j Chunk ingestion: LLM synthesis + sentence-transformer "
            "embedding → :Chunk nodes in Neo4j"
        ),
        schedule_interval="0 1 * * 6",  # 01:00 UTC every Saturday
        start_date=datetime(2025, 1, 4, tzinfo=timezone.utc),  # first Saturday 2025
        catchup=False,
        default_args=_DEFAULT_ARGS,
        tags=["neo4j", "embedding", "chunk-ingestion", "weekly"],
        max_active_runs=1,
    ) as dag:

        # One task per ticker — run in parallel
        ticker_tasks = []
        for _ticker in _TICKERS:
            t = PythonOperator(
                task_id=f"ingest_{_ticker.lower()}",
                python_callable=_ingest_chunks_for_ticker,
                op_kwargs={"ticker": _ticker},
                provide_context=True,
                execution_timeout=None,  # LLM synthesis is unbounded
            )
            ticker_tasks.append(t)

        # Summary task runs after all ticker tasks complete
        summary_task = PythonOperator(
            task_id="log_summary",
            python_callable=_log_summary,
            provide_context=True,
            trigger_rule="all_done",  # run even if some tickers failed
        )

        # All ticker tasks feed into summary
        ticker_tasks >> summary_task
