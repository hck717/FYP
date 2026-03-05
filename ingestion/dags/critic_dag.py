"""Airflow DAG — Nightly Critic Agent run.

Schedule: 02:00 UTC every day (runs after overnight market data ingestion).

Tasks
-----
1. fetch_recent_runs     — query agent_run_telemetry for distinct run_ids
                           from the last 24 hours.
2. run_nli_check         — for each run_id, call run_nli_hallucination_check()
                           with agent output text pulled from query_logs /
                           agent_run_telemetry.  Source chunks are fetched
                           from the citation_tracking table (chunk_text column,
                           if present) or from Qdrant payload.
3. compute_cur           — compute Citation Utilisation Rate per run_id.
4. log_summary           — emit an Airflow log summary of unverified claim %
                           and low-CUR runs.

All tasks are gracefully tolerant of missing data — the DAG never fails the
pipeline if PostgreSQL or Qdrant is temporarily unavailable.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Airflow imports ────────────────────────────────────────────────────────────
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False
    logger.warning("[critic_dag] Airflow not installed — DAG will not register.")

# ── default args ───────────────────────────────────────────────────────────────

_DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=30),
}


# ── task functions ─────────────────────────────────────────────────────────────

def _fetch_recent_run_ids(**context) -> List[str]:
    """Task 1: Fetch distinct run_ids from agent_run_telemetry (last 24h).

    Pushes the list to XCom under key 'run_ids'.
    """
    import sys
    import os
    # Ensure the repo root is on sys.path so the agents package is importable
    repo_root = os.getenv("FYP_REPO_ROOT", "/opt/airflow")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from agents.critic.agent import _fetch_recent_runs  # type: ignore[import]
        run_ids = _fetch_recent_runs(lookback_hours=24)
        logger.info("[critic_dag] Fetched %d run_id(s) from last 24h.", len(run_ids))
    except Exception as exc:
        logger.warning("[critic_dag] Could not fetch run_ids: %s", exc)
        run_ids = []

    context["task_instance"].xcom_push(key="run_ids", value=run_ids)
    return run_ids


def _fetch_source_chunks_for_run(run_id: str) -> Dict[str, List[str]]:
    """Helper: pull source chunk text from citation_tracking for a given run_id.

    Returns a dict of agent_name → [chunk_text, ...].
    Falls back to empty dict if the table schema doesn't have chunk_text.
    """
    try:
        import psycopg2  # type: ignore[import]
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "airflow"),
            user=os.getenv("POSTGRES_USER", "airflow"),
            password=os.getenv("POSTGRES_PASSWORD", "airflow"),
        )
        chunks_by_agent: Dict[str, List[str]] = {}
        with conn, conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT agent_name, chunk_text
                    FROM citation_tracking
                    WHERE run_id = %s AND chunk_text IS NOT NULL
                    """,
                    (run_id,),
                )
                for row in cur.fetchall():
                    agent_name, chunk_text = row
                    chunks_by_agent.setdefault(agent_name, []).append(chunk_text)
            except Exception:
                # chunk_text column may not exist in all schema versions
                pass
        conn.close()
        return chunks_by_agent
    except Exception as exc:
        logger.debug("[critic_dag] Could not fetch source chunks for run_id=%s: %s", run_id, exc)
        return {}


def _fetch_agent_outputs_for_run(run_id: str) -> Dict[str, Any]:
    """Helper: pull agent output text from query_logs for a given run_id.

    Returns a dict of agent_name → output dict (best-effort).
    """
    try:
        import psycopg2  # type: ignore[import]
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "airflow"),
            user=os.getenv("POSTGRES_USER", "airflow"),
            password=os.getenv("POSTGRES_PASSWORD", "airflow"),
        )
        outputs: Dict[str, Any] = {}
        with conn, conn.cursor() as cur:
            try:
                # query_logs stores the full agent output JSON in 'agent_outputs' column
                cur.execute(
                    """
                    SELECT agent_outputs
                    FROM query_logs
                    WHERE session_id = %s
                    ORDER BY recorded_at DESC
                    LIMIT 1
                    """,
                    (run_id,),
                )
                row = cur.fetchone()
                if row and row[0]:
                    raw = row[0]
                    if isinstance(raw, str):
                        raw = json.loads(raw)
                    if isinstance(raw, dict):
                        outputs = raw
            except Exception:
                pass
        conn.close()
        return outputs
    except Exception as exc:
        logger.debug("[critic_dag] Could not fetch agent outputs for run_id=%s: %s", run_id, exc)
        return {}


def _run_nli_check(**context) -> Dict[str, Any]:
    """Task 2: Run NLI hallucination check for each run_id fetched in Task 1."""
    import sys
    repo_root = os.getenv("FYP_REPO_ROOT", "/opt/airflow")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    run_ids: List[str] = context["task_instance"].xcom_pull(key="run_ids") or []
    if not run_ids:
        logger.info("[critic_dag] No run_ids — skipping NLI check.")
        return {"nli_checked": 0, "total_unverified": 0}

    try:
        from agents.critic.agent import run_nli_hallucination_check  # type: ignore[import]
    except Exception as exc:
        logger.warning("[critic_dag] Could not import critic agent: %s", exc)
        return {"nli_checked": 0, "total_unverified": 0, "error": str(exc)}

    total_claims = 0
    total_unverified = 0

    for run_id in run_ids:
        agent_outputs = _fetch_agent_outputs_for_run(run_id)
        source_chunks = _fetch_source_chunks_for_run(run_id)
        if not agent_outputs:
            logger.debug("[critic_dag] No agent outputs for run_id=%s — skipping NLI.", run_id)
            continue
        try:
            results = run_nli_hallucination_check(
                run_id=run_id,
                agent_outputs=agent_outputs,
                source_chunks=source_chunks,
            )
            unverified = sum(1 for r in results if not r.verified)
            total_claims += len(results)
            total_unverified += unverified
            logger.info(
                "[critic_dag] run_id=%s  claims=%d  unverified=%d",
                run_id, len(results), unverified,
            )
        except Exception as exc:
            logger.warning("[critic_dag] NLI check failed for run_id=%s: %s", run_id, exc)

    summary = {
        "nli_checked": len(run_ids),
        "total_claims": total_claims,
        "total_unverified": total_unverified,
        "unverified_pct": round(100 * total_unverified / total_claims, 1) if total_claims else 0.0,
    }
    context["task_instance"].xcom_push(key="nli_summary", value=summary)
    return summary


def _compute_cur(**context) -> Dict[str, Any]:
    """Task 3: Compute Citation Utilisation Rate for each run_id."""
    import sys
    repo_root = os.getenv("FYP_REPO_ROOT", "/opt/airflow")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    run_ids: List[str] = context["task_instance"].xcom_pull(key="run_ids") or []
    if not run_ids:
        logger.info("[critic_dag] No run_ids — skipping CUR computation.")
        return {"cur_checked": 0, "low_cur_runs": 0}

    try:
        from agents.critic.agent import compute_citation_utilisation_rate  # type: ignore[import]
    except Exception as exc:
        logger.warning("[critic_dag] Could not import critic agent: %s", exc)
        return {"cur_checked": 0, "low_cur_runs": 0, "error": str(exc)}

    low_cur_runs = 0
    cur_values = []

    for run_id in run_ids:
        try:
            result = compute_citation_utilisation_rate(run_id)
            cur_values.append(result.cur)
            if result.low_cur:
                low_cur_runs += 1
        except Exception as exc:
            logger.warning("[critic_dag] CUR computation failed for run_id=%s: %s", run_id, exc)

    avg_cur = sum(cur_values) / len(cur_values) if cur_values else 0.0
    summary = {
        "cur_checked": len(run_ids),
        "low_cur_runs": low_cur_runs,
        "avg_cur": round(avg_cur, 3),
    }
    context["task_instance"].xcom_push(key="cur_summary", value=summary)
    return summary


def _log_summary(**context) -> None:
    """Task 4: Log a human-readable summary of the critic run."""
    nli_summary = context["task_instance"].xcom_pull(key="nli_summary") or {}
    cur_summary = context["task_instance"].xcom_pull(key="cur_summary") or {}

    logger.info(
        "[critic_dag] ══ NIGHTLY CRITIC SUMMARY ══\n"
        "  NLI: checked=%s  claims=%s  unverified=%s (%.1f%%)\n"
        "  CUR: checked=%s  low_cur_runs=%s  avg_cur=%.3f",
        nli_summary.get("nli_checked", 0),
        nli_summary.get("total_claims", 0),
        nli_summary.get("total_unverified", 0),
        nli_summary.get("unverified_pct", 0.0),
        cur_summary.get("cur_checked", 0),
        cur_summary.get("low_cur_runs", 0),
        cur_summary.get("avg_cur", 0.0),
    )


# ── DAG definition ─────────────────────────────────────────────────────────────

if _AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="critic_nightly",
        description="Nightly critic agent: NLI hallucination check + Citation Utilisation Rate",
        schedule_interval="0 2 * * *",  # 02:00 UTC daily
        start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        catchup=False,
        default_args=_DEFAULT_ARGS,
        tags=["critic", "quality-assurance", "nightly"],
        max_active_runs=1,
    ) as dag:

        task_fetch_runs = PythonOperator(
            task_id="fetch_recent_runs",
            python_callable=_fetch_recent_run_ids,
            provide_context=True,
        )

        task_nli = PythonOperator(
            task_id="run_nli_check",
            python_callable=_run_nli_check,
            provide_context=True,
        )

        task_cur = PythonOperator(
            task_id="compute_cur",
            python_callable=_compute_cur,
            provide_context=True,
        )

        task_summary = PythonOperator(
            task_id="log_summary",
            python_callable=_log_summary,
            provide_context=True,
        )

        # Pipeline: fetch → [nli, cur in parallel] → summary
        task_fetch_runs >> [task_nli, task_cur] >> task_summary
