"""Airflow DAG — Weekly Qdrant Boost.

Schedule: 03:00 UTC every Sunday (runs after the nightly critic DAG has a full
week of citation_tracking data to aggregate).

Purpose (C1 — Self-Improving RAG)
----------------------------------
Over the course of a week, the ``citation_tracking`` table records which Qdrant
document chunks were actually *used* (cited) in the final investment notes vs.
which were retrieved but discarded.  This DAG boosts the Qdrant payload field
``boost_factor`` for every chunk that was cited at least once, making those
chunks score higher in future hybrid retrieval.

Tasks
-----
1. fetch_cited_chunks    — query citation_tracking for was_cited=True rows
                           in the last 7 days; aggregate by chunk_id.
2. boost_qdrant_payloads — for each cited chunk_id, fetch the current payload
                           from Qdrant, increment boost_factor, and upsert.
3. log_boost_summary     — emit a log of how many chunks were boosted and the
                           distribution of boost_factor increments.

Idempotent: re-running the DAG for the same week will boost the same chunks
again (additive), which is acceptable — citation frequency naturally limits
over-boosting because rarely-cited chunks receive fewer increments.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Airflow imports ────────────────────────────────────────────────────────────
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False
    logger.warning("[qdrant_boost_dag] Airflow not installed — DAG will not register.")

# ── default args ───────────────────────────────────────────────────────────────

_DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=30),
}

# ── Qdrant / PostgreSQL helpers ────────────────────────────────────────────────

_QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
_QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
_QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "financial_documents")
_BOOST_INCREMENT = float(os.getenv("QDRANT_BOOST_INCREMENT", "0.05"))
_BOOST_MAX = float(os.getenv("QDRANT_BOOST_MAX", "2.0"))


def _pg_connect():
    import psycopg2  # type: ignore[import]
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "airflow"),
        user=os.getenv("POSTGRES_USER", "airflow"),
        password=os.getenv("POSTGRES_PASSWORD", "airflow"),
    )


def _qdrant_client():
    """Return a Qdrant client instance."""
    from qdrant_client import QdrantClient  # type: ignore[import]
    return QdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT)


# ── task functions ─────────────────────────────────────────────────────────────

def _fetch_cited_chunks(**context) -> List[Tuple[str, int]]:
    """Task 1: Fetch cited chunk_ids from citation_tracking (last 7 days).

    Returns a list of (chunk_id, citation_count) tuples sorted by citation_count desc.
    Pushes the list to XCom under key 'cited_chunks'.
    """
    try:
        conn = _pg_connect()
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, COUNT(*) AS citation_count
                FROM citation_tracking
                WHERE was_cited = TRUE
                  AND retrieved_at >= NOW() - INTERVAL '7 days'
                  AND chunk_id IS NOT NULL
                GROUP BY chunk_id
                ORDER BY citation_count DESC
                """
            )
            rows = cur.fetchall()
        conn.close()
        cited: List[Tuple[str, int]] = [(str(r[0]), int(r[1])) for r in rows if r[0]]
        logger.info("[qdrant_boost_dag] Fetched %d cited chunk_id(s) from last 7 days.", len(cited))
    except Exception as exc:
        logger.warning("[qdrant_boost_dag] Could not fetch cited chunks: %s", exc)
        cited = []

    context["task_instance"].xcom_push(key="cited_chunks", value=cited)
    return cited


def _boost_qdrant_payloads(**context) -> Dict[str, Any]:
    """Task 2: Increment boost_factor in Qdrant payloads for cited chunks.

    For each (chunk_id, citation_count) pair:
    - Retrieve the current payload from Qdrant.
    - Increment boost_factor by BOOST_INCREMENT * citation_count.
    - Cap at BOOST_MAX.
    - Set the updated payload back via set_payload.

    Skips chunks whose Qdrant ID cannot be resolved (graceful degradation).
    """
    cited_chunks: List[Tuple[str, int]] = (
        context["task_instance"].xcom_pull(key="cited_chunks") or []
    )
    if not cited_chunks:
        logger.info("[qdrant_boost_dag] No cited chunks to boost.")
        return {"boosted": 0, "skipped": 0}

    try:
        client = _qdrant_client()
    except Exception as exc:
        logger.warning("[qdrant_boost_dag] Cannot connect to Qdrant: %s", exc)
        return {"boosted": 0, "skipped": len(cited_chunks), "error": str(exc)}

    boosted = 0
    skipped = 0
    boost_deltas: List[float] = []

    for chunk_id, citation_count in cited_chunks:
        try:
            # chunk_id may be a UUID string or an integer string — try both
            try:
                qdrant_id = int(chunk_id)
            except ValueError:
                qdrant_id = chunk_id  # type: ignore[assignment]

            # Retrieve current point payload
            results = client.retrieve(
                collection_name=_QDRANT_COLLECTION,
                ids=[qdrant_id],
                with_payload=True,
                with_vectors=False,
            )
            if not results:
                logger.debug(
                    "[qdrant_boost_dag] chunk_id=%s not found in Qdrant — skipping.",
                    chunk_id,
                )
                skipped += 1
                continue

            point = results[0]
            payload = dict(point.payload or {})

            # Compute new boost_factor
            current_boost = float(payload.get("boost_factor", 1.0))
            delta = _BOOST_INCREMENT * citation_count
            new_boost = min(current_boost + delta, _BOOST_MAX)
            payload["boost_factor"] = new_boost

            # Upsert the updated payload
            client.set_payload(
                collection_name=_QDRANT_COLLECTION,
                payload={"boost_factor": new_boost},
                points=[qdrant_id],
            )
            boosted += 1
            boost_deltas.append(delta)
            logger.debug(
                "[qdrant_boost_dag] chunk_id=%s  boost: %.3f → %.3f  (Δ=%.3f, citations=%d)",
                chunk_id, current_boost, new_boost, delta, citation_count,
            )

        except Exception as exc:
            logger.warning(
                "[qdrant_boost_dag] Failed to boost chunk_id=%s: %s", chunk_id, exc
            )
            skipped += 1

    avg_delta = sum(boost_deltas) / len(boost_deltas) if boost_deltas else 0.0
    summary = {
        "boosted": boosted,
        "skipped": skipped,
        "avg_boost_delta": round(avg_delta, 4),
        "total_cited_chunks": len(cited_chunks),
    }
    logger.info("[qdrant_boost_dag] Boost complete: %s", summary)
    context["task_instance"].xcom_push(key="boost_summary", value=summary)
    return summary


def _log_boost_summary(**context) -> None:
    """Task 3: Emit a readable summary of the weekly boost run."""
    boost_summary = context["task_instance"].xcom_pull(key="boost_summary") or {}
    cited_chunks: List[Tuple[str, int]] = (
        context["task_instance"].xcom_pull(key="cited_chunks") or []
    )

    # Distribution of citation counts
    if cited_chunks:
        counts = [c for _, c in cited_chunks]
        dist = Counter(counts)
        dist_str = ", ".join(f"{k}x cited: {v} chunks" for k, v in sorted(dist.items()))
    else:
        dist_str = "no cited chunks"

    logger.info(
        "[qdrant_boost_dag] ══ WEEKLY QDRANT BOOST SUMMARY ══\n"
        "  Cited chunks fetched: %d\n"
        "  Successfully boosted: %d\n"
        "  Skipped (not in Qdrant): %d\n"
        "  Average boost delta: %.4f\n"
        "  Citation distribution: %s",
        len(cited_chunks),
        boost_summary.get("boosted", 0),
        boost_summary.get("skipped", 0),
        boost_summary.get("avg_boost_delta", 0.0),
        dist_str,
    )


# ── DAG definition ─────────────────────────────────────────────────────────────

if _AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="qdrant_boost_weekly",
        description="Weekly Qdrant boost: increment boost_factor for cited chunks (C1 Self-Improving RAG)",
        schedule_interval="0 3 * * 0",  # 03:00 UTC every Sunday
        start_date=datetime(2025, 1, 5, tzinfo=timezone.utc),  # first Sunday in 2025
        catchup=False,
        default_args=_DEFAULT_ARGS,
        tags=["qdrant", "self-improving-rag", "weekly"],
        max_active_runs=1,
    ) as dag:

        task_fetch_cited = PythonOperator(
            task_id="fetch_cited_chunks",
            python_callable=_fetch_cited_chunks,
            provide_context=True,
        )

        task_boost = PythonOperator(
            task_id="boost_qdrant_payloads",
            python_callable=_boost_qdrant_payloads,
            provide_context=True,
        )

        task_summary = PythonOperator(
            task_id="log_boost_summary",
            python_callable=_log_boost_summary,
            provide_context=True,
        )

        # Linear pipeline: fetch → boost → summary
        task_fetch_cited >> task_boost >> task_summary
