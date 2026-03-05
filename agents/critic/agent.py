"""Critic Agent — offline quality-assurance layer for the equity research pipeline.

Runs nightly via Airflow (see airflow/dags/critic_dag.py).  Performs two independent
checks on the most recent pipeline runs stored in PostgreSQL:

1. NLI Hallucination Check
   For each agent output in the run, compare every factual claim (sentence containing
   a number, %, or named entity) against the source document chunks that were retrieved.
   Uses a cross-encoder (ms-marco-MiniLM-L-6-v2) as a lightweight NLI proxy: scores
   the (claim, source_chunk) pair; if the maximum entailment score across all chunks
   < NLI_THRESHOLD the claim is flagged as unverified / potentially hallucinated.

2. Citation Utilisation Rate (CUR)
   CUR = (chunks that were actually cited in the final note) /
         (total chunks retrieved across all agents for the run)
   A low CUR (<0.30) indicates the retrieval pipeline is fetching far more context
   than the LLM uses — a signal to tighten the Qdrant top-k or boost recency decay.

Both metrics are written to the ``critic_run_log`` PostgreSQL table (see docker/init-db.sql).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── thresholds ────────────────────────────────────────────────────────────────

NLI_THRESHOLD: float = float(os.getenv("CRITIC_NLI_THRESHOLD", "0.10"))
"""Cross-encoder score below which a claim is considered unverified."""

CUR_WARN_THRESHOLD: float = float(os.getenv("CRITIC_CUR_WARN_THRESHOLD", "0.30"))
"""CUR below this value triggers a WARNING log and is stored with flag low_cur=True."""


# ── result dataclasses ────────────────────────────────────────────────────────

@dataclass
class HallucinationResult:
    run_id: str
    agent_name: str
    claim: str
    max_entailment_score: float
    verified: bool
    best_source_chunk_id: Optional[str] = None
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CURResult:
    run_id: str
    total_chunks_retrieved: int
    total_chunks_cited: int
    cur: float
    low_cur: bool
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── NLI model (cached at module level) ────────────────────────────────────────

def _get_cross_encoder():
    """Lazily load and cache the cross-encoder model."""
    if not hasattr(_get_cross_encoder, "_model"):
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]
            _get_cross_encoder._model = CrossEncoder(  # type: ignore[attr-defined]
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
            )
            logger.info("[critic] Cross-encoder loaded.")
        except Exception as exc:
            logger.warning("[critic] Cross-encoder unavailable: %s", exc)
            _get_cross_encoder._model = None  # type: ignore[attr-defined]
    return _get_cross_encoder._model  # type: ignore[attr-defined]


# ── helpers ────────────────────────────────────────────────────────────────────

def _extract_factual_claims(text: str) -> List[str]:
    """Split text into sentences; return those that contain a number, %, or proper noun.

    This is a lightweight heuristic — not full NLP.  Sentences that are pure
    narrative without any quantitative or named-entity anchor are skipped since
    they are too vague to verify against a source chunk.
    """
    # Sentence split on '.', '!', '?' followed by whitespace or end
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims: List[str] = []
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Keep if it has a digit, a percent, or a capitalised word (>= 3 chars) as proxy
        # for named entity / ticker / metric name
        has_digit = bool(re.search(r"\d", sent))
        has_percent = "%" in sent
        has_proper = bool(re.search(r"\b[A-Z][a-z]{2,}", sent))
        if has_digit or has_percent or has_proper:
            claims.append(sent)
    return claims


def _score_claim_against_chunks(
    cross_encoder,
    claim: str,
    chunks: List[str],
) -> Tuple[float, int]:
    """Return (max_score, best_chunk_index) for a claim vs. a list of source chunks.

    Falls back to simple substring overlap scoring if the cross-encoder is None.
    """
    if not chunks:
        return 0.0, -1

    if cross_encoder is not None:
        pairs = [(claim, chunk) for chunk in chunks]
        try:
            scores: List[float] = cross_encoder.predict(pairs).tolist()
            best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
            return float(scores[best_idx]), best_idx
        except Exception as exc:
            logger.debug("[critic] Cross-encoder predict failed: %s", exc)

    # Fallback: normalised token overlap
    claim_tokens = set(claim.lower().split())
    best_score, best_idx = 0.0, 0
    for i, chunk in enumerate(chunks):
        chunk_tokens = set(chunk.lower().split())
        if not chunk_tokens:
            continue
        overlap = len(claim_tokens & chunk_tokens) / len(claim_tokens | chunk_tokens)
        if overlap > best_score:
            best_score, best_idx = overlap, i
    return best_score, best_idx


# ── PostgreSQL helpers ─────────────────────────────────────────────────────────

def _pg_connect():
    """Return a live psycopg2 connection using env vars."""
    import psycopg2  # type: ignore[import]
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "airflow"),
        user=os.getenv("POSTGRES_USER", "airflow"),
        password=os.getenv("POSTGRES_PASSWORD", "airflow"),
    )


def _fetch_recent_runs(lookback_hours: int = 24) -> List[str]:
    """Return distinct run_ids from agent_run_telemetry in the last N hours."""
    try:
        conn = _pg_connect()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT run_id
                FROM agent_run_telemetry
                WHERE recorded_at >= %s
                ORDER BY run_id
                """,
                (cutoff,),
            )
            rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows if r[0]]
    except Exception as exc:
        logger.warning("[critic] Could not fetch recent run_ids: %s", exc)
        return []


def _fetch_citation_data(run_id: str) -> Tuple[int, int]:
    """Return (total_retrieved, total_cited) from citation_tracking for run_id.

    If the citation_tracking table has no run_id column, falls back to counting
    all rows in the last 24h as a proxy (graceful degradation).
    """
    try:
        conn = _pg_connect()
        with conn, conn.cursor() as cur:
            # Attempt run_id-scoped query first
            try:
                cur.execute(
                    """
                    SELECT COUNT(*), COALESCE(SUM(CASE WHEN was_cited THEN 1 ELSE 0 END), 0)
                    FROM citation_tracking
                    WHERE run_id = %s
                    """,
                    (run_id,),
                )
                row = cur.fetchone()
                total_retrieved = int(row[0]) if row else 0
                total_cited = int(row[1]) if row else 0
            except Exception:
                # Fallback: count rows from last 24h
                cur.execute(
                    """
                    SELECT COUNT(*), COALESCE(SUM(CASE WHEN was_cited THEN 1 ELSE 0 END), 0)
                    FROM citation_tracking
                    WHERE retrieved_at >= NOW() - INTERVAL '24 hours'
                    """
                )
                row = cur.fetchone()
                total_retrieved = int(row[0]) if row else 0
                total_cited = int(row[1]) if row else 0
        conn.close()
        return total_retrieved, total_cited
    except Exception as exc:
        logger.warning("[critic] Could not fetch citation data for run_id=%s: %s", run_id, exc)
        return 0, 0


def _write_hallucination_result(result: HallucinationResult) -> None:
    """Persist one HallucinationResult row to critic_run_log."""
    try:
        conn = _pg_connect()
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO critic_run_log
                    (run_id, agent_name, check_type, claim, score, verified,
                     best_source_chunk_id, checked_at)
                VALUES (%s, %s, 'nli_hallucination', %s, %s, %s, %s, %s)
                """,
                (
                    result.run_id,
                    result.agent_name,
                    result.claim[:1000],
                    result.max_entailment_score,
                    result.verified,
                    result.best_source_chunk_id,
                    result.checked_at,
                ),
            )
        conn.close()
    except Exception as exc:
        logger.debug("[critic] Could not write hallucination result (non-fatal): %s", exc)


def _write_cur_result(result: CURResult) -> None:
    """Persist one CURResult row to critic_run_log."""
    try:
        conn = _pg_connect()
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO critic_run_log
                    (run_id, agent_name, check_type, score, verified,
                     notes, checked_at)
                VALUES (%s, 'orchestration', 'citation_utilisation_rate', %s, %s, %s, %s)
                """,
                (
                    result.run_id,
                    result.cur,
                    not result.low_cur,
                    json.dumps({
                        "total_retrieved": result.total_chunks_retrieved,
                        "total_cited": result.total_chunks_cited,
                        "low_cur": result.low_cur,
                    }),
                    result.checked_at,
                ),
            )
        conn.close()
    except Exception as exc:
        logger.debug("[critic] Could not write CUR result (non-fatal): %s", exc)


# ── public API ─────────────────────────────────────────────────────────────────

def run_nli_hallucination_check(
    run_id: str,
    agent_outputs: Dict[str, Any],
    source_chunks: Dict[str, List[str]],
) -> List[HallucinationResult]:
    """Check agent output text for hallucinated claims vs. retrieved source chunks.

    Parameters
    ----------
    run_id:
        Unique identifier for the pipeline run (from agent_run_telemetry).
    agent_outputs:
        Mapping of agent_name → output dict (as returned by each agent).
        The function inspects common text fields: 'summary', 'qualitative_summary',
        'quantitative_summary', 'final_summary'.
    source_chunks:
        Mapping of agent_name → list of raw source chunk strings that were
        retrieved during the run (passed in by the Airflow DAG which fetches
        them from Qdrant or the PostgreSQL citation_tracking table).

    Returns
    -------
    List of HallucinationResult — one per (claim, agent) pair evaluated.
    Unverified claims have ``verified=False``.
    """
    cross_encoder = _get_cross_encoder()
    results: List[HallucinationResult] = []

    text_fields = ["summary", "qualitative_summary", "quantitative_summary", "final_summary",
                   "investment_note", "risk_assessment"]

    for agent_name, output in agent_outputs.items():
        if not isinstance(output, dict):
            continue
        chunks = source_chunks.get(agent_name, [])
        # Collect all text from known text fields
        combined_text = " ".join(
            str(output.get(f, "")) for f in text_fields if output.get(f)
        )
        if not combined_text.strip():
            logger.debug("[critic] No text found for agent=%s — skipping NLI.", agent_name)
            continue

        claims = _extract_factual_claims(combined_text)
        logger.info("[critic] agent=%s  claims=%d  chunks=%d", agent_name, len(claims), len(chunks))

        for claim in claims:
            score, best_idx = _score_claim_against_chunks(cross_encoder, claim, chunks)
            verified = score >= NLI_THRESHOLD
            best_chunk_id = str(best_idx) if best_idx >= 0 else None
            result = HallucinationResult(
                run_id=run_id,
                agent_name=agent_name,
                claim=claim,
                max_entailment_score=score,
                verified=verified,
                best_source_chunk_id=best_chunk_id,
            )
            results.append(result)
            _write_hallucination_result(result)

            if not verified:
                logger.warning(
                    "[critic] UNVERIFIED claim  run=%s  agent=%s  score=%.3f  claim=%r",
                    run_id, agent_name, score, claim[:120],
                )

    unverified_count = sum(1 for r in results if not r.verified)
    logger.info(
        "[critic] NLI check complete  run=%s  total_claims=%d  unverified=%d",
        run_id, len(results), unverified_count,
    )
    return results


def compute_citation_utilisation_rate(run_id: str) -> CURResult:
    """Compute CUR = cited_chunks / total_retrieved_chunks for a pipeline run.

    Reads ``citation_tracking`` in PostgreSQL.  Writes result to ``critic_run_log``.

    Parameters
    ----------
    run_id:
        The pipeline run identifier.

    Returns
    -------
    CURResult with cur value, low_cur flag, and chunk counts.
    """
    total_retrieved, total_cited = _fetch_citation_data(run_id)

    if total_retrieved == 0:
        logger.warning("[critic] No citation_tracking rows found for run_id=%s.", run_id)
        cur_value = 0.0
    else:
        cur_value = total_cited / total_retrieved

    low_cur = cur_value < CUR_WARN_THRESHOLD
    if low_cur:
        logger.warning(
            "[critic] LOW CUR  run=%s  cur=%.3f  retrieved=%d  cited=%d  "
            "(threshold=%.2f) — consider tightening Qdrant top-k or boosting recency decay.",
            run_id, cur_value, total_retrieved, total_cited, CUR_WARN_THRESHOLD,
        )
    else:
        logger.info(
            "[critic] CUR OK  run=%s  cur=%.3f  retrieved=%d  cited=%d",
            run_id, cur_value, total_retrieved, total_cited,
        )

    result = CURResult(
        run_id=run_id,
        total_chunks_retrieved=total_retrieved,
        total_chunks_cited=total_cited,
        cur=cur_value,
        low_cur=low_cur,
    )
    _write_cur_result(result)
    return result


def run_critic_for_run(
    run_id: str,
    agent_outputs: Optional[Dict[str, Any]] = None,
    source_chunks: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper: run both NLI check and CUR for a single pipeline run.

    Parameters
    ----------
    run_id:
        Pipeline run identifier.
    agent_outputs:
        If None, skips NLI check (used when no output data is available).
    source_chunks:
        If None, NLI check runs with empty chunk list (all claims will be
        marked unverified due to no evidence — use only when chunks unavailable).

    Returns
    -------
    Summary dict with keys: run_id, nli_results, cur_result, unverified_count.
    """
    nli_results: List[HallucinationResult] = []
    if agent_outputs is not None:
        nli_results = run_nli_hallucination_check(
            run_id=run_id,
            agent_outputs=agent_outputs,
            source_chunks=source_chunks or {},
        )

    cur_result = compute_citation_utilisation_rate(run_id)

    unverified_count = sum(1 for r in nli_results if not r.verified)
    return {
        "run_id": run_id,
        "nli_results": nli_results,
        "nli_total_claims": len(nli_results),
        "nli_unverified_count": unverified_count,
        "cur_result": cur_result,
        "cur": cur_result.cur,
        "low_cur": cur_result.low_cur,
    }


__all__ = [
    "run_nli_hallucination_check",
    "compute_citation_utilisation_rate",
    "run_critic_for_run",
    "HallucinationResult",
    "CURResult",
    "NLI_THRESHOLD",
    "CUR_WARN_THRESHOLD",
]
