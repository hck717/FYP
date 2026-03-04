"""Episodic memory for the orchestration pipeline.

Stores past query failures (INSUFFICIENT_DATA / ReAct loops) so the planner
can pre-empt known failure patterns on future runs.

PostgreSQL table: agent_episodic_memory
  - query_signature: normalised first-30-char slug of the user query
  - ticker: the resolved ticker that caused the failure
  - failure_agent: which agent produced no output (business_analyst | quant_fundamental | ...)
  - failure_reason: short reason code (e.g. INSUFFICIENT_DATA, TIMEOUT, ERROR)
  - react_iterations_used: how many ReAct passes were consumed before giving up
  - recorded_at: timestamp
  - query_embedding: 384-dim all-MiniLM-L6-v2 embedding of the full query text
                     (stored as JSONB array) for semantic similarity lookup

On lookup, the planner computes cosine similarity between the current query
embedding and stored embeddings. If similarity > SIMILARITY_THRESHOLD, the
planner injects a pre-emptive web-search signal into the plan so the
failing agent is bypassed (or the web search agent is added) from the start.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# Cosine similarity threshold above which a stored failure is considered a "match"
SIMILARITY_THRESHOLD = float(os.getenv("EPISODIC_MEMORY_SIMILARITY_THRESHOLD", "0.85"))

# Maximum number of stored memories to check (keep lookup O(n) small)
MAX_MEMORIES_TO_CHECK = int(os.getenv("EPISODIC_MEMORY_MAX_CHECK", "200"))

# Minimum number of ReAct retries that had to occur before we bother recording
MIN_ITERATIONS_TO_RECORD = int(os.getenv("EPISODIC_MEMORY_MIN_ITERATIONS", "2"))


# ---------------------------------------------------------------------------
# DB connection helpers
# ---------------------------------------------------------------------------

def _get_pg_conn():
    """Open a new psycopg2 connection using env vars (mirrors other agents)."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "airflow"),
        user=os.getenv("POSTGRES_USER", "airflow"),
        password=os.getenv("POSTGRES_PASSWORD", "airflow"),
    )


def ensure_table_exists() -> None:
    """Create the agent_episodic_memory table if it does not exist yet.

    Called lazily on first write so we never hard-fail at import time if
    Postgres is down (e.g. during unit tests).
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS agent_episodic_memory (
        id                  SERIAL PRIMARY KEY,
        query_signature     TEXT        NOT NULL,
        ticker              TEXT        NOT NULL,
        failure_agent       TEXT        NOT NULL,
        failure_reason      TEXT        NOT NULL DEFAULT 'UNKNOWN',
        react_iterations_used INT       NOT NULL DEFAULT 1,
        query_embedding     JSONB,
        recorded_at         TIMESTAMP   DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_episodic_memory_ticker
        ON agent_episodic_memory (ticker);
    CREATE INDEX IF NOT EXISTS idx_episodic_memory_agent
        ON agent_episodic_memory (failure_agent);
    """
    try:
        conn = _get_pg_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
        conn.close()
    except Exception as exc:
        logger.warning("[episodic_memory] Could not ensure table: %s", exc)


# ---------------------------------------------------------------------------
# Embedding helper (reuses the already-cached sentence-transformers model)
# ---------------------------------------------------------------------------

_embedding_cache: Dict[str, List[float]] = {}


def _embed_query(text: str) -> Optional[List[float]]:
    """Embed text with all-MiniLM-L6-v2 (384-dim) — same model as Neo4j chunks.

    Returns None gracefully if sentence_transformers is unavailable.
    """
    if text in _embedding_cache:
        return _embedding_cache[text]
    try:
        # Prefer the already-loaded model cached by LocalEmbeddingClient
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        import os as _os
        _old_hf = _os.environ.get("HF_HUB_OFFLINE")
        _os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
        finally:
            if _old_hf is None:
                _os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                _os.environ["HF_HUB_OFFLINE"] = _old_hf
        vec = model.encode(text, normalize_embeddings=True).tolist()
        _embedding_cache[text] = vec
        return vec
    except Exception as exc:
        logger.debug("[episodic_memory] Embedding unavailable: %s", exc)
        return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Dot product of two L2-normalised vectors (= cosine similarity)."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Query signature
# ---------------------------------------------------------------------------

def _query_signature(query: str) -> str:
    """Create a compact, normalised signature for deduplication logging."""
    normalised = " ".join(query.lower().split())[:120]
    return hashlib.md5(normalised.encode()).hexdigest()[:16] + "_" + normalised[:30]


# ---------------------------------------------------------------------------
# Public API: write
# ---------------------------------------------------------------------------

def record_failure(
    user_query: str,
    ticker: str,
    failure_agent: str,
    failure_reason: str = "INSUFFICIENT_DATA",
    react_iterations_used: int = 1,
) -> None:
    """Persist a query failure event to agent_episodic_memory.

    Args:
        user_query:           The original user query text.
        ticker:               The ticker that caused the failure.
        failure_agent:        Which agent failed (e.g. "business_analyst").
        failure_reason:       Short code — INSUFFICIENT_DATA | TIMEOUT | ERROR.
        react_iterations_used: Number of ReAct passes consumed.
    """
    if react_iterations_used < MIN_ITERATIONS_TO_RECORD:
        logger.debug(
            "[episodic_memory] Skipping record: only %d iteration(s), threshold=%d",
            react_iterations_used,
            MIN_ITERATIONS_TO_RECORD,
        )
        return

    sig = _query_signature(user_query)
    embedding = _embed_query(user_query)

    try:
        ensure_table_exists()
        conn = _get_pg_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_episodic_memory
                        (query_signature, ticker, failure_agent, failure_reason,
                         react_iterations_used, query_embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        sig,
                        ticker.upper(),
                        failure_agent,
                        failure_reason,
                        react_iterations_used,
                        json.dumps(embedding) if embedding is not None else None,
                    ),
                )
        conn.close()
        logger.info(
            "[episodic_memory] Recorded failure: query=%r ticker=%s agent=%s reason=%s",
            sig, ticker, failure_agent, failure_reason,
        )
    except Exception as exc:
        logger.warning("[episodic_memory] Failed to record: %s", exc)


# ---------------------------------------------------------------------------
# Public API: read
# ---------------------------------------------------------------------------

def lookup_similar_failures(
    user_query: str,
    tickers: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return stored failures that are semantically similar to user_query.

    Uses cosine similarity on all-MiniLM-L6-v2 embeddings.
    Falls back to ticker-only exact match if embeddings are unavailable.

    Args:
        user_query: Current query text.
        tickers:    Optional list of resolved tickers to scope the search.

    Returns:
        List of dicts with keys: ticker, failure_agent, failure_reason,
        react_iterations_used, similarity.  Sorted by similarity descending.
    """
    query_vec = _embed_query(user_query)

    try:
        ensure_table_exists()
        conn = _get_pg_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if tickers:
                placeholders = ",".join(["%s"] * len(tickers))
                cur.execute(
                    f"""
                    SELECT ticker, failure_agent, failure_reason,
                           react_iterations_used, query_embedding
                    FROM agent_episodic_memory
                    WHERE ticker = ANY(ARRAY[{placeholders}]::text[])
                    ORDER BY recorded_at DESC
                    LIMIT %s
                    """,
                    [t.upper() for t in tickers] + [MAX_MEMORIES_TO_CHECK],
                )
            else:
                cur.execute(
                    """
                    SELECT ticker, failure_agent, failure_reason,
                           react_iterations_used, query_embedding
                    FROM agent_episodic_memory
                    ORDER BY recorded_at DESC
                    LIMIT %s
                    """,
                    (MAX_MEMORIES_TO_CHECK,),
                )
            rows = cur.fetchall()
        conn.close()
    except Exception as exc:
        logger.warning("[episodic_memory] Lookup failed: %s", exc)
        return []

    matches: List[Dict[str, Any]] = []
    for row in rows:
        sim = 0.0
        if query_vec is not None and row.get("query_embedding"):
            try:
                stored_vec = (
                    row["query_embedding"]
                    if isinstance(row["query_embedding"], list)
                    else json.loads(row["query_embedding"])
                )
                sim = _cosine_similarity(query_vec, stored_vec)
            except Exception:
                pass
        elif query_vec is None:
            # No embedding available — fall back: treat every stored failure
            # for the same ticker as a "match" with similarity 1.0
            if tickers and row["ticker"].upper() in [t.upper() for t in tickers]:
                sim = 1.0

        if sim >= SIMILARITY_THRESHOLD:
            matches.append(
                {
                    "ticker": row["ticker"],
                    "failure_agent": row["failure_agent"],
                    "failure_reason": row["failure_reason"],
                    "react_iterations_used": row["react_iterations_used"],
                    "similarity": round(sim, 4),
                }
            )

    matches.sort(key=lambda x: x["similarity"], reverse=True)
    logger.debug(
        "[episodic_memory] %d match(es) (similarity>%.2f) for query=%r",
        len(matches), SIMILARITY_THRESHOLD, user_query[:60],
    )
    return matches


def build_preemptive_plan_hints(
    failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Translate past failures into planner hints.

    Returns a dict consumed by node_planner:
      - force_web_search: bool — if True, add run_web_search=True regardless of complexity
      - degraded_agents: List[str] — agents known to fail for this query pattern
    """
    if not failures:
        return {}

    force_web_search = any(
        f["failure_agent"] in ("business_analyst", "quant_fundamental")
        and f["failure_reason"] == "INSUFFICIENT_DATA"
        for f in failures
    )
    degraded_agents = list(dict.fromkeys(f["failure_agent"] for f in failures))

    return {
        "force_web_search": force_web_search,
        "degraded_agents": degraded_agents,
    }


__all__ = [
    "record_failure",
    "lookup_similar_failures",
    "build_preemptive_plan_hints",
    "ensure_table_exists",
    "SIMILARITY_THRESHOLD",
]
