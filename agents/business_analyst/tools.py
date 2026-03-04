"""Connectors and helper utilities for the Business Analyst agent."""

from __future__ import annotations

import json
import logging
import math
import os
import re as _re_top
from contextlib import closing
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg2
from neo4j import GraphDatabase, basic_auth
from psycopg2.extras import RealDictCursor
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import requests

from .config import BusinessAnalystConfig
from .schema import Chunk, CRAGStatus, RetrievalResult, SentimentSnapshot

# Cross-encoder reranker — loaded lazily so import never hard-fails.
_CrossEncoder = None  # type: ignore[assignment]
# sentence-transformers SentenceTransformer — loaded lazily for the same reason.
_SentenceTransformer = None  # type: ignore[assignment]

def _get_cross_encoder_class():
    global _CrossEncoder
    if _CrossEncoder is None:
        try:
            from sentence_transformers import CrossEncoder as _CE  # type: ignore[import]
            _CrossEncoder = _CE
        except ImportError:
            pass
    return _CrossEncoder

def _get_sentence_transformer_class():
    global _SentenceTransformer
    if _SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as _ST  # type: ignore[import]
            _SentenceTransformer = _ST
        except ImportError:
            pass
    return _SentenceTransformer

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------------
class EmbeddingClient:
    """Thin wrapper around Ollama's embedding endpoint."""

    def __init__(self, base_url: str, model: str, timeout: Optional[int]) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, text: str) -> List[float]:
        # Try new /api/embed endpoint first (Ollama >= 0.1.26, e.g. nomic-embed-text).
        # Only fall back to the legacy /api/embeddings endpoint if the primary call
        # fails with a non-404 error (e.g. empty vector returned).  A 404 on
        # /api/embed means the Ollama instance is too old or the model is not loaded;
        # in that case /api/embeddings will also 404 (it was removed in Ollama ≥ 0.2),
        # so we surface the original error immediately instead of masking it.
        primary_exc: Optional[Exception] = None
        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings")
            if embeddings and len(embeddings) > 0 and embeddings[0]:
                return embeddings[0]
            # Endpoint responded but returned an empty vector — note and fall through
            primary_exc = RuntimeError(
                f"/api/embed returned empty embeddings for model '{self.model}'. "
                "Ensure the model is pulled: `ollama pull {self.model}`"
            )
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                # 404 on /api/embed → do NOT try /api/embeddings (also absent in modern Ollama)
                raise RuntimeError(
                    f"Ollama embedding endpoint /api/embed returned 404 for model '{self.model}'. "
                    "Ensure Ollama ≥ 0.1.26 is running and the model is pulled: "
                    f"`ollama pull {self.model}`"
                ) from exc
            primary_exc = exc
        except Exception as exc:
            primary_exc = exc

        if primary_exc is not None:
            logger.debug(
                "EmbeddingClient /api/embed failed (%s), trying legacy /api/embeddings", primary_exc
            )

        # Fallback to legacy /api/embeddings endpoint (Ollama < 0.1.26)
        try:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            embedding = resp.json().get("embedding")
            if not embedding:
                raise RuntimeError("Embedding endpoint returned empty vector")
            return embedding
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                # Both endpoints returned 404 — surface the original /api/embed error
                if primary_exc is not None:
                    raise RuntimeError(
                        f"Both /api/embed and /api/embeddings returned 404 for model '{self.model}'. "
                        "Ensure Ollama ≥ 0.1.26 is running and the model is pulled: "
                        f"`ollama pull {self.model}`"
                    ) from primary_exc
            raise


# ----------------------------------------------------------------------------
# Local sentence-transformers embedding client (for models NOT served via Ollama,
# e.g. all-MiniLM-L6-v2 used for Neo4j chunk ingestion vectors).
# ----------------------------------------------------------------------------
class LocalEmbeddingClient:
    """Embedding client that runs sentence-transformers locally (no Ollama required).

    Used for the Neo4j vector index, which was built with all-MiniLM-L6-v2
    (384-dim) via sentence_transformers, not via Ollama.  Falls back to the
    Ollama-backed EmbeddingClient if sentence_transformers is not installed.
    """

    _model_cache: dict = {}  # class-level cache so the model is loaded once per process

    def __init__(self, model: str, fallback: Optional["EmbeddingClient"] = None) -> None:
        self.model_name = model
        self.fallback = fallback
        self._st_model = None
        self._load_attempted = False

    def _load(self):
        if self._load_attempted:
            return
        self._load_attempted = True
        ST = _get_sentence_transformer_class()
        if ST is None:
            logger.warning(
                "sentence_transformers not installed — LocalEmbeddingClient will use fallback."
            )
            return
        if self.model_name in self.__class__._model_cache:
            self._st_model = self.__class__._model_cache[self.model_name]
            return
        import os as _os
        _old_tqdm = _os.environ.get("TQDM_DISABLE")
        _old_hf_offline = _os.environ.get("HF_HUB_OFFLINE")
        _os.environ["TQDM_DISABLE"] = "1"
        _os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self._st_model = ST(self.model_name)
            self.__class__._model_cache[self.model_name] = self._st_model
            logger.info("LocalEmbeddingClient loaded model: %s", self.model_name)
        except Exception as exc:
            logger.warning(
                "LocalEmbeddingClient failed to load '%s' (%s) — will use fallback.",
                self.model_name, exc,
            )
        finally:
            if _old_tqdm is None:
                _os.environ.pop("TQDM_DISABLE", None)
            else:
                _os.environ["TQDM_DISABLE"] = _old_tqdm
            if _old_hf_offline is None:
                _os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                _os.environ["HF_HUB_OFFLINE"] = _old_hf_offline

    def embed(self, text: str) -> List[float]:
        self._load()
        if self._st_model is not None:
            try:
                vec = self._st_model.encode(text, normalize_embeddings=True)
                return vec.tolist()
            except Exception as exc:
                logger.warning("LocalEmbeddingClient encode failed (%s) — trying fallback.", exc)
        if self.fallback is not None:
            return self.fallback.embed(text)
        raise RuntimeError(
            f"LocalEmbeddingClient: sentence_transformers model '{self.model_name}' unavailable "
            "and no fallback configured."
        )


# ----------------------------------------------------------------------------
# Neo4j
# ----------------------------------------------------------------------------
class Neo4jConnector:
    def __init__(self, config: BusinessAnalystConfig) -> None:
        self.config = config
        uri = config.neo4j_uri
        auth = basic_auth(config.neo4j_user, config.neo4j_password)
        self.driver = GraphDatabase.driver(uri, auth=auth, encrypted=config.neo4j_verify)

    def fetch_company_overview(self, ticker: Optional[str]) -> Optional[Dict[str, Any]]:
        if not ticker:
            return None
        cypher = """
        MATCH (c:Company {ticker:$ticker})
        RETURN c LIMIT 1
        """
        with self.driver.session(database=None) as session:
            record = session.run(cypher, ticker=ticker).single()
            if not record:
                return None
            node = record["c"]
            return dict(node)

    def fetch_graph_facts(self, ticker: Optional[str], limit: int = 25) -> List[Dict[str, Any]]:
        if not ticker:
            return []
        cypher = """
        MATCH (c:Company {ticker:$ticker})-[r:FACES_RISK|HAS_STRATEGY|COMPETES_WITH|HAS_FACT]->(n)
        RETURN type(r) AS rel_type, properties(n) AS node_props, properties(r) AS rel_props
        LIMIT $limit
        """
        with self.driver.session(database=None) as session:
            rows = session.run(cypher, ticker=ticker, limit=limit)
            facts: List[Dict[str, Any]] = []
            for row in rows:
                facts.append(
                    {
                        "relationship": row["rel_type"],
                        "node": row["node_props"],
                        "relationship_properties": row["rel_props"],
                    }
                )
            return facts

    def vector_search(
        self,
        vector: Sequence[float],
        ticker: Optional[str],
        top_k: int,
        index_name: Optional[str] = None,
    ) -> List[Chunk]:
        index_name = index_name or self.config.neo4j_chunk_index
        cypher = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $vector)
        YIELD node, score
        WITH node, score
        WHERE $ticker IS NULL OR node.ticker = $ticker
        RETURN node.chunk_id AS chunk_id,
               node.text AS text,
               node.section AS section,
               node.filing_date AS filing_date,
               node.ticker AS ticker_symbol,
               score
        LIMIT $top_k
        """
        with self.driver.session(database=None) as session:
            rows = session.run(
                cypher,
                index_name=index_name,
                top_k=top_k,
                vector=vector,
                ticker=ticker,
            ).data()
        chunks: List[Chunk] = []
        for row in rows:
            metadata = {
                "section": row.get("section"),
                "filing_date": row.get("filing_date"),
                "ticker": row.get("ticker_symbol"),
            }
            chunk_id = row.get("chunk_id") or f"neo4j::{metadata['ticker']}::{metadata['section']}::{len(chunks)}"
            score = self._normalise_score(row.get("score"))
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=row.get("text", ""),
                    score=score,
                    source="neo4j",
                    metadata=metadata,
                )
            )
        return chunks

    @staticmethod
    def _normalise_score(raw: Optional[float]) -> float:
        if raw is None:
            return 0.0
        if 0 <= raw <= 1:
            return raw
        # Neo4j returns cosine distance (0 good) → convert to similarity
        if raw >= 0:
            return max(0.0, 1.0 - raw)
        return 0.0

    def insert_chunks(
        self,
        ticker: str,
        chunks: List[Dict[str, Any]],
        embedding_client: Optional["EmbeddingClient"] = None,
    ) -> int:
        rows = []
        for chunk in chunks:
            row = dict(chunk)
            if embedding_client is not None and "embedding" not in row:
                try:
                    row["embedding"] = embedding_client.embed(row["text"])
                except Exception as exc:
                    logger.warning("Embedding failed for chunk %s: %s", row.get("chunk_id"), exc)
                    row["embedding"] = None
            rows.append(row)

        cypher = """
        UNWIND $rows AS row
        MERGE (c:Company {ticker: row.ticker})
        MERGE (chunk:Chunk {chunk_id: row.chunk_id})
          SET chunk.text = row.text,
              chunk.section = row.section,
              chunk.filing_date = row.filing_date,
              chunk.ticker = row.ticker,
              chunk.embedding = row.embedding
        MERGE (c)-[:HAS_CHUNK]->(chunk)
        """
        with self.driver.session(database=None) as session:
            session.run(cypher, rows=rows)
        return len(rows)

    def fetch_community_summary(self, ticker: Optional[str]) -> Optional[str]:
        """Build a graph-community summary for *ticker* using relationship-count centrality.

        Avoids GDS PageRank (community edition has no GDS plugin).  Instead we
        count outgoing relationship types from the Company node and collect a
        few sample neighbour property snippets to characterise the local
        community.

        Returns a human-readable string like:
            "Apple (AAPL) is most centrally connected via HAS_STRATEGY (12 edges),
             FACES_RISK (8 edges), COMPETES_WITH (5 edges).  Top connected entities:
             [{'name': 'AI Integration', 'type': 'Strategy'}, ...]"
        or *None* if Neo4j is unreachable or no Company node exists.
        """
        if not ticker:
            return None
        cypher = """
        MATCH (c:Company {ticker: $ticker})-[r]->(n)
        WITH type(r)        AS rel_type,
             count(*)       AS rel_count,
             collect(properties(n))[..3] AS sample_nodes
        ORDER BY rel_count DESC
        LIMIT 10
        RETURN rel_type, rel_count, sample_nodes
        """
        try:
            with self.driver.session(database=None) as session:
                result = session.run(cypher, ticker=ticker)
                rows = [dict(rec) for rec in result]
        except Exception as exc:  # pragma: no cover
            logger.warning("[Neo4j] community summary query failed for %s: %s", ticker, exc)
            return None

        if not rows:
            return None

        # Build a compact, readable summary string
        rel_parts: List[str] = []
        all_samples: List[Dict[str, Any]] = []
        for row in rows:
            rel_type = row.get("rel_type", "UNKNOWN")
            count = row.get("rel_count", 0)
            rel_parts.append(f"{rel_type} ({count} edges)")
            # Flatten sample neighbour nodes, dropping embedding/vector fields
            for node in row.get("sample_nodes") or []:
                clean = {
                    k: v for k, v in node.items()
                    if k not in ("embedding", "vector") and v is not None
                }
                if clean:
                    all_samples.append(clean)

        company_name = ticker  # fallback to ticker symbol
        # Try to fetch the Company node name for a friendlier label
        try:
            with self.driver.session(database=None) as session:
                name_result = session.run(
                    "MATCH (c:Company {ticker: $ticker}) RETURN c.name AS name LIMIT 1",
                    ticker=ticker,
                )
                name_row = name_result.single()
                if name_row and name_row.get("name"):
                    company_name = name_row["name"]
        except Exception:  # pragma: no cover
            pass

        rel_summary = ", ".join(rel_parts[:5]) if rel_parts else "no outgoing relationships found"
        sample_str = json.dumps(all_samples[:5], ensure_ascii=False) if all_samples else "[]"
        return (
            f"{company_name} ({ticker}) is most centrally connected via {rel_summary}. "
            f"Top connected entities: {sample_str}"
        )

    def close(self) -> None:
        self.driver.close()


# ----------------------------------------------------------------------------
# PostgreSQL
# ----------------------------------------------------------------------------
class PostgresConnector:
    def __init__(self, config: BusinessAnalystConfig) -> None:
        self.config = config

    def fetch_sentiment(self, ticker: Optional[str]) -> Optional[SentimentSnapshot]:
        if not ticker:
            return None
        # NOTE: live DB uses 'date' column; 'trend' added via migration when present
        sql = """
        SELECT bullish_pct, bearish_pct, neutral_pct,
               COALESCE(trend, 'unknown') AS trend
        FROM sentiment_trends
        WHERE ticker = %s
        ORDER BY date DESC
        LIMIT 1
        """
        conn = psycopg2.connect(
            host=self.config.postgres_host,
            port=self.config.postgres_port,
            dbname=self.config.postgres_db,
            user=self.config.postgres_user,
            password=self.config.postgres_password,
        )
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker,))
            row = cur.fetchone()
            if not row:
                return None
            return SentimentSnapshot(
                bullish_pct=float(row.get("bullish_pct", 0.0)),
                bearish_pct=float(row.get("bearish_pct", 0.0)),
                neutral_pct=float(row.get("neutral_pct", 0.0)),
                trend=row.get("trend", "unknown"),
            )


# ----------------------------------------------------------------------------
# Qdrant (news fallback)
# ----------------------------------------------------------------------------
class QdrantConnector:
    def __init__(self, config: BusinessAnalystConfig) -> None:
        self.config = config
        self.client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)

    def vector_search(
        self,
        vector: Sequence[float],
        ticker: Optional[str],
        top_k: int,
    ) -> List[Chunk]:
        qdrant_filter: Optional[qmodels.Filter] = None
        if ticker:
            qdrant_filter = qmodels.Filter(
                must=[qmodels.FieldCondition(
                    key="ticker_symbol",
                    match=qmodels.MatchValue(value=ticker),
                )]
            )
        response = self.client.query_points(
            collection_name=self.config.qdrant_collection,
            query=list(vector),
            limit=top_k * 6,  # Over-fetch heavily to get top_k *unique* articles after dedup
            with_payload=True,
            query_filter=qdrant_filter,
        )
        hits = response.points
        chunks: List[Chunk] = []
        seen_ids: set = set()
        seen_title_keys: set = set()  # normalised title key for near-duplicate suppression
        for hit in hits:
            payload = hit.payload or {}
            # Build a human-readable chunk_id from ticker + title slug when no stored chunk_id
            if payload.get("chunk_id"):
                chunk_id = payload["chunk_id"]
            else:
                ticker_sym = payload.get("ticker_symbol", "")
                title_raw = payload.get("title") or ""
                title_slug = title_raw[:40].replace(" ", "_").replace("/", "-")
                chunk_id = f"qdrant::{ticker_sym}::{title_slug}" if title_slug else f"qdrant::{hit.id}"
            # Primary dedup: exact chunk_id
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            # Secondary dedup: normalised title key (strips punctuation, lowercases) so
            # near-identical articles like "Tesla Suing California..." vs
            # "Tesla Is Suing California..." collapse to one entry.
            title_raw = payload.get("title") or ""
            import re as _re
            title_key = _re.sub(r"[^a-z0-9]", "", title_raw.lower())[:60]
            if title_key and title_key in seen_title_keys:
                continue
            if title_key:
                seen_title_keys.add(title_key)
            text = payload.get("content") or payload.get("text") or payload.get("summary") or ""
            metadata = {
                k: payload.get(k)
                for k in ["ticker_symbol", "data_name", "source", "section", "filing_date"]
                if payload.get(k) is not None
            }
            score = hit.score or 0.0
            score = score if 0 <= score <= 1 else max(0.0, 1.0 - score)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=score,
                    source="qdrant",
                    metadata=metadata,
                )
            )
            if len(chunks) >= top_k:
                break
        return chunks


# ----------------------------------------------------------------------------
# Keyword scorer (BM25-lite)
# ----------------------------------------------------------------------------
def keyword_overlap_score(text: str, query: str) -> float:
    text_tokens = set(_tokenise(text))
    query_tokens = set(_tokenise(query))
    if not text_tokens or not query_tokens:
        return 0.0
    overlap = len(text_tokens & query_tokens)
    return overlap / len(query_tokens)


def _tokenise(text: str) -> List[str]:
    return [t.lower() for t in text.split() if t.isalpha()]


def _sigmoid_normalise(scores: List[float]) -> List[float]:
    """Map raw cross-encoder logits to [0, 1] via sigmoid."""
    import math as _math
    return [1.0 / (1.0 + _math.exp(-s)) for s in scores]


# ----------------------------------------------------------------------------
# Ticker identity helper
# ----------------------------------------------------------------------------

def _chunk_ticker_matches(chunk: "Chunk", ticker: str) -> bool:
    """Return True if the chunk provably belongs to *ticker*.

    Checks (in priority order):
    1. ``chunk.metadata["ticker"]``       — set by Neo4jConnector
    2. ``chunk.metadata["ticker_symbol"]`` — set by QdrantConnector
    3. ``chunk.chunk_id`` prefix heuristic — e.g. ``qdrant::AAPL::…`` or
       ``neo4j::AAPL::…``

    Returns True when the ticker is *unknown* (missing from all three
    sources) to avoid silently discarding potentially valid chunks.
    """
    meta = chunk.metadata or {}
    for key in ("ticker", "ticker_symbol"):
        val = meta.get(key)
        if val:
            return str(val).upper() == ticker.upper()
    # Fallback: parse chunk_id  (format: <source>::<TICKER>::<rest>)
    parts = chunk.chunk_id.split("::")
    if len(parts) >= 2:
        candidate = parts[1].upper()
        # Only trust the heuristic when it looks like a real ticker symbol
        if 1 <= len(candidate) <= 6 and candidate.isalpha():
            return candidate == ticker.upper()
    # Cannot determine ticker from available metadata — let it through
    return True


# ----------------------------------------------------------------------------
# Hybrid retriever & evaluator
# ----------------------------------------------------------------------------
class HybridRetriever:
    """Dense + sparse + graph retrieval with cross-encoder reranking.

    Reranking weight mix:
        final_score = BM25_WEIGHT * lexical + CE_WEIGHT * cross_encoder
    where BM25_WEIGHT = 0.30 and CE_WEIGHT = 0.70 (mirrors README spec).

    If sentence-transformers is unavailable or the model fails to load,
    the reranker silently degrades to BM25-only scoring with a warning.
    """

    BM25_WEIGHT: float = 0.30
    CE_WEIGHT: float = 0.70

    def __init__(
        self,
        config: BusinessAnalystConfig,
        embedding_client: Any,
        neo4j: Neo4jConnector,
        qdrant: QdrantConnector,
        qdrant_embedding_client: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.embedding_client = embedding_client
        self.neo4j = neo4j
        self.qdrant = qdrant
        # Use a separate embedder for Qdrant if dimensions differ (e.g. nomic-embed-text 768-dim)
        self.qdrant_embedding_client = qdrant_embedding_client or embedding_client
        # Deferred — loaded on first call to _cross_encoder_scores() to avoid
        # the sentence_transformers / torch / sympy / mpmath import chain at startup
        # (hangs on Python 3.14 during mpmath ctx_mp.py:init_builtins()).
        self._cross_encoder = None
        self._ce_load_attempted: bool = False

    # ------------------------------------------------------------------
    # Cross-encoder loader
    # ------------------------------------------------------------------

    def _load_cross_encoder(self):
        """Load cross-encoder model; return None on any failure."""
        import os as _os
        import sys as _sys
        CE = _get_cross_encoder_class()
        if CE is None:
            logger.warning("sentence-transformers not installed — reranker disabled, using BM25-only")
            return None
        try:
            # Suppress verbose "LOAD REPORT" and tqdm progress bar by redirecting
            # file descriptors 1 (stdout) and 2 (stderr) to /dev/null at the OS level.
            # Python-level sys.stdout/stderr redirect is insufficient because
            # sentence_transformers / tqdm write directly to the underlying fd.
            _old_tqdm = _os.environ.get("TQDM_DISABLE")
            _old_hf_offline = _os.environ.get("HF_HUB_OFFLINE")
            _os.environ["TQDM_DISABLE"] = "1"
            # Force local cache — prevents HuggingFace Hub remote fetch (which times out
            # when the model is already cached locally).
            _os.environ["HF_HUB_OFFLINE"] = "1"
            _devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
            _saved_stdout = _os.dup(1)
            _saved_stderr = _os.dup(2)
            _os.dup2(_devnull_fd, 1)
            _os.dup2(_devnull_fd, 2)
            _os.close(_devnull_fd)
            try:
                model = CE(self.config.reranker_model)
            finally:
                _os.dup2(_saved_stdout, 1)
                _os.dup2(_saved_stderr, 2)
                _os.close(_saved_stdout)
                _os.close(_saved_stderr)
                if _old_tqdm is None:
                    _os.environ.pop("TQDM_DISABLE", None)
                else:
                    _os.environ["TQDM_DISABLE"] = _old_tqdm
                if _old_hf_offline is None:
                    _os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    _os.environ["HF_HUB_OFFLINE"] = _old_hf_offline
            logger.info("Cross-encoder loaded: %s", self.config.reranker_model)
            return model
        except Exception as exc:
            logger.warning("Cross-encoder load failed (%s) — falling back to BM25-only", exc)
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, ticker: Optional[str]) -> RetrievalResult:
        neo4j_vector = self.embedding_client.embed(query)
        chunks = self.neo4j.vector_search(neo4j_vector, ticker, self.config.top_k)
        if len(chunks) < max(1, self.config.top_k // 2):
            qdrant_vector = self.qdrant_embedding_client.embed(query)
            chunks.extend(self.qdrant.vector_search(qdrant_vector, ticker, self.config.top_k))

        # Post-merge ticker guard: drop any chunk that belongs to a different company.
        # This catches (a) Neo4j chunks that slipped through the post-hoc WHERE filter
        # when the ANN window was dominated by other tickers, and (b) Qdrant chunks
        # where the payload field name may have differed from the expected key.
        if ticker:
            before = len(chunks)
            chunks = [c for c in chunks if _chunk_ticker_matches(c, ticker)]
            dropped = before - len(chunks)
            if dropped:
                logger.warning(
                    "[HybridRetriever] Dropped %d cross-ticker chunk(s) for ticker=%s "
                    "after post-merge filter.",
                    dropped,
                    ticker,
                )

        ranked, bm25_debug = self._rerank(query, chunks)
        facts = self.neo4j.fetch_graph_facts(ticker, limit=25)
        return RetrievalResult(chunks=ranked, graph_facts=facts, bm25_debug=bm25_debug)

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def _rerank(self, query: str, chunks: List[Chunk]) -> Tuple[List[Chunk], Dict[str, float]]:
        """Blend BM25 lexical score (30%) with cross-encoder score (70%).

        Enhancements:
          2B — Time-Decayed Contrastive RAG:
               Apply exponential time-decay to base scores:
               ``final_score = base_score * exp(-DECAY_LAMBDA * days_old)``
               where DECAY_LAMBDA = 0.005 (≈ half-life ~139 days).
               Chunks are tagged ``temporal_band = "recent"`` (≤30 days) or
               ``"historical"`` (≥335 days) in metadata so the LLM context can
               analyse the delta between the two windows.

          2C — Dynamic Reranker Weighting:
               If the query contains specific alphanumeric product/model strings
               (e.g. "M2 Ultra", "H100", "RTX 4090") the CE weight shifts to 0.40
               and BM25 to 0.60 — precise keyword matches matter more for product
               queries than semantic proximity.

        Falls back to dense + BM25 blend when cross-encoder is unavailable or
        uniformly mis-calibrated (max CE score < 0.4).
        Returns a (sorted_chunks, bm25_debug) tuple where bm25_debug maps
        chunk_id → raw BM25 token-overlap score (before blending).
        """
        if not chunks:
            return chunks, {}

        # 2C: Detect product/model-specific query — shift weights to BM25-heavy
        _PRODUCT_RE = _re_top.compile(
            r"\b[A-Z][A-Za-z0-9]*\s*\d+\s*[A-Za-z]*\b"  # e.g. M2 Ultra, H100, RTX4090
        )
        is_product_query = bool(_PRODUCT_RE.search(query))
        if is_product_query:
            ce_w = 0.40
            bm25_w = 0.60
            logger.debug("_rerank: product query detected — CE=0.40 / BM25=0.60")
        else:
            ce_w = self.CE_WEIGHT   # 0.70
            bm25_w = self.BM25_WEIGHT  # 0.30

        # Snapshot the incoming dense scores before any modification
        dense_scores = [float(min(1.0, c.score)) for c in chunks]
        for i, chunk in enumerate(chunks):
            chunk.score = dense_scores[i]

        # 2B: Compute time-decay factor for each chunk
        _DECAY_LAMBDA = 0.005  # half-life ≈ 139 days
        _now = datetime.now(tz=timezone.utc)
        time_decay_factors: List[float] = []
        for chunk in chunks:
            filing_date_raw = (chunk.metadata or {}).get("filing_date") or (chunk.metadata or {}).get("published_at")
            days_old: float = 0.0
            temporal_band: Optional[str] = None
            if filing_date_raw:
                try:
                    if isinstance(filing_date_raw, str):
                        # Parse ISO-8601 or YYYY-MM-DD
                        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
                            try:
                                dt = datetime.strptime(filing_date_raw[:19], fmt[:len(filing_date_raw[:19])])
                                if dt.tzinfo is None:
                                    dt = dt.replace(tzinfo=timezone.utc)
                                days_old = max(0.0, (_now - dt).days)
                                break
                            except ValueError:
                                continue
                    elif isinstance(filing_date_raw, (int, float)):
                        # Unix timestamp
                        dt = datetime.fromtimestamp(filing_date_raw, tz=timezone.utc)
                        days_old = max(0.0, (_now - dt).days)
                except Exception:
                    pass
            # Tag temporal band for contrastive analysis
            if days_old <= 30:
                temporal_band = "recent"
            elif days_old >= 335:
                temporal_band = "historical"
            if temporal_band and chunk.metadata is not None:
                chunk.metadata["temporal_band"] = temporal_band
            decay = math.exp(-_DECAY_LAMBDA * days_old)
            time_decay_factors.append(decay)

        bm25_scores = [keyword_overlap_score(c.text, query) for c in chunks]

        # Attempt cross-encoder scoring
        ce_scores = self._cross_encoder_scores(query, chunks)

        # Only use CE when scores are meaningful (max >= 0.4).
        use_ce = ce_scores is not None and max(ce_scores) >= 0.4

        for i, chunk in enumerate(chunks):
            bm25 = bm25_scores[i]
            decay = time_decay_factors[i]
            if use_ce:
                ce = ce_scores[i]  # type: ignore[index]
                base = float(ce_w * ce + bm25_w * bm25)
            else:
                # Dense + BM25 fallback: 70% dense score + 30% BM25
                base = float(0.7 * dense_scores[i] + 0.3 * bm25)
            # Apply time-decay (2B)
            chunk.score = float(min(1.0, base * decay))

        if ce_scores is not None and not use_ce:
            logger.debug(
                "_rerank: CE max score %.3f < 0.4 — using dense+BM25 fallback "
                "(cross-encoder mis-calibrated for this corpus/query)",
                max(ce_scores),
            )

        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        bm25_debug: Dict[str, float] = {
            chunks[i].chunk_id: round(bm25_scores[i], 4) for i in range(len(chunks))
        }
        return sorted_chunks, bm25_debug

    def _cross_encoder_scores(
        self, query: str, chunks: List[Chunk]
    ) -> Optional[List[float]]:
        """Return normalised [0,1] cross-encoder scores, or None on failure.

        The cross-encoder model is loaded lazily on the first call to avoid
        importing sentence_transformers / torch at toolkit construction time,
        which hangs on Python 3.14 due to mpmath ctx_mp.py:init_builtins().
        """
        if self._cross_encoder is None and not self._ce_load_attempted:
            self._ce_load_attempted = True
            self._cross_encoder = self._load_cross_encoder()
        if self._cross_encoder is None:
            return None
        import os as _os
        try:
            pairs = [(query, c.text) for c in chunks]
            # Suppress tqdm progress bar printed to stderr during predict()
            _old_tqdm = _os.environ.get("TQDM_DISABLE")
            _os.environ["TQDM_DISABLE"] = "1"
            try:
                raw_scores: List[float] = self._cross_encoder.predict(pairs, show_progress_bar=False).tolist()  # type: ignore[union-attr]
            finally:
                if _old_tqdm is None:
                    _os.environ.pop("TQDM_DISABLE", None)
                else:
                    _os.environ["TQDM_DISABLE"] = _old_tqdm
            return _sigmoid_normalise(raw_scores)
        except Exception as exc:
            logger.warning("Cross-encoder predict failed: %s — using BM25-only", exc)
            return None


@dataclass
class CRAGEvaluation:
    status: CRAGStatus
    confidence: float


class CRAGEvaluator:
    def __init__(self, config: BusinessAnalystConfig) -> None:
        self.config = config

    def evaluate(
        self,
        chunks: Sequence[Chunk],
        ticker: Optional[str] = None,
    ) -> CRAGEvaluation:
        """Score retrieval quality.

        When *ticker* is provided, only chunks that provably belong to that
        ticker are considered.  A chunk that belongs to a *different* company
        is never counted as evidence of a CORRECT retrieval — it would
        downstream contaminate the LLM context with off-ticker information.
        """
        if not chunks:
            return CRAGEvaluation(CRAGStatus.INCORRECT, 0.0)

        if ticker:
            ticker_chunks = [c for c in chunks if _chunk_ticker_matches(c, ticker)]
            if not ticker_chunks:
                # Every retrieved chunk belongs to a different company — treat as
                # INCORRECT so the pipeline falls back to web search.
                logger.warning(
                    "[CRAGEvaluator] All %d retrieved chunk(s) are off-ticker for ticker=%s "
                    "— classifying as INCORRECT.",
                    len(chunks),
                    ticker,
                )
                return CRAGEvaluation(CRAGStatus.INCORRECT, 0.0)
            top_score = ticker_chunks[0].score
        else:
            top_score = chunks[0].score

        if top_score >= self.config.crag_correct_threshold:
            return CRAGEvaluation(CRAGStatus.CORRECT, top_score)
        if top_score >= self.config.crag_ambiguous_threshold:
            return CRAGEvaluation(CRAGStatus.AMBIGUOUS, top_score)
        return CRAGEvaluation(CRAGStatus.INCORRECT, top_score)


# ----------------------------------------------------------------------------
# Toolkit façade used by the agent + health checks
# ----------------------------------------------------------------------------
class BusinessAnalystToolkit:
    def __init__(self, config: Optional[BusinessAnalystConfig] = None) -> None:
        self.config = config or BusinessAnalystConfig()
        # Neo4j chunk-ingestion embedder: all-MiniLM-L6-v2 (384-dim), run locally via
        # sentence_transformers — this model is NOT served through Ollama.
        # A fallback to the Ollama EmbeddingClient is wired in so the toolkit still
        # initialises if sentence_transformers is unavailable.
        _neo4j_ollama_fallback = EmbeddingClient(
            self.config.ollama_base_url,
            self.config.embedding_model,
            self.config.request_timeout,
        )
        self.embedding = LocalEmbeddingClient(
            self.config.embedding_model,
            fallback=_neo4j_ollama_fallback,
        )
        # Qdrant embedder: nomic-embed-text, 768-dim (matches how load_qdrant.py indexes)
        self.qdrant_embedding = EmbeddingClient(
            self.config.ollama_base_url,
            self.config.qdrant_embedding_model,
            self.config.request_timeout,
        )
        self.neo4j = Neo4jConnector(self.config)
        self.pg = PostgresConnector(self.config)
        self.qdrant = QdrantConnector(self.config)
        self.retriever = HybridRetriever(
            self.config,
            self.embedding,
            self.neo4j,
            self.qdrant,
            qdrant_embedding_client=self.qdrant_embedding,
        )
        self.evaluator = CRAGEvaluator(self.config)

    def fetch_company_overview(self, ticker: Optional[str]) -> Optional[Dict[str, Any]]:
        return self.neo4j.fetch_company_overview(ticker)

    def fetch_community_summary(self, ticker: Optional[str]) -> Optional[str]:
        """Return graph-community summary string for *ticker* (see Neo4jConnector)."""
        try:
            return self.neo4j.fetch_community_summary(ticker)
        except Exception as exc:
            logger.warning("Community summary fetch failed for %s: %s", ticker, exc)
            return None

    def fetch_sentiment(self, ticker: Optional[str]) -> Optional[SentimentSnapshot]:
        try:
            return self.pg.fetch_sentiment(ticker)
        except Exception as exc:
            logger.warning("Postgres sentiment fetch failed: %s", exc)
            return None

    def retrieve(self, query: str, ticker: Optional[str]) -> RetrievalResult:
        try:
            return self.retriever.retrieve(query, ticker)
        except Exception as exc:
            logger.error("Hybrid retrieval failed: %s", exc)
            return RetrievalResult(chunks=[], graph_facts=[], bm25_debug={})

    def evaluate(self, chunks: Sequence[Chunk], ticker: Optional[str] = None) -> CRAGEvaluation:
        return self.evaluator.evaluate(chunks, ticker=ticker)

    def healthcheck(self) -> Dict[str, Any]:
        health: Dict[str, Any] = {"neo4j": False, "postgres": False, "qdrant": False, "ollama": False}
        try:
            with self.neo4j.driver.session(database=None) as _session:
                _session.run("RETURN 1")
            health["neo4j"] = True
        except Exception as exc:
            health["neo4j_error"] = str(exc)

        try:
            conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                dbname=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password,
            )
            conn.close()
            health["postgres"] = True
        except Exception as exc:
            health["postgres_error"] = str(exc)

        try:
            self.qdrant.client.get_collections()
            health["qdrant"] = True
        except Exception as exc:
            health["qdrant_error"] = str(exc)

        try:
            self.embedding.embed("health check ping")
            health["ollama"] = True
        except Exception as exc:
            health["ollama_error"] = str(exc)

        return health

    def close(self) -> None:
        self.neo4j.close()


__all__ = [
    "BusinessAnalystToolkit",
    "EmbeddingClient",
    "LocalEmbeddingClient",
    "HybridRetriever",
    "CRAGEvaluator",
    "CRAGEvaluation",
    "Neo4jConnector",
    "PostgresConnector",
    "QdrantConnector",
]
