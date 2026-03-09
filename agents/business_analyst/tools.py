"""Connectors and helper utilities for the Business Analyst agent."""

from __future__ import annotations

import json
import logging
import math
import os
import re as _re_top
import time
from collections import OrderedDict
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import hashlib
import threading

import psycopg2
from neo4j import GraphDatabase, basic_auth
from psycopg2.extras import RealDictCursor
import requests

try:
    from orchestration.data_availability import check_all, ticker_data_profile
    _ORCHESTRATION_AVAILABLE = True
except ModuleNotFoundError:
    # orchestration package not on sys.path (e.g. running outside the repo root).
    # The metadata profile degrades gracefully: has_pg_fundamentals / has_pg_timeseries
    # will be False, but all other pipeline functionality is unaffected.
    _ORCHESTRATION_AVAILABLE = False

    def check_all(*args, **kwargs):  # type: ignore[misc]
        return {}

    def ticker_data_profile(*args, **kwargs):  # type: ignore[misc]
        return {}

from .config import BusinessAnalystConfig
from .schema import Chunk, CRAGStatus, MetadataProfile, RetrievalResult, SentimentSnapshot

_CrossEncoder = None  # type: ignore[assignment]

def _get_cross_encoder_class():
    global _CrossEncoder
    if _CrossEncoder is None:
        try:
            from sentence_transformers import CrossEncoder as _CE  # type: ignore[import]
            _CrossEncoder = _CE
        except ImportError:
            pass
    return _CrossEncoder

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Process-level metadata cache (singleton, thread-safe)
# ----------------------------------------------------------------------------

@dataclass(slots=True)
class _MetadataCacheEntry:
    profile: MetadataProfile
    expires_at: float


class _MetadataCache:
    """Thread-safe in-process TTL cache for MetadataProfile objects.

    A single instance is shared across all toolkit instances within a
    process so repeated calls within the same pipeline run hit the cache
    rather than re-querying Neo4j / PostgreSQL.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: Dict[str, _MetadataCacheEntry] = {}

    def get(self, ticker: str) -> Optional[MetadataProfile]:
        with self._lock:
            entry = self._store.get(ticker.upper())
            if entry is None:
                return None
            if time.monotonic() > entry.expires_at:
                del self._store[ticker.upper()]
                return None
            return entry.profile

    def set(self, ticker: str, profile: MetadataProfile, ttl: float) -> None:
        with self._lock:
            self._store[ticker.upper()] = _MetadataCacheEntry(
                profile=profile,
                expires_at=time.monotonic() + ttl,
            )


# Singleton — one per process
_METADATA_CACHE = _MetadataCache()


# ----------------------------------------------------------------------------
# Process-level semantic cache (LRU + TTL)
# ----------------------------------------------------------------------------

@dataclass(slots=True)
class _SemanticCacheEntry:
    retrieval: RetrievalResult
    expires_at: float


class _SemanticCache:
    """Thread-safe LRU + TTL cache keyed on (ticker, query_hash).

    A cache hit returns a previously computed RetrievalResult so the
    pipeline can skip the embedding + vector-search round-trip for
    repeated or near-identical queries (e.g. hot tickers like AAPL).
    """

    def __init__(self, max_entries: int = 128) -> None:
        self._lock = threading.Lock()
        self._store: OrderedDict[str, _SemanticCacheEntry] = OrderedDict()
        self._max = max_entries

    @staticmethod
    def _key(ticker: Optional[str], query: str) -> str:
        raw = f"{(ticker or '').upper()}::{query.strip().lower()}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, ticker: Optional[str], query: str) -> Optional[RetrievalResult]:
        key = self._key(ticker, query)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.monotonic() > entry.expires_at:
                del self._store[key]
                return None
            # LRU: move to end
            self._store.move_to_end(key)
            return entry.retrieval

    def set(self, ticker: Optional[str], query: str, retrieval: RetrievalResult, ttl: float) -> None:
        key = self._key(ticker, query)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = _SemanticCacheEntry(
                retrieval=retrieval,
                expires_at=time.monotonic() + ttl,
            )
            # Evict LRU entry if over capacity
            while len(self._store) > self._max:
                self._store.popitem(last=False)


# Singleton — one per process (max_entries updated on first toolkit init)
_SEMANTIC_CACHE = _SemanticCache(max_entries=128)



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
        # Only query relationship types that are guaranteed to exist (EODHD-ingested).
        # FACES_RISK and COMPETES_WITH are FMP-sourced and absent when FMP DAG is paused;
        # including them triggers Neo4j GqlStatusObject WARNING notifications.
        cypher = """
        MATCH (c:Company {ticker:$ticker})-[r:HAS_STRATEGY|HAS_FACT]->(n)
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

    def fetch_insider_signals(self, ticker: str, limit: int = 20) -> List[Dict]:
        """Fetch insider trading signals from Neo4j."""
        cypher = """
        MATCH (c:Company {ticker: $ticker})-[:HAS_INSIDER_TRADE]->(i:Insider)
        RETURN properties(i) AS insider
        ORDER BY i.transactionDate DESC
        LIMIT $limit
        """
        with self.driver.session(database=None) as session:
            result = session.run(cypher, ticker=ticker, limit=limit)
            return [dict(row["insider"]) for row in result]

    def fetch_etf_constituents(self, ticker: str, limit: int = 50) -> List[Dict]:
        """Fetch ETF/index constituent holdings for the given ETF ticker (Row 15).

        Queries Neo4j for ETF_HOLDS_CONSTITUENT relationships from the ETF node.
        Returns a list of constituent dicts with keys: constituent_ticker, weight,
        shares, name (where available).
        """
        cypher = """
        MATCH (etf:ETF {ticker: $ticker})-[r:ETF_HOLDS_CONSTITUENT]->(c)
        RETURN properties(c) AS constituent, properties(r) AS holding
        ORDER BY r.weight DESC
        LIMIT $limit
        """
        try:
            with self.driver.session(database=None) as session:
                result = session.run(cypher, ticker=ticker, limit=limit)
                rows = []
                for row in result:
                    entry = {}
                    holding = row.get("holding") or {}
                    constituent = row.get("constituent") or {}
                    entry.update({k: v for k, v in constituent.items() if k not in ("embedding",)})
                    entry.update({k: v for k, v in holding.items()})
                    rows.append(entry)
                return rows
        except Exception as exc:
            logger.warning("fetch_etf_constituents failed for %s: %s", ticker, exc)
            return []

    def fetch_chunk_count(self, ticker: Optional[str]) -> int:
        """Return the number of Chunk nodes for *ticker* in Neo4j (fast COUNT query)."""
        if not ticker:
            return 0
        cypher = """
        MATCH (c:Company {ticker: $ticker})-[:HAS_CHUNK]->(ch:Chunk)
        RETURN count(ch) AS n
        """
        try:
            with self.driver.session(database=None) as session:
                record = session.run(cypher, ticker=ticker).single()
                return int(record["n"]) if record else 0
        except Exception as exc:
            logger.warning("[Neo4j] chunk count query failed for %s: %s", ticker, exc)
            return 0

    def is_vector_index_online(self) -> bool:
        """Return True when the chunk_embedding vector index is ONLINE."""
        cypher = "SHOW INDEXES YIELD name, state WHERE name = $name RETURN state"
        try:
            with self.driver.session(database=None) as session:
                record = session.run(cypher, name=self.config.neo4j_chunk_index).single()
                if record:
                    return str(record["state"]).upper() == "ONLINE"
        except Exception as exc:
            logger.warning("[Neo4j] vector index status query failed: %s", exc)
        return False

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
        # NOTE: live DB uses 'as_of_date' column; 'trend' added via migration when present
        sql = """
        SELECT bullish_pct, bearish_pct, neutral_pct,
               COALESCE(trend, 'unknown') AS trend
        FROM sentiment_trends
        WHERE ticker = %s
        ORDER BY as_of_date DESC
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

    def fetch_esg(self, ticker: str) -> Optional[Dict]:
        """Fetch latest ESG scores for the ticker."""
        sql = """
            SELECT ticker, env_score, social_score, gov_score, esg_total, as_of_date
            FROM esg_scores
            WHERE ticker = %s
            ORDER BY as_of_date DESC
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
            return dict(row)

    def fetch_social_sentiment(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Fetch recent social sentiment data for the ticker."""
        sql = """
            SELECT ticker, platform, score, sentiment_label, date
            FROM social_sentiment
            WHERE ticker = %s
            ORDER BY date DESC
            LIMIT %s
        """
        conn = psycopg2.connect(
            host=self.config.postgres_host,
            port=self.config.postgres_port,
            dbname=self.config.postgres_db,
            user=self.config.postgres_user,
            password=self.config.postgres_password,
        )
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def _pg_connect(self):
        return psycopg2.connect(
            host=self.config.postgres_host,
            port=self.config.postgres_port,
            dbname=self.config.postgres_db,
            user=self.config.postgres_user,
            password=self.config.postgres_password,
        )

    def fetch_company_profile(self, ticker: str) -> Optional[Dict]:
        """Fetch latest company profile for the ticker (Row 3: Company Profiles).

        Queries raw_fundamentals WHERE data_name = 'company_profile'.
        """
        sql = """
        SELECT payload, as_of_date
        FROM raw_fundamentals
        WHERE ticker_symbol = %s
          AND data_name = 'company_profile'
        ORDER BY as_of_date DESC
        LIMIT 1
        """
        conn = self._pg_connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker,))
            row = cur.fetchone()
        if not row:
            return None
        payload = row["payload"]
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                pass
        return payload if isinstance(payload, dict) else {}

    def fetch_news(self, ticker: str, limit: int = 20) -> List[Dict]:
        """Fetch recent news items for the ticker (Rows 4/17: Financial News).

        Queries raw_fundamentals WHERE data_name = 'news'.
        """
        sql = """
        SELECT payload, as_of_date
        FROM raw_fundamentals
        WHERE ticker_symbol = %s
          AND data_name = 'news'
        ORDER BY as_of_date DESC
        LIMIT %s
        """
        conn = self._pg_connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
            rows = cur.fetchall()
        results = []
        for row in rows:
            payload = row["payload"]
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    pass
            # News payload is typically a list of articles; flatten it
            if isinstance(payload, list):
                results.extend(payload)
            elif isinstance(payload, dict):
                results.append(payload)
        return results[:limit]

    def fetch_insider_transactions(self, ticker: str, limit: int = 20) -> List[Dict]:
        """Fetch insider transactions for the ticker (Row 5: Insider Transactions)."""
        sql = """
        SELECT ticker, insider_name, transaction_type, shares, price, transaction_date
        FROM insider_transactions
        WHERE ticker = %s
        ORDER BY transaction_date DESC
        LIMIT %s
        """
        conn = self._pg_connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def fetch_institutional_holders(self, ticker: str, limit: int = 20) -> List[Dict]:
        """Fetch major institutional/mutual fund holders (Row 6)."""
        sql = """
        SELECT ticker, holder_name, shares, shares_change, ownership_pct, as_of_date
        FROM institutional_holders
        WHERE ticker = %s
        ORDER BY as_of_date DESC, shares DESC
        LIMIT %s
        """
        conn = self._pg_connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def fetch_financial_calendar(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Fetch financial calendar events for the ticker (Row 16: Financial Calendar)."""
        sql = """
        SELECT ticker, event_type, event_date, eps_estimate, revenue_estimate
        FROM financial_calendar
        WHERE ticker = %s
        ORDER BY event_date DESC
        LIMIT %s
        """
        conn = self._pg_connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def fetch_textual_documents(self, ticker: str, limit: int = 20) -> List[Dict]:
        """Fetch textual document metadata for the ticker (Row 24: Textual Documents Metadata)."""
        sql = """
        SELECT ticker, doc_type, filename, filepath, institution, date_approx,
               file_size_bytes, md5_hash
        FROM textual_documents
        WHERE ticker = %s
        ORDER BY date_approx DESC
        LIMIT %s
        """
        conn = self._pg_connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
            rows = cur.fetchall()
        return [dict(r) for r in rows]


# ----------------------------------------------------------------------------
# Keyword scorer (BM25-lite)
# ----------------------------------------------------------------------------
class PgVectorConnector:
    """Semantic search over the text_chunks table using pgvector cosine similarity.

    Requires the pgvector extension and the text_chunks table (created by
    load_postgres.py / docker/init-db.sql).  Gracefully returns [] if the
    table or extension is not available.
    """

    def __init__(self, config: BusinessAnalystConfig) -> None:
        self.config = config

    def _pg_connect(self):
        return psycopg2.connect(
            host=self.config.postgres_host,
            port=self.config.postgres_port,
            dbname=self.config.postgres_db,
            user=self.config.postgres_user,
            password=self.config.postgres_password,
        )

    def vector_search(
        self,
        vector: List[float],
        ticker: Optional[str],
        top_k: int = 10,
    ) -> List[Chunk]:
        """Return the *top_k* most similar chunks from text_chunks for *ticker*."""
        # Build the query embedding as a Postgres vector literal
        emb_str = "[" + ",".join(f"{x:.8f}" for x in vector) + "]"
        ticker_filter = "AND ticker = %s" if ticker else ""
        # params order must match placeholder order in SQL:
        # 1) emb_str  → score calc  (SELECT)
        # 2) ticker   → ticker filter (WHERE, only if present)
        # 3) emb_str  → ORDER BY
        # 4) top_k    → LIMIT
        params: list = [emb_str]
        if ticker:
            params.append(ticker)
        params.extend([emb_str, top_k])

        sql = f"""
        SELECT chunk_id, text, section, filing_date, ticker,
               1 - (embedding <=> %s::vector) AS score
        FROM text_chunks
        WHERE embedding IS NOT NULL
          {ticker_filter}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        try:
            conn = self._pg_connect()
            with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        except Exception as exc:
            logger.warning("[PgVector] vector_search failed: %s", exc)
            return []

        chunks: List[Chunk] = []
        for row in rows:
            metadata = {
                "section": row.get("section"),
                "filing_date": row.get("filing_date"),
                "ticker": row.get("ticker"),
            }
            chunk_id = row.get("chunk_id") or f"pgvec::{row.get('ticker')}::{len(chunks)}"
            score = float(row.get("score") or 0.0)
            score = max(0.0, min(1.0, score))
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=row.get("text", ""),
                    score=score,
                    source="pgvector",
                    metadata=metadata,
                )
            )
        return chunks

    def chunk_count(self, ticker: Optional[str]) -> int:
        """Return the number of embedded rows in text_chunks for *ticker*."""
        if not ticker:
            return 0
        sql = "SELECT COUNT(*) AS n FROM text_chunks WHERE ticker = %s AND embedding IS NOT NULL"
        try:
            conn = self._pg_connect()
            with closing(conn), conn.cursor() as cur:
                cur.execute(sql, (ticker,))
                row = cur.fetchone()
                return int(row[0]) if row else 0
        except Exception as exc:
            logger.warning("[PgVector] chunk_count failed for %s: %s", ticker, exc)
            return 0

    def has_embedding_index(self) -> bool:
        """Return True when an ivfflat/hnsw index exists on text_chunks.embedding."""
        sql = """
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'text_chunks'
          AND indexdef ILIKE '%embedding%'
        LIMIT 1
        """
        try:
            conn = self._pg_connect()
            with closing(conn), conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchone() is not None
        except Exception as exc:
            logger.warning("[PgVector] has_embedding_index check failed: %s", exc)
            return False


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


def _rrf_fuse(
    ranked_lists: List[List[Chunk]],
    k: int = 60,
) -> List[Chunk]:
    """Reciprocal Rank Fusion over multiple ranked chunk lists.

    For each list, the RRF score for a chunk at rank ``r`` (1-indexed) is
    ``1 / (k + r)``.  Scores are summed across lists so a chunk appearing
    in multiple lists gets a composite boost.

    Args:
        ranked_lists: Each element is an ordered list of Chunk objects
                      (best first).  Duplicate chunk_ids across lists are
                      merged — the chunk text/metadata from the first
                      occurrence is kept.
        k:            RRF constant (default 60, from the original paper).

    Returns:
        A single deduplicated list sorted by descending RRF score, with
        ``chunk.score`` updated to the normalised RRF composite score.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunk_map:
                chunk_map[cid] = chunk

    # Normalise RRF scores to [0, 1] by dividing by the theoretical maximum
    # (a chunk appearing at rank 1 in every list: n_lists * 1/(k+1))
    n = max(len(ranked_lists), 1)
    max_score = n / (k + 1)
    result: List[Chunk] = []
    for cid, rrf in sorted(rrf_scores.items(), key=lambda x: -x[1]):
        base_chunk = chunk_map[cid]
        normed = min(rrf / max_score, 1.0)
        result.append(
            Chunk(
                chunk_id=base_chunk.chunk_id,
                text=base_chunk.text,
                score=normed,
                source=base_chunk.source,
                metadata=base_chunk.metadata,
            )
        )
    return result


# ----------------------------------------------------------------------------
# Ticker identity helper
# ----------------------------------------------------------------------------

def _chunk_ticker_matches(chunk: "Chunk", ticker: str) -> bool:
    """Return True if the chunk provably belongs to *ticker*.

    Checks (in priority order):
    1. ``chunk.metadata["ticker"]``       — set by Neo4jConnector
    2. ``chunk.metadata["ticker_symbol"]`` — set by source connector
    3. ``chunk.chunk_id`` prefix heuristic — e.g. ``neo4j::AAPL::…``

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
        self.embedding = EmbeddingClient(
            self.config.ollama_base_url,
            self.config.embedding_model,
            self.config.request_timeout,
        )
        self.neo4j = Neo4jConnector(self.config)
        self.pg = PostgresConnector(self.config)
        self.pgvec = PgVectorConnector(self.config)
        self.evaluator = CRAGEvaluator(self.config)
        # Update singleton semantic cache size from config (idempotent on repeated init)
        global _SEMANTIC_CACHE
        _SEMANTIC_CACHE._max = self.config.semantic_cache_max_entries

    # ------------------------------------------------------------------
    # Metadata pre-check (cached)
    # ------------------------------------------------------------------

    def get_metadata_profile(self, ticker: str) -> MetadataProfile:
        """Return a cached MetadataProfile for *ticker*.

        The profile captures:
        - Neo4j Chunk node count + vector index readiness
        - pgvector row count + embedding index presence
        - Sentiment data availability

        Results are cached in-process for ``config.metadata_cache_ttl`` seconds
        (default 60 s) to avoid repeated DB round-trips within a single run.
        """
        cached = _METADATA_CACHE.get(ticker)
        if cached is not None:
            logger.debug("[MetadataCache] HIT for ticker=%s", ticker)
            return cached

        logger.debug("[MetadataCache] MISS for ticker=%s — querying backends", ticker)

        # Neo4j
        neo4j_chunk_count = 0
        neo4j_index_ready = False
        try:
            neo4j_chunk_count = self.neo4j.fetch_chunk_count(ticker)
            neo4j_index_ready = self.neo4j.is_vector_index_online()
        except Exception as exc:
            logger.warning("[MetadataProfile] Neo4j query failed for %s: %s", ticker, exc)

        # pgvector
        pgvec_count = 0
        pgvec_table_ready = False
        pgvec_index = False
        try:
            pgvec_count = self.pgvec.chunk_count(ticker)
            pgvec_table_ready = True
            pgvec_index = self.pgvec.has_embedding_index()
        except Exception as exc:
            logger.warning("[MetadataProfile] pgvector query failed for %s: %s", ticker, exc)

        # Sentiment
        has_sentiment = False
        sentiment_last_updated: Optional[str] = None
        try:
            snap = self.pg.fetch_sentiment(ticker)
            has_sentiment = snap is not None
        except Exception as exc:
            logger.warning("[MetadataProfile] sentiment check failed for %s: %s", ticker, exc)

        # Fundamentals / timeseries flags (reuse check_all from data_availability)
        has_pg_fundamentals = False
        has_pg_timeseries = False
        try:
            avail_all = check_all(tickers=[ticker])
            av = ticker_data_profile(avail_all, ticker)
            has_pg_fundamentals = bool(av.get("has_fundamentals", False))
            has_pg_timeseries = bool(av.get("has_timeseries", False))
        except Exception as exc:
            logger.debug("[MetadataProfile] data_availability profile failed for %s: %s", ticker, exc)

        profile = MetadataProfile(
            ticker=ticker,
            neo4j_chunk_count=neo4j_chunk_count,
            neo4j_chunk_index_ready=neo4j_index_ready,
            pgvector_chunk_count=pgvec_count,
            pgvector_table_ready=pgvec_table_ready,
            pgvector_embedding_index=pgvec_index,
            has_sentiment=has_sentiment,
            sentiment_last_updated=sentiment_last_updated,
            last_checked=time.monotonic(),
            has_neo4j_chunks=neo4j_chunk_count > 0,
            has_pg_fundamentals=has_pg_fundamentals,
            has_pg_timeseries=has_pg_timeseries,
        )
        _METADATA_CACHE.set(ticker, profile, ttl=self.config.metadata_cache_ttl)
        logger.info(
            "[MetadataProfile] ticker=%s neo4j_chunks=%d pgvec_chunks=%d "
            "sentiment=%s neo4j_index=%s",
            ticker, neo4j_chunk_count, pgvec_count, has_sentiment, neo4j_index_ready,
        )
        return profile

    # ------------------------------------------------------------------
    # Fast-path retrieval (vector + BM25, no full graph traversal)
    # ------------------------------------------------------------------

    def retrieve_fast(self, query: str, ticker: Optional[str]) -> RetrievalResult:
        """Stage-1 fast hybrid retrieval for SIMPLE and NUMERICAL query paths.

        Uses a smaller top_k budget (``config.fast_path_top_k``) and skips
        graph traversal / cross-encoder reranking to minimise latency.
        Checks the semantic cache first; populates it on miss.
        """
        # Semantic cache check
        cached = _SEMANTIC_CACHE.get(ticker, query)
        if cached is not None:
            logger.info("[SemanticCache] HIT for ticker=%s (fast path)", ticker)
            return cached

        try:
            vector = self.embedding.embed(query)
            chunks = self.neo4j.vector_search(vector, ticker, top_k=self.config.fast_path_top_k)

            # BM25 re-scoring blend
            for i, chunk in enumerate(chunks):
                bm25 = keyword_overlap_score(chunk.text, query)
                chunks[i] = Chunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=0.7 * chunk.score + 0.3 * bm25,
                    source=chunk.source,
                    metadata=chunk.metadata,
                )
            chunks.sort(key=lambda c: -c.score)
            result = RetrievalResult(chunks=chunks, graph_facts=[], bm25_debug={})
        except Exception as exc:
            logger.error("Fast-path retrieval failed: %s", exc)
            result = RetrievalResult(chunks=[], graph_facts=[], bm25_debug={})

        _SEMANTIC_CACHE.set(ticker, query, result, ttl=self.config.semantic_cache_ttl)
        return result

    # ------------------------------------------------------------------
    # Multi-stage retrieval (Stage 1 recall → Stage 2 rerank + graph → RRF)
    # ------------------------------------------------------------------

    def retrieve_multi_stage(self, query: str, ticker: Optional[str]) -> RetrievalResult:
        """Two-stage retrieval for COMPLEX query paths.

        Stage 1 — Fast bi-encoder recall:
            - Neo4j vector search top-``multi_stage_recall_k`` (dense)
            - pgvector vector search top-``multi_stage_recall_k`` (dense)
            - BM25 keyword scoring over Neo4j results

        Stage 2 — Cross-encoder precision rerank + graph traversal:
            - Cross-encoder (ms-marco-MiniLM-L-6-v2) re-scores the Stage-1
              recall pool; degrades gracefully if sentence-transformers absent.
            - Graph-traversal facts (HAS_STRATEGY / HAS_FACT) appended as
              supplementary context.

        Fusion — Reciprocal Rank Fusion (RRF) merges all scored lists into a
        single ordered result, returning the top ``config.top_k`` chunks.

        Results are cached in the semantic cache (TTL = ``config.semantic_cache_ttl``).
        """
        # Semantic cache check
        cached = _SEMANTIC_CACHE.get(ticker, query)
        if cached is not None:
            logger.info("[SemanticCache] HIT for ticker=%s (multi-stage path)", ticker)
            return cached

        recall_k = self.config.multi_stage_recall_k
        final_k = self.config.top_k

        try:
            vector = self.embedding.embed(query)
        except Exception as exc:
            logger.error("Embedding failed in multi-stage retrieval: %s", exc)
            return RetrievalResult(chunks=[], graph_facts=[], bm25_debug={})

        # --- Stage 1: Bi-encoder recall ---
        neo4j_chunks: List[Chunk] = []
        pgvec_chunks: List[Chunk] = []
        try:
            neo4j_chunks = self.neo4j.vector_search(vector, ticker, top_k=recall_k)
        except Exception as exc:
            logger.warning("[MultiStage] Neo4j recall failed: %s", exc)
        try:
            pgvec_chunks = self.pgvec.vector_search(vector, ticker, top_k=recall_k)
        except Exception as exc:
            logger.warning("[MultiStage] pgvector recall failed: %s", exc)

        # BM25 re-scoring on Neo4j results
        bm25_ranked: List[Chunk] = []
        for chunk in neo4j_chunks:
            bm25 = keyword_overlap_score(chunk.text, query)
            bm25_ranked.append(
                Chunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=bm25,
                    source=chunk.source,
                    metadata=chunk.metadata,
                )
            )
        bm25_ranked.sort(key=lambda c: -c.score)

        # --- Stage 2: Cross-encoder rerank ---
        # Merge recall pool (deduplicated by chunk_id)
        recall_pool: List[Chunk] = []
        seen_ids: set = set()
        for chunk in neo4j_chunks + pgvec_chunks:
            if chunk.chunk_id not in seen_ids:
                recall_pool.append(chunk)
                seen_ids.add(chunk.chunk_id)

        ce_ranked: List[Chunk] = []
        CE = _get_cross_encoder_class()
        if CE is not None and recall_pool:
            try:
                ce_model = CE(self.config.reranker_model)
                pairs = [(query, c.text[:512]) for c in recall_pool]
                raw_scores: List[float] = ce_model.predict(pairs).tolist()  # type: ignore[union-attr]
                norm_scores = _sigmoid_normalise(raw_scores)
                ce_ranked = [
                    Chunk(
                        chunk_id=recall_pool[i].chunk_id,
                        text=recall_pool[i].text,
                        score=norm_scores[i],
                        source=recall_pool[i].source,
                        metadata=recall_pool[i].metadata,
                    )
                    for i in range(len(recall_pool))
                ]
                ce_ranked.sort(key=lambda c: -c.score)
                logger.debug(
                    "[MultiStage] Cross-encoder reranked %d chunks; top score=%.3f",
                    len(ce_ranked), ce_ranked[0].score if ce_ranked else 0.0,
                )
            except Exception as exc:
                logger.warning("[MultiStage] Cross-encoder rerank failed: %s", exc)
                ce_ranked = sorted(recall_pool, key=lambda c: -c.score)
        else:
            # Degrade gracefully: sort recall pool by bi-encoder score
            ce_ranked = sorted(recall_pool, key=lambda c: -c.score)

        # --- RRF Fusion ---
        ranked_lists = [neo4j_chunks, pgvec_chunks, bm25_ranked, ce_ranked]
        fused = _rrf_fuse([lst for lst in ranked_lists if lst], k=60)
        fused = fused[:final_k]

        # --- Graph traversal facts (supplementary context) ---
        graph_facts: List[Dict[str, Any]] = []
        try:
            graph_facts = self.neo4j.fetch_graph_facts(ticker, limit=25)
        except Exception as exc:
            logger.warning("[MultiStage] graph facts fetch failed: %s", exc)

        result = RetrievalResult(
            chunks=fused,
            graph_facts=graph_facts,
            bm25_debug={"bm25_pool_size": len(bm25_ranked), "recall_pool_size": len(recall_pool)},
        )
        _SEMANTIC_CACHE.set(ticker, query, result, ttl=self.config.semantic_cache_ttl)
        return result

    # ------------------------------------------------------------------
    # Existing retrieval (kept as default / backward compat)
    # ------------------------------------------------------------------

    def retrieve(self, query: str, ticker: Optional[str]) -> RetrievalResult:
        """Default retrieval (wraps retrieve_multi_stage for the complex path)."""
        return self.retrieve_multi_stage(query, ticker)

    def fetch_community_summary(self, ticker: Optional[str]) -> Optional[str]:
        """Return graph-community summary string for *ticker* (see Neo4jConnector)."""
        try:
            return self.neo4j.fetch_community_summary(ticker)
        except Exception as exc:
            logger.warning("Community summary fetch failed for %s: %s", ticker, exc)
            return None

    def fetch_company_overview(self, ticker: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return the Neo4j Company node properties for *ticker*."""
        try:
            return self.neo4j.fetch_company_overview(ticker)
        except Exception as exc:
            logger.warning("Company overview fetch failed for %s: %s", ticker, exc)
            return None

    def fetch_sentiment(self, ticker: Optional[str]) -> Optional[SentimentSnapshot]:
        try:
            return self.pg.fetch_sentiment(ticker)
        except Exception as exc:
            logger.warning("Postgres sentiment fetch failed: %s", exc)
            return None

    def fetch_esg(self, ticker: str) -> Optional[Dict]:
        """Fetch latest ESG scores for the ticker."""
        try:
            return self.pg.fetch_esg(ticker)
        except Exception as exc:
            logger.warning("ESG fetch failed for %s: %s", ticker, exc)
            return None

    def fetch_social_sentiment(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Fetch recent social sentiment data for the ticker."""
        try:
            return self.pg.fetch_social_sentiment(ticker, limit)
        except Exception as exc:
            logger.warning("Social sentiment fetch failed for %s: %s", ticker, exc)
            return []

    def fetch_insider_signals(self, ticker: str, limit: int = 20) -> List[Dict]:
        """Fetch insider trading signals from Neo4j."""
        try:
            return self.neo4j.fetch_insider_signals(ticker, limit)
        except Exception as exc:
            logger.warning("Insider signals fetch failed for %s: %s", ticker, exc)
            return []

    def fetch_company_profile(self, ticker: str) -> Optional[Dict]:
        """Fetch latest company profile for the ticker (Row 3)."""
        try:
            return self.pg.fetch_company_profile(ticker)
        except Exception as exc:
            logger.warning("Company profile fetch failed for %s: %s", ticker, exc)
            return None

    def fetch_news(self, ticker: str, limit: int = 20) -> List[Dict]:
        """Fetch recent news items for the ticker (Rows 4/17)."""
        try:
            return self.pg.fetch_news(ticker, limit)
        except Exception as exc:
            logger.warning("News fetch failed for %s: %s", ticker, exc)
            return []

    def fetch_insider_transactions(self, ticker: str, limit: int = 20) -> List[Dict]:
        """Fetch insider transactions for the ticker from PostgreSQL (Row 5)."""
        try:
            return self.pg.fetch_insider_transactions(ticker, limit)
        except Exception as exc:
            logger.warning("Insider transactions fetch failed for %s: %s", ticker, exc)
            return []

    def fetch_institutional_holders(self, ticker: str, limit: int = 20) -> List[Dict]:
        """Fetch major institutional/mutual fund holders (Row 6)."""
        try:
            return self.pg.fetch_institutional_holders(ticker, limit)
        except Exception as exc:
            logger.warning("Institutional holders fetch failed for %s: %s", ticker, exc)
            return []

    def fetch_financial_calendar(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Fetch financial calendar events for the ticker (Row 16)."""
        try:
            return self.pg.fetch_financial_calendar(ticker, limit)
        except Exception as exc:
            logger.warning("Financial calendar fetch failed for %s: %s", ticker, exc)
            return []

    def fetch_etf_constituents(self, ticker: str, limit: int = 50) -> List[Dict]:
        """Fetch ETF/index constituent holdings from Neo4j (Row 15)."""
        try:
            return self.neo4j.fetch_etf_constituents(ticker, limit)
        except Exception as exc:
            logger.warning("ETF constituents fetch failed for %s: %s", ticker, exc)
            return []

    def fetch_textual_documents(self, ticker: str, limit: int = 20) -> List[Dict]:
        """Fetch textual document metadata for the ticker (Row 24)."""
        try:
            return self.pg.fetch_textual_documents(ticker, limit)
        except Exception as exc:
            logger.warning("Textual documents fetch failed for %s: %s", ticker, exc)
            return []

    def evaluate(self, chunks: Sequence[Chunk], ticker: Optional[str] = None) -> CRAGEvaluation:
        return self.evaluator.evaluate(chunks, ticker=ticker)

    def healthcheck(self) -> Dict[str, Any]:
        health: Dict[str, Any] = {"neo4j": False, "postgres": False, "ollama": False}
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
    "CRAGEvaluator",
    "CRAGEvaluation",
    "Neo4jConnector",
    "PostgresConnector",
]
