"""Connectors and helper utilities for the Business Analyst agent.

Changes vs original:
  - fetch_sentiment_with_fallback(): if PostgreSQL sentiment_trends is empty,
    falls back to (1) VADER local scoring over recent Neo4j chunks, then
    (2) TextBlob if VADER unavailable, then (3) raw chunk-count heuristic.
    Result is returned as a SentimentSnapshot with source annotation.
  - get_sentiment_snapshot() added as a clean single-call entry point that
    orchestrates pg → local fallback → logs clearly.
  - Rule-based query pre-classifier added (_rule_based_classify) to cheaply
    detect COMPLEX queries before calling the LLM classifier.
"""

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


def _serialize_value(v: Any) -> Any:
    """Serialize a value for JSON, handling datetime and other non-serializable types."""
    if v is None:
        return None
    if hasattr(v, 'isoformat'):
        return v.isoformat()
    if hasattr(v, '__float__') and not isinstance(v, (str, bool)):
        return float(v)
    return v


# ──────────────────────────────────────────────────────────────────────────────
# Chunk content-quality filter
# ──────────────────────────────────────────────────────────────────────────────

# Legal/disclaimer phrases that flag boilerplate broker-report chunks.
_BOILERPLATE_PHRASES = (
    "standardized options",
    "characteristics and risks of standardized options",
    "important disclosures",
    "this report has been prepared by",
    "for important disclosures",
    "analyst certification",
    "reg ac certification",
    "ubs securities llc",
    "ubs ag",
    "barclays capital",
    "morgan stanley & co",
    "as of the date of this report",
    "past performance is not",
    "conflicts of interest",
    "the information contained herein",
    "this material is not a product of",
    "please refer to important disclosures",
    "additional information is available upon request",
    "redistribution or reproduction is prohibited",
    "all rights reserved",
)


def _is_boilerplate(text: str) -> bool:
    """
    Return True if *text* is likely garbled OCR output or a legal-disclaimer
    boilerplate chunk that should be excluded from retrieval.

    Two checks:
      1. Garbled-text: more than 40 % of whitespace-separated tokens are a
         single character (space-separated chars from bad PDF font encoding).
      2. Boilerplate: the chunk begins with or contains a known legal
         disclaimer phrase.
    """
    if not text:
        return True

    # ── Garbled-text check ────────────────────────────────────────────────────
    tokens = text.split()
    if tokens:
        single_char_ratio = sum(1 for t in tokens if len(t) == 1) / len(tokens)
        if single_char_ratio > 0.40:
            return True

    # ── Boilerplate-phrase check ──────────────────────────────────────────────
    lower = text.lower()
    for phrase in _BOILERPLATE_PHRASES:
        if phrase in lower:
            return True

    return False


# ──────────────────────────────────────────────────────────────────────────────
# Rule-based pre-classifier (fast, no LLM call needed)
# ──────────────────────────────────────────────────────────────────────────────

# Keywords that reliably indicate a COMPLEX (analytical) query.
_COMPLEX_KEYWORDS = frozenset([
    "moat", "competitive advantage", "risk", "threat", "guidance",
    "sentiment trend", "outlook", "strategy", "valuation", "forecast",
    "why", "explain", "compare", "versus", "vs", "analysis",
    "long-term", "short-term", "catalyst", "headwind", "tailwind",
    "margin", "growth rate", "dcf", "intrinsic value",
])

def rule_based_classify(query: str) -> Optional[str]:
    """
    Fast rule-based pre-classifier that returns 'complex' when the query
    contains analytical keywords, or None to fall through to the LLM classifier.

    Returning None means "no strong signal — let the LLM decide".
    This runs before any LLM call so it adds zero latency on obvious cases.
    """
    q_lower = query.lower()
    for kw in _COMPLEX_KEYWORDS:
        if kw in q_lower:
            logger.debug("[rule_based_classify] keyword=%r → complex", kw)
            return "complex"
    return None  # Fall through to LLM classifier


# ──────────────────────────────────────────────────────────────────────────────
# Process-level metadata cache (singleton, thread-safe)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class _MetadataCacheEntry:
    profile: MetadataProfile
    expires_at: float


class _MetadataCache:
    """Thread-safe in-process TTL cache for MetadataProfile objects."""

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


_METADATA_CACHE = _MetadataCache()


# ──────────────────────────────────────────────────────────────────────────────
# Process-level semantic cache (LRU + TTL)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class _SemanticCacheEntry:
    retrieval: RetrievalResult
    expires_at: float


class _SemanticCache:
    """Thread-safe LRU + TTL cache keyed on (ticker, query_hash)."""

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
            while len(self._store) > self._max:
                self._store.popitem(last=False)


_SEMANTIC_CACHE = _SemanticCache(max_entries=128)


# ──────────────────────────────────────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────────────────────────────────────
class EmbeddingClient:
    """Thin wrapper around Ollama's embedding endpoint."""

    def __init__(self, base_url: str, model: str, timeout: Optional[int]) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, text: str) -> List[float]:
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
            primary_exc = RuntimeError(
                f"/api/embed returned empty embeddings for model '{self.model}'. "
                "Ensure the model is pulled: `ollama pull {self.model}`"
            )
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise RuntimeError(
                    f"Ollama embedding endpoint /api/embed returned 404 for model '{self.model}'. "
                    "Ensure Ollama >= 0.1.26 is running and the model is pulled: "
                    f"`ollama pull {self.model}`"
                ) from exc
            primary_exc = exc
        except Exception as exc:
            primary_exc = exc

        if primary_exc is not None:
            logger.debug(
                "EmbeddingClient /api/embed failed (%s), trying legacy /api/embeddings", primary_exc
            )

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
                if primary_exc is not None:
                    raise RuntimeError(
                        f"Both /api/embed and /api/embeddings returned 404 for model '{self.model}'. "
                        "Ensure Ollama >= 0.1.26 is running and the model is pulled: "
                        f"`ollama pull {self.model}`"
                    ) from primary_exc
            raise


# ──────────────────────────────────────────────────────────────────────────────
# Local sentiment fallback helpers
# ──────────────────────────────────────────────────────────────────────────────

def _vader_sentiment(texts: List[str]) -> float:
    """
    Compute aggregate sentiment score using VADER (pip install vaderSentiment).
    Returns mean compound score in [-1, +1]. Raises ImportError if not installed.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore[import]
    sia = SentimentIntensityAnalyzer()
    scores = [sia.polarity_scores(t)["compound"] for t in texts if t.strip()]
    return sum(scores) / len(scores) if scores else 0.0


def _textblob_sentiment(texts: List[str]) -> float:
    """
    Compute aggregate sentiment polarity using TextBlob (pip install textblob).
    Returns mean polarity in [-1, +1]. Raises ImportError if not installed.
    """
    from textblob import TextBlob  # type: ignore[import]
    scores = [TextBlob(t).sentiment.polarity for t in texts if t.strip()]
    return sum(scores) / len(scores) if scores else 0.0


def _local_sentiment_from_chunks(chunks: List["Chunk"]) -> Optional["SentimentSnapshot"]:
    """
    Derive a SentimentSnapshot from a list of text chunks using local NLP.

    Priority order:
      1. VADER (vaderSentiment) — fastest, finance-aware lexicon
      2. TextBlob              — fallback if VADER not installed
      3. Word-count heuristic  — last resort, no external deps

    Returns None only if the chunk list is empty.
    """
    if not chunks:
        return None

    texts = [c.text for c in chunks[:30]]  # Cap at 30 chunks for speed

    score: Optional[float] = None
    source_label = "local"

    # Attempt 1: VADER
    try:
        score = _vader_sentiment(texts)
        source_label = "vader"
        logger.info("[SentimentFallback] Used VADER on %d chunks → score=%.3f", len(texts), score)
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("[SentimentFallback] VADER failed: %s", exc)

    # Attempt 2: TextBlob
    if score is None:
        try:
            score = _textblob_sentiment(texts)
            source_label = "textblob"
            logger.info("[SentimentFallback] Used TextBlob on %d chunks → score=%.3f", len(texts), score)
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("[SentimentFallback] TextBlob failed: %s", exc)

    # Attempt 3: Simple positive/negative keyword ratio
    if score is None:
        _POS = frozenset(["growth", "strong", "beat", "record", "bullish", "profit", "gain", "surge", "positive"])
        _NEG = frozenset(["loss", "miss", "decline", "weak", "risk", "bearish", "concern", "drop", "negative"])
        pos = neg = 0
        for text in texts:
            tokens = set(text.lower().split())
            pos += len(tokens & _POS)
            neg += len(tokens & _NEG)
        total = pos + neg
        score = (pos - neg) / total if total > 0 else 0.0
        source_label = "keyword_heuristic"
        logger.info(
            "[SentimentFallback] Used keyword heuristic (pos=%d neg=%d) → score=%.3f",
            pos, neg, score,
        )

    # Convert scalar score to SentimentSnapshot percentages
    # score ∈ [-1, +1] → map to bullish/bearish/neutral buckets
    clamped = max(-1.0, min(1.0, score))
    if clamped > 0.1:
        bullish_pct = 50.0 + clamped * 50.0
        bearish_pct = max(0.0, 50.0 - clamped * 50.0)
        neutral_pct = max(0.0, 100.0 - bullish_pct - bearish_pct)
        trend = "bullish"
    elif clamped < -0.1:
        bearish_pct = 50.0 + abs(clamped) * 50.0
        bullish_pct = max(0.0, 50.0 - abs(clamped) * 50.0)
        neutral_pct = max(0.0, 100.0 - bullish_pct - bearish_pct)
        trend = "bearish"
    else:
        bullish_pct = bearish_pct = 25.0
        neutral_pct = 50.0
        trend = "neutral"

    return SentimentSnapshot(
        bullish_pct=round(bullish_pct, 2),
        bearish_pct=round(bearish_pct, 2),
        neutral_pct=round(neutral_pct, 2),
        trend=f"{trend} (local/{source_label})",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Neo4j
# ──────────────────────────────────────────────────────────────────────────────
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
        """Return structured facts from Neo4j for *ticker*.

        Strategy (schema-aware):
          1. Company node properties → financial highlights, sector, description
          2. CONTAINS relationships → peer/competitor Company nodes (if any)
          3. Diverse text chunks → one representative chunk per unique section
             (description, highlights, valuation, ESG, analyst_ratings, earnings_call, etc.)

        This replaces the old HAS_STRATEGY|HAS_FACT query which returned 0 results
        because those relationship types do not exist in the current graph schema.
        """
        if not ticker:
            return []

        facts: List[Dict[str, Any]] = []

        with self.driver.session(database=None) as session:
            # ── 1. Company node properties ────────────────────────────────────
            try:
                company_row = session.run(
                    "MATCH (c:Company {ticker: $ticker}) RETURN properties(c) AS props LIMIT 1",
                    ticker=ticker,
                ).single()
                if company_row:
                    props = dict(company_row["props"])
                    # Omit embedding vectors and very long string blobs
                    clean_props = {
                        k: v for k, v in props.items()
                        if k != "embedding" and not isinstance(v, (list, bytes))
                        and (not isinstance(v, str) or len(v) < 2000)
                    }
                    if clean_props:
                        facts.append({
                            "relationship": "COMPANY_PROFILE",
                            "node": clean_props,
                            "relationship_properties": {},
                        })
            except Exception as exc:
                logger.debug("[Neo4j] fetch_graph_facts company props failed for %s: %s", ticker, exc)

            # ── 2. CONTAINS peer relationships (Company → Company) ────────────
            try:
                peer_rows = session.run(
                    """
                    MATCH (c:Company {ticker: $ticker})-[:CONTAINS]->(peer:Company)
                    RETURN peer.ticker AS peer_ticker, peer.Name AS peer_name,
                           peer.Sector AS sector, peer.Industry AS industry
                    LIMIT 10
                    """,
                    ticker=ticker,
                ).data()
                for row in peer_rows:
                    facts.append({
                        "relationship": "CONTAINS_PEER",
                        "node": {k: v for k, v in row.items() if v is not None},
                        "relationship_properties": {},
                    })
            except Exception as exc:
                logger.debug("[Neo4j] fetch_graph_facts peers failed for %s: %s", ticker, exc)

            # ── 3. One chunk per unique section (diverse knowledge coverage) ──
            try:
                chunk_rows = session.run(
                    """
                    MATCH (ch:Chunk {ticker: $ticker})
                    WITH ch.section AS section, collect(ch)[0] AS sample_chunk
                    RETURN section, sample_chunk.text AS text,
                           sample_chunk.filing_date AS filing_date
                    ORDER BY section
                    LIMIT $limit
                    """,
                    ticker=ticker,
                    limit=limit - len(facts),
                ).data()
                for row in chunk_rows:
                    section = row.get("section") or "unknown"
                    text = row.get("text") or ""
                    if text:
                        facts.append({
                            "relationship": f"HAS_CHUNK::{section}",
                            "node": {
                                "section": section,
                                "text": text[:500],  # truncate for prompt budget
                                "filing_date": row.get("filing_date"),
                            },
                            "relationship_properties": {},
                        })
            except Exception as exc:
                logger.debug("[Neo4j] fetch_graph_facts chunks failed for %s: %s", ticker, exc)

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
               node.institution AS institution,
               node.source_file AS source_file,
               node.source_name AS source_name,
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
            section = row.get("section") or ""
            institution = row.get("institution") or ""
            source_file = row.get("source_file") or ""
            filing_date = row.get("filing_date") or ""
            ticker_sym = row.get("ticker_symbol") or ""

            # Prefer the stored source_name (populated during ingestion).
            # Fall back to deriving one from source_file when the node pre-dates
            # the ingestion fix (source_name not yet populated).
            stored_source_name = row.get("source_name") or ""
            if stored_source_name:
                source_name = stored_source_name
            elif source_file:
                # Strip one or two .pdf extensions from the filename stem
                stem = source_file
                for _ in range(2):
                    if stem.lower().endswith(".pdf"):
                        stem = stem[:-4]
                source_name = stem.strip() or source_file
            else:
                source_name = section.replace("_", " ").title() if section else "document"
            metadata = {
                "section": section,
                "filing_date": filing_date,
                "ticker": ticker_sym,
                "institution": institution,
                "source_file": source_file,
                "source_name": source_name,
            }
            chunk_id = row.get("chunk_id") or f"neo4j::{ticker_sym}::{section}::{len(chunks)}"
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
        if raw >= 0:
            return max(0.0, 1.0 - raw)
        return 0.0

    def insert_chunks(
        self,
        ticker: str,
        chunks: List[Dict[str, Any]],
        embedding_client: Optional["EmbeddingClient"] = None,
        embedding_model_version: Optional[str] = None,
    ) -> int:
        """Insert or update Chunk nodes in Neo4j.

        Args:
            ticker: Company ticker symbol.
            chunks: List of chunk dicts (must contain at least ``chunk_id``,
                ``text``, ``section``, ``filing_date``).
            embedding_client: If provided, embeddings are computed on-the-fly
                for chunks that do not already have an ``embedding`` key.
            embedding_model_version: Version string stamped onto each Chunk node
                as ``chunk.embedding_version`` so retrieval can filter by version.
                Defaults to the config value (``EMBEDDING_MODEL_VERSION`` env var).
        """
        version = embedding_model_version or self.config.embedding_model_version
        rows = []
        for chunk in chunks:
            row = dict(chunk)
            row.setdefault("embedding_version", version)
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
              chunk.embedding = row.embedding,
              chunk.embedding_version = row.embedding_version
        MERGE (c)-[:HAS_CHUNK]->(chunk)
        """
        with self.driver.session(database=None) as session:
            session.run(cypher, rows=rows)
        return len(rows)

    def fetch_community_summary(self, ticker: Optional[str]) -> Optional[str]:
        """Build a graph-community summary for *ticker* using relationship-count centrality."""
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
        except Exception as exc:
            logger.warning("[Neo4j] community summary query failed for %s: %s", ticker, exc)
            return None

        if not rows:
            return None

        rel_parts: List[str] = []
        all_samples: List[Dict[str, Any]] = []
        for row in rows:
            rel_type = row.get("rel_type", "UNKNOWN")
            count = row.get("rel_count", 0)
            rel_parts.append(f"{rel_type} ({count} edges)")
            for node in row.get("sample_nodes") or []:
                clean = {
                    k: _serialize_value(v) for k, v in node.items()
                    if k not in ("embedding", "vector") and v is not None
                }
                if clean:
                    all_samples.append(clean)

        company_name = ticker
        try:
            with self.driver.session(database=None) as session:
                name_result = session.run(
                    "MATCH (c:Company {ticker: $ticker}) RETURN c.name AS name LIMIT 1",
                    ticker=ticker,
                )
                name_row = name_result.single()
                if name_row and name_row.get("name"):
                    company_name = name_row["name"]
        except Exception:
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
        """Fetch ETF/index constituent holdings for the given ETF ticker."""
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
        """Return the number of Chunk nodes for *ticker* in Neo4j."""
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

    def fetch_recent_chunks(self, ticker: str, limit: int = 30) -> List["Chunk"]:
        """
        Fetch the most recent text chunks for *ticker* from Neo4j.
        Used by the sentiment fallback to get recent document text without
        a vector search (no query embedding needed).
        """
        cypher = """
        MATCH (c:Company {ticker: $ticker})-[:HAS_CHUNK]->(ch:Chunk)
        RETURN ch.chunk_id AS chunk_id, ch.text AS text,
               ch.section AS section, ch.filing_date AS filing_date
        ORDER BY ch.filing_date DESC
        LIMIT $limit
        """
        try:
            with self.driver.session(database=None) as session:
                rows = session.run(cypher, ticker=ticker, limit=limit).data()
            return [
                Chunk(
                    chunk_id=row["chunk_id"] or f"neo4j::{ticker}::{i}",
                    text=row.get("text", ""),
                    score=0.0,
                    source="neo4j",
                    metadata={"section": row.get("section"), "filing_date": row.get("filing_date"), "ticker": ticker},
                )
                for i, row in enumerate(rows)
            ]
        except Exception as exc:
            logger.warning("[Neo4j] fetch_recent_chunks failed for %s: %s", ticker, exc)
            return []

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


# ──────────────────────────────────────────────────────────────────────────────
# PostgreSQL
# ──────────────────────────────────────────────────────────────────────────────
class PostgresConnector:
    def __init__(self, config: BusinessAnalystConfig) -> None:
        self.config = config

    def fetch_sentiment(
        self,
        ticker: Optional[str],
        max_age_days: int = 7,
    ) -> Optional["SentimentSnapshot"]:
        """Fetch the most recent sentiment row from PostgreSQL.

        Args:
            ticker: Ticker symbol (case-insensitive).
            max_age_days: If the latest row is older than this many days the
                method returns ``None`` so the caller can fall back to local NLP.
                Defaults to 7 days.  Set to 0 to disable the age check.

        Returns a ``SentimentSnapshot`` whose ``source`` field is annotated with
        the as_of_date so callers can display data freshness.
        """
        if not ticker:
            return None
        sql = """
        SELECT bullish_pct, bearish_pct, neutral_pct,
               COALESCE(trend, 'unknown') AS trend,
               as_of_date
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

            as_of_date = row.get("as_of_date")
            # Recency check: if the data is stale, signal caller to use fallback
            if max_age_days > 0 and as_of_date is not None:
                try:
                    if isinstance(as_of_date, str):
                        from datetime import date as _date
                        as_of_date = _date.fromisoformat(as_of_date[:10])
                    age_days = (datetime.now(timezone.utc).date() - as_of_date).days
                    if age_days > max_age_days:
                        logger.warning(
                            "[Sentiment] PostgreSQL row for %s is %d days old "
                            "(max_age_days=%d) — treating as stale, activating local fallback.",
                            ticker, age_days, max_age_days,
                        )
                        return None
                except Exception as age_exc:
                    logger.debug("[Sentiment] Could not parse as_of_date for %s: %s", ticker, age_exc)

            date_str = str(as_of_date) if as_of_date else "unknown"
            return SentimentSnapshot(
                bullish_pct=float(row.get("bullish_pct", 0.0)),
                bearish_pct=float(row.get("bearish_pct", 0.0)),
                neutral_pct=float(row.get("neutral_pct", 0.0)),
                trend=row.get("trend", "unknown"),
                source=f"postgresql:sentiment_trends (as_of={date_str})",
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
        # First try the new dedicated company_profiles table
        sql = """
        SELECT *
        FROM company_profiles
        WHERE ticker = %s
        LIMIT 1
        """
        try:
            conn = self._pg_connect()
            with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (ticker,))
                row = cur.fetchone()
            if row:
                return {k: (float(v) if isinstance(v, (int, float)) and k not in ("ticker", "id") else
                            str(v) if v is not None else None)
                        for k, v in dict(row).items()}
        except Exception as exc:
            logger.debug("[BA] fetch_company_profile primary query failed for %s: %s", ticker, exc)

        # Fallback to raw_fundamentals data_name='company_profile'
        sql = """
        SELECT payload, as_of_date
        FROM raw_fundamentals
        WHERE ticker_symbol = %s
          AND data_name = 'company_profile'
        ORDER BY as_of_date DESC
        LIMIT 1
        """
        try:
            conn = self._pg_connect()
            with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (ticker,))
                row = cur.fetchone()
            if row:
                payload = row["payload"]
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        pass
                if isinstance(payload, dict) and payload:
                    return payload
        except Exception as exc:
            logger.debug("[BA] fetch_company_profile raw_fundamentals fallback failed for %s: %s", ticker, exc)

        # Final fallback: company_profile_neo4j.json General section
        try:
            base_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "ingestion", "etl", "agent_data", ticker.upper()
            )
            neo4j_path = os.path.normpath(os.path.join(base_dir, "company_profile_neo4j.json"))
            if os.path.exists(neo4j_path):
                with open(neo4j_path, "r", encoding="utf-8") as f:
                    neo4j_data = json.load(f)
                g = neo4j_data.get("General", {})
                if g:
                    logger.info("[BA] fetch_company_profile: using company_profile_neo4j.json fallback for %s", ticker)
                    return {
                        "ticker":              ticker,
                        "name":                g.get("Name"),
                        "exchange":            g.get("Exchange"),
                        "sector":              g.get("Sector"),
                        "industry":            g.get("Industry"),
                        "gic_sector":          g.get("GicSector"),
                        "description":         g.get("Description"),
                        "address":             g.get("Address"),
                        "city":                g.get("AddressData", {}).get("City") if isinstance(g.get("AddressData"), dict) else None,
                        "state":               g.get("AddressData", {}).get("State") if isinstance(g.get("AddressData"), dict) else None,
                        "country":             g.get("CountryName") or (g.get("AddressData", {}).get("Country") if isinstance(g.get("AddressData"), dict) else None),
                        "phone":               g.get("Phone"),
                        "web_url":             g.get("WebURL"),
                        "full_time_employees": g.get("FullTimeEmployees"),
                        "fiscal_year_end":     g.get("FiscalYearEnd"),
                        "ipo_date":            g.get("IPODate"),
                        "currency":            g.get("CurrencyCode"),
                        "isin":                g.get("ISIN"),
                        "cusip":               g.get("CUSIP"),
                        "cik":                 g.get("CIK"),
                        "is_delisted":         False,
                    }
        except Exception as exc:
            logger.debug("[BA] fetch_company_profile neo4j json fallback failed for %s: %s", ticker, exc)
        return None

    def fetch_analyst_ratings(self, ticker: str) -> Optional[Dict]:
        """Fetch latest analyst ratings for the ticker.

        Falls back to raw_fundamentals data_name='analyst_ratings', then to
        company_profile_neo4j.json AnalystRatings section.
        """
        sql = """
        SELECT *
        FROM analyst_ratings
        WHERE ticker = %s
        ORDER BY as_of_date DESC
        LIMIT 1
        """
        try:
            conn = self._pg_connect()
            with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (ticker,))
                row = cur.fetchone()
            if row:
                return {k: (float(v) if isinstance(v, (int, float)) and k not in ("ticker", "id") else
                            str(v) if v is not None else None)
                        for k, v in dict(row).items()}
        except Exception as exc:
            logger.debug("[BA] fetch_analyst_ratings primary query failed for %s: %s", ticker, exc)

        # Fallback 1: raw_fundamentals data_name='analyst_ratings'
        sql = """
        SELECT payload, as_of_date
        FROM raw_fundamentals
        WHERE ticker_symbol = %s
          AND data_name = 'analyst_ratings'
        ORDER BY as_of_date DESC
        LIMIT 1
        """
        try:
            conn = self._pg_connect()
            with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (ticker,))
                row = cur.fetchone()
            if row:
                payload = row["payload"]
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        pass
                if isinstance(payload, dict) and payload:
                    logger.info("[BA] fetch_analyst_ratings: using raw_fundamentals fallback for %s", ticker)
                    def _fn(x):
                        try:
                            return float(x) if x is not None else None
                        except (TypeError, ValueError):
                            return None
                    return {
                        "ticker":           ticker,
                        "rating":           _fn(payload.get("Rating")),
                        "target_price":     _fn(payload.get("TargetPrice")),
                        "strong_buy_count": _fn(payload.get("StrongBuy")),
                        "buy_count":        _fn(payload.get("Buy")),
                        "hold_count":       _fn(payload.get("Hold")),
                        "sell_count":       _fn(payload.get("Sell")),
                        "strong_sell_count": _fn(payload.get("StrongSell")),
                    }
        except Exception as exc:
            logger.debug("[BA] fetch_analyst_ratings raw_fundamentals fallback failed for %s: %s", ticker, exc)

        # Fallback 2: company_profile_neo4j.json AnalystRatings section
        try:
            base_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "ingestion", "etl", "agent_data", ticker.upper()
            )
            neo4j_path = os.path.normpath(os.path.join(base_dir, "company_profile_neo4j.json"))
            if os.path.exists(neo4j_path):
                with open(neo4j_path, "r", encoding="utf-8") as f:
                    neo4j_data = json.load(f)
                ar = neo4j_data.get("AnalystRatings", {})
                if ar:
                    logger.info("[BA] fetch_analyst_ratings: using company_profile_neo4j.json fallback for %s", ticker)
                    def _fn(x):
                        try:
                            return float(x) if x is not None else None
                        except (TypeError, ValueError):
                            return None
                    return {
                        "ticker":           ticker,
                        "rating":           _fn(ar.get("Rating")),
                        "target_price":     _fn(ar.get("TargetPrice")),
                        "strong_buy_count": _fn(ar.get("StrongBuy")),
                        "buy_count":        _fn(ar.get("Buy")),
                        "hold_count":       _fn(ar.get("Hold")),
                        "sell_count":       _fn(ar.get("Sell")),
                        "strong_sell_count": _fn(ar.get("StrongSell")),
                    }
        except Exception as exc:
            logger.debug("[BA] fetch_analyst_ratings neo4j json fallback failed for %s: %s", ticker, exc)
        return None

    def fetch_news(self, ticker: str, limit: int = 20) -> List[Dict]:
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
            if isinstance(payload, list):
                results.extend(payload)
            elif isinstance(payload, dict):
                results.append(payload)
        return results[:limit]

    def fetch_insider_transactions(self, ticker: str, limit: int = 20) -> List[Dict]:
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


# ──────────────────────────────────────────────────────────────────────────────
# PgVector connector
# ──────────────────────────────────────────────────────────────────────────────
class PgVectorConnector:
    """Semantic search over the text_chunks table using pgvector cosine similarity."""

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
        emb_str = "[" + ",".join(f"{x:.8f}" for x in vector) + "]"
        ticker_filter = "AND ticker = %s" if ticker else ""
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


# ──────────────────────────────────────────────────────────────────────────────
# Keyword scorer (BM25-lite)
# ──────────────────────────────────────────────────────────────────────────────
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
    import math as _math
    return [1.0 / (1.0 + _math.exp(-s)) for s in scores]


def _rrf_fuse(
    ranked_lists: List[List[Chunk]],
    k: int = 60,
) -> List[Chunk]:
    """Reciprocal Rank Fusion over multiple ranked chunk lists."""
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Chunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunk_map:
                chunk_map[cid] = chunk

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


def _apply_time_decay(chunks: List[Chunk], lambda_: float) -> List[Chunk]:
    """Apply exponential time-decay to chunk scores based on filing_date metadata.

    Decay multiplier = exp(-lambda_ * age_days / 365).  Chunks with no
    parseable date are left unchanged (neutral multiplier of 1.0).

    Returns a new sorted list (highest score first).
    """
    if lambda_ <= 0.0:
        return chunks
    now = datetime.now(timezone.utc).date()
    result: List[Chunk] = []
    for chunk in chunks:
        filing_date_raw = (chunk.metadata or {}).get("filing_date")
        decay = 1.0
        if filing_date_raw:
            try:
                if hasattr(filing_date_raw, "year"):  # already a date/datetime
                    fd = filing_date_raw if not hasattr(filing_date_raw, "date") else filing_date_raw.date()
                else:
                    fd = datetime.fromisoformat(str(filing_date_raw)[:10]).date()
                age_days = max((now - fd).days, 0)
                decay = math.exp(-lambda_ * age_days / 365.0)
            except Exception:
                pass  # unparseable date — no penalty
        result.append(
            Chunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=chunk.score * decay,
                source=chunk.source,
                metadata=chunk.metadata,
            )
        )
    result.sort(key=lambda c: -c.score)
    return result


def _mmr_select(
    chunks: List[Chunk],
    final_k: int,
    mmr_lambda: float,
) -> List[Chunk]:
    """Maximal Marginal Relevance selection using TF-IDF bag-of-words similarity.

    mmr_lambda=1.0 → pure relevance (equivalent to top-k by score).
    mmr_lambda=0.0 → pure diversity.

    Uses normalised unigram token overlap as a fast proxy for cosine similarity
    (no embedding call required).
    """
    if mmr_lambda >= 1.0 or len(chunks) <= final_k:
        return chunks[:final_k]

    def _token_set(text: str) -> set:
        return set(_re_top.findall(r"\b\w+\b", text.lower()))

    def _jaccard(a: set, b: set) -> float:
        union = a | b
        return len(a & b) / len(union) if union else 0.0

    token_sets = [_token_set(c.text) for c in chunks]
    selected: List[Chunk] = []
    selected_indices: List[int] = []
    remaining = list(range(len(chunks)))

    while len(selected) < final_k and remaining:
        if not selected_indices:
            # First pick: highest relevance score
            best_idx = max(remaining, key=lambda i: chunks[i].score)
        else:
            best_score = float("-inf")
            best_idx = remaining[0]
            for i in remaining:
                relevance = chunks[i].score
                max_sim = max(
                    _jaccard(token_sets[i], token_sets[j]) for j in selected_indices
                )
                mmr = mmr_lambda * relevance - (1.0 - mmr_lambda) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
        selected.append(chunks[best_idx])
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return selected


# ──────────────────────────────────────────────────────────────────────────────
# Ticker identity helper
# ──────────────────────────────────────────────────────────────────────────────

def _chunk_ticker_matches(chunk: "Chunk", ticker: str) -> bool:
    meta = chunk.metadata or {}
    for key in ("ticker", "ticker_symbol"):
        val = meta.get(key)
        if val:
            return str(val).upper() == ticker.upper()
    parts = chunk.chunk_id.split("::")
    if len(parts) >= 2:
        candidate = parts[1].upper()
        if 1 <= len(candidate) <= 6 and candidate.isalpha():
            return candidate == ticker.upper()
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
        """
        Score retrieval quality against CRAG thresholds.
        Thresholds are now configurable via env vars (CRAG_CORRECT_THRESHOLD,
        CRAG_AMBIGUOUS_THRESHOLD) — no code change needed to tune them.
        """
        if not chunks:
            return CRAGEvaluation(CRAGStatus.INCORRECT, 0.0)

        if ticker:
            ticker_chunks = [c for c in chunks if _chunk_ticker_matches(c, ticker)]
            if not ticker_chunks:
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

        # Log every CRAG judgment for debugging and threshold calibration
        if top_score >= self.config.crag_correct_threshold:
            status = CRAGStatus.CORRECT
        elif top_score >= self.config.crag_ambiguous_threshold:
            status = CRAGStatus.AMBIGUOUS
        else:
            status = CRAGStatus.INCORRECT

        logger.info(
            "[CRAG] score=%.3f → %s (thresholds: correct≥%.2f, ambiguous≥%.2f)",
            top_score, status.value,
            self.config.crag_correct_threshold,
            self.config.crag_ambiguous_threshold,
        )
        return CRAGEvaluation(status, top_score)


# ──────────────────────────────────────────────────────────────────────────────
# Toolkit façade
# ──────────────────────────────────────────────────────────────────────────────
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
        global _SEMANTIC_CACHE
        _SEMANTIC_CACHE._max = self.config.semantic_cache_max_entries

    # ------------------------------------------------------------------
    # Metadata pre-check (cached)
    # ------------------------------------------------------------------

    def get_metadata_profile(self, ticker: str) -> MetadataProfile:
        cached = _METADATA_CACHE.get(ticker)
        if cached is not None:
            logger.debug("[MetadataCache] HIT for ticker=%s", ticker)
            return cached

        logger.debug("[MetadataCache] MISS for ticker=%s — querying backends", ticker)

        neo4j_chunk_count = 0
        neo4j_index_ready = False
        try:
            neo4j_chunk_count = self.neo4j.fetch_chunk_count(ticker)
            neo4j_index_ready = self.neo4j.is_vector_index_online()
        except Exception as exc:
            logger.warning("[MetadataProfile] Neo4j query failed for %s: %s", ticker, exc)

        pgvec_count = 0
        pgvec_table_ready = False
        pgvec_index = False
        try:
            pgvec_count = self.pgvec.chunk_count(ticker)
            pgvec_table_ready = True
            pgvec_index = self.pgvec.has_embedding_index()
        except Exception as exc:
            logger.warning("[MetadataProfile] pgvector query failed for %s: %s", ticker, exc)

        has_sentiment = False
        sentiment_last_updated: Optional[str] = None
        try:
            # Use max_age_days=0 to skip the recency gate here — we just want
            # to know if *any* row exists and when it was last updated.
            snap = self.pg.fetch_sentiment(ticker, max_age_days=0)
            has_sentiment = snap is not None
            if snap is not None and snap.source:
                # Extract date from source annotation, e.g. "postgresql:sentiment_trends (as_of=2025-01-15)"
                import re as _re_local
                _m = _re_local.search(r"as_of=(\S+)\)", snap.source)
                if _m:
                    sentiment_last_updated = _m.group(1)
        except Exception as exc:
            logger.warning("[MetadataProfile] sentiment check failed for %s: %s", ticker, exc)

        has_pg_fundamentals = False
        has_pg_timeseries = False
        try:
            # Lightweight direct SQL — avoids check_all()'s ThreadPoolExecutor
            # which opens new Neo4j + Postgres connections and can hang on the
            # host due to Docker NAT latency (8-12 s).  The existing self.pg
            # connection config is already proven to work by the sentiment check
            # above, so reuse the same pattern.
            _avail_conn = psycopg2.connect(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                dbname=self.config.postgres_db,
                user=self.config.postgres_user,
                password=self.config.postgres_password,
            )
            with closing(_avail_conn), _avail_conn.cursor() as _cur:
                _cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM financial_statements WHERE ticker = %s LIMIT 1)",
                    (ticker,),
                )
                has_pg_fundamentals = bool((_cur.fetchone() or [False])[0])
                _cur.execute(
                    "SELECT EXISTS(SELECT 1 FROM raw_timeseries WHERE ticker_symbol = %s LIMIT 1)",
                    (ticker,),
                )
                has_pg_timeseries = bool((_cur.fetchone() or [False])[0])
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
    # Sentiment: PostgreSQL → local NLP fallback
    # ------------------------------------------------------------------

    def get_sentiment_snapshot(self, ticker: str) -> Optional[SentimentSnapshot]:
        """
        Main entry point for sentiment data.

        Flow:
          1. Query PostgreSQL sentiment_trends (primary source, EODHD data).
             If the latest row is older than 7 days it is treated as stale and
             we fall through to the local fallback immediately.
          2. If empty/stale/unavailable → fetch recent Neo4j text chunks and score
             locally using VADER → TextBlob → keyword heuristic (in that order).
          3. If Neo4j also has no chunks → return None.

        The fallback is logged clearly so you can diagnose ingestion lag.
        """
        # Primary: PostgreSQL (with 7-day recency gate)
        try:
            snap = self.pg.fetch_sentiment(ticker, max_age_days=7)
            if snap is not None:
                logger.info(
                    "[Sentiment] ticker=%s → source=postgresql trend=%s",
                    ticker, snap.trend,
                )
                return snap
            logger.warning(
                "[Sentiment] sentiment_trends is EMPTY or STALE (>7 days) for "
                "ticker=%s — activating local chunk-based fallback.",
                ticker,
            )
        except Exception as exc:
            logger.warning(
                "[Sentiment] PostgreSQL query failed for %s: %s — activating fallback.",
                ticker, exc,
            )

        # Fallback: local NLP over recent Neo4j chunks
        try:
            recent_chunks = self.neo4j.fetch_recent_chunks(ticker, limit=30)
            if not recent_chunks:
                logger.warning(
                    "[Sentiment] No Neo4j chunks found for ticker=%s — cannot compute local sentiment.",
                    ticker,
                )
                return None
            snap = _local_sentiment_from_chunks(recent_chunks)
            if snap:
                logger.info(
                    "[Sentiment] ticker=%s → source=local_fallback trend=%s (used %d chunks)",
                    ticker, snap.trend, len(recent_chunks),
                )
            return snap
        except Exception as exc:
            logger.error("[Sentiment] Local fallback failed for %s: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Fast-path retrieval
    # ------------------------------------------------------------------

    def retrieve_fast(self, query: str, ticker: Optional[str]) -> RetrievalResult:
        cached = _SEMANTIC_CACHE.get(ticker, query)
        if cached is not None:
            logger.info("[SemanticCache] HIT for ticker=%s (fast path)", ticker)
            return cached

        try:
            vector = self.embedding.embed(query)
            chunks = self.neo4j.vector_search(vector, ticker, top_k=self.config.fast_path_top_k)

            # Filter out garbled / boilerplate chunks before scoring
            before = len(chunks)
            chunks = [c for c in chunks if not _is_boilerplate(c.text)]
            if len(chunks) < before:
                logger.info(
                    "[retrieve_fast] Filtered %d boilerplate/garbled chunks (kept %d)",
                    before - len(chunks), len(chunks),
                )

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
    # Multi-stage retrieval
    # ------------------------------------------------------------------

    def retrieve_multi_stage(self, query: str, ticker: Optional[str]) -> RetrievalResult:
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

        # Filter out garbled / boilerplate chunks before reranking
        _n4j_before = len(neo4j_chunks)
        _pgv_before = len(pgvec_chunks)
        neo4j_chunks = [c for c in neo4j_chunks if not _is_boilerplate(c.text)]
        pgvec_chunks = [c for c in pgvec_chunks if not _is_boilerplate(c.text)]
        _filtered = (_n4j_before - len(neo4j_chunks)) + (_pgv_before - len(pgvec_chunks))
        if _filtered:
            logger.info(
                "[MultiStage] Filtered %d boilerplate/garbled chunks (neo4j kept %d, pg kept %d)",
                _filtered, len(neo4j_chunks), len(pgvec_chunks),
            )

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
            ce_ranked = sorted(recall_pool, key=lambda c: -c.score)

        ranked_lists = [neo4j_chunks, pgvec_chunks, bm25_ranked, ce_ranked]
        fused = _rrf_fuse([lst for lst in ranked_lists if lst], k=60)

        # ── Time-decay re-ranking ──────────────────────────────────────────────
        # Penalise older chunks by exp(-lambda * age_days / 365) before final cut.
        fused = _apply_time_decay(fused, self.config.time_decay_lambda)
        logger.debug(
            "[MultiStage] Time-decay applied (lambda=%.2f); top score after decay=%.3f",
            self.config.time_decay_lambda,
            fused[0].score if fused else 0.0,
        )

        # ── MMR selection ─────────────────────────────────────────────────────
        # Replace naive top-k cut with Maximal Marginal Relevance to reduce
        # redundant chunks and increase topical diversity in final result.
        fused = _mmr_select(fused, final_k, self.config.mmr_lambda)
        logger.debug(
            "[MultiStage] MMR selected %d/%d chunks (mmr_lambda=%.2f)",
            len(fused), final_k, self.config.mmr_lambda,
        )

        # Section-diversity guarantee: ensure at least min_chunks_per_section
        # chunks from each key section (earnings_call, broker_report, 10-K,
        # annual_report) appear in the final result, even when their RRF scores
        # are lower than other sections.
        _min_sec = self.config.min_chunks_per_section
        if _min_sec > 0 and recall_pool:
            _key_sections = ("earnings_call", "broker_report", "10-K", "annual_report")
            _fused_ids = {c.chunk_id for c in fused}
            _section_counts: Dict[str, int] = {}
            for c in fused:
                sec = (c.metadata or {}).get("section") or ""
                _section_counts[sec] = _section_counts.get(sec, 0) + 1

            # Fallback pool sorted by raw vector score (best from recall_pool)
            _fallback_pool = sorted(recall_pool, key=lambda c: -c.score)
            _added = 0
            for _sec in _key_sections:
                _deficit = _min_sec - _section_counts.get(_sec, 0)
                if _deficit <= 0:
                    continue
                _candidates = [
                    c for c in _fallback_pool
                    if c.chunk_id not in _fused_ids
                    and (c.metadata or {}).get("section") == _sec
                ]
                for c in _candidates[:_deficit]:
                    fused.append(c)
                    _fused_ids.add(c.chunk_id)
                    _added += 1
            if _added:
                logger.info(
                    "[MultiStage] Section-diversity: injected %d chunk(s) to satisfy "
                    "min_chunks_per_section=%d for sections %s",
                    _added, _min_sec, _key_sections,
                )

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

    def retrieve(self, query: str, ticker: Optional[str]) -> RetrievalResult:
        """Default retrieval (wraps retrieve_multi_stage for the complex path)."""
        return self.retrieve_multi_stage(query, ticker)

    def fetch_community_summary(self, ticker: Optional[str]) -> Optional[str]:
        try:
            return self.neo4j.fetch_community_summary(ticker)
        except Exception as exc:
            logger.warning("Community summary fetch failed for %s: %s", ticker, exc)
            return None

    def fetch_company_overview(self, ticker: Optional[str]) -> Optional[Dict[str, Any]]:
        try:
            return self.neo4j.fetch_company_overview(ticker)
        except Exception as exc:
            logger.warning("Company overview fetch failed for %s: %s", ticker, exc)
            return None

    def fetch_sentiment(self, ticker: Optional[str]) -> Optional[SentimentSnapshot]:
        """Backward-compat wrapper — prefer get_sentiment_snapshot() for new code."""
        if not ticker:
            return None
        return self.get_sentiment_snapshot(ticker)

    def fetch_esg(self, ticker: str) -> Optional[Dict]:
        try:
            return self.pg.fetch_esg(ticker)
        except Exception as exc:
            logger.warning("ESG fetch failed for %s: %s", ticker, exc)
            return None

    def fetch_social_sentiment(self, ticker: str, limit: int = 10) -> List[Dict]:
        try:
            return self.pg.fetch_social_sentiment(ticker, limit)
        except Exception as exc:
            logger.warning("Social sentiment fetch failed for %s: %s", ticker, exc)
            return []

    def fetch_insider_signals(self, ticker: str, limit: int = 20) -> List[Dict]:
        try:
            return self.neo4j.fetch_insider_signals(ticker, limit)
        except Exception as exc:
            logger.warning("Insider signals fetch failed for %s: %s", ticker, exc)
            return []

    def fetch_company_profile(self, ticker: str) -> Optional[Dict]:
        try:
            return self.pg.fetch_company_profile(ticker)
        except Exception as exc:
            logger.warning("Company profile fetch failed for %s: %s", ticker, exc)
            return None

    def fetch_news(self, ticker: str, limit: int = 20) -> List[Dict]:
        try:
            return self.pg.fetch_news(ticker, limit)
        except Exception as exc:
            logger.warning("News fetch failed for %s: %s", ticker, exc)
            return []

    def fetch_insider_transactions(self, ticker: str, limit: int = 20) -> List[Dict]:
        try:
            return self.pg.fetch_insider_transactions(ticker, limit)
        except Exception as exc:
            logger.warning("Insider transactions fetch failed for %s: %s", ticker, exc)
            return []

    def fetch_institutional_holders(self, ticker: str, limit: int = 20) -> List[Dict]:
        try:
            return self.pg.fetch_institutional_holders(ticker, limit)
        except Exception as exc:
            logger.warning("Institutional holders fetch failed for %s: %s", ticker, exc)
            return []

    def fetch_financial_calendar(self, ticker: str, limit: int = 10) -> List[Dict]:
        try:
            return self.pg.fetch_financial_calendar(ticker, limit)
        except Exception as exc:
            logger.warning("Financial calendar fetch failed for %s: %s", ticker, exc)
            return []

    def fetch_etf_constituents(self, ticker: str, limit: int = 50) -> List[Dict]:
        try:
            return self.neo4j.fetch_etf_constituents(ticker, limit)
        except Exception as exc:
            logger.warning("ETF constituents fetch failed for %s: %s", ticker, exc)
            return []

    def fetch_textual_documents(self, ticker: str, limit: int = 20) -> List[Dict]:
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
    "rule_based_classify",
]
