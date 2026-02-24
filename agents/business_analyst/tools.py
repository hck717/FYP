# agents/business_analyst/tools.py
"""
Data connectors for the Business Analyst Agent.

Provides:
  - Neo4j: company graph facts + vector search (hybrid retrieval)
  - Qdrant: recent news semantic search
  - PostgreSQL: sentiment_trends data
  - Utility: JSON extraction, BM25 scoring
"""
import os
import re
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠️  neo4j driver not installed: pip install neo4j")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
NEO4J_URI       = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER      = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD  = os.getenv("NEO4J_PASSWORD", "changeme_neo4j_password")

QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION     = os.getenv("QDRANT_COLLECTION_NAME", "financial_documents")
RAG_TOP_K       = int(os.getenv("RAG_TOP_K", "8"))

PG_HOST         = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT         = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB           = os.getenv("POSTGRES_DB", "airflow")
PG_USER         = os.getenv("POSTGRES_USER", "airflow")
PG_PASS         = os.getenv("POSTGRES_PASSWORD", "airflow")

EMBED_MODEL     = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANK_MODEL    = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Lazy-loaded singletons ─────────────────────────────────────────────────────
_embedder: Optional[SentenceTransformer] = None
_reranker: Optional[CrossEncoder] = None
_neo4j_driver = None
_qdrant_client: Optional[QdrantClient] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info(f"[tools] Loading embedder: {EMBED_MODEL}")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"[tools] Loading reranker: {RERANK_MODEL}")
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is None and NEO4J_AVAILABLE:
        try:
            _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with _neo4j_driver.session() as s:
                s.run("RETURN 1")
            logger.info("[tools] Neo4j connected")
        except Exception as e:
            logger.warning(f"[tools] Neo4j connection failed: {e}")
            _neo4j_driver = None
    return _neo4j_driver


def get_qdrant() -> Optional[QdrantClient]:
    global _qdrant_client
    if _qdrant_client is None:
        try:
            _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            _qdrant_client.get_collections()
            logger.info("[tools] Qdrant connected")
        except Exception as e:
            logger.warning(f"[tools] Qdrant connection failed: {e}")
            _qdrant_client = None
    return _qdrant_client


# ── PostgreSQL: Sentiment ──────────────────────────────────────────────────────
def fetch_sentiment(ticker: str) -> Dict:
    """
    Fetch latest sentiment_trends row for a ticker from PostgreSQL.
    Table schema (from ingestion/etl/load_postgres.py):
      sentiment_trends(ticker, date, bullish_pct, bearish_pct, neutral_pct, ingested_at)
    Returns dict with pct fields, or empty dict on failure.
    """
    try:
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, dbname=PG_DB,
            user=PG_USER, password=PG_PASS,
            connect_timeout=5
        )
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT bullish_pct, bearish_pct, neutral_pct, date
                    FROM sentiment_trends
                    WHERE ticker = %s
                    ORDER BY date DESC
                    LIMIT 2
                """, (ticker,))
                rows = cur.fetchall()
        conn.close()

        if not rows:
            return {}

        latest = dict(rows[0])
        # Compute simple trend: compare latest vs previous if available
        trend = "stable"
        if len(rows) == 2:
            prev_bull = float(rows[1].get("bullish_pct") or 0)
            curr_bull = float(latest.get("bullish_pct") or 0)
            diff = curr_bull - prev_bull
            if diff > 3:
                trend = "improving"
            elif diff < -3:
                trend = "deteriorating"

        return {
            "bullish_pct": float(latest.get("bullish_pct") or 0),
            "bearish_pct": float(latest.get("bearish_pct") or 0),
            "neutral_pct": float(latest.get("neutral_pct") or 0),
            "trend": trend,
            "source": "postgresql:sentiment_trends",
        }
    except Exception as e:
        logger.warning(f"[tools] PostgreSQL sentiment fetch failed: {e}")
        return {}


# ── Neo4j: Company properties ──────────────────────────────────────────────────
def fetch_company_profile(ticker: str) -> Dict:
    """
    Fetch :Company node properties from Neo4j.
    Returns dict of company properties (name, sector, market_cap, pe_ratio, etc.)
    """
    driver = get_neo4j_driver()
    if not driver:
        return {}
    try:
        with driver.session() as session:
            result = session.run(
                "MATCH (c:Company {ticker: $ticker}) RETURN properties(c) AS props",
                ticker=ticker
            )
            record = result.single()
            if not record:
                return {}
            props = dict(record["props"])

        # Map known Neo4j property names → clean output keys
        return {
            "name":          props.get("Name") or props.get("name") or ticker,
            "sector":        props.get("Sector") or props.get("sector"),
            "market_cap":    _safe_float(props.get("Highlights_MarketCapitalization")),
            "pe_ratio":      _safe_float(props.get("Valuation_TrailingPE") or props.get("Highlights_PERatio")),
            "profit_margin": _safe_float(props.get("Highlights_ProfitMargin")),
            "description":   props.get("Description") or props.get("description") or "",
        }
    except Exception as e:
        logger.warning(f"[tools] Neo4j company profile fetch failed: {e}")
        return {}


def fetch_graph_facts(ticker: str, keyword: str = "business") -> List[str]:
    """
    Cypher structural traversal — fetch nodes directly linked to the company.
    Returns list of text strings (strategy, risk, product nodes).
    """
    driver = get_neo4j_driver()
    if not driver:
        return []
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Company {ticker: $ticker})-[:HAS_STRATEGY|FACES_RISK|OFFERS_PRODUCT]->(n)
                WHERE toLower(n.description) CONTAINS toLower($keyword)
                   OR toLower(coalesce(n.title,'')) CONTAINS toLower($keyword)
                RETURN coalesce(n.title,'') + ': ' + coalesce(n.description,'') AS text
                LIMIT 12
            """, ticker=ticker, keyword=keyword)
            rows = [r["text"] for r in result if r["text"].strip()]
        return rows
    except Exception as e:
        logger.warning(f"[tools] Neo4j graph traversal failed: {e}")
        return []


def neo4j_vector_search(query: str, ticker: str, k: int = 10) -> List[str]:
    """
    Neo4j native vector index search on :Chunk nodes.
    Falls back to empty list if vector index not available.
    """
    driver = get_neo4j_driver()
    if not driver:
        return []
    try:
        embedding = get_embedder().encode(query).tolist()
        with driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('chunk_embedding', $k, $embedding)
                YIELD node, score
                WHERE node.ticker CONTAINS $ticker
                RETURN node.text AS text, node.chunk_id AS chunk_id, score
                ORDER BY score DESC
            """, k=k, embedding=embedding, ticker=ticker)
            rows = []
            for r in result:
                text = r.get("text", "")
                cid  = r.get("chunk_id", "unknown")
                if text:
                    rows.append(f"[{cid}] {text}")
        return rows
    except Exception as e:
        logger.warning(f"[tools] Neo4j vector search failed: {e}")
        return []


# ── Qdrant: News semantic search ───────────────────────────────────────────────
def qdrant_news_search(query: str, ticker: str, k: int = RAG_TOP_K) -> List[str]:
    """
    Semantic search over financial_documents collection in Qdrant.
    Filters by ticker_symbol payload field.
    Returns list of news text strings.
    """
    client = get_qdrant()
    if not client:
        return []
    try:
        embedding = get_embedder().encode(query).tolist()
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        hits: List[ScoredPoint] = client.search(
            collection_name=COLLECTION,
            query_vector=embedding,
            query_filter=Filter(
                must=[FieldCondition(
                    key="ticker_symbol",
                    match=MatchValue(value=ticker)
                )]
            ) if ticker else None,
            limit=k,
            with_payload=True,
        )
        results = []
        for hit in hits:
            payload = hit.payload or {}
            title   = payload.get("title", "")
            content = payload.get("content") or payload.get("text") or ""
            date    = payload.get("date") or payload.get("published_date") or "unknown"
            results.append(f"[news|{date}] {title}: {content[:400]}")
        return results
    except Exception as e:
        logger.warning(f"[tools] Qdrant news search failed: {e}")
        return []


# ── Hybrid Retrieval + CRAG ────────────────────────────────────────────────────
def hybrid_retrieve(
    query: str,
    ticker: str,
    k_final: int = 5,
) -> Tuple[List[str], float]:
    """
    Hybrid retrieval: Neo4j vector + Cypher graph + BM25 → Cross-Encoder rerank.
    Returns (top_k_docs, crag_confidence_score).

    Scoring: 30% BM25 + 70% Cross-Encoder (same as CRAG POC v4.2)
    """
    # Determine keyword from query for Cypher traversal
    q_lower = query.lower()
    keyword = "business"
    for kw in ["risk", "strategy", "revenue", "growth", "competition", "supply"]:
        if kw in q_lower:
            keyword = kw
            break

    # --- A: Neo4j vector search (dense) ---
    vec_docs = neo4j_vector_search(query, ticker, k=15)

    # --- B: Neo4j Cypher traversal (structured) ---
    graph_docs = fetch_graph_facts(ticker, keyword)

    # --- C: Qdrant news (recency signal) ---
    news_docs = qdrant_news_search(query, ticker, k=5)

    # Combine + deduplicate
    all_docs = list(dict.fromkeys(vec_docs + graph_docs + news_docs))
    if not all_docs:
        return [], 0.0

    # --- D: BM25 sparse scoring ---
    bm25_scores = [0.0] * len(all_docs)
    if BM25_AVAILABLE:
        try:
            tokenized = [d.lower().split() for d in all_docs]
            bm25 = BM25Okapi(tokenized)
            raw  = bm25.get_scores(query.lower().split())
            max_raw = max(raw) if max(raw) > 0 else 1.0
            bm25_scores = [s / max_raw for s in raw]
        except Exception:
            pass

    # --- E: Cross-Encoder rerank (semantic) ---
    reranker = get_reranker()
    pairs  = [[query, doc] for doc in all_docs]
    ce_scores = reranker.predict(pairs).tolist()

    # Hybrid score: 30% BM25 + 70% Cross-Encoder
    hybrid = [
        0.3 * bm25_scores[i] + 0.7 * ce_scores[i]
        for i in range(len(all_docs))
    ]
    ranked = sorted(zip(all_docs, hybrid), key=lambda x: x[1], reverse=True)

    top_docs  = [doc for doc, _ in ranked[:k_final]]
    top_score = ranked[0][1] if ranked else 0.0

    logger.info(f"[tools] hybrid_retrieve: {len(all_docs)} candidates → top score {top_score:.3f}")
    return top_docs, top_score


def crag_evaluate(score: float) -> str:
    """Map confidence score to CRAG status."""
    if score > 0.7:
        return "CORRECT"
    elif score >= 0.5:
        return "AMBIGUOUS"
    else:
        return "INCORRECT"


# ── Utilities ──────────────────────────────────────────────────────────────────
def extract_json_from_response(content: str) -> Optional[dict]:
    """Extract JSON from LLM response (handles markdown fences)."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass
    # Try extracting largest {...} block
    match2 = re.search(r"\{.*\}", content, re.DOTALL)
    if match2:
        try:
            return json.loads(match2.group(0))
        except json.JSONDecodeError:
            pass
    logger.warning("[tools] Could not extract JSON from response")
    return None


def _safe_float(val) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None
