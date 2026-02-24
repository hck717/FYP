# agents/business_analyst/tools.py
"""
Data connectors for the Business Analyst Agent.

Provides:
  - Neo4j: company graph facts + vector search (hybrid retrieval)
  - Qdrant: recent news semantic search
  - PostgreSQL: sentiment_trends data
  - Utility: JSON extraction, BM25 scoring

Host resolution:
  Each connector tries .env hostname (Docker-internal) first, then localhost.
  No .env changes needed when running from Mac terminal.

Import design — ALL heavy packages lazy-imported inside getters:
  neo4j / pandas       → get_neo4j_driver()
  qdrant_client        → get_qdrant()
  sentence_transformers→ get_embedder() / get_reranker()

Embedding model resolution:
  .env may set EMBEDDING_MODEL=nomic-embed-text (Ollama name).
  sentence-transformers uses HuggingFace IDs, not Ollama names.
  We map known Ollama names → HuggingFace equivalents, then fall back
  to all-MiniLM-L6-v2 if the resolved model still fails to load.
"""
import os
import re
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
logger = logging.getLogger(__name__)

# Suppress neo4j schema-warning noise (missing props/rels printed for every query)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

# ── Config ──────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme_neo4j_password")

QDRANT_HOST    = os.getenv("QDRANT_HOST",           "localhost")
QDRANT_PORT    = int(os.getenv("QDRANT_PORT",        "6333"))
COLLECTION     = os.getenv("QDRANT_COLLECTION_NAME", "financial_documents")
RAG_TOP_K      = int(os.getenv("RAG_TOP_K",          "8"))

PG_HOST        = os.getenv("POSTGRES_HOST",     "localhost")
PG_PORT        = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB          = os.getenv("POSTGRES_DB",       "airflow")
PG_USER        = os.getenv("POSTGRES_USER",     "airflow")
PG_PASS        = os.getenv("POSTGRES_PASSWORD", "airflow")

_EMBED_MODEL_RAW = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANK_MODEL     = os.getenv("RERANKER_MODEL",  "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Embedding model name resolution ──────────────────────────────────────────────
# .env may use Ollama model names. Map them to HuggingFace sentence-transformers IDs.
_OLLAMA_TO_HF: Dict[str, str] = {
    "nomic-embed-text":          "nomic-ai/nomic-embed-text-v1",
    "nomic-embed-text:latest":   "nomic-ai/nomic-embed-text-v1",
    "mxbai-embed-large":         "mixedbread-ai/mxbai-embed-large-v1",
    "mxbai-embed-large:latest":  "mixedbread-ai/mxbai-embed-large-v1",
    "all-minilm":                "all-MiniLM-L6-v2",
    "all-minilm:latest":         "all-MiniLM-L6-v2",
    "bge-large":                 "BAAI/bge-large-en-v1.5",
    "bge-m3":                    "BAAI/bge-m3",
}
_EMBED_FALLBACK = "all-MiniLM-L6-v2"  # always works, 384-dim, no HF login needed

EMBED_MODEL = _OLLAMA_TO_HF.get(_EMBED_MODEL_RAW, _EMBED_MODEL_RAW)

# ── Lazy-loaded singletons ─────────────────────────────────────────────────────
_embedder:      Any = None
_reranker:      Any = None
_neo4j_driver:  Any = None
_qdrant_client: Any = None
_embed_model_used: str = EMBED_MODEL  # track which model actually loaded


def _neo4j_localhost_uri() -> str:
    m = re.match(r"(bolt://)([^:]+)(:\d+)", NEO4J_URI)
    return f"{m.group(1)}localhost{m.group(3)}" if m else "bolt://localhost:7687"


# ── Getters ────────────────────────────────────────────────────────────────────

def get_embedder():
    """
    Lazy-load SentenceTransformer.
    Tries EMBED_MODEL (HF-resolved from .env) first.
    Falls back to all-MiniLM-L6-v2 if that model fails (e.g. needs HF login,
    wrong dimension, download error).
    """
    global _embedder, _embed_model_used
    if _embedder is not None:
        return _embedder

    from sentence_transformers import SentenceTransformer  # noqa: lazy

    models_to_try = [EMBED_MODEL]
    if EMBED_MODEL != _EMBED_FALLBACK:
        models_to_try.append(_EMBED_FALLBACK)

    for model_name in models_to_try:
        try:
            logger.info(f"[tools] Loading embedder: {model_name}")
            _embedder = SentenceTransformer(model_name, trust_remote_code=True)
            _embed_model_used = model_name
            logger.info(f"[tools] Embedder ready: {model_name}")
            return _embedder
        except Exception as e:
            logger.warning(f"[tools] Embedder load failed ({model_name}): {e}")

    raise RuntimeError(f"[tools] Could not load any embedding model. Tried: {models_to_try}")


def get_reranker():
    """Lazy-load CrossEncoder."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder  # noqa: lazy
        logger.info(f"[tools] Loading reranker: {RERANK_MODEL}")
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def get_neo4j_driver():
    """
    Lazy-import neo4j. Tries Docker URI first, then localhost fallback.
    """
    global _neo4j_driver
    if _neo4j_driver is not None:
        return _neo4j_driver

    try:
        from neo4j import GraphDatabase  # noqa: lazy (pulls pandas)
    except Exception as e:
        logger.warning(f"[tools] neo4j import failed: {e}")
        return None

    uris = [NEO4J_URI]
    lb   = _neo4j_localhost_uri()
    if lb != NEO4J_URI:
        uris.append(lb)

    for uri in uris:
        try:
            drv = GraphDatabase.driver(uri, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with drv.session() as s:
                s.run("RETURN 1")
            logger.info(f"[tools] Neo4j connected via {uri}")
            _neo4j_driver = drv
            return _neo4j_driver
        except Exception as e:
            logger.warning(f"[tools] Neo4j failed ({uri}): {e}")

    logger.warning("[tools] Neo4j unavailable on all URIs")
    return None


def get_qdrant():
    """
    Lazy-import qdrant_client. Tries Docker host first, then localhost fallback.
    """
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    try:
        from qdrant_client import QdrantClient  # noqa: lazy
    except Exception as e:
        logger.warning(f"[tools] qdrant_client import failed: {e}")
        return None

    hosts = [QDRANT_HOST]
    if QDRANT_HOST != "localhost":
        hosts.append("localhost")

    for host in hosts:
        try:
            client = QdrantClient(host=host, port=QDRANT_PORT)
            client.get_collections()
            logger.info(f"[tools] Qdrant connected via {host}:{QDRANT_PORT}")
            _qdrant_client = client
            return _qdrant_client
        except Exception as e:
            logger.warning(f"[tools] Qdrant failed ({host}:{QDRANT_PORT}): {e}")

    logger.warning("[tools] Qdrant unavailable on all hosts")
    return None


# ── PostgreSQL: Sentiment ──────────────────────────────────────────────────────

def fetch_sentiment(ticker: str) -> Dict:
    """Fetch latest sentiment_trends row. Tries PG_HOST then localhost."""
    hosts = [PG_HOST]
    if PG_HOST != "localhost":
        hosts.append("localhost")

    conn = None
    for host in hosts:
        try:
            conn = psycopg2.connect(
                host=host, port=PG_PORT, dbname=PG_DB,
                user=PG_USER, password=PG_PASS,
                connect_timeout=5,
            )
            logger.debug(f"[tools] PostgreSQL connected via {host}")
            break
        except Exception as e:
            logger.warning(f"[tools] PostgreSQL failed ({host}): {e}")

    if conn is None:
        return {}

    try:
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
        trend  = "stable"
        if len(rows) == 2:
            diff = float(latest.get("bullish_pct") or 0) - float(rows[1].get("bullish_pct") or 0)
            if diff > 3:    trend = "improving"
            elif diff < -3: trend = "deteriorating"

        return {
            "bullish_pct": float(latest.get("bullish_pct") or 0),
            "bearish_pct": float(latest.get("bearish_pct") or 0),
            "neutral_pct": float(latest.get("neutral_pct") or 0),
            "trend":       trend,
            "source":      "postgresql:sentiment_trends",
        }
    except Exception as e:
        logger.warning(f"[tools] PostgreSQL query failed: {e}")
        return {}


# ── Neo4j queries ────────────────────────────────────────────────────────────────

def fetch_company_profile(ticker: str) -> Dict:
    driver = get_neo4j_driver()
    if not driver:
        return {}
    try:
        with driver.session() as session:
            result = session.run(
                "MATCH (c:Company {ticker: $ticker}) RETURN properties(c) AS props",
                ticker=ticker,
            )
            record = result.single()
            if not record:
                return {}
            props = dict(record["props"])
        return {
            "name":          props.get("Name")    or props.get("name")    or ticker,
            "sector":        props.get("Sector")  or props.get("sector"),
            "market_cap":    _safe_float(props.get("Highlights_MarketCapitalization")),
            "pe_ratio":      _safe_float(props.get("Valuation_TrailingPE") or props.get("Highlights_PERatio")),
            "profit_margin": _safe_float(props.get("Highlights_ProfitMargin")),
            "description":   props.get("Description") or props.get("description") or "",
        }
    except Exception as e:
        logger.warning(f"[tools] Neo4j company profile failed: {e}")
        return {}


def fetch_graph_facts(ticker: str, keyword: str = "business") -> List[str]:
    """
    Flexible Cypher that works even when graph edges don't exist yet.
    Falls back to a broader company-description search if no relationship edges found.
    """
    driver = get_neo4j_driver()
    if not driver:
        return []
    try:
        with driver.session() as session:
            # Primary: structured relationship traversal
            result = session.run("""
                MATCH (c:Company {ticker: $ticker})-[r]->(n)
                WHERE type(r) IN ['HAS_STRATEGY','FACES_RISK','OFFERS_PRODUCT']
                  AND (
                    toLower(coalesce(n.description,'')) CONTAINS toLower($keyword)
                    OR toLower(coalesce(n.name,''))  CONTAINS toLower($keyword)
                    OR toLower(coalesce(n.text,''))  CONTAINS toLower($keyword)
                  )
                RETURN coalesce(n.name, n.title, '') + ': '
                     + coalesce(n.description, n.text, '') AS text
                LIMIT 12
            """, ticker=ticker, keyword=keyword)
            rows = [r["text"] for r in result if r["text"].strip(" :")]

            # Fallback: if no edge data, return Company description as context
            if not rows:
                fb = session.run(
                    "MATCH (c:Company {ticker: $ticker}) "
                    "RETURN coalesce(c.Description, c.description, '') AS text",
                    ticker=ticker,
                )
                rec = fb.single()
                if rec and rec["text"]:
                    rows = [rec["text"][:800]]

        return rows
    except Exception as e:
        logger.warning(f"[tools] Neo4j graph traversal failed: {e}")
        return []


def neo4j_vector_search(query: str, ticker: str, k: int = 10) -> List[str]:
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
            return [
                f"[{r.get('chunk_id','unknown')}] {r.get('text','')}"
                for r in result if r.get("text")
            ]
    except Exception as e:
        logger.warning(f"[tools] Neo4j vector search failed: {e}")
        return []


# ── Qdrant: News semantic search ───────────────────────────────────────────────

def qdrant_news_search(query: str, ticker: str, k: int = RAG_TOP_K) -> List[str]:
    client = get_qdrant()
    if not client:
        return []
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # noqa: lazy
        embedding = get_embedder().encode(query).tolist()
        hits = client.search(
            collection_name=COLLECTION,
            query_vector=embedding,
            query_filter=Filter(must=[FieldCondition(
                key="ticker_symbol", match=MatchValue(value=ticker),
            )]) if ticker else None,
            limit=k,
            with_payload=True,
        )
        results = []
        for hit in hits:
            p       = hit.payload or {}
            title   = p.get("title", "")
            content = p.get("content") or p.get("text") or ""
            date    = p.get("date") or p.get("published_date") or "unknown"
            results.append(f"[news|{date}] {title}: {content[:400]}")
        return results
    except Exception as e:
        logger.warning(f"[tools] Qdrant news search failed: {e}")
        return []


# ── Hybrid Retrieval ───────────────────────────────────────────────────────────

def hybrid_retrieve(query: str, ticker: str, k_final: int = 5) -> Tuple[List[str], float]:
    """Neo4j vector + graph + Qdrant news → BM25 + Cross-Encoder rerank."""
    q_lower = query.lower()
    keyword = next(
        (kw for kw in ["risk","strategy","revenue","growth","competition","supply","moat"]
         if kw in q_lower), "business"
    )

    vec_docs   = neo4j_vector_search(query, ticker, k=15)
    graph_docs = fetch_graph_facts(ticker, keyword)
    news_docs  = qdrant_news_search(query, ticker, k=5)

    all_docs = list(dict.fromkeys(vec_docs + graph_docs + news_docs))
    if not all_docs:
        return [], 0.0

    bm25_scores = [0.0] * len(all_docs)
    if BM25_AVAILABLE:
        try:
            tokenized   = [d.lower().split() for d in all_docs]
            bm25        = BM25Okapi(tokenized)
            raw         = bm25.get_scores(query.lower().split())
            max_raw     = max(raw) if max(raw) > 0 else 1.0
            bm25_scores = [s / max_raw for s in raw]
        except Exception:
            pass

    reranker  = get_reranker()
    ce_scores = reranker.predict([[query, d] for d in all_docs]).tolist()

    hybrid    = [0.3 * bm25_scores[i] + 0.7 * ce_scores[i] for i in range(len(all_docs))]
    ranked    = sorted(zip(all_docs, hybrid), key=lambda x: x[1], reverse=True)
    top_docs  = [doc for doc, _ in ranked[:k_final]]
    top_score = ranked[0][1] if ranked else 0.0

    logger.info(f"[tools] hybrid_retrieve: {len(all_docs)} candidates → top score {top_score:.3f}")
    return top_docs, top_score


def crag_evaluate(score: float) -> str:
    if score > 0.7:    return "CORRECT"
    elif score >= 0.5: return "AMBIGUOUS"
    else:              return "INCORRECT"


# ── Utilities ──────────────────────────────────────────────────────────────────

def extract_json_from_response(content: str) -> Optional[dict]:
    for pattern in [r"```(?:json)?\s*(\{.*?\})\s*```", r"\{.*\}"]:
        m = re.search(pattern, content, re.DOTALL)
        if m:
            try:    return json.loads(m.group(1) if "```" in pattern else m.group(0))
            except: pass
    try:    return json.loads(content.strip())
    except: pass
    logger.warning("[tools] Could not extract JSON from response")
    return None


def _safe_float(val) -> Optional[float]:
    try:    return float(val) if val is not None else None
    except: return None
