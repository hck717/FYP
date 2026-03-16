"""
Step 6: Chunking + FAISS vector embeddings + evidence pack retrieval.

Two index-building modes are supported:

1. **FAISS mode** (original, default for local-PDF runs):
   ``build_index(chunks)`` — embeds chunks in-process with HuggingFace
   ``all-MiniLM-L6-v2`` and loads them into a FAISS IndexFlatIP.

2. **Neo4j mode** (DB run, preferred when ingestion pipeline has been run):
   ``build_index_neo4j(ticker, transcript_pages, broker_pages)`` — fetches
   pre-computed Ollama embeddings from Neo4j (dim=768) and loads them into
   the same FAISS structure.  No local embedding model is needed.  Falls back
   to FAISS mode automatically if Neo4j is unavailable or has no embeddings.

Both modes return the same ``EvidenceIndex`` NamedTuple so all downstream
``retrieve`` / ``retrieve_broker_evidence`` calls are unchanged.

Speed optimisation:
  - retrieve_broker_evidence reuses the already-computed FAISS vectors stored in
    EvidenceIndex.vectors instead of re-computing broker chunks.

Run:
    python agent_step6_embeddings.py
"""

from __future__ import annotations

import logging
import os
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Try importing FAISS — only required for FAISS mode
try:
    import faiss as _faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _faiss = None  # type: ignore[assignment]
    _FAISS_AVAILABLE = False
    logger.warning("[step6] faiss not installed — FAISS mode unavailable; Neo4j mode will be used.")

# Try importing HuggingFace embeddings — only required for FAISS mode
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
    _HF_AVAILABLE = True
except ImportError:
    _HuggingFaceEmbeddings = None  # type: ignore[assignment,misc]
    _HF_AVAILABLE = False
    logger.warning("[step6] langchain_community.embeddings not available — FAISS mode unavailable.")

# ── Neo4j config (mirrors agent_step1_neo4j.py) ───────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_env_path = _REPO_ROOT / ".env"
if _env_path.exists():
    try:
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())
    except Exception:
        pass

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "SecureNeo4jPass2025!")

from agent_step1_load import list_stock_files, load_pdf_pages
from agent_step3_parse_quality import (
    flag_quality_issues, tag_transcript_sections,
    tag_broker_sections, filter_usable,
    _normalize_spaced_text,
)

# ── Config ────────────────────────────────────────────────────────────────────
TICKER               = "AAPL"
BASE_DIR             = Path("data_reports")
CHUNK_SIZE           = 500
CHUNK_OVERLAP        = 100
EMBED_MODEL          = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K                = 4
BROKER_TOP_K_PER_DOC = 1

BROKER_THESIS_QUERIES = [
    "investment thesis price target valuation",
    "EPS estimate earnings forecast",
    "key risks downside bear case",
    "catalysts upside bull case",
]

# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_documents(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    for page in pages:
        normalized_text = _normalize_spaced_text(page.page_content)
        splits = splitter.split_text(normalized_text)
        for i, text in enumerate(splits):
            if text.strip():
                chunks.append(Document(
                    page_content=text,
                    metadata={**page.metadata, "chunk_index": i},
                ))
    return chunks


# ── Embedding + FAISS index ───────────────────────────────────────────────────

class EvidenceIndex(NamedTuple):
    """FAISS index + model (optional) + chunks + pre-normalised vectors."""
    faiss_index: object               # faiss.IndexFlatIP
    model:       Optional[object]     # HuggingFaceEmbeddings, or None in Neo4j mode
    chunks:      list[Document]
    vectors:     np.ndarray           # shape (N, dim), L2-normalised


def build_index(chunks: list[Document]) -> EvidenceIndex:
    if not _FAISS_AVAILABLE or not _HF_AVAILABLE:
        raise ImportError(
            "FAISS mode requires both `faiss` and `langchain_community` packages. "
            "Install them or use build_index_neo4j() instead."
        )
    print(f"  Loading embedding model: {EMBED_MODEL} ...")
    model  = _HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    texts  = [c.page_content for c in chunks]
    print(f"  Embedding {len(texts)} chunks ...")
    vecs   = np.array(model.embed_documents(texts), dtype="float32")

    norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms  = np.where(norms == 0, 1.0, norms)
    vecs  /= norms                   # L2-normalise in place

    dim    = vecs.shape[1]
    index  = _faiss.IndexFlatIP(dim)
    index.add(vecs)
    print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")

    return EvidenceIndex(faiss_index=index, model=model, chunks=chunks, vectors=vecs)


def build_index_neo4j(
    ticker: str,
    transcript_pages: list[Document],
    broker_pages: list[Document],
) -> EvidenceIndex:
    """
    Build an EvidenceIndex using pre-computed embeddings fetched from Neo4j.

    Chunks that have no embedding in Neo4j (embedding IS NULL) are silently
    skipped.  If no embeddings are available at all, falls back to
    ``build_index`` (FAISS mode with local HuggingFace model).

    Parameters
    ----------
    ticker:
        Ticker symbol, used to query Neo4j.
    transcript_pages:
        Document list from agent_step1_neo4j.load_neo4j_pages (transcript
        docs); used to reconstruct the Document list in the same order as
        the embeddings.
    broker_pages:
        Document list from agent_step1_neo4j.load_neo4j_pages (broker docs).

    Returns
    -------
    EvidenceIndex
        Compatible with all existing retrieve / retrieve_broker_evidence calls.
        ``model`` is None (no local embedding model is loaded).
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.warning("[step6] neo4j driver not available — falling back to FAISS mode.")
        all_pages = transcript_pages + broker_pages
        chunks    = chunk_documents(all_pages)
        return build_index(chunks)

    logger.info("[step6] Building index from Neo4j embeddings for ticker=%s", ticker)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Company {ticker: $ticker})-[:HAS_CHUNK]->(ch:Chunk)
                WHERE ch.embedding IS NOT NULL
                RETURN ch.chunk_id   AS chunk_id,
                       ch.text       AS text,
                       ch.section    AS section,
                       ch.source_name AS source_name,
                       ch.filing_date AS filing_date,
                       ch.institution AS institution,
                       ch.embedding   AS embedding
                ORDER BY ch.chunk_id
                """,
                ticker=ticker,
            )
            rows = [dict(r) for r in result]
    finally:
        driver.close()

    if not rows:
        logger.warning(
            "[step6] No embedded chunks found in Neo4j for ticker=%s — "
            "falling back to FAISS mode.", ticker,
        )
        all_pages = transcript_pages + broker_pages
        chunks    = chunk_documents(all_pages)
        return build_index(chunks)

    # Build Documents + embedding matrix from Neo4j rows
    chunks: list[Document] = []
    emb_list: list[list[float]] = []

    for i, row in enumerate(rows):
        text = row.get("text") or ""
        if not text.strip():
            continue
        # Map Neo4j section → doc_type used by downstream filters
        section = row.get("section") or ""
        doc_type = "transcript" if section == "earnings_call" else "broker"

        chunks.append(Document(
            page_content=text,
            metadata={
                "ticker":       ticker,
                "doc_type":     doc_type,
                "doc_name":     row.get("source_name") or row.get("chunk_id", "unknown"),
                "period":       "",          # period not stored in Neo4j; patched below
                "page_number":  i + 1,
                "section":      section,
                "filing_date":  row.get("filing_date", ""),
                "institution":  row.get("institution", ""),
                "chunk_id":     row.get("chunk_id", ""),
            },
        ))
        emb_list.append(row["embedding"])

    if not chunks:
        logger.warning("[step6] All Neo4j chunks were empty — falling back to FAISS mode.")
        all_pages = transcript_pages + broker_pages
        return build_index(chunk_documents(all_pages))

    # Patch "period" metadata using the transcript_pages ordering:
    # latest pages come first in transcript_pages, so chunks whose
    # chunk_id appears in transcript_pages[0..n_latest-1] get "latest".
    _latest_ids: set[str] = {
        p.metadata.get("chunk_id", "") for p in transcript_pages
        if p.metadata.get("period") == "latest"
    }
    _prev_ids: set[str] = {
        p.metadata.get("chunk_id", "") for p in transcript_pages
        if p.metadata.get("period") == "previous"
    }
    for chunk in chunks:
        cid = chunk.metadata.get("chunk_id", "")
        if cid in _latest_ids:
            chunk.metadata["period"] = "latest"
        elif cid in _prev_ids:
            chunk.metadata["period"] = "previous"

    vecs = np.array(emb_list, dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vecs /= norms

    if not _FAISS_AVAILABLE:
        raise ImportError(
            "faiss is required for build_index_neo4j. "
            "Install it with: pip install faiss-cpu"
        )

    dim   = vecs.shape[1]
    index = _faiss.IndexFlatIP(dim)
    index.add(vecs)
    logger.info("[step6] Neo4j index built: %d vectors, dim=%d", index.ntotal, dim)

    return EvidenceIndex(faiss_index=index, model=None, chunks=chunks, vectors=vecs)


# ── Ollama query embedder (used by retrieve in Neo4j mode) ────────────────────

def _embed_query_ollama(query: str) -> np.ndarray:
    """Embed a single query string using Ollama (nomic-embed-text)."""
    import requests
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    resp = requests.post(
        f"{ollama_url.rstrip('/')}/api/embed",
        json={"model": embed_model, "input": [query]},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    embeddings = data.get("embeddings")
    if not embeddings:
        raise RuntimeError("Ollama returned empty embeddings for query")
    return np.array(embeddings[0], dtype="float32")


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _embed_query(evidence: EvidenceIndex, query: str) -> np.ndarray:
    """Embed a query using whichever method is available."""
    if evidence.model is not None:
        # FAISS mode — use HuggingFace model
        return np.array(evidence.model.embed_query(query), dtype="float32")
    else:
        # Neo4j mode — use Ollama
        return _embed_query_ollama(query)


def retrieve(
    evidence: EvidenceIndex,
    query: str,
    top_k: int = TOP_K,
    filter_doc_type: str | None = None,
    filter_section:  str | None = None,
) -> list[Document]:
    q_vec = _embed_query(evidence, query)
    q_vec /= max(np.linalg.norm(q_vec), 1e-9)
    q_vec  = q_vec.reshape(1, -1)

    fetch_k = top_k * 5 if (filter_doc_type or filter_section) else top_k
    fetch_k = min(fetch_k, evidence.faiss_index.ntotal)

    scores, indices = evidence.faiss_index.search(q_vec, fetch_k)
    results: list[Document] = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = evidence.chunks[idx]
        meta  = chunk.metadata
        if filter_doc_type and meta.get("doc_type") != filter_doc_type:
            continue
        if filter_section and meta.get("section") != filter_section:
            continue
        results.append(Document(
            page_content=chunk.page_content,
            metadata={**meta, "retrieval_score": round(float(score), 4)},
        ))
        if len(results) >= top_k:
            break

    return results


def retrieve_broker_evidence(
    evidence: EvidenceIndex,
    top_k_per_doc: int = BROKER_TOP_K_PER_DOC,
    queries: list[str] | None = None,
) -> list[Document]:
    """
    Reuse the pre-normalised vectors stored in EvidenceIndex.vectors
    instead of re-embedding broker chunks from scratch.
    Query embedding uses Ollama (Neo4j mode) or HuggingFace (FAISS mode).
    """
    if queries is None:
        queries = BROKER_THESIS_QUERIES

    # Embed queries
    q_vecs = []
    for q in queries:
        v = _embed_query(evidence, q)
        v /= max(np.linalg.norm(v), 1e-9)
        q_vecs.append(v)

    # Build per-doc list of chunk indices (broker docs only)
    doc_chunk_indices: dict[str, list[int]] = defaultdict(list)
    for i, chunk in enumerate(evidence.chunks):
        if chunk.metadata.get("doc_type") == "broker":
            doc_chunk_indices[chunk.metadata.get("doc_name", "")].append(i)

    if not doc_chunk_indices:
        return []

    selected: list[Document] = []

    for doc_name, chunk_idxs in sorted(doc_chunk_indices.items()):
        if not chunk_idxs:
            continue
        # Reuse already-computed normalised vectors
        doc_vecs = evidence.vectors[chunk_idxs]          # shape (k, dim)

        # Max-score across all queries
        scores = np.zeros(len(chunk_idxs))
        for q_vec in q_vecs:
            scores = np.maximum(scores, doc_vecs @ q_vec)

        top_idxs = np.argsort(scores)[::-1][:top_k_per_doc]
        for local_i in top_idxs:
            global_i = chunk_idxs[local_i]
            chunk    = evidence.chunks[global_i]
            selected.append(Document(
                page_content=chunk.page_content,
                metadata={**chunk.metadata, "retrieval_score": round(float(scores[local_i]), 4)},
            ))

    return selected


def format_evidence_pack(chunks: list[Document]) -> str:
    parts = []
    for c in chunks:
        m     = c.metadata
        label = f"[{m.get('doc_name', 'unknown')} p.{m.get('page_number', '?')}]"
        parts.append(f"{label}\n{c.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"=== Steps 1-3: Load + quality + section tag for {TICKER} ===\n")
    broker_pdfs, transcript_pdfs = list_stock_files(BASE_DIR, TICKER)
    latest_t   = transcript_pdfs[0]
    previous_t = transcript_pdfs[1]

    transcript_pages  = []
    transcript_pages += load_pdf_pages([latest_t],   TICKER, "transcript", "latest")
    transcript_pages += load_pdf_pages([previous_t], TICKER, "transcript", "previous")
    broker_pages       = load_pdf_pages(broker_pdfs, TICKER, "broker")

    flag_quality_issues(transcript_pages)
    flag_quality_issues(broker_pages)
    tag_transcript_sections(transcript_pages)
    tag_broker_sections(broker_pages)

    usable_transcript = filter_usable(transcript_pages)
    usable_broker     = filter_usable(broker_pages)
    all_pages         = usable_transcript + usable_broker

    chunks   = chunk_documents(all_pages)
    print(f"  {len(all_pages)} pages → {len(chunks)} chunks")

    evidence = build_index(chunks)

    for q in ["revenue guidance for next quarter", "broker price target and rating"]:
        print(f"\n--- Query: '{q}' ---")
        for r in retrieve(evidence, q, top_k=3):
            m = r.metadata
            print(f"  [{m.get('doc_name','?')} p.{m.get('page_number','?')} score={m.get('retrieval_score','?')}]")
