"""
Step 6: Chunking + PGvector vector embeddings + evidence pack retrieval.

Two index-building modes are supported:

1. **build_index (PDF mode)** (original, default for local-PDF runs):
   ``build_index(chunks)`` — embeds chunks in-process with HuggingFace
   ``all-MiniLM-L6-v2`` and loads them into an in-memory FAISS IndexFlatIP.

2. **build_index_neo4j (PGvector mode)** (DB run, preferred when ingestion pipeline has been run):
   ``build_index_neo4j(ticker, transcript_pages, broker_pages)`` — fetches
   pre-computed Ollama embeddings from PostgreSQL text_chunks (dim=768) and loads them into
   the EvidenceIndex.  No local embedding model or FAISS is needed.  Falls back to
   ``build_index`` (PDF mode) automatically if PostgreSQL is unavailable or has no embeddings.

Both modes return the same ``EvidenceIndex`` NamedTuple so all downstream
``retrieve`` / ``retrieve_broker_evidence`` calls are unchanged.

Speed optimisation:
  - retrieve_broker_evidence reuses the already-computed PG vectors stored in
    EvidenceIndex.vectors instead of re-computing broker chunks.

Run:
    python agent_step6_embeddings.py
"""

from __future__ import annotations

import logging
import os
import ast
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_faiss = None
_FAISS_AVAILABLE = False

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
    _HF_AVAILABLE = True
except ImportError:
    _HuggingFaceEmbeddings = None  # type: ignore[assignment,misc]
    _HF_AVAILABLE = False
    logger.warning("[step6] langchain_community.embeddings not available — PDF mode unavailable.")

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

try:
    import psycopg2 as _psycopg2
    _PSYCOPG_AVAILABLE = True
except ImportError:
    _psycopg2 = None  # type: ignore[assignment]
    _PSYCOPG_AVAILABLE = False

PG_HOST     = os.getenv("POSTGRES_HOST",     "postgres")
PG_PORT     = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB       = os.getenv("POSTGRES_DB",       "airflow")
PG_USER     = os.getenv("POSTGRES_USER",     "airflow")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")

OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("EMBEDDING_MODEL",   "nomic-embed-text")

from agents.stock_research_agent.agent_step1_load import list_stock_files, load_pdf_pages
from agents.stock_research_agent.agent_step3_parse_quality import (
    flag_quality_issues, tag_transcript_sections,
    tag_broker_sections, filter_usable,
    _normalize_spaced_text,
)

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


class EvidenceIndex(NamedTuple):
    faiss_index: object
    model:       Optional[object]
    chunks:      list[Document]
    vectors:     np.ndarray


def _pg_connect():
    return _psycopg2.connect(  # type: ignore[operator]
        host=PG_HOST, port=PG_PORT,
        dbname=PG_DB, user=PG_USER, password=PG_PASSWORD,
    )


def _coerce_embedding(embedding: object) -> list[float]:
    """Normalize pgvector payload into a float list."""
    if embedding is None:
        return []
    if isinstance(embedding, (list, tuple)):
        return [float(x) for x in embedding]
    if isinstance(embedding, np.ndarray):
        return embedding.astype("float32").tolist()
    if isinstance(embedding, str):
        s = embedding.strip()
        if not s:
            return []
        # pgvector text format: "[0.1,0.2,...]"
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return []
            return [float(x) for x in inner.split(",")]
        # fallback if driver returns python-list repr
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [float(x) for x in parsed]
        except Exception:
            pass
    raise ValueError(f"Unsupported embedding payload type: {type(embedding)!r}")


def build_index(chunks: list[Document]) -> EvidenceIndex:
    global _faiss, _FAISS_AVAILABLE
    if not _FAISS_AVAILABLE:
        try:
            import faiss as _faiss_mod
            _faiss = _faiss_mod
            _FAISS_AVAILABLE = True
        except ImportError as exc:
            raise ImportError(
                "PDF mode requires `faiss` to build local index. "
                "Use PG mode (build_index_neo4j) or install faiss."
            ) from exc

    if not _HF_AVAILABLE:
        raise ImportError(
            "PDF mode requires `langchain_community` embeddings package. "
            "Use PG mode (build_index_neo4j) or install langchain_community."
        )
    print(f"  Loading embedding model: {EMBED_MODEL} ...")
    model  = _HuggingFaceEmbeddings(model_name=EMBED_MODEL)  # type: ignore[operator]
    texts  = [c.page_content for c in chunks]
    print(f"  Embedding {len(texts)} chunks ...")
    vecs   = np.array(model.embed_documents(texts), dtype="float32")

    norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms  = np.where(norms == 0, 1.0, norms)
    vecs  /= norms

    dim    = vecs.shape[1]
    index  = _faiss.IndexFlatIP(dim)  # type: ignore[operator]
    index.add(vecs)  # type: ignore[operator]
    print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")

    return EvidenceIndex(faiss_index=index, model=model, chunks=chunks, vectors=vecs)


def _enrich_chunks_from_textual_documents(
    ticker: str,
    chunk_rows: list[dict],
) -> dict[str, dict]:
    """
    Enrich chunk rows with doc_name, institution, doc_type from textual_documents.

    Joins text_chunks (by source_file) with textual_documents (by filename).
    Falls back to deriving doc_name from source_name if available, or chunk_id.
    """
    if not chunk_rows:
        return {}

    conn = _pg_connect()
    enrichment: dict[str, dict] = {}
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT td.filename, td.institution, td.doc_type
                FROM textual_documents td
                WHERE td.ticker = %s
                  AND td.doc_type IN ('earnings_call', 'broker_report')
                """,
                (ticker,),
            )
            rows = cur.fetchall()
        for row in rows:
            fname, institution, doc_type = row
            if fname:
                doc_name = fname
                if doc_name.lower().endswith(".pdf"):
                    doc_name = doc_name[:-4]
            else:
                doc_name = ""
            enrichment[fname or ""] = {
                "doc_name":    doc_name,
                "institution": institution or "",
                "doc_type":   doc_type or "",
            }
    finally:
        conn.close()
    return enrichment


def build_index_neo4j(
    ticker: str,
    transcript_pages: list[Document],
    broker_pages: list[Document],
) -> EvidenceIndex:
    """
    Build an EvidenceIndex using pre-computed embeddings fetched from PostgreSQL.

    Fetches all chunks for the ticker from the text_chunks table (via pgvector
    HNSW index), enriches them with doc_name/institution from textual_documents,
    and returns an EvidenceIndex with all vectors loaded for in-memory retrieval.

    Falls back to ``build_index`` (FAISS mode with local HuggingFace model) if
    PostgreSQL is unavailable or no embedded chunks are found.
    """
    if not _PSYCOPG_AVAILABLE:
        logger.warning("[step6] psycopg2 not available — falling back to FAISS mode.")
        all_pages = transcript_pages + broker_pages
        chunks    = chunk_documents(all_pages)
        return build_index(chunks)

    logger.info("[step6] Building index from PostgreSQL/pgvector for ticker=%s", ticker)

    conn = _pg_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, text, section, filing_date, embedding, source_file, source_name
                FROM text_chunks
                WHERE ticker = %s
                  AND embedding IS NOT NULL
                  AND section IN ('earnings_call', 'broker_report')
                ORDER BY chunk_id
                """,
                (ticker,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        logger.warning(
            "[step6] No embedded chunks found in PostgreSQL for ticker=%s — "
            "falling back to FAISS mode.", ticker,
        )
        all_pages = transcript_pages + broker_pages
        chunks    = chunk_documents(all_pages)
        return build_index(chunks)

    chunk_rows = []
    emb_list: list[list[float]] = []
    expected_dim: int | None = None
    skipped_bad_dim = 0
    for row in rows:
        chunk_id, text, section, filing_date, embedding, source_file, source_name = row
        if not text or not text.strip():
            continue
        emb = _coerce_embedding(embedding)
        if not emb:
            continue
        if expected_dim is None:
            expected_dim = len(emb)
        if len(emb) != expected_dim:
            skipped_bad_dim += 1
            continue
        chunk_rows.append({
            "chunk_id":     chunk_id,
            "text":         text,
            "section":      section or "",
            "filing_date":  filing_date or "",
            "source_file":  source_file or "",
            "source_name":  source_name or "",
        })
        emb_list.append(emb)

    if not chunk_rows:
        logger.warning("[step6] All PostgreSQL chunks were empty — falling back to FAISS mode.")
        all_pages = transcript_pages + broker_pages
        return build_index(chunk_documents(all_pages))

    if skipped_bad_dim:
        logger.warning(
            "[step6] Skipped %d chunks due to embedding dimension mismatch",
            skipped_bad_dim,
        )

    enrichment = _enrich_chunks_from_textual_documents(ticker, chunk_rows)

    transcript_map: dict[str, Document] = {p.metadata.get("chunk_id", ""): p for p in transcript_pages}
    broker_map:     dict[str, Document] = {p.metadata.get("chunk_id", ""): p for p in broker_pages}

    latest_ids: set[str] = {p.metadata.get("chunk_id", "") for p in transcript_pages if p.metadata.get("period") == "latest"}
    prev_ids:   set[str] = {p.metadata.get("chunk_id", "") for p in transcript_pages if p.metadata.get("period") == "previous"}

    docs: list[Document] = []
    for row in chunk_rows:
        cid  = row["chunk_id"]
        text = row["text"]
        section = row["section"]

        pg_doc = transcript_map.get(cid) or broker_map.get(cid)

        if pg_doc:
            base_meta = dict(pg_doc.metadata)
        else:
            doc_type = "transcript" if section == "earnings_call" else "broker"
            source_file = row.get("source_file") or ""
            enrich = enrichment.get(source_file, {})

            if section == "earnings_call":
                period = "latest" if cid in latest_ids else ("previous" if cid in prev_ids else "")
            else:
                period = ""

            base_meta = {
                "ticker":      ticker,
                "doc_type":    enrich.get("doc_type") or doc_type,
                "doc_name":    enrich.get("doc_name") or source_file or cid,
                "period":      period,
                "page_number": 0,
                "section":     section,
                "filing_date": row["filing_date"],
                "institution": enrich.get("institution") or "",
                "chunk_id":    cid,
            }

        docs.append(Document(page_content=text, metadata=base_meta))

    if not emb_list:
        logger.warning("[step6] No valid embeddings after normalization — falling back to PDF mode.")
        all_pages = transcript_pages + broker_pages
        return build_index(chunk_documents(all_pages))

    vecs = np.array(emb_list, dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vecs /= norms

    dummy_index = None
    logger.info("[step6] PGvector index built: %d vectors, dim=%d", len(docs), vecs.shape[1])

    return EvidenceIndex(faiss_index=dummy_index, model=None, chunks=docs, vectors=vecs)


def _embed_query_ollama(query: str) -> np.ndarray:
    """Embed a single query string using Ollama (nomic-embed-text)."""
    import requests
    embed_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    resp = requests.post(
        f"{embed_url.rstrip('/')}/api/embed",
        json={"model": embed_model, "input": [query]},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    embeddings = data.get("embeddings")
    if not embeddings:
        raise RuntimeError("Ollama returned empty embeddings for query")
    return np.array(embeddings[0], dtype="float32")


def _embed_query(evidence: EvidenceIndex, query: str) -> np.ndarray:
    """Embed a query using whichever method is available."""
    if evidence.model is not None:
        return np.array(evidence.model.embed_query(query), dtype="float32")  # type: ignore[union-attr]
    else:
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

    fetch_k = top_k * 5 if (filter_doc_type or filter_section) else top_k
    fetch_k = min(fetch_k, len(evidence.chunks))

    scores = evidence.vectors @ q_vec
    top_indices = np.argsort(scores)[::-1][:fetch_k]

    results: list[Document] = []
    for idx in top_indices:
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
            metadata={**meta, "retrieval_score": round(float(scores[idx]), 4)},
        ))
        if len(results) >= top_k:
            break

    return results


def retrieve_broker_evidence(
    evidence: EvidenceIndex,
    top_k_per_doc: int = BROKER_TOP_K_PER_DOC,
    queries: list[str] | None = None,
) -> list[Document]:
    if queries is None:
        queries = BROKER_THESIS_QUERIES

    q_vecs = []
    for q in queries:
        v = _embed_query(evidence, q)
        v /= max(np.linalg.norm(v), 1e-9)
        q_vecs.append(v)

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
        doc_vecs = evidence.vectors[chunk_idxs]

        scores = np.zeros(len(chunk_idxs))
        for q_vec in q_vecs:
            scores = np.maximum(scores, doc_vecs @ q_vec)

        top_local_idxs = np.argsort(scores)[::-1][:top_k_per_doc]
        for local_i in top_local_idxs:
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
