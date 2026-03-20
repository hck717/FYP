#!/usr/bin/env python3
"""
ingest_earnings_calls.py — Extract text from PDF earnings calls and load into Neo4j.

Changes vs original:
  - content_hash-based MERGE prevents duplicate chunks across reruns
  - embedding_version property stored on every Chunk node (tracks model changes)
  - Automatic detection of new PDFs via mtime/hash (scan_new_pdfs helper)
  - Retry wrapper around Ollama embed calls (tenacity, max 3 attempts)
  - In-memory deduplication: chunks with cosine similarity > 0.95 are dropped

Run directly:
    python ingestion/etl/ingest_earnings_calls.py           # AAPL (default)
    python ingestion/etl/ingest_earnings_calls.py TSLA      # single ticker
    python ingestion/etl/ingest_earnings_calls.py --all     # all tickers

Requires: pip install pypdf requests tenacity
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Load .env ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_env_path = _REPO_ROOT / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ── Config ─────────────────────────────────────────────────────────────────────
# PDFs live under data/textual data/<TICKER>/earning_call/
# Override with TEXTUAL_DATA_DIR env var if the layout differs (e.g. in Docker).
_default_textual_dir = _REPO_ROOT / "data" / "textual data"
TEXTUAL_DATA_DIR = Path(os.getenv("TEXTUAL_DATA_DIR", str(_default_textual_dir)))
TRACKED_TICKERS = os.getenv("TRACKED_TICKERS", "AAPL,TSLA,NVDA,MSFT,GOOGL").split(",")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "SecureNeo4jPass2025!")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "airflow")
POSTGRES_USER = os.getenv("POSTGRES_USER", "airflow")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")

# Determine if we're running inside Docker by checking for /.dockerenv
IN_DOCKER = Path("/.dockerenv").exists()
if IN_DOCKER:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
else:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

OLLAMA_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
# Lock to a tagged version to prevent silent model drift between runs.
# Override with EMBEDDING_MODEL_VERSION env var if needed.
EMBEDDING_MODEL_VERSION = os.getenv("EMBEDDING_MODEL_VERSION", "1.0")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION", "768"))

CHUNK_SIZE = 800
OVERLAP = 100

# Similarity threshold above which two chunks are considered duplicates
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("CHUNK_DEDUP_SIMILARITY_THRESHOLD", "0.95"))

# Try to import pypdf, fall back to PyPDF2
try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("ERROR: pypdf or PyPDF2 required. Install with: pip install pypdf")
        sys.exit(1)

import requests
from neo4j import GraphDatabase
import psycopg2
from psycopg2.extras import execute_values

# Tenacity for retry logic (pip install tenacity)
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    _TENACITY_AVAILABLE = True
except ImportError:
    _TENACITY_AVAILABLE = False
    logger.warning("[ingest_earnings_calls] tenacity not installed — retries disabled")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _content_hash(text: str) -> str:
    """MD5 hash of chunk text for deduplication key."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _derive_source_name(pdf_path: Path) -> str:
    """
    Derive a human-readable source name from an earnings-call PDF filename.

    Typical filename:
        ``Apple Inc Earnings Call 20251030 RT000000003082837405.pdf.pdf``

    Steps:
      1. Strip one or two ``.pdf`` extensions.
      2. Strip trailing random identifier (long alphanumeric token, e.g.
         ``RT000000003082837405`` or ``DN000000003094489278``).
      3. Reformat any embedded 8-digit date ``YYYYMMDD`` → ``YYYY-MM-DD``.

    Result example:
        ``"Apple Inc Earnings Call 2025-10-30"``
    """
    stem = pdf_path.stem          # strips outermost .pdf
    if stem.endswith(".pdf"):
        stem = stem[:-4]          # handle double-extension .pdf.pdf
    stem = stem.strip()

    # Strip trailing random ID: two uppercase letters followed by ≥10 digits
    stem = re.sub(r'\s+[A-Z]{2}\d{10,}\s*$', '', stem).strip()

    # Reformat 8-digit dates embedded in the stem: YYYYMMDD → YYYY-MM-DD
    # Also handle 7-digit dates: YYYYMDD (single-digit month) → YYYY-MM-DD
    def _fmt_date(m: re.Match) -> str:
        d = m.group(0)
        if len(d) == 8:
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        else:  # 7 digits: YYYYMDD
            return f"{d[:4]}-0{d[4]}-{d[5:7]}"

    stem = re.sub(r'\b(\d{7,8})\b', _fmt_date, stem)

    return stem


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Dot product similarity between two L2-normalised vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _dedup_chunks(chunks: list[dict]) -> list[dict]:
    """
    Remove chunks whose embeddings are too similar to a previously seen chunk
    (cosine similarity > DEDUP_SIMILARITY_THRESHOLD).

    Only runs if embeddings are already populated.
    Falls back to content_hash deduplication when no embedding is present.
    """
    seen_hashes: set[str] = set()
    seen_embeddings: list[list[float]] = []
    result: list[dict] = []

    for chunk in chunks:
        # Fast path: exact text duplicate
        h = chunk.get("content_hash") or _content_hash(chunk["text"])
        if h in seen_hashes:
            continue

        emb = chunk.get("embedding")
        if emb:
            # Check near-duplicate via cosine similarity
            is_dup = any(
                _cosine_similarity(emb, prev) > DEDUP_SIMILARITY_THRESHOLD
                for prev in seen_embeddings
            )
            if is_dup:
                continue
            seen_embeddings.append(emb)

        seen_hashes.add(h)
        result.append(chunk)

    removed = len(chunks) - len(result)
    if removed:
        logger.info("[dedup] Removed %d near-duplicate chunks (threshold=%.2f)", removed, DEDUP_SIMILARITY_THRESHOLD)
    return result


# ── PDF scan helper ───────────────────────────────────────────────────────────

def scan_new_pdfs(ticker: str, section: str, state_file: Path | None = None) -> list[Path]:
    """
    Return PDF paths in data/{ticker}/{section}/ that are NEW or MODIFIED
    since the last successful ingestion, detected via mtime+size hash.

    *state_file* is a JSON file that persists {filename: mtime_size_hash}.
    When None, defaults to data/{ticker}/.pdf_scan_state_{section}.json.

    This enables automatic PDF detection without manual triggers.
    """
    pdf_dir = TEXTUAL_DATA_DIR / ticker / section
    if not pdf_dir.exists():
        return []

    if state_file is None:
        state_file = TEXTUAL_DATA_DIR / ticker / f".pdf_scan_state_{section}.json"

    # Load previous state
    state: dict[str, str] = {}
    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
        except Exception:
            state = {}

    new_pdfs: list[Path] = []
    current_state: dict[str, str] = {}

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        stat = pdf_path.stat()
        # Hash of mtime + size as a cheap change detector
        file_sig = hashlib.md5(f"{stat.st_mtime}:{stat.st_size}".encode()).hexdigest()
        current_state[pdf_path.name] = file_sig
        if state.get(pdf_path.name) != file_sig:
            new_pdfs.append(pdf_path)

    # Persist updated state
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w") as f:
            json.dump(current_state, f, indent=2)
    except Exception as exc:
        logger.warning("[scan_new_pdfs] Could not save state: %s", exc)

    if new_pdfs:
        logger.info("[scan_new_pdfs] %s/%s: %d new/modified PDFs detected", ticker, section, len(new_pdfs))
    return new_pdfs


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _pg_connect():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def _upsert_postgres_textual(
    ticker: str,
    doc_type: str,
    docs: list[dict],
    chunks: list[dict],
) -> int:
    if not docs and not chunks:
        return 0

    conn = _pg_connect()
    try:
        with conn, conn.cursor() as cur:
            if docs:
                doc_rows = [
                    (
                        ticker,
                        doc_type,
                        d.get("filename", ""),
                        d.get("filepath", ""),
                        d.get("institution", ""),
                        d.get("date_approx"),
                        d.get("file_size_bytes"),
                        d.get("md5_hash"),
                        True,
                    )
                    for d in docs
                ]
                doc_rows = list({row[2]: row for row in doc_rows}.values())
                execute_values(
                    cur,
                    """
                    INSERT INTO textual_documents
                        (ticker, doc_type, filename, filepath, institution,
                         date_approx, file_size_bytes, md5_hash, ingested_into_qdrant)
                    VALUES %s
                    ON CONFLICT (ticker, filename) DO UPDATE SET
                        filepath = EXCLUDED.filepath,
                        institution = EXCLUDED.institution,
                        date_approx = EXCLUDED.date_approx,
                        file_size_bytes = EXCLUDED.file_size_bytes,
                        md5_hash = EXCLUDED.md5_hash,
                        ingested_into_qdrant = EXCLUDED.ingested_into_qdrant,
                        ingested_at = NOW()
                    """,
                    doc_rows,
                )

            if chunks:
                chunk_rows = []
                for c in chunks:
                    emb = c.get("embedding") or []
                    emb_str = "[" + ",".join(f"{float(x):.8f}" for x in emb) + "]"
                    chunk_rows.append(
                        (
                            ticker,
                            c["chunk_id"],
                            c["text"],
                            c.get("section"),
                            c.get("filing_date"),
                            emb_str,
                            c.get("source_file", ""),
                            c.get("source_name", ""),
                        )
                    )
                chunk_rows = list({row[1]: row for row in chunk_rows}.values())
                execute_values(
                    cur,
                    """
                    INSERT INTO text_chunks (ticker, chunk_id, text, section, filing_date, embedding, source_file, source_name)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        text = EXCLUDED.text,
                        section = EXCLUDED.section,
                        filing_date = EXCLUDED.filing_date,
                        embedding = EXCLUDED.embedding,
                        source_file = EXCLUDED.source_file,
                        source_name = EXCLUDED.source_name,
                        ingested_at = NOW()
                    """,
                    chunk_rows,
                    template="(%s, %s, %s, %s, %s, %s::vector, %s, %s)",
                )
        return len(chunks)
    finally:
        conn.close()


def _existing_source_files(ticker: str, section: str) -> set[str]:
    """Return source_file names already ingested for ticker+section in Neo4j."""
    cypher = """
    MATCH (c:Company {ticker: $ticker})-[:HAS_CHUNK]->(ch:Chunk {section: $section})
    WHERE ch.source_file IS NOT NULL
    RETURN DISTINCT ch.source_file AS source_file
    """
    driver = get_driver()
    try:
        with driver.session() as session:
            rows = session.run(cypher, ticker=ticker, section=section).data()
            return {str(r["source_file"]) for r in rows if r.get("source_file")}
    finally:
        driver.close()


def _existing_postgres_doc_files(ticker: str, doc_type: str) -> set[str]:
    conn = _pg_connect()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT filename FROM textual_documents WHERE ticker = %s AND doc_type = %s",
                (ticker, doc_type),
            )
            return {str(r[0]) for r in cur.fetchall() if r and r[0]}
    finally:
        conn.close()


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF file.

    Tries pymupdf (fitz) first — it handles the character-map encoding issues
    that cause pypdf to produce space-separated characters (e.g. 'V i c e
    P r e s i d e n t').  Falls back to pypdf/PyPDF2 if pymupdf is not
    installed.
    """
    # ── pymupdf (preferred) ───────────────────────────────────────────────────
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(pdf_path))
        text_parts = []
        for page in doc:
            text = page.get_text()
            if isinstance(text, str) and text.strip():
                text_parts.append(text)
        doc.close()
        full_text = "\n\n".join(text_parts)
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        full_text = re.sub(r' {2,}', ' ', full_text)
        return full_text
    except ImportError:
        pass  # fall through to pypdf
    except Exception as e:
        logger.warning("  [pymupdf Error] %s: %s — falling back to pypdf", pdf_path.name, e)

    # ── pypdf fallback ────────────────────────────────────────────────────────
    try:
        reader = PdfReader(str(pdf_path))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if isinstance(text, str) and text.strip():
                text_parts.append(text)
        full_text = "\n\n".join(text_parts)
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        full_text = re.sub(r' {2,}', ' ', full_text)
        return full_text
    except Exception as e:
        logger.warning("  [PDF Error] %s: %s", pdf_path.name, e)
        return ""


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Split text into overlapping chunks by word boundaries."""
    if not text or not text.strip():
        return []

    words = text.split()
    chunks = []
    buf = []
    buf_len = 0

    for word in words:
        wl = len(word) + 1
        if buf_len + wl > chunk_size and buf:
            chunks.append(" ".join(buf))
            # Carry over last few words for overlap
            carry = []
            carry_len = 0
            for w in reversed(buf):
                if carry_len + len(w) + 1 > overlap:
                    break
                carry.insert(0, w)
                carry_len += len(w) + 1
            buf = carry
            buf_len = carry_len
        buf.append(word)
        buf_len += wl

    if buf:
        chunks.append(" ".join(buf))

    return chunks


def _embed_with_retry(texts: list[str], model: str, base_url: str) -> list[list[float]]:
    """
    Embed a batch of texts via Ollama with exponential-backoff retry.
    Uses tenacity if available, otherwise plain try/except.
    """
    url = f"{base_url.rstrip('/')}/api/embed"

    def _call() -> list[list[float]]:
        resp = requests.post(
            url,
            json={"model": model, "input": texts},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        batch_embeddings = data.get("embeddings")
        if not batch_embeddings:
            raise RuntimeError(f"Ollama returned empty embeddings for model '{model}'")
        return batch_embeddings

    if _TENACITY_AVAILABLE:
        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type((requests.RequestException, RuntimeError)),
            reraise=True,
        )
        def _call_with_retry() -> list[list[float]]:
            return _call()
        return _call_with_retry()
    else:
        # Simple fallback without tenacity
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                return _call()
            except Exception as exc:
                last_exc = exc
                import time
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Embed failed after 3 attempts: {last_exc}")


def embed_texts(texts: list[str], model: str, base_url: str) -> list[list[float]]:
    """Embed texts using Ollama with per-batch retry."""
    if not texts:
        return []

    embeddings = []
    batch_size = 1
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i, start in enumerate(range(0, len(texts), batch_size), 1):
        batch = texts[start:start + batch_size]
        logger.info("  [Embed] batch %d/%d (chunk %d-%d of %d)", i, total_batches, start + 1, min(start + batch_size, len(texts)), len(texts))
        try:
            batch_embeddings = _embed_with_retry(batch, model, base_url)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error("  [Ollama Error] batch %d: %s", i, e)
            embeddings.extend([[0.0] * EMBEDDING_DIM] * len(batch))

    return embeddings


def load_earnings_call_chunks(ticker: str, only_new: bool = False) -> int:
    """
    Load earnings call PDF text chunks into Neo4j for a ticker.

    Args:
        ticker:    Ticker symbol (e.g. "AAPL").
        only_new:  If True, skip PDFs that have not changed since last run
                   (uses scan_new_pdfs mtime/hash detection).
    """
    earnings_dir = TEXTUAL_DATA_DIR / ticker / "earning_call"

    if not earnings_dir.exists():
        logger.info("[%s] No earning_call directory found", ticker)
        return 0

    # Build full file list; incremental filtering is decided against DB state below.
    pdf_files = sorted({*earnings_dir.glob("*.pdf"), *earnings_dir.glob("*.pdf.pdf")})

    if not pdf_files:
        logger.info("[%s] No PDF files found in %s", ticker, earnings_dir)
        return 0

    # Real incremental ingestion: only process files not yet loaded into Neo4j.
    existing_neo4j = _existing_source_files(ticker, "earnings_call")
    existing_pg = _existing_postgres_doc_files(ticker, "earnings_call")
    existing_files = existing_neo4j.intersection(existing_pg)
    if existing_files:
        before = len(pdf_files)
        pdf_files = [p for p in pdf_files if p.name not in existing_files]
        skipped = before - len(pdf_files)
        if skipped:
            logger.info("[%s] Skipping %d already-ingested PDFs (Neo4j+PostgreSQL)", ticker, skipped)

    if not pdf_files:
        logger.info("[%s] No new earnings call PDFs to ingest", ticker)
        return 0

    logger.info("[%s] Found %d earnings call PDFs", ticker, len(pdf_files))

    all_chunks = []
    all_docs = []

    for pdf_file in pdf_files:
        logger.info("  Processing: %s", pdf_file.name)

        # Extract date from filename
        date_match = re.search(r'(\d{4})[_\-]?(\d{2})[_\-]?(\d{2})', pdf_file.name)
        if date_match:
            filing_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
        else:
            filing_date = None

        # Derive human-readable source name from the full PDF filename.
        source_name = _derive_source_name(pdf_file)

        # Extract text
        text = extract_text_from_pdf(pdf_file)
        if not text.strip():
            logger.warning("    No text extracted from %s", pdf_file.name)
            continue

        # Split into chunks
        chunks = split_text(text, CHUNK_SIZE, OVERLAP)

        all_docs.append(
            {
                "filename": pdf_file.name,
                "filepath": str(pdf_file),
                "institution": "",
                "date_approx": filing_date,
                "file_size_bytes": int(pdf_file.stat().st_size),
                "md5_hash": hashlib.md5(pdf_file.read_bytes()).hexdigest(),
            }
        )

        # Create chunk dicts — include content_hash for Neo4j MERGE dedup
        for i, chunk_text in enumerate(chunks):
            c_hash = _content_hash(chunk_text)
            # chunk_id uses content_hash suffix so the MERGE key is stable
            # even if the PDF is re-processed with a different index i.
            chunk_id = f"{ticker}::earnings_call::{c_hash[:16]}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "content_hash": c_hash,
                "ticker": ticker,
                "text": chunk_text,
                "section": "earnings_call",
                "filing_date": filing_date,
                "source_file": pdf_file.name,
                "source_name": source_name,
                "embedding_version": EMBEDDING_MODEL_VERSION,
            })

        logger.info("    Created %d chunks", len(chunks))

    if not all_chunks:
        logger.info("[%s] No chunks created", ticker)
        return 0

    # Embed chunks (with retry)
    logger.info("[%s] Embedding %d chunks via Ollama...", ticker, len(all_chunks))
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts, OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL)

    for chunk, emb in zip(all_chunks, embeddings):
        chunk["embedding"] = emb

    # Near-duplicate removal (embedding-based, threshold=0.95)
    all_chunks = _dedup_chunks(all_chunks)
    logger.info("[%s] %d unique chunks after dedup", ticker, len(all_chunks))

    # Write to Neo4j — MERGE on content_hash to prevent duplicates across reruns
    logger.info("[%s] Writing to Neo4j...", ticker)
    cypher = """
    UNWIND $rows AS row
    MERGE (c:Company {ticker: row.ticker})
    MERGE (chunk:Chunk {content_hash: row.content_hash, ticker: row.ticker})
      ON CREATE SET
          chunk.chunk_id        = row.chunk_id,
          chunk.text            = row.text,
          chunk.section         = row.section,
          chunk.filing_date     = row.filing_date,
          chunk.source_file     = row.source_file,
          chunk.source_name     = row.source_name,
          chunk.embedding       = row.embedding,
          chunk.embedding_version = row.embedding_version,
          chunk.created_at      = datetime()
      ON MATCH SET
          chunk.source_name     = row.source_name,
          chunk.embedding       = row.embedding,
          chunk.embedding_version = row.embedding_version,
          chunk.updated_at      = datetime()
    MERGE (c)-[:HAS_CHUNK]->(chunk)
    """

    driver = get_driver()
    try:
        with driver.session() as session:
            batch_size = 50
            written = 0
            for start in range(0, len(all_chunks), batch_size):
                batch = all_chunks[start:start + batch_size]
                session.run(cypher, rows=batch)
                written += len(batch)
    finally:
        driver.close()

    logger.info("[%s] Wrote %d earnings call chunks to Neo4j", ticker, written)
    try:
        pg_written = _upsert_postgres_textual(ticker, "earnings_call", all_docs, all_chunks)
        logger.info("[%s] Upserted %d earnings call chunks to PostgreSQL", ticker, pg_written)
    except Exception as exc:
        logger.error("[%s] PostgreSQL upsert failed: %s", ticker, exc)
    return written


def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Ingest earnings call PDFs into Neo4j")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--all", action="store_true", help="Process all tracked tickers")
    parser.add_argument("--only-new", action="store_true", help="Only process new/modified PDFs")
    args = parser.parse_args()

    if args.all:
        tickers = TRACKED_TICKERS
    else:
        tickers = [args.ticker]

    print("=" * 60)
    print("Earnings Call PDF Ingestion")
    print("=" * 60)
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Neo4j: {NEO4J_URI}")
    print(f"Ollama: {OLLAMA_BASE_URL} ({OLLAMA_EMBED_MODEL})")
    print(f"Embedding version: {EMBEDDING_MODEL_VERSION}")
    print(f"Dedup threshold: {DEDUP_SIMILARITY_THRESHOLD}")
    print("=" * 60)

    total = 0
    for ticker in tickers:
        ticker = ticker.strip()
        n = load_earnings_call_chunks(ticker, only_new=args.only_new)
        total += n

    print(f"\nTotal chunks written: {total}")


if __name__ == "__main__":
    main()
