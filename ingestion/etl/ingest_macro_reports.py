#!/usr/bin/env python3
"""
ingest_macro_reports.py — Extract text from MACRO PDFs and load into Neo4j.

Source directory:
    data/textual data/MACRO/*.pdf

Incremental behavior:
    - Only ingest files whose source_file is not already present in Neo4j
      for ticker=_MACRO, section=macro_report.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
from pathlib import Path

import requests
from neo4j import GraphDatabase
import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

# Load .env
_REPO_ROOT = Path(__file__).resolve().parents[2]
_env_path = _REPO_ROOT / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

_default_textual_dir = _REPO_ROOT / "data" / "textual data"
TEXTUAL_DATA_DIR = Path(os.getenv("TEXTUAL_DATA_DIR", str(_default_textual_dir)))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "SecureNeo4jPass2025!")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "airflow")
POSTGRES_USER = os.getenv("POSTGRES_USER", "airflow")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")

IN_DOCKER = Path("/.dockerenv").exists()
if IN_DOCKER:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
else:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

OLLAMA_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_MODEL_VERSION = os.getenv("EMBEDDING_MODEL_VERSION", "1.0")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION", "768"))

CHUNK_SIZE = 800
OVERLAP = 100

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("ERROR: pypdf or PyPDF2 required. Install with: pip install pypdf")
        sys.exit(1)


def _content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _derive_source_name(pdf_path: Path) -> str:
    stem = pdf_path.stem
    if stem.endswith(".pdf"):
        stem = stem[:-4]
    stem = stem.strip()

    date_prefix = re.match(r"^(\d{8})[_\-](.+)$", stem)
    if date_prefix:
        date_str = date_prefix.group(1)
        rest = date_prefix.group(2).replace("_", " ")
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return f"{rest} ({formatted_date})"

    return stem.replace("_", " ")


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
                            "textual_pdf",
                        )
                    )
                chunk_rows = list({row[1]: row for row in chunk_rows}.values())
                execute_values(
                    cur,
                    """
                    INSERT INTO text_chunks (ticker, chunk_id, text, section, filing_date, embedding, source)
                    VALUES %s
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        text = EXCLUDED.text,
                        section = EXCLUDED.section,
                        filing_date = EXCLUDED.filing_date,
                        embedding = EXCLUDED.embedding,
                        source = EXCLUDED.source,
                        ingested_at = NOW()
                    """,
                    chunk_rows,
                    template="(%s, %s, %s, %s, %s, %s::vector, %s)",
                )
        return len(chunks)
    finally:
        conn.close()


def _existing_source_files(ticker: str, section: str) -> set[str]:
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
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        text_parts = []
        for page in doc:
            text = page.get_text()
            if isinstance(text, str) and text.strip():
                text_parts.append(text)
        doc.close()
        full_text = "\n\n".join(text_parts)
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)
        full_text = re.sub(r" {2,}", " ", full_text)
        return full_text
    except ImportError:
        pass
    except Exception as e:
        logger.warning("  [pymupdf Error] %s: %s — falling back to pypdf", pdf_path.name, e)

    try:
        reader = PdfReader(str(pdf_path))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if isinstance(text, str) and text.strip():
                text_parts.append(text)
        full_text = "\n\n".join(text_parts)
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)
        full_text = re.sub(r" {2,}", " ", full_text)
        return full_text
    except Exception as e:
        logger.warning("  [PDF Error] %s: %s", pdf_path.name, e)
        return ""


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
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
    url = f"{base_url.rstrip('/')}/api/embed"

    def _call() -> list[list[float]]:
        resp = requests.post(url, json={"model": model, "input": texts}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        batch_embeddings = data.get("embeddings")
        if not batch_embeddings:
            raise RuntimeError(f"Ollama returned empty embeddings for model '{model}'")
        return batch_embeddings

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
    if not texts:
        return []

    embeddings = []
    batch_size = 1
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i, start in enumerate(range(0, len(texts), batch_size), 1):
        batch = texts[start:start + batch_size]
        logger.info(
            "  [Embed] batch %d/%d (chunk %d-%d of %d)",
            i,
            total_batches,
            start + 1,
            min(start + batch_size, len(texts)),
            len(texts),
        )
        try:
            batch_embeddings = _embed_with_retry(batch, model, base_url)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error("  [Ollama Error] batch %d: %s", i, e)
            embeddings.extend([[0.0] * EMBEDDING_DIM] * len(batch))

    return embeddings


def load_macro_report_chunks(only_new: bool = False) -> int:
    macro_dir = TEXTUAL_DATA_DIR / "MACRO"
    ticker = "_MACRO"
    section = "macro_report"

    if not macro_dir.exists():
        logger.info("[MACRO] No MACRO directory found: %s", macro_dir)
        return 0

    pdf_files = sorted({*macro_dir.glob("*.pdf"), *macro_dir.glob("*.pdf.pdf")})

    if not pdf_files:
        logger.info("[MACRO] No PDF files found in %s", macro_dir)
        return 0

    existing_neo4j = _existing_source_files(ticker, section)
    existing_pg = _existing_postgres_doc_files(ticker, "macro_report")
    existing_files = existing_neo4j.intersection(existing_pg)
    before = len(pdf_files)
    pdf_files = [p for p in pdf_files if p.name not in existing_files]
    skipped = before - len(pdf_files)
    if skipped:
        logger.info("[MACRO] Skipping %d already-ingested PDFs (Neo4j+PostgreSQL)", skipped)

    if not pdf_files:
        logger.info("[MACRO] No new macro PDFs to ingest")
        return 0

    logger.info("[MACRO] Found %d macro PDFs to ingest", len(pdf_files))

    all_chunks = []
    all_docs = []
    for pdf_file in pdf_files:
        logger.info("  Processing: %s", pdf_file.name)

        source_name = _derive_source_name(pdf_file)
        date_match = re.search(r"(\d{4})[_\-]?(\d{2})[_\-]?(\d{2})", pdf_file.name)
        filing_date = (
            f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
            if date_match
            else None
        )

        text = extract_text_from_pdf(pdf_file)
        if not text.strip():
            logger.warning("    No text extracted from %s", pdf_file.name)
            continue

        chunks = split_text(text, CHUNK_SIZE, OVERLAP)
        inst = source_name.split(" ", 1)[0] if source_name else ""
        all_docs.append(
            {
                "filename": pdf_file.name,
                "filepath": str(pdf_file),
                "institution": inst,
                "date_approx": filing_date,
                "file_size_bytes": int(pdf_file.stat().st_size),
                "md5_hash": hashlib.md5(pdf_file.read_bytes()).hexdigest(),
            }
        )
        for chunk_text in chunks:
            c_hash = _content_hash(chunk_text)
            chunk_id = f"{ticker}::{section}::{c_hash[:16]}"
            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content_hash": c_hash,
                    "ticker": ticker,
                    "text": chunk_text,
                    "section": section,
                    "filing_date": filing_date,
                    "source_file": pdf_file.name,
                    "source_name": source_name,
                    "embedding_version": EMBEDDING_MODEL_VERSION,
                }
            )

        logger.info("    Created %d chunks", len(chunks))

    if not all_chunks:
        logger.info("[MACRO] No chunks created")
        return 0

    logger.info("[MACRO] Embedding %d chunks via Ollama...", len(all_chunks))
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts, OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL)
    for chunk, emb in zip(all_chunks, embeddings):
        chunk["embedding"] = emb

    logger.info("[MACRO] Writing to Neo4j...")
    cypher = """
    UNWIND $rows AS row
    MERGE (c:Company {ticker: row.ticker})
    MERGE (chunk:Chunk {content_hash: row.content_hash, ticker: row.ticker})
      ON CREATE SET
          chunk.chunk_id          = row.chunk_id,
          chunk.text              = row.text,
          chunk.section           = row.section,
          chunk.filing_date       = row.filing_date,
          chunk.source_file       = row.source_file,
          chunk.source_name       = row.source_name,
          chunk.embedding         = row.embedding,
          chunk.embedding_version = row.embedding_version,
          chunk.created_at        = datetime()
      ON MATCH SET
          chunk.source_name       = row.source_name,
          chunk.embedding         = row.embedding,
          chunk.embedding_version = row.embedding_version,
          chunk.updated_at        = datetime()
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

    logger.info("[MACRO] Wrote %d macro-report chunks to Neo4j", written)
    try:
        pg_written = _upsert_postgres_textual(ticker, "macro_report", all_docs, all_chunks)
        logger.info("[MACRO] Upserted %d macro-report chunks to PostgreSQL", pg_written)
    except Exception as exc:
        logger.error("[MACRO] PostgreSQL upsert failed: %s", exc)
    return written


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Ingest MACRO PDFs into Neo4j")
    parser.add_argument("--only-new", action="store_true", help="Only process new PDFs")
    args = parser.parse_args()

    print("=" * 60)
    print("Macro Report PDF Ingestion")
    print("=" * 60)
    print(f"Path: {TEXTUAL_DATA_DIR / 'MACRO'}")
    print(f"Neo4j: {NEO4J_URI}")
    print(f"Ollama: {OLLAMA_BASE_URL} ({OLLAMA_EMBED_MODEL})")
    print(f"Embedding version: {EMBEDDING_MODEL_VERSION}")
    print("=" * 60)

    total = load_macro_report_chunks(only_new=args.only_new)
    print(f"\nTotal chunks written: {total}")


if __name__ == "__main__":
    main()
