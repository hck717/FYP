#!/usr/bin/env python3
"""
ingest_earnings_calls.py — Extract text from PDF earnings calls and load into Neo4j.

Run directly:
    python ingestion/etl/ingest_earnings_calls.py           # AAPL (default)
    python ingestion/etl/ingest_earnings_calls.py TSLA      # single ticker
    python ingestion/etl/ingest_earnings_calls.py --all     # all tickers

Requires: pip install pypdf requests
"""

from __future__ import annotations

import json
import os
import sys
import re
from pathlib import Path

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
TEXTUAL_DATA_DIR = _REPO_ROOT / "data"
TRACKED_TICKERS = os.getenv("TRACKED_TICKERS", "AAPL,TSLA,NVDA,MSFT,GOOGL").split(",")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "SecureNeo4jPass2025!")

# Determine if we're running inside Docker by checking for /.dockerenv
IN_DOCKER = Path("/.dockerenv").exists()
if IN_DOCKER:
    # Inside Docker container - use host.docker.internal (works with extra_hosts in docker-compose)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
else:
    # Running on host - use localhost
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

OLLAMA_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIMENSION", "768"))

CHUNK_SIZE = 800
OVERLAP = 100

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
import numpy as np
from neo4j import GraphDatabase


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(str(pdf_path))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        full_text = "\n\n".join(text_parts)
        
        # Clean up whitespace
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        full_text = re.sub(r' {2,}', ' ', full_text)
        
        return full_text
    except Exception as e:
        print(f"  [PDF Error] {pdf_path.name}: {e}")
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


def embed_texts(texts: list[str], model: str, base_url: str) -> list[list[float]]:
    """Embed texts using Ollama."""
    if not texts:
        return []
    
    url = f"{base_url.rstrip('/')}/api/embed"
    embeddings = []
    batch_size = 8
    
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        try:
            resp = requests.post(
                url,
                json={"model": model, "input": batch},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            batch_embeddings = data.get("embeddings")
            if batch_embeddings:
                embeddings.extend(batch_embeddings)
            else:
                print(f"  [Ollama] Empty embeddings for batch {start}")
                embeddings.extend([[0.0] * EMBEDDING_DIM] * len(batch))
        except Exception as e:
            print(f"  [Ollama Error] {e}")
            embeddings.extend([[0.0] * EMBEDDING_DIM] * len(batch))
    
    return embeddings


def load_earnings_call_chunks(ticker: str) -> int:
    """Load earnings call PDF text chunks into Neo4j for a ticker."""
    earnings_dir = TEXTUAL_DATA_DIR / ticker / "earning_call"
    
    if not earnings_dir.exists():
        print(f"[{ticker}] No earning_call directory found")
        return 0
    
    pdf_files = list(earnings_dir.glob("*.pdf")) + list(earnings_dir.glob("*.pdf.pdf"))
    
    if not pdf_files:
        print(f"[{ticker}] No PDF files found in {earnings_dir}")
        return 0
    
    print(f"[{ticker}] Found {len(pdf_files)} earnings call PDFs")
    
    all_chunks = []
    
    for pdf_file in pdf_files:
        print(f"  Processing: {pdf_file.name}")
        
        # Extract date from filename
        date_match = re.search(r'(\d{4})[_\-]?(\d{2})[_\-]?(\d{2})', pdf_file.name)
        if date_match:
            filing_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
        else:
            filing_date = None
        
        # Extract text
        text = extract_text_from_pdf(pdf_file)
        if not text.strip():
            print(f"    No text extracted from {pdf_file.name}")
            continue
        
        # Split into chunks
        chunks = split_text(text, CHUNK_SIZE, OVERLAP)
        
        # Create chunk dicts
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{ticker}::earnings_call::{pdf_file.stem[:20]}::{i}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "ticker": ticker,
                "text": chunk_text,
                "section": "earnings_call",
                "filing_date": filing_date,
                "source_file": pdf_file.name,
            })
        
        print(f"    Created {len(chunks)} chunks")
    
    if not all_chunks:
        print(f"[{ticker}] No chunks created")
        return 0
    
    # Embed chunks
    print(f"[{ticker}] Embedding {len(all_chunks)} chunks via Ollama...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts, OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL)
    
    for chunk, emb in zip(all_chunks, embeddings):
        chunk["embedding"] = emb
    
    # Write to Neo4j
    print(f"[{ticker}] Writing to Neo4j...")
    cypher = """
    UNWIND $rows AS row
    MERGE (c:Company {ticker: row.ticker})
    MERGE (chunk:Chunk {chunk_id: row.chunk_id})
      SET chunk.text       = row.text,
          chunk.section    = row.section,
          chunk.filing_date = row.filing_date,
          chunk.ticker     = row.ticker,
          chunk.embedding  = row.embedding,
          chunk.source_file = row.source_file
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
    
    print(f"[{ticker}] Wrote {written} earnings call chunks to Neo4j")
    return written


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest earnings call PDFs into Neo4j")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--all", action="store_true", help="Process all tracked tickers")
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
    print("=" * 60)
    
    total = 0
    for ticker in tickers:
        ticker = ticker.strip()
        n = load_earnings_call_chunks(ticker)
        total += n
    
    print(f"\nTotal chunks written: {total}")


if __name__ == "__main__":
    main()
