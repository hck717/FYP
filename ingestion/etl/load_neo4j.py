"""
load_neo4j.py  —  ETL loader: agent_data CSV/JSON  →  Neo4j
============================================================

Called by the Airflow DAG after each ticker's scrape task.
Also callable directly:

    python load_neo4j.py            # loads AAPL (default)
    python load_neo4j.py TSLA       # loads one ticker
    python load_neo4j.py --all      # loads all 5 tickers

Key design decisions
--------------------
* All writes use MERGE (idempotent — safe to re-run).
* numpy scalar types (np.int64, np.float64, etc.) are coerced to Python
  native types before passing to Neo4j; NaN / Inf floats become None.
* company_profile data is stored as a :Company node with all scalar
  properties from the EODHD fundamentals response, including extended
  Highlights, Valuation, SharesStats, and SplitsDividends fields.
* etf_index_constituents creates :Company-[:CONTAINS]->:Company edges.
* All other data types with a storage_destination of "neo4j" are stored
  as generic :DataRecord nodes linked to the :Company node.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from neo4j import GraphDatabase

# ── Path resolution ───────────────────────────────────────────────────────────
# In the Airflow container __file__ is /opt/airflow/etl/load_neo4j.py,
# so parent == /opt/airflow/etl and agent_data lives at parent / "agent_data".
# For local dev, __file__ is .../ingestion/etl/load_neo4j.py — same layout.
_THIS_ETL_DIR    = Path(__file__).resolve().parent          # .../etl/
_DEFAULT_ETL_DIR = _THIS_ETL_DIR / "agent_data"
BASE_ETL_DIR = Path(os.getenv("BASE_ETL_DIR", str(_DEFAULT_ETL_DIR)))

# ── Neo4j connection ──────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://neo4j:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "SecureNeo4jPass2025!")


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ── Schema / index management ─────────────────────────────────────────────────

CHUNK_INDEX_NAME      = os.getenv("NEO4J_CHUNK_INDEX", "chunk_embedding")
# Use EMBEDDING_DIMENSION env var (defaults to 768 for nomic-embed-text via Ollama).
# Legacy NEO4J_EMBEDDING_DIMENSION is checked first for backward compat.
CHUNK_EMBED_DIM       = int(os.getenv("NEO4J_EMBEDDING_DIMENSION",
                                       os.getenv("EMBEDDING_DIMENSION", "768")))
CHUNK_EMBED_SIMILARITY = "cosine"

# Ollama endpoint for embeddings
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


def ensure_neo4j_schema() -> None:
    """
    Create the chunk_embedding vector index and a unique constraint on Chunk
    nodes if they do not already exist.  Safe to call on every startup (all
    operations use IF NOT EXISTS).
    """
    driver = get_driver()
    try:
        with driver.session() as session:
            # Unique constraint on Chunk.chunk_id
            session.run(
                "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
                "FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE"
            )

            # Vector index for semantic search
            session.run(
                f"""
                CREATE VECTOR INDEX {CHUNK_INDEX_NAME} IF NOT EXISTS
                FOR (ch:Chunk) ON ch.embedding
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {CHUNK_EMBED_DIM},
                        `vector.similarity_function`: '{CHUNK_EMBED_SIMILARITY}'
                    }}
                }}
                """
            )
        print(
            f"[Neo4j Schema] vector index '{CHUNK_INDEX_NAME}' "
            f"(dim={CHUNK_EMBED_DIM}, similarity={CHUNK_EMBED_SIMILARITY}) ensured."
        )
    except Exception as exc:
        print(f"[Neo4j Schema] WARNING: could not create index/constraint: {exc}")
    finally:
        driver.close()


# ── Text chunking helper ──────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """
    Split *text* into overlapping word-boundary chunks of at most *chunk_size*
    characters with *overlap* characters of context carry-over.
    """
    if not text or not text.strip():
        return []
    words = text.split()
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for word in words:
        wl = len(word) + 1  # +1 for the space
        if buf_len + wl > chunk_size and buf:
            chunks.append(" ".join(buf))
            # Keep last `overlap` characters worth of words as context
            carry: list[str] = []
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


# ── Company-profile → Chunk ingestion ────────────────────────────────────────

def _build_profile_chunks(
    ticker_symbol: str,
    profile: dict,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[dict]:
    """
    Extract human-readable text segments from a company_profile JSON dict,
    split them into overlapping chunks, and return a list of chunk dicts ready
    for Neo4j ingestion.

    Covered sections: General (description, officers), Highlights (key metrics),
    Valuation, ESGScores, AnalystRatings, and financial_news headlines (if present
    in the same ticker folder).
    """
    from datetime import datetime as _dt

    now_str = _dt.utcnow().strftime("%Y-%m-%d")
    chunks: list[dict] = []

    def _add_chunks(text: str, section: str) -> None:
        for i, seg in enumerate(_split_text(text, chunk_size, overlap)):
            chunks.append(
                {
                    "chunk_id": f"{ticker_symbol}::{section}::{i}",
                    "text": seg,
                    "section": section,
                    "filing_date": now_str,
                    "ticker": ticker_symbol,
                    "embedding": None,  # filled later
                }
            )

    # ── General description ──────────────────────────────────────────────────
    gen = profile.get("General", {})
    desc = str(gen.get("Description") or "").strip()
    if desc:
        _add_chunks(desc, "description")

    # ── Officers summary ─────────────────────────────────────────────────────
    officers = gen.get("Officers") or {}
    if isinstance(officers, dict):
        officer_lines = [
            f"{v.get('Name','?')} – {v.get('Title','?')}"
            for v in officers.values()
            if isinstance(v, dict)
        ]
        if officer_lines:
            _add_chunks(
                f"{gen.get('Name', ticker_symbol)} leadership: "
                + "; ".join(officer_lines),
                "officers",
            )

    # ── Highlights ───────────────────────────────────────────────────────────
    hl = profile.get("Highlights", {})
    if hl:
        hl_text = (
            f"{gen.get('Name', ticker_symbol)} ({ticker_symbol}) financial highlights: "
            + " | ".join(
                f"{k}={v}"
                for k, v in hl.items()
                if v not in (None, "", "None", "0", 0)
            )
        )
        _add_chunks(hl_text, "highlights")

    # ── Valuation ────────────────────────────────────────────────────────────
    val = profile.get("Valuation", {})
    if val:
        val_text = (
            f"{ticker_symbol} valuation metrics: "
            + " | ".join(
                f"{k}={v}"
                for k, v in val.items()
                if v not in (None, "", "None", "0", 0)
            )
        )
        _add_chunks(val_text, "valuation")

    # ── ESG ──────────────────────────────────────────────────────────────────
    esg = profile.get("ESGScores", {})
    if esg:
        esg_text = (
            f"{ticker_symbol} ESG scores: "
            + " | ".join(
                f"{k}={v}"
                for k, v in esg.items()
                if v not in (None, "", "None", "0", 0)
            )
        )
        _add_chunks(esg_text, "esg")

    # ── Analyst ratings ──────────────────────────────────────────────────────
    ar = profile.get("AnalystRatings", {})
    if ar:
        ar_text = (
            f"{ticker_symbol} analyst ratings: "
            + " | ".join(f"{k}={v}" for k, v in ar.items() if v not in (None, ""))
        )
        _add_chunks(ar_text, "analyst_ratings")

    return chunks


def _ollama_embed(texts: list[str], model: str, base_url: str) -> list[list[float]]:
    """
    Embed a list of texts using Ollama's /api/embed endpoint.
    Returns a list of float vectors (one per text).
    Raises RuntimeError if Ollama is unreachable or the model is not pulled.
    """
    url = f"{base_url.rstrip('/')}/api/embed"
    embeddings: list[list[float]] = []
    # Send in batches of 16 to avoid overloading Ollama
    batch_size = 16
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = requests.post(
            url,
            json={"model": model, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        batch_embeddings = data.get("embeddings")
        if not batch_embeddings:
            raise RuntimeError(
                f"Ollama /api/embed returned empty embeddings for model '{model}'. "
                "Ensure the model is pulled: `ollama pull {model}`"
            )
        embeddings.extend(batch_embeddings)
    return embeddings


def ingest_chunks_for_ticker(
    ticker_symbol: str,
    embedding_model: str | None = None,
    chunk_size: int = 512,
    overlap: int = 50,
    base_etl_dir: "Path | str | None" = None,
) -> int:
    """
    Read company_profile.json for *ticker_symbol*, build text chunks, embed them
    using Ollama's nomic-embed-text (768-dim), and upsert into Neo4j.

    The embedding model and Ollama URL are read from env vars:
      EMBEDDING_MODEL   (default: nomic-embed-text)
      OLLAMA_BASE_URL   (default: http://localhost:11434)

    Returns the total number of chunks written.
    """
    model = embedding_model or OLLAMA_EMBED_MODEL
    ollama_url = OLLAMA_BASE_URL

    if base_etl_dir is None:
        base_etl_dir = BASE_ETL_DIR
    ticker_dir = Path(base_etl_dir) / ticker_symbol
    profile_path = ticker_dir / "company_profile.json"

    if not profile_path.exists():
        print(f"[Chunk Ingestion] No company_profile.json for {ticker_symbol} — skipping")
        return 0

    with open(profile_path) as f:
        profile: dict = json.load(f)

    chunks = _build_profile_chunks(ticker_symbol, profile, chunk_size, overlap)
    if not chunks:
        print(f"[Chunk Ingestion] No text found for {ticker_symbol} — skipping")
        return 0

    print(f"[Chunk Ingestion] {ticker_symbol}: embedding {len(chunks)} chunks "
          f"via Ollama '{model}' at {ollama_url} …")
    texts = [c["text"] for c in chunks]
    try:
        embeddings = _ollama_embed(texts, model, ollama_url)
    except Exception as exc:
        print(f"[Chunk Ingestion] ERROR embedding chunks for {ticker_symbol}: {exc}")
        raise

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    cypher = """
    UNWIND $rows AS row
    MERGE (c:Company {ticker: row.ticker})
    MERGE (chunk:Chunk {chunk_id: row.chunk_id})
      SET chunk.text       = row.text,
          chunk.section    = row.section,
          chunk.filing_date = row.filing_date,
          chunk.ticker     = row.ticker,
          chunk.embedding  = row.embedding
    MERGE (c)-[:HAS_CHUNK]->(chunk)
    """

    driver = get_driver()
    try:
        with driver.session() as session:
            # Write in batches of 50 to avoid parameter size limits
            batch_size = 50
            written = 0
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                session.run(cypher, rows=batch)
                written += len(batch)
    finally:
        driver.close()

    print(f"[Chunk Ingestion] {ticker_symbol}: wrote {written} chunks to Neo4j")
    return written


# ── Type coercion helpers ─────────────────────────────────────────────────────

def _coerce_value(v):
    """
    Convert a value to a Neo4j-safe Python native type.

    Neo4j properties must be: bool, int, float, str, list of the above, or None.
    numpy scalars, NaN, Inf, and complex dicts/lists of non-primitives are
    handled here.
    """
    if v is None:
        return None

    # numpy bool
    if isinstance(v, (np.bool_,)):
        return bool(v)

    # numpy integers
    if isinstance(v, (np.integer,)):
        return int(v)

    # numpy floats — includes np.float64, np.float32, etc.
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f

    # Python floats
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v

    # Python int / bool — pass through
    if isinstance(v, (int, bool)):
        return v

    # Strings — strip but keep None check
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in ("nan", "none", "null", "nat", ""):
            return None
        return s

    # Lists — recursively coerce items, drop non-primitive elements
    if isinstance(v, list):
        safe = []
        for item in v:
            cv = _coerce_value(item)
            if cv is None or isinstance(cv, (bool, int, float, str)):
                safe.append(cv)
        return safe if safe else None

    # dicts, other objects → JSON string (Neo4j can't store arbitrary dicts)
    try:
        return json.dumps(v, default=str)
    except Exception:
        return str(v)


def _coerce_props(d: dict) -> dict:
    """Return a new dict with all values coerced to Neo4j-safe types."""
    result = {}
    for k, v in d.items():
        safe = _coerce_value(v)
        if safe is not None:
            result[str(k)] = safe
    return result


def _safe_float(v) -> float | None:
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


# ── Transaction functions ─────────────────────────────────────────────────────

def _load_company_profile(tx, ticker_symbol: str, row: dict):
    """
    MERGE a :Company node and SET all scalar properties from the profile row.
    Covers: General fields + Highlights_* + Valuation_* + SharesStats_* +
    SplitsDividends_* columns (as written by _flatten_fundamentals in the DAG).
    """
    props = _coerce_props(row)
    props["ticker"] = ticker_symbol
    tx.run(
        """
        MERGE (c:Company {ticker: $ticker})
        SET c += $props
        """,
        ticker=ticker_symbol,
        props=props,
    )


def _load_etf_constituent(tx, ticker_symbol: str, row_dict: dict):
    """
    MERGE :Company ETF and constituent nodes, and MERGE the [:CONTAINS] edge.
    """
    constituent = (
        row_dict.get("Code")
        or row_dict.get("ticker")
        or row_dict.get("symbol")
    )
    if not constituent:
        return

    weight = _safe_float(
        row_dict.get("Weight") or row_dict.get("weight") or 0.0
    ) or 0.0

    tx.run(
        """
        MERGE (etf:Company {ticker: $etf_ticker})
        SET etf.is_etf = true
        MERGE (c:Company {ticker: $constituent})
        MERGE (etf)-[r:CONTAINS]->(c)
        SET r.weight = $weight
        """,
        etf_ticker=ticker_symbol,
        constituent=str(constituent),
        weight=weight,
    )


def _load_generic_record(tx, ticker_symbol: str, data_name: str, row_index: int, row: dict):
    """
    Store any neo4j-destined row as a :DataRecord node linked to :Company.
    """
    props = _coerce_props(row)
    props["data_name"]    = data_name
    props["row_index"]    = row_index
    props["ticker"]       = ticker_symbol

    tx.run(
        """
        MERGE (c:Company {ticker: $ticker})
        MERGE (d:DataRecord {ticker: $ticker, data_name: $data_name, row_index: $row_index})
        SET d += $props
        MERGE (c)-[:HAS_DATA]->(d)
        """,
        ticker=ticker_symbol,
        data_name=data_name,
        row_index=row_index,
        props=props,
    )


# ── Per-ticker loader ─────────────────────────────────────────────────────────

def load_neo4j_for_ticker(ticker_symbol: str) -> int:
    """
    Load all neo4j-destined CSV files for one ticker into Neo4j.
    Also ensures the vector index schema exists before writing.
    Returns total number of rows written.
    """
    ensure_neo4j_schema()

    ticker_dir    = BASE_ETL_DIR / ticker_symbol
    metadata_path = ticker_dir / "metadata.json"

    if not metadata_path.exists():
        print(f"[Neo4j Loader] No metadata.json for {ticker_symbol} — skipping")
        return 0

    with open(metadata_path) as f:
        metadata: dict = json.load(f)

    count  = 0
    driver = get_driver()

    with driver.session() as session:
        for data_name, info in metadata.items():
            dest = info.get("storage_destination")
            if dest != "neo4j":
                continue

            csv_path = ticker_dir / f"{data_name}.csv"
            if not csv_path.exists():
                print(f"[Neo4j Loader] Missing CSV for {data_name} at {csv_path}")
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[Neo4j Loader] Failed to read {csv_path}: {e}")
                continue

            if df.empty:
                print(f"[Neo4j Loader] Empty CSV for {data_name} — skipping")
                continue

            print(f"[Neo4j Loader] Loading {data_name} for {ticker_symbol} ({len(df)} rows)")

            if data_name == "company_profile":
                for _, row in df.iterrows():
                    try:
                        session.execute_write(
                            _load_company_profile, ticker_symbol, row.to_dict()
                        )
                        count += 1
                    except Exception as exc:
                        print(f"[Neo4j Loader] {data_name} row error: {exc}")

            elif data_name == "etf_index_constituents":
                for _, row in df.iterrows():
                    try:
                        session.execute_write(
                            _load_etf_constituent, ticker_symbol, row.to_dict()
                        )
                        count += 1
                    except Exception as exc:
                        print(f"[Neo4j Loader] {data_name} row error: {exc}")

            else:
                # Generic fallback — any other neo4j-destined data type
                for row_idx, (_, row) in enumerate(df.iterrows()):
                    try:
                        session.execute_write(
                            _load_generic_record, ticker_symbol, data_name, row_idx, row.to_dict()
                        )
                        count += 1
                    except Exception as exc:
                        print(f"[Neo4j Loader] {data_name} row {row_idx} error: {exc}")

    driver.close()
    print(f"[Neo4j Loader] {ticker_symbol}: {count} total writes")

    # Ingest company profile text chunks with Ollama embeddings (768-dim)
    # Skip for the _MACRO pseudo-ticker (no company_profile.json there)
    if ticker_symbol != "_MACRO":
        try:
            chunk_count = ingest_chunks_for_ticker(
                ticker_symbol,
                base_etl_dir=BASE_ETL_DIR,
            )
            count += chunk_count
        except Exception as exc:
            print(f"[Neo4j Loader] WARNING: chunk ingestion failed for {ticker_symbol}: {exc}")

    return count


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    # Load .env for local runs
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as ef:
            for line in ef:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())

    # Re-read Neo4j env after loading .env
    NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "SecureNeo4jPass2025!")
    OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
    OLLAMA_EMBED_MODEL = os.getenv("EMBEDDING_MODEL",  "nomic-embed-text")

    parser = argparse.ArgumentParser(description="Load EODHD agent data into Neo4j")
    parser.add_argument("ticker", nargs="?", default="AAPL",
                        help="Ticker symbol to load (default: AAPL)")
    parser.add_argument("--all", action="store_true",
                        help="Load all 5 tickers")
    args = parser.parse_args()

    if args.all:
        tickers = os.getenv("TRACKED_TICKERS", "AAPL,TSLA,NVDA,MSFT,GOOGL").split(",")
        for t in tickers:
            print(f"\n=== Loading {t.strip()} ===")
            print(f"  writes: {load_neo4j_for_ticker(t.strip())}")
    else:
        print(load_neo4j_for_ticker(args.ticker))
