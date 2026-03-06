"""
ingest_neo4j_chunks.py  —  Bootstrap Neo4j vector index + company-profile chunks
==================================================================================

Run this script ONCE (or at any time) to:
  1. Create the `chunk_embedding` vector index in Neo4j (idempotent).
  2. Read each ticker's company_profile.json from agent_data/.
  3. Split the rich text into overlapping 512-char chunks.
  4. Embed them with all-MiniLM-L6-v2 (sentence-transformers, dim=384).
  5. Upsert Chunk nodes + HAS_CHUNK relationships into Neo4j.

Usage:
    # All 5 tickers (default):
    python ingest_neo4j_chunks.py

    # Single ticker:
    python ingest_neo4j_chunks.py AAPL

    # All tickers from env TRACKED_TICKERS:
    TRACKED_TICKERS=AAPL,MSFT python ingest_neo4j_chunks.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Resolve ETL dir so load_neo4j can be imported both locally and in container
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Load .env for local runs
_env_path = _THIS_DIR.parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _ef:
        for _line in _ef:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# Re-read Neo4j env after loading .env so module-level defaults pick up values.
# When running locally (outside Docker), override the docker-internal hostname
# bolt://neo4j:7687 with the localhost equivalent so we can reach the container.
import load_neo4j as _lnj  # noqa: E402

_raw_neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
# Replace the Docker service hostname with localhost for local execution
if "neo4j:7687" in _raw_neo4j_uri and not os.getenv("RUNNING_IN_DOCKER"):
    _raw_neo4j_uri = _raw_neo4j_uri.replace("neo4j:7687", "localhost:7687")

_lnj.NEO4J_URI      = _raw_neo4j_uri
_lnj.NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
_lnj.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "SecureNeo4jPass2025!")

from load_neo4j import ensure_neo4j_schema, ingest_chunks_for_ticker  # noqa: E402

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
# Use a sentence-transformers specific env var so it doesn't conflict with the
# Ollama EMBEDDING_MODEL (e.g. nomic-embed-text) used elsewhere.
EMBEDDING_MODEL = os.getenv("ST_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE      = int(os.getenv("BUSINESS_ANALYST_CHUNK_SIZE", "512"))
OVERLAP         = int(os.getenv("BUSINESS_ANALYST_OVERLAP",    "50"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap Neo4j vector index and ingest company-profile chunks"
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default=None,
        help="Single ticker symbol to ingest (default: all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all tickers from TRACKED_TICKERS env var or built-in default list",
    )
    args = parser.parse_args()

    # Step 1: ensure schema (idempotent)
    print("=== Ensuring Neo4j schema (vector index) ===")
    ensure_neo4j_schema()

    # Step 2: determine tickers
    if args.ticker:
        tickers = [args.ticker.strip().upper()]
    else:
        env_tickers = os.getenv("TRACKED_TICKERS", "")
        tickers = (
            [t.strip() for t in env_tickers.split(",") if t.strip()]
            if env_tickers
            else DEFAULT_TICKERS
        )

    # Step 3: ingest chunks
    total = 0
    for ticker in tickers:
        print(f"\n=== Ingesting chunks for {ticker} ===")
        written = ingest_chunks_for_ticker(
            ticker,
            embedding_model=EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP,
        )
        total += written

    print(f"\n=== Done.  Total chunks written: {total} ===")


if __name__ == "__main__":
    main()
