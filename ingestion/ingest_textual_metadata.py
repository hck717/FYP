#!/usr/bin/env python3
"""
ingest_textual_metadata.py — One-shot script to ingest PDF metadata into PostgreSQL.

Reads metadata.json from each ticker's textual data directory and upserts
document metadata into the `textual_documents` table.

Run directly (not via Airflow):
    python ingestion/ingest_textual_metadata.py

No binary content is stored; PDFs are later ingested into Qdrant separately.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
_env_path = _REPO_ROOT / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ── Path / config ─────────────────────────────────────────────────────────────
TEXTUAL_DATA_DIR = _REPO_ROOT / "data" / "textual data"
TRACKED_TICKERS  = os.getenv("TRACKED_TICKERS", "AAPL,TSLA,NVDA,MSFT,GOOGL").split(",")

# ── PostgreSQL env overrides for local run ────────────────────────────────────
# Force localhost — .env sets POSTGRES_HOST=postgres (Docker service name),
# but this script runs directly on the host machine.
os.environ["POSTGRES_HOST"] = "localhost"

# Import ETL loader (adds ensure_tables + _insert_textual_documents)
sys.path.insert(0, str(Path(__file__).resolve().parent / "etl"))
import load_postgres as _lp  # noqa: E402
from load_postgres import ensure_tables, _insert_textual_documents  # noqa: E402

# Patch host for local run (module-level constant is set at import time)
_lp.PG_HOST = os.getenv("POSTGRES_HOST", "localhost")


def load_ticker_metadata(ticker: str) -> list[dict]:
    """Read metadata.json for one ticker and return the list of document dicts."""
    meta_path = TEXTUAL_DATA_DIR / ticker / "metadata.json"
    if not meta_path.exists():
        print(f"[TextualIngest] {ticker}: metadata.json not found at {meta_path}")
        return []

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    docs = meta.get("documents", [])
    if not docs:
        print(f"[TextualIngest] {ticker}: no documents in metadata.json")
    return docs


def main() -> None:
    print("=== Textual Document Metadata Ingestion ===")
    print(f"Textual data dir : {TEXTUAL_DATA_DIR}")
    print(f"Tickers          : {', '.join(TRACKED_TICKERS)}")
    print()

    # Ensure textual_documents table exists
    ensure_tables()

    total = 0
    for ticker in TRACKED_TICKERS:
        ticker = ticker.strip()
        docs = load_ticker_metadata(ticker)
        if not docs:
            continue

        # Normalise field names to what _insert_textual_documents expects
        normalised = []
        for d in docs:
            normalised.append({
                "ticker":              d.get("ticker", ticker),
                "doc_type":            d.get("document_type", ""),
                "filename":            d.get("filename", ""),
                "filepath":            d.get("filepath", ""),
                "institution":         d.get("institution", ""),
                "date_approx":         d.get("date_approx"),
                "file_size_bytes":     d.get("file_size_bytes"),
                "md5_hash":            d.get("md5_hash"),
                "ingested_into_qdrant": d.get("ingested", False),
            })

        n = _insert_textual_documents(normalised)
        total += n
        print(f"[TextualIngest] {ticker}: {n} documents upserted")

    print(f"\nDone. Total documents upserted: {total}")


if __name__ == "__main__":
    main()
