"""
Step 1: Discover PDF files for a given ticker.
Step 2: Load PDF pages with metadata, using pdfplumber fallback if pypdf fails.

Run:
    python agent_step1_load.py
"""

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# ── Config ────────────────────────────────────────────────────────────────────
TICKER   = "AAPL"   # change to GOOGL, MSFT, NVDA, TSLA
BASE_DIR = Path("data_reports")

# ── Step 1: Discover files ─────────────────────────────────────────────────────

def find_subdir(root: Path, keyword: str) -> Path | None:
    """Find first subdirectory whose name contains keyword (case-insensitive)."""
    matches = [p for p in root.iterdir() if p.is_dir() and keyword in p.name.lower()]
    return matches[0] if matches else None


def list_stock_files(base_dir: Path, ticker: str):
    """
    Return (broker_pdfs, transcript_pdfs) for the given ticker.
    transcript_pdfs is sorted so index 0 = latest, index 1 = previous.
    """
    stock_dir = base_dir / ticker
    if not stock_dir.exists():
        raise FileNotFoundError(f"Ticker folder not found: {stock_dir}")

    broker_dir  = find_subdir(stock_dir, "broker")
    earning_dir = find_subdir(stock_dir, "earn")   # matches 'earning', 'earningcall', etc.

    if broker_dir is None:
        raise ValueError(f"Cannot find broker folder under {stock_dir}")
    if earning_dir is None:
        raise ValueError(f"Cannot find earning folder under {stock_dir}")

    broker_pdfs     = sorted(broker_dir.glob("*.pdf"))
    transcript_pdfs = sorted(earning_dir.glob("*.pdf"))

    if len(transcript_pdfs) < 2:
        raise ValueError(f"Need at least 2 transcripts, found {len(transcript_pdfs)} for {ticker}")

    # Sort by date parsed from filename (most recent first).
    # Filenames contain dates like 20251030 or 2026129 (YYYYMMDD or YYYYMDD).
    # We extract the first 7-8 digit sequence and zero-pad it to YYYYMMDD for comparison.
    def extract_date_from_filename(p: Path) -> str:
        match = re.search(r"(\d{7,8})", p.name)
        if match:
            raw = match.group(1)
            # If 7 digits (e.g. 2026129 = 2026/1/29), zero-pad month and day
            # Format: YYYY + remaining → pad to 8 digits total
            if len(raw) == 7:
                raw = raw[:4] + raw[4:].zfill(4)  # e.g. 2026129 -> 20260129
            return raw  # YYYYMMDD string, sortable lexicographically
        # No date found: fall back to filename alphabetical order
        print(f"[WARN] Could not parse date from filename: {p.name}. Using filename sort.")
        return p.name

    transcript_pdfs = sorted(
        transcript_pdfs,
        key=extract_date_from_filename,
        reverse=True   # most recent first
    )[:2]

    return broker_pdfs, transcript_pdfs


# ── Step 2: Load PDF pages with metadata ──────────────────────────────────────

# This list records which parser was used for each file.
# Useful for audit / debugging.
parse_log = []


def _load_with_pdfplumber(path: Path, ticker: str, doc_type: str, period: str) -> list[Document]:
    """
    Fallback parser using pdfplumber.
    Returns one Document per page, same structure as PyPDFLoader output.
    """
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append(Document(
                page_content=text,
                metadata={
                    "ticker":      ticker,
                    "doc_type":    doc_type,
                    "doc_name":    path.name,
                    "period":      period,
                    "page_number": i + 1,
                }
            ))
    return pages


def load_pdf_pages(paths: list[Path], ticker: str, doc_type: str, period: str = "") -> list[Document]:
    """
    Load a list of PDF files into Document objects (one per page).

    Strategy:
      1. Try PyPDFLoader (fast, standard).
      2. If it crashes (e.g. malformed font metadata), fall back to pdfplumber.
      3. If both fail, log the error and skip the file.

    Each Document carries metadata:
      ticker, doc_type, doc_name, period, page_number
    """
    all_pages = []

    for path in paths:
        # ── Attempt 1: PyPDFLoader ────────────────────────────────────────────
        try:
            docs = PyPDFLoader(str(path)).load()

            # Overwrite metadata with our standard fields
            for d in docs:
                d.metadata = {
                    "ticker":      ticker,
                    "doc_type":    doc_type,
                    "doc_name":    path.name,
                    "period":      period,
                    "page_number": int(d.metadata.get("page", 0)) + 1,
                }

            all_pages.extend(docs)
            parse_log.append({
                "file":   path.name,
                "parser": "PyPDFLoader",
                "pages":  len(docs),
                "error":  None,
            })

        except Exception as e1:
            # ── Attempt 2: pdfplumber fallback ───────────────────────────────
            print(f"[WARN] PyPDFLoader failed on '{path.name}'")
            print(f"       Reason : {type(e1).__name__}: {e1}")
            print(f"       Action : falling back to pdfplumber ...")

            try:
                fallback_docs = _load_with_pdfplumber(path, ticker, doc_type, period)
                all_pages.extend(fallback_docs)
                parse_log.append({
                    "file":   path.name,
                    "parser": "pdfplumber",
                    "pages":  len(fallback_docs),
                    "error":  str(e1),
                })
                print(f"       Result : pdfplumber OK — {len(fallback_docs)} pages loaded.\n")

            except Exception as e2:
                # Both failed — skip this file, keep going
                print(f"[ERROR] pdfplumber also failed on '{path.name}'")
                print(f"        Reason : {type(e2).__name__}: {e2}")
                print(f"        Action : skipping this file. Check it manually.\n")
                parse_log.append({
                    "file":   path.name,
                    "parser": "FAILED",
                    "pages":  0,
                    "error":  str(e2),
                })

    return all_pages


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1
    print(f"=== Step 1: Discover files for {TICKER} ===\n")
    broker_pdfs, transcript_pdfs = list_stock_files(BASE_DIR, TICKER)

    latest_t  = transcript_pdfs[0]
    previous_t = transcript_pdfs[1]

    print(f"Latest transcript  : {latest_t.name}")
    print(f"Previous transcript: {previous_t.name}")
    print(f"Broker reports ({len(broker_pdfs)}):")
    for p in broker_pdfs:
        print(f"  - {p.name}")

    # Step 2
    print(f"\n=== Step 2: Load PDF pages ===\n")

    transcript_pages  = []
    transcript_pages += load_pdf_pages([latest_t],   TICKER, "transcript", "latest")
    transcript_pages += load_pdf_pages([previous_t], TICKER, "transcript", "previous")
    broker_pages       = load_pdf_pages(broker_pdfs,  TICKER, "broker")

    # Summary
    print("\n=== Parse Log ===")
    log_df = pd.DataFrame(parse_log)
    print(log_df.to_string(index=False))

    print(f"\nTotal transcript pages : {len(transcript_pages)}")
    print(f"Total broker pages     : {len(broker_pages)}")

    # Spot-check: show metadata of first page of each doc type
    print("\n--- Sample metadata (transcript page 1) ---")
    print(transcript_pages[0].metadata)
    print("\n--- Sample metadata (broker page 1) ---")
    print(broker_pages[0].metadata)
