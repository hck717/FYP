"""
Step 1 (Neo4j variant): Load macro report chunks and latest earnings call
from Neo4j.

The ingestion pipeline stores macro reports as Chunk nodes attached to a
special Company node with ticker="_MACRO", and earnings calls attached to
regular ticker Company nodes.

This module retrieves those chunks and returns them as
``langchain_core.documents.Document`` objects with the same metadata
schema that downstream steps expect.

Data source
-----------
Neo4j nodes look like::

    (:Company {ticker: "_MACRO"})-[:HAS_CHUNK]->(:Chunk {
        section:      "macro_report",
        text:         "...",
        chunk_id:     "_MACRO::macro_report::<hash>",
        filing_date:  "2025-10-30",
        source_file:  "Federal Reserve Report 20251030 ....pdf",
        source_name:  "Federal Reserve Report 2025-10-30",
        institution:  "Federal Reserve",
        embedding:    [float, ...]
    })

    (:Company {ticker: "AAPL"})-[:HAS_CHUNK]->(:Chunk {
        section:      "earnings_call",
        text:         "...",
        filing_date:  "2025-01-29",
        source_file:  "Apple Inc Earnings Call 20250129 ....pdf",
        source_name:  "Apple Inc Earnings Call 2025-01-29",
        embedding:    [float, ...]
    })

Run standalone (for debugging)::

    python agent_step1_neo4j.py          # default AAPL
    python agent_step1_neo4j.py NVDA
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── Load .env ─────────────────────────────────────────────────────────────────
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

# ── Neo4j config (mirrors ingestion/etl/ingest_*.py) ─────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "SecureNeo4jPass2025!")


def _get_driver():
    from neo4j import GraphDatabase  # lazy import — not always available
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_doc_name(raw_name: str) -> str:
    """Ensure doc_name ends with exactly one '.pdf' suffix."""
    import re as _re
    if not raw_name:
        return raw_name
    # Collapse any run of trailing .pdf/.PDF into a single .pdf
    name = _re.sub(r'(\.pdf)+$', '.pdf', raw_name, flags=_re.IGNORECASE)
    if not name.lower().endswith(".pdf"):
        name = name + ".pdf"
    return name


def _chunks_to_docs(
    chunks: list[dict],
    ticker: str,
    doc_type: str,
) -> list[Document]:
    """
    Convert Neo4j chunk records to Document objects.
    
    ``page_number`` is synthesised from the chunk's position in the list
    (1-indexed) because Neo4j chunks don't carry a page number.
    
    ``doc_name`` is normalised to end with ``.pdf`` so it matches the
    filenames used elsewhere (downstream steps use doc_name as citation keys).
    """
    docs: list[Document] = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("text") or ""
        if not text.strip():
            continue
        raw_name = (
            chunk.get("source_file")
            or chunk.get("source_name")
            or chunk.get("chunk_id", "unknown")
        )
        docs.append(Document(
            page_content=text,
            metadata={
                "ticker":       ticker,
                "doc_type":     doc_type,
                "doc_name":     _normalise_doc_name(raw_name),
                "page_number":  i + 1,
                # Extra Neo4j fields
                "institution":  chunk.get("institution", ""),
                "filing_date":  chunk.get("filing_date", ""),
                "section":      chunk.get("section", ""),
                "chunk_id":     chunk.get("chunk_id", ""),
            },
        ))
    return docs


def _fetch_chunks(ticker: str, section: str) -> list[dict]:
    """
    Fetch all Chunk nodes for a ticker/section from Neo4j,
    ordered by chunk_id (stable ordering).
    """
    driver = _get_driver()
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Company {ticker: $ticker})-[:HAS_CHUNK]->(ch:Chunk {section: $section})
                RETURN ch.chunk_id     AS chunk_id,
                       ch.text         AS text,
                       ch.section      AS section,
                       ch.filing_date  AS filing_date,
                       ch.source_file  AS source_file,
                       ch.source_name  AS source_name,
                       ch.institution  AS institution,
                       ch.embedding    AS embedding
                ORDER BY ch.chunk_id
                """,
                ticker=ticker,
                section=section,
            )
            return [dict(r) for r in result]
    finally:
        driver.close()


# ── Public API ────────────────────────────────────────────────────────────────

def load_macro_and_earnings(
    target_ticker: str,
) -> Tuple[List[Document], List[Document], List[str]]:
    """
    High-level loader. Fetches:
    - ALL macro report chunks (ticker="_MACRO", section="macro_report")
    - LATEST earnings call chunk for target ticker (section="earnings_call")
    
    Returns
    -------
    macro_pages :
        Document list of all macro report chunks, ordered by chunk_id.
    earnings_pages :
        Document list containing ONLY the latest earnings call for target_ticker.
    macro_doc_names :
        List of unique source_name values from macro chunks (for per-report linkage).
    """
    logger.info("[step1_neo4j] Fetching macro chunks and earnings for ticker=%s", target_ticker)

    # Fetch all macro chunks
    macro_chunks = _fetch_chunks("_MACRO", "macro_report")
    logger.info("[step1_neo4j] Fetched %d macro report chunks", len(macro_chunks))

    if not macro_chunks:
        raise ValueError(
            f"No macro_report chunks found in Neo4j for ticker '_MACRO'. "
            f"Run `ingestion/etl/ingest_macro_reports.py` first."
        )

    # Fetch latest earnings call for target ticker
    earnings_chunks_all = _fetch_chunks(target_ticker, "earnings_call")
    logger.info("[step1_neo4j] Fetched %d earnings call chunks for ticker=%s", len(earnings_chunks_all), target_ticker)

    if not earnings_chunks_all:
        logger.warning(
            "[step1_neo4j] No earnings_call chunks found in Neo4j for ticker %s. "
            "Using only macro data.",
            target_ticker,
        )
        earnings_chunks = []
    else:
        # Pick the latest (newest filing_date)
        def _get_date(chunk: dict) -> str:
            date = chunk.get("filing_date") or ""
            return date if date and date != "0000-00-00" else ""
        
        sorted_chunks = sorted(earnings_chunks_all, key=_get_date, reverse=True)
        # Use only the latest document's chunks (group by source_name)
        latest_source = sorted_chunks[0].get("source_name") or sorted_chunks[0].get("source_file")
        earnings_chunks = [c for c in sorted_chunks if (c.get("source_name") or c.get("source_file")) == latest_source]
        logger.info("[step1_neo4j] Using latest earnings call: %s (%d chunks)", latest_source, len(earnings_chunks))

    # Convert to Documents
    macro_docs = _chunks_to_docs(macro_chunks, "_MACRO", "macro_report")
    earnings_docs = _chunks_to_docs(earnings_chunks, target_ticker, "earnings_call")

    # Extract unique macro report names for per-report processing
    macro_names = list(set(c.get("source_name") or c.get("source_file") or c.get("chunk_id") for c in macro_chunks))
    macro_names = sorted([n for n in macro_names if n])  # Remove None/empty, sort

    logger.info(
        "[step1_neo4j] Loaded: %d macro docs from %d reports, %d earnings docs",
        len(macro_docs), len(macro_names), len(earnings_docs),
    )

    return macro_docs, earnings_docs, macro_names


# ── Standalone debug run ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"=== Neo4j Step 1: Load macro + earnings for {ticker_arg} ===\n")

    try:
        macro_pages, earnings_pages, macro_names = load_macro_and_earnings(ticker_arg)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"Macro reports  : {len(macro_names)} unique sources")
    print(f"Macro docs     : {len(macro_pages)}")
    print(f"Earnings docs  : {len(earnings_pages)}")

    if macro_pages:
        print("\n--- Sample macro doc metadata ---")
        print(macro_pages[0].metadata)
        print(f"  text[:120]: {macro_pages[0].page_content[:120]!r}")

    if earnings_pages:
        print("\n--- Sample earnings doc metadata ---")
        print(earnings_pages[0].metadata)
        print(f"  text[:120]: {earnings_pages[0].page_content[:120]!r}")
