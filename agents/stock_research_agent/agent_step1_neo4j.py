"""
Step 1 (Neo4j variant): Load earnings-call and broker-report text chunks
from Neo4j instead of local PDF files.

The ingestion pipeline (`ingestion/etl/ingest_earnings_calls.py` and
`ingest_broker_reports.py`) extracts PDF text, chunks it, embeds it
with Ollama (nomic-embed-text, dim=768), and stores it in Neo4j as
`:Chunk` nodes attached to `:Company` nodes via `:HAS_CHUNK` edges.

This module retrieves those chunks and returns them as
``langchain_core.documents.Document`` objects with the same metadata
schema that the downstream pipeline steps (3–7) expect:

    ticker, doc_type, doc_name, period, page_number

Data source
-----------
Neo4j nodes for a ticker look like::

    (:Company {ticker: "AAPL"})-[:HAS_CHUNK]->(:Chunk {
        section:      "earnings_call" | "broker_report",
        text:         "...",
        chunk_id:     "AAPL::earnings_call::<hash>",
        filing_date:  "2025-10-30",
        source_file:  "Apple Inc Earnings Call 20251030 ....pdf",
        source_name:  "Apple Inc Earnings Call 2025-10-30",
        institution:  "...",   # broker_report only
        embedding:    [float, ...]
    })

Compatibility
-------------
The returned ``Document`` list is a drop-in replacement for the output
of ``agent_step1_load.load_pdf_pages``.  The rest of the pipeline
(steps 3–7) works unchanged.

Run standalone (for debugging)::

    python agent_step1_neo4j.py          # default AAPL
    python agent_step1_neo4j.py NVDA
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

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
    """Ensure doc_name ends with exactly one '.pdf' suffix.

    Neo4j may store source_file with a duplicated extension (e.g. 'foo.pdf.pdf')
    as an ingestion artefact.  Strip all trailing '.pdf' repetitions first, then
    re-add a single '.pdf' so the result matches the PDF-mode reference filenames.
    """
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
    period: str = "",
) -> list[Document]:
    """
    Convert Neo4j chunk records to Document objects.

    ``page_number`` is synthesised from the chunk's position in the list
    (1-indexed) because Neo4j chunks don't carry a page number — they are
    word-boundary splits across the whole document.

    ``doc_name`` is normalised to end with ``.pdf`` so it matches the
    filenames used in PDF mode (downstream steps use doc_name as citation keys).
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
                "period":       period,
                "page_number":  i + 1,
                # Extra Neo4j fields (used downstream for broker parsing)
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
    ordered by chunk_id (stable ordering for reproducibility).
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


def _extract_date_from_name(name: str) -> str:
    """
    Attempt to extract a YYYY-MM-DD date from a source_name string.
    Handles formats like:
      'Apple Inc Earnings Call 2026-01-29'
      'Apple Inc Earnings Call 20261029 ...'
    Returns '' if no date found.
    """
    import re
    # Try ISO format: YYYY-MM-DD
    m = re.search(r'(\d{4}-\d{2}-\d{2})', name)
    if m:
        return m.group(1)
    # Try compact YYYYMMDD (8 digits)
    m = re.search(r'(\d{4})(\d{2})(\d{2})', name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return ""


def _pick_two_transcripts(
    chunks: list[dict],
) -> Tuple[list[dict], list[dict], Optional[str], Optional[str]]:
    """
    Split transcript chunks into two groups: latest and previous.

    Grouping is by ``source_name`` (unique per source PDF).  The groups
    are sorted by ``filing_date`` descending so the most-recent call is
    "latest".  Returns (latest_chunks, previous_chunks, latest_name, previous_name).
    Raises ``ValueError`` if fewer than 2 distinct source documents are found.

    When ``filing_date`` is missing or '0000-00-00' (ingestion artefact),
    the date is parsed from the source_name string as a fallback.
    """
    # Group by source_name
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        key = chunk.get("source_name") or chunk.get("source_file") or chunk.get("chunk_id", "unknown")
        groups[key].append(chunk)

    if len(groups) < 2:
        raise ValueError(
            f"Need at least 2 distinct earnings-call transcripts in Neo4j, "
            f"found {len(groups)}: {list(groups.keys())}"
        )

    # Sort group keys by the best filing_date among chunks in that group.
    # Fall back to parsing date from source_name when filing_date is absent
    # or set to the sentinel value '0000-00-00'.
    def _group_date(name: str) -> str:
        raw_dates = [c.get("filing_date") or "" for c in groups[name]]
        valid = [d for d in raw_dates if d and d != "0000-00-00"]
        if valid:
            return max(valid)
        # Fallback: extract from the group key (source_name)
        return _extract_date_from_name(name)

    sorted_keys = sorted(groups.keys(), key=_group_date, reverse=True)
    latest_key   = sorted_keys[0]
    previous_key = sorted_keys[1]

    logger.debug(
        "[step1_neo4j] Transcript ordering: latest=%r (date=%s), previous=%r (date=%s)",
        latest_key, _group_date(latest_key),
        previous_key, _group_date(previous_key),
    )

    return groups[latest_key], groups[previous_key], latest_key, previous_key


# ── Public API ────────────────────────────────────────────────────────────────

def list_stock_files_neo4j(ticker: str) -> Tuple[list[dict], list[dict], list[dict], Optional[str], Optional[str]]:
    """
    Neo4j equivalent of ``agent_step1_load.list_stock_files``.

    Returns
    -------
    broker_chunks :
        All broker-report Chunk dicts for *ticker*.
    transcript_chunks :
        Latest + previous transcript Chunk dicts combined (latest first).
    latest_name :
        Human-readable name of the latest transcript.
    previous_name :
        Human-readable name of the previous transcript.
    """
    logger.info("[step1_neo4j] Fetching chunks for ticker=%s", ticker)

    broker_chunks = _fetch_chunks(ticker, "broker_report")
    transcript_chunks_all = _fetch_chunks(ticker, "earnings_call")

    logger.info(
        "[step1_neo4j] %s: %d broker chunks, %d earnings-call chunks",
        ticker, len(broker_chunks), len(transcript_chunks_all),
    )

    if not transcript_chunks_all:
        raise ValueError(
            f"No earnings_call chunks found in Neo4j for ticker {ticker!r}. "
            f"Run `ingestion/etl/ingest_earnings_calls.py {ticker}` first."
        )
    if not broker_chunks:
        logger.warning(
            "[step1_neo4j] No broker_report chunks found in Neo4j for ticker %s. "
            "Broker analysis will be skipped.",
            ticker,
        )

    latest_chunks, previous_chunks, latest_name, previous_name = _pick_two_transcripts(
        transcript_chunks_all
    )

    return broker_chunks, latest_chunks, previous_chunks, latest_name, previous_name


def load_neo4j_pages(
    ticker: str,
) -> Tuple[List[Document], List[Document], str, str]:
    """
    High-level loader.  Fetches all chunks from Neo4j and returns:

    - ``transcript_pages``  : Document list (latest transcript then previous)
    - ``broker_pages``      : Document list (all broker reports)
    - ``latest_name``       : filename/source name of the latest transcript
    - ``previous_name``     : filename/source name of the previous transcript

    This is the entry-point used by ``agent.py`` when running in DB mode.
    """
    broker_chunks, latest_chunks, previous_chunks, latest_name, previous_name = (
        list_stock_files_neo4j(ticker)
    )

    latest_docs   = _chunks_to_docs(latest_chunks,   ticker, "transcript", "latest")
    previous_docs = _chunks_to_docs(previous_chunks, ticker, "transcript", "previous")
    broker_docs   = _chunks_to_docs(broker_chunks,   ticker, "broker")

    logger.info(
        "[step1_neo4j] Loaded: %d latest-transcript docs, %d previous-transcript docs, %d broker docs",
        len(latest_docs), len(previous_docs), len(broker_docs),
    )

    transcript_pages = latest_docs + previous_docs
    _latest_name: str = latest_name if latest_name is not None else ""
    _previous_name: str = previous_name if previous_name is not None else ""
    return transcript_pages, broker_docs, _latest_name, _previous_name


# ── Standalone debug run ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"=== Neo4j Step 1: Discover + load for {ticker_arg} ===\n")

    try:
        t_pages, b_pages, latest_n, prev_n = load_neo4j_pages(ticker_arg)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"Latest transcript  : {latest_n}")
    print(f"Previous transcript: {prev_n}")
    print(f"Broker docs        : {len(b_pages)}")
    print(f"Transcript docs    : {len(t_pages)}")

    if t_pages:
        print("\n--- Sample transcript doc metadata ---")
        print(t_pages[0].metadata)
        print(f"  text[:120]: {t_pages[0].page_content[:120]!r}")

    if b_pages:
        print("\n--- Sample broker doc metadata ---")
        print(b_pages[0].metadata)
        print(f"  text[:120]: {b_pages[0].page_content[:120]!r}")
