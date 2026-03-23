"""
Step 3: Parse quality checks + transcript section tagging.

For each loaded Document:
  - Flag blank pages (< MIN_CHARS chars after stripping)
  - Flag garbage pages (high ratio of non-ASCII chars)
  - Tag transcript pages with section: 'prepared_remarks' or 'qa'

Section tagging uses rule-based markers found in earnings call transcripts:
  - QA section begins when a line matches QA_MARKERS (e.g. "question-and-answer",
    "q&a session", "questions from", etc.)
  - Everything before that marker is prepared_remarks.
  - Tags are added to each page's metadata as 'section'.

Broker pages get section = 'broker_report' (no section split needed).

Run:
    python agent_step3_parse_quality.py
"""

import re
import sys
from pathlib import Path

from langchain_core.documents import Document

# Import Step 1/2 helpers
from agents.stock_research_agent.agent_step1_load import list_stock_files, load_pdf_pages, parse_log

# ── Config ────────────────────────────────────────────────────────────────────
TICKER   = "AAPL"   # change to GOOGL, MSFT, NVDA, TSLA
BASE_DIR = Path("data_reports")

MIN_CHARS        = 30    # pages with fewer chars (after strip) are flagged blank
MAX_GARBAGE_RATIO = 0.25  # if >25% of chars are non-ASCII, flag as garbage

# Regex patterns that signal the START of a Q&A section.
# Lowercase match. We check each page's full text and also its first ~500 chars.
QA_MARKERS = [
    # Standalone "Questions And Answers" header (section title)
    r"^questions\s+and\s+answers\s*$",
    # "Question-and-Answer Session" as a heading (NOT preceded by "will be a / there will be")
    r"(?<!will be a )(?<!there will be )(?<!be a )question[\s\-&]*and[\s\-&]*answer\s+session",
    r"\bq&a\b",
    r"\bq\s*&\s*a\s+session\b",
    # "open the floor to questions"
    r"open\s+the\s+floor\s+(to\s+)?questions",
    # "take our first question" or "go ahead and take our first question"
    r"(take|go\s+ahead\s+and\s+take)\s+.*?(our\s+)?first\s+question",
    r"first\s+question\s+(comes?\s+from|is\s+from)",
    r"^\s*operator[:\s]",                    # "Operator:" line starting Q&A
    # Bloomberg transcript: line starting with "Q  -  Analyst Name" (capital letter)
    r"^Q\s+-\s+[A-Z]",
]
QA_PATTERN = re.compile("|".join(QA_MARKERS), re.IGNORECASE | re.MULTILINE)


def _normalize_spaced_text(text: str) -> str:
    """
    Some PDFs extract text with spaces between every character
    (e.g. "Q  -  A m i t" or "$ 1 4 3 . 8  b i l l i o n").
    Collapse runs of single characters/digits separated by ONE space.

    Rules:
      - A "spaced-char run" is 3+ consecutive tokens each of length 1 (alpha or digit or '.'),
        separated by exactly ONE space.
      - Runs are collapsed into a single token.
      - Normal multi-character words are never touched.
    """
    # Tokenize on single spaces (not double spaces, which are word separators in these PDFs)
    # We'll process character by character using a simple state machine on the token list.
    tokens = re.split(r"( )", text)   # split on single spaces, keep as delimiter

    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # Check if this could be part of a spaced-char run:
        # - token is 1 char (alpha, digit, or punctuation like '.' '$' '%')
        # - next token is a single space
        # - token after that is also 1 char
        is_single = len(token) == 1 and (token.isalnum() or token in ".$%,")
        if is_single and i + 2 < len(tokens) and tokens[i + 1] == " ":
            # Peek ahead to see how long the run is
            run = [token]
            j = i + 1
            while j + 1 < len(tokens):
                sep      = tokens[j]
                next_tok = tokens[j + 1]
                if sep == " " and len(next_tok) == 1 and (next_tok.isalnum() or next_tok in ".$%,"):
                    run.append(next_tok)
                    j += 2
                else:
                    break
            if len(run) >= 3:
                result.append("".join(run))
                i = j
                continue
        result.append(token)
        i += 1
    return "".join(result)


# ── Quality checks ────────────────────────────────────────────────────────────

def check_page_quality(doc: Document) -> dict:
    """
    Returns a dict with:
      is_blank   : bool  — fewer than MIN_CHARS printable chars
      is_garbage : bool  — >MAX_GARBAGE_RATIO non-ASCII chars
      char_count : int
      garbage_ratio : float
    """
    text = doc.page_content.strip()
    char_count = len(text)
    if char_count == 0:
        return {"is_blank": True, "is_garbage": False, "char_count": 0, "garbage_ratio": 0.0}

    non_ascii = sum(1 for c in text if ord(c) > 127)
    garbage_ratio = non_ascii / char_count

    return {
        "is_blank":      char_count < MIN_CHARS,
        "is_garbage":    garbage_ratio > MAX_GARBAGE_RATIO,
        "char_count":    char_count,
        "garbage_ratio": round(garbage_ratio, 4),
    }


def flag_quality_issues(pages: list[Document]) -> list[Document]:
    """
    Add quality metadata to each page. Returns same list (mutated in place).
    Also prints a summary of flagged pages.
    """
    blank_count   = 0
    garbage_count = 0

    for doc in pages:
        q = check_page_quality(doc)
        doc.metadata["is_blank"]      = q["is_blank"]
        doc.metadata["is_garbage"]    = q["is_garbage"]
        doc.metadata["char_count"]    = q["char_count"]
        doc.metadata["garbage_ratio"] = q["garbage_ratio"]
        if q["is_blank"]:
            blank_count += 1
        if q["is_garbage"]:
            garbage_count += 1

    print(f"  Quality flags: {blank_count} blank pages, {garbage_count} garbage pages "
          f"(out of {len(pages)} total)")
    return pages


# ── Transcript section tagging ────────────────────────────────────────────────

def tag_transcript_sections(transcript_pages: list[Document]) -> list[Document]:
    """
    Tag each transcript page with section = 'prepared_remarks' or 'qa'.

    Strategy:
      - We process pages in order per (doc_name, period).
      - Once a page contains a QA_MARKER, that page and all subsequent pages
        in the same document are tagged 'qa'.
      - Pages before that are 'prepared_remarks'.
      - If no QA marker is found, all pages are 'prepared_remarks'.
    """
    # Group pages by document
    from collections import defaultdict
    docs_map: dict[str, list[Document]] = defaultdict(list)
    for page in transcript_pages:
        key = page.metadata["doc_name"]
        docs_map[key].append(page)

    for doc_name, pages in docs_map.items():
        in_qa = False
        for page_idx, page in enumerate(pages):
            text = page.page_content
            # Normalize spaced-out characters (PDF extraction artifact) before matching
            normalized = _normalize_spaced_text(text)

            # Only look for Q&A markers after the first 2 pages (page_idx >= 2).
            # The first 1-2 pages always contain operator boilerplate that says
            # "there will be a question-and-answer session" — a forward reference,
            # not the actual Q&A section start.
            if not in_qa and page_idx >= 2 and QA_PATTERN.search(normalized):
                in_qa = True

            page.metadata["section"] = "qa" if in_qa else "prepared_remarks"

        qa_pages   = sum(1 for p in pages if p.metadata["section"] == "qa")
        prep_pages = len(pages) - qa_pages
        print(f"  {doc_name}: {prep_pages} prepared_remarks pages, {qa_pages} qa pages")

    return transcript_pages


DISCLAIMER_PATTERNS = re.compile(
    r"(rating\s+definitions?|important\s+disclosures?|analyst\s+certification|"
    r"conflicts?\s+of\s+interest|regulatory\s+disclosures?|"
    r"required\s+disclosures?|disclosures?\s+appendix|"
    r"research\s+disclosures?|legal\s+disclaimer|"
    r"please\s+refer\s+to\s+.*disclosures?|"
    r"this\s+report\s+has\s+been\s+prepared\s+by|"
    r"for\s+important\s+disclosures|"
    r"member\s+(?:sipc|finra)|finra\s+member|"
    r"non-us\s+analyst\s+disclosure)",
    re.IGNORECASE,
)


def is_disclaimer_page(page: Document) -> bool:
    """
    Return True if this broker page is a boilerplate disclaimer/certification page
    that should be suppressed before retrieval.

    Heuristic: the page must BOTH
      1. match a disclaimer keyword, AND
      2. have low analytical content (< 60 words or no dollar/percent signs)
         OR have the disclaimer keyword in the first 200 chars (i.e. it's a header page).
    """
    text = page.page_content
    normalized = _normalize_spaced_text(text)
    if not DISCLAIMER_PATTERNS.search(normalized):
        return False
    # Extra guard: don't suppress pages that have real analytical content
    has_numbers = bool(re.search(r"[\$\%]|\d+\.\d+x|\bEPS\b|\bP/E\b", normalized, re.IGNORECASE))
    word_count  = len(normalized.split())
    # Suppress if: few words, OR disclaimer is in the first 200 chars (title page)
    first_200 = normalized[:200]
    if DISCLAIMER_PATTERNS.search(first_200):
        return True
    if word_count < 80 and not has_numbers:
        return True
    return False


def tag_broker_sections(broker_pages: list[Document]) -> list[Document]:
    """Broker pages don't need section splitting — tag them uniformly.
    Also flag disclaimer/boilerplate pages so they can be filtered before retrieval."""
    disc_count = 0
    for page in broker_pages:
        page.metadata["section"] = "broker_report"
        is_disc = is_disclaimer_page(page)
        page.metadata["is_disclaimer"] = is_disc
        if is_disc:
            disc_count += 1
    if disc_count:
        print(f"  Flagged {disc_count} disclaimer/boilerplate broker pages (will be excluded from retrieval)")
    return broker_pages


# ── Filtering helpers ─────────────────────────────────────────────────────────

def filter_usable(pages: list[Document]) -> list[Document]:
    """Return only pages that are neither blank, garbage, nor disclaimer boilerplate."""
    return [
        p for p in pages
        if not p.metadata.get("is_blank")
        and not p.metadata.get("is_garbage")
        and not p.metadata.get("is_disclaimer")
    ]


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"=== Step 1+2: Load files for {TICKER} ===\n")
    broker_pdfs, transcript_pdfs = list_stock_files(BASE_DIR, TICKER)

    latest_t   = transcript_pdfs[0]
    previous_t = transcript_pdfs[1]

    print(f"Latest transcript  : {latest_t.name}")
    print(f"Previous transcript: {previous_t.name}")

    transcript_pages  = []
    transcript_pages += load_pdf_pages([latest_t],   TICKER, "transcript", "latest")
    transcript_pages += load_pdf_pages([previous_t], TICKER, "transcript", "previous")
    broker_pages       = load_pdf_pages(broker_pdfs,  TICKER, "broker")

    print(f"\n=== Step 3a: Quality checks ===\n")
    print("  Transcripts:")
    flag_quality_issues(transcript_pages)
    print("  Broker reports:")
    flag_quality_issues(broker_pages)

    print(f"\n=== Step 3b: Section tagging ===\n")
    tag_transcript_sections(transcript_pages)
    tag_broker_sections(broker_pages)

    # Filter out blank/garbage before downstream steps
    usable_transcript = filter_usable(transcript_pages)
    usable_broker     = filter_usable(broker_pages)

    print(f"\n=== Summary ===")
    print(f"Transcript pages: {len(transcript_pages)} total → {len(usable_transcript)} usable")
    print(f"Broker pages    : {len(broker_pages)} total → {len(usable_broker)} usable")

    # Show section distribution
    from collections import Counter
    t_sections = Counter(p.metadata["section"] for p in usable_transcript)
    print(f"\nTranscript section distribution: {dict(t_sections)}")

    # Spot-check
    print("\n--- Sample transcript page (prepared_remarks) ---")
    prep_sample = next((p for p in usable_transcript if p.metadata["section"] == "prepared_remarks"), None)
    if prep_sample:
        print(f"  doc: {prep_sample.metadata['doc_name']}  page: {prep_sample.metadata['page_number']}")
        print(f"  text[:200]: {prep_sample.page_content[:200]!r}")

    print("\n--- Sample transcript page (qa) ---")
    qa_sample = next((p for p in usable_transcript if p.metadata["section"] == "qa"), None)
    if qa_sample:
        print(f"  doc: {qa_sample.metadata['doc_name']}  page: {qa_sample.metadata['page_number']}")
        print(f"  text[:200]: {qa_sample.page_content[:200]!r}")
