"""
Step 4: Deterministic broker label extraction.

For each broker report document, scan the full concatenated text for
rating keywords and map them to: bullish / neutral / bearish.

Keyword mapping:
  bullish  : Buy, Overweight, Outperform, Strong Buy, Positive, Accumulate
  neutral  : Hold, Neutral, Market Perform, Equal Weight, In Line
  bearish  : Sell, Underweight, Underperform, Reduce, Negative

Strategy:
  - Concatenate all pages of a single broker document.
  - Search for rating keywords near "rating", "recommendation", "we rate",
    "our view", or standalone at the start of a sentence / in a table.
  - Count keyword hits for each category.
  - Winning category (most hits) becomes the rating.
  - Tie or zero hits → flag as 'unknown'.

Output per broker doc:
  {
    "doc_name": "...",
    "ticker": "...",
    "rating": "bullish" | "neutral" | "bearish" | "unknown",
    "rating_raw": ["Buy", ...],   # all matched keywords
    "hit_counts": {"bullish": N, "neutral": N, "bearish": N},
  }

Run:
    python agent_step4_broker_labels.py
"""

import re
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document

from agents.stock_research_agent.agent_step1_load import list_stock_files, load_pdf_pages, parse_log
from agents.stock_research_agent.agent_step3_parse_quality import flag_quality_issues, filter_usable

# ── Config ────────────────────────────────────────────────────────────────────
TICKER   = "AAPL"
BASE_DIR = Path("data_reports")

# ── Rating keyword maps ───────────────────────────────────────────────────────
RATING_KEYWORDS: dict[str, list[str]] = {
    "bullish": [
        "strong buy", "buy", "overweight", "outperform", "positive",
        "accumulate", "add", "top pick",
    ],
    "neutral": [
        "hold", "neutral", "market perform", "marketperform",
        "equal weight", "equal-weight", "in line", "inline", "sector perform",
        "sector weight",
    ],
    "bearish": [
        "sell", "underweight", "underperform", "reduce", "negative",
        "below average",
    ],
}

# Build a flat pattern: each keyword mapped to its category.
# We sort by length descending so "strong buy" matches before "buy".
_ALL_KEYWORDS: list[tuple[str, str]] = []
for category, words in RATING_KEYWORDS.items():
    for word in sorted(words, key=len, reverse=True):
        _ALL_KEYWORDS.append((word, category))

# Contexts that raise confidence a nearby keyword is a rating statement.
RATING_CONTEXT_PATTERN = re.compile(
    r"(rating|recommendation|reiterat|we\s+rate|our\s+view|price\s+target|pt\b|"
    r"initiating|initiate|maintain|upgrade|downgrade|reiterating)",
    re.IGNORECASE,
)


def _find_rating_hits(text: str) -> dict:
    """
    Scan text for rating keywords. Returns hit_counts and raw matched words.
    
    Two-pass approach:
      Pass 1: Find keywords in 'high-confidence' windows (within 80 chars of
              a rating-context phrase). These count double.
      Pass 2: Find standalone keywords elsewhere (count once).
    """
    text_lower = text.lower()
    hit_counts = {"bullish": 0, "neutral": 0, "bearish": 0}
    raw_hits:  list[str] = []

    # Find all context anchor positions
    context_positions = {m.start() for m in RATING_CONTEXT_PATTERN.finditer(text)}

    for keyword, category in _ALL_KEYWORDS:
        # Find all positions of this keyword (whole-word match)
        pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(text_lower):
            raw_hits.append(m.group())
            # Check proximity to any context anchor
            near_context = any(abs(m.start() - cp) <= 80 for cp in context_positions)
            hit_counts[category] += 2 if near_context else 1

    return {"hit_counts": hit_counts, "raw_hits": raw_hits}


def extract_broker_label(doc_pages: list[Document]) -> dict:
    """
    Given all pages from a single broker document, return a rating dict.
    """
    doc_name = doc_pages[0].metadata["doc_name"]
    ticker   = doc_pages[0].metadata["ticker"]

    # Concatenate full document text
    full_text = "\n".join(p.page_content for p in doc_pages)

    result = _find_rating_hits(full_text)
    hit_counts = result["hit_counts"]
    raw_hits   = list(dict.fromkeys(result["raw_hits"]))  # deduplicate, preserve order

    # Determine winner
    max_hits = max(hit_counts.values())
    if max_hits == 0:
        rating = "unknown"
    else:
        winners = [cat for cat, cnt in hit_counts.items() if cnt == max_hits]
        rating = winners[0] if len(winners) == 1 else "unknown"  # tie → unknown

    return {
        "doc_name":   doc_name,
        "ticker":     ticker,
        "rating":     rating,
        "rating_raw": raw_hits,
        "hit_counts": hit_counts,
    }


def extract_all_broker_labels(broker_pages: list[Document]) -> list[dict]:
    """
    Group broker pages by doc_name, then extract rating for each doc.
    Returns list of rating dicts sorted by doc_name.
    """
    # Group pages by document
    docs_map: dict[str, list[Document]] = defaultdict(list)
    for page in broker_pages:
        docs_map[page.metadata["doc_name"]].append(page)

    results = []
    for doc_name, pages in sorted(docs_map.items()):
        label = extract_broker_label(pages)
        results.append(label)

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"=== Steps 1-3: Load + quality check for {TICKER} ===\n")
    broker_pdfs, transcript_pdfs = list_stock_files(BASE_DIR, TICKER)

    broker_pages = load_pdf_pages(broker_pdfs, TICKER, "broker")
    flag_quality_issues(broker_pages)
    usable_broker = filter_usable(broker_pages)

    print(f"\n=== Step 4: Broker label extraction ===\n")
    labels = extract_all_broker_labels(usable_broker)

    for lbl in labels:
        flag = " *** UNKNOWN ***" if lbl["rating"] == "unknown" else ""
        print(f"  [{lbl['rating'].upper():8s}]{flag}  {lbl['doc_name']}")
        print(f"             hits={lbl['hit_counts']}  raw={lbl['rating_raw'][:8]}")

    print(f"\nSummary: {len(labels)} broker docs")
    from collections import Counter
    rating_dist = Counter(l["rating"] for l in labels)
    print(f"  Distribution: {dict(rating_dist)}")
    unknowns = [l["doc_name"] for l in labels if l["rating"] == "unknown"]
    if unknowns:
        print(f"  Unknown ratings (review manually): {unknowns}")
