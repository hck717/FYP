"""
Step 5: Deterministic transcript feature extraction.

Extracts interpretable signals from transcript pages WITHOUT using an LLM.
These features go into the evidence pack for DeepSeek in Step 7.

Features extracted per document (latest vs previous transcript):
  1. KPI mentions        — counts of key financial metric keywords per section
  2. Hedge word counts   — "may", "could", "might", "potentially", etc.
  3. Certainty words     — "will", "expect", "confident", "committed", etc.
  4. Hedge/certainty ratio — hedge / (hedge + certainty), higher = more cautious
  5. "but"/"however" counts — mid-sentence pivots suggesting caveats
  6. Guidance ranges     — regex extraction of forward-looking "$X–$Y" or "X% to Y%"
  7. Dropped KPIs        — KPIs mentioned in previous transcript but NOT in latest
  8. Prep vs QA tone     — hedge ratio split between prepared_remarks and qa sections
  9. Q&A evasiveness     — phrases like "I'll get back to you", "not going to guide",
                           "decline to comment", "as I mentioned" (deflection)

All counts are normalized per 1000 words for comparability.

Output: a dict per ticker / period summarising the above features.

Run:
    python agent_step5_transcript_features.py
"""

import re
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document

from agent_step1_load import list_stock_files, load_pdf_pages
from agent_step3_parse_quality import (
    flag_quality_issues, tag_transcript_sections, filter_usable,
    _normalize_spaced_text,
)

# ── Config ────────────────────────────────────────────────────────────────────
TICKER   = "AAPL"
BASE_DIR = Path("data_reports")

# ── Word lists ────────────────────────────────────────────────────────────────

KPI_KEYWORDS: list[str] = [
    # Revenue / growth
    "revenue", "net revenue", "total revenue", "sales", "growth",
    # Profit
    "gross margin", "gross profit", "operating income", "operating margin",
    "net income", "earnings", "ebitda", "ebit",
    # Per-share
    "eps", "earnings per share", "diluted eps",
    # Cash
    "free cash flow", "cash flow", "capex", "capital expenditure",
    # Guidance
    "guidance", "outlook", "forecast", "full year", "next quarter",
    # Segment / product
    "services", "cloud", "hardware", "subscription", "active users",
    "monthly active", "daily active", "units sold", "backlog",
    # Returns
    "buyback", "dividend", "share repurchase",
]

HEDGE_WORDS: list[str] = [
    "may", "might", "could", "potentially", "possibly", "uncertain",
    "uncertainty", "subject to", "risk", "risks", "headwind", "headwinds",
    "challenging", "difficult", "volatility", "volatile", "approximate",
    "approximately", "around", "roughly", "unclear", "depend", "depends",
    "depending", "if", "assuming", "assumption",
]

CERTAINTY_WORDS: list[str] = [
    "will", "expect", "expects", "expected", "confident", "confidence",
    "committed", "commit", "strong", "strongly", "clear", "clearly",
    "definite", "definitely", "sure", "certainly", "certain", "plan",
    "plans", "planned", "on track", "accelerat",
]

EVASIVE_PHRASES: list[str] = [
    "not going to guide",
    "don't guide",
    "we don't provide",
    "decline to",
    "not in a position",
    "as i mentioned",
    "as we've said",
    "i'll get back to you",
    "i'll follow up",
    "we'll discuss",
    "stay tuned",
    "not going to comment",
    "can't comment",
    "cannot comment",
    "no further details",
    "we're not going to break out",
    "we don't break out",
    "not going to break that out",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


def _count_keywords(text: str, keywords: list[str]) -> dict[str, int]:
    """Count occurrences of each keyword/phrase (case-insensitive, whole word)."""
    text_lower = text.lower()
    counts = {}
    for kw in keywords:
        pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
        counts[kw] = len(pattern.findall(text_lower))
    return counts


def _count_phrases(text: str, phrases: list[str]) -> int:
    """Count total occurrences of a list of phrases."""
    text_lower = text.lower()
    return sum(text_lower.count(p) for p in phrases)


def _extract_guidance_ranges(text: str) -> list[str]:
    """
    Extract guidance range strings like:
      "$1.2B to $1.4B", "10% to 12%", "$45–$50", "1.2 billion to 1.4 billion"
    Returns list of matched strings (deduplicated).
    """
    patterns = [
        r"\$[\d,\.]+\s*(?:billion|million|B|M)?\s*(?:to|–|-)\s*\$[\d,\.]+\s*(?:billion|million|B|M)?",
        r"[\d\.]+\s*%\s*(?:to|–|-)\s*[\d\.]+\s*%",
        r"[\d\.]+\s*(?:billion|million)\s*(?:to|–|-)\s*[\d\.]+\s*(?:billion|million)",
    ]
    combined = re.compile("|".join(patterns), re.IGNORECASE)
    matches = combined.findall(text)
    return list(dict.fromkeys(m.strip() for m in matches if m.strip()))


def _hedge_certainty_ratio(hedge_total: int, certainty_total: int) -> float:
    """
    Returns hedge / (hedge + certainty).
    0.0 = fully certain, 1.0 = fully hedged. Returns 0.5 if both zero.
    """
    total = hedge_total + certainty_total
    if total == 0:
        return 0.5
    return round(hedge_total / total, 4)


# ── Core feature extractor ────────────────────────────────────────────────────

def extract_transcript_features(pages: list[Document], doc_name: str) -> dict:
    """
    Given all usable pages from ONE transcript document, return a feature dict.
    Pages must already have 'section' metadata (prepared_remarks / qa).
    """
    # Split pages by section
    prep_pages = [p for p in pages if p.metadata.get("section") == "prepared_remarks"]
    qa_pages   = [p for p in pages if p.metadata.get("section") == "qa"]

    prep_text = " ".join(_normalize_spaced_text(p.page_content) for p in prep_pages)
    qa_text   = " ".join(_normalize_spaced_text(p.page_content) for p in qa_pages)
    full_text = prep_text + " " + qa_text

    # Word counts
    prep_wc  = _word_count(prep_text)
    qa_wc    = _word_count(qa_text)
    total_wc = _word_count(full_text)

    # KPI mentions (full doc)
    kpi_counts = _count_keywords(full_text, KPI_KEYWORDS)
    kpi_total  = sum(kpi_counts.values())
    kpi_per1k  = round(kpi_total / max(total_wc, 1) * 1000, 2)

    # Hedge / certainty — full doc
    hedge_counts    = _count_keywords(full_text, HEDGE_WORDS)
    certainty_counts = _count_keywords(full_text, CERTAINTY_WORDS)
    hedge_total      = sum(hedge_counts.values())
    certainty_total  = sum(certainty_counts.values())
    hedge_ratio      = _hedge_certainty_ratio(hedge_total, certainty_total)

    # Hedge / certainty — prepared remarks only
    prep_hedge      = sum(_count_keywords(prep_text, HEDGE_WORDS).values())
    prep_certainty  = sum(_count_keywords(prep_text, CERTAINTY_WORDS).values())
    prep_hedge_ratio = _hedge_certainty_ratio(prep_hedge, prep_certainty)

    # Hedge / certainty — Q&A only
    qa_hedge        = sum(_count_keywords(qa_text, HEDGE_WORDS).values())
    qa_certainty    = sum(_count_keywords(qa_text, CERTAINTY_WORDS).values())
    qa_hedge_ratio  = _hedge_certainty_ratio(qa_hedge, qa_certainty)

    # "but" / "however" counts
    but_count     = len(re.findall(r"\bbut\b", full_text, re.IGNORECASE))
    however_count = len(re.findall(r"\bhowever\b", full_text, re.IGNORECASE))

    # Guidance ranges
    guidance_ranges = _extract_guidance_ranges(full_text)

    # Q&A evasiveness
    evasive_count = _count_phrases(qa_text, EVASIVE_PHRASES)

    return {
        "doc_name":          doc_name,
        "word_count":        total_wc,
        "prep_word_count":   prep_wc,
        "qa_word_count":     qa_wc,
        # KPIs
        "kpi_total":         kpi_total,
        "kpi_per_1k_words":  kpi_per1k,
        "kpi_counts":        kpi_counts,
        # Tone
        "hedge_total":       hedge_total,
        "certainty_total":   certainty_total,
        "hedge_ratio":       hedge_ratio,         # full doc
        "prep_hedge_ratio":  prep_hedge_ratio,    # prepared remarks only
        "qa_hedge_ratio":    qa_hedge_ratio,      # Q&A only
        "qa_vs_prep_hedge_delta": round(qa_hedge_ratio - prep_hedge_ratio, 4),
        # Pivots
        "but_count":         but_count,
        "however_count":     however_count,
        "pivot_per_1k_words": round((but_count + however_count) / max(total_wc, 1) * 1000, 2),
        # Guidance
        "guidance_ranges":   guidance_ranges,
        "guidance_range_count": len(guidance_ranges),
        # Evasiveness
        "evasive_count":     evasive_count,
    }


def compare_kpi_coverage(latest_features: dict, prev_features: dict) -> dict:
    """
    Find KPIs mentioned in previous transcript but NOT in latest (dropped KPIs)
    and vice versa (new KPIs).
    """
    latest_mentioned = {kw for kw, cnt in latest_features["kpi_counts"].items() if cnt > 0}
    prev_mentioned   = {kw for kw, cnt in prev_features["kpi_counts"].items()  if cnt > 0}

    dropped  = sorted(prev_mentioned - latest_mentioned)
    added    = sorted(latest_mentioned - prev_mentioned)
    retained = sorted(latest_mentioned & prev_mentioned)

    return {
        "dropped_kpis":  dropped,
        "added_kpis":    added,
        "retained_kpis": retained,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from agent_step3_parse_quality import tag_broker_sections

    print(f"=== Steps 1-3: Load + quality + section tag for {TICKER} ===\n")
    broker_pdfs, transcript_pdfs = list_stock_files(BASE_DIR, TICKER)
    latest_t   = transcript_pdfs[0]
    previous_t = transcript_pdfs[1]

    transcript_pages  = []
    transcript_pages += load_pdf_pages([latest_t],   TICKER, "transcript", "latest")
    transcript_pages += load_pdf_pages([previous_t], TICKER, "transcript", "previous")

    flag_quality_issues(transcript_pages)
    tag_transcript_sections(transcript_pages)
    usable = filter_usable(transcript_pages)

    # Split by period
    latest_pages   = [p for p in usable if p.metadata["period"] == "latest"]
    previous_pages = [p for p in usable if p.metadata["period"] == "previous"]

    print(f"\n=== Step 5: Transcript feature extraction ===\n")
    latest_feat   = extract_transcript_features(latest_pages,   latest_t.name)
    previous_feat = extract_transcript_features(previous_pages, previous_t.name)

    def _print_features(feat: dict):
        print(f"  doc         : {feat['doc_name']}")
        print(f"  words       : total={feat['word_count']}  prep={feat['prep_word_count']}  qa={feat['qa_word_count']}")
        print(f"  KPI total   : {feat['kpi_total']}  ({feat['kpi_per_1k_words']} per 1k words)")
        print(f"  Hedge ratio : {feat['hedge_ratio']}  (prep={feat['prep_hedge_ratio']} | qa={feat['qa_hedge_ratio']}) delta={feat['qa_vs_prep_hedge_delta']}")
        print(f"  Pivots(but/however): {feat['but_count']}/{feat['however_count']}  ({feat['pivot_per_1k_words']} per 1k words)")
        print(f"  Guidance ranges ({feat['guidance_range_count']}): {feat['guidance_ranges'][:5]}")
        print(f"  Evasive phrases (qa): {feat['evasive_count']}")
        # Top KPIs
        top_kpis = sorted(feat["kpi_counts"].items(), key=lambda x: x[1], reverse=True)[:8]
        print(f"  Top KPIs    : {top_kpis}")

    print("--- Latest transcript ---")
    _print_features(latest_feat)
    print("\n--- Previous transcript ---")
    _print_features(previous_feat)

    print("\n--- KPI coverage comparison ---")
    comparison = compare_kpi_coverage(latest_feat, previous_feat)
    print(f"  Dropped KPIs : {comparison['dropped_kpis']}")
    print(f"  Added KPIs   : {comparison['added_kpis']}")
    print(f"  Retained     : {comparison['retained_kpis']}")
