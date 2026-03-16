"""
Step 4b: Deterministic per-broker structured data extraction.

For each broker document, extract via regex:
  - rating          : bullish / neutral / bearish / unknown  (from Step 4)
  - price_target    : first dollar amount near "price target" / "PT" / "target price"
  - current_price   : stock price mentioned near "current" / "price" at time of report
  - upside_pct      : calculated (price_target / current_price - 1) if both found
  - eps_estimates   : list of EPS values mentioned (e.g. FY25E, FY26E)
  - valuation_note  : short snippet near P/E, EV/EBITDA, P/S multiples
  - key_risks       : first 200-char snippet near "risk" / "downside" / "bear case"

All fields are null if not found — explicit and honest.

Output (per doc):
  {
    "doc_name":        str,
    "ticker":          str,
    "rating":          str,
    "price_target":    float | None,
    "current_price":   float | None,
    "upside_pct":      float | None,   # e.g. 0.15 = +15%
    "eps_estimates":   list[str],      # raw matched strings
    "valuation_note":  str | None,     # snippet
    "key_risks":       str | None,     # snippet
    "page_refs":       dict,           # which page each field was found on
  }

Run:
    python agent_step4b_broker_parse.py
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document

from agent_step1_load import list_stock_files, load_pdf_pages
from agent_step3_parse_quality import (
    flag_quality_issues, tag_broker_sections, filter_usable,
    _normalize_spaced_text,
)
from agent_step4_broker_labels import extract_all_broker_labels

# ── Config ────────────────────────────────────────────────────────────────────
TICKER   = "AAPL"
BASE_DIR = Path("data_reports")

# ── Regex patterns ─────────────────────────────────────────────────────────────

# Price target: "$123", "$1,234", "$123.45" near "price target" / "PT" / "target price" / "target"
_PT_CONTEXT = re.compile(
    r"(?:price\s+target|target\s+price|\bPT\b|12[-\s]?month\s+(?:price\s+)?target|"
    r"our\s+target|raise[sd]?\s+(?:our\s+)?(?:price\s+)?target|"
    r"lower[sd]?\s+(?:our\s+)?(?:price\s+)?target|"
    r"maintain[s]?\s+(?:a\s+)?(?:price\s+)?target)\s*(?:of|to|at|from)?\s*"
    r"\$?\s*([\d,]+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
# Standalone dollar amount on same line as "price target" keyword (fallback)
_PT_FALLBACK = re.compile(
    r"(?:price\s+target|target\s+price|\bPT\b)[^\n$]{0,40}\$\s*([\d,]+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)

# Current / stock price near report date
_CURRENT_PRICE = re.compile(
    r"(?:current\s+(?:share\s+)?price|closing\s+price|stock\s+(?:price|trades?)|"
    r"shares?\s+(?:trading|trade[sd]?)\s+at|price\s+as\s+of)\s*[:\-–]?\s*\$?\s*([\d,]+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)

# EPS estimates: patterns like "$1.23", "EPS of $1.23", "FY25E EPS $1.23", "FY2025E $5.67"
_EPS_CONTEXT = re.compile(
    r"(?:(?:FY|CY|Q)\d{2,4}[E]?\s+)?(?:EPS|earnings\s+per\s+share|diluted\s+EPS)"
    r"[^\n$]{0,30}\$?\s*([\d,]+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_EPS_YEAR = re.compile(
    r"(?:FY|CY|fiscal\s+year)\s*(\d{2,4})[Ee]?\s+(?:EPS|earnings)\s*[:\-–=]?\s*\$?\s*([\d,]+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)

# Valuation multiples: P/E, EV/EBITDA, P/S, NTM multiples
_VALUATION = re.compile(
    r"(?:P/E|price[- /]to[- ]earnings|EV/EBITDA|EV/Sales|P/S|NTM\s+P/E|"
    r"forward\s+P/E|fwd\s+P/E|multiple|valuation)[^\n]{0,120}",
    re.IGNORECASE,
)

# Risk snippets: first occurrence near "risk" / "downside"
_RISK = re.compile(
    r"(?:key\s+risks?|downside\s+risks?|bear\s+case|risks?\s+(?:include|to\s+our)|"
    r"risks?\s+to\s+the\s+(?:thesis|view|rating))[^\n]{0,250}",
    re.IGNORECASE,
)


def _clean_number(raw: str) -> float | None:
    """Parse a matched number string like '1,234.56' → 1234.56."""
    try:
        return float(raw.replace(",", ""))
    except (ValueError, TypeError):
        return None


def _extract_price_target(pages: list[Document]) -> tuple[float | None, int | None]:
    """Return (price_target, page_number) or (None, None)."""
    for page in pages:
        text = _normalize_spaced_text(page.page_content)
        m = _PT_CONTEXT.search(text)
        if not m:
            m = _PT_FALLBACK.search(text)
        if m:
            val = _clean_number(m.group(1))
            if val and 1 < val < 10_000:   # sanity range for stock prices
                return val, page.metadata.get("page_number")
    return None, None


def _extract_current_price(pages: list[Document]) -> tuple[float | None, int | None]:
    """Return (current_price, page_number) or (None, None)."""
    for page in pages:
        text = _normalize_spaced_text(page.page_content)
        m = _CURRENT_PRICE.search(text)
        if m:
            val = _clean_number(m.group(1))
            if val and 1 < val < 10_000:
                return val, page.metadata.get("page_number")
    return None, None


def _extract_eps_estimates(pages: list[Document]) -> list[str]:
    """Return list of raw EPS snippets found across all pages."""
    seen: set[str] = set()
    results: list[str] = []
    for page in pages:
        text = _normalize_spaced_text(page.page_content)
        # Prefer labelled year matches
        for m in _EPS_YEAR.finditer(text):
            label = f"FY{m.group(1)}E ${m.group(2)}"
            if label not in seen:
                seen.add(label)
                results.append(label)
        # Fallback: generic EPS mentions
        for m in _EPS_CONTEXT.finditer(text):
            snippet = m.group(0).strip()[:60]
            if snippet not in seen:
                seen.add(snippet)
                results.append(snippet)
    return results[:8]   # cap at 8 to keep output manageable


def _extract_valuation_note(pages: list[Document]) -> tuple[str | None, int | None]:
    """Return (first valuation snippet, page_number) or (None, None)."""
    for page in pages:
        text = _normalize_spaced_text(page.page_content)
        m = _VALUATION.search(text)
        if m:
            snippet = m.group(0).strip()[:200]
            return snippet, page.metadata.get("page_number")
    return None, None


def _extract_key_risks(pages: list[Document]) -> tuple[str | None, int | None]:
    """Return (first risk snippet, page_number) or (None, None)."""
    for page in pages:
        text = _normalize_spaced_text(page.page_content)
        m = _RISK.search(text)
        if m:
            snippet = m.group(0).strip()[:250]
            return snippet, page.metadata.get("page_number")
    return None, None


# ── Main extractor ─────────────────────────────────────────────────────────────

def parse_broker_doc(pages: list[Document], rating: str) -> dict:
    """
    Extract structured fields from all pages of one broker document.
    Returns a fully populated dict with None for any field not found.
    """
    doc_name = pages[0].metadata["doc_name"]
    ticker   = pages[0].metadata["ticker"]

    pt, pt_page          = _extract_price_target(pages)
    cp, cp_page          = _extract_current_price(pages)
    eps                  = _extract_eps_estimates(pages)
    val_note, val_page   = _extract_valuation_note(pages)
    risks, risk_page     = _extract_key_risks(pages)

    upside = None
    if pt is not None and cp is not None and cp > 0:
        upside = round((pt / cp) - 1, 4)

    return {
        "doc_name":       doc_name,
        "ticker":         ticker,
        "rating":         rating,
        "price_target":   pt,
        "current_price":  cp,
        "upside_pct":     upside,
        "eps_estimates":  eps,
        "valuation_note": val_note,
        "key_risks":      risks,
        "page_refs": {
            "price_target":   pt_page,
            "current_price":  cp_page,
            "valuation_note": val_page,
            "key_risks":      risk_page,
        },
    }


def parse_all_broker_docs(
    broker_pages: list[Document],
    broker_labels: list[dict],
) -> list[dict]:
    """
    Group broker pages by doc_name, match with rating labels, and parse each doc.
    Returns list of structured dicts sorted by doc_name.
    """
    # Build rating lookup
    rating_map = {l["doc_name"]: l["rating"] for l in broker_labels}

    # Group pages by document (only non-disclaimer pages should arrive here)
    docs_map: dict[str, list[Document]] = defaultdict(list)
    for page in broker_pages:
        docs_map[page.metadata["doc_name"]].append(page)

    results = []
    for doc_name, pages in sorted(docs_map.items()):
        rating = rating_map.get(doc_name, "unknown")
        parsed = parse_broker_doc(pages, rating)
        results.append(parsed)

    return results


# ── Formatting helpers ─────────────────────────────────────────────────────────

def format_broker_table_md(broker_parsed: list[dict]) -> str:
    """Render parsed broker data as a Markdown table."""
    header = (
        "| Document | Rating | Price Target | Current | Upside | EPS Estimates | Valuation | Key Risk |\n"
        "|----------|--------|-------------:|--------:|-------:|---------------|-----------|----------|\n"
    )
    rows = []
    for b in broker_parsed:
        pt_str  = f"${b['price_target']:.0f}"  if b["price_target"]  else "—"
        cp_str  = f"${b['current_price']:.0f}" if b["current_price"] else "—"
        up_str  = f"{b['upside_pct']:+.1%}"    if b["upside_pct"] is not None else "—"
        eps_str = "; ".join(b["eps_estimates"][:3]) if b["eps_estimates"] else "—"
        val_str = (b["valuation_note"][:60] + "…") if b["valuation_note"] else "—"
        risk_str= (b["key_risks"][:60] + "…")       if b["key_risks"]      else "—"
        # Rating badge
        badge = {"bullish": "**BULL**", "neutral": "NEUT", "bearish": "*BEAR*"}.get(
            b["rating"], b["rating"].upper()
        )
        rows.append(
            f"| {b['doc_name'][:45]} | {badge} | {pt_str} | {cp_str} | {up_str} | "
            f"{eps_str} | {val_str} | {risk_str} |"
        )
    return header + "\n".join(rows)


def format_broker_summary_for_llm(broker_parsed: list[dict]) -> str:
    """
    Compact text representation of parsed broker data for injection into LLM prompt.
    One block per broker doc.
    """
    parts = []
    for b in broker_parsed:
        lines = [f"--- {b['doc_name']} | Rating: {b['rating'].upper()} ---"]
        if b["price_target"]:
            lines.append(f"  Price Target: ${b['price_target']:.2f}")
        if b["current_price"]:
            lines.append(f"  Current Price: ${b['current_price']:.2f}")
        if b["upside_pct"] is not None:
            lines.append(f"  Implied Upside: {b['upside_pct']:+.1%}")
        if b["eps_estimates"]:
            lines.append(f"  EPS Estimates: {'; '.join(b['eps_estimates'][:4])}")
        if b["valuation_note"]:
            pg = b["page_refs"]["valuation_note"]
            lines.append(f"  Valuation [{b['doc_name']} p.{pg}]: {b['valuation_note'][:120]}")
        if b["key_risks"]:
            pg = b["page_refs"]["key_risks"]
            lines.append(f"  Key Risks [{b['doc_name']} p.{pg}]: {b['key_risks'][:120]}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"=== Steps 1-3: Load + quality check for {TICKER} ===\n")
    broker_pdfs, transcript_pdfs = list_stock_files(BASE_DIR, TICKER)

    broker_pages_raw = load_pdf_pages(broker_pdfs, TICKER, "broker")
    flag_quality_issues(broker_pages_raw)
    tag_broker_sections(broker_pages_raw)
    usable_broker = filter_usable(broker_pages_raw)

    print(f"\n=== Step 4: Broker labels ===\n")
    from agent_step4_broker_labels import extract_all_broker_labels
    # Labels use ALL pages (before disclaimer filter) to maximise keyword hits
    # but pass ALL broker pages (raw) for label extraction
    all_broker_nondisclaimer = [p for p in broker_pages_raw
                                if not p.metadata.get("is_blank")
                                and not p.metadata.get("is_garbage")]
    labels = extract_all_broker_labels(all_broker_nondisclaimer)

    print(f"\n=== Step 4b: Deterministic broker parsing ===\n")
    parsed = parse_all_broker_docs(usable_broker, labels)

    for b in parsed:
        print(f"\n  {b['doc_name']}")
        print(f"    Rating    : {b['rating']}")
        print(f"    PT        : {b['price_target']}  |  Current: {b['current_price']}  |  Upside: {b['upside_pct']}")
        print(f"    EPS       : {b['eps_estimates'][:3]}")
        print(f"    Val note  : {str(b['valuation_note'])[:100]}")
        print(f"    Risk snip : {str(b['key_risks'])[:100]}")

    print(f"\n\n=== Markdown Table ===\n")
    print(format_broker_table_md(parsed))
