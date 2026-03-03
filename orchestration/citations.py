"""Citation extraction and formatting for the orchestration summarizer.

Pulls structured citations from all agent outputs and renders them
into a numbered reference list that the summarizer appends to its narrative.

Citation sources by agent:
  business_analyst    → Qdrant vector chunks  (qdrant::TICKER::slug)
                      → Neo4j graph chunks    (neo4j::TICKER::section::n)
                      → PostgreSQL sentiment  (postgresql:sentiment_trends)
  quant_fundamental   → PostgreSQL tables:   raw_fundamentals, raw_timeseries,
                                              market_eod_us, ratios_ttm, etc.
  web_search          → Live URLs from Perplexity Sonar (breaking_news, risk_flags,
                         competitor_signals, raw_citations)
  financial_modelling → PostgreSQL DCF inputs, raw_timeseries (technicals),
                         raw_fundamentals (earnings/dividends/factor scores),
                         Neo4j peer group (Comps)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ── Human-readable database labels ────────────────────────────────────────────

_DB_LABELS: Dict[str, str] = {
    "qdrant":                       "Qdrant vector store (news/filings)",
    "neo4j":                        "Neo4j knowledge graph",
    "postgresql":                   "PostgreSQL",
    "postgresql:sentiment_trends":  "PostgreSQL · sentiment_trends table",
    "postgresql:raw_fundamentals":  "PostgreSQL · raw_fundamentals table",
    "postgresql:raw_timeseries":    "PostgreSQL · raw_timeseries (price history)",
    "postgresql:market_eod_us":     "PostgreSQL · market_eod_us (benchmark)",
    "postgresql:ratios_ttm":        "PostgreSQL · ratios_ttm",
    "postgresql:key_metrics_ttm":   "PostgreSQL · key_metrics_ttm",
    "postgresql:financial_scores":  "PostgreSQL · financial_scores",
}

_SECTION_LABELS: Dict[str, str] = {
    "income_statement":  "Income Statement",
    "balance_sheet":     "Balance Sheet",
    "cash_flow":         "Cash Flow Statement",
    "risk_factors":      "Risk Factors (filing)",
    "mda":               "Management Discussion & Analysis",
    "earnings":          "Earnings Release",
    "press_release":     "Press Release",
    "news":              "News Article",
}


# ── Citation dataclass ─────────────────────────────────────────────────────────

class Citation:
    """A single resolved citation with display metadata."""

    __slots__ = ("index", "source_agent", "db", "label", "detail", "url", "chunk_id")

    def __init__(
        self,
        index: int,
        source_agent: str,
        db: str,
        label: str,
        detail: str = "",
        url: str = "",
        chunk_id: str = "",
    ) -> None:
        self.index        = index
        self.source_agent = source_agent
        self.db           = db
        self.label        = label
        self.detail       = detail
        self.url          = url
        self.chunk_id     = chunk_id

    def ref(self) -> str:
        """Short inline reference, e.g. '[1]'."""
        return f"[{self.index}]"

    def footnote(self) -> str:
        """Full footnote line for the References section."""
        parts = [f"[{self.index}]", self.label]
        if self.detail:
            parts.append(f"— {self.detail}")
        if self.url:
            parts.append(f"({self.url})")
        parts.append(f"[{self.source_agent}]")
        return "  " + " ".join(parts)


# ── Extraction helpers ─────────────────────────────────────────────────────────

def _chunk_id_to_label(chunk_id: str, ticker: Optional[str] = None) -> Tuple[str, str, str]:
    """Parse a chunk_id string into (db, label, detail).

    Formats handled:
      qdrant::AAPL::Apple_Reports_Record_Q  →  Qdrant, "Apple Reports Record Q", news
      neo4j::AAPL::risk_factors::0          →  Neo4j,  "Risk Factors (filing)",  AAPL
      neo4j::AAPL::mda::2                   →  Neo4j,  "MDA",                    AAPL
    """
    if not chunk_id:
        return "unknown", chunk_id, ""

    parts = chunk_id.split("::")
    if len(parts) < 2:
        return "unknown", chunk_id, ""

    backend = parts[0].lower()  # "qdrant" or "neo4j"

    if backend == "qdrant":
        # qdrant::TICKER::slug
        tk     = parts[1] if len(parts) > 1 else (ticker or "")
        slug   = parts[2] if len(parts) > 2 else ""
        # Convert slug to readable title (underscores → spaces, trim to 60 chars)
        title  = slug.replace("_", " ").replace("-", " ").strip()[:60]
        label  = title or f"{tk} document"
        detail = f"{tk} · Qdrant vector store"
        return "qdrant", label, detail

    if backend == "neo4j":
        # neo4j::TICKER::section::n
        tk      = parts[1] if len(parts) > 1 else (ticker or "")
        section = parts[2] if len(parts) > 2 else "filing"
        sec_label = _SECTION_LABELS.get(section, section.replace("_", " ").title())
        label   = f"{tk} — {sec_label}"
        detail  = "Neo4j knowledge graph"
        return "neo4j", label, detail

    return backend, chunk_id, ""


def _extract_inline_chunk_ids(text: str) -> List[str]:
    """Find all [qdrant::...] and [neo4j::...] inline citation tokens in a string."""
    return re.findall(r'\[(qdrant::[^\]]+|neo4j::[^\]]+)\]', text or "")


def _walk_and_collect_chunk_ids(obj: Any, found: List[str]) -> None:
    """Recursively collect all inline chunk_id tokens from nested dicts/lists/strings."""
    if isinstance(obj, str):
        found.extend(_extract_inline_chunk_ids(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk_and_collect_chunk_ids(v, found)
    elif isinstance(obj, list):
        for item in obj:
            _walk_and_collect_chunk_ids(item, found)


# ── Main public function ───────────────────────────────────────────────────────

def build_citation_block(
    ba_output: Optional[Dict[str, Any]],
    quant_output: Optional[Dict[str, Any]],
    web_output: Optional[Dict[str, Any]],
    fm_output: Optional[Dict[str, Any]] = None,
    ticker: Optional[str] = None,
    index_offset: int = 0,
) -> Tuple[str, Dict[str, Citation]]:
    """Extract all citations from agent outputs and return a formatted reference block.

    Args:
        ba_output:     Business Analyst agent output dict (or None).
        quant_output:  Quant Fundamental agent output dict (or None).
        web_output:    Web Search agent output dict (or None).
        fm_output:     Financial Modelling agent output dict (or None).
        ticker:        Primary ticker symbol for label generation.
        index_offset:  Starting index offset — used when building multi-ticker
                       citation blocks so each call produces globally-unique [N]
                       numbers.  Pass the number of citations already assigned in
                       prior calls; the first citation in this block will be
                       ``index_offset + 1``.

    Returns:
        (reference_block_text, chunk_id_to_citation_map)

        reference_block_text:     Markdown-formatted References section.
        chunk_id_to_citation_map: Maps chunk_id strings → Citation objects so the
                                  summarizer can inject inline [N] numbers.
    """
    citations: List[Citation] = []
    seen_ids: Dict[str, int] = {}  # chunk_id/url → citation index (dedup)

    def _add(source_agent: str, db: str, label: str, detail: str = "",
             url: str = "", chunk_id: str = "") -> Citation:
        dedup_key = chunk_id or url or label
        if dedup_key in seen_ids:
            return citations[seen_ids[dedup_key] - 1]
        idx = index_offset + len(citations) + 1
        c = Citation(idx, source_agent, db, label, detail, url, chunk_id)
        citations.append(c)
        seen_ids[dedup_key] = idx
        return c

    # ── 1. Business Analyst citations ─────────────────────────────────────────
    if ba_output:
        # 1a. Inline chunk_id tokens embedded in prose fields
        inline_ids: List[str] = []
        _walk_and_collect_chunk_ids(ba_output, inline_ids)

        # 1b. Explicit sources arrays (competitive_moat.sources, key_risks[].source)
        moat = ba_output.get("competitive_moat") or {}
        for cid in (moat.get("sources") or []):
            if isinstance(cid, str) and cid:
                inline_ids.append(cid)

        for risk in (ba_output.get("key_risks") or []):
            if isinstance(risk, dict) and risk.get("source"):
                inline_ids.append(risk["source"])

        # Deduplicate while preserving order
        seen_cids: set = set()
        for cid in inline_ids:
            cid_clean = cid.strip().rstrip("].,;")
            if cid_clean and cid_clean not in seen_cids:
                seen_cids.add(cid_clean)
                db, label, detail = _chunk_id_to_label(cid_clean, ticker)
                _add("business_analyst", db, label, detail, chunk_id=cid_clean)

        # 1c. Sentiment — always sourced from PostgreSQL
        if ba_output.get("sentiment"):
            sent_src = (ba_output["sentiment"].get("source") or "postgresql:sentiment_trends")
            _add(
                "business_analyst",
                "postgresql",
                "Market Sentiment Data",
                _DB_LABELS.get(sent_src, sent_src),
                chunk_id=sent_src,
            )

        # 1d. CRAG fallback_triggered — note the web search fallback was used
        if ba_output.get("fallback_triggered"):
            _add(
                "business_analyst",
                "web_fallback",
                "Web Search Fallback (CRAG INCORRECT)",
                "Local knowledge base context was insufficient; live web search was used.",
            )

    # ── 2. Quant Fundamental citations ────────────────────────────────────────
    if quant_output:
        tk = quant_output.get("ticker") or ticker or ""
        tk_label = tk if tk else "Market"

        # Always present when quant agent ran
        _add(
            "quant_fundamental",
            "postgresql",
            f"{tk_label} — Income Statement / Balance Sheet / Cash Flow",
            "PostgreSQL · raw_fundamentals (FMP ingestion)",
            chunk_id="postgresql:raw_fundamentals",
        )
        _add(
            "quant_fundamental",
            "postgresql",
            f"{tk_label} — TTM Ratios & Key Metrics",
            "PostgreSQL · ratios_ttm, key_metrics_ttm (FMP ingestion)",
            chunk_id="postgresql:ratios_ttm",
        )

        # Price history used for beta / Sharpe / momentum
        ds = quant_output.get("data_sources") or {}
        if ds.get("price_history") or (quant_output.get("momentum_risk") and
                any(v is not None for v in (quant_output.get("momentum_risk") or {}).values())):
            _add(
                "quant_fundamental",
                "postgresql",
                f"{tk_label} — Price History (Beta, Sharpe, Momentum)",
                "PostgreSQL · raw_timeseries (EOD ingestion)",
                chunk_id="postgresql:raw_timeseries",
            )

        # Anomaly flags use rolling multi-period history
        if quant_output.get("anomaly_flags"):
            _add(
                "quant_fundamental",
                "postgresql",
                f"{tk_label} — Rolling Financial History (Anomaly Detection)",
                "PostgreSQL · raw_fundamentals (12-period rolling window)",
                chunk_id="postgresql:raw_fundamentals:rolling",
            )

        # Financial scores (Piotroski, Beneish)
        qf = quant_output.get("quality_factors") or {}
        if qf.get("piotroski_f_score") is not None or qf.get("beneish_m_score") is not None:
            _add(
                "quant_fundamental",
                "postgresql",
                f"{tk_label} — Financial Scores (Piotroski F-Score, Beneish M-Score)",
                "PostgreSQL · financial_scores (FMP ingestion)",
                chunk_id="postgresql:financial_scores",
            )

    # ── 3. Web Search citations ───────────────────────────────────────────────
    if web_output:
        # 3a. Breaking news articles
        for item in (web_output.get("breaking_news") or []):
            if not isinstance(item, dict):
                continue
            title = item.get("title") or "News article"
            url   = item.get("url") or ""
            date  = item.get("published_date") or ""
            tier  = item.get("source_tier")
            detail_parts = []
            if date:
                detail_parts.append(date)
            if tier:
                detail_parts.append(f"Source tier {tier}")
            if not item.get("verified", True):
                detail_parts.append("unverified")
            _add(
                "web_search",
                "web",
                title[:80],
                ", ".join(detail_parts),
                url=url,
                chunk_id=url,
            )

        # 3b. Risk flag sources
        for flag in (web_output.get("unknown_risk_flags") or []):
            if not isinstance(flag, dict):
                continue
            risk = flag.get("risk") or "Risk signal"
            url  = flag.get("source_url") or ""
            sev  = flag.get("severity") or ""
            _add(
                "web_search",
                "web",
                f"Risk: {risk[:60]}",
                f"Severity: {sev}" if sev else "",
                url=url,
                chunk_id=f"risk::{url}",
            )

        # 3c. Competitor signal sources
        for sig in (web_output.get("competitor_signals") or []):
            if not isinstance(sig, dict):
                continue
            company = sig.get("company") or "Competitor"
            signal  = sig.get("signal") or ""
            url     = sig.get("source_url") or ""
            _add(
                "web_search",
                "web",
                f"{company}: {signal[:50]}",
                "",
                url=url,
                chunk_id=f"competitor::{url}",
            )

        # 3d. Raw Perplexity citations (URLs not already captured above)
        for url in (web_output.get("raw_citations") or []):
            if isinstance(url, str) and url and url not in seen_ids:
                _add(
                    "web_search",
                    "web",
                    url.split("/")[-1][:60] or url[:60],
                    "Perplexity Sonar citation",
                    url=url,
                    chunk_id=url,
                )

    # ── 4. Financial Modelling citations ──────────────────────────────────────
    if fm_output:
        tk = fm_output.get("ticker") or ticker or ""
        tk_label = tk if tk else "Market"

        # DCF and WACC — raw_fundamentals (FCF inputs) + raw_timeseries (β, Rf)
        val = fm_output.get("valuation") or {}
        if val.get("dcf") or val.get("implied_price_range"):
            _add(
                "financial_modelling",
                "postgresql",
                f"{tk_label} — DCF & WACC Valuation",
                "PostgreSQL · raw_fundamentals (FCF, EBITDA, debt, equity) + "
                "raw_timeseries (60-day β, 10Y Treasury Rf) — all math in Python",
                chunk_id="postgresql:fm:dcf_wacc",
            )

        # Comps — Neo4j peer selection + PostgreSQL multiples
        if val.get("comps"):
            _add(
                "financial_modelling",
                "neo4j",
                f"{tk_label} — Comparable Company Analysis (Comps)",
                "Neo4j · COMPETES_WITH edges (peer group) + "
                "PostgreSQL · raw_fundamentals (EV/EBITDA, P/E, P/S multiples)",
                chunk_id="neo4j:fm:comps_peer_group",
            )

        # Technicals — raw_timeseries EOD prices
        if fm_output.get("technicals"):
            _add(
                "financial_modelling",
                "postgresql",
                f"{tk_label} — Technical Analysis",
                "PostgreSQL · raw_timeseries (EOD OHLCV) — "
                "SMA/EMA/RSI/MACD/BB/ATR/HV30/Stochastic computed in Python",
                chunk_id="postgresql:fm:technicals",
            )

        # Earnings — raw_fundamentals earnings history
        if fm_output.get("earnings"):
            _add(
                "financial_modelling",
                "postgresql",
                f"{tk_label} — Earnings Analysis (EPS Surprise, Beat/Miss Streak)",
                "PostgreSQL · raw_fundamentals (earnings_history, analyst_estimates)",
                chunk_id="postgresql:fm:earnings",
            )

        # Dividends — raw_fundamentals dividends
        if fm_output.get("dividends") and any(
            v is not None for v in (fm_output.get("dividends") or {}).values()
        ):
            _add(
                "financial_modelling",
                "postgresql",
                f"{tk_label} — Dividend Analysis",
                "PostgreSQL · raw_fundamentals (dividends, splits)",
                chunk_id="postgresql:fm:dividends",
            )

        # Factor Scores — Piotroski, Beneish, Altman Z
        if fm_output.get("factor_scores") and any(
            v is not None for v in (fm_output.get("factor_scores") or {}).values()
        ):
            _add(
                "financial_modelling",
                "postgresql",
                f"{tk_label} — Factor Scores (Piotroski F, Beneish M, Altman Z)",
                "PostgreSQL · raw_fundamentals — all scores computed in Python",
                chunk_id="postgresql:fm:factor_scores",
            )

    # ── Format the reference block ────────────────────────────────────────────
    if not citations:
        return "", {}

    lines = ["", "---", "### References"]

    # Group by agent
    agent_order = ["business_analyst", "quant_fundamental", "web_search", "financial_modelling", "web_fallback"]
    agent_labels = {
        "business_analyst":   "Business Analyst (qualitative research)",
        "quant_fundamental":  "Quant Fundamental (financial data)",
        "web_search":         "Web Search (live sources)",
        "financial_modelling": "Financial Modelling (DCF, WACC, Comps, Technicals)",
        "web_fallback":       "Web Fallback",
    }

    by_agent: Dict[str, List[Citation]] = {a: [] for a in agent_order}
    for c in citations:
        bucket = c.source_agent if c.source_agent in by_agent else "business_analyst"
        by_agent[bucket].append(c)

    for agent_key in agent_order:
        group = by_agent[agent_key]
        if not group:
            continue
        lines.append(f"\n**{agent_labels[agent_key]}**")
        for c in group:
            lines.append(c.footnote())

    chunk_id_map: Dict[str, Citation] = {c.chunk_id: c for c in citations if c.chunk_id}
    return "\n".join(lines), chunk_id_map


def inject_inline_numbers(text: str, chunk_id_map: Dict[str, Citation]) -> str:
    """Replace [qdrant::...] and [neo4j::...] inline tokens with numeric refs [N].

    If a token is not in the map (e.g. it was filtered as ungrounded), it is
    removed to keep the prose clean.
    """
    if not chunk_id_map or not text:
        return text

    def _replace(m: re.Match) -> str:
        raw = m.group(1).rstrip("].,;")
        if raw in chunk_id_map:
            return chunk_id_map[raw].ref()
        # Try prefix match
        for cid, cit in chunk_id_map.items():
            if cid and (raw.startswith(cid) or cid.startswith(raw)):
                return cit.ref()
        return ""  # strip unresolved token

    return re.sub(r'\[(qdrant::[^\]]+|neo4j::[^\]]+)\]', _replace, text)


__all__ = ["Citation", "build_citation_block", "inject_inline_numbers"]
