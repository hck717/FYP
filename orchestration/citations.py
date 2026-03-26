"""Citation extraction and formatting for the orchestration summarizer.

Pulls structured citations from all agent outputs and renders them
into a numbered reference list that the summarizer appends to its narrative.

Citation sources by agent:
  business_analyst    → Neo4j graph chunks    (neo4j::TICKER::section::n)
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
from urllib.parse import urlparse


# ── Human-readable database labels ────────────────────────────────────────────

_DB_LABELS: Dict[str, str] = {
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

# High-confidence finance/news sources for web citations.
_HIGH_QUALITY_WEB_DOMAINS = {
    "sec.gov", "reuters.com", "bloomberg.com", "wsj.com", "ft.com", "cnbc.com",
    "marketwatch.com", "barrons.com", "apnews.com", "investor.tesla.com",
    "investor.apple.com", "investor.microsoft.com", "abc.xyz", "nasdaq.com",
    "nyse.com", "federalreserve.gov", "bis.doc.gov", "ec.europa.eu", "imf.org",
    "worldbank.org", "oecd.org", "tradingeconomics.com", "finance.yahoo.com",
}

_LOW_QUALITY_WEB_DOMAINS = {
    "youtube.com", "youtu.be", "tiktok.com", "reddit.com", "x.com", "twitter.com",
    "stocktwits.com", "seekingalpha.com", "fool.com", "motleyfool.com", "medium.com",
    "substack.com", "blogspot.com", "wordpress.com",
}

_WEB_PUBLISHER_ALIASES = {
    "sec.gov": "SEC",
    "reuters.com": "Reuters",
    "bloomberg.com": "Bloomberg",
    "wsj.com": "Wall Street Journal",
    "ft.com": "Financial Times",
    "cnbc.com": "CNBC",
    "marketwatch.com": "MarketWatch",
    "barrons.com": "Barron's",
    "apnews.com": "AP News",
    "finance.yahoo.com": "Yahoo Finance",
    "nasdaq.com": "Nasdaq",
}


def _url_domain(url: str) -> str:
    try:
        domain = urlparse(url or "").netloc.lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def _is_high_quality_web_source(url: str) -> bool:
    domain = _url_domain(url)
    if not domain:
        return False
    if any(domain == d or domain.endswith("." + d) for d in _LOW_QUALITY_WEB_DOMAINS):
        return False
    return any(domain == d or domain.endswith("." + d) for d in _HIGH_QUALITY_WEB_DOMAINS)


def _publisher_from_url(url: str) -> str:
    domain = _url_domain(url)
    if not domain:
        return "Web Source"
    for key, pub in _WEB_PUBLISHER_ALIASES.items():
        if domain == key or domain.endswith("." + key):
            return pub
    return domain


def _human_web_label(title: str, url: str) -> str:
    """Return a clean citation label for web sources.

    Avoid raw fragments like `watch?v=...` in the reference list.
    """
    t = (title or "").strip()
    if t and not re.match(r"^(?:watch\?v=|https?://)", t, flags=re.IGNORECASE):
        return t[:90]
    pub = _publisher_from_url(url)
    path = (urlparse(url).path or "").strip("/") if url else ""
    if path:
        tail = path.split("/")[-1].replace("-", " ").replace("_", " ").strip()
        if tail and tail.lower() not in {"watch", "video"}:
            tail = re.sub(r"\s+", " ", tail)
            return f"{pub}: {tail[:64]}"
    return f"{pub} article"


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
        label = re.sub(r'\s+', ' ', str(self.label or '')).strip()
        detail = re.sub(r'\s+', ' ', str(self.detail or '')).strip()
        parts = [f"[{self.index}]", label]
        if detail:
            parts.append(f"— {detail}")
        if self.url:
            parts.append(f"({self.url})")
        parts.append(f"[{self.source_agent}]")
        return "  " + " ".join(parts)


# ── Extraction helpers ─────────────────────────────────────────────────────────

def _chunk_id_to_label(
    chunk_id: str,
    ticker: Optional[str] = None,
    source_name: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Parse a chunk_id string into (db, label, detail).

    Formats handled:
      neo4j::AAPL::risk_factors::0          →  Neo4j,  "Risk Factors (filing)",  AAPL
      neo4j::AAPL::mda::2                   →  Neo4j,  "MDA",                    AAPL
      AAPL::broker_report::abc123def456789  →  Neo4j,  source_name or "Broker Report",  AAPL
      AAPL::earnings_call::abc123def456789  →  Neo4j,  source_name or "Earnings Call",  AAPL
      AAPL::risk_factors::0                 →  Neo4j,  "Risk Factors (filing)",  AAPL
    """
    if not chunk_id:
        return "unknown", chunk_id, ""

    parts = chunk_id.split("::")
    if len(parts) < 2:
        return "unknown", chunk_id, ""

    backend = parts[0].lower()  # "neo4j" or ticker symbol

    if backend == "neo4j":
        # neo4j::TICKER::section::n
        tk        = parts[1] if len(parts) > 1 else (ticker or "")
        section   = parts[2] if len(parts) > 2 else "filing"
        sec_label = _SECTION_LABELS.get(section, section.replace("_", " ").title())
        label     = f"{tk} — {sec_label}"
        detail    = "Neo4j knowledge graph"
        return "neo4j", label, detail

    # Ticker-prefixed format: TICKER::section::hash  (broker_report, earnings_call, etc.)
    # parts[0] is the ticker, parts[1] is the section, parts[2] is the hash
    tk      = parts[0].upper()
    section = parts[1] if len(parts) > 1 else "document"
    if section == "broker_report":
        # Use source_name (e.g. "Wells Fargo_GOOGL_2025.pdf") if available,
        # otherwise fall back to a generic label
        if source_name:
            # Strip common extensions and clean up underscores/dashes
            display = re.sub(r'\.(pdf|txt|docx?)$', '', source_name, flags=re.IGNORECASE)
            display = display.replace('_', ' ').strip()
            label   = f"{tk} — Broker Report: {display}"
        else:
            label = f"{tk} — Broker Research Report"
        return "neo4j", label, "Neo4j knowledge graph · broker_report"

    if section == "earnings_call":
        if source_name:
            display = re.sub(r'\.(pdf|txt|docx?)$', '', source_name, flags=re.IGNORECASE)
            display = display.replace('_', ' ').strip()
            label   = f"{tk} — Earnings Call: {display}"
        else:
            # Fallback label with chunk hint so reference list remains human-readable,
            # but never shows only raw hashes.
            hint = parts[2][:8] if len(parts) > 2 else "unknown"
            label = f"{tk} — Earnings Call Transcript ({hint})"
        return "neo4j", label, "Neo4j knowledge graph · earnings_call"

    # Other ticker-prefixed sections (annual_report, 10-K, press_release, etc.)
    sec_label = _SECTION_LABELS.get(section, section.replace("_", " ").title())
    label     = f"{tk} — {sec_label}"
    return "neo4j", label, "Neo4j knowledge graph"


def _extract_inline_chunk_ids(text: str) -> List[Tuple[str, str]]:
    """Find all inline citation tokens in a string.

    Handles two formats:
      Old:  [neo4j::TICKER::section::n]          → ("", "neo4j::TICKER::section::n")
      New:  [source_name | TICKER::section::hash] → ("source_name", "TICKER::section::hash")
      Bare: [TICKER::section::hash]               → ("", "TICKER::section::hash")

    Returns a list of (source_name, chunk_id) tuples.
    """
    results: List[Tuple[str, str]] = []
    # Match any bracket containing "::" — covers all three formats above
    for raw in re.findall(r'\[([^\]]+::[^\]]+)\]', text or ""):
        raw = raw.strip().rstrip("].,;")
        if " | " in raw:
            sname, cid = raw.split(" | ", 1)
            results.append((sname.strip(), cid.strip().rstrip("].,;")))
        else:
            results.append(("", raw))
    return results


def _walk_and_collect_chunk_ids(obj: Any, found: List[Tuple[str, str]]) -> None:
    """Recursively collect all inline chunk_id tokens from nested dicts/lists/strings.

    Each element of *found* is a (source_name, chunk_id) tuple.
    """
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
    sr_output: Optional[Dict[str, Any]] = None,
    macro_output: Optional[Dict[str, Any]] = None,
    insider_news_output: Optional[Dict[str, Any]] = None,
    ticker: Optional[str] = None,
    index_offset: int = 0,
) -> Tuple[str, Dict[str, Citation]]:
    """Extract all citations from agent outputs and return a formatted reference block.

    Args:
        ba_output:     Business Analyst agent output dict (or None).
        quant_output:  Quant Fundamental agent output dict (or None).
        web_output:    Web Search agent output dict (or None).
        fm_output:     Financial Modelling agent output dict (or None).
        sr_output:     Stock Research agent output dict (or None).
        macro_output:  Macro agent output dict (or None).
        insider_news_output: Insider+News agent output dict (or None).
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
    seen_ids: Dict[str, int] = {}  # chunk_id/url → local citations list index (dedup)

    def _add(source_agent: str, db: str, label: str, detail: str = "",
             url: str = "", chunk_id: str = "") -> Citation:
        dedup_key = chunk_id or url or label
        if dedup_key in seen_ids:
            return citations[seen_ids[dedup_key]]
        idx = index_offset + len(citations) + 1
        c = Citation(idx, source_agent, db, label, detail, url, chunk_id)
        citations.append(c)
        seen_ids[dedup_key] = len(citations) - 1
        return c

    # ── 1. Business Analyst citations ─────────────────────────────────────────
    if ba_output:
        # 1a. Inline chunk_id tokens embedded in prose fields
        # Each entry is a (source_name, chunk_id) tuple
        inline_pairs: List[Tuple[str, str]] = []
        _walk_and_collect_chunk_ids(ba_output, inline_pairs)

        # 1b. Explicit sources arrays (competitive_moat.sources, key_risks[].source)
        # These are bare chunk_id strings (no source_name), so we wrap them as ("", cid)
        moat = ba_output.get("competitive_moat") or {}
        for cid in (moat.get("sources") or []):
            if isinstance(cid, str) and cid:
                inline_pairs.append(("", cid))

        for risk in (ba_output.get("key_risks") or []):
            if isinstance(risk, dict) and risk.get("source"):
                inline_pairs.append(("", risk["source"]))

        # Deduplicate while preserving order; key is chunk_id
        seen_cids: set = set()
        for sname, cid in inline_pairs:
            cid_clean = cid.strip().rstrip("].,;")
            if cid_clean and cid_clean not in seen_cids:
                seen_cids.add(cid_clean)
                db, label, detail = _chunk_id_to_label(
                    cid_clean, ticker, source_name=sname or None
                )
                _add("business_analyst", db, label, detail, chunk_id=cid_clean)

        # 1c. Sentiment — always sourced from PostgreSQL
        if ba_output.get("sentiment"):
            _sent = ba_output["sentiment"]
            sent_src = (_sent.get("source") if isinstance(_sent, dict) else None) or "postgresql:sentiment_trends"
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
            url = item.get("url") or ""
            if not _is_high_quality_web_source(url):
                # Keep reference list focused on investable-quality sources.
                continue
            title = _human_web_label(str(item.get("title") or ""), url)
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
            if isinstance(url, str) and url and url not in seen_ids and _is_high_quality_web_source(url):
                _add(
                    "web_search",
                    "web",
                    _human_web_label("", url),
                    f"Perplexity Sonar citation · {_publisher_from_url(url)}",
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

    # ── 5. Stock Research citations ───────────────────────────────────────────
    if sr_output:
        # From explicit per-task citations: [{'doc_name': ..., 'page': ...}, ...]
        for c in (sr_output.get("citations") or []):
            if not isinstance(c, dict):
                continue
            doc = str(c.get("doc_name") or "Stock Research Document").strip()
            page = c.get("page")
            detail = f"page {page}" if page is not None else ""
            chunk_id = str(c.get("chunk_id") or f"stock_research::{doc}::{page}")
            _add(
                "stock_research",
                "neo4j",
                doc,
                detail,
                chunk_id=chunk_id,
            )

    # ── 6. Macro citations ────────────────────────────────────────────────────
    if macro_output:
        source = str(macro_output.get("data_source") or "neo4j").lower()
        db_detail = "PostgreSQL/Neo4j macro report store"
        # Add explicit citation rows from the macro agent
        for c in (macro_output.get("citations") or []):
            if not isinstance(c, dict):
                continue
            doc = str(c.get("doc_name") or "Macro Research Document").strip()
            page = c.get("page")
            detail = f"page {page}" if page is not None else db_detail
            chunk_id = str(c.get("chunk_id") or f"macro::{doc}::{page}")
            _add(
                "macro",
                "postgresql" if source in ("pg", "neo4j") else source,
                doc,
                detail,
                chunk_id=chunk_id,
            )
        # Fallback anchor citation if no explicit per-doc citations were emitted
        if not (macro_output.get("citations") or []):
            _add(
                "macro",
                "postgresql" if source in ("pg", "neo4j") else source,
                f"{(ticker or str(macro_output.get('ticker') or 'TICKER')).upper()} — Macro Regime and Thematic Analysis",
                db_detail,
                chunk_id="postgresql:macro:regime_and_themes",
            )

    # ── 7. Insider + News citations ───────────────────────────────────────────
    if insider_news_output:
        source = str(insider_news_output.get("data_source") or "postgresql").lower()
        coverage = insider_news_output.get("data_coverage") or {}
        insider_n = coverage.get("insider_transactions_count", 0)
        news_n = coverage.get("news_articles_count", 0)
        # Add explicit citations emitted by the insider/news synthesis
        for c in (insider_news_output.get("citations") or []):
            if not isinstance(c, dict):
                continue
            doc = str(c.get("doc_name") or "Insider/News Document").strip()
            page = c.get("page")
            detail = f"page {page}" if page is not None else "PostgreSQL insider/news store"
            chunk_id = str(c.get("chunk_id") or f"insider_news::{doc}::{page}")
            _add(
                "insider_news",
                "postgresql" if source == "pg" else source,
                doc,
                detail,
                chunk_id=chunk_id,
            )
        # Fallback source anchor if citations are empty
        if not (insider_news_output.get("citations") or []):
            _add(
                "insider_news",
                "postgresql" if source == "pg" else source,
                f"{(ticker or str(insider_news_output.get('ticker') or 'TICKER')).upper()} — Insider Transactions and News Articles",
                f"PostgreSQL · insider_transactions/news_articles (insider={insider_n}, news={news_n})",
                chunk_id="postgresql:insider_news:coverage",
            )

    # ── Format the reference block ────────────────────────────────────────────
    if not citations:
        return "", {}

    lines = ["", "---", "### References"]

    # Group by agent
    agent_order = [
        "business_analyst",
        "quant_fundamental",
        "financial_modelling",
        "stock_research",
        "macro",
        "insider_news",
        "web_search",
        "web_fallback",
    ]
    agent_labels = {
        "business_analyst":   "Business Analyst (qualitative research)",
        "stock_research":     "Stock Research (earnings calls & broker reports)",
        "macro":              "Macro (regime, thematic, transmission channels)",
        "insider_news":       "Insider & News (transactions and sentiment)",
        "quant_fundamental":  "Quant Fundamental (financial data)",
        "web_search":         "Web Search (live sources)",
        "financial_modelling": "Financial Modelling (DCF, WACC, Comps, Technicals)",
        "web_fallback":       "Web Fallback",
    }

    by_agent: Dict[str, List[Citation]] = {a: [] for a in agent_order}
    for c in citations:
        bucket = c.source_agent if c.source_agent in by_agent else "business_analyst"
        by_agent[bucket].append(c)

    # Re-number citations to follow the final display order exactly.
    # This keeps references clean and avoids out-of-order numbers by section.
    ordered: List[Citation] = []
    for agent_key in agent_order:
        group = by_agent[agent_key]
        if not group:
            continue
        group_sorted = sorted(group, key=lambda c: c.index)
        ordered.extend(group_sorted)

    for i, c in enumerate(ordered, start=1):
        c.index = index_offset + i

    for agent_key in agent_order:
        group = by_agent[agent_key]
        if not group:
            continue
        lines.append(f"\n**{agent_labels[agent_key]}**")
        group_sorted = sorted(group, key=lambda c: c.index)
        for c in group_sorted:
            lines.append(c.footnote())

    chunk_id_map: Dict[str, Citation] = {c.chunk_id: c for c in citations if c.chunk_id}
    return "\n".join(lines), chunk_id_map


def inject_inline_numbers(text: str, chunk_id_map: Dict[str, Citation]) -> str:
    """Replace inline citation tokens with numeric refs [N].

    Handles:
      Old:  [neo4j::TICKER::section::n]          → [N]
      New:  [source_name | TICKER::section::hash] → [N]
      Bare: [TICKER::section::hash]               → [N]

    If a token is not in the map (e.g. it was filtered as ungrounded), it is
    removed to keep the prose clean.
    """
    if not chunk_id_map or not text:
        return text

    def _resolve(cid: str) -> str:
        """Look up cid in chunk_id_map; try prefix match if exact match fails."""
        if cid in chunk_id_map:
            return chunk_id_map[cid].ref()
        for key, cit in chunk_id_map.items():
            if key and (cid.startswith(key) or key.startswith(cid)):
                return cit.ref()
        return ""  # strip unresolved token

    def _replace(m: re.Match) -> str:
        raw = m.group(1).strip().rstrip("].,;")
        # New format: "source_name | chunk_id"
        if " | " in raw:
            cid = raw.split(" | ", 1)[1].strip().rstrip("].,;")
        else:
            cid = raw
        return _resolve(cid)

    # Match any bracket containing "::" — covers all inline citation formats
    return re.sub(r'\[([^\]]+::[^\]]+)\]', _replace, text)


__all__ = ["Citation", "build_citation_block", "inject_inline_numbers"]
