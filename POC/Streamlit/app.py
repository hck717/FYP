"""Streamlit UI for The Agentic Investment Analyst.

Run from the repo root:
    streamlit run POC/streamlit/app.py

Architecture:
    This app wraps the orchestration layer (orchestration/graph.py) and streams
    live node-by-node progress to the UI using orchestration.graph.stream().
    Per-agent real-time progress is shown via a thread-safe queue that
    node_parallel_agents pushes to as each agent finishes.
    Per-agent result cards are rendered once the full pipeline completes.
"""

from __future__ import annotations

import os
import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Ensure repo root is on sys.path so orchestration imports work ─────────────
def _discover_repo_root() -> Path:
    """Find project root in both local and Docker-mounted layouts."""
    env_root = os.getenv("REPO_ROOT")
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if p.exists():
            return p

    this_file = Path(__file__).resolve()
    search_roots = [this_file.parent, *this_file.parents]
    for candidate in search_roots:
        orch_graph = candidate / "orchestration" / "graph.py"
        orch_nodes = candidate / "orchestration" / "nodes.py"
        agents_dir = candidate / "agents"
        if orch_graph.exists() and orch_nodes.exists() and agents_dir.exists():
            return candidate

    # Last-resort fallback: previous local layout assumption.
    return this_file.parents[2] if len(this_file.parents) > 2 else this_file.parent


_REPO_ROOT = _discover_repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Load .env file for environment variables ───────────────────────────────
from dotenv import load_dotenv
load_dotenv(dotenv_path=_REPO_ROOT / ".env", override=False)

import streamlit as st
from charts import (
    # Core research charts
    chart_revenue_trend,
    chart_margin_trends,
    chart_peer_comps,
    chart_football_field,
    chart_dcf_waterfall,
    chart_sensitivity_heatmap,
    chart_moe_consensus,
    chart_technicals,
    chart_sentiment_donut,
    chart_factor_scorecard,
    chart_eps_trend,
    # Live charts (previously stubs — now use real data when available)
    chart_ebitda_trend,
    chart_fcf_trend,
    chart_price_history,
    chart_price_performance,
    # Backward-compat stub aliases (kept so old imports don't break)
    chart_ebitda_trend_stub,
    chart_fcf_trend_stub,
    chart_price_history_stub,
    chart_price_performance_stub,
    # Backward-compat aliases (kept so old chart_hints strings still work)
    chart_dcf_scenarios,
    chart_quarterly_trends,
    chart_factor_radar,
    chart_altman_z,
)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Agentic Investment Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
SUPPORTED_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

EXAMPLE_QUERIES = [
    "What is Apple's competitive moat and current valuation?",
    "Give me a full investment analysis of NVDA.",
    "Compare MSFT vs GOOGL on fundamentals and valuation.",
    "Is TSLA overvalued? Show DCF and technicals.",
    "What are the latest risks and news for AAPL?",
]

# Canonical language keys expected by orchestration translation/scoring paths.
_UI_OUTPUT_LANGUAGE_MAP = {
    "English": "english",
    "Cantonese": "cantonese",
    "Mandarin": "mandarin",
    "Spanish": "spanish",
    "French": "french",
    "German": "german",
    "Japanese": "japanese",
    "Korean": "korean",
    "Portuguese": "portuguese",
    "Italian": "italian",
    "Dutch": "dutch",
    "Russian": "russian",
    "Arabic": "arabic",
    "Hindi": "hindi",
    "Thai": "thai",
    "Vietnamese": "vietnamese",
    "Indonesian": "indonesian",
}

# Node display labels / descriptions
NODE_LABELS = {
    "planner":         ("Planning", "Resolving ticker, selecting agents, setting complexity…"),
    "parallel_agents": ("Running Agents", "Dispatching selected agents…"),
    "react_check":     ("ReAct Check", "Evaluating agent outputs for gaps or errors…"),
    "summarizer":      ("Summarizing", "Generating final research note with DeepSeek-R1…"),
    "memory_update":   ("Memory Update", "Persisting failure patterns to episodic memory…"),
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

    :root {
        --ui-font: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
        --num-font: 'JetBrains Mono', 'SF Mono', Monaco, monospace;
    }

    [data-theme="light"], .stApp[data-theme="light"] {
        --text-main: #1a1a1a;
        --text-muted: #4b5563;
        --line-subtle: rgba(0, 0, 0, 0.1);
    }

    [data-theme="dark"], .stApp[data-theme="dark"] {
        --text-main: #e6eaf0;
        --text-muted: #a8b3c2;
        --line-subtle: rgba(230, 234, 240, 0.14);
    }

    /* Fallback for default (dark mode default) */
    :root, [data-theme="dark"] {
        --text-main: #e6eaf0;
        --text-muted: #a8b3c2;
        --line-subtle: rgba(230, 234, 240, 0.14);
    }

    html, body, [class*="css"], .stMarkdown, .stText, .stCaption {
        font-family: var(--ui-font);
        color: var(--text-main);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        font-feature-settings: "liga" 1, "kern" 1;
    }

    .block-container {
        max-width: 100% !important;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    h1, h2, h3 {
        font-family: var(--ui-font);
        letter-spacing: -0.02em;
        line-height: 1.15;
        font-weight: 700;
    }

    h1 { font-size: 2.35rem; }
    h2 { font-size: 1.75rem; }
    h3 { font-size: 1.25rem; }

    p, li, .stMarkdown p {
        font-size: 1.02rem;
        line-height: 1.75;
        color: var(--text-main);
    }

    .report-wrap {
        max-width: 920px;
        margin: 0 auto;
    }

    .report-wrap p, .report-wrap li {
        max-width: 78ch;
        word-break: break-word;
        word-wrap: break-word;
        overflow-wrap: break-word;
        line-height: 1.6;
        hyphens: auto;
    }

    .report-wrap {
        overflow-x: hidden;
    }

    /* Ensure text in markdown doesn't overflow */
    .report-wrap * {
        word-break: break-word;
        overflow-wrap: break-word;
    }

    .stCaption, small, .metric-label {
        font-size: 0.9rem;
        color: var(--text-muted);
    }

    .stMetricValue, .metric-value {
        font-family: var(--num-font);
        letter-spacing: -0.01em;
    }

    hr {
        border: none;
        border-top: 1px solid var(--line-subtle);
        margin: 1.5rem 0;
    }

    .agent-card { border: 1px solid #2d2d2d; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
    .status-ok   { color: #22c55e; font-weight: 600; }
    .status-warn { color: #f59e0b; font-weight: 600; }
    .status-err  { color: #ef4444; font-weight: 600; }
    .metric-label { font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.1rem; font-weight: 600; }

    /* Inline code in markdown should not look like bright green chips */
    .stMarkdown code {
        color: inherit !important;
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 4px !important;
        padding: 0.05rem 0.2rem !important;
        font-family: var(--ui-font) !important;
        font-size: 0.95em !important;
    }

    /* Keep code blocks readable/monospace */
    pre code, .stCodeBlock code {
        font-family: var(--num-font) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_text_for_display(text: str) -> str:
    """Clean text to fix rendering issues with invisible Unicode characters and word wrapping."""
    if not text:
        return text
    import re

    # Remove zero-width characters and other invisible / control Unicode
    text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)

    # Normalise line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Fix concatenated text where spaces are missing between words (e.g., "wordanother" -> "word another")
    # Look for patterns where lowercase/number is followed directly by uppercase (likely word boundary)
    # But be careful not to break acronyms and proper nouns
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    
    # Also handle cases where a period/punctuation is directly followed by word without space
    # e.g., "end.Next" -> "end. Next"
    text = re.sub(r'([.!?:;,])([A-Z])', r'\1 \2', text)

    # Escape dollar signs that might be interpreted as LaTeX by Streamlit
    # Match single $ that aren't part of currency amounts (e.g., "$100")
    # Look for $ not followed by digits or $ already escaped
    text = re.sub(r'\$(?![0-9,.])', r'\\$', text)
    text = re.sub(r'(?<![\\])\$(?![0-9,.])', r'\\$', text)

    return text


def _escape_markdown_special_chars(text: str) -> str:
    """Escape special markdown characters that could break rendering."""
    if not text:
        return text
    
    import re
    
    # Escape characters that have special meaning in markdown but we want to display literally
    # But be careful not to break intentional markdown formatting
    
    # Escape backslashes first (to avoid double-escaping)
    text = text.replace('\\', '\\\\')
    
    # Escape dollar signs that aren't part of currency (already done in _clean_text_for_display)
    # but we'll do it again here for safety
    text = re.sub(r'(?<!\d)\$(?!\d)', r'\$', text)
    
    # Escape pipe characters that might break tables
    # But preserve markdown table pipes
    text = re.sub(r'\|(?!.*\|.*\n)', r'\\|', text)
    
    return text


def _sanitize_llm_markdown(text: str) -> str:
    """Sanitize model markdown so headings, LaTeX, and HTML cannot break layout."""
    if not text:
        return text

    import re

    cleaned = _clean_text_for_display(text)
    
    # Remove raw HTML heading tags if model emits them.
    cleaned = re.sub(r"<h[1-6][^>]*>.*?</h[1-6]>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove LaTeX math blocks that might break Streamlit rendering
    # Remove $...$ (inline math)
    cleaned = re.sub(r'\$[^$\n]+\$', lambda m: m.group(0).replace('$', '\\$'), cleaned)
    
    # Remove $$...$$ (display math)
    cleaned = re.sub(r'\$\$[^$]+\$\$', lambda m: m.group(0).replace('$', '\\$'), cleaned)
    
    # Remove \(...\) and \[...\] (LaTeX delimiters)
    cleaned = re.sub(r'\\\([^)]*\\\)', '', cleaned)
    cleaned = re.sub(r'\\\[[^\]]*\\\]', '', cleaned)
    
    # Remove HTML/XML-like tags that aren't markdown (e.g., <sub>, <sup>, <font>)
    cleaned = re.sub(r'<(?!/?(?:b|i|em|strong|code|pre|a|ul|ol|li|p|div|span|blockquote|hr)[>\s/])[^>]*>', '', cleaned, flags=re.IGNORECASE)
    
    # Downgrade all ATX headings to level-3 for consistent typography.
    cleaned = re.sub(r"(?m)^\s*#{1,2}\s+", "### ", cleaned)
    cleaned = re.sub(r"(?m)^\s*#{3,}\s+", "### ", cleaned)
    
    # Ensure section headers are always on their own lines (prevents
    # "... sentence. ## Header" from rendering as plain paragraph text).
    cleaned = re.sub(r"(?<!\n)\s*(##\s+)", r"\n\n\1", cleaned)
    
    # Ensure a blank line after each section header for paragraph separation.
    cleaned = re.sub(r"(?m)^(###\s+[^\n]+)\n(?!\n)", r"\1\n\n", cleaned)

    # Collapse excessive blank lines from malformed LLM markdown.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    
    # Remove any remaining problematic escape sequences
    cleaned = re.sub(r'\\[nrtfvb]', ' ', cleaned)  # Remove escape chars but keep space
    
    # Escape special markdown characters
    cleaned = _escape_markdown_special_chars(cleaned)
    
    return cleaned.strip()


def _collect_ba_references(output: Dict[str, Any]) -> List[str]:
    """Best-effort extraction of BA reference identifiers/URLs for display."""
    refs: List[str] = []

    def _add(val: Any) -> None:
        if isinstance(val, str):
            s = val.strip()
            if s and s not in refs:
                refs.append(s)

    moat = output.get("competitive_moat") or {}
    if isinstance(moat, dict):
        for s in (moat.get("sources") or []):
            _add(s)

    mgmt = output.get("management_guidance") or {}
    if isinstance(mgmt, dict):
        for s in (mgmt.get("sources") or []):
            _add(s)

    val_ctx = output.get("valuation_context") or {}
    if isinstance(val_ctx, dict):
        for s in (val_ctx.get("sources") or []):
            _add(s)

    for r in (output.get("key_risks") or []):
        if isinstance(r, dict):
            _add(r.get("source"))

    for s in (output.get("sources") or []):
        _add(s)

    # New BA-level structured citations (stock-research style)
    for c in (output.get("citations") or []):
        if isinstance(c, dict):
            doc = str(c.get("doc_name") or "").strip()
            page = c.get("page")
            if doc:
                if page is not None:
                    _add(f"{doc} (p. {page})")
                else:
                    _add(doc)

    return refs


def _render_ba_references(output: Dict[str, Any]) -> None:
    """Render Business Analyst citation diagnostics and source list."""
    citation_report = output.get("citation_report") or {}
    refs = _collect_ba_references(output)

    ba_citations = output.get("citations") or []

    if not citation_report and not refs and not ba_citations:
        return

    with st.expander("References & citations", expanded=False):
        if citation_report:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cited", _fmt(citation_report.get("total_cited")))
            c2.metric("Grounded", _fmt(citation_report.get("grounded")))
            c3.metric("Ungrounded", _fmt(citation_report.get("ungrounded")))
            c4.metric("Grounding %", f"{_fmt(citation_report.get('grounding_rate_pct'))}%")

        if ba_citations:
            st.caption(f"Total citations: {len(ba_citations)}")

            grouped: Dict[str, set] = {}
            for c in ba_citations:
                if not isinstance(c, dict):
                    continue
                doc = str(c.get("doc_name") or "unknown")
                page = c.get("page")
                grouped.setdefault(doc, set())
                if page is not None:
                    grouped[doc].add(page)

            for doc, pages in list(grouped.items())[:20]:
                if pages:
                    page_list = ", ".join(str(p) for p in sorted(pages))
                    st.markdown(f"- **{doc}** (p. {page_list})")
                else:
                    st.markdown(f"- **{doc}**")

        if refs and not ba_citations:
            st.caption("Evidence sources")
            for src in refs[:20]:
                st.markdown(f"- `{src}`")


def _render_stock_research_references(output: Dict[str, Any]) -> None:
    """Render stock-research document/page citations in a compact grouped form."""
    citations = output.get("citations") or []
    if not citations:
        return

    grouped: Dict[str, set] = {}
    for c in citations:
        if not isinstance(c, dict):
            continue
        doc = str(c.get("doc_name") or "unknown")
        page = c.get("page")
        grouped.setdefault(doc, set())
        if page is not None:
            grouped[doc].add(page)

    with st.expander("References & citations", expanded=False):
        st.caption(f"Total citations: {len(citations)}")
        for doc, pages in list(grouped.items())[:20]:
            if pages:
                page_list = ", ".join(str(p) for p in sorted(pages))
                st.markdown(f"- **{doc}** (p. {page_list})")
            else:
                st.markdown(f"- **{doc}**")


def _render_data_tables(state: Dict[str, Any]) -> None:
    """Render financial data tables from quant and fm outputs."""
    import pandas as pd
    
    tickers: List[str] = state.get("tickers") or []
    if not tickers and state.get("ticker"):
        tickers = [state["ticker"]]
    if not tickers:
        return

    qf_outputs: List[Dict] = state.get("quant_fundamental_outputs") or []
    if not qf_outputs and state.get("quant_fundamental_output"):
        qf_outputs = [state["quant_fundamental_output"]]
    
    fm_outputs: List[Dict] = state.get("financial_modelling_outputs") or []
    if not fm_outputs and state.get("financial_modelling_output"):
        fm_outputs = [state["financial_modelling_output"]]

    has_data = False

    for idx, ticker in enumerate(tickers):
        qf = qf_outputs[idx] if idx < len(qf_outputs) else {}
        fm = fm_outputs[idx] if idx < len(fm_outputs) else {}

        if not qf and not fm:
            continue

        qt = qf.get("quarterly_trends") or []
        key_metrics = qf.get("key_metrics") or {}
        vf = qf.get("value_factors") or {}

        if not qt and not key_metrics and not vf:
            continue

        has_data = True
        with st.expander(f"📊 Financial Data Tables — {ticker}", expanded=True):
            # Quarterly Revenue
            if qt:
                rev_rows = []
                for row in sorted(qt, key=lambda r: str(r.get("period", "")))[-4:]:
                    period = str(row.get("period", ""))
                    revenue = row.get("revenue")
                    if revenue is not None:
                        try:
                            rev_val = float(revenue) / 1_000_000
                            rev_rows.append({"Quarter": period, "Revenue ($M)": f"${rev_val:,.1f}M"})
                        except (TypeError, ValueError):
                            rev_rows.append({"Quarter": period, "Revenue ($M)": "N/A"})

                if rev_rows:
                    st.markdown("**Quarterly Revenue**")
                    df_rev = pd.DataFrame(rev_rows)
                    st.table(df_rev)

            # Annual Revenue — aggregate from quarterly_trends grouped by year
            if qt:
                annual_rev_map: Dict[str, float] = {}
                for row in sorted(qt, key=lambda r: str(r.get("period", ""))):
                    year = str(row.get("period", ""))[:4]
                    revenue = row.get("revenue")
                    if year.isdigit() and revenue is not None:
                        try:
                            annual_rev_map[year] = annual_rev_map.get(year, 0.0) + float(revenue)
                        except (TypeError, ValueError):
                            pass
                if annual_rev_map:
                    annual_rev_rows = [
                        {"Fiscal Year": yr, "Revenue ($M)": f"${total / 1_000_000:,.1f}M"}
                        for yr, total in sorted(annual_rev_map.items())
                    ]
                    st.markdown("**Annual Revenue (Sum of Quarterly)**")
                    st.table(pd.DataFrame(annual_rev_rows))

            # Quarterly Operating Earnings (EPS)
            if qt:
                eps_rows = []
                for row in sorted(qt, key=lambda r: str(r.get("period", "")))[-4:]:
                    period = str(row.get("period", ""))
                    eps = row.get("eps_diluted") or row.get("operating_earnings_per_share")
                    if eps is not None:
                        try:
                            eps_val = float(eps)
                            eps_rows.append({"Quarter": period, "EPS ($)": f"${eps_val:.2f}"})
                        except (TypeError, ValueError):
                            eps_rows.append({"Quarter": period, "EPS ($)": "N/A"})

                if eps_rows:
                    st.markdown("**Quarterly EPS (Diluted)**")
                    df_eps = pd.DataFrame(eps_rows)
                    st.table(df_eps)

            # Annual EPS — sum diluted EPS per year from quarterly_trends
            if qt:
                annual_eps_map: Dict[str, float] = {}
                for row in sorted(qt, key=lambda r: str(r.get("period", ""))):
                    year = str(row.get("period", ""))[:4]
                    eps = row.get("eps_diluted") or row.get("operating_earnings_per_share")
                    if year.isdigit() and eps is not None:
                        try:
                            annual_eps_map[year] = annual_eps_map.get(year, 0.0) + float(eps)
                        except (TypeError, ValueError):
                            pass
                if annual_eps_map:
                    annual_eps_rows = [
                        {"Fiscal Year": yr, "EPS ($, Annual)": f"${annual_eps_map[yr]:.2f}"}
                        for yr in sorted(annual_eps_map.keys())
                    ]
                    st.markdown("**Annual EPS (Sum of Quarterly Diluted EPS)**")
                    st.table(pd.DataFrame(annual_eps_rows))

            # Annual Valuation (P/E)
            pe_trailing = vf.get("pe_trailing")
            if pe_trailing is not None:
                try:
                    pe_val = float(pe_trailing)
                    fiscal_year = key_metrics.get("fiscal_year", "TTM")
                    st.markdown("**Valuation (Trailing P/E)**")
                    df_pe = pd.DataFrame([{"Period": fiscal_year, "P/E (TTM)": f"{pe_val:.1f}x"}])
                    st.table(df_pe)
                except (TypeError, ValueError):
                    pass

    if not has_data:
        st.caption("No financial data tables available.")


def _fmt(val: Any, decimals: int = 2, suffix: str = "") -> str:
    """Format a numeric value for display; returns '—' for None/missing."""
    if val is None:
        return "—"
    try:
        return f"{float(val):,.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(val)


def _pct(val: Any) -> str:
    return _fmt(val, 1, "%")


def _parse_institution_from_doc_name(doc_name: str) -> str:
    """Extract institution name from a broker doc_name string.

    Format: "<Institution> <TICKER> <title...>"
    Strategy: take everything before the first occurrence of an all-caps ticker
    segment (2-5 uppercase letters), then strip trailing whitespace.
    Falls back to the first token if no ticker pattern is found.
    """
    import re as _re
    if not doc_name:
        return "?"
    # Find the position of the first ALL-CAPS word that looks like a ticker (2-5 chars)
    m = _re.search(r'\b[A-Z]{2,5}\b', doc_name)
    if m:
        institution = doc_name[:m.start()].strip()
        return institution if institution else doc_name.split()[0]
    return doc_name.split()[0]


def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """Safely navigate a nested dict."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def _normalize_output_language(value: Optional[str]) -> Optional[str]:
    """Normalize UI language labels / free-text to canonical language keys."""
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    if raw in _UI_OUTPUT_LANGUAGE_MAP:
        lang = _UI_OUTPUT_LANGUAGE_MAP[raw]
    else:
        lang = raw.lower()

    return None if lang == "english" else lang


def _enforce_summary_language(
    state: Dict[str, Any],
    expected_language: Optional[str],
) -> Dict[str, Any]:
    """UI-level safeguard: force final_summary to the selected language."""
    if not expected_language:
        state["output_language"] = None
        return state

    summary = str(state.get("final_summary") or "")
    if not summary.strip():
        state["output_language"] = expected_language
        return state

    try:
        from orchestration.llm import translate_text  # type: ignore[import]
        state["final_summary"] = translate_text(summary, expected_language)
        state["output_language"] = expected_language
    except Exception as exc:
        st.warning(f"Language translation safeguard failed: {exc}")
        state["output_language"] = expected_language
    return state


# ── Sidebar: data availability ────────────────────────────────────────────────

def _render_sidebar_availability() -> None:
    """Check and display backend health in the sidebar."""
    # Check if running on cloud
    is_cloud = os.getenv("STREAMLIT_CLOUD", "").lower() == "true" or "streamlit.cloud" in os.getenv("HOSTNAME", "")
    
    st.sidebar.header("Backend Status")
    
    if is_cloud:
        st.sidebar.warning("⚠️ Cloud Deployment")
        st.sidebar.caption("""
When hosting on Streamlit Cloud, backend services (PostgreSQL, Neo4j) 
must be accessible from the cloud. Options:
1. Deploy databases to the cloud
2. Use a VPN/tunnel to expose local services
3. Run Streamlit locally instead
        """)
    
    if st.sidebar.button("Refresh Status", key="refresh_avail"):
        st.session_state.pop("avail_report", None)

    if "avail_report" not in st.session_state:
        with st.sidebar:
            with st.spinner("Checking backends…"):
                try:
                    from orchestration.data_availability import check_all
                    st.session_state["avail_report"] = check_all()
                except Exception as exc:
                    st.session_state["avail_report"] = {"_error": str(exc)}

    report = st.session_state.get("avail_report", {})

    if "_error" in report:
        st.sidebar.error(f"Availability check failed: {report['_error']}")
        return

    tiers = {
        "PostgreSQL":  report.get("postgres", {}),
        "Neo4j":       report.get("neo4j", {}),
        "DeepSeek":    report.get("deepseek", {}),
    }

    for name, info in tiers.items():
        reachable = info.get("reachable", False)
        icon = "🟢" if reachable else "🔴"
        st.sidebar.write(f"{icon} **{name}**")

    degraded = report.get("degraded_tiers", [])
    if degraded:
        st.sidebar.warning("Degraded: " + ", ".join(degraded))

    fully_ready = report.get("tickers_fully_ready", [])
    partially   = report.get("tickers_partially_ready", [])

    if fully_ready:
        st.sidebar.success("Full data: " + " · ".join(fully_ready))
    if partially:
        st.sidebar.info("Partial data: " + " · ".join(partially))

    summary = report.get("summary", "")
    if summary:
        with st.sidebar.expander("Details"):
            st.caption(summary)


# ── Agent result renderers ────────────────────────────────────────────────────

def _render_thinking_trace(trace: List[Dict[str, Any]]) -> None:
    """Render the pipeline thinking trace as a collapsible timeline."""
    if not trace:
        return
    with st.expander("Thinking process", expanded=False):
        for step in trace:
            ts     = step.get("ts", "")
            symbol = step.get("symbol", "OK")
            name   = step.get("name", "")
            detail = step.get("detail", "")
            color  = "status-err" if symbol == "!!" else "status-ok"
            icon   = "✗" if symbol == "!!" else "✓"
            st.markdown(
                f"<span class='{color}'>{icon}</span> "
                f"<code>{ts}</code> **{name}**"
                + (f"<br><span style='color:#888;font-size:0.85em;margin-left:1.5em'>{detail}</span>" if detail else ""),
                unsafe_allow_html=True,
            )


def _render_business_analyst(output: Dict[str, Any], ticker: str) -> None:
    """Render Business Analyst agent card."""
    with st.expander(f"Business Analyst — {ticker}", expanded=True):
        crag = output.get("crag_status", "—")
        confidence = output.get("confidence")
        crag_color = (
            "status-ok" if crag == "CORRECT"
            else "status-warn" if crag == "AMBIGUOUS"
            else "status-err"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**CRAG Status:** <span class='{crag_color}'>{crag}</span> "
                        f"(confidence: {_fmt(confidence)})", unsafe_allow_html=True)

        # Sentiment
        sentiment = output.get("sentiment") or {}
        if sentiment:
            st.markdown("**Sentiment**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Bullish", _pct(sentiment.get("bullish_pct")))
            c2.metric("Bearish", _pct(sentiment.get("bearish_pct")))
            c3.metric("Neutral", _pct(sentiment.get("neutral_pct")))
            c4.metric("Trend", str(sentiment.get("trend", "—")).capitalize())

        # Competitive moat
        moat = output.get("competitive_moat") or {}
        if moat:
            st.markdown("**Competitive Moat**")
            rating = moat.get("rating", "—")
            strengths = moat.get("key_strengths") or []
            st.markdown(f"Rating: **{rating}**")
            if strengths:
                for s in strengths[:5]:
                    st.markdown(f"- {s}")

        # Key risks
        risks = output.get("key_risks") or []
        if risks:
            st.markdown("**Key Risks**")
            for r in risks[:5]:
                if isinstance(r, dict):
                    desc = r.get("risk") or r.get("description") or str(r)
                    sev  = r.get("severity", "")
                    st.markdown(f"- {desc}" + (f" *(severity: {sev})*" if sev else ""))
                else:
                    st.markdown(f"- {r}")

        # Narrative
        narrative = output.get("competitive_moat_summary") or output.get("summary") or ""
        if not narrative:
            # Try to extract from nested moat dict
            narrative = moat.get("summary") or moat.get("narrative") or ""
        if narrative:
            with st.expander("Full narrative"):
                # Clean and wrap narrative text properly
                narrative_clean = _clean_text_for_display(narrative)
                st.markdown(
                    f'<div style="word-break: break-word; overflow-wrap: break-word; '
                    f'line-height: 1.6;">{narrative_clean}</div>',
                    unsafe_allow_html=True
                )

        # Missing context
        missing = output.get("missing_context") or []
        if missing:
            st.caption("Missing context: " + "; ".join(str(m) for m in missing[:3]))

        # Thinking trace
        _render_thinking_trace(output.get("thinking_trace") or [])

        # References / citation diagnostics
        _render_ba_references(output)


def _render_quant_fundamental(output: Dict[str, Any], ticker: str) -> None:
    """Render Quant Fundamental agent card."""
    with st.expander(f"Quant Fundamental — {ticker}", expanded=True):

        # Value factors table
        vf = output.get("value_factors") or {}
        qf = output.get("quality_factors") or {}
        mr = output.get("momentum_risk") or {}

        if vf or qf:
            st.markdown("**Factors**")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("*Value*")
                v_rows = {
                    "P/E":        vf.get("pe_trailing") or vf.get("pe_ratio"),
                    "EV/EBITDA":  vf.get("ev_ebitda"),
                    "P/FCF":      vf.get("p_fcf"),
                    "EV/Revenue": vf.get("ev_revenue"),
                }
                for label, val in v_rows.items():
                    st.markdown(f"<span class='metric-label'>{label}</span> "
                                f"<span class='metric-value'>{_fmt(val)}</span>",
                                unsafe_allow_html=True)

            with col2:
                st.markdown("*Quality*")
                q_rows = {
                    "ROE":            qf.get("roe"),
                    "ROIC":           qf.get("roic"),
                    "Piotroski F":    qf.get("piotroski_f_score"),
                    "Beneish M":      qf.get("beneish_m_score"),
                }
                for label, val in q_rows.items():
                    st.markdown(f"<span class='metric-label'>{label}</span> "
                                f"<span class='metric-value'>{_fmt(val)}</span>",
                                unsafe_allow_html=True)

        # Momentum/Risk
        if mr:
            st.markdown("**Momentum & Risk**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Beta (60d)",    _fmt(mr.get("beta_60d")))
            c2.metric("Sharpe (12m)",  _fmt(mr.get("sharpe_ratio_12m") or mr.get("sharpe_12m")))
            c3.metric("Return (12m)",  _pct(mr.get("return_12m_pct") or mr.get("return_12m")))

        # Anomaly flags
        anomalies = output.get("anomaly_flags") or []
        if anomalies:
            st.markdown("**Anomaly Flags**")
            for flag in anomalies[:5]:
                if isinstance(flag, dict):
                    metric = flag.get("metric") or flag.get("field", "?")
                    z      = flag.get("z_score")
                    desc   = flag.get("description") or flag.get("note", "")
                    st.warning(f"{metric}: z={_fmt(z)} — {desc}")
                else:
                    st.warning(str(flag))

        # LLM narrative
        narrative = output.get("quantitative_summary") or output.get("summary") or ""
        if narrative:
            with st.expander("Quantitative narrative"):
                # Clean and wrap narrative text properly
                narrative_clean = _clean_text_for_display(narrative)
                st.markdown(
                    f'<div style="word-break: break-word; overflow-wrap: break-word; '
                    f'line-height: 1.6;">{narrative_clean}</div>',
                    unsafe_allow_html=True
                )

        # Thinking trace
        _render_thinking_trace(output.get("thinking_trace") or [])


def _render_financial_modelling(output: Dict[str, Any], ticker: str) -> None:
    """Render Financial Modelling agent card."""
    with st.expander(f"Financial Modelling — {ticker}", expanded=True):

        valuation = output.get("valuation") or {}
        dcf       = valuation.get("dcf") or {}
        comps     = valuation.get("comps") or {}
        technicals = output.get("technicals") or {}
        current_price = output.get("current_price")

        # DCF intrinsic values
        if dcf:
            st.markdown("**DCF Valuation**")
            base  = dcf.get("intrinsic_value_base")
            bear  = dcf.get("intrinsic_value_bear")
            bull  = dcf.get("intrinsic_value_bull")
            wacc  = dcf.get("wacc_used") or dcf.get("wacc")
            upside = None
            if base and current_price:
                try:
                    upside = (float(base) / float(current_price) - 1) * 100
                except (TypeError, ValueError):
                    pass

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Bear",   f"${_fmt(bear)}")
            c2.metric("Base",   f"${_fmt(base)}")
            c3.metric("Bull",   f"${_fmt(bull)}")
            c4.metric("WACC",   f"{float(wacc) * 100:.1f}%" if wacc is not None else "—")
            if current_price:
                c5.metric("Current Price", f"${_fmt(current_price)}")

            if upside is not None:
                delta_str = f"{upside:+.1f}% vs current"
                color = "status-ok" if upside > 0 else "status-err"
                st.markdown(f"Base upside: <span class='{color}'>{delta_str}</span>",
                            unsafe_allow_html=True)

            # Sensitivity matrix
            sensitivity = dcf.get("sensitivity_matrix")
            if sensitivity and isinstance(sensitivity, dict):
                try:
                    import pandas as pd
                    df = pd.DataFrame(sensitivity)
                    st.caption("DCF Sensitivity Matrix (intrinsic value by WACC × growth)")
                    st.dataframe(df, width="stretch")
                except Exception:
                    pass

        # Comps
        if comps:
            st.markdown("**Comparable Analysis**")
            peer_ev_ebitda = comps.get("peer_ev_ebitda_median") or comps.get("ev_ebitda")
            peer_pe        = comps.get("peer_pe_median") or comps.get("pe_trailing")
            implied_ev     = comps.get("implied_ev_ebitda_value")

            c1, c2, c3 = st.columns(3)
            c1.metric("EV/EBITDA",        _fmt(peer_ev_ebitda))
            c2.metric("P/E (trailing)",   _fmt(peer_pe))
            c3.metric("Implied Price (EV/EBITDA)", f"${_fmt(implied_ev)}")

        # Technicals
        if technicals:
            st.markdown("**Technicals**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RSI (14)",  _fmt(technicals.get("rsi_14")))
            c2.metric("MACD",      _fmt(technicals.get("macd") or technicals.get("macd_histogram")))
            c3.metric("ATR",       _fmt(technicals.get("atr_14") or technicals.get("atr")))
            c4.metric("SMA 50",    _fmt(technicals.get("sma_50")))

            signal = technicals.get("trend_signal") or technicals.get("signal")
            if signal:
                st.caption(f"Signal: **{signal}**")

        # Factor scores
        factors = output.get("factor_scores") or {}
        if factors:
            st.markdown("**Factor Scores**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Piotroski", _fmt(factors.get("piotroski_f_score")))
            c2.metric("Beneish M", _fmt(factors.get("beneish_m_score")))
            c3.metric("Altman Z",  _fmt(factors.get("altman_z_score")))

        # MoE Consensus — show per-persona narratives (Optimist / Pessimist / Realist)
        moe = output.get("moe_consensus") or {}
        consensus_narrative = moe.get("consensus_narrative") or ""
        personas = moe.get("personas") or []
        if consensus_narrative or personas:
            with st.expander("MoE Consensus reasoning", expanded=False):
                if consensus_narrative:
                    # Clean and wrap narrative text properly
                    cons_clean = _clean_text_for_display(consensus_narrative)
                    st.markdown(
                        f'<div style="word-break: break-word; overflow-wrap: break-word; '
                        f'line-height: 1.6;">**Consensus:** {cons_clean}</div>',
                        unsafe_allow_html=True
                    )
                for p in personas:
                    name_p = p.get("name") or p.get("persona") or "Agent"
                    narr_p = p.get("narrative") or p.get("reasoning") or ""
                    if narr_p:
                        # Clean and wrap persona narrative text properly
                        persona_clean = _clean_text_for_display(narr_p)
                        st.markdown(
                            f'<div style="word-break: break-word; overflow-wrap: break-word; '
                            f'line-height: 1.6;">**{name_p}:** {persona_clean}</div>',
                            unsafe_allow_html=True
                        )

        # Narrative
        narrative = output.get("quantitative_summary") or output.get("summary") or ""
        if narrative:
            with st.expander("Full narrative"):
                # Clean and wrap narrative text properly to prevent LaTeX interpretation
                narrative_clean = _sanitize_llm_markdown(narrative)
                import html
                escaped_narrative = html.escape(narrative_clean)
                st.markdown(
                    f'<div style="word-break: break-word; overflow-wrap: break-word; '
                    f'line-height: 1.6; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">'
                    f'{escaped_narrative}</div>',
                    unsafe_allow_html=True
                )

        # Thinking trace
        _render_thinking_trace(output.get("thinking_trace") or [])


def _render_web_search(output: Dict[str, Any], ticker: str) -> None:
    """Render Web Search agent card."""
    with st.expander(f"Web Search — {ticker}", expanded=True):

        sentiment = output.get("sentiment_signal", "—")
        confidence = output.get("confidence")
        fallback   = output.get("fallback_triggered", False)

        col1, col2 = st.columns([2, 1])
        with col1:
            color = (
                "status-ok" if "BULLISH" in str(sentiment).upper()
                else "status-err" if "BEARISH" in str(sentiment).upper()
                else "status-warn"
            )
            st.markdown(f"**Sentiment:** <span class='{color}'>{sentiment}</span> "
                        f"(confidence: {_fmt(confidence)})",
                        unsafe_allow_html=True)
        if fallback:
            st.warning("Fallback triggered — Perplexity API error or parse failure.")

        # Breaking news
        news = output.get("breaking_news") or []
        if news:
            st.markdown("**Breaking News**")
            for item in news[:5]:
                if isinstance(item, dict):
                    title   = item.get("title") or item.get("headline") or str(item)
                    summary = item.get("summary") or item.get("snippet") or ""
                    source  = item.get("source") or ""
                    st.markdown(f"**{title}**" + (f"  \n{summary}" if summary else "") +
                                (f"  \n*{source}*" if source else ""))
                    st.divider()
                else:
                    st.markdown(f"- {item}")

        # Risk flags
        risks = output.get("unknown_risk_flags") or []
        if risks:
            st.markdown("**Unknown Risk Flags**")
            for r in risks[:5]:
                if isinstance(r, dict):
                    desc = r.get("flag") or r.get("description") or str(r)
                    st.error(f"- {desc}")
                else:
                    st.error(f"- {r}")

        # Competitor signals
        comps = output.get("competitor_signals") or []
        if comps:
            st.markdown("**Competitor Signals**")
            for c in comps[:4]:
                if isinstance(c, dict):
                    company = c.get("company") or c.get("ticker") or "?"
                    signal  = c.get("signal") or c.get("summary") or str(c)
                    st.markdown(f"- **{company}**: {signal}")
                else:
                    st.markdown(f"- {c}")

        # Rationale
        rationale = output.get("sentiment_rationale") or ""
        if rationale:
            with st.expander("Sentiment rationale"):
                st.markdown(rationale)


def _render_stock_research(output: Dict[str, Any], ticker: str) -> None:
    """Render Stock Research agent card (broker reports + earnings call analysis)."""
    data_source = str(output.get("data_source", "pdf")).lower()
    source_badge = "PostgreSQL/pgvector" if data_source == "pg" else "PDF"
    with st.expander(f"Stock Research — {ticker}", expanded=True):

        # Data source badge
        badge_color = "status-ok" if data_source == "pg" else "status-warn"
        st.markdown(
            f"Data source: <span class='{badge_color}'>{source_badge}</span>",
            unsafe_allow_html=True,
        )

        # Error check
        err = output.get("error")
        if err:
            st.error(f"Agent error: {err}")

        # Transcript names
        latest = output.get("latest_transcript") or ""
        previous = output.get("previous_transcript") or ""
        if latest or previous:
            col1, col2 = st.columns(2)
            col1.caption(f"**Latest:** {latest}")
            col2.caption(f"**Previous:** {previous}")

        # NLP features
        features = output.get("features") or {}
        feat_latest = features.get("latest") or {}
        feat_prev   = features.get("previous") or {}
        kpi_diff    = features.get("kpi_diff") or {}
        if feat_latest:
            st.markdown("**NLP Signal Features (latest transcript)**")
            c1, c2, c3, c4 = st.columns(4)
            _hedge = feat_latest.get("hedge_ratio")
            _tone = round(1.0 - _hedge, 4) if _hedge is not None else None
            c1.metric("KPI Density",    _fmt(feat_latest.get("kpi_per_1k_words")))
            c2.metric("Hedge Ratio",    _fmt(_hedge))
            c3.metric("Evasive Count",  _fmt(feat_latest.get("evasive_count")))
            c4.metric("Tone Score",     _fmt(_tone))
            if feat_prev:
                _kpi_latest = feat_latest.get("kpi_per_1k_words")
                _kpi_prev   = feat_prev.get("kpi_per_1k_words")
                _hedge_prev = feat_prev.get("hedge_ratio")
                delta_kpi   = round(_kpi_latest - _kpi_prev, 4) if (_kpi_latest is not None and _kpi_prev is not None) else None
                delta_hedge = round((_hedge or 0) - (_hedge_prev or 0), 4) if (_hedge is not None and _hedge_prev is not None) else None
                if delta_kpi is not None or delta_hedge is not None:
                    st.caption(
                        f"Period-over-period: KPI density Δ{_fmt(delta_kpi)}  "
                        f"Hedge ratio Δ{_fmt(delta_hedge)}"
                    )

        # Broker rating distribution
        broker_parsed = output.get("broker_parsed") or []
        if broker_parsed:
            st.markdown("**Broker Ratings**")
            counts: Dict[str, int] = {"bullish": 0, "neutral": 0, "bearish": 0}
            for b in broker_parsed:
                r = (b.get("rating") or "neutral").lower()
                if r in counts:
                    counts[r] += 1
                else:
                    counts["neutral"] += 1
            c1, c2, c3 = st.columns(3)
            c1.metric("Bullish",  counts["bullish"])
            c2.metric("Neutral",  counts["neutral"])
            c3.metric("Bearish",  counts["bearish"])

            # Per-broker price targets
            pt_rows = [
                b for b in broker_parsed
                if b.get("price_target") not in (None, "", "N/A")
            ]
            if pt_rows:
                with st.expander("Price targets by broker", expanded=False):
                    for b in pt_rows:
                        inst   = b.get("institution") or _parse_institution_from_doc_name(b.get("doc_name", ""))
                        rating = (b.get("rating") or "neutral").capitalize()
                        pt     = b.get("price_target") or "—"
                        color  = (
                            "status-ok"  if "bullish" in rating.lower() else
                            "status-err" if "bearish" in rating.lower() else
                            "status-warn"
                        )
                        st.markdown(
                            f"**{inst}** — <span class='{color}'>{rating}</span> | PT: {pt}",
                            unsafe_allow_html=True,
                        )

        # Three main analysis sections
        transcript_comparison = output.get("transcript_comparison") or ""
        qa_behavior           = output.get("qa_behavior") or ""
        broker_consensus      = output.get("broker_consensus") or ""

        if transcript_comparison:
            with st.expander("Transcript comparison (YoY)", expanded=False):
                st.markdown(transcript_comparison)
        if qa_behavior:
            with st.expander("Q&A behaviour analysis", expanded=False):
                st.markdown(qa_behavior)
        if broker_consensus:
            with st.expander("Broker consensus synthesis", expanded=False):
                st.markdown(broker_consensus)

        # Thinking trace
        _render_thinking_trace(output.get("thinking_trace") or [])

        # References / document citations
        _render_stock_research_references(output)


def _render_macro(output: Dict[str, Any], ticker: str) -> None:
    """Render Macro agent card (macroeconomic analysis)."""
    data_source = str(output.get("data_source", "none")).lower()
    source_badge = "PostgreSQL/Neo4j" if data_source in ("pg", "neo4j") else data_source
    with st.expander(f"Macro Analysis — {ticker}", expanded=True):

        st.markdown(
            f"Data source: <span class='status-ok'>{source_badge}</span>",
            unsafe_allow_html=True,
        )

        err = output.get("error")
        if err:
            st.error(f"Agent error: {err}")

        regime = output.get("regime", "")
        if regime:
            st.markdown(f"**Market Regime:** {regime}")

        macro_themes = output.get("macro_themes") or []
        if macro_themes:
            st.markdown("**Macro Themes**")
            for theme in macro_themes:
                direction = theme.get("direction", "neutral")
                confidence = theme.get("confidence", 0)
                st.markdown(
                    f"- **{theme.get('theme', '')}** — {direction} "
                    f"(confidence: {confidence:.0%})"
                )

        per_report_summaries = output.get("per_report_summaries") or []
        if per_report_summaries:
            with st.expander("Report Summaries", expanded=False):
                for report in per_report_summaries:
                    st.markdown(f"**{report.get('report_name', '')}**")
                    st.markdown(f"_{report.get('summary', '')}_")
                    st.caption(f"Stock relevance: {report.get('stock_relevance', '')}")

        top_drivers = output.get("top_macro_drivers") or []
        if top_drivers:
            st.markdown(f"**Top Macro Drivers:** {', '.join(top_drivers)}")

        top_risk = output.get("top_risk", "")
        if top_risk:
            st.markdown(f"**Top Risk:** {top_risk}")

        risk_scenario = output.get("risk_scenario", "")
        if risk_scenario:
            st.markdown(f"**Risk Scenario:** {risk_scenario}")

        citations = output.get("citations") or []
        if citations:
            with st.expander(f"Citations ({len(citations)})", expanded=False):
                for c in citations:
                    st.caption(c.get("doc_name", ""))


def _render_insider_news(output: Dict[str, Any], ticker: str) -> None:
    """Render Insider News agent card (insider trading + news sentiment)."""
    data_source = str(output.get("data_source", "pg")).lower()
    source_badge = "PostgreSQL" if data_source == "pg" else data_source
    with st.expander(f"Insider & News Analysis — {ticker}", expanded=True):

        st.markdown(
            f"Data source: <span class='status-ok'>{source_badge}</span>",
            unsafe_allow_html=True,
        )

        data_coverage = output.get("data_coverage") or {}
        insider_count = data_coverage.get("insider_transactions_count", 0)
        news_count = data_coverage.get("news_articles_count", 0)
        date_range = data_coverage.get("date_range", "N/A")

        c1, c2, c3 = st.columns(3)
        c1.metric("Insider Transactions", insider_count)
        c2.metric("News Articles", news_count)
        c3.caption(f"Date Range: {date_range}")

        err = output.get("error")
        if err:
            st.error(f"Agent error: {err}")

        insider_analysis = output.get("insider_analysis") or {}
        if insider_analysis:
            with st.expander("Insider Activity Analysis", expanded=True):
                activity = insider_analysis.get("activity_summary", "")
                if activity:
                    st.markdown(f"**Summary:** {activity}")

                buy_sell = insider_analysis.get("buy_sell_ratio")
                if buy_sell:
                    st.metric("Buy/Sell Ratio", f"{buy_sell:.2f}")

                net_position = insider_analysis.get("net_position", "")
                if net_position:
                    st.markdown(f"**Net Position:** {net_position}")

                conviction = insider_analysis.get("conviction", "")
                if conviction:
                    st.markdown(f"**Conviction:** {conviction}")

                sentiment = insider_analysis.get("insider_sentiment", "")
                if sentiment:
                    st.markdown(f"**Sentiment:** {sentiment}")

                red_flags = insider_analysis.get("red_flags") or []
                if red_flags:
                    st.markdown("**Red Flags:**")
                    for flag in red_flags:
                        st.markdown(f"- {flag}")

        news_analysis = output.get("news_analysis") or {}
        if news_analysis:
            with st.expander("News Sentiment Analysis", expanded=True):
                sentiment_summary = news_analysis.get("sentiment_summary", "")
                if sentiment_summary:
                    st.markdown(f"**Summary:** {sentiment_summary}")

                avg_score = news_analysis.get("avg_sentiment_score")
                if avg_score is not None:
                    st.metric("Avg Sentiment Score", f"{avg_score:.2f}")

                sentiment_trend = news_analysis.get("sentiment_trend", "")
                if sentiment_trend:
                    st.markdown(f"**Trend:** {sentiment_trend}")

                positive_catalysts = news_analysis.get("positive_catalysts") or []
                negative_catalysts = news_analysis.get("negative_catalysts") or []
                if positive_catalysts:
                    st.markdown("**Positive Catalysts:**")
                    for cat in positive_catalysts:
                        st.markdown(f"- {cat}")
                if negative_catalysts:
                    st.markdown("**Negative Catalysts:**")
                    for cat in negative_catalysts:
                        st.markdown(f"- {cat}")

        investment_thesis = output.get("investment_thesis") or {}
        if investment_thesis:
            with st.expander("Investment Thesis", expanded=True):
                combined = investment_thesis.get("combined_thesis", "")
                if combined:
                    st.markdown(combined)

                bull_case = investment_thesis.get("bull_case", "")
                if bull_case:
                    st.markdown(f"**Bull Case:** {bull_case}")

                bear_case = investment_thesis.get("bear_case", "")
                if bear_case:
                    st.markdown(f"**Bear Case:** {bear_case}")

                recommendation = investment_thesis.get("recommendation", "")
                if recommendation:
                    st.markdown(f"**Recommendation:** {recommendation}")

                conviction = investment_thesis.get("conviction", "")
                if conviction:
                    st.markdown(f"**Conviction:** {conviction}")

        citations = output.get("citations") or []
        if citations:
            with st.expander(f"Citations ({len(citations)})", expanded=False):
                for c in citations:
                    st.caption(c.get("doc_name", ""))


def _render_visualisations(state: Dict[str, Any]) -> None:
    """Render Plotly charts driven by planner chart_hints + available data.

    Called once after the pipeline completes, before agent detail cards.
    For multi-ticker queries the charts for each ticker are shown in tabs.

    Chart set mirrors buy-side equity research report structure:
      Section 1 — Price & Performance (stubs if no price history)
      Section 2 — P&L Trends: Revenue, Margins, Net Income, EBITDA (stub), EPS
      Section 3 — Valuation: Football Field, DCF Scenarios, Sensitivity Heatmap
      Section 4 — Peer Comparables
      Section 5 — Cash Flow: FCF (stub)
      Section 6 — Technical Indicators
      Section 7 — Sentiment & Quality Factors
      Section 8 — MoE Analyst Consensus
    """
    plan        = state.get("plan") or {}
    chart_hints = plan.get("chart_hints") or []
    tickers: List[str] = state.get("tickers") or []
    if not tickers and state.get("ticker"):
        tickers = [state["ticker"]]
    if not tickers:
        return

    # Agent outputs per ticker (positional match)
    fm_outputs = state.get("financial_modelling_outputs") or (
        [state["financial_modelling_output"]] if state.get("financial_modelling_output") else []
    )
    qf_outputs = state.get("quant_fundamental_outputs") or (
        [state["quant_fundamental_output"]] if state.get("quant_fundamental_output") else []
    )
    ba_outputs = state.get("business_analyst_outputs") or (
        [state["business_analyst_output"]] if state.get("business_analyst_output") else []
    )

    # Full default chart suite — used when planner doesn't specify hints
    _DEFAULT_HINTS = [
        # Price & performance (stubs — no historical series in pipeline output)
        "price_history",
        "price_performance",
        # P&L trends
        "revenue_trend",
        "margin_trends",
        "eps_trend",
        "ebitda_trend",
        # Valuation
        "football_field",
        "dcf_waterfall",
        "sensitivity_heatmap",
        # Peer comps
        "peer_comps",
        # Cash flow
        "fcf_trend",
        # Technicals
        "technicals",
        # Sentiment & quality
        "sentiment_donut",
        "factor_scorecard",
        # MoE consensus
        "moe_consensus",
    ]

    # Fallback: if chart_hints is empty, use the full default suite
    if not chart_hints:
        chart_hints = list(_DEFAULT_HINTS)
    else:
        # Deduplicate while preserving order
        seen: set = set()
        deduped: List[str] = []
        for h in chart_hints:
            if h not in seen:
                seen.add(h)
                deduped.append(h)
        chart_hints = deduped

    st.subheader("Visualisations")

    # If multiple tickers → tabbed layout; single ticker → plain layout
    if len(tickers) > 1:
        tab_objs = st.tabs(tickers)
    else:
        tab_objs = [st.container()]

    for idx, (ticker, tab) in enumerate(zip(tickers, tab_objs)):
        fm   = fm_outputs[idx] if idx < len(fm_outputs) else {}
        qf   = qf_outputs[idx] if idx < len(qf_outputs) else {}
        ba   = ba_outputs[idx] if idx < len(ba_outputs) else {}

        valuation     = (fm or {}).get("valuation") or {}
        dcf           = valuation.get("dcf") or {}
        comps         = valuation.get("comps") or {}
        technicals    = (fm or {}).get("technicals") or {}
        moe           = (fm or {}).get("moe_consensus") or {}
        earnings_rec  = (fm or {}).get("earnings") or {}
        current_price = (fm or {}).get("current_price")
        price_hist    = (fm or {}).get("price_history") or []
        bench_hist    = (fm or {}).get("benchmark_history") or []
        q_trends      = (qf or {}).get("quarterly_trends") or []
        key_metrics   = (qf or {}).get("key_metrics") or {}
        sentiment     = (ba or {}).get("sentiment") or {}

        rendered = False

        with tab:
            for hint in chart_hints:

                # ── Price & Performance ──────────────────────────────────────
                if hint == "price_history":
                    st.plotly_chart(
                        chart_price_history(price_hist, technicals, ticker, current_price),
                        width="stretch",
                        key=f"price_history_{ticker}_{idx}",
                    )
                    rendered = True

                elif hint == "price_performance":
                    st.plotly_chart(
                        chart_price_performance(price_hist, bench_hist, ticker),
                        width="stretch",
                        key=f"price_perf_{ticker}_{idx}",
                    )
                    rendered = True

                # ── P&L Trends ───────────────────────────────────────────────
                elif hint == "revenue_trend":
                    st.plotly_chart(
                        chart_revenue_trend(q_trends, dcf, ticker),
                        width="stretch",
                        key=f"revenue_{ticker}_{idx}",
                    )
                    rendered = True

                elif hint == "margin_trends":
                    st.plotly_chart(
                        chart_margin_trends(q_trends, key_metrics, ticker),
                        width="stretch",
                        key=f"margins_{ticker}_{idx}",
                    )
                    rendered = True

                elif hint == "eps_trend":
                    st.plotly_chart(
                        chart_eps_trend(q_trends, ticker, earnings_rec),
                        width="stretch",
                        key=f"eps_{ticker}_{idx}",
                    )
                    rendered = True

                elif hint == "ebitda_trend":
                    st.plotly_chart(
                        chart_ebitda_trend(q_trends, ticker),
                        width="stretch",
                        key=f"ebitda_{ticker}_{idx}",
                    )
                    rendered = True

                # ── Valuation ────────────────────────────────────────────────
                elif hint == "football_field":
                    st.plotly_chart(
                        chart_football_field(dcf, comps, current_price, ticker),
                        width="stretch",
                        key=f"football_{ticker}_{idx}",
                    )
                    rendered = True

                elif hint in ("dcf_waterfall", "dcf_scenarios") and dcf:
                    st.plotly_chart(
                        chart_dcf_waterfall(dcf, current_price, ticker),
                        width="stretch",
                        key=f"dcf_{hint}_{ticker}_{idx}",
                    )
                    rendered = True

                elif hint == "sensitivity_heatmap":
                    sm = dcf.get("sensitivity_matrix")
                    if sm and isinstance(sm, dict):
                        st.plotly_chart(
                            chart_sensitivity_heatmap(sm, ticker, current_price),
                            width="stretch",
                            key=f"sensitivity_{ticker}_{idx}",
                        )
                        rendered = True

                # ── Peer Comps ───────────────────────────────────────────────
                elif hint == "peer_comps":
                    st.plotly_chart(
                        chart_peer_comps(comps, ticker),
                        width="stretch",
                        key=f"peer_comps_{ticker}_{idx}",
                    )
                    rendered = True

                # ── Cash Flow ────────────────────────────────────────────────
                elif hint == "fcf_trend":
                    st.plotly_chart(
                        chart_fcf_trend(q_trends, key_metrics, ticker),
                        width="stretch",
                        key=f"fcf_{ticker}_{idx}",
                    )
                    rendered = True

                # ── Technicals ───────────────────────────────────────────────
                elif hint == "technicals":
                    st.plotly_chart(
                        chart_technicals(technicals, ticker, current_price),
                        width="stretch",
                        key=f"technicals_{ticker}_{idx}",
                    )
                    rendered = True

                # ── Sentiment & Quality ──────────────────────────────────────
                elif hint == "sentiment_donut":
                    st.plotly_chart(
                        chart_sentiment_donut(sentiment, ticker),
                        width="stretch",
                        key=f"sentiment_{ticker}_{idx}",
                    )
                    rendered = True

                elif hint in ("factor_scorecard", "factor_radar") and qf:
                    st.plotly_chart(
                        chart_factor_scorecard(qf, ticker),
                        width="stretch",
                        key=f"factor_{hint}_{ticker}_{idx}",
                    )
                    rendered = True

                elif hint == "altman_z" and qf:
                    st.plotly_chart(
                        chart_altman_z(qf, ticker),
                        width="stretch",
                        key=f"altman_z_{ticker}_{idx}",
                    )
                    rendered = True

                # ── MoE Consensus ────────────────────────────────────────────
                elif hint == "moe_consensus":
                    st.plotly_chart(
                        chart_moe_consensus(moe, current_price, ticker),
                        width="stretch",
                        key=f"moe_{ticker}_{idx}",
                    )
                    rendered = True

                # ── Backward-compat: quarterly_trends ────────────────────────
                elif hint == "quarterly_trends" and q_trends:
                    st.plotly_chart(
                        chart_quarterly_trends(q_trends, ticker),
                        width="stretch",
                        key=f"q_trends_{ticker}_{idx}",
                    )
                    rendered = True

            if not rendered:
                st.caption(f"No chart data available for {ticker}.")


def _render_agent_outputs(state: Dict[str, Any]) -> None:
    """Render all agent result cards from the final state."""
    tickers: List[str] = state.get("tickers") or []
    if not tickers and state.get("ticker"):
        tickers = [state["ticker"]]

    # ── Planner Reasoning ────────────────────────────────────────────────────
    plan = state.get("plan") or {}
    plan_reasoning = plan.get("reasoning") or ""
    planner_trace  = state.get("planner_trace") or ""
    if plan_reasoning or planner_trace:
        with st.expander("Planner reasoning", expanded=False):
            if planner_trace:
                st.markdown("**Chain-of-thought (DeepSeek reasoner):**")
                st.markdown(
                    f"<div style='font-size:0.85em;color:#9ca3af;white-space:pre-wrap'>{planner_trace}</div>",
                    unsafe_allow_html=True,
                )
                if plan_reasoning:
                    st.divider()
            if plan_reasoning:
                st.markdown(plan_reasoning)

    # ── Summarizer Thinking Trace ─────────────────────────────────────────────
    summarizer_trace = state.get("summarizer_trace") or ""
    if summarizer_trace:
        with st.expander("Summarizer reasoning (chain-of-thought)", expanded=False):
            st.markdown(
                f"<div style='font-size:0.85em;color:#9ca3af;white-space:pre-wrap'>{summarizer_trace}</div>",
                unsafe_allow_html=True,
            )

    # ── Business Analyst ─────────────────────────────────────────────────────
    ba_outputs: List[Dict] = state.get("business_analyst_outputs") or []
    if not ba_outputs and state.get("business_analyst_output"):
        ba_outputs = [state["business_analyst_output"]]
    for i, out in enumerate(ba_outputs):
        t = tickers[i] if i < len(tickers) else out.get("ticker", "?")
        _render_business_analyst(out, t)

    # ── Quant Fundamental ────────────────────────────────────────────────────
    qf_outputs: List[Dict] = state.get("quant_fundamental_outputs") or []
    if not qf_outputs and state.get("quant_fundamental_output"):
        qf_outputs = [state["quant_fundamental_output"]]
    for i, out in enumerate(qf_outputs):
        t = tickers[i] if i < len(tickers) else out.get("ticker", "?")
        _render_quant_fundamental(out, t)

    # ── Financial Modelling ──────────────────────────────────────────────────
    fm_outputs: List[Dict] = state.get("financial_modelling_outputs") or []
    if not fm_outputs and state.get("financial_modelling_output"):
        fm_outputs = [state["financial_modelling_output"]]
    for i, out in enumerate(fm_outputs):
        t = tickers[i] if i < len(tickers) else out.get("ticker", "?")
        _render_financial_modelling(out, t)

    # ── Web Search ───────────────────────────────────────────────────────────
    ws_outputs: List[Dict] = state.get("web_search_outputs") or []
    if not ws_outputs and state.get("web_search_output"):
        ws_outputs = [state["web_search_output"]]
    for i, out in enumerate(ws_outputs):
        t = tickers[i] if i < len(tickers) else out.get("ticker", "?")
        _render_web_search(out, t)

    # ── Stock Research ───────────────────────────────────────────────────────
    sr_outputs: List[Dict] = state.get("stock_research_outputs") or []
    if not sr_outputs and state.get("stock_research_output"):
        sr_outputs = [state["stock_research_output"]]
    for i, out in enumerate(sr_outputs):
        t = tickers[i] if i < len(tickers) else out.get("ticker", "?")
        _render_stock_research(out, t)

    # ── Macro ────────────────────────────────────────────────────────────────
    macro_outputs: List[Dict] = state.get("macro_outputs") or []
    if not macro_outputs and state.get("macro_output"):
        macro_outputs = [state["macro_output"]]
    for i, out in enumerate(macro_outputs):
        t = tickers[i] if i < len(tickers) else out.get("ticker", "?")
        _render_macro(out, t)

    # ── Insider News ─────────────────────────────────────────────────────────
    insider_news_outputs: List[Dict] = state.get("insider_news_outputs") or []
    if not insider_news_outputs and state.get("insider_news_output"):
        insider_news_outputs = [state["insider_news_output"]]
    for i, out in enumerate(insider_news_outputs):
        t = tickers[i] if i < len(tickers) else out.get("ticker", "?")
        _render_insider_news(out, t)


def _render_plan_info(plan: Optional[Dict[str, Any]], state: Dict[str, Any]) -> None:
    """Show plan metadata in a compact info bar."""
    if not plan:
        return
    ticker   = state.get("ticker") or "—"
    tickers  = state.get("tickers") or []
    complexity = plan.get("complexity") or "—"
    agents_run = [
        a for a, flag in [
            ("BA",  state.get("run_business_analyst")),
            ("QF",  state.get("run_quant_fundamental")),
            ("FM",  state.get("run_financial_modelling")),
            ("WS",  state.get("run_web_search")),
            ("SR",  state.get("run_stock_research")),
        ] if flag
    ]
    react_iter = state.get("react_iteration") or 0
    react_max  = state.get("react_max_iterations") or 1

    st.info(
        f"**Ticker(s):** {', '.join(tickers) or ticker} | "
        f"**Complexity:** {complexity} | "
        f"**Agents:** {', '.join(agents_run) or '—'} | "
        f"**ReAct passes:** {react_iter}/{react_max}"
    )


def _render_planner_feedback_learning(state: Dict[str, Any]) -> None:
    """Show how planner learns from worst historical feedback runs."""
    try:
        from orchestration import feedback
    except Exception as exc:
        with st.expander("Planner feedback learning", expanded=False):
            st.caption(f"Unavailable: {exc}")
        return

    with st.expander("Planner feedback learning", expanded=False):
        user_query = str(state.get("user_query") or "")
        st.caption("This panel previews the worst-case context block injected into planner routing.")

        try:
            worst_cases = feedback.get_worst_cases(limit=5)
        except Exception as exc:
            st.warning(f"Could not fetch worst cases: {exc}")
            return

        if not worst_cases:
            st.info("No worst-case context injected (cold start or feedback DB unavailable).")
            return

        lines = ["PAST FAILURES TO AVOID (worst scored runs - learn from these):"]
        for i, w in enumerate(worst_cases, 1):
            tags = w.get("issue_tags") or []
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except Exception:
                    tags = []
            weaknesses = w.get("weaknesses") or []
            if isinstance(weaknesses, str):
                try:
                    weaknesses = json.loads(weaknesses)
                except Exception:
                    weaknesses = [weaknesses]

            comment = str(w.get("comment") or "")
            human_signal = ""
            if w.get("helpful") is False:
                human_signal = f" | USER DOWNVOTE: {', '.join(str(t) for t in tags)}"
                if comment:
                    human_signal += f" - '{comment}'"

            lines.append(
                f"  [{i}] Query: \"{str(w.get('user_query') or '')[:80]}\" | "
                f"Ticker: {w.get('ticker', 'N/A')} | "
                f"Score: {float(w.get('overall_score') or 0):.1f}/10 | "
                f"Blamed: {w.get('agent_blamed', 'unknown')} | "
                f"Weaknesses: {'; '.join(str(x) for x in weaknesses[:2]) or 'N/A'}"
                f"{human_signal}"
            )

        lines.append(
            "ACTION: Adjust your routing to avoid repeating these failure patterns. "
            "If a similar query + ticker combination appears, increase react_max_iterations "
            "or force run_web_search=true as a fallback."
        )

        worst_case_context = "\n".join(lines)

        st.markdown("**How prompt is improved (planner user-message augmentation):**")
        preview = f"{user_query}\n\n---\n{worst_case_context}\n---"
        st.code(preview, language="text")
        st.caption(f"Injected examples: {len(worst_cases)}")


# ── Feedback Widget ───────────────────────────────────────────────────────────

def _render_feedback_widget(run_id: str, state: Dict[str, Any]) -> None:
    """Render the explicit user feedback widget."""
    from orchestration import feedback
    
    feedback_key = f"feedback_submitted_{run_id}"
    
    if feedback_key not in st.session_state:
        st.session_state[feedback_key] = False
    
    with st.expander("Was this analysis helpful?", expanded=True):
        col1, col2 = st.columns([1, 4])
        
        with col1:
            helpful = st.radio(
                "Your feedback",
                options=["Yes", "No"],
                horizontal=True,
                key=f"helpful_radio_{run_id}",
            )
        
        with col2:
            issue_options = [
                "Wrong numbers",
                "Missing citations",
                "Analysis too shallow",
                "Missing sections",
                "Too long/verbose",
                "Other",
            ]
            selected_issues = st.multiselect(
                "What could be better? (select all that apply)",
                options=issue_options,
                key=f"issues_{run_id}",
            )
        
        comment = st.text_area(
            "Additional comments (optional)",
            placeholder="Tell us more about your experience...",
            key=f"comment_{run_id}",
        )
        
        if st.button("Submit Feedback", key=f"submit_{run_id}"):
            is_helpful = helpful == "Yes"
            
            success = feedback.store_user_feedback(
                run_id=run_id,
                session_id=st.session_state.get("session_id"),
                helpful=is_helpful,
                comment=comment if comment else None,
                issue_tags=selected_issues if selected_issues else None,
                report_version=None,
            )
            
            if success:
                st.session_state[feedback_key] = True
                st.success("Thank you for your feedback!")
            else:
                st.error("Failed to submit feedback. Please try again.")
        
        if st.session_state[feedback_key]:
            st.info("You have already submitted feedback for this report.")


# ── Streaming runner ──────────────────────────────────────────────────────────

# Human-readable agent display names for the live progress panel
_AGENT_DISPLAY = {
    "business_analyst":   "Business Analyst",
    "quant_fundamental":  "Quant Fundamental",
    "financial_modelling":"Financial Modelling",
    "web_search":         "Web Search",
    "stock_research":     "Stock Research",
    "macro":              "Macro",
    "insider_news":       "Insider & News",
}


def _run_with_streaming(query: str, output_language: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Stream orchestration events into the UI; return final state on completion.

    Architecture
    ------------
    LangGraph's .stream() generator blocks the calling thread while each node
    runs.  The parallel_agents node runs all four agents concurrently inside a
    ThreadPoolExecutor, but only yields ONE event (when ALL agents finish).

    To show live per-agent progress while parallel_agents is executing, we:
      1. Run the LangGraph stream in a background thread, pushing (node_name,
         node_output) tuples into a thread-safe ``output_queue``.
      2. Subscribe to the session's ``progress_queue`` — nodes.py pushes a
         progress event each time an individual agent starts or finishes.
      3. The Streamlit main thread polls both queues in a tight loop, updating
         the UI as events arrive.
    """
    from orchestration.graph import (
        stream as orch_stream,
        subscribe_agent_progress,
        unsubscribe_agent_progress,
    )
    import uuid

    # ── Subscribe to progress queue BEFORE starting stream ───────────────────
    # Generate session_id first so we can subscribe before parallel_agents runs
    session_id = str(uuid.uuid4())[:12]
    progress_queue = subscribe_agent_progress(session_id)
    
    # ── Queues ───────────────────────────────────────────────────────────────
    # output_queue: node-level events from the LangGraph stream thread
    output_queue: queue.Queue = queue.Queue()
    session_id_holder: List[str] = [session_id]  # store for cleanup

    _STREAM_DONE = object()   # sentinel to signal stream exhaustion
    _STREAM_ERROR = object()  # sentinel to signal stream exception

    def _stream_worker() -> None:
        """Background thread: run the LangGraph generator and push events."""
        try:
            # Pass the pre-generated session_id to ensure consistency
            for node_name, node_output in orch_stream(query, session_id=session_id, output_language=output_language):
                output_queue.put((node_name, node_output))
        except Exception as exc:
            output_queue.put((_STREAM_ERROR, exc))
        finally:
            output_queue.put((_STREAM_DONE, None))

    thread = threading.Thread(target=_stream_worker, daemon=True)
    thread.start()

    # ── UI state ──────────────────────────────────────────────────────────────
    progress_area = st.empty()        # top-level node status lines
    agent_area    = st.empty()        # live per-agent progress during parallel_agents
    status_lines: List[str] = []
    final_state: Dict[str, Any] = {}
    pass_count = 0
    in_parallel_agents = False        # True while we're waiting for parallel_agents node

    # Track per-agent live state: {agent_name: "running"|"done"|"error"}
    agent_statuses: Dict[str, str] = {}
    agent_elapsed:  Dict[str, int]  = {}
    agent_excerpts: Dict[str, str]  = {}
    agent_thinking_traces: Dict[str, List[Dict[str, Any]]] = {}

    def _render_agent_progress() -> None:
        """Re-render the per-agent progress box."""
        if not agent_statuses:
            return
        lines = ["**Agents running:**"]
        for agent, disp in _AGENT_DISPLAY.items():
            status = agent_statuses.get(agent)
            if status is None:
                continue
            elapsed = agent_elapsed.get(agent, 0)
            if status == "running":
                icon = "⏳"
                tag  = "working…"
            elif status == "done":
                icon = "✅"
                tag  = f"done ({elapsed/1000:.1f}s)"
            else:
                icon = "❌"
                tag  = f"error ({elapsed/1000:.1f}s)"
            line = f"- {icon} **{disp}**: {tag}"
            excerpt = agent_excerpts.get(agent)
            if excerpt and status == "done":
                line += f"\n  *{excerpt}*"
            lines.append(line)
            # Show thinking trace in real-time for done agents
            thinking_trace = agent_thinking_traces.get(agent)
            if thinking_trace and status == "done":
                trace_lines = []
                for step in thinking_trace[-3:]:  # Show last 3 steps
                    ts = step.get("ts", "")
                    name = step.get("name", "")
                    detail = step.get("detail", "")
                    trace_lines.append(f"  `{ts}` {name}: {detail[:50]}...")
                if trace_lines:
                    lines.append("  **Thinking:**")
                    lines.extend(trace_lines)
        agent_area.markdown("\n".join(lines))

    try:
        with st.spinner("Running pipeline…"):
            stream_done = False
            while not stream_done:
                # ── Drain output_queue (node-level events) ────────────────
                while True:
                    try:
                        node_name, node_output = output_queue.get_nowait()
                    except queue.Empty:
                        break

                    if node_name is _STREAM_DONE:
                        stream_done = True
                        break
                    if node_name is _STREAM_ERROR:
                        raise node_output  # type: ignore[misc]

                    # ── Handle __session__ — already subscribed above ────────────────
                    if node_name == "__session__":
                        # Already subscribed to progress queue before stream started
                        continue

                    label, desc = NODE_LABELS.get(node_name, (node_name, ""))

                    if node_name == "parallel_agents":
                        pass_count += 1
                        in_parallel_agents = True  # Start tracking while agents run
                        agent_area.empty()           # clear per-agent box
                        agent_statuses.clear()
                        # Build dynamic agent list from which run_* flags are True
                        _run_flag_map = [
                            ("run_business_analyst",   "Business Analyst"),
                            ("run_quant_fundamental",  "Quant Fundamental"),
                            ("run_financial_modelling","Financial Modelling"),
                            ("run_web_search",         "Web Search"),
                            ("run_stock_research",     "Stock Research"),
                            ("run_macro",              "Macro"),
                            ("run_insider_news",       "Insider & News"),
                        ]
                        active_agents = [
                            name for key, name in _run_flag_map
                            if node_output.get(key)
                        ] if isinstance(node_output, dict) else []
                        agent_list_str = " · ".join(active_agents) if active_agents else "selected agents"
                        line = f"**Pass {pass_count} — {label}:** {agent_list_str}"

                    elif node_name == "react_check":
                        iteration = node_output.get("react_iteration") or pass_count
                        react_max = node_output.get("react_max_iterations") or 1
                        gaps = [
                            a for a, (out_key, run_key) in [
                                ("BA",  ("business_analyst_outputs",  "run_business_analyst")),
                                ("QF",  ("quant_fundamental_outputs", "run_quant_fundamental")),
                                ("FM",  ("financial_modelling_outputs","run_financial_modelling")),
                                ("WS",  ("web_search_outputs",         "run_web_search")),
                            ]
                            if node_output.get(run_key) and not node_output.get(out_key)
                        ]
                        gap_str = f" — retrying: {', '.join(gaps)}" if gaps else " — all agents OK"
                        line = f"**{label} (pass {iteration}/{react_max}):**{gap_str}"
                        if gaps:
                            in_parallel_agents = True  # about to re-enter parallel_agents

                    elif node_name == "planner":
                        ticker     = node_output.get("ticker") or "?"
                        tickers    = node_output.get("tickers") or []
                        plan       = node_output.get("plan") or {}
                        complexity = plan.get("complexity") or "?"
                        line = (
                            f"**{label}:** ticker={', '.join(tickers) or ticker}, "
                            f"complexity={complexity}"
                        )
                        in_parallel_agents = True  # next node will be parallel_agents

                    elif node_name == "memory_update":
                        line = f"**{label}:** {desc}"
                        in_parallel_agents = False

                    else:
                        line = f"**{label}:** {desc}"

                    status_lines.append(line)
                    progress_area.markdown("\n\n".join(status_lines))

                    if isinstance(node_output, dict):
                        final_state.update(node_output)

                # ── Drain progress_queue (per-agent events) ─────────────────────────
                # Always drain when we have events, not just during parallel_agents
                if progress_queue is not None:
                    drained = False
                    while not drained:
                        try:
                            event = progress_queue.get_nowait()
                        except queue.Empty:
                            drained = True
                            break
                        agent = event.get("agent", "")
                        if agent == "__done__":
                            drained = True
                            in_parallel_agents = False  # Agents finished
                            break
                        status = event.get("status", "")
                        elapsed = event.get("elapsed_ms", 0)
                        if status == "started":
                            agent_statuses[agent] = "running"
                        elif status == "done":
                            agent_statuses[agent] = "done"
                            agent_elapsed[agent]  = elapsed
                            excerpt = event.get("excerpt")
                            if excerpt:
                                agent_excerpts[agent] = excerpt
                            # Store thinking trace for real-time display
                            thinking_trace = event.get("thinking_trace")
                            if thinking_trace:
                                agent_thinking_traces[agent] = thinking_trace
                        elif status == "error":
                            agent_statuses[agent] = "error"
                            agent_elapsed[agent]  = elapsed
                    _render_agent_progress()

                if not stream_done:
                    time.sleep(0.05)  # 50ms poll interval

    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        if session_id_holder:
            unsubscribe_agent_progress(session_id_holder[0])
        return None

    if session_id_holder:
        unsubscribe_agent_progress(session_id_holder[0])

    progress_area.empty()
    agent_area.empty()
    return final_state


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("Agentic Investment Analyst")
        st.caption("Powered by LangGraph · DeepSeek · Neo4j · PostgreSQL")
        st.divider()
        
        # ── API Settings ───────────────────────────────────────────────────────
        st.markdown("### API Settings")
        
        # DeepSeek API Key (auto-loaded from env in Docker)
        env_deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        default_deepseek_api_key = st.session_state.get("deepseek_api_key") or env_deepseek_api_key

        deepseek_api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=default_deepseek_api_key,
            help="Enter your DeepSeek API key. Get one at https://platform.deepseek.com",
            key="deepseek_key_input"
        )

        if deepseek_api_key:
            st.session_state["deepseek_api_key"] = deepseek_api_key
            os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
            # Treat env-provided key as ready to use; explicit test is optional.
            if "api_verified" not in st.session_state:
                st.session_state["api_verified"] = True
        
        # Test API button
        if st.button("Test API Connection", key="test_api_btn"):
            if not deepseek_api_key:
                st.error("Please enter your DeepSeek API key first.")
            else:
                with st.spinner("Testing connection..."):
                    try:
                        import requests
                        response = requests.get(
                            "https://api.deepseek.com/v1/models",
                            headers={"Authorization": f"Bearer {deepseek_api_key}"},
                            timeout=10
                        )
                        if response.status_code == 200:
                            st.session_state["api_verified"] = True
                            st.session_state["deepseek_api_key"] = deepseek_api_key
                            os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
                            st.success("✓ API key is valid!")
                        else:
                            st.session_state["api_verified"] = False
                            st.error(f"✗ API error: {response.status_code}")
                    except Exception as e:
                        st.session_state["api_verified"] = False
                        st.error(f"✗ Connection failed: {e}")
        else:
            # Check if already verified
            if st.session_state.get("api_verified") and deepseek_api_key:
                os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
                st.success("✓ API verified")
        
        # Show API status
        if deepseek_api_key:
            st.success("✓ DeepSeek key loaded")
        else:
            st.warning("⚠️ API key required for analysis")

        st.divider()

        # Perplexity API Key (for web search agent)
        st.markdown("### Web Search (Optional)")
        st.caption("Required only when the analysis needs recent news or live web data.")
        perplexity_api_key = st.text_input(
            "Perplexity API Key",
            type="password",
            value=st.session_state.get("perplexity_api_key", os.getenv("PERPLEXITY_API_KEY", "")),
            help="Enter your Perplexity API key to enable the web search agent. Get one at https://www.perplexity.ai/settings/api",
            key="perplexity_key_input",
        )
        if perplexity_api_key:
            st.session_state["perplexity_api_key"] = perplexity_api_key
            os.environ["PERPLEXITY_API_KEY"] = perplexity_api_key
            st.success("✓ Perplexity key set")
        else:
            st.info("No Perplexity key — web search agent disabled")

        st.divider()
        _render_sidebar_availability()
        st.divider()
        st.markdown("**Supported tickers**")
        st.markdown("AAPL · MSFT · GOOGL · TSLA · NVDA")

    # ── Main area with tabs ─────────────────────────────────────────────────────
    st.title("Agentic Investment Analyst")
    
    tab1, tab2 = st.tabs(["Analysis", "Database Health"])
    
    with tab1:
        st.markdown(
            "Ask any investment question about AAPL, MSFT, GOOGL, TSLA, or NVDA. "
            "The orchestration pipeline will dispatch the appropriate agents in parallel "
            "and synthesise a research note."
        )

        # Example query buttons
        st.markdown("**Example queries:**")
        cols = st.columns(len(EXAMPLE_QUERIES))
        for col, q in zip(cols, EXAMPLE_QUERIES):
            if col.button(q[:40] + "…", key=f"ex_{q[:20]}", width="stretch"):
                st.session_state["query_input"] = q

        # Query input
        query = st.text_area(
            "Your investment question",
            value=st.session_state.get("query_input", ""),
            height=80,
            placeholder="e.g. What is Apple's competitive moat and valuation?",
            key="query_text",
        )
        st.session_state["query_input"] = query

        # Ticker override (optional — orchestration planner resolves automatically)
        with st.expander("Advanced options"):
            ticker_override = st.selectbox(
                "Force ticker (optional — overrides planner resolution)",
                options=["(auto)", *SUPPORTED_TICKERS],
                index=0,
                key="ticker_override",
            )
            if ticker_override != "(auto)":
                # Inject the ticker into the query if not already present
                query = query or ""
                if ticker_override not in query.upper():
                    query = f"[{ticker_override}] {query}"
            
            # Language selection for output
            output_language = st.selectbox(
                "Output language (translation)",
                options=["English", "Cantonese", "Mandarin", "Spanish", "French", "German", "Japanese", "Korean", "Portuguese", "Italian", "Dutch", "Russian", "Arabic", "Hindi", "Thai", "Vietnamese", "Indonesian"],
                index=0,
                key="output_language",
            )

    with tab2:
        st.markdown("### Database Health Check")
        st.markdown("Inspect PostgreSQL and Neo4j databases for data coverage and quality.")
        
        if st.button("Run Database Inspection", key="db_inspect"):
            with st.spinner("Connecting to databases..."):
                try:
                    # Import and run the inspection
                    import sys
                    from pathlib import Path
                    etl_root = _REPO_ROOT / "ingestion" / "etl"
                    if etl_root.exists():
                        sys.path.insert(0, str(etl_root))
                    from ingestion.etl.inspect_db import check_postgres, check_neo4j
                    
                    import io
                    from contextlib import redirect_stdout, redirect_stderr
                    
                    # PostgreSQL results
                    pg_output = io.StringIO()
                    try:
                        with redirect_stdout(pg_output), redirect_stderr(pg_output):
                            pg_ok = check_postgres()
                    except Exception as e:
                        pg_ok = False
                    
                    # Neo4j results  
                    neo4j_output = io.StringIO()
                    try:
                        with redirect_stdout(neo4j_output), redirect_stderr(neo4j_output):
                            neo4j_ok = check_neo4j()
                    except Exception as e:
                        neo4j_ok = False
                    
                    # Display summary cards
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("PostgreSQL", "✓ Connected" if pg_ok else "✗ Error", 
                                 "OK" if pg_ok else "Check logs")
                    with col2:
                        st.metric("Neo4j", "✓ Connected" if neo4j_ok else "✗ Error",
                                 "OK" if neo4j_ok else "Check logs")
                    
                    st.divider()
                    
                    # Show detailed results in expandable sections
                    with st.expander("PostgreSQL Details", expanded=True):
                        pg_result = pg_output.getvalue()
                        if pg_result:
                            # Parse and display key metrics
                            lines = pg_result.split('\n')
                            metrics = {}
                            for line in lines:
                                if ':' in line and ('rows' in line.lower() or 'coverage' in line.lower()):
                                    metrics[line.strip()] = True
                            
                            # Show as metrics
                            metric_cols = st.columns(3)
                            idx = 0
                            for line in lines:
                                if "raw_timeseries" in line and "rows" in line:
                                    try:
                                        count = int([x for x in line.split() if x.isdigit()][0])
                                        metric_cols[idx % 3].metric("Price Data", f"{count:,}")
                                        idx += 1
                                    except:
                                        pass
                                if "financial_statements" in line and "rows" in line:
                                    try:
                                        count = int([x for x in line.split() if x.isdigit()][0])
                                        metric_cols[idx % 3].metric("Financials", f"{count:,}")
                                        idx += 1
                                    except:
                                        pass
                                if "news_articles" in line and "rows" in line:
                                    try:
                                        count = int([x for x in line.split() if x.isdigit()][0])
                                        metric_cols[idx % 3].metric("News", f"{count:,}")
                                        idx += 1
                                    except:
                                        pass
                            
                            st.text("Raw Output:")
                            st.code(pg_result[:3000] if len(pg_result) > 3000 else pg_result, language="text")
                    
                    with st.expander("Neo4j Details", expanded=True):
                        neo4j_result = neo4j_output.getvalue()
                        
                        # Parse key metrics
                        lines = neo4j_result.split('\n')
                        idx = 0
                        metric_cols = st.columns(3)
                        for line in lines:
                            if "Company nodes" in line:
                                try:
                                    count = int([x for x in line.split() if x.isdigit()][0])
                                    metric_cols[idx % 3].metric("Companies", str(count))
                                    idx += 1
                                except:
                                    pass
                            if "Chunk nodes" in line:
                                try:
                                    count = int([x for x in line.split() if x.isdigit()][0])
                                    metric_cols[idx % 3].metric("Chunks", f"{count:,}")
                                    idx += 1
                                except:
                                    pass
                            if "embedding dimension" in line:
                                if "768" in line:
                                    metric_cols[idx % 3].metric("Embedding", "768-dim ✓")
                                idx += 1
                            
                            st.text("Raw Output:")
                            st.code(neo4j_result[:3000] if len(neo4j_result) > 3000 else neo4j_result, language="text")
                        
                except Exception as e:
                    st.error(f"Failed to inspect databases: {e}")
                    st.info("Make sure PostgreSQL and Neo4j are running.")

    run_btn = st.button("Run Analysis", type="primary", width="stretch")

    # ── Run ───────────────────────────────────────────────────────────────────
    if run_btn:
        # Check DeepSeek API key is present (auto-loaded from docker-compose env)
        deepseek_api_key = st.session_state.get("deepseek_api_key") or os.getenv("DEEPSEEK_API_KEY", "")
        if not deepseek_api_key:
            st.error("Please set a DeepSeek API key in environment or sidebar.")
            st.info("Tip: docker-compose already passes DEEPSEEK_API_KEY to the streamlit service.")
            st.stop()
        
        query = query or ""
        if not query.strip():
            st.warning("Please enter a question.")
            return

        # Clear previous results
        for key in ("last_state", "last_query"):
            st.session_state.pop(key, None)

        st.markdown("---")
        st.subheader("Live Progress")
        # Get output language from the widget's session state
        output_lang = st.session_state.get("output_language", "English")
        output_lang_param = _normalize_output_language(output_lang)
        final_state = _run_with_streaming(query.strip(), output_lang_param)

        if final_state:
            final_state = _enforce_summary_language(final_state, output_lang_param)

        if final_state:
            st.session_state["last_state"] = final_state
            st.session_state["last_query"] = query.strip()
            st.rerun()

    # ── Display persisted results ─────────────────────────────────────────────
    if "last_state" in st.session_state:
        state = st.session_state["last_state"]

        st.markdown("---")

        # Plan info bar
        _render_plan_info(state.get("plan"), state)

        # Planner learning from prior worst feedback
        _render_planner_feedback_learning(state)

        # Agent errors
        errors: Dict[str, str] = state.get("agent_errors") or {}
        if errors:
            st.error("**Agent errors detected:**")
            for agent, msg in errors.items():
                st.error(f"  • **{agent}**: {msg}")

        # Visualisations (chart library — driven by planner chart_hints)
        # Rendered BEFORE the research note so charts appear at the top.
        _render_visualisations(state)

        # Financial Data Tables
        _render_data_tables(state)

        # Final research note
        final_summary = state.get("final_summary") or ""
        if final_summary:
            st.subheader("Research Note")
            
            # Clean and display the text
            final_summary_clean = _sanitize_llm_markdown(final_summary)
            
            # Check if text has proper markdown structure
            has_markdown_sections = "\n## " in final_summary_clean or "\n### " in final_summary_clean
            
            # Show toggle for raw vs formatted view
            col1, col2 = st.columns([1, 5])
            with col1:
                show_raw = st.checkbox("Raw text", value=not has_markdown_sections, key="show_raw_text")
            with col2:
                pass

            with st.expander("Render debug", expanded=False):
                st.caption("Raw LLM preview (first 2000 chars)")
                st.code(final_summary_clean[:2000], language="markdown")
            
            if show_raw:
                # Use plain text display with proper wrapping (not markdown to avoid interpretation)
                import html
                escaped_text = html.escape(final_summary_clean)
                st.markdown(
                    f'<pre style="white-space: pre-wrap; word-wrap: break-word; '
                    f'overflow-wrap: break-word; font-family: monospace; '
                    f'padding: 1rem; background-color: #f0f0f0; border-radius: 4px; '
                    f'line-height: 1.5; font-size: 0.9em; color: #333;">{escaped_text}</pre>',
                    unsafe_allow_html=True
                )
            else:
                # For markdown view, ensure proper rendering with escaped special chars
                st.markdown('<div class="report-wrap">', unsafe_allow_html=True)
                st.markdown(final_summary_clean, unsafe_allow_html=False)  # Disable HTML to prevent injection
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.download_button(
                label="Download Research Note (.md)",
                data=final_summary,
                file_name=f"research_{state.get('ticker', 'query')}.md",
                mime="text/markdown",
            )

        # RLAIF Quality Score Display
        rl_scores = state.get("rl_feedback_scores")
        run_id = state.get("rl_feedback_run_id")
        if rl_scores:
            with st.expander("AI Quality Score (RLAIF)", expanded=False):
                overall = float(rl_scores.get("overall_score") or 0)
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Overall", f"{overall:.1f}/10")
                col2.metric("Citations", f"{float(rl_scores.get('citation_completeness') or 0):.1f}")
                col3.metric("Analysis", f"{float(rl_scores.get('analysis_depth') or 0):.1f}")
                col4.metric("Structure", f"{float(rl_scores.get('structure_compliance') or 0):.1f}")
                col5.metric("Language", f"{float(rl_scores.get('language_quality') or 0):.1f}")

                if overall < 7.0:
                    st.warning(
                        "⚠️ This report scored below the quality threshold (7.0). "
                        f"Agent blamed: {rl_scores.get('agent_blamed', 'unknown')}"
                    )

                weaknesses = rl_scores.get("weaknesses") or []
                if weaknesses:
                    st.markdown("**Areas for improvement:**")
                    for w in weaknesses[:3]:
                        st.markdown(f"- {w}")

                if run_id:
                    st.caption(f"Run ID: {run_id}")

        # Explicit User Feedback Widget
        run_id = state.get("rl_feedback_run_id")
        if run_id:
            _render_feedback_widget(run_id, state)

        # Per-agent result cards
        st.subheader("Agent Details")
        _render_agent_outputs(state)

        # ReAct trace (debug)
        react_steps = state.get("react_steps") or []
        if react_steps:
            with st.expander(f"ReAct trace ({len(react_steps)} steps)"):
                for i, step in enumerate(react_steps):
                    st.markdown(f"**Step {i+1}:** `{step.get('tool', '?')}` — "
                                f"{str(step.get('observation', ''))[:200]}")

        # Pipeline graph (orchestration DAG visualisation)
        with st.expander("Pipeline graph (orchestration DAG)", expanded=False):
            try:
                from orchestration.graph import get_graph as _get_orch_graph
                _compiled = _get_orch_graph()
                mermaid_str = _compiled.get_graph().draw_mermaid()
                # Try using st.mermaid if available (Streamlit 1.28+)
                try:
                    _mermaid_fn = getattr(st, "mermaid", None)
                    if callable(_mermaid_fn):
                        _mermaid_fn(mermaid_str)
                    else:
                        raise AttributeError("st.mermaid unavailable")
                except (AttributeError, Exception):
                    # Fallback: show as code with link to render
                    st.code(mermaid_str, language="mermaid")
                    st.markdown("[📊 Render this Mermaid diagram online](https://mermaid.live)")
            except Exception as _graph_err:
                st.caption(f"Pipeline graph unavailable: {_graph_err}")


if __name__ == "__main__":
    main()
