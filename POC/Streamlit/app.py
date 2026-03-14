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
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Ensure repo root is on sys.path so orchestration imports work ─────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
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

# Node display labels / descriptions
NODE_LABELS = {
    "planner":         ("Planning", "Resolving ticker, selecting agents, setting complexity…"),
    "parallel_agents": ("Running Agents", "Business Analyst · Quant Fundamental · Financial Modelling · Web Search"),
    "react_check":     ("ReAct Check", "Evaluating agent outputs for gaps or errors…"),
    "summarizer":      ("Summarizing", "Generating final research note with DeepSeek-R1…"),
    "memory_update":   ("Memory Update", "Persisting failure patterns to episodic memory…"),
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .agent-card { border: 1px solid #2d2d2d; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
    .status-ok   { color: #22c55e; font-weight: 600; }
    .status-warn { color: #f59e0b; font-weight: 600; }
    .status-err  { color: #ef4444; font-weight: 600; }
    .metric-label { font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.1rem; font-weight: 600; }
    
    /* Fix for text rendering issues */
    .stMarkdown {
        font-feature-settings: "liga" 1, "kern" 1;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Better font for code/technical content */
    code {
        font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', Monaco, monospace;
        font-size: 0.85em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_text_for_display(text: str) -> str:
    """Clean text to fix rendering issues with invisible Unicode characters."""
    if not text:
        return text
    import re

    # Remove zero-width characters and other invisible / control Unicode
    text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)

    # Normalise line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    return text


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
                qt_sorted = sorted(qt, key=lambda r: str(r.get("period", "")))
                rev_rows = []
                for row in qt_sorted[-4:]:
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
                from collections import defaultdict
                annual_rev_map: dict = defaultdict(float)
                for row in qt_sorted:
                    year = str(row.get("period", ""))[:4]
                    revenue = row.get("revenue")
                    if year.isdigit() and revenue is not None:
                        try:
                            annual_rev_map[year] += float(revenue)
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
                qt_sorted = sorted(qt, key=lambda r: str(r.get("period", "")))
                eps_rows = []
                for row in qt_sorted[-4:]:
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
                annual_eps_map: dict = defaultdict(float)
                annual_eps_count: dict = defaultdict(int)
                for row in qt_sorted:
                    year = str(row.get("period", ""))[:4]
                    eps = row.get("eps_diluted") or row.get("operating_earnings_per_share")
                    if year.isdigit() and eps is not None:
                        try:
                            annual_eps_map[year] += float(eps)
                            annual_eps_count[year] += 1
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


def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """Safely navigate a nested dict."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


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
                st.markdown(narrative)

        # Missing context
        missing = output.get("missing_context") or []
        if missing:
            st.caption("Missing context: " + "; ".join(str(m) for m in missing[:3]))

        # Thinking trace
        _render_thinking_trace(output.get("thinking_trace") or [])


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
                st.markdown(narrative)

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
                    st.markdown(f"**Consensus:** {consensus_narrative}")
                for p in personas:
                    name_p = p.get("name") or p.get("persona") or "Agent"
                    narr_p = p.get("narrative") or p.get("reasoning") or ""
                    if narr_p:
                        st.markdown(f"**{name_p}:** {narr_p}")

        # Narrative
        narrative = output.get("quantitative_summary") or output.get("summary") or ""
        if narrative:
            with st.expander("Full narrative"):
                st.markdown(narrative)

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
                        line = f"**Pass {pass_count} — {label}:** {desc}"

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
        
        # DeepSeek API Key
        deepseek_api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=st.session_state.get("deepseek_api_key", ""),
            help="Enter your DeepSeek API key. Get one at https://platform.deepseek.com",
            key="deepseek_key_input"
        )
        
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
        if not st.session_state.get("api_verified"):
            st.warning("⚠️ API key required for analysis")
        
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
                    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ingestion" / "etl"))
                    from inspect_db import check_postgres, check_neo4j
                    
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
        # Check API key is verified
        if not st.session_state.get("api_verified"):
            st.error("Please enter and verify your DeepSeek API key in the sidebar first!")
            st.info("1. Enter your DeepSeek API key in the sidebar\n2. Click 'Test API Connection'\n3. Wait for '✓ API key is valid!' message")
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
        # Convert to the format expected by orchestration (None for English)
        output_lang_param = output_lang if output_lang != "English" else None
        final_state = _run_with_streaming(query.strip(), output_lang_param)

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
            final_summary_clean = _clean_text_for_display(final_summary)
            
            # Check if text has proper formatting
            has_proper_spaces = " " in final_summary_clean[:100] and "\n" in final_summary_clean[:500]
            
            # Show toggle for raw vs formatted view
            col1, col2 = st.columns([1, 5])
            with col1:
                show_raw = st.checkbox("Raw text", value=not has_proper_spaces, key="show_raw_text")
            with col2:
                pass
            
            if show_raw:
                st.text(final_summary_clean)
            else:
                st.markdown(final_summary_clean)
            
            st.download_button(
                label="Download Research Note (.md)",
                data=final_summary,
                file_name=f"research_{state.get('ticker', 'query')}.md",
                mime="text/markdown",
            )

        # RLAIF Quality Score Display
        rl_scores = state.get("rl_feedback_scores")
        run_id = state.get("rl_feedback_run_id")
        if rl_scores and run_id:
            with st.expander("AI Quality Score (RLAIF)", expanded=False):
                overall = rl_scores.get("overall_score", 0)
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Overall", f"{overall:.1f}/10")
                col2.metric("Factual", f"{rl_scores.get('factual_accuracy', 0):.1f}")
                col3.metric("Citations", f"{rl_scores.get('citation_completeness', 0):.1f}")
                col4.metric("Analysis", f"{rl_scores.get('analysis_depth', 0):.1f}")
                col5.metric("Structure", f"{rl_scores.get('structure_compliance', 0):.1f}")
                
                if overall < 7.0:
                    st.warning(f"⚠️ This report scored below the quality threshold (7.0). Agent blamed: {rl_scores.get('agent_blamed', 'unknown')}")
                
                weaknesses = rl_scores.get("weaknesses", [])
                if weaknesses:
                    st.markdown("**Areas for improvement:**")
                    for w in weaknesses[:3]:
                        st.markdown(f"- {w}")

        # Explicit User Feedback Widget
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
                    st.mermaid(mermaid_str)
                except (AttributeError, Exception):
                    # Fallback: show as code with link to render
                    st.code(mermaid_str, language="mermaid")
                    st.markdown("[📊 Render this Mermaid diagram online](https://mermaid.live)")
            except Exception as _graph_err:
                st.caption(f"Pipeline graph unavailable: {_graph_err}")


if __name__ == "__main__":
    main()
