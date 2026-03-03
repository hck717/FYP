"""Streamlit UI for The Agentic Investment Analyst.

Run from the repo root:
    streamlit run POC/streamlit/streamlit.py

Architecture:
    This app wraps the orchestration layer (orchestration/graph.py) and streams
    live node-by-node progress to the UI using orchestration.graph.stream().
    Per-agent result cards are rendered once the full pipeline completes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Ensure repo root is on sys.path so orchestration imports work ─────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

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
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    st.sidebar.header("Backend Status")

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
        "Qdrant":      report.get("qdrant", {}),
        "Neo4j":       report.get("neo4j", {}),
        "Ollama":      report.get("ollama", {}),
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
                    "P/E":        vf.get("pe_ratio"),
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
            c2.metric("Sharpe (12m)",  _fmt(mr.get("sharpe_12m")))
            c3.metric("Return (12m)",  _pct(mr.get("return_12m")))

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
            wacc  = dcf.get("wacc")
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
            c4.metric("WACC",   _pct(wacc))
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
                    st.dataframe(df, use_container_width=True)
                except Exception:
                    pass

        # Comps
        if comps:
            st.markdown("**Comparable Analysis**")
            peer_ev_ebitda = comps.get("peer_ev_ebitda_median")
            peer_pe        = comps.get("peer_pe_median")
            implied_ev     = comps.get("implied_ev_ebitda_value")

            c1, c2, c3 = st.columns(3)
            c1.metric("Peer EV/EBITDA (median)", _fmt(peer_ev_ebitda))
            c2.metric("Peer P/E (median)",        _fmt(peer_pe))
            c3.metric("Implied Price (EV/EBITDA)", f"${_fmt(implied_ev)}")

        # Technicals
        if technicals:
            st.markdown("**Technicals**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RSI (14)",  _fmt(technicals.get("rsi_14")))
            c2.metric("MACD",      _fmt(technicals.get("macd")))
            c3.metric("ATR",       _fmt(technicals.get("atr")))
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

        # Narrative
        narrative = output.get("quantitative_summary") or output.get("summary") or ""
        if narrative:
            with st.expander("Full narrative"):
                st.markdown(narrative)


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


def _render_agent_outputs(state: Dict[str, Any]) -> None:
    """Render all agent result cards from the final state."""
    tickers: List[str] = state.get("tickers") or []
    if not tickers and state.get("ticker"):
        tickers = [state["ticker"]]

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


# ── Streaming runner ──────────────────────────────────────────────────────────

def _run_with_streaming(query: str) -> Optional[Dict[str, Any]]:
    """Stream orchestration events into the UI; return final state on completion."""
    from orchestration.graph import stream as orch_stream

    progress_area = st.empty()
    status_lines: List[str] = []
    final_state: Dict[str, Any] = {}
    pass_count = 0

    try:
        with st.spinner("Running pipeline…"):
            for node_name, node_output in orch_stream(query):
                label, desc = NODE_LABELS.get(node_name, (node_name, ""))

                # Track parallel_agents passes for ReAct display
                if node_name == "parallel_agents":
                    pass_count += 1
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
                elif node_name == "planner":
                    ticker   = node_output.get("ticker") or "?"
                    tickers  = node_output.get("tickers") or []
                    plan     = node_output.get("plan") or {}
                    complexity = plan.get("complexity") or "?"
                    line = (
                        f"**{label}:** ticker={', '.join(tickers) or ticker}, "
                        f"complexity={complexity}"
                    )
                else:
                    line = f"**{label}:** {desc}"

                status_lines.append(line)
                progress_area.markdown("\n\n".join(status_lines))

                # Accumulate state — merge each node_output into final_state
                if isinstance(node_output, dict):
                    final_state.update(node_output)

    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        return None

    progress_area.empty()
    return final_state


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("Agentic Investment Analyst")
        st.caption("Powered by LangGraph · Ollama · Qdrant · Neo4j · PostgreSQL")
        st.divider()
        _render_sidebar_availability()
        st.divider()
        st.markdown("**Supported tickers**")
        st.markdown("AAPL · MSFT · GOOGL · TSLA · NVDA")

    # ── Main area ─────────────────────────────────────────────────────────────
    st.title("Agentic Investment Analyst")
    st.markdown(
        "Ask any investment question about AAPL, MSFT, GOOGL, TSLA, or NVDA. "
        "The orchestration pipeline will dispatch the appropriate agents in parallel "
        "and synthesise a research note."
    )

    # Example query buttons
    st.markdown("**Example queries:**")
    cols = st.columns(len(EXAMPLE_QUERIES))
    for col, q in zip(cols, EXAMPLE_QUERIES):
        if col.button(q[:40] + "…", key=f"ex_{q[:20]}", use_container_width=True):
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
            if ticker_override not in query.upper():
                query = f"[{ticker_override}] {query}"

    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    # ── Run ───────────────────────────────────────────────────────────────────
    if run_btn:
        if not query.strip():
            st.warning("Please enter a question.")
            return

        # Clear previous results
        for key in ("last_state", "last_query"):
            st.session_state.pop(key, None)

        st.markdown("---")
        st.subheader("Live Progress")
        final_state = _run_with_streaming(query.strip())

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

        # Final research note
        final_summary = state.get("final_summary") or ""
        if final_summary:
            st.subheader("Research Note")
            st.markdown(final_summary)
            st.download_button(
                label="Download Research Note (.md)",
                data=final_summary,
                file_name=f"research_{state.get('ticker', 'query')}.md",
                mime="text/markdown",
            )

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


if __name__ == "__main__":
    main()
