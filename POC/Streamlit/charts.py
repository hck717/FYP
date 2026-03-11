"""Plotly chart library for the Agentic Investment Analyst Streamlit UI.

Each function accepts raw agent output dicts and returns a ``plotly.graph_objects.Figure``.
All functions are pure / deterministic — no LLM calls, no I/O.

Design philosophy: buy-side equity research aesthetics — Bloomberg terminal dark theme,
Goldman/Morgan Stanley research deck style. Charts mirror what you would find in a
real equity research PDF: price performance, P&L trends, margin evolution, peer comps,
DCF football field, sensitivity heatmap, MoE consensus dumbbell, and factor scorecard.

Usage (from streamlit.py):
    from charts import (
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
        chart_fcf_trend,
        chart_price_history,
        chart_price_performance,
        chart_ebitda_trend,
        chart_eps_trend,
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_float(val: Any) -> Optional[float]:
    """Safely convert *val* to float; return None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _bn(val: Any) -> Optional[float]:
    """Convert to billions; returns None on failure."""
    f = _to_float(val)
    return f / 1e9 if f is not None else None


def _pv(val: Any) -> Optional[float]:
    """Convert to plain float (for percentages already in fractional or % form)."""
    return _to_float(val)


def _rolling_sma(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    """Pure-Python rolling simple moving average; returns None for insufficient data."""
    result: List[Optional[float]] = []
    for i, _ in enumerate(values):
        if i < window - 1:
            result.append(None)
        else:
            chunk = [x for x in values[i - window + 1 : i + 1] if x is not None]
            result.append(sum(chunk) / len(chunk) if chunk else None)
    return result


# ── Colour palette ────────────────────────────────────────────────────────────
_BEAR_COLOUR    = "#ef4444"   # red-500
_BASE_COLOUR    = "#3b82f6"   # blue-500
_BULL_COLOUR    = "#22c55e"   # green-500
_NEUTRAL_COLOUR = "#f59e0b"   # amber-500
_GREY           = "#6b7280"   # grey-500
_DARK_BG        = "#0f172a"   # slate-900
_CARD_BG        = "#1e293b"   # slate-800
_TEXT           = "#f1f5f9"   # slate-100
_GRID           = "#334155"   # slate-700
_VIOLET         = "#a78bfa"   # violet-400
_TEAL           = "#2dd4bf"   # teal-400
_INDIGO         = "#818cf8"   # indigo-400

_LAYOUT_BASE: Dict[str, Any] = dict(
    paper_bgcolor=_DARK_BG,
    plot_bgcolor=_CARD_BG,
    font=dict(color=_TEXT, family="Inter, sans-serif", size=12),
    margin=dict(l=60, r=40, t=50, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11)),
)


def _apply_base_layout(fig: go.Figure, title: str = "", height: Optional[int] = None) -> go.Figure:
    """Apply the shared dark-theme layout to a figure."""
    updates: Dict[str, Any] = {**_LAYOUT_BASE}
    if title:
        updates["title"] = dict(text=title, x=0.02, font=dict(size=14, color=_TEXT))
    if height:
        updates["height"] = height
    fig.update_layout(**updates)
    fig.update_xaxes(gridcolor=_GRID, zerolinecolor=_GRID)
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor=_GRID)
    return fig


def _stub_fig(title: str, message: str = "Data not available for this view") -> go.Figure:
    """Return a clean 'data unavailable' placeholder figure."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"<b>{message}</b>",
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14, color=_GREY),
        bgcolor="rgba(0,0,0,0)",
    )
    _apply_base_layout(fig, title, height=220)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# ── 1. Revenue / Sales Trend ──────────────────────────────────────────────────

def chart_revenue_trend(
    quarterly_trends: List[Dict[str, Any]],
    dcf: Dict[str, Any],
    ticker: str,
) -> go.Figure:
    """Revenue bar chart (4Q actuals) with gross profit overlay.

    If DCF scenario_table contains projected annual revenue, a forward estimate
    line is appended after the actuals. Gross margin % shown on secondary axis.
    """
    if not quarterly_trends:
        return _stub_fig(f"{ticker} — Revenue Trend", "No quarterly revenue data available")

    rows = sorted(quarterly_trends, key=lambda r: str(r.get("period", "")))
    periods  = [str(r.get("period", "?")) for r in rows]
    rev      = [_bn(r.get("revenue"))      for r in rows]
    gp       = [_bn(r.get("gross_profit")) for r in rows]
    gm       = [_pv(r.get("gross_margin")) for r in rows]

    # Forward estimate from DCF scenario_table (base case annual)
    fwd_labels: List[str] = []
    fwd_rev:    List[Optional[float]] = []
    for row in (dcf.get("scenario_table") or []):
        if str(row.get("scenario", "")).lower() == "base":
            for yr in (row.get("projections") or [])[:3]:
                label = str(yr.get("year", ""))
                rv    = _bn(yr.get("revenue"))
                if label and rv is not None:
                    fwd_labels.append(f"{label}E")
                    fwd_rev.append(rv)
            break

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Actual revenue bars
    if any(v is not None for v in rev):
        fig.add_trace(go.Bar(
            x=periods, y=rev,
            name="Revenue (Actual)",
            marker_color=_BASE_COLOUR,
            hovertemplate="%{x}<br>Revenue: $%{y:.2f}B<extra></extra>",
        ), secondary_y=False)

    # Actual gross profit bars
    if any(v is not None for v in gp):
        fig.add_trace(go.Bar(
            x=periods, y=gp,
            name="Gross Profit",
            marker_color=_VIOLET,
            opacity=0.75,
            hovertemplate="%{x}<br>Gross Profit: $%{y:.2f}B<extra></extra>",
        ), secondary_y=False)

    # Forward revenue line
    if fwd_labels and fwd_rev:
        fig.add_trace(go.Scatter(
            x=fwd_labels, y=fwd_rev,
            name="Revenue (Base Est.)",
            mode="lines+markers",
            line=dict(color=_TEAL, width=2, dash="dot"),
            marker=dict(size=7, symbol="diamond"),
            hovertemplate="%{x}<br>Est. Revenue: $%{y:.2f}B<extra></extra>",
        ), secondary_y=False)

    # Gross margin % line
    if any(v is not None for v in gm):
        # Normalise: if values look like fractions (0.xx) convert to %
        gm_display = [v * 100 if v is not None and abs(v) <= 1.5 else v for v in gm]
        fig.add_trace(go.Scatter(
            x=periods, y=gm_display,
            name="Gross Margin %",
            mode="lines+markers",
            line=dict(color=_NEUTRAL_COLOUR, width=2),
            marker=dict(size=6),
            hovertemplate="%{x}<br>Gross Margin: %{y:.1f}%<extra></extra>",
        ), secondary_y=True)

    fig.update_layout(
        barmode="group",
        title=dict(text=f"{ticker} — Revenue & Gross Profit Trend", x=0.02, font=dict(size=14, color=_TEXT)),
        **_LAYOUT_BASE,
    )
    fig.update_yaxes(title_text="USD (Billions)", secondary_y=False, gridcolor=_GRID)
    fig.update_yaxes(title_text="Gross Margin (%)", secondary_y=True, showgrid=False)
    fig.update_xaxes(gridcolor=_GRID, zerolinecolor=_GRID)
    return fig


# ── 2. EBIT / Operating Income Trend ─────────────────────────────────────────

def chart_ebitda_trend(
    quarterly_trends: List[Dict[str, Any]],
    ticker: str,
) -> go.Figure:
    """EBIT / Operating Income bars + EBIT margin % line on secondary axis.

    Uses ``operating_income`` and ``ebit_margin`` from quarterly_trends.
    Labelled as "EBIT / Operating Income" (not true EBITDA — D&A not disaggregated).
    """
    if not quarterly_trends:
        return _stub_fig(f"{ticker} — EBIT / Operating Income Trend", "No quarterly data available")

    rows = sorted(quarterly_trends, key=lambda r: str(r.get("period", "")))
    periods = [str(r.get("period", "?")) for r in rows]
    ebit    = [_bn(r.get("operating_income")) for r in rows]

    def _norm_pct(v: Any) -> Optional[float]:
        f = _pv(v)
        if f is None:
            return None
        return f * 100 if abs(f) <= 1.5 else f

    ebit_m = [_norm_pct(r.get("ebit_margin")) for r in rows]

    if not any(v is not None for v in ebit):
        return _stub_fig(f"{ticker} — EBIT / Operating Income Trend", "Operating income not available")

    bar_colours = [_BULL_COLOUR if (v or 0) >= 0 else _BEAR_COLOUR for v in ebit]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=periods, y=ebit,
        name="EBIT / Op. Income",
        marker_color=bar_colours,
        text=[f"${v:.2f}B" if v is not None else "—" for v in ebit],
        textposition="outside",
        textfont=dict(size=10, color=_TEXT),
        cliponaxis=False,
        hovertemplate="%{x}<br>EBIT: $%{y:.2f}B<extra></extra>",
    ), secondary_y=False)

    if any(v is not None for v in ebit_m):
        fig.add_trace(go.Scatter(
            x=periods, y=ebit_m,
            name="EBIT Margin %",
            mode="lines+markers",
            line=dict(color=_NEUTRAL_COLOUR, width=2),
            marker=dict(size=7, symbol="circle"),
            hovertemplate="%{x}<br>EBIT Margin: %{y:.1f}%<extra></extra>",
        ), secondary_y=True)

    fig.update_layout(
        title=dict(
            text=f"{ticker} — EBIT / Operating Income Trend  (Note: excl. D&A)",
            x=0.02, font=dict(size=14, color=_TEXT),
        ),
        **_LAYOUT_BASE,
        height=300,
    )
    fig.update_yaxes(title_text="USD (Billions)", secondary_y=False, gridcolor=_GRID)
    fig.update_yaxes(title_text="EBIT Margin (%)", secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="Quarter", gridcolor=_GRID, zerolinecolor=_GRID)
    return fig


# Stub alias retained for backward-compat (streamlit.py may still import it)
def chart_ebitda_trend_stub(ticker: str) -> go.Figure:
    """Backward-compat alias → chart_ebitda_trend with empty data."""
    return chart_ebitda_trend([], ticker)


# ── 3. EPS Trend with beat/miss markers ──────────────────────────────────────

def chart_eps_trend(
    quarterly_trends: List[Dict[str, Any]],
    ticker: str,
    earnings: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """Diluted EPS bar chart from quarterly_trends.

    Uses ``eps_diluted`` per quarter if available; falls back to net income (billions).
    Annotates the most recent quarter with a beat/miss badge from the
    ``earnings`` record (``surprise_pct``, ``beat_streak``).
    """
    if not quarterly_trends:
        return _stub_fig(f"{ticker} — EPS Trend", "No quarterly earnings data available")

    rows = sorted(quarterly_trends, key=lambda r: str(r.get("period", "")))
    periods = [str(r.get("period", "?")) for r in rows]

    # Prefer eps_diluted; fall back to net income in $B
    eps_vals = [_to_float(r.get("eps_diluted")) for r in rows]
    use_eps = any(v is not None for v in eps_vals)

    if use_eps:
        y_vals   = eps_vals
        y_label  = "Diluted EPS (USD)"
        fmt_fn   = lambda v: f"${v:.2f}" if v is not None else "—"
        hover_t  = "%{x}<br>EPS: $%{y:.2f}<extra></extra>"
        title_sfx = "Diluted EPS"
    else:
        y_vals  = [_bn(r.get("net_income")) for r in rows]
        y_label = "Net Income (USD Billions)"
        fmt_fn  = lambda v: f"${v:.2f}B" if v is not None else "—"
        hover_t = "%{x}<br>Net Income: $%{y:.2f}B<extra></extra>"
        title_sfx = "Net Income"

    if not any(v is not None for v in y_vals):
        return _stub_fig(f"{ticker} — EPS Trend", "EPS / net income data not available")

    colours = [_BULL_COLOUR if (v or 0) >= 0 else _BEAR_COLOUR for v in y_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=periods, y=y_vals,
        name=title_sfx,
        marker_color=colours,
        text=[fmt_fn(v) for v in y_vals],
        textposition="outside",
        textfont=dict(size=10, color=_TEXT),
        cliponaxis=False,
        hovertemplate=hover_t,
    ))

    # Beat / miss annotation on most recent quarter
    if earnings:
        surp = _to_float((earnings or {}).get("surprise_pct"))
        beat_streak = (earnings or {}).get("beat_streak") or 0
        miss_streak = (earnings or {}).get("miss_streak") or 0
        if surp is not None and periods:
            beat = surp >= 0
            sign = "+" if surp >= 0 else ""
            streak_txt = ""
            if beat and beat_streak:
                streak_txt = f"  ({beat_streak}Q beat streak)"
            elif not beat and miss_streak:
                streak_txt = f"  ({miss_streak}Q miss streak)"
            annot_col  = _BULL_COLOUR if beat else _BEAR_COLOUR
            badge_icon = "BEAT" if beat else "MISS"
            fig.add_annotation(
                x=periods[-1],
                y=y_vals[-1] if y_vals[-1] is not None else 0,
                text=f"<b>{badge_icon} {sign}{surp:.1f}%{streak_txt}</b>",
                showarrow=True,
                arrowhead=2,
                arrowcolor=annot_col,
                font=dict(color=annot_col, size=10),
                bgcolor=_CARD_BG,
                bordercolor=annot_col,
                borderwidth=1,
                ay=-40,
            )

    _apply_base_layout(fig, f"{ticker} — {title_sfx} Trend", height=300)
    fig.update_xaxes(title_text="Quarter")
    fig.update_yaxes(title_text=y_label, gridcolor=_GRID)
    fig.update_layout(margin=dict(l=70, r=60, t=55, b=50))
    return fig


# ── 4. Key Margin Trends ──────────────────────────────────────────────────────

def chart_margin_trends(
    quarterly_trends: List[Dict[str, Any]],
    key_metrics: Dict[str, Any],
    ticker: str,
) -> go.Figure:
    """Multi-line margin chart: Gross, EBIT, Net margins over time.

    Gross and EBIT margins come from quarterly_trends.
    Net margin is derived (net_income / revenue) per quarter.
    A full-year LTM reference line is drawn from key_metrics if available.
    """
    if not quarterly_trends:
        return _stub_fig(f"{ticker} — Margin Trends", "No margin data available")

    rows = sorted(quarterly_trends, key=lambda r: str(r.get("period", "")))
    periods = [str(r.get("period", "?")) for r in rows]

    def _norm_pct(v: Any) -> Optional[float]:
        f = _pv(v)
        if f is None:
            return None
        return f * 100 if abs(f) <= 1.5 else f

    gross_m = [_norm_pct(r.get("gross_margin")) for r in rows]
    ebit_m  = [_norm_pct(r.get("ebit_margin"))  for r in rows]

    # Net margin = net_income / revenue
    net_m: List[Optional[float]] = []
    for r in rows:
        rev = _to_float(r.get("revenue"))
        ni  = _to_float(r.get("net_income"))
        if rev and rev != 0 and ni is not None:
            net_m.append((ni / rev) * 100)
        else:
            net_m.append(None)

    # Check we have at least one series
    if not any(any(v is not None for v in s) for s in [gross_m, ebit_m, net_m]):
        return _stub_fig(f"{ticker} — Margin Trends", "No margin data available")

    fig = go.Figure()

    margin_series = [
        ("Gross Margin",  gross_m, _BULL_COLOUR,    "solid"),
        ("EBIT Margin",   ebit_m,  _BASE_COLOUR,    "dot"),
        ("Net Margin",    net_m,   _NEUTRAL_COLOUR, "dash"),
    ]
    for name, data, col, dash in margin_series:
        if any(v is not None for v in data):
            fig.add_trace(go.Scatter(
                x=periods, y=data,
                name=name,
                mode="lines+markers",
                line=dict(color=col, width=2, dash=dash),
                marker=dict(size=7),
                hovertemplate=f"%{{x}}<br>{name}: %{{y:.1f}}%<extra></extra>",
            ))

    # LTM reference lines from key_metrics
    for km_key, label, col in [
        ("gross_margin", "LTM Gross", _BULL_COLOUR),
        ("ebit_margin",  "LTM EBIT",  _BASE_COLOUR),
    ]:
        v = _to_float(key_metrics.get(km_key))
        if v is not None:
            v_display = v * 100 if abs(v) <= 1.5 else v
            fig.add_hline(
                y=v_display, line_dash="longdash", line_color=col, line_width=1, opacity=0.5,
                annotation_text=f"  {label} {v_display:.1f}%",
                annotation_font=dict(color=col, size=9),
                annotation_position="right",
            )

    _apply_base_layout(fig, f"{ticker} — Key Margin Trends  (Gross | EBIT | Net)", height=300)
    fig.update_xaxes(title_text="Quarter")
    fig.update_yaxes(title_text="Margin (%)", gridcolor=_GRID)
    return fig


# ── 5. Peer Group Comparison Bar Charts ──────────────────────────────────────

def chart_peer_comps(
    comps: Dict[str, Any],
    ticker: str,
) -> go.Figure:
    """Horizontal grouped bar: ticker vs peer median for EV/EBITDA, P/E, P/FCF.

    Premium/discount annotations are added to the right of each bar group.
    Bars are colour-coded: red = expensive vs peers, green = cheap, blue = in-line.
    """
    peer_group = comps.get("peer_group") or []

    # Safety check: if peer_group is not a list or is empty strings, skip peer processing
    if not isinstance(peer_group, list) or not peer_group or any(isinstance(p, str) for p in peer_group):
        peer_group = []

    metrics = [
        ("ev_ebitda",   "EV/EBITDA"),
        ("pe_trailing", "P/E Trailing"),
        ("pe_forward",  "P/E Forward"),
        ("p_fcf",       "P/FCF"),
    ]

    ticker_vals: Dict[str, Optional[float]] = {k: _to_float(comps.get(k)) for k, _ in metrics}

    # Peer medians — use precomputed key if present, else compute from peer_group
    peer_medians: Dict[str, Optional[float]] = {}
    med_key_map  = {"ev_ebitda": "peer_ev_ebitda_median", "pe_trailing": "peer_pe_median"}
    peer_aliases = {
        "ev_ebitda":   ["ev_ebitda"],
        "pe_trailing": ["pe_trailing", "pe"],
        "pe_forward":  ["pe_forward"],
        "p_fcf":       ["p_fcf"],
    }
    for key, _ in metrics:
        mk = med_key_map.get(key)
        if mk and comps.get(mk) is not None:
            peer_medians[key] = _to_float(comps[mk])
            continue
        vals = []
        for peer in peer_group:
            for alias in peer_aliases.get(key, [key]):
                v = _to_float(peer.get(alias))
                if v is not None:
                    vals.append(v)
                    break
        if vals:
            s = sorted(vals)
            n = len(s)
            peer_medians[key] = (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]
        else:
            peer_medians[key] = None

    display = [(k, lbl) for k, lbl in metrics
               if ticker_vals.get(k) is not None or peer_medians.get(k) is not None]
    if not display:
        return _stub_fig(f"{ticker} — Peer Comparables", "No comparable data available")

    metric_labels = [lbl for _, lbl in display]
    t_vals = [ticker_vals.get(k) or 0  for k, _ in display]
    p_vals = [peer_medians.get(k) or 0 for k, _ in display]

    bar_colours = []
    for tv, pv in zip(t_vals, p_vals):
        if pv == 0:
            bar_colours.append(_BASE_COLOUR)
        elif tv > pv * 1.05:
            bar_colours.append(_BEAR_COLOUR)
        elif tv < pv * 0.95:
            bar_colours.append(_BULL_COLOUR)
        else:
            bar_colours.append(_BASE_COLOUR)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=metric_labels, x=p_vals,
        name="Peer Median",
        orientation="h",
        marker_color=_GREY,
        opacity=0.65,
        hovertemplate="%{y} Peer Median: %{x:.1f}x<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=metric_labels, x=t_vals,
        name=ticker,
        orientation="h",
        marker_color=bar_colours,
        text=[f"{v:.1f}x" for v in t_vals],
        textposition="outside",
        textfont=dict(size=11),
        cliponaxis=False,
        hovertemplate=f"{ticker} " + "%{y}: %{x:.1f}x<extra></extra>",
    ))

    annotations = []
    for tv, pv, lbl in zip(t_vals, p_vals, metric_labels):
        if pv > 0 and tv > 0:
            prem = (tv / pv - 1) * 100
            col  = _BEAR_COLOUR if prem > 5 else (_BULL_COLOUR if prem < -5 else _GREY)
            sign = "+" if prem >= 0 else ""
            annotations.append(dict(
                x=max(tv, pv) * 1.05, y=lbl,
                text=f"<b>{sign}{prem:.0f}%</b>",
                showarrow=False,
                font=dict(color=col, size=11),
                xanchor="left",
            ))

    # Add peer names as subtitle annotation
    peer_names = [p.get("ticker") or p.get("name") or "" for p in peer_group[:6]]
    peer_str = "  Peers: " + " · ".join(p for p in peer_names if p) if peer_names else ""

    _apply_base_layout(fig, f"{ticker} — Peer Comparable Analysis{peer_str}", height=max(260, 70 * len(display)))
    fig.update_layout(
        barmode="group",
        annotations=annotations,
        margin=dict(l=100, r=100, t=55, b=50),
    )
    fig.update_xaxes(title_text="Multiple (x)")
    return fig


# ── 6. Valuation Football Field ───────────────────────────────────────────────

def chart_football_field(
    dcf: Dict[str, Any],
    comps: Dict[str, Any],
    current_price: Optional[float],
    ticker: str,
) -> go.Figure:
    """Classic equity research football field / valuation range chart.

    Each methodology is shown as a horizontal range bar (low → high) with a
    point estimate diamond. Current price shown as vertical dotted line.

    Methodologies:
      - DCF: Bear → Bull range, Base as point estimate
      - EV/EBITDA Comps: implied value range around median
      - P/E Comps: implied value range (if available)
    """
    ranges: List[Tuple[str, float, float, float, str]] = []
    # (label, low, high, point, colour)

    # DCF range
    bear = _to_float(dcf.get("intrinsic_value_bear"))
    base = _to_float(dcf.get("intrinsic_value_base"))
    bull = _to_float(dcf.get("intrinsic_value_bull"))
    if bear is not None and bull is not None and base is not None:
        ranges.append(("DCF Intrinsic Value", bear, bull, base, _BASE_COLOUR))

    # EV/EBITDA comps implied
    implied_ev = _to_float(comps.get("implied_ev_ebitda_value"))
    peer_ev_med = _to_float(comps.get("peer_ev_ebitda_median"))
    ticker_ev   = _to_float(comps.get("ev_ebitda"))
    if implied_ev is not None:
        # Build a ±15% range around implied
        lo = implied_ev * 0.85
        hi = implied_ev * 1.15
        ranges.append(("EV/EBITDA Comps", lo, hi, implied_ev, _TEAL))

    # P/E comps implied — rough calculation if we have pe_forward
    pe_fwd = _to_float(comps.get("pe_forward"))
    peer_pe = _to_float(comps.get("peer_pe_median"))
    if pe_fwd is not None and peer_pe is not None and current_price is not None:
        # Implied price at peer median P/E
        eps_implied = current_price / pe_fwd if pe_fwd > 0 else None
        if eps_implied:
            implied_pe_price = eps_implied * peer_pe
            ranges.append(("P/E Comps", implied_pe_price * 0.85, implied_pe_price * 1.15,
                            implied_pe_price, _VIOLET))

    if not ranges:
        return _stub_fig(f"{ticker} — Valuation Football Field", "Insufficient valuation data")

    fig = go.Figure()
    colours = [r[4] for r in ranges]

    for i, (label, lo, hi, pt, col) in enumerate(ranges):
        # Range bar (base = lo, width = hi - lo)
        fig.add_trace(go.Bar(
            y=[label],
            x=[hi - lo],
            base=[lo],
            orientation="h",
            marker_color=col,
            opacity=0.35,
            hovertemplate=f"{label}: ${lo:,.0f} – ${hi:,.0f}<extra></extra>",
            showlegend=False,
            width=0.5,
        ))
        # Point estimate diamond
        fig.add_trace(go.Scatter(
            y=[label], x=[pt],
            mode="markers+text",
            marker=dict(color=col, size=14, symbol="diamond",
                        line=dict(color="white", width=1)),
            text=[f"<b>${pt:,.0f}</b>"],
            textposition="top center",
            textfont=dict(size=10, color=col),
            hovertemplate=f"{label} estimate: ${pt:,.0f}<extra></extra>",
            showlegend=True,
            name=label,
        ))

    # Current price line
    if current_price is not None:
        fig.add_vline(
            x=current_price,
            line_dash="dot",
            line_color=_NEUTRAL_COLOUR,
            line_width=2,
            annotation_text=f"  ${current_price:,.0f}  Current",
            annotation_position="top",
            annotation_font=dict(color=_NEUTRAL_COLOUR, size=11),
        )

    all_lows  = [r[1] for r in ranges]
    all_highs = [r[2] for r in ranges]
    x_min = min(all_lows) * 0.85
    x_max = max(all_highs) * 1.2
    if current_price:
        x_min = min(x_min, current_price * 0.75)
        x_max = max(x_max, current_price * 1.15)

    _apply_base_layout(fig, f"{ticker} — Valuation Football Field", height=max(260, 100 * len(ranges) + 60))
    fig.update_xaxes(title_text="Implied Value (USD)", range=[x_min, x_max],
                     tickprefix="$", tickformat=",.0f")
    fig.update_layout(margin=dict(l=160, r=60, t=55, b=50))
    return fig


# ── 7. DCF Waterfall / Scenario Bridge ───────────────────────────────────────

def chart_dcf_waterfall(
    dcf: Dict[str, Any],
    current_price: Optional[float],
    ticker: str,
) -> go.Figure:
    """Waterfall chart: current price → Bear / Base / Bull intrinsic values.

    Shows the gap between current market price and each scenario as a step,
    making the upside/downside magnitude visually immediate. Falls back to a
    horizontal bar chart if waterfall is unsuitable (e.g. negative prices).
    """
    bear = _to_float(dcf.get("intrinsic_value_bear"))
    base = _to_float(dcf.get("intrinsic_value_base"))
    bull = _to_float(dcf.get("intrinsic_value_bull"))
    wacc = _to_float(dcf.get("wacc_used") or dcf.get("wacc"))

    if base is None:
        return _stub_fig(f"{ticker} — DCF Scenarios", "No DCF intrinsic value data available")

    cp = current_price or 0

    # Build measure/x/y for waterfall
    scenarios = [("Bear", bear, _BEAR_COLOUR), ("Base", base, _BASE_COLOUR), ("Bull", bull, _BULL_COLOUR)]
    valid = [(s, v, c) for s, v, c in scenarios if v is not None]

    # Build a clean horizontal bar chart (simpler and more legible than waterfall for this data)
    labels  = [s for s, _, _ in valid]
    values  = [v for _, v, _ in valid]
    colours = [c for _, _, c in valid]

    text_labels = []
    for s, v, _ in valid:
        txt = f"<b>${v:,.0f}</b>"
        if cp > 0:
            upside = (v / cp - 1) * 100
            sign = "+" if upside >= 0 else ""
            txt += f"  <span style='color:{'#22c55e' if upside >= 0 else '#ef4444'}'>{sign}{upside:.1f}%</span>"
        # Add probability from scenario_table
        for row in (dcf.get("scenario_table") or []):
            if str(row.get("scenario", "")).lower() == s.lower():
                p = _to_float(row.get("probability"))
                if p is not None:
                    txt += f"  ({p:.0%})"
                break
        text_labels.append(txt)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=values,
        orientation="h",
        marker_color=colours,
        text=text_labels,
        textposition="outside",
        textfont=dict(size=11, color=_TEXT),
        cliponaxis=False,
        hovertemplate="%{y}: $%{x:,.2f}<extra></extra>",
        showlegend=False,
    ))

    if cp > 0:
        fig.add_vline(
            x=cp,
            line_dash="dot",
            line_color=_NEUTRAL_COLOUR,
            line_width=2,
            annotation_text=f"  ${cp:,.0f}  Current",
            annotation_position="top",
            annotation_font=dict(color=_NEUTRAL_COLOUR, size=11),
        )

    wacc_str = f"  WACC: {wacc * 100:.1f}%" if wacc is not None else ""
    x_max = max(values) * 1.40 if values else 1
    if cp:
        x_max = max(x_max, cp * 1.1)

    _apply_base_layout(fig, f"{ticker} — DCF Intrinsic Value Scenarios{wacc_str}", height=260)
    fig.update_xaxes(title_text="Intrinsic Value (USD)", range=[0, x_max],
                     tickprefix="$", tickformat=",.0f")
    fig.update_layout(margin=dict(l=70, r=130, t=50, b=50))
    return fig


# ── 8. Sensitivity Heatmap ────────────────────────────────────────────────────

def chart_sensitivity_heatmap(
    sensitivity_matrix: Dict[str, Any],
    ticker: str,
    current_price: Optional[float] = None,
) -> go.Figure:
    """WACC × terminal-growth-rate annotated heatmap.

    Colorscale anchored so current_price appears near the midpoint.
    """
    if not sensitivity_matrix:
        return _stub_fig(f"{ticker} — DCF Sensitivity", "No sensitivity matrix available")

    wacc_labels = list(sensitivity_matrix.keys())
    tgr_labels  = list(next(iter(sensitivity_matrix.values())).keys())

    z = []
    for wacc_k in wacc_labels:
        row_vals = []
        for tgr_k in tgr_labels:
            v = sensitivity_matrix[wacc_k].get(tgr_k)
            row_vals.append(float(v) if v is not None else 0.0)
        z.append(row_vals)

    all_vals = [v for row in z for v in row if v > 0]
    z_min = min(all_vals) if all_vals else 0
    z_max = max(all_vals) if all_vals else 1

    if current_price and z_min < current_price < z_max:
        zmid = current_price
    else:
        zmid = (z_min + z_max) / 2

    text_matrix = [[f"${v:,.0f}" for v in row] for row in z]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=tgr_labels,
        y=wacc_labels,
        colorscale="RdYlGn",
        zmid=zmid,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        hovertemplate="WACC: %{y}<br>TGR: %{x}<br>Value: $%{z:,.2f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Intrinsic<br>Value", font=dict(color=_TEXT, size=11)),
            tickfont=dict(color=_TEXT),
            tickprefix="$",
            tickformat=",.0f",
        ),
    ))

    _apply_base_layout(fig, f"{ticker} — DCF Sensitivity  (WACC × Terminal Growth Rate)", height=280)
    fig.update_xaxes(title_text="Terminal Growth Rate")
    fig.update_yaxes(title_text="WACC")
    return fig


# ── 9. Free Cash Flow Trend ───────────────────────────────────────────────────

def chart_fcf_trend(
    quarterly_trends: List[Dict[str, Any]],
    key_metrics: Dict[str, Any],
    ticker: str,
) -> go.Figure:
    """FCF proxy = net_income × fcf_conversion ratio from key_metrics.

    ``fcf_conversion`` (FCF/Net Income) comes from the QF agent's key_metrics.
    Applied uniformly to all quarterly net income values.  A caption note
    explains the proxy methodology.
    """
    if not quarterly_trends:
        return _stub_fig(f"{ticker} — Free Cash Flow Trend", "No quarterly data available")

    fcf_conv = _to_float(key_metrics.get("fcf_conversion"))
    rows = sorted(quarterly_trends, key=lambda r: str(r.get("period", "")))
    periods = [str(r.get("period", "?")) for r in rows]
    ni_vals = [_bn(r.get("net_income")) for r in rows]

    if not any(v is not None for v in ni_vals):
        return _stub_fig(f"{ticker} — Free Cash Flow Trend", "Net income data not available")

    if fcf_conv is not None:
        fcf_vals: List[Optional[float]] = [
            v * fcf_conv if v is not None else None for v in ni_vals
        ]
        conv_pct = fcf_conv * 100 if abs(fcf_conv) <= 5 else fcf_conv
        note = f"FCF = Net Income × {conv_pct:.0f}% conversion (LTM)"
    else:
        # No conversion ratio → show net income as FCF proxy with explicit warning
        fcf_vals = list(ni_vals)
        note = "FCF proxy: net income (fcf_conversion unavailable)"

    bar_colours = [_TEAL if (v or 0) >= 0 else _BEAR_COLOUR for v in fcf_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=periods, y=fcf_vals,
        name="FCF (proxy)",
        marker_color=bar_colours,
        text=[f"${v:.2f}B" if v is not None else "—" for v in fcf_vals],
        textposition="outside",
        textfont=dict(size=10, color=_TEXT),
        cliponaxis=False,
        hovertemplate="%{x}<br>FCF Proxy: $%{y:.2f}B<extra></extra>",
    ))

    # Zero reference line
    fig.add_hline(y=0, line_dash="dot", line_color=_GREY, line_width=1, opacity=0.6)

    _apply_base_layout(fig, f"{ticker} — Free Cash Flow Trend  ({note})", height=300)
    fig.update_xaxes(title_text="Quarter")
    fig.update_yaxes(title_text="FCF Proxy (USD Billions)", gridcolor=_GRID)
    fig.update_layout(margin=dict(l=70, r=60, t=55, b=50))
    return fig


# Stub alias retained for backward-compat
def chart_fcf_trend_stub(ticker: str) -> go.Figure:
    """Backward-compat alias → chart_fcf_trend with empty data."""
    return chart_fcf_trend([], {}, ticker)


# ── 10. Historical Price Chart ────────────────────────────────────────────────

def chart_price_history(
    price_history: List[Dict[str, Any]],
    technicals: Dict[str, Any],
    ticker: str,
    current_price: Optional[float] = None,
) -> go.Figure:
    """Candlestick OHLC + volume bars + 50/200-day SMAs + Bollinger band.

    *price_history* is the serialised OHLCV list forwarded from the FM agent
    (``fm_output["price_history"]``), sorted oldest-first.

    Overlays:
    - 50-day and 200-day SMA lines (computed in pure Python from close prices)
    - Bollinger Bands from technicals snapshot (shaded fill between upper/lower)
    - Current price horizontal dashed line
    - 52W high / 52W low reference lines
    """
    if not price_history:
        return _stub_fig(
            f"{ticker} — Historical Share Price",
            "Price series not available — no data forwarded from pipeline",
        )

    dates   = [r["date"]   for r in price_history]
    opens   = [_to_float(r.get("open"))   for r in price_history]
    highs   = [_to_float(r.get("high"))   for r in price_history]
    lows    = [_to_float(r.get("low"))    for r in price_history]
    closes  = [_to_float(r.get("close"))  for r in price_history]
    volumes = [_to_float(r.get("volume")) for r in price_history]

    # SMA
    sma50  = _rolling_sma(closes, 50)
    sma200 = _rolling_sma(closes, 200)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )

    # ── Candlestick ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dates,
        open=opens, high=highs, low=lows, close=closes,
        name="OHLC",
        increasing_line_color=_BULL_COLOUR,
        decreasing_line_color=_BEAR_COLOUR,
        increasing_fillcolor=_BULL_COLOUR,
        decreasing_fillcolor=_BEAR_COLOUR,
        line_width=1,
        hovertext=[
            f"O:{o:.2f}  H:{h:.2f}  L:{l:.2f}  C:{c:.2f}"
            for o, h, l, c in zip(
                [v or 0 for v in opens],
                [v or 0 for v in highs],
                [v or 0 for v in lows],
                [v or 0 for v in closes],
            )
        ],
    ), row=1, col=1)

    # ── SMAs ─────────────────────────────────────────────────────────────────
    if any(v is not None for v in sma50):
        fig.add_trace(go.Scatter(
            x=dates, y=sma50,
            name="SMA 50",
            mode="lines",
            line=dict(color=_NEUTRAL_COLOUR, width=1.5),
            hovertemplate="SMA50: %{y:.2f}<extra></extra>",
        ), row=1, col=1)

    if any(v is not None for v in sma200):
        fig.add_trace(go.Scatter(
            x=dates, y=sma200,
            name="SMA 200",
            mode="lines",
            line=dict(color=_VIOLET, width=1.5, dash="dot"),
            hovertemplate="SMA200: %{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # ── Bollinger Bands (from technicals snapshot — single current values) ───
    bb_upper = _to_float(technicals.get("bollinger_upper"))
    bb_lower = _to_float(technicals.get("bollinger_lower"))
    if bb_upper is not None and bb_lower is not None and dates:
        fig.add_hrect(
            y0=bb_lower, y1=bb_upper,
            fillcolor="rgba(99,102,241,0.07)",
            line_width=0,
            row=1, col=1,
        )
        for val, label in [(bb_upper, "BB Upper"), (bb_lower, "BB Lower")]:
            fig.add_hline(
                y=val, line_dash="dot", line_color=_INDIGO, line_width=1, opacity=0.7,
                annotation_text=f"  {label} {val:.2f}",
                annotation_font=dict(color=_INDIGO, size=9),
                annotation_position="right",
                row=1, col=1,
            )

    # ── Current price ─────────────────────────────────────────────────────────
    if current_price is not None:
        fig.add_hline(
            y=current_price, line_dash="dash", line_color=_TEXT, line_width=1.5, opacity=0.8,
            annotation_text=f"  Current ${current_price:.2f}",
            annotation_font=dict(color=_TEXT, size=9),
            annotation_position="right",
            row=1, col=1,
        )

    # ── 52W High / Low ────────────────────────────────────────────────────────
    high52 = _to_float(technicals.get("52w_high") or technicals.get("high_52w"))
    low52  = _to_float(technicals.get("52w_low")  or technicals.get("low_52w"))
    for val, label, col in [(high52, "52W High", _BULL_COLOUR), (low52, "52W Low", _BEAR_COLOUR)]:
        if val is not None:
            fig.add_hline(
                y=val, line_dash="longdash", line_color=col, line_width=1, opacity=0.5,
                annotation_text=f"  {label} ${val:.2f}",
                annotation_font=dict(color=col, size=9),
                annotation_position="right",
                row=1, col=1,
            )

    # ── Volume bars ───────────────────────────────────────────────────────────
    vol_colours = []
    for i, (o, c) in enumerate(zip(opens, closes)):
        if o is not None and c is not None:
            vol_colours.append(_BULL_COLOUR if c >= o else _BEAR_COLOUR)
        else:
            vol_colours.append(_GREY)

    fig.add_trace(go.Bar(
        x=dates, y=volumes,
        name="Volume",
        marker_color=vol_colours,
        opacity=0.6,
        showlegend=False,
        hovertemplate="Vol: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=f"{ticker} — Historical Share Price (OHLCV)", x=0.02, font=dict(size=14, color=_TEXT)),
        **_LAYOUT_BASE,
        height=520,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
    )
    fig.update_xaxes(gridcolor=_GRID, zerolinecolor=_GRID)
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor=_GRID)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1, showgrid=False)
    return fig


# Stub alias retained for backward-compat
def chart_price_history_stub(ticker: str) -> go.Figure:
    """Backward-compat alias → chart_price_history with empty data."""
    return chart_price_history([], {}, ticker)


# ── 11. Price Performance vs Index (indexed to 100) ──────────────────────────

def chart_price_performance(
    price_history: List[Dict[str, Any]],
    benchmark_history: List[Dict[str, Any]],
    ticker: str,
) -> go.Figure:
    """Multi-line indexed performance chart.

    Both the ticker and S&P 500 benchmark are indexed to 100 at the first
    common date.  Returns a stub if neither series is available.
    """
    if not price_history and not benchmark_history:
        return _stub_fig(
            f"{ticker} — Price Performance vs S&P 500",
            "No price or benchmark history available",
        )

    def _build_series(rows: List[Dict[str, Any]]) -> Tuple[List[str], List[Optional[float]]]:
        dates  = [r["date"]  for r in rows]
        closes = [_to_float(r.get("close")) for r in rows]
        return dates, closes

    t_dates, t_closes = _build_series(price_history)
    b_dates, b_closes = _build_series(benchmark_history)

    # Find first common date for indexing
    t_map = {d: c for d, c in zip(t_dates, t_closes) if c is not None}
    b_map = {d: c for d, c in zip(b_dates, b_closes) if c is not None}

    common = sorted(set(t_map) & set(b_map))
    if not common:
        # No overlap — index each series to its own start independently
        common_t = t_dates[0] if t_dates else None
        common_b = b_dates[0] if b_dates else None
        base_t = t_map.get(common_t) if common_t else None
        base_b = b_map.get(common_b) if common_b else None
    else:
        base_t = t_map[common[0]]
        base_b = b_map[common[0]]

    def _index(vals: List[Optional[float]], base: Optional[float]) -> List[Optional[float]]:
        if base is None or base == 0:
            return [None] * len(vals)
        return [v / base * 100 if v is not None else None for v in vals]

    t_indexed = _index(t_closes, base_t)
    b_indexed = _index(b_closes, base_b)

    fig = go.Figure()

    if any(v is not None for v in t_indexed):
        fig.add_trace(go.Scatter(
            x=t_dates, y=t_indexed,
            name=ticker,
            mode="lines",
            line=dict(color=_BASE_COLOUR, width=2.5),
            hovertemplate=f"{ticker}: %{{y:.1f}}<extra></extra>",
        ))

    if any(v is not None for v in b_indexed):
        fig.add_trace(go.Scatter(
            x=b_dates, y=b_indexed,
            name="S&P 500",
            mode="lines",
            line=dict(color=_GREY, width=1.5, dash="dot"),
            hovertemplate="S&P 500: %{y:.1f}<extra></extra>",
        ))

    # 100 baseline
    fig.add_hline(y=100, line_dash="longdash", line_color=_GRID, line_width=1, opacity=0.8)

    _apply_base_layout(fig, f"{ticker} — Price Performance vs S&P 500  (Indexed to 100)", height=320)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Indexed Return (base = 100)", gridcolor=_GRID)
    return fig


# Stub alias retained for backward-compat
def chart_price_performance_stub(ticker: str) -> go.Figure:
    """Backward-compat alias → chart_price_performance with empty data."""
    return chart_price_performance([], [], ticker)


# ── 12. MoE Consensus Dumbbell ───────────────────────────────────────────────

def chart_moe_consensus(
    moe: Dict[str, Any],
    current_price: Optional[float],
    ticker: str,
) -> go.Figure:
    """Persona Bear–Base–Bull dumbbell chart.

    One horizontal range per persona (bear→bull), diamond = base case.
    Consensus row appended if present.
    """
    personas   = moe.get("personas") or []
    cons_bear  = _to_float(moe.get("consensus_bear"))
    cons_base  = _to_float(moe.get("consensus_base"))
    cons_bull  = _to_float(moe.get("consensus_bull"))

    rows: List[tuple] = []
    persona_colours = {
        "optimist":  _BULL_COLOUR,
        "realist":   _BASE_COLOUR,
        "pessimist": _BEAR_COLOUR,
    }
    for p in personas:
        name  = str(p.get("name") or p.get("persona") or "Agent").strip()
        pb    = _to_float(p.get("bear"))
        pbase = _to_float(p.get("base"))
        pbull = _to_float(p.get("bull"))
        col   = persona_colours.get(name.lower(), _NEUTRAL_COLOUR)
        if any(v is not None for v in [pb, pbase, pbull]):
            rows.append((name, pb, pbase, pbull, col))

    if any(v is not None for v in [cons_bear, cons_base, cons_bull]):
        rows.append(("Consensus", cons_bear, cons_base, cons_bull, _VIOLET))

    if not rows:
        return _stub_fig(f"{ticker} — Analyst Consensus", "No MoE consensus data available")

    fig = go.Figure()

    for label, bear, base, bull, col in rows:
        if bear is not None and bull is not None:
            fig.add_trace(go.Scatter(
                x=[float(bear), float(bull)],
                y=[label, label],
                mode="lines",
                line=dict(color=col, width=5),
                name=label,
                legendgroup=label,
                hoverinfo="skip",
            ))
            for val, marker_col, sym, hover_lbl in [
                (bear, _BEAR_COLOUR, "circle", "Bear"),
                (bull, _BULL_COLOUR, "circle", "Bull"),
            ]:
                fig.add_trace(go.Scatter(
                    x=[float(val)], y=[label],
                    mode="markers",
                    marker=dict(color=marker_col, size=10, symbol=sym),
                    legendgroup=label,
                    showlegend=False,
                    hovertemplate=f"{label} {hover_lbl}: $%{{x:,.0f}}<extra></extra>",
                ))

        if base is not None:
            upside_txt = ""
            if current_price and current_price > 0:
                up = (float(base) / current_price - 1) * 100
                upside_txt = f"  {'+' if up >= 0 else ''}{up:.1f}%"
            fig.add_trace(go.Scatter(
                x=[float(base)], y=[label],
                mode="markers+text",
                marker=dict(color=col, size=14, symbol="diamond",
                            line=dict(color="white", width=1)),
                text=[f"<b>${float(base):,.0f}</b>{upside_txt}"],
                textposition="top center",
                textfont=dict(size=10, color=_TEXT),
                legendgroup=label,
                showlegend=False,
                hovertemplate=f"{label} Base: $%{{x:,.0f}}<extra></extra>",
            ))

    if current_price is not None:
        fig.add_vline(
            x=current_price,
            line_dash="dot",
            line_color=_NEUTRAL_COLOUR,
            line_width=2,
            annotation_text=f"  ${current_price:,.0f}",
            annotation_position="top",
            annotation_font=dict(color=_NEUTRAL_COLOUR, size=11),
        )

    _apply_base_layout(
        fig,
        f"{ticker} — Persona Price Targets  (Bear | Base ◆ | Bull)",
        height=max(280, 100 * len(rows)),
    )
    fig.update_xaxes(title_text="Price Target (USD)", tickprefix="$", tickformat=",.0f")
    return fig


# ── 13. Technical Indicators Panel ───────────────────────────────────────────

def chart_technicals(
    technicals: Dict[str, Any],
    ticker: str,
    current_price: Optional[float] = None,
) -> go.Figure:
    """Four-row technical indicator panel.

    Row 1: 52-week range swimlane with SMA markers.
    Row 2: RSI (14) bar with overbought/oversold bands.
    Row 3: MACD histogram + signal dot.
    Row 4: Stochastic %K / %D bars with bands.
    """
    if not technicals:
        return _stub_fig(f"{ticker} — Technical Indicators", "No technicals data available")

    lo52   = _to_float(technicals.get("52w_low"))
    hi52   = _to_float(technicals.get("52w_high"))
    sma20  = _to_float(technicals.get("sma_20"))
    sma50  = _to_float(technicals.get("sma_50"))
    sma200 = _to_float(technicals.get("sma_200"))
    supp   = _to_float(technicals.get("support"))
    res    = _to_float(technicals.get("resistance"))
    rsi    = _to_float(technicals.get("rsi_14"))
    macd_val  = _to_float(technicals.get("macd"))
    macd_sig  = _to_float(technicals.get("macd_signal"))
    macd_hist = _to_float(technicals.get("macd_histogram"))
    stoch_k = _to_float(technicals.get("stochastic_k"))
    stoch_d = _to_float(technicals.get("stochastic_d"))

    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[0.35, 0.22, 0.22, 0.21],
        shared_xaxes=False,
        subplot_titles=[
            "52-Week Price Range & Key Levels",
            "RSI (14)",
            "MACD",
            "Stochastic %K / %D",
        ],
        vertical_spacing=0.10,
    )

    # ── Row 1: 52-week swimlane ───────────────────────────────────────────────
    if lo52 is not None and hi52 is not None:
        fig.add_trace(go.Bar(
            x=[hi52 - lo52], y=["Range"],
            base=[lo52], orientation="h",
            marker_color="rgba(100,116,139,0.25)",
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)
        if current_price and lo52 < current_price < hi52:
            fig.add_trace(go.Bar(
                x=[current_price - lo52], y=["Range"],
                base=[lo52], orientation="h",
                marker_color="rgba(34,197,94,0.35)",
                hoverinfo="skip", showlegend=False,
            ), row=1, col=1)

    level_items = [
        ("52W Low",    lo52,          _BEAR_COLOUR,    "triangle-right"),
        ("52W High",   hi52,          _BULL_COLOUR,    "triangle-left"),
        ("Support",    supp,          "#fb923c",       "line-ns"),
        ("Resistance", res,           "#f472b6",       "line-ns"),
        ("SMA 200",    sma200,        _GREY,           "circle"),
        ("SMA 50",     sma50,         "#60a5fa",       "circle"),
        ("SMA 20",     sma20,         _VIOLET,         "circle"),
        ("Current",    current_price, _NEUTRAL_COLOUR, "diamond"),
    ]
    for label, val, col, sym in level_items:
        if val is None:
            continue
        fig.add_trace(go.Scatter(
            x=[val], y=["Range"],
            mode="markers+text",
            marker=dict(size=14, color=col, symbol=sym,
                        line=dict(color="white", width=1)),
            text=[f"${val:,.0f}"],
            textposition="top center" if label in ("SMA 20", "SMA 50", "SMA 200", "Current") else "bottom center",
            textfont=dict(size=9, color=col),
            name=label,
            hovertemplate=f"{label}: $%{{x:,.2f}}<extra></extra>",
            showlegend=True,
        ), row=1, col=1)

    fig.update_yaxes(showticklabels=False, row=1, col=1)
    if lo52 is not None and hi52 is not None:
        pad = (hi52 - lo52) * 0.05
        fig.update_xaxes(range=[lo52 - pad, hi52 + pad], tickprefix="$", row=1, col=1)

    # ── Row 2: RSI ────────────────────────────────────────────────────────────
    if rsi is not None:
        rsi_col = _BEAR_COLOUR if rsi > 70 else (_BULL_COLOUR if rsi < 30 else _BASE_COLOUR)
        fig.add_trace(go.Bar(
            x=["RSI"], y=[rsi],
            marker_color=rsi_col,
            text=[f"{rsi:.1f}"], textposition="outside",
            textfont=dict(size=12),
            hovertemplate="RSI (14): %{y:.1f}<extra></extra>",
            showlegend=False,
        ), row=2, col=1)
        fig.update_yaxes(range=[0, 110], row=2, col=1)
        fig.add_shape(type="rect", x0=-0.5, x1=0.5, y0=70, y1=100,
                      fillcolor="rgba(239,68,68,0.12)", line_width=0, row=2, col=1)
        fig.add_shape(type="rect", x0=-0.5, x1=0.5, y0=0, y1=30,
                      fillcolor="rgba(34,197,94,0.12)", line_width=0, row=2, col=1)
        for level, label in [(70, "OB 70"), (30, "OS 30")]:
            fig.add_hline(y=level, line_dash="dash", line_color=_GREY, line_width=1,
                          annotation_text=label, annotation_font_size=9,
                          annotation_font_color=_GREY, row=2, col=1)

    # ── Row 3: MACD ───────────────────────────────────────────────────────────
    hist_val = macd_hist if macd_hist is not None else macd_val
    if hist_val is not None:
        fig.add_trace(go.Bar(
            x=["MACD"], y=[hist_val],
            marker_color=_BULL_COLOUR if hist_val >= 0 else _BEAR_COLOUR,
            text=[f"{hist_val:+.3f}"], textposition="outside",
            textfont=dict(size=12),
            hovertemplate="MACD: %{y:+.4f}<extra></extra>",
            showlegend=False,
        ), row=3, col=1)
        if macd_sig is not None:
            fig.add_trace(go.Scatter(
                x=["MACD"], y=[macd_sig],
                mode="markers",
                marker=dict(color=_NEUTRAL_COLOUR, size=12, symbol="circle-open",
                            line=dict(width=2)),
                name="Signal",
                hovertemplate="Signal: %{y:+.4f}<extra></extra>",
                showlegend=False,
            ), row=3, col=1)
        fig.add_hline(y=0, line_color=_GREY, line_width=1, row=3, col=1)

    # ── Row 4: Stochastic ─────────────────────────────────────────────────────
    if stoch_k is not None or stoch_d is not None:
        for lbl, val, col in [("%K", stoch_k, _BASE_COLOUR), ("%D", stoch_d, _NEUTRAL_COLOUR)]:
            if val is None:
                continue
            fig.add_trace(go.Bar(
                x=[lbl], y=[val],
                marker_color=col,
                text=[f"{val:.1f}"], textposition="outside",
                textfont=dict(size=12),
                hovertemplate=f"{lbl}: %{{y:.1f}}<extra></extra>",
                showlegend=False,
            ), row=4, col=1)
        fig.update_yaxes(range=[0, 110], row=4, col=1)
        fig.add_shape(type="rect", x0=-0.5, x1=1.5, y0=80, y1=100,
                      fillcolor="rgba(239,68,68,0.12)", line_width=0, row=4, col=1)
        fig.add_shape(type="rect", x0=-0.5, x1=1.5, y0=0, y1=20,
                      fillcolor="rgba(34,197,94,0.12)", line_width=0, row=4, col=1)
        for level in (80, 20):
            fig.add_hline(y=level, line_dash="dash", line_color=_GREY, line_width=1, row=4, col=1)

    fig.update_layout(
        title=dict(text=f"{ticker} — Technical Indicators", x=0.02, font=dict(size=14, color=_TEXT)),
        height=620,
        **_LAYOUT_BASE,
    )
    fig.update_xaxes(gridcolor=_GRID, zerolinecolor=_GRID)
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor=_GRID)
    return fig


# ── 14. Sentiment Donut ───────────────────────────────────────────────────────

def chart_sentiment_donut(
    sentiment: Dict[str, Any],
    ticker: str,
) -> go.Figure:
    """Bullish / Bearish / Neutral donut chart from BA agent."""
    labels_d: List[str]  = []
    values_d: List[float] = []
    colours_d: List[str] = []
    for label, key, col in [
        ("Bullish", "bullish_pct", _BULL_COLOUR),
        ("Neutral", "neutral_pct", _NEUTRAL_COLOUR),
        ("Bearish", "bearish_pct", _BEAR_COLOUR),
    ]:
        val = _to_float(sentiment.get(key))
        if val is not None:
            labels_d.append(label)
            values_d.append(val)
            colours_d.append(col)

    if not values_d:
        return _stub_fig(f"{ticker} — Market Sentiment", "No sentiment data available")

    if max(values_d) <= 1.0:
        values_d = [v * 100 for v in values_d]

    dominant_idx = values_d.index(max(values_d))

    fig = go.Figure(data=go.Pie(
        labels=labels_d,
        values=values_d,
        hole=0.55,
        marker_colors=colours_d,
        texttemplate="%{label}<br><b>%{value:.1f}%</b>",
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        textfont=dict(size=12),
        pull=[0.04 if i == dominant_idx else 0 for i in range(len(labels_d))],
    ))

    fig.update_layout(
        annotations=[dict(
            text=f"<b>{labels_d[dominant_idx]}</b>",
            x=0.5, y=0.5,
            font=dict(size=16, color=colours_d[dominant_idx]),
            showarrow=False,
        )]
    )

    trend = sentiment.get("trend", "")
    trend_str = f"  Trend: {trend}" if trend else ""
    _apply_base_layout(fig, f"{ticker} — Market Sentiment{trend_str}")
    return fig


# ── 15. Quality Factor Scorecard ──────────────────────────────────────────────

def chart_factor_scorecard(
    qf_output: Dict[str, Any],
    ticker: str,
) -> go.Figure:
    """Horizontal bar scorecard for quality, profitability and risk factors.

    Each factor is normalised to 0–100 and colour-coded red/amber/green.
    Raw values are annotated outside the bar for precision.
    """
    quality  = qf_output.get("quality_factors") or {}
    momentum = qf_output.get("momentum_risk") or {}
    km       = qf_output.get("key_metrics") or {}

    piotroski = _to_float(quality.get("piotroski_f_score"))
    roe       = _to_float(quality.get("roe"))
    roic      = _to_float(quality.get("roic"))
    gross_m   = _to_float(km.get("gross_margin"))
    sharpe    = _to_float(momentum.get("sharpe_ratio_12m"))
    beta      = _to_float(momentum.get("beta"))
    altman    = _to_float(quality.get("altman_z_score"))
    beneish   = _to_float(quality.get("beneish_m_score"))

    def _clamp(val: Optional[float], lo: float, hi: float) -> Optional[float]:
        if val is None:
            return None
        return max(0.0, min(1.0, (val - lo) / (hi - lo))) * 100

    def _altman_score(z: Optional[float]) -> Optional[float]:
        if z is None:
            return None
        if z >= 3.0:
            return 100.0
        if z <= 1.23:
            return 0.0
        return (z - 1.23) / (3.0 - 1.23) * 100

    def _beneish_score(m: Optional[float]) -> Optional[float]:
        # Beneish M < -2.22 = unlikely manipulator (good), > -1.78 = potential manipulator (bad)
        if m is None:
            return None
        if m <= -2.22:
            return 100.0
        if m >= -1.78:
            return 0.0
        return (-2.22 - m) / (-2.22 - (-1.78)) * 100 + 50

    # (factor_name, normalised_0-100, raw_value, raw_format_string)
    factors = [
        ("Piotroski F-Score",  _clamp(piotroski, 0, 9),        piotroski, "{:.0f} / 9"),
        ("ROE",                _clamp(roe, 0.0, 0.40),          roe,       "{:.1%}"),
        ("ROIC",               _clamp(roic, 0.0, 0.40),         roic,      "{:.1%}"),
        ("Gross Margin",       _clamp(gross_m, 0.0, 0.80),      gross_m,   "{:.1%}"),
        ("Sharpe (12M)",       _clamp(max(0, sharpe) if sharpe is not None else None, 0.0, 3.0),
                               sharpe, "{:.2f}"),
        ("Altman Z-Score",     _altman_score(altman),            altman,    "{:.2f}"),
        ("Beneish M-Score",    _beneish_score(beneish),          beneish,   "{:.2f}"),
        ("Beta (stability)",   _clamp(2.5 - (beta or 0), 0.0, 2.5) if beta is not None else None,
                               beta, "{:.2f}"),
    ]

    avail = [(n, score, raw, fmt) for n, score, raw, fmt in factors if score is not None]
    if not avail:
        return _stub_fig(f"{ticker} — Factor Scorecard", "No factor data available")

    names  = [x[0] for x in avail]
    scores = [x[1] for x in avail]
    raws   = [x[2] for x in avail]
    fmts   = [x[3] for x in avail]

    bar_colours = [
        _BEAR_COLOUR if s < 35 else (_NEUTRAL_COLOUR if s < 65 else _BULL_COLOUR)
        for s in scores
    ]

    raw_labels = []
    for raw, fmt in zip(raws, fmts):
        try:
            raw_labels.append(fmt.format(raw))
        except Exception:
            raw_labels.append(str(raw) if raw is not None else "—")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names,
        x=scores,
        orientation="h",
        marker_color=bar_colours,
        text=raw_labels,
        textposition="outside",
        textfont=dict(size=11, color=_TEXT),
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.0f}/100 (raw: %{text})<extra></extra>",
        showlegend=False,
    ))

    # Background zone bands
    for x0, x1, col in [
        (0,  35,  "rgba(239,68,68,0.08)"),
        (35, 65,  "rgba(245,158,11,0.08)"),
        (65, 100, "rgba(34,197,94,0.08)"),
    ]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=-0.5, y1=len(names) - 0.5,
                      fillcolor=col, line_width=0)

    _apply_base_layout(
        fig,
        f"{ticker} — Quality Factor Scorecard  (0 = weak | 100 = strong)",
        height=max(280, 50 * len(avail)),
    )
    fig.update_xaxes(range=[0, 130], title_text="Normalised Score (0–100)")
    fig.update_layout(margin=dict(l=155, r=90, t=50, b=50))
    return fig


# ── Backward-compat aliases (old hint names still work) ──────────────────────
# These are thin wrappers so that any chart_hints strings from the LLM planner
# that still use the old names don't crash _render_visualisations.

def chart_dcf_scenarios(
    dcf: Dict[str, Any],
    current_price: Optional[float],
    ticker: str,
) -> go.Figure:
    """Alias → chart_dcf_waterfall."""
    return chart_dcf_waterfall(dcf, current_price, ticker)


def chart_quarterly_trends(
    quarterly_trends: List[Dict[str, Any]],
    ticker: str,
) -> go.Figure:
    """Alias → chart_revenue_trend (no DCF forward data)."""
    return chart_revenue_trend(quarterly_trends, {}, ticker)


def chart_factor_radar(
    qf_output: Dict[str, Any],
    ticker: str,
) -> go.Figure:
    """Alias → chart_factor_scorecard."""
    return chart_factor_scorecard(qf_output, ticker)


def chart_altman_z(
    qf_output: Dict[str, Any],
    ticker: str,
) -> go.Figure:
    """Standalone Altman Z-score gauge (kept for backward compat with old hints)."""
    quality = qf_output.get("quality_factors") or {}
    z = _to_float(quality.get("altman_z_score"))

    if z is None:
        return _stub_fig(f"{ticker} — Altman Z-Score", "No Altman Z-score available")

    gauge_max = max(6.0, z * 1.2)
    if z <= 1.23:
        zone_label = "Distress Zone"
        gauge_colour = _BEAR_COLOUR
    elif z <= 3.0:
        zone_label = "Grey Zone"
        gauge_colour = _NEUTRAL_COLOUR
    else:
        zone_label = "Safe Zone"
        gauge_colour = _BULL_COLOUR

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=z,
        number=dict(font=dict(color=gauge_colour, size=38), valueformat=".2f"),
        delta=dict(
            reference=2.99,
            valueformat=".2f",
            increasing=dict(color=_BULL_COLOUR),
            decreasing=dict(color=_BEAR_COLOUR),
        ),
        gauge=dict(
            axis=dict(range=[0, gauge_max], tickwidth=1, tickcolor=_GRID,
                      tickfont=dict(color=_TEXT, size=10)),
            bar=dict(color=gauge_colour, thickness=0.5),
            bgcolor=_CARD_BG,
            borderwidth=0,
            steps=[
                dict(range=[0, 1.23],        color="rgba(239,68,68,0.25)"),
                dict(range=[1.23, 3.0],      color="rgba(245,158,11,0.25)"),
                dict(range=[3.0, gauge_max], color="rgba(34,197,94,0.25)"),
            ],
            threshold=dict(line=dict(color=gauge_colour, width=4), thickness=0.75, value=z),
        ),
        title=dict(text=f"<b>{zone_label}</b>", font=dict(color=gauge_colour, size=13)),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        margin=dict(l=30, r=30, t=60, b=20),
        title=dict(
            text=f"{ticker} — Altman Z-Score  (< 1.23 Distress | 1.23–3.0 Grey | > 3.0 Safe)",
            x=0.02,
            font=dict(size=13, color=_TEXT),
        ),
        height=270,
    )
    return fig
