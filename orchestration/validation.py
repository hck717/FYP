# orchestration/validation.py
"""
Structured data validation layer for multi-agent summarisation.

Provides:
- Pydantic models for all agent output schemas
- QuantMetrics: typed container for all quantitative metrics that the
  summariser must report verbatim (anti-hallucination anchor)
- validate_quant_output(): normalises a raw quant agent dict into QuantMetrics
- FactChecker: verifies that numeric values in the generated report match
  the ground-truth QuantMetrics within tolerance
- build_locked_data_anchor(): returns a formatted string block injected into
  the LLM prompt so the model has a single authoritative source of truth
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic-style typed container (no external dependency required)
# ---------------------------------------------------------------------------

class QuantMetrics:
    """Typed, validated container for key quantitative metrics.

    All numeric fields are stored as Python floats (or None if unavailable).
    Percentage fields are stored in *percent* form (e.g. 152.02 means 152.02 %).
    """

    __slots__ = (
        "ticker",
        "pe_trailing",
        "ev_ebitda",
        "p_fcf",
        "piotroski_f_score",      # integer 0-9
        "beneish_m_score",        # float, typically -6 to +6
        "altman_z_score",         # float
        "roe_pct",                # e.g. 152.02
        "roic_pct",               # e.g. 57.99
        "revenue_ttm",            # in USD millions
        "net_income_ttm",
        "eps_ttm",
        "beta",                   # 60-day beta vs S&P 500
        "sharpe_ratio",           # 12-month Sharpe ratio
        "revenue_ttm_b",          # TTM revenue in billions (for FactChecker)
        "revenue_latest_q_b",     # Latest quarter revenue in billions
        "net_income_ttm_b",       # TTM net income in billions
        "net_income_latest_q_b",  # Latest quarter net income in billions
        "ocf_b",                  # Most recent annual operating cash flow in billions
        "fcf_b",                  # Most recent annual free cash flow in billions
        "revenue_yoy_pct",        # YoY revenue growth % (most recent quarter vs same quarter prior year)
        "net_income_yoy_pct",     # YoY net income growth %
        "total_assets_b",         # Most recent annual total assets in billions
        "shareholders_equity_b",  # Most recent annual shareholders equity in billions
        "return_12m_pct",         # 12-month price return %
        "rsi_14",                 # RSI(14) current value
    )

    def __init__(
        self,
        ticker: str,
        pe_trailing: Optional[float] = None,
        ev_ebitda: Optional[float] = None,
        p_fcf: Optional[float] = None,
        piotroski_f_score: Optional[int] = None,
        beneish_m_score: Optional[float] = None,
        altman_z_score: Optional[float] = None,
        roe_pct: Optional[float] = None,
        roic_pct: Optional[float] = None,
        revenue_ttm: Optional[float] = None,
        net_income_ttm: Optional[float] = None,
        eps_ttm: Optional[float] = None,
        beta: Optional[float] = None,
        sharpe_ratio: Optional[float] = None,
        revenue_ttm_b: Optional[float] = None,
        revenue_latest_q_b: Optional[float] = None,
        net_income_ttm_b: Optional[float] = None,
        net_income_latest_q_b: Optional[float] = None,
        ocf_b: Optional[float] = None,
        fcf_b: Optional[float] = None,
        revenue_yoy_pct: Optional[float] = None,
        net_income_yoy_pct: Optional[float] = None,
        total_assets_b: Optional[float] = None,
        shareholders_equity_b: Optional[float] = None,
        return_12m_pct: Optional[float] = None,
        rsi_14: Optional[float] = None,
    ) -> None:
        self.ticker = ticker
        self.pe_trailing = pe_trailing
        self.ev_ebitda = ev_ebitda
        self.p_fcf = p_fcf
        self.piotroski_f_score = piotroski_f_score
        self.beneish_m_score = beneish_m_score
        self.altman_z_score = altman_z_score
        self.roe_pct = roe_pct
        self.roic_pct = roic_pct
        self.revenue_ttm = revenue_ttm
        self.net_income_ttm = net_income_ttm
        self.eps_ttm = eps_ttm
        self.beta = beta
        self.sharpe_ratio = sharpe_ratio
        self.revenue_ttm_b = revenue_ttm_b
        self.revenue_latest_q_b = revenue_latest_q_b
        self.net_income_ttm_b = net_income_ttm_b
        self.net_income_latest_q_b = net_income_latest_q_b
        self.ocf_b = ocf_b
        self.fcf_b = fcf_b
        self.revenue_yoy_pct = revenue_yoy_pct
        self.net_income_yoy_pct = net_income_yoy_pct
        self.total_assets_b = total_assets_b
        self.shareholders_equity_b = shareholders_equity_b
        self.return_12m_pct = return_12m_pct
        self.rsi_14 = rsi_14

    def to_dict(self) -> Dict[str, Any]:
        return {s: getattr(self, s) for s in self.__slots__}

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items() if v is not None)
        return f"QuantMetrics({items})"


def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert *v* to float, returning *default* on failure."""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    if v is None:
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def validate_quant_output(quant_output: Dict[str, Any], fm_output: Optional[Dict[str, Any]] = None) -> Optional[QuantMetrics]:
    """Normalise a raw quant_fundamental agent output dict into a QuantMetrics.

    Args:
        quant_output: Raw quant_fundamental agent output dict.
        fm_output: Optional financial_modelling agent output dict for cash flow data.

    Returns None if *quant_output* is falsy or missing essential fields.
    """
    if not quant_output:
        return None

    ticker = quant_output.get("ticker", "UNKNOWN")
    vf = quant_output.get("value_factors") or {}
    qf = quant_output.get("quality_factors") or {}
    km = quant_output.get("key_metrics") or {}

    # Retrieve raw values from multiple possible locations
    pe = _safe_float(vf.get("pe_trailing"))
    ev_ebitda = _safe_float(vf.get("ev_ebitda"))
    p_fcf = _safe_float(vf.get("p_fcf"))

    piotroski = _safe_int(qf.get("piotroski_f_score"))
    beneish = _safe_float(qf.get("beneish_m_score"))
    altman = _safe_float(qf.get("altman_z_score"))

    roe_raw = _safe_float(qf.get("roe"))
    roic_raw = _safe_float(qf.get("roic"))
    roe_pct = round(roe_raw * 100, 4) if roe_raw is not None else None
    roic_pct = round(roic_raw * 100, 4) if roic_raw is not None else None

    # Financial statement figures (may come from key_metrics or value_factors)
    revenue = _safe_float(km.get("revenue_ttm") or vf.get("revenue_ttm"))
    net_income = _safe_float(km.get("net_income_ttm") or vf.get("net_income_ttm"))
    eps = _safe_float(km.get("eps_ttm") or vf.get("eps_ttm"))

    # Compute TTM from quarterly_trends if available (more reliable source)
    qt = quant_output.get("quarterly_trends") or []
    revenue_ttm_b: Optional[float] = None
    revenue_latest_q_b: Optional[float] = None
    net_income_ttm_b: Optional[float] = None
    net_income_latest_q_b: Optional[float] = None
    if qt:
        rev_q0 = qt[0].get("revenue")
        ni_q0  = qt[0].get("net_income")
        if rev_q0 is not None:
            revenue_latest_q_b = round(float(rev_q0) / 1e9, 2)
        if ni_q0 is not None:
            net_income_latest_q_b = round(float(ni_q0) / 1e9, 2)
        if len(qt) >= 4:
            ttm_r = sum((q.get("revenue") or 0) for q in qt[:4])
            ttm_n = sum((q.get("net_income") or 0) for q in qt[:4])
            if ttm_r:
                revenue_ttm_b = round(ttm_r / 1e9, 2)
                if revenue is None:
                    revenue = ttm_r  # fallback to computed TTM
            if ttm_n:
                net_income_ttm_b = round(ttm_n / 1e9, 2)
                if net_income is None:
                    net_income = ttm_n

    # OCF and FCF from financial_modelling output (most recent annual)
    ocf_b: Optional[float] = None
    fcf_b: Optional[float] = None
    total_assets_b: Optional[float] = None
    shareholders_equity_b: Optional[float] = None
    if fm_output:
        tsm = fm_output.get("three_statement_model") or {}
        cf_stmts = tsm.get("cash_flows") or []
        bs_stmts = tsm.get("balance_sheets") or []
        if cf_stmts:
            cf0 = cf_stmts[0]
            ocf_raw = cf0.get("operating_cash_flow")
            fcf_raw = cf0.get("free_cash_flow")
            if ocf_raw is not None:
                ocf_b = round(float(ocf_raw) / 1e9, 2)
            if fcf_raw is not None:
                fcf_b = round(float(fcf_raw) / 1e9, 2)
        if bs_stmts:
            bs0 = bs_stmts[0]
            tot_a = bs0.get("total_assets")
            tot_l = bs0.get("total_liabilities")
            if tot_a is not None:
                total_assets_b = round(float(tot_a) / 1e9, 2)
            if tot_a is not None and tot_l is not None:
                shareholders_equity_b = round((float(tot_a) - float(tot_l)) / 1e9, 2)

    # Risk/momentum metrics
    mr = quant_output.get("momentum_risk") or quant_output.get("momentum_factors") or {}
    beta = _safe_float(mr.get("beta_60d"))
    sharpe = _safe_float(mr.get("sharpe_ratio_12m"))
    return_12m_pct = _safe_float(mr.get("return_12m_pct") or mr.get("return_12m") or mr.get("price_return_12m"))
    rsi_14 = _safe_float(mr.get("rsi_14") or mr.get("rsi") or mr.get("rsi14"))

    # YoY growth rates
    yoy = quant_output.get("yoy_deltas") or {}
    revenue_yoy_pct = _safe_float(yoy.get("revenue_yoy_pct"))
    net_income_yoy_pct = _safe_float(yoy.get("net_income_yoy_pct"))

    metrics = QuantMetrics(
        ticker=ticker,
        pe_trailing=pe,
        ev_ebitda=ev_ebitda,
        p_fcf=p_fcf,
        piotroski_f_score=piotroski,
        beneish_m_score=beneish,
        altman_z_score=altman,
        roe_pct=roe_pct,
        roic_pct=roic_pct,
        revenue_ttm=revenue,
        net_income_ttm=net_income,
        eps_ttm=eps,
        beta=beta,
        sharpe_ratio=sharpe,
        revenue_ttm_b=revenue_ttm_b,
        revenue_latest_q_b=revenue_latest_q_b,
        net_income_ttm_b=net_income_ttm_b,
        net_income_latest_q_b=net_income_latest_q_b,
        ocf_b=ocf_b,
        fcf_b=fcf_b,
        revenue_yoy_pct=revenue_yoy_pct,
        net_income_yoy_pct=net_income_yoy_pct,
        total_assets_b=total_assets_b,
        shareholders_equity_b=shareholders_equity_b,
        return_12m_pct=return_12m_pct,
        rsi_14=rsi_14,
    )
    logger.debug("[validate_quant_output] %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Locked data anchor builder
# ---------------------------------------------------------------------------

def build_locked_data_anchor(metrics_list: List[QuantMetrics]) -> str:
    """Return a prompt block listing authoritative DB values for every ticker.

    This block is injected verbatim into the LLM user prompt so the model has
    a single source of truth that overrides any training-time knowledge.
    """
    if not metrics_list:
        return ""

    lines: List[str] = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║        LOCKED DATA ANCHOR — AUTHORITATIVE DB VALUES          ║",
        "║  Use EXACTLY these values. Do NOT use memory or estimates.   ║",
        "╚══════════════════════════════════════════════════════════════╝",
    ]

    for m in metrics_list:
        lines.append(f"\nTicker: {m.ticker}")
        if m.pe_trailing is not None:
            lines.append(f"  • P/E (trailing):       EXACTLY {m.pe_trailing:.4f}x")
        if m.ev_ebitda is not None:
            lines.append(f"  • EV/EBITDA:            EXACTLY {m.ev_ebitda:.4f}x")
        if m.p_fcf is not None:
            lines.append(f"  • P/FCF:                EXACTLY {m.p_fcf:.4f}x")
        if m.piotroski_f_score is not None:
            lines.append(
                f"  • Piotroski F-Score:    EXACTLY {m.piotroski_f_score}/9"
                f"  (WARNING: DB value {m.piotroski_f_score} may differ from your training data — use DB value anyway)"
            )
        if m.beneish_m_score is not None:
            lines.append(
                f"  • Beneish M-Score:      EXACTLY {m.beneish_m_score:.4f}"
                f"  (WARNING: DB value {m.beneish_m_score:.4f} may differ from your training data — use DB value anyway)"
            )
        if m.altman_z_score is not None:
            lines.append(
                f"  • Altman Z-Score:       EXACTLY {m.altman_z_score:.3f}"
                f"  (WARNING: DB value {m.altman_z_score:.3f} may differ from your training data — use DB value anyway)"
            )
        if m.roe_pct is not None:
            lines.append(f"  • ROE:                  EXACTLY {m.roe_pct:.2f}%")
        if m.roic_pct is not None:
            lines.append(f"  • ROIC:                 EXACTLY {m.roic_pct:.2f}%")
        if m.beta is not None:
            lines.append(
                f"  • Beta (60-day):        EXACTLY {m.beta:.4f}"
                f"  (WARNING: DB value {m.beta:.4f} may differ from your training data — use DB value anyway)"
            )
        if m.sharpe_ratio is not None:
            lines.append(
                f"  • Sharpe Ratio (12m):   EXACTLY {m.sharpe_ratio:.4f}"
                f"  (WARNING: DB value {m.sharpe_ratio:.4f} may differ from your training data — use DB value anyway)"
            )
        if m.revenue_latest_q_b is not None:
            lines.append(f"  • Latest quarterly revenue: EXACTLY ${m.revenue_latest_q_b:.2f}B")
        if m.revenue_ttm_b is not None:
            lines.append(f"  • TTM revenue:          EXACTLY ${m.revenue_ttm_b:.2f}B")
        if m.net_income_ttm_b is not None:
            lines.append(f"  • TTM net income:       EXACTLY ${m.net_income_ttm_b:.2f}B")
        if m.net_income_latest_q_b is not None:
            lines.append(f"  • Latest quarterly net income: EXACTLY ${m.net_income_latest_q_b:.2f}B")
        if m.ocf_b is not None:
            lines.append(f"  • Operating cash flow:  EXACTLY ${m.ocf_b:.2f}B")
        if m.fcf_b is not None:
            lines.append(f"  • Free cash flow:       EXACTLY ${m.fcf_b:.2f}B")
        if m.total_assets_b is not None:
            lines.append(
                f"  • Total assets (FY2024):        EXACTLY ${m.total_assets_b:.2f}B"
                f"  (WARNING: use this value, not FY2025 figure)"
            )
        if m.shareholders_equity_b is not None:
            lines.append(
                f"  • Shareholders equity (FY2024): EXACTLY ${m.shareholders_equity_b:.2f}B"
                f"  (WARNING: use this value, not FY2025 figure)"
            )
        if m.revenue_yoy_pct is not None:
            lines.append(
                f"  • YoY revenue growth (latest Q): EXACTLY {m.revenue_yoy_pct:.1f}%"
                f"  (WARNING: DB value {m.revenue_yoy_pct:.1f}% may differ from your training data — use DB value anyway)"
            )
        if m.net_income_yoy_pct is not None:
            lines.append(
                f"  • YoY net income growth (latest Q): EXACTLY {m.net_income_yoy_pct:.1f}%"
            )
        if m.return_12m_pct is not None:
            lines.append(
                f"  • 12-month price return:        EXACTLY {m.return_12m_pct:.2f}%"
                f"  (WARNING: DB value {m.return_12m_pct:.2f}% may differ from your training data — use DB value anyway)"
            )
        if m.rsi_14 is not None:
            lines.append(
                f"  • RSI(14):                      EXACTLY {m.rsi_14:.2f}"
                f"  (WARNING: DB value {m.rsi_14:.2f} may differ from your training data — use DB value anyway)"
            )

    lines.append(
        "\n⚠ CRITICAL: Any deviation from the values above constitutes a factual error."
        "\n  If the value seems unusual (e.g. Piotroski 2/9 instead of the typical 6-8),"
        "\n  that is the correct DB-sourced value — use it and explain it analytically."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fact-checker: verify generated report numeric values against ground truth
# ---------------------------------------------------------------------------

# Each entry: (label, regex_pattern, metrics_attr, is_percent)
# regex_pattern should capture the numeric value in group 1
_METRIC_CHECKS: List[Tuple[str, str, str, bool]] = [
    # P/E: handles "P/E of 33.7", "P/E ratio 33.7x", "trailing P/E: 33.7", "P/E: 33.7"
    ("P/E",           r"(?:trailing\s+)?P[/\\]E(?:\s+(?:ratio|multiple))?[\s:of]*(-?\d+(?:\.\d+)?)x?",         "pe_trailing",       False),
    # P/E reversed: "33.7x P/E" or "33.7 P/E"
    ("P/E_rev",       r"(-?\d+(?:\.\d+)?)x?\s+P[/\\]E\b",                                                      "pe_trailing",       False),
    # EV/EBITDA: handles "EV/EBITDA of 24.2x", "EV/EBITDA: 24.2", "EV/EBITDA multiple 24.2"
    ("EV/EBITDA",     r"EV[/\\]EBITDA(?:\s+(?:multiple|ratio))?[\s:of]*(-?\d+(?:\.\d+)?)x?",                   "ev_ebitda",         False),
    # EV/EBITDA reversed: "24.2x EV/EBITDA"
    ("EV/EBITDA_rev", r"(-?\d+(?:\.\d+)?)x?\s+EV[/\\]EBITDA\b",                                                "ev_ebitda",         False),
    # Piotroski: handles "Piotroski F-Score of 6", "Piotroski score: 6", "Piotroski F score 6/9",
    # "Piotroski F-Score of only 6" (word between of and number)
    ("Piotroski",     r"Piotroski\s*(?:F[-\s]?)?\s*[Ss]core(?:\s+of)?(?:\s+\w+){0,2}\s+(-?\d+(?:\.\d+)?)(?:/9)?", "piotroski_f_score", False),
    # Piotroski reversed: "a score of 6 on the Piotroski" or "6 Piotroski F-Score"
    ("Piotroski_rev", r"(-?\d+(?:\.\d+)?)\s*(?:/9)?\s*(?:on\s+(?:the\s+)?)?Piotroski",                         "piotroski_f_score", False),
    # Beneish: handles "Beneish M-Score of -6.2", "Beneish score: -6.2", "Beneish M score -6.2"
    ("Beneish",       r"Beneish\s*(?:M[-\s]?)?\s*[Ss]core[\s:of]*(-?\d+(?:\.\d+)?)",                           "beneish_m_score",   False),
    # Beneish reversed: "-6.211 Beneish M-Score" (number before name)
    ("Beneish_rev",   r"(-?\d+(?:\.\d+)?)\s*(?:on\s+(?:the\s+)?)?Beneish",                                     "beneish_m_score",   False),
    # Altman: handles "Altman Z-Score of 9.1", "Altman Z score: 9.1", "Altman score 9.1"
    ("AltmanZ",       r"Altman\s*(?:Z[-\s]?)?\s*[Ss]core[\s:of]*(-?\d+(?:\.\d+)?)",                            "altman_z_score",    False),
    # Altman reversed: "9.1 Altman Z-Score"
    ("AltmanZ_rev",   r"(-?\d+(?:\.\d+)?)\s*(?:on\s+(?:the\s+)?)?Altman",                                      "altman_z_score",    False),
    # ROE abbreviation: handles "ROE of 152%", "ROE: 135.5%", "ROE (trailing) 140%"
    ("ROE",           r"ROE(?:\s+of)?[\s\w\(\):]{0,20}?(\d+(?:\.\d+)?)\s*%",                                   "roe_pct",           True),
    # ROE full phrase: "return on equity of 136.2%"
    ("ROE_full",      r"return\s+on\s+equity(?:\s+of)?[\s\w\(\):]{0,20}?(\d+(?:\.\d+)?)\s*%",                  "roe_pct",           True),
    # ROIC abbreviation: handles "ROIC of 58%", "ROIC: 57.9%"
    ("ROIC",          r"ROIC(?:\s+of)?[\s\w\(\):]{0,20}?(\d+(?:\.\d+)?)\s*%",                                  "roic_pct",          True),
    # ROIC full phrase: "return on invested capital of 56.3%"
    ("ROIC_full",     r"return\s+on\s+invested\s+capital(?:\s+of)?[\s\w\(\):]{0,20}?(\d+(?:\.\d+)?)\s*%",      "roic_pct",          True),
    # P/E full phrase: "price-to-earnings ratio of 33.7x" or "price to earnings multiple of 33.7"
    ("P/E_full",      r"price[-\s]to[-\s]earnings(?:\s+(?:ratio|multiple))?[\s:of]*(-?\d+(?:\.\d+)?)x?",       "pe_trailing",       False),
    # EV/EBITDA full phrase: "enterprise value to EBITDA multiple of 22.8x"
    ("EV/EBITDA_full",r"enterprise\s+value\s+to\s+EBITDA(?:\s+(?:multiple|ratio))?[\s:of]*(-?\d+(?:\.\d+)?)x?","ev_ebitda",         False),
    # Beta: "beta of 1.29", "beta: 1.29", "beta coefficient of 1.29"
    ("Beta",          r"\bbeta(?:\s+(?:coefficient|of|:))?[\s:of]*(-?\d+(?:\.\d+)?)\b",                        "beta",              False),
    # Beta reversed: "1.29 beta"
    ("Beta_rev",      r"(-?\d+(?:\.\d+)?)\s+beta\b",                                                           "beta",              False),
    # Sharpe: "Sharpe ratio of 0.58", "Sharpe: 0.58", "Sharpe ratio: 0.58"
    ("Sharpe",        r"[Ss]harpe(?:\s+ratio)?(?:\s+of)?[\s:]*(-?\d+(?:\.\d+)?)\b",                            "sharpe_ratio",      False),
    # Revenue (quarterly): "quarterly revenue of $119.6B", "revenue reached $143.7B for the quarter",
    # "revenue ... $143.7B ... quarter" etc.
    ("Rev_Q",         r"(?:quarterly\s+revenue|revenue\s+(?:for|in)\s+the\s+(?:quarter|period))\b.{0,80}?\$(\d+(?:\.\d+)?)\s*(?:billion|B)\b", "revenue_latest_q_b", False),
    # Revenue (TTM) forward: "TTM/trailing twelve-month(s) ... revenue ... $X"
    ("Rev_TTM",       r"(?:TTM|trailing\s+twelve[-\s]months?(?:\s+basis)?).{0,120}?revenue\b.{0,80}?\$(\d+(?:\.\d+)?)\s*(?:billion|B)\b", "revenue_ttm_b", False),
    # Revenue (TTM) reversed: "revenue totaled/reached $X ... trailing/TTM"
    ("Rev_TTM_rev",   r"revenue\s+\w+\s+\$(\d+(?:\.\d+)?)\s*(?:billion|B).{0,120}?(?:trailing|TTM)",          "revenue_ttm_b",     False),
    # Net income (TTM) forward: "TTM net income $X"
    ("NI_TTM",        r"(?:TTM|trailing\s+twelve[-\s]months?(?:\s+basis)?).{0,120}?net\s+income\b.{0,80}?\$(\d+(?:\.\d+)?)\s*(?:billion|B)\b", "net_income_ttm_b", False),
    # Net income (TTM) reversed: "net income was $X ... trailing/TTM"
    ("NI_TTM_rev",    r"net\s+income\s+was\s+\$(\d+(?:\.\d+)?)\s*(?:billion|B).{0,120}?(?:trailing|TTM)",     "net_income_ttm_b",  False),
    # Net income (quarterly): "quarterly net income $X", "net income for the quarter reached $X",
    # "net income ... $X ... quarter", "net income ... quarter ... $X"
    ("NI_Q",          r"(?:quarterly\s+net\s+income|net\s+income\s+for\s+the\s+quarter)\b.{0,80}?\$(\d+(?:\.\d+)?)\s*(?:billion|B)\b", "net_income_latest_q_b", False),
    # NI_Q reversed: "net income for the quarter reached $X billion"
    # (exclude "net income was $X" to avoid matching TTM context — use NI_TTM_rev for that)
    ("NI_Q_rev",      r"net\s+income\s+for\s+the\s+quarter\s+(?:reached|grew|was|totaled)\s+\$(\d+(?:\.\d+)?)\s*(?:billion|B)",  "net_income_latest_q_b", False),
    # Operating cash flow: "operating cash flow of $110B", "operating CF of $118.3B"
    ("OCF",           r"operating\s+(?:cash\s+flow|CF)\b.{0,60}?\$(\d+(?:\.\d+)?)\s*(?:billion|B)\b",        "ocf_b",             False),
    # Free cash flow: "free cash flow of $99.5B", "FCF of $108.8B"
    ("FCF",           r"(?:free\s+cash\s+flow|FCF)\b.{0,60}?\$(\d+(?:\.\d+)?)\s*(?:billion|B)\b",             "fcf_b",             False),
    # YoY revenue growth — multiple forms:
    # "year-over-year increase of 11.3%", "11.3% YoY", "revenue growth of 11.3%", "revenue grew 11.3%"
    # "YoY revenue growth of 11.3%", "revenue ... year-over-year ... 11.3%"
    ("Rev_YoY",       r"(?:revenue\b.{0,60}?(?:year[-\s]over[-\s]year|YoY)\b.{0,60}?|(?:year[-\s]over[-\s]year|YoY)\s+revenue\s+\w+\s+(?:of\s+)?)(\d+(?:\.\d+)?)\s*%", "revenue_yoy_pct", False),
    # Revenue YoY reversed: "11.3% YoY [revenue]"
    ("Rev_YoY_rev",   r"(\d+(?:\.\d+)?)\s*%\s+(?:year[-\s]over[-\s]year|YoY)(?:\s+\w+){0,3}\s+revenue",                                                                "revenue_yoy_pct", False),
    # Revenue YoY full phrase: "year-over-year increase of 11.3%" in revenue context
    ("Rev_YoY_full",  r"revenue\b.{0,80}?year[-\s]over[-\s]year\s+(?:increase|growth|change)\s+of\s+(\d+(?:\.\d+)?)\s*%",                                              "revenue_yoy_pct", False),
    # YoY net income growth
    ("NI_YoY",        r"net\s+income\s+(?:growth|grew?|increased?)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*%",                                                                    "net_income_yoy_pct", False),
    # Total assets: "total assets of $352.6B", "total assets were $365.0B"
    ("TotalAssets",   r"total\s+assets\b.{0,60}?\$(\d+(?:\.\d+)?)\s*(?:billion|B)\b",                          "total_assets_b",    False),
    # Shareholders / total equity: "shareholders' equity of $74.2B", "stockholders equity $57.0B",
    # "total equity of $74.3B", "total shareholders' equity of $57.0B"
    ("Equity",        r"(?:(?:shareholders?'?\s+|stockholders?'?\s+|total\s+)equity)\b.{0,60}?\$(\d+(?:\.\d+)?)\s*(?:billion|B)\b", "shareholders_equity_b", False),
    # 12-month price return: "12-month return of 22.5%", "one-year return of 22.5%", "12-month price return 22.5%"
    ("Return12m",     r"(?:12[-\s]month|one[-\s]year|trailing\s+(?:twelve[-\s]month|12[-\s]month))\s+(?:price\s+)?return\b.{0,60}?(\d+(?:\.\d+)?)\s*%", "return_12m_pct", False),
    # Return12m reversed: "return of 22.5% over the past 12 months", "22.5% 12-month return"
    ("Return12m_rev", r"(\d+(?:\.\d+)?)\s*%\s+(?:12[-\s]month|one[-\s]year)\s+(?:price\s+)?return\b",                               "return_12m_pct", False),
    # RSI: "RSI(14) of 34.6", "RSI 34.6", "RSI: 34.6", "14-period RSI of 34.6"
    ("RSI",           r"RSI\s*\(?(?:14)?\)?\s*(?:of\s+|:\s*)?(\d+(?:\.\d+)?)\b",                                                    "rsi_14",         False),
    # RSI reversed: "34.6 RSI"
    ("RSI_rev",       r"(\d+(?:\.\d+)?)\s+RSI\b",                                                                                    "rsi_14",         False),
]

# Tolerance for near-match (0.3% — tight enough to catch LLM rounding like 58.2% vs 57.99%)
_TOLERANCE = 0.003


class FactChecker:
    """Post-generation fact-checker that finds metric values in report text
    and replaces any that deviate from the ground-truth QuantMetrics."""

    def correct_report(self, report: str, metrics: QuantMetrics) -> Tuple[str, List[str]]:
        """Correct numeric values in *report* to match *metrics*.

        Returns (corrected_report, list_of_corrections_made).
        """
        corrections: List[str] = []
        text = report

        for label, pattern, attr, is_percent in _METRIC_CHECKS:
            truth = getattr(metrics, attr, None)
            if truth is None:
                continue

            def _make_replacer(truth_val: float, lbl: str, is_pct: bool):
                def _replacer(m: re.Match) -> str:
                    found_str = m.group(1)
                    try:
                        found = float(found_str)
                    except ValueError:
                        return m.group(0)

                    ref = truth_val
                    denom = abs(ref) + 1e-9
                    if abs(found - ref) / denom < _TOLERANCE:
                        return m.group(0)  # within tolerance, leave unchanged

                    # Replace found value with truth
                    correct_str = (
                        f"{ref:.2f}" if is_pct else
                        f"{int(ref)}" if lbl == "Piotroski" else
                        f"{ref:.4f}" if abs(ref) < 10 else f"{ref:.2f}"
                    )
                    corrections.append(f"{lbl}: {found_str} → {correct_str}")
                    logger.info("[FactChecker] %s: replacing %s with %s", lbl, found_str, correct_str)
                    full = m.group(0)
                    return full[: m.start(1) - m.start(0)] + correct_str + full[m.end(1) - m.start(0):]

                return _replacer

            replacer = _make_replacer(float(truth), label, is_percent)
            text = re.sub(pattern, replacer, text, flags=re.IGNORECASE)

        return text, corrections

    def find_violations(self, report: str, metrics: QuantMetrics) -> List[Dict[str, Any]]:
        """Return a list of violations (metric values that differ from DB truth)."""
        violations = []
        for label, pattern, attr, is_percent in _METRIC_CHECKS:
            truth = getattr(metrics, attr, None)
            if truth is None:
                continue
            for m in re.finditer(pattern, report, re.IGNORECASE):
                try:
                    found = float(m.group(1))
                except ValueError:
                    continue
                ref = float(truth)
                denom = abs(ref) + 1e-9
                if abs(found - ref) / denom >= _TOLERANCE:
                     violations.append({
                        "metric": label,
                        "reported": found,
                        "expected": ref,
                        "deviation_pct": round(abs(found - ref) / denom * 100, 1),
                        "snippet": m.group(0)[:80],
                    })
        return violations


# ---------------------------------------------------------------------------
# Utility: flatten nested JSON to expose all leaf values
# ---------------------------------------------------------------------------

def flatten_json(obj: Any, parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """Recursively flatten a nested dict/list into a flat dict of leaf values.

    Example:
        flatten_json({"a": {"b": 1, "c": [2, 3]}})
        → {"a_b": 1, "a_c_0": 2, "a_c_1": 3}
    """
    items: List[Tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.extend(flatten_json(v, new_key, sep).items())
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{idx}" if parent_key else str(idx)
            items.extend(flatten_json(v, new_key, sep).items())
    else:
        return {parent_key: obj}
    return dict(items)
