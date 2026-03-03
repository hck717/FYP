"""Value factor calculations: P/E, EV/EBITDA, P/FCF, EV/Revenue."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..schema import FinancialsBundle, ValueFactors
from ..tools import _safe_div, _safe_float

logger = logging.getLogger(__name__)


def compute_value_factors(bundle: FinancialsBundle) -> ValueFactors:
    """Compute value factors from the FinancialsBundle.

    Data source priority for each metric:
    1. key_metrics_ttm / ratios_ttm   — pre-computed, most current
    2. enterprise_values              — for EV-based metrics
    3. Derived from income + balance  — fallback
    """
    inc = bundle.income
    bal = bundle.balance
    cf = bundle.cashflow
    ent = bundle.enterprise
    km_ttm = bundle.key_metrics_ttm
    rt = bundle.ratios_ttm
    km = bundle.key_metrics

    # ------------------------------------------------------------------
    # P/E trailing — prefer pre-computed TTM ratio
    # ------------------------------------------------------------------
    pe: Optional[float] = None
    # Try ratios_ttm first (FMP key: priceToEarningsRatioTTM)
    pe_raw = (
        rt.get("priceToEarningsRatioTTM")
        or rt.get("peRatioTTM")
        or rt.get("priceEarningsRatioTTM")
        or km_ttm.get("peRatioTTM")
        or km.get("peRatio")
    )
    pe = _safe_float(pe_raw)
    if pe and pe <= 0:
        pe = None  # Negative P/E is not meaningful for value factor

    # ------------------------------------------------------------------
    # EV/EBITDA — prefer enterprise_values payload
    # ------------------------------------------------------------------
    ev_ebitda: Optional[float] = None
    ev_raw = (
        ent.get("enterpriseValue")
        or ent.get("enterpriseValueTTM")
        or km_ttm.get("enterpriseValueOverEBITDATTM")
    )
    # If km_ttm provides the ratio directly, use it
    ev_ebitda_direct = (
        km_ttm.get("evToEBITDATTM")
        or km_ttm.get("evToEbitdaTTM")
        or km_ttm.get("enterpriseValueOverEBITDATTM")
        or km.get("evToEbitda")
        or rt.get("enterpriseValueMultipleTTM")
    )
    if ev_ebitda_direct:
        ev_ebitda = _safe_float(ev_ebitda_direct)
        if ev_ebitda and ev_ebitda <= 0:
            ev_ebitda = None
    else:
        # Derive: EV / EBITDA
        ev = _safe_float(ev_raw)
        ebitda = _safe_float(
            inc.get("ebitda")
            or inc.get("ebitdaRatio")  # ratio won't work — try other keys
        )
        if ebitda and inc.get("ebitdaRatio"):
            # ebitdaRatio is a margin, not absolute — ignore
            ebitda = None
        # Try absolute EBITDA from key_metrics
        if ebitda is None:
            ebitda = _safe_float(km_ttm.get("netIncomePerShareTTM"))  # not useful
            ebitda = None  # reset — derive from income
            operating_income = _safe_float(inc.get("operatingIncome") or inc.get("ebit"))
            da = _safe_float(cf.get("depreciationAndAmortization"))
            if operating_income is not None and da is not None:
                ebitda = operating_income + da
        ev_ebitda = _safe_div(ev, ebitda)
        if ev_ebitda and ev_ebitda <= 0:
            ev_ebitda = None

    # ------------------------------------------------------------------
    # P/FCF — price to free cash flow
    # ------------------------------------------------------------------
    p_fcf: Optional[float] = None
    p_fcf_raw = (
        km_ttm.get("evToFreeCashFlowTTM")
        or km_ttm.get("pfcfRatioTTM")
        or km_ttm.get("priceToFreeCashFlowsRatioTTM")
        or km.get("pfcfRatio")
        or km.get("priceToFreeCashFlowsRatio")
        or rt.get("priceToFreeCashFlowRatioTTM")
    )
    p_fcf = _safe_float(p_fcf_raw)
    if p_fcf and p_fcf <= 0:
        p_fcf = None

    # ------------------------------------------------------------------
    # EV/Revenue
    # ------------------------------------------------------------------
    ev_revenue: Optional[float] = None
    ev_rev_raw = (
        km_ttm.get("evToSalesTTM")
        or km_ttm.get("priceToSalesRatioTTM")  # fallback: P/S as approximation
        or km.get("evToSales")
    )
    ev_revenue = _safe_float(ev_rev_raw)
    if ev_revenue and ev_revenue <= 0:
        ev_revenue = None

    return ValueFactors(
        pe_trailing=round(pe, 2) if pe else None,
        ev_ebitda=round(ev_ebitda, 2) if ev_ebitda else None,
        p_fcf=round(p_fcf, 2) if p_fcf else None,
        ev_revenue=round(ev_revenue, 2) if ev_revenue else None,
    )


__all__ = ["compute_value_factors"]
