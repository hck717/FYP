"""Value factor calculations: P/E, EV/EBITDA, P/FCF, EV/Revenue.

Data sources (Fundamental Math Agent scope):
  - bundle.valuation_metrics: trailing_pe, forward_pe, ev_ebitda, ev_revenue, peg_ratio,
                               market_cap, free_cash_flow
  - bundle.key_metrics_ttm:   PERatio, PEGRatio (fallback)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..schema import FinancialsBundle, ValueFactors
from ..tools import _safe_float

logger = logging.getLogger(__name__)


def compute_value_factors(bundle: FinancialsBundle) -> ValueFactors:
    """Compute value factors from valuation_metrics and key_metrics_ttm.

    Uses the dedicated valuation_metrics table (trailing_pe, ev_ebitda,
    ev_revenue, market_cap, free_cash_flow) which is populated by the
    EODHD ingestion pipeline.
    """
    vm = bundle.valuation_metrics or {}
    km = bundle.key_metrics_ttm or {}

    # ------------------------------------------------------------------
    # P/E trailing — valuation_metrics.trailing_pe preferred
    # ------------------------------------------------------------------
    pe = (
        _safe_float(vm.get("trailing_pe"))
        or _safe_float(vm.get("pe_ratio"))
        or _safe_float(km.get("PERatio"))
    )
    if pe is not None and pe <= 0:
        pe = None

    # ------------------------------------------------------------------
    # EV/EBITDA — valuation_metrics.ev_ebitda
    # ------------------------------------------------------------------
    ev_ebitda = _safe_float(vm.get("ev_ebitda"))
    if ev_ebitda is not None and ev_ebitda <= 0:
        ev_ebitda = None

    # ------------------------------------------------------------------
    # P/FCF — computed from market_cap / free_cash_flow
    # Tries: valuation_metrics, then cashflow statement (direct or OCF - CapEx)
    # ------------------------------------------------------------------
    p_fcf: Optional[float] = None
    mc = _safe_float(vm.get("market_cap"))
    if mc and mc > 0:
        fcf = _safe_float(vm.get("free_cash_flow"))
        if fcf is None or fcf <= 0:
            # Fall back to cashflow statement on the bundle
            cf = getattr(bundle, "cashflow", None) or {}
            fcf = _safe_float(cf.get("freeCashFlow"))
            if fcf is None or fcf <= 0:
                ocf = _safe_float(cf.get("totalCashFromOperatingActivities")
                                  or cf.get("operatingCashFlow"))
                capex = _safe_float(cf.get("capitalExpenditures"))
                if ocf and capex is not None:
                    # capitalExpenditures in EODHD is stored as a positive number
                    fcf = ocf - abs(capex)
        if fcf and fcf > 0:
            p_fcf = round(mc / fcf, 2)

    # ------------------------------------------------------------------
    # EV/Revenue — valuation_metrics.ev_revenue
    # ------------------------------------------------------------------
    ev_revenue = (
        _safe_float(vm.get("ev_revenue"))
        or _safe_float(vm.get("price_sales_ttm"))  # P/S as rough proxy
    )
    if ev_revenue is not None and ev_revenue <= 0:
        ev_revenue = None

    logger.info(
        "[ValueFactors] pe=%.2f ev_ebitda=%.2f p_fcf=%.2f ev_revenue=%.2f",
        pe or 0, ev_ebitda or 0, p_fcf or 0, ev_revenue or 0,
    )

    return ValueFactors(
        pe_trailing=round(pe, 4) if pe is not None else None,
        ev_ebitda=round(ev_ebitda, 4) if ev_ebitda is not None else None,
        p_fcf=round(p_fcf, 2) if p_fcf is not None else None,
        ev_revenue=round(ev_revenue, 4) if ev_revenue is not None else None,
    )


__all__ = ["compute_value_factors"]
