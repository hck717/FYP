"""Value factor calculations: P/E, EV/EBITDA, P/FCF, EV/Revenue.

Data sources (Fundamental Math Agent scope):
  - bundle.valuation_metrics: trailing_pe, forward_pe, ev_ebitda, ev_revenue, peg_ratio
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
    ev_revenue) which is populated by the EODHD ingestion pipeline.
    P/FCF is not available from allowed data sources and remains null.
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
    # P/FCF — not available from allowed data; stays null
    # ------------------------------------------------------------------
    p_fcf: Optional[float] = None

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
        "[ValueFactors] pe=%.2f ev_ebitda=%.2f ev_revenue=%.2f",
        pe or 0, ev_ebitda or 0, ev_revenue or 0,
    )

    return ValueFactors(
        pe_trailing=round(pe, 4) if pe is not None else None,
        ev_ebitda=round(ev_ebitda, 4) if ev_ebitda is not None else None,
        p_fcf=p_fcf,
        ev_revenue=round(ev_revenue, 4) if ev_revenue is not None else None,
    )


__all__ = ["compute_value_factors"]
