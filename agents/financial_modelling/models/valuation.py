"""Comparable Company Analysis (Comps) engine.

Computes EV/EBITDA, P/E (trailing + forward), P/S, EV/Revenue for the target
ticker and compares against peer median multiples sourced from PostgreSQL.

Peer group is resolved by the FMDataFetcher (Neo4j → PG fallback).
All arithmetic is deterministic Python — no LLM involvement.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional

from ..config import FinancialModellingConfig
from ..schema import CompsResult, FMDataBundle

logger = logging.getLogger(__name__)


def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _median(values: List[float]) -> Optional[float]:
    clean = [v for v in values if v is not None and not (v != v)]  # drop NaN
    if not clean:
        return None
    return statistics.median(clean)


class CompsEngine:
    """Computes peer comparable multiples and sector premium/discount."""

    def __init__(self, config: FinancialModellingConfig) -> None:
        self.config = config

    def compute(self, bundle: FMDataBundle) -> CompsResult:
        """Compute Comps multiples for bundle.ticker vs. peer_fundamentals."""
        result = CompsResult()
        result.peer_group = list(bundle.peer_fundamentals.keys())

        # ── Target multiples ─────────────────────────────────────────────────
        km_ttm = bundle.key_metrics_ttm
        ratios_ttm = bundle.ratios_ttm
        enterprise = bundle.enterprise
        income = bundle.income

        result.ev_ebitda = self._ev_ebitda(km_ttm, ratios_ttm, enterprise, income)
        result.pe_trailing = self._pe_trailing(km_ttm, ratios_ttm)
        result.pe_forward = self._pe_forward(bundle)
        result.ps_ttm = self._ps_ttm(km_ttm, ratios_ttm)
        result.ev_revenue = self._ev_revenue(km_ttm, ratios_ttm, enterprise, income)

        # ── Peer median multiples ────────────────────────────────────────────
        peer_ev_ebitdas: List[float] = []
        peer_pe_trailing: List[float] = []

        for peer_ticker, peer_data in bundle.peer_fundamentals.items():
            peer_km_ttm = peer_data.get("key_metrics_ttm") or {}
            peer_ratios_ttm = peer_data.get("ratios_ttm") or {}
            peer_ent = peer_data.get("enterprise") or {}
            peer_inc = peer_data.get("income") or {}

            ev_eb = self._ev_ebitda(peer_km_ttm, peer_ratios_ttm, peer_ent, peer_inc)
            if ev_eb is not None and ev_eb > 0:
                peer_ev_ebitdas.append(ev_eb)

            pe_t = self._pe_trailing(peer_km_ttm, peer_ratios_ttm)
            if pe_t is not None and pe_t > 0:
                peer_pe_trailing.append(pe_t)

        # ── vs_sector_avg: use EV/EBITDA if available, else P/E trailing ────
        target_multiple = result.ev_ebitda
        peer_median = _median(peer_ev_ebitdas)

        if target_multiple is None or peer_median is None:
            target_multiple = result.pe_trailing
            peer_median = _median(peer_pe_trailing)

        if target_multiple is not None and peer_median is not None and peer_median > 0:
            pct = (target_multiple - peer_median) / peer_median * 100
            sign = "+" if pct >= 0 else ""
            label = "premium" if pct >= 0 else "discount"
            result.vs_sector_avg = f"{label} {sign}{pct:.0f}%"

        return result

    # ── Individual multiple helpers ──────────────────────────────────────────

    def _ev_ebitda(
        self,
        km_ttm: Dict[str, Any],
        ratios_ttm: Dict[str, Any],
        enterprise: Dict[str, Any],
        income: Dict[str, Any],
    ) -> Optional[float]:
        # Try precomputed TTM key first
        v = _safe_float(km_ttm.get("evToEBITDATTM") or km_ttm.get("enterpriseValueMultipleTTM"))
        if v is not None and v > 0:
            return round(v, 2)
        # Compute from components
        ev = _safe_float(enterprise.get("enterpriseValue") or enterprise.get("enterpriseValueTTM"))
        ebitda = _safe_float(
            income.get("ebitda")
            or income.get("EBITDA")
            or km_ttm.get("ebitdaTTM")
        )
        if ev and ebitda and ebitda > 0:
            return round(ev / ebitda, 2)
        return None

    def _pe_trailing(
        self,
        km_ttm: Dict[str, Any],
        ratios_ttm: Dict[str, Any],
    ) -> Optional[float]:
        v = _safe_float(
            ratios_ttm.get("priceToEarningsRatioTTM")
            or km_ttm.get("peRatioTTM")
            or km_ttm.get("priceEarningsRatioTTM")
        )
        if v is not None and v > 0:
            return round(v, 2)
        return None

    def _pe_forward(self, bundle: FMDataBundle) -> Optional[float]:
        """Derive forward P/E from analyst_estimates consensus EPS and current price."""
        estimates = bundle.analyst_estimates
        if not estimates:
            return None
        # Take the most recent analyst estimate
        est = estimates[0] if isinstance(estimates[0], dict) else {}
        fwd_eps = _safe_float(
            est.get("estimatedEpsAvg")
            or est.get("estimatedEPS")
            or est.get("epsEstimated")
        )
        current_price = _safe_float(
            bundle.key_metrics_ttm.get("stockPriceTTM")
            or bundle.ratios_ttm.get("stockPriceTTM")
        )
        if fwd_eps and fwd_eps > 0 and current_price and current_price > 0:
            return round(current_price / fwd_eps, 2)
        return None

    def _ps_ttm(
        self,
        km_ttm: Dict[str, Any],
        ratios_ttm: Dict[str, Any],
    ) -> Optional[float]:
        v = _safe_float(
            ratios_ttm.get("priceToSalesRatioTTM")
            or km_ttm.get("priceToSalesRatioTTM")
            or km_ttm.get("psTTM")
        )
        if v is not None and v > 0:
            return round(v, 2)
        return None

    def _ev_revenue(
        self,
        km_ttm: Dict[str, Any],
        ratios_ttm: Dict[str, Any],
        enterprise: Dict[str, Any],
        income: Dict[str, Any],
    ) -> Optional[float]:
        v = _safe_float(
            km_ttm.get("evToSalesTTM")
            or km_ttm.get("enterpriseValueOverEBITDATTM")
        )
        if v is not None and v > 0 and "revenue" not in str(v):
            # evToSales ≠ evToEBITDA; verify via raw components
            pass
        # Try computing from EV / revenue
        ev = _safe_float(enterprise.get("enterpriseValue") or enterprise.get("enterpriseValueTTM"))
        revenue = _safe_float(income.get("revenue"))
        if ev and revenue and revenue > 0:
            return round(ev / revenue, 2)
        # Fallback to TTM key
        v2 = _safe_float(km_ttm.get("evToSalesTTM"))
        if v2 is not None and v2 > 0:
            return round(v2, 2)
        return None


__all__ = ["CompsEngine"]
