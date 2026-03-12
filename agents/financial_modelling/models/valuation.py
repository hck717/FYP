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

import numpy as np

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


def winsorize_peer_multiples(multiples: List[float], limits: tuple = (0.05, 0.05)) -> List[float]:
    """Cap extreme peer multiples at 5th/95th percentile to reduce outlier impact.
    
    Args:
        multiples: List of peer multiples (e.g., EV/EBITDA values)
        limits: Tuple of (lower_percentile, upper_percentile) - defaults to 5th/95th
    
    Returns:
        List of winsorized multiples
    """
    if not multiples:
        return []
    
    clean = [m for m in multiples if m is not None and m > 0 and (m == m)]  # positive and not NaN
    if len(clean) < 5:
        logger.debug("Not enough peers (%d) for winsorization, returning original", len(clean))
        return multiples
    
    try:
        lower = np.percentile(clean, limits[0] * 100)
        upper = np.percentile(clean, (1 - limits[1]) * 100)
        
        winsorized = [max(lower, min(m, upper)) for m in clean]
        
        # Log if any values were adjusted
        adjusted_count = sum(1 for m in clean if m < lower or m > upper)
        if adjusted_count > 0:
            logger.info("Winsorized %d peer multiples: range [%.2f, %.2f] -> [%.2f, %.2f]",
                       adjusted_count, min(clean), max(clean), lower, upper)
        
        return winsorized
    except Exception as e:
        logger.warning("Winsorization failed: %s, returning original", e)
        return multiples
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

        result.ev_ebitda  = self._ev_ebitda(km_ttm, ratios_ttm, enterprise, income)
        result.ev_ebit    = self._ev_ebit(km_ttm, enterprise, income)
        result.pe_trailing = self._pe_trailing(km_ttm, ratios_ttm)
        result.pe_forward  = self._pe_forward(bundle)
        result.ps_ttm      = self._ps_ttm(km_ttm, ratios_ttm)
        result.ev_revenue  = self._ev_revenue(km_ttm, ratios_ttm, enterprise, income)
        result.p_fcf       = self._p_fcf(km_ttm, bundle)
        result.peg_ratio   = self._peg_ratio(result.pe_trailing, bundle)

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

        # ── Apply outlier detection (winsorization) to peer multiples ─────────────
        peer_ev_ebitdas_winsorized = winsorize_peer_multiples(peer_ev_ebitdas)
        peer_pe_trailing_winsorized = winsorize_peer_multiples(peer_pe_trailing)

        # ── vs_sector_avg: use EV/EBITDA if available, else P/E trailing ────
        target_multiple = result.ev_ebitda
        peer_median_evebitda = _median(peer_ev_ebitdas_winsorized)
        peer_median = peer_median_evebitda

        if target_multiple is None or peer_median is None:
            target_multiple = result.pe_trailing
            peer_median = _median(peer_pe_trailing_winsorized)

        if target_multiple is not None and peer_median is not None and peer_median > 0:
            pct = (target_multiple - peer_median) / peer_median * 100
            sign = "+" if pct >= 0 else ""
            label = "premium" if pct >= 0 else "discount"
            result.vs_sector_avg = f"{label} {sign}{pct:.0f}%"

        # ── Implied price from peer EV/EBITDA median ─────────────────────────
        # implied_ev_ebitda_value = peer_median_ev_ebitda × EBITDA / shares_outstanding
        if peer_median_evebitda is not None and peer_median_evebitda > 0:
            income_annual = bundle.income_annual or {}
            ebitda = _safe_float(
                income.get("ebitda")
                or income.get("EBITDA")
                or income_annual.get("ebitda")
                or income_annual.get("EBITDA")
                or km_ttm.get("ebitdaTTM")
                or km_ttm.get("EBITDA")
            )
            shares = _safe_float(
                km_ttm.get("weightedAverageSharesOutstanding")
                or km_ttm.get("weightedAverageSharesDiluted")
                or km_ttm.get("sharesOutstanding")
                or km_ttm.get("SharesOutstanding")
            )
            if ebitda and ebitda > 0 and shares and shares > 0:
                net_debt = _safe_float(
                    enterprise.get("netDebt")
                    or enterprise.get("NetDebt")
                ) or 0.0
                implied_ev = peer_median_evebitda * ebitda
                implied_equity = implied_ev - net_debt
                result.implied_ev_ebitda_value = round(implied_equity / shares, 2)

        return result

    # ── Individual multiple helpers ──────────────────────────────────────────

    # Industry thresholds for EV/EBITDA validation
    EV_EBITDA_THRESHOLDS = {
        "technology": 40,
        "software": 35,
        "semiconductors": 30,
        "utilities": 15,
        "financials": 12,
        "healthcare": 20,
        "consumer_discretionary": 25,
        "consumer_staples": 18,
        "energy": 10,
        "industrials": 20,
        "materials": 12,
        "real_estate": 20,
        "telecommunications": 10,
    }

    def _ev_ebitda(
        self,
        km_ttm: Dict[str, Any],
        ratios_ttm: Dict[str, Any],
        enterprise: Dict[str, Any],
        income: Dict[str, Any],
    ) -> Optional[float]:
        # Try precomputed TTM key first (FMP field names)
        v = _safe_float(km_ttm.get("evToEBITDATTM") or km_ttm.get("enterpriseValueMultipleTTM"))
        if v is not None and v > 0:
            if v > 50:
                logger.warning(f"EV/EBITDA {v}x unusually high (>50), validating...")
                v = self._validate_ev_ebitda(v, km_ttm, enterprise, income)
            return round(v, 2) if v else None
        # EODHD field names (enterprise_values payload stored in bundle.enterprise)
        v = _safe_float(enterprise.get("EnterpriseValueEbitda") or km_ttm.get("EnterpriseValueEbitda"))
        if v is not None and v > 0:
            if v > 50:
                logger.warning(f"EV/EBITDA {v}x unusually high (>50), validating...")
                v = self._validate_ev_ebitda(v, km_ttm, enterprise, income)
            return round(v, 2) if v else None
        # Compute from components
        ev = _safe_float(
            enterprise.get("enterpriseValue")
            or enterprise.get("EnterpriseValue")
            or enterprise.get("enterpriseValueTTM")
        )
        ebitda = _safe_float(
            income.get("ebitda")
            or income.get("EBITDA")
            or km_ttm.get("ebitdaTTM")
            or km_ttm.get("EBITDA")   # EODHD key_metrics_ttm
        )
        if ev and ebitda and ebitda > 0:
            ev_ebitda = round(ev / ebitda, 2)
            if ev_ebitda > 50:
                logger.warning(f"Computed EV/EBITDA {ev_ebitda}x unusually high (>50), validating...")
                validated = self._validate_ev_ebitda(ev_ebitda, km_ttm, enterprise, income)
                return round(validated, 2) if validated else None
            return ev_ebitda
        return None

    def _validate_ev_ebitda(
        self,
        current_value: float,
        km_ttm: Dict[str, Any],
        enterprise: Dict[str, Any],
        income: Dict[str, Any],
    ) -> Optional[float]:
        """Validate and correct implausible EV/EBITDA values."""
        # Detect sector from key_metrics
        sector = None
        for key in ("sector", "Sector", "GicSector", "gicSector", "industry"):
            v = km_ttm.get(key)
            if v and isinstance(v, str):
                sector = v.lower()
                break
        
        # Get threshold based on sector
        threshold = 40  # default for mature tech
        if sector:
            for sector_key, sector_threshold in self.EV_EBITDA_THRESHOLDS.items():
                if sector_key in sector:
                    threshold = sector_threshold
                    break
        
        if current_value <= threshold:
            return current_value
        
        logger.warning(f"EV/EBITDA {current_value}x exceeds {threshold}x threshold for sector '{sector}', attempting manual calculation...")
        
        # Fallback: compute manually from raw components
        ev = _safe_float(
            enterprise.get("enterpriseValue")
            or enterprise.get("EnterpriseValue")
            or enterprise.get("marketCapitalization")
        )
        ebitda = _safe_float(
            income.get("ebitda")
            or income.get("EBITDA")
            or km_ttm.get("ebitdaTTM")
            or km_ttm.get("EBITDA")
        )
        
        if ev and ebitda and ebitda > 0:
            manual_ev_ebitda = round(ev / ebitda, 2)
            if 5 < manual_ev_ebitda < threshold:
                logger.info(f"Manual EV/EBITDA calculation yielded {manual_ev_ebitda}x (was {current_value}x)")
                return manual_ev_ebitda
        
        # If manual calculation also yields high value, return None to trigger peer median fallback
        logger.warning(f"Could not validate EV/EBITDA, returning None to use peer median")
        return None

    def _pe_trailing(
        self,
        km_ttm: Dict[str, Any],
        ratios_ttm: Dict[str, Any],
    ) -> Optional[float]:
        v = _safe_float(
            # FMP field names
            ratios_ttm.get("priceToEarningsRatioTTM")
            or km_ttm.get("peRatioTTM")
            or km_ttm.get("priceEarningsRatioTTM")
            # EODHD field names
            or km_ttm.get("PERatio")
            or ratios_ttm.get("PERatio")
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
            # FMP field names
            ratios_ttm.get("priceToSalesRatioTTM")
            or km_ttm.get("priceToSalesRatioTTM")
            or km_ttm.get("psTTM")
            # EODHD field names
            or km_ttm.get("PriceSalesTTM")
            or ratios_ttm.get("PriceSalesTTM")
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
        # Try computing from EV / revenue (most reliable)
        ev = _safe_float(
            enterprise.get("enterpriseValue")
            or enterprise.get("EnterpriseValue")
            or enterprise.get("enterpriseValueTTM")
        )
        revenue = _safe_float(income.get("revenue"))
        if ev and revenue and revenue > 0:
            return round(ev / revenue, 2)
        # EODHD pre-computed EV/Revenue
        v = _safe_float(
            enterprise.get("EnterpriseValueRevenue")
            or km_ttm.get("EnterpriseValueRevenue")
            or km_ttm.get("evToSalesTTM")
        )
        if v is not None and v > 0:
            return round(v, 2)
        return None

    def _ev_ebit(
        self,
        km_ttm: Dict[str, Any],
        enterprise: Dict[str, Any],
        income: Dict[str, Any],
    ) -> Optional[float]:
        """EV / EBIT — pure operating multiple (excludes D&A from denominator vs EV/EBITDA)."""
        ev = _safe_float(
            enterprise.get("enterpriseValue")
            or enterprise.get("EnterpriseValue")
            or enterprise.get("enterpriseValueTTM")
        )
        ebit = _safe_float(
            income.get("ebit")
            or income.get("operatingIncome")
            or km_ttm.get("ebitTTM")
        )
        if ev and ebit and ebit > 0:
            return round(ev / ebit, 2)
        return None

    def _p_fcf(
        self,
        km_ttm: Dict[str, Any],
        bundle: FMDataBundle,
    ) -> Optional[float]:
        """P/FCF = Market Cap / Free Cash Flow to Firm."""
        # Try pre-computed TTM ratio
        v = _safe_float(km_ttm.get("priceToFreeCashFlowsRatioTTM") or km_ttm.get("pfcfRatioTTM"))
        if v is not None and v > 0:
            return round(v, 2)
        # Compute from components
        market_cap = _safe_float(
            bundle.enterprise.get("marketCapitalization")
            or km_ttm.get("marketCap")
            or km_ttm.get("marketCapTTM")
        )
        fcff = _safe_float(
            bundle.cashflow.get("freeCashFlowToFirm")
            or km_ttm.get("freeCashFlowToFirmTTM")
            or km_ttm.get("freeCashFlowPerShareTTM")
        )
        if market_cap and fcff and fcff > 0:
            return round(market_cap / fcff, 2)
        return None

    def _peg_ratio(
        self,
        pe_trailing: Optional[float],
        bundle: FMDataBundle,
    ) -> Optional[float]:
        """PEG Ratio = P/E ÷ (expected EPS growth rate × 100).

        Uses analyst EPS estimates to derive forward EPS growth rate.
        PEG < 1 = potentially undervalued relative to growth; > 2 = expensive relative to growth.
        """
        if pe_trailing is None or pe_trailing <= 0:
            return None

        estimates = bundle.analyst_estimates
        if not estimates or len(estimates) < 1:
            return None

        # Derive EPS growth rate from analyst estimate vs. trailing EPS
        est = estimates[0] if isinstance(estimates[0], dict) else {}
        fwd_eps = _safe_float(
            est.get("estimatedEpsAvg")
            or est.get("estimatedEPS")
            or est.get("epsEstimated")
        )
        # Trailing EPS: derive from P/E and current price if available
        current_price = _safe_float(
            bundle.key_metrics_ttm.get("stockPriceTTM")
            or bundle.ratios_ttm.get("stockPriceTTM")
        )
        if (current_price is None or current_price <= 0) and bundle.price_history:
            row = bundle.price_history[0]
            current_price = _safe_float(
                row.get("adjusted_close") or row.get("adjClose") or row.get("close")
            )

        trailing_eps = (current_price / pe_trailing) if current_price and pe_trailing > 0 else None

        if fwd_eps and trailing_eps and trailing_eps > 0:
            eps_growth_rate = (fwd_eps - trailing_eps) / trailing_eps  # as decimal
            if eps_growth_rate > 0:
                # PEG = P/E ÷ (EPS growth % per year)
                peg = pe_trailing / (eps_growth_rate * 100)
                return round(peg, 2)

        return None


__all__ = ["CompsEngine", "winsorize_peer_multiples"]
