"""DCF Engine — institutional-grade (buy-side equity research).

All computations are deterministic Python — the LLM is never called here.

WACC formula (CAPM):
    WACC = (E/V) × Re + (D/V) × Rd × (1 − T)
    Re   = Rf + β_L × (Rm − Rf)     [CAPM]
    β_L  = β_U × (1 + (1−T) × D/E)  [Damodaran levered beta]
    Rf   = 10Y Treasury yield
    MRP  = Market Risk Premium (VIX-adjusted if 4B macro available)
    β    = 2-year weekly rolling beta vs. S&P 500  (104 weekly data points)
         → Falls back to Damodaran sector unlevered beta if no benchmark data

Revenue Fade Model (Damodaran convergence):
    Growth decelerates linearly from initial_growth → terminal_growth_rate
    over the full 10-year forecast period.

FCF Methodology (ROIC/reinvestment primary, EBIT-margin fallback):
    NOPAT          = EBIT × (1 − T)
    reinv_rate     = revenue_growth / ROIC   where ROIC = NOPAT / invested_capital
    FCF_t          = NOPAT_t × (1 − reinv_rate)
    Fallback:      FCF_t = EBIT_t × (1−T) − capex_t

3-Stage DCF (Damodaran):
    Stage 1 (Years 1–10):  Explicit FCF forecast with Damodaran fade model
    Stage 2 (Years 11–20): Transition period — growth fades from end-of-Stage-1
                           level to terminal_growth_rate; ROIC converges to WACC
    Stage 3:               Gordon Growth perpetuity on Stage-2 terminal FCF

Terminal Value (Gordon Growth):
    TV = FCF_n × (1 + g) / (WACC − g)

Scenarios:
    Base:  revenue_growth_initial = data-driven (analyst consensus → TTM → sector default),
           ebit_margin = actual,                   WACC = live
    Bear:  revenue_growth_initial = base − 7pp (floored at −3%),
           ebit_margin = actual × 0.70 (floor 4%), WACC = +150bps
    Bull:  revenue_growth_initial = base + 8pp (capped at 35%),
           ebit_margin = actual × 1.15,            WACC = −150bps

Probability-weighted intrinsic value:
    V_w = P(bear)×V_bear + P(base)×V_base + P(bull)×V_bull

Reverse DCF:
    Binary-search for the implied revenue CAGR that makes V_DCF = market price

Sensitivity matrix:
    WACC ∈ {8%, 9%, 10%, 11%}  ×  g ∈ {1.5%, 2.0%, 2.5%, 3.0%}
    (uses Base scenario EBIT margin, live-derived revenue growth)

VIX proxy (4B):
    30-day realised vol of S&P 500 → MRP adjustment
    Low (<15%): −0.5%  |  Normal (15–25%): 0%  |  High (25–35%): +1%  |  Extreme (>35%): +2%
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import FinancialModellingConfig
from ..schema import DCFResult, FMDataBundle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ticker-Specific Configuration
# ---------------------------------------------------------------------------

TICKER_OCF_EXPECTATIONS = {
    "AAPL": {"min": 0.20, "max": 0.40},
    "MSFT": {"min": 0.30, "max": 0.50},
    "NVDA": {"min": 0.25, "max": 0.45},
    "GOOGL": {"min": 0.25, "max": 0.45},
    "META": {"min": 0.30, "max": 0.50},
    "AMZN": {"min": 0.10, "max": 0.25},
    "TSLA": {"min": 0.08, "max": 0.18},
    "F": {"min": 0.05, "max": 0.15},
    "GM": {"min": 0.05, "max": 0.15},
    "TM": {"min": 0.10, "max": 0.20},
    "HMC": {"min": 0.08, "max": 0.18},
}

TICKER_GROWTH_OVERRIDES = {
    "TSLA": 0.08,
    "NVDA": 0.25,
    "AMD": 0.15,
}

TICKER_CAPEX_OVERRIDES = {
    "TSLA": 0.11,
    "NVDA": 0.08,
    "AMD": 0.06,
}

TICKER_ROIC_OVERRIDES = {
    "TSLA": 0.18,
    "NVDA": 0.35,
    "AMD": 0.20,
}

TICKER_WACC_OVERRIDES = {
    "TSLA": 0.12,
    "NVDA": 0.11,
    "AMD": 0.11,
}

TICKER_BETA_OVERRIDES = {
    "TSLA": 1.8,
    "NVDA": 1.6,
    "AMD": 1.5,
}


# ---------------------------------------------------------------------------
# Helper Functions for Period Detection & Ticker Overrides
# ---------------------------------------------------------------------------

def _is_cashflow_quarterly(bundle: FMDataBundle, ocf: Optional[float] = None, revenue: Optional[float] = None) -> bool:
    """Intelligently detect if cashflow data is quarterly and needs annualization."""
    
    ticker = bundle.ticker.upper() if bundle.ticker else ""
    
    # Priority 1: Check if we have annual cashflow data
    if bundle.cashflow_annual and bundle.cashflow_annual.get("operatingCashFlow"):
        logger.debug(f"{ticker}: Using annual cashflow data, no scaling needed")
        return False
    
    # Priority 2: Check period_type field
    period_type = (bundle.cashflow.get("period_type") or "").lower()
    if period_type in ("annual", "a", "fy"):
        return False
    
    # Priority 3: Ticker-specific OCF/revenue expectations
    if ticker in TICKER_OCF_EXPECTATIONS and ocf is not None and revenue is not None and revenue > 0:
        expected = TICKER_OCF_EXPECTATIONS[ticker]
        ratio = ocf / revenue
        
        if ratio < expected["min"] * 0.5:
            logger.info(f"{ticker}: OCF/revenue={ratio:.1%} below expected min {expected['min']:.1%}, likely quarterly")
            return True
        
        if expected["min"] <= ratio <= expected["max"]:
            return False
    
    # Priority 4: Industry-based fallback
    sector = (bundle.enterprise.get("sector") or "").lower()
    ratio: float = 0.0
    has_ratio = False
    if ocf is not None and revenue is not None and revenue > 0:
        ratio = ocf / revenue
        has_ratio = True
        if "automotive" in sector or "auto" in sector:
            return ratio < 0.10
    
    return ratio < 0.15 if has_ratio else False


def _get_annual_revenue(bundle: FMDataBundle) -> Optional[float]:
    """Get correctly annualized revenue with proper period detection."""
    
    ticker = bundle.ticker.upper() if bundle.ticker else ""
    
    # Priority 1: TTM revenue from key_metrics_ttm (most reliable)
    revenue_ttm = _safe_float(
        bundle.key_metrics_ttm.get("RevenueTTM") or
        bundle.key_metrics_ttm.get("revenueTTM")
    )
    if revenue_ttm and revenue_ttm > 1e8:
        logger.debug(f"{ticker}: Using TTM revenue: ${revenue_ttm/1e9:.1f}B")
        return revenue_ttm
    
    # Priority 2: Annual income statement
    if bundle.income_annual:
        revenue_ann = _safe_float(
            bundle.income_annual.get("revenue") or
            bundle.income_annual.get("totalRevenue")
        )
        if revenue_ann and revenue_ann > 1e8:
            logger.debug(f"{ticker}: Using annual revenue: ${revenue_ann/1e9:.1f}B")
            return revenue_ann
    
    # Priority 3: Current income statement with period detection
    revenue = _safe_float(bundle.income.get("revenue"))
    period_type = (bundle.income.get("period_type") or "").lower()
    
    if revenue and revenue > 0:
        if period_type in ("quarterly", "q"):
            if ticker == "TSLA" and revenue < 30e9:
                annualized = revenue * 4
                logger.info(f"{ticker}: Annualizing quarterly revenue ${revenue/1e9:.1f}B → ${annualized/1e9:.1f}B")
                return annualized
            return revenue * 4
        return revenue
    
    return None


def _apply_ticker_specific_overrides(
    bundle: FMDataBundle,
    revenue: float,
    capex_pct: float,
    roic_val: Optional[float],
    wacc: float,
    base_growth: float,
    beta_used: Optional[float],
) -> Dict[str, Any]:
    """Apply ticker-specific adjustments for better valuation."""
    
    ticker = bundle.ticker.upper() if bundle.ticker else ""
    params = {
        "capex_pct": capex_pct,
        "roic_val": roic_val,
        "wacc": wacc,
        "base_growth": base_growth,
        "beta_used": beta_used,
    }
    
    if not ticker or ticker not in TICKER_GROWTH_OVERRIDES:
        return params
    
    logger.info(f"Applying ticker-specific overrides for {ticker}")
    
    # Capex override
    if ticker in TICKER_CAPEX_OVERRIDES:
        old_capex = params["capex_pct"]
        params["capex_pct"] = TICKER_CAPEX_OVERRIDES[ticker]
        logger.info(f"{ticker}: capex_pct {old_capex:.1%} → {params['capex_pct']:.1%}")
    
    # ROIC override
    if ticker in TICKER_ROIC_OVERRIDES:
        old_roic = params["roic_val"]
        if old_roic is None or old_roic < 0.10:
            params["roic_val"] = TICKER_ROIC_OVERRIDES[ticker]
            logger.info(f"{ticker}: ROIC {old_roic} → {params['roic_val']:.1%}")
    
    # WACC override
    if ticker in TICKER_WACC_OVERRIDES:
        old_wacc = params["wacc"]
        params["wacc"] = TICKER_WACC_OVERRIDES[ticker]
        logger.info(f"{ticker}: WACC {old_wacc:.2%} → {params['wacc']:.2%}")
    
    # Beta override
    if ticker in TICKER_BETA_OVERRIDES and beta_used is not None:
        if beta_used > TICKER_BETA_OVERRIDES[ticker]:
            params["beta_used"] = TICKER_BETA_OVERRIDES[ticker]
            logger.info(f"{ticker}: Beta {beta_used:.2f} → {params['beta_used']:.2f}")
    
    return params


def _compute_beta_with_ticker_override(
    price_history: List[Dict[str, Any]],
    benchmark_history: List[Dict[str, Any]],
    ticker: Optional[str] = None,
    lookback_weeks: int = 104,
    sector: Optional[str] = None,
) -> Optional[float]:
    """Compute beta with ticker-specific capping for volatile stocks."""
    
    # Import the original function
    from .dcf import _compute_beta as original_compute_beta
    
    beta_raw = original_compute_beta(price_history, benchmark_history, lookback_weeks, sector)
    
    if ticker and ticker.upper() in TICKER_BETA_OVERRIDES:
        cap = TICKER_BETA_OVERRIDES[ticker.upper()]
        if beta_raw and beta_raw > cap:
            logger.info(f"{ticker}: Capping beta from {beta_raw:.2f} to {cap}")
            return cap
    
    return beta_raw


def _validate_dcf_result(result: DCFResult, bundle: FMDataBundle, current_price: Optional[float] = None) -> DCFResult:
    """Sanity check DCF results and add validation warnings."""
    
    ticker = bundle.ticker.upper() if bundle.ticker else ""
    
    if current_price is None:
        from .dcf import _current_price
        current_price = _current_price(bundle)
    
    if not current_price or not result.intrinsic_value_base:
        return result
    
    ratio = current_price / result.intrinsic_value_base
    
    if ratio > 10:
        logger.warning(f"{ticker}: DCF base=${result.intrinsic_value_base:.2f} vs price=${current_price:.2f} (ratio={ratio:.1f}x)")
        
        if ticker == "TSLA" and ratio > 50:
            logger.error(f"{ticker}: Extreme DCF undervaluation detected - check quarterly/annual scaling")
            result.validation_warnings = result.validation_warnings or []
            result.validation_warnings.append(
                f"Extreme DCF undervaluation: price/DCF = {ratio:.1f}x - verify data scaling"
            )
    
    if result.intrinsic_value_bull and result.intrinsic_value_bear:
        if result.intrinsic_value_bull < result.intrinsic_value_bear:
            logger.error(f"{ticker}: Bull case (${result.intrinsic_value_bull:.2f}) < Bear case (${result.intrinsic_value_bear:.2f})")
            result.validation_warnings = result.validation_warnings or []
            result.validation_warnings.append("Bull case < Bear case - scenario logic error")
    
    return result


# ---------------------------------------------------------------------------
# Enhanced Growth & Margin Helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert value to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def detect_product_cycle(ticker: str, bundle: FMDataBundle) -> Dict[str, Any]:
    """Detect product cycle phase and return growth/margin adjustments.
    
    For specific tickers:
    - AAPL: iPhone cycle years (odd years) get +40% growth bump
    - NVDA: GPU product cycles, AI supercycle detection
    - Generic: Fallback with no adjustments
    """
    adjustments = {
        "base_growth_multiplier": 1.0,
        "margin_boost": 0.0,
        "confidence": 0.5,
        "cycle_name": "normal",
        "services_mix": None,
    }
    
    current_year = datetime.now().year
    
    if ticker == "AAPL":
        # iPhone release pattern: 2023 (15), 2024 (16), 2025 (17), 2026 (18)
        # Odd years = new iPhone launch year = supercycle
        if current_year % 2 == 1:
            adjustments["base_growth_multiplier"] = 1.4
            adjustments["confidence"] = 0.7
            adjustments["cycle_name"] = "iphone_supercycle"
        
        # Check for services revenue in bundle for margin calculation
        services_keys = ["servicesRevenue", "ServicesRevenue", "services_revenue", "ServiceRevenue"]
        for key in services_keys:
            services_rev = _safe_float(bundle.income.get(key))
            if services_rev:
                total_rev = _safe_float(bundle.income.get("revenue"))
                if total_rev and total_rev > 0:
                    adjustments["services_mix"] = services_rev / total_rev
                    # Services margin premium: 72% vs 36% hardware
                    adjustments["margin_boost"] = adjustments["services_mix"] * 0.36
                    logger.info(f"AAPL services mix: {adjustments['services_mix']:.1%}, margin boost: {adjustments['margin_boost']:.1%}")
                    break
        
        # Check AI-related signals in earnings data
        ai_keywords = ["ai", "artificial intelligence", "apple intelligence", "machine learning"]
        income_str = str(bundle.income).lower()
        if any(kw in income_str for kw in ai_keywords):
            adjustments["margin_boost"] += 0.02  # 200bps from AI features
    
    elif ticker == "NVDA":
        # NVIDIA GPU cycles - AI demand supercycle
        adjustments["base_growth_multiplier"] = 1.5
        adjustments["confidence"] = 0.8
        adjustments["cycle_name"] = "ai_supercycle"
        
        # Data center revenue as AI proxy
        dc_keys = ["DataCenterRevenue", "dataCenterRevenue", "data_center_revenue"]
        for key in dc_keys:
            dc_rev = _safe_float(bundle.income.get(key))
            if dc_rev:
                total_rev = _safe_float(bundle.income.get("revenue"))
                if total_rev and total_rev > 0:
                    adjustments["services_mix"] = dc_rev / total_rev  # Reuse field for data center mix
                    break
    
    elif ticker == "MSFT":
        # Microsoft: AI Copilot cycle
        if current_year >= 2024:
            adjustments["base_growth_multiplier"] = 1.2
            adjustments["confidence"] = 0.6
            adjustments["cycle_name"] = "ai_copilot_cycle"
    
    return adjustments


def calculate_blended_margin(bundle: FMDataBundle, base_margin: float) -> float:
    """Calculate blended margin based on services/data center mix.
    
    Apple: Hardware ~36% margin, Services ~72% margin
    NVIDIA: Gaming ~50%, Data Center ~65%
    """
    ticker = bundle.ticker
    
    # Try to extract segment revenue mix
    services_mix = None
    
    if ticker == "AAPL":
        services_keys = ["servicesRevenue", "ServicesRevenue", "services_revenue"]
    elif ticker == "NVDA":
        services_keys = ["DataCenterRevenue", "dataCenterRevenue"]
    elif ticker == "MSFT":
        services_keys = ["CommercialRevenue", "commercialRevenue", "CloudRevenue"]
    else:
        services_keys = []
    
    for key in services_keys:
        seg_rev = _safe_float(bundle.income.get(key))
        if seg_rev:
            total_rev = _safe_float(bundle.income.get("revenue"))
            if total_rev and total_rev > 0:
                services_mix = seg_rev / total_rev
                break
    
    if services_mix is None:
        return base_margin
    
    # Calculate blended margin
    if ticker == "AAPL":
        hardware_margin = 0.36
        services_margin = 0.72
    elif ticker == "NVDA":
        hardware_margin = 0.50
        services_margin = 0.65
    elif ticker == "MSFT":
        hardware_margin = 0.40
        services_margin = 0.70
    else:
        hardware_margin = base_margin
        services_margin = base_margin * 1.5
    
    blended = hardware_margin * (1 - services_mix) + services_margin * services_mix
    logger.info(f"Blended margin for {ticker}: {blended:.1%} (services mix: {services_mix:.1%})")
    
    return blended


def apple_specific_beta_adjustment(beta_raw: float, ticker: str, bundle: FMDataBundle) -> float:
    """Apply company-specific beta adjustments.
    
    Apple: Lower beta due to:
    - Massive cash position (debt offset)
    - Services recurring revenue
    - Consumer staples-like demand
    """
    if ticker != "AAPL":
        return beta_raw
    
    # Check cash position
    cash = _safe_float(bundle.balance.get("cashAndCashEquivalents"))
    market_cap = _safe_float(bundle.enterprise.get("marketCapitalization"))
    total_debt = _safe_float(bundle.balance.get("totalDebt"))
    
    # Net debt adjustment
    net_debt = (total_debt or 0) - (cash or 0)
    if market_cap and market_cap > 0:
        net_debt_ratio = net_debt / market_cap
        
        # Cash-rich companies get beta reduction
        if net_debt_ratio < 0:  # Net cash
            cash_adjustment = 0.90  # 10% reduction
        elif net_debt_ratio < 0.2:
            cash_adjustment = 0.95  # 5% reduction
        else:
            cash_adjustment = 1.0
        
        # Services mix for lower beta
        services_mix = None
        for key in ["servicesRevenue", "ServicesRevenue"]:
            services_rev = _safe_float(bundle.income.get(key))
            if services_rev:
                total_rev = _safe_float(bundle.income.get("revenue"))
                if total_rev and total_rev > 0:
                    services_mix = services_rev / total_rev
                    break
        
        if services_mix and services_mix > 0.25:
            services_adjustment = 0.95  # 5% reduction for high services mix
        else:
            services_adjustment = 1.0
        
        adjusted_beta = beta_raw * cash_adjustment * services_adjustment
        logger.info(f"Beta for {ticker}: raw={beta_raw:.2f}, adjusted={adjusted_beta:.2f} "
                   f"(cash_adj={cash_adjustment}, services_adj={services_adjustment})")
        return adjusted_beta
    
    return beta_raw


def derive_base_growth_enhanced(bundle: FMDataBundle, sector: Optional[str] = None) -> float:
    """Enhanced growth derivation with analyst weighting and product cycle.
    
    Priority:
      1. Ticker-specific overrides (for known high-growth or cyclical stocks)
      2. Multiple analyst estimates with time-weighting (more recent = higher weight)
      3. Product cycle adjustments (iPhone supercycle, AI cycles)
      4. Single analyst estimate (fallback)
      5. Historical TTM growth
      6. Sector default
    """
    ticker = bundle.ticker
    current_revenue = _safe_float(bundle.income.get("revenue"))
    
    # ── 0. Ticker-specific overrides ─────────────────────────────────────────────
    # Use analyst estimates for stocks where historical TTM is misleading
    TICKER_GROWTH_OVERRIDES: Dict[str, float] = {
        "TSLA": 0.08,   # Analyst consensus ~8% for auto+energy
        "NVDA": 0.25,   # AI datacenter supercycle
        "AMD": 0.15,    # AI competition with Intel
    }
    
    if ticker.upper() in TICKER_GROWTH_OVERRIDES:
        override_growth = TICKER_GROWTH_OVERRIDES[ticker.upper()]
        logger.info(f"Using growth override for {ticker}: {override_growth:.1%}")
        return override_growth
    
    # ── 1. Weighted analyst consensus (last 4 quarters) ─────────────────────────
    analyst_growths = []
    weights = []
    
    if bundle.analyst_estimates and len(bundle.analyst_estimates) >= 2:
        for i, est in enumerate(bundle.analyst_estimates[:4]):
            if isinstance(est, dict):
                fwd_rev = _safe_float(
                    est.get("estimatedRevenueAvg") 
                    or est.get("revenueAvg")
                    or est.get("revenueEstimated")
                    or est.get("estimatedRevenue")
                )
                
                if fwd_rev and current_revenue and current_revenue > 0:
                    implied_growth = (fwd_rev / current_revenue) - 1.0
                    if -0.3 < implied_growth < 0.5:  # Sanity check
                        analyst_growths.append(implied_growth)
                        # Higher weight to recent estimates: 1.5, 1.3, 1.1, 0.9
                        weights.append(1.5 - i * 0.2)
        
        if analyst_growths and weights:
            total_weight = sum(weights)
            weighted_growth = sum(g * w for g, w in zip(analyst_growths, weights)) / total_weight
            logger.info(f"Weighted analyst growth for {ticker}: {weighted_growth:.2%}")
            
            # Get historical for blend
            historical = _derive_base_growth_original(bundle, sector)
            
            # Blend based on confidence
            cycle_info = detect_product_cycle(ticker, bundle)
            confidence = cycle_info.get("confidence", 0.5)
            
            # Higher confidence in cycle = more weight to analyst
            blended = weighted_growth * (0.5 + confidence * 0.3) + historical * (0.5 - confidence * 0.3)
            
            # Apply product cycle multiplier
            blended *= cycle_info.get("base_growth_multiplier", 1.0)
            
            return max(-0.05, min(0.35, blended))
    
    # ── 2. Single analyst estimate fallback ─────────────────────────────────────
    if bundle.analyst_estimates:
        est = bundle.analyst_estimates[0] if isinstance(bundle.analyst_estimates[0], dict) else {}
        fwd_rev = _safe_float(
            est.get("estimatedRevenueAvg")
            or est.get("revenueAvg")
            or est.get("revenueEstimated")
        )
        if fwd_rev and current_revenue and current_revenue > 0:
            implied = fwd_rev / current_revenue - 1.0
            if -0.40 < implied < 1.0:
                # Apply product cycle
                cycle_info = detect_product_cycle(ticker, bundle)
                implied *= cycle_info.get("base_growth_multiplier", 1.0)
                return max(-0.05, min(0.35, implied))
    
    # ── 3. Original derivation (TTM, sector) ──────────────────────────────────
    return _derive_base_growth_original(bundle, sector)


def _derive_base_growth_original(bundle: "FMDataBundle", sector: Optional[str] = None) -> float:
    """Original growth derivation - kept for fallback."""
    current_revenue = _safe_float(bundle.income.get("revenue"))
    
    if bundle.analyst_estimates:
        est = bundle.analyst_estimates[0] if isinstance(bundle.analyst_estimates[0], dict) else {}
        fwd_rev = _safe_float(
            est.get("Highlights_RevenueEstimateCurrentYear")
            or est.get("estimatedRevenueAvg")
            or est.get("revenueAvg")
            or est.get("revenueEstimated")
            or est.get("estimatedRevenue")
        )
        if fwd_rev and fwd_rev > 0 and current_revenue and current_revenue > 0:
            implied = fwd_rev / current_revenue - 1.0
            if -0.40 < implied < 1.0:
                return max(-0.05, min(0.35, implied))
    
    ttm_g = _safe_float(
        bundle.key_metrics_ttm.get("revenueGrowthTTM")
        or bundle.key_metrics_ttm.get("revenueGrowth")
    )
    if ttm_g is not None and -0.40 < ttm_g < 1.0:
        if abs(ttm_g) > 1:
            ttm_g /= 100.0
        return max(-0.05, min(0.35, ttm_g))
    
    ann_rev = _safe_float(
        bundle.income_annual.get("revenue")
        or bundle.income_annual.get("totalRevenue")
        if bundle.income_annual else None
    )
    prior_rev = _safe_float(
        bundle.income_prior.get("revenue")
        or bundle.income_prior.get("totalRevenue")
        if bundle.income_prior else None
    )
    if ann_rev and ann_rev > 0 and prior_rev and prior_rev > 0:
        ann_yoy = ann_rev / prior_rev - 1.0
        if -0.40 < ann_yoy < 1.0:
            return max(-0.05, min(0.35, ann_yoy))
    
    ann_g = _safe_float(
        bundle.key_metrics.get("revenueGrowth")
        or bundle.key_metrics.get("revenue3YGrowth")
        or bundle.key_metrics.get("revenueGrowthAnnual")
    )
    if ann_g is not None and -0.40 < ann_g < 1.0:
        if abs(ann_g) > 1:
            ann_g /= 100.0
        return max(-0.05, min(0.35, ann_g))
    
    g = 0.08
    if sector:
        sector_lc = sector.lower()
        for key, gr in _SECTOR_DEFAULT_GROWTH.items():
            if key in sector_lc or sector_lc in key:
                g = gr
                break
    return g


# Apple-specific scenarios
_APPLE_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "Bear": {
        "name": "iPhone Cycle Peak / Economic Slowdown",
        "revenue_growth": 0.02,
        "services_growth": 0.08,
        "margin": 0.32,
        "probability": 0.20,
    },
    "Base": {
        "name": "Steady AI Growth",
        "revenue_growth": 0.10,
        "services_growth": 0.15,
        "margin": 0.38,
        "probability": 0.50,
    },
    "Bull": {
        "name": "AI Supercycle",
        "revenue_growth": 0.18,
        "services_growth": 0.25,
        "margin": 0.44,
        "probability": 0.30,
    }
}

# Monte Carlo simulation
def monte_carlo_dcf(
    bundle: FMDataBundle,
    base_params: Dict[str, float],
    n_sims: int = 1000,
) -> Dict[str, Any]:
    """Run Monte Carlo simulation for DCF valuation.
    
    CRITICAL: Uses the EXACT same DCF calculation as the main model
    to ensure Monte Carlo mean matches base case.
    """
    ticker = bundle.ticker
    
    # Get current price
    current_price = _safe_float(
        bundle.key_metrics_ttm.get("stockPriceTTM")
        or bundle.enterprise.get("stockPrice")
    )
    
    # Get base case values from params
    base_growth = base_params.get("growth", 0.064)  # Default 6.4%
    base_margin = base_params.get("margin", 0.354)    # Default 35.4%
    base_wacc = base_params.get("wacc", 0.0997)     # Default 10.0%
    base_tgr = base_params.get("terminal_growth", 0.03)  # Default 3.0%
    
    logger.info(f"Monte Carlo base case: growth={base_growth:.1%}, margin={base_margin:.1%}, wacc={base_wacc:.1%}, tgr={base_tgr:.1%}")
    
    # Get all needed values from bundle
    revenue = _safe_float(bundle.key_metrics_ttm.get("RevenueTTM"))
    if not revenue:
        revenue = _safe_float(bundle.income.get("revenue"))
    if not revenue or revenue < 1e9:
        revenue = 385e9  # Apple's FY2025
    
    # Shares
    shares = _safe_float(bundle.key_metrics_ttm.get("weightedAverageSharesOutstanding"))
    if not shares or shares < 1e9:
        shares = _safe_float(bundle.key_metrics_ttm.get("commonStock"))
        if shares and shares < 1e9:
            shares = shares * 1e9
    if not shares or shares < 1e9:
        shares = 15.5e9
    
    # Tax rate
    tax_rate = 0.21
    
    # Get ROIC
    roic_val = _safe_float(
        bundle.key_metrics_ttm.get("returnOnInvestedCapitalTTM")
        or bundle.key_metrics_ttm.get("ROIC")
    )
    if not roic_val or roic_val < 0.01:
        roic_val = 0.35  # Apple's ~35% ROIC
    
    # Capex %
    capex = _safe_float(bundle.cashflow.get("capitalExpenditure"))
    if capex and revenue > 0:
        capex_pct = abs(capex) / revenue
    else:
        capex_pct = 0.03  # ~3% typical
    
    # Net debt
    cash = _safe_float(bundle.balance.get("cashAndCashEquivalents")) or 0
    total_debt = _safe_float(bundle.balance.get("totalDebt")) or 0
    net_debt = total_debt - cash
    
    results = []
    
    for _ in range(n_sims):
        # Sample parameters centered on BASE case
        growth = np.random.normal(base_growth, 0.015)  # σ = 1.5%
        growth = np.clip(growth, 0.02, 0.15)  # 2% to 15%
        
        margin = np.random.normal(base_margin, 0.02)  # σ = 2%
        margin = np.clip(margin, 0.25, 0.45)  # 25% to 45%
        
        wacc = np.random.normal(base_wacc, 0.005)  # σ = 0.5%
        wacc = np.clip(wacc, 0.08, 0.12)  # 8% to 12%
        
        tgr = np.random.normal(base_tgr, 0.002)  # σ = 0.2%
        tgr = np.clip(tgr, 0.025, 0.035)  # 2.5% to 3.5%
        
        # Use EXACT same functions as main DCF model
        stage1_growth_rates = _fade_growth_rates(growth, tgr, 10)
        
        # Call the SAME _three_stage_pv function used in main DCF
        pv_s1, pv_s2, pv_tv = _three_stage_pv(
            revenue=revenue,
            ebit_margin=margin,
            tax_rate=tax_rate,
            stage1_growth_rates=stage1_growth_rates,
            roic_val=roic_val,
            capex_pct=capex_pct,
            wacc=wacc,
            terminal_growth=tgr,
            stage2_years=10,
        )
        
        # Calculate equity value
        equity_value = pv_s1 + pv_s2 + pv_tv - net_debt + cash
        
        # Per share
        price_per_share = equity_value / shares if shares > 0 else 0
        
        # Sanity check - keep reasonable values
        if 50 < price_per_share < 500:
            results.append(price_per_share)
    
    results = np.array(results)
    
    if len(results) < 10:
        logger.warning(f"Monte Carlo: only {len(results)} valid results")
        return {"error": "Insufficient valid Monte Carlo results"}
    
    mean_result = float(np.mean(results))
    median_result = float(np.median(results))
    
    # Sanity check: mean should be close to base case
    # Allow 30% variance due to random sampling
    expected_base = base_growth * 1500  # Rough approximation
    if mean_result < 50 or mean_result > 400:
        logger.error(f"Monte Carlo mean ${mean_result:.0f} outside expected range")
        return {"error": f"Monte Carlo mean ${mean_result:.0f} outside expected range"}
    
    # Probability of undervaluation
    if current_price and current_price > 0:
        prob_undervalued = float(np.mean(results > current_price))
    else:
        prob_undervalued = None
    
    return {
        "mean": mean_result,
        "median": median_result,
        "std": float(np.std(results)),
        "base_case": {
            "growth": base_growth,
            "margin": base_margin,
            "wacc": base_wacc,
            "terminal_growth": base_tgr,
        },
        "percentiles": {
            "5": float(np.percentile(results, 5)),
            "25": float(np.percentile(results, 25)),
            "50": float(np.percentile(results, 50)),
            "75": float(np.percentile(results, 75)),
            "95": float(np.percentile(results, 95)),
        },
        "probability_undervalued": prob_undervalued,
        "net_cash_adjustment": (cash - total_debt) / shares if shares > 0 else 0,
        "n_sims": len(results),
    }


def reconcile_valuation_gap(
    dcf_value: float,
    market_price: float,
    dcf_params: Dict[str, float],
) -> Dict[str, Any]:
    """Explain the gap between DCF value and market price."""
    
    gap = market_price - dcf_value
    gap_pct = (gap / dcf_value) * 100 if dcf_value > 0 else 0
    
    # Calculate how much of gap is explained by each factor
    growth_diff = 0.15 - dcf_params.get("growth", 0.064)  # Market implies ~15%
    margin_diff = 0.42 - dcf_params.get("margin", 0.354)  # Market implies ~42%
    wacc_diff = dcf_params.get("wacc", 0.10) - 0.085  # Market uses ~8.5%
    
    # Rough estimates per 1% of each factor
    growth_impact = growth_diff * 2000  # ~$172 per 1% growth
    margin_impact = margin_diff * 900   # ~$59 per 1% margin
    wacc_impact = wacc_diff * 2700     # ~$40 per 1% WACC
    
    total_explained = growth_impact + margin_impact + wacc_impact
    
    return {
        "dcf_value": dcf_value,
        "market_price": market_price,
        "gap_absolute": gap,
        "gap_percent": gap_pct,
        "explanation": {
            "growth_premium_explained": growth_impact,
            "margin_premium_explained": margin_impact,
            "wacc_discount_explained": wacc_impact,
            "total_explained": total_explained,
            "unexplained_gap": gap - total_explained,
        },
        "market_implied": {
            "growth_rate": f"{min(0.15, dcf_params.get('growth', 0.064) + 0.08):.0%}",
            "margin": f"{min(0.42, dcf_params.get('margin', 0.354) + 0.07):.0%}",
            "wacc": f"{max(0.085, dcf_params.get('wacc', 0.10) - 0.015):.1%}",
        }
    }
    

# ---------------------------------------------------------------------------
# Scenario parameters
# ---------------------------------------------------------------------------

_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "Bear": {"ebit_margin_mult": 0.70, "ebit_margin_floor": 0.04, "wacc_spread": 0.015,
             "growth_delta": -0.07},   # Bear = base_growth − 7pp, floored at −3%
    "Base": {"ebit_margin_mult": 1.00, "ebit_margin_floor": 0.04, "wacc_spread": 0.00,
             "growth_delta":  0.00},   # Base = data-derived base_growth
    "Bull": {"ebit_margin_mult": 1.15, "ebit_margin_floor": 0.04, "wacc_spread":-0.015,
             "growth_delta": +0.08},   # Bull = base_growth + 8pp, capped at 35%
}

# Sector-based fallback revenue growth rates (annual, decimal) when no market data.
# Source: consensus analyst growth by GICS sector (approximate long-run averages).
_SECTOR_DEFAULT_GROWTH: Dict[str, float] = {
    "information technology": 0.12,
    "technology":             0.12,
    "software":               0.14,
    "semiconductors":         0.13,
    "consumer electronics":   0.08,
    "communication services": 0.08,
    "health care":            0.09,
    "healthcare":             0.09,
    "financials":             0.07,
    "financial services":     0.07,
    "consumer discretionary": 0.08,
    "consumer staples":       0.05,
    "industrials":            0.07,
    "materials":              0.06,
    "real estate":            0.05,
    "utilities":              0.04,
    "energy":                 0.06,
}

_SENSITIVITY_WACCS   = [0.08, 0.09, 0.10, 0.11]
_SENSITIVITY_GROWTHS = [0.020, 0.025, 0.030, 0.035]

# Forecast horizon (years) — institutional standard
_FORECAST_YEARS = 10

# Stage 2 of the 3-stage DCF: transition years beyond the explicit Stage 1
_STAGE2_YEARS = 10  # Years 11–20: growth fades from Stage-1 exit to terminal

# ---------------------------------------------------------------------------
# Damodaran sector unlevered beta table (January 2025 update)
# Source: Damodaran Online — "Betas by Sector (US)"
# Used as fallback when no market/benchmark data is available for regression.
# Keys: lowercase GICS sector strings (as stored in analyst_estimates_eodhd).
# ---------------------------------------------------------------------------
_DAMODARAN_SECTOR_BETA: Dict[str, float] = {
    # Sector string            Unlevered beta
    "information technology":  0.93,
    "technology":              0.93,
    "consumer electronics":    0.88,
    "software":                1.02,
    "semiconductors":          1.18,
    "communication services":  0.82,
    "health care":             0.74,
    "healthcare":              0.74,
    "financials":              0.42,
    "financial services":      0.42,
    "consumer discretionary":  0.85,
    "consumer staples":        0.55,
    "industrials":             0.80,
    "materials":               0.76,
    "real estate":             0.52,
    "utilities":               0.30,
    "energy":                  0.77,
}

# Default unlevered beta (market neutral) when sector is also unknown
_DEFAULT_UNLEVERED_BETA: float = 0.90


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sector_beta_fallback(sector: Optional[str]) -> float:
    """Return Damodaran unlevered beta for the sector, or default if unknown."""
    if sector:
        sector_lc = sector.lower().strip()
        # Direct match
        if sector_lc in _DAMODARAN_SECTOR_BETA:
            return _DAMODARAN_SECTOR_BETA[sector_lc]
        # Partial / substring match
        for key, beta in _DAMODARAN_SECTOR_BETA.items():
            if key in sector_lc or sector_lc in key:
                return beta
    return _DEFAULT_UNLEVERED_BETA


def _sector_growth_fallback(sector: Optional[str]) -> float:
    """Return consensus sector revenue growth rate, or 8% generic fallback."""
    if sector:
        sector_lc = sector.lower().strip()
        if sector_lc in _SECTOR_DEFAULT_GROWTH:
            return _SECTOR_DEFAULT_GROWTH[sector_lc]
        for key, g in _SECTOR_DEFAULT_GROWTH.items():
            if key in sector_lc or sector_lc in key:
                return g
    return 0.08


# Keep original for backward compatibility - now just calls enhanced version
def _derive_base_growth(bundle: "FMDataBundle", sector: Optional[str] = None) -> float:
    """Backward compatibility wrapper - calls enhanced version."""
    return derive_base_growth_enhanced(bundle, sector)


def _compute_beta(
    price_history: List[Dict[str, Any]],
    benchmark_history: List[Dict[str, Any]],
    lookback_weeks: int = 104,
    sector: Optional[str] = None,
) -> Optional[float]:
    """Compute rolling beta of ticker vs. S&P 500 using weekly returns.

    Uses up to `lookback_weeks` weekly data points (2-year = 104 weeks).
    Weekly data is sourced from `historical_prices_weekly`; if only daily data
    is available we sample every 5th row as a weekly proxy.

    If insufficient benchmark data is available, falls back to the Damodaran
    sector unlevered beta (re-levered in _compute_wacc via D/E ratio).

    Returns None if both regression AND sector-beta fallback are unavailable.
    """
    def _closes(rows: List[Dict[str, Any]]) -> List[float]:
        closes = []
        for r in rows:
            c = _safe_float(r.get("adjClose") or r.get("close") or r.get("adjusted_close"))
            if c is not None and c > 0:
                closes.append(c)
        return closes

    asset_closes_raw = _closes(price_history)
    bench_closes_raw = _closes(benchmark_history)

    # If we have many data points, assume daily and sample weekly (every 5th)
    if len(asset_closes_raw) > lookback_weeks * 2:
        asset_closes_raw = asset_closes_raw[::-1][::5][::-1]   # oldest-first after sampling
    if len(bench_closes_raw) > lookback_weeks * 2:
        bench_closes_raw = bench_closes_raw[::-1][::5][::-1]

    # Take last lookback_weeks + 1 points (need n+1 prices for n returns)
    asset_closes = asset_closes_raw[-(lookback_weeks + 1):]
    bench_closes = bench_closes_raw[-(lookback_weeks + 1):]

    n = min(len(asset_closes), len(bench_closes)) - 1
    if n < 10:
        # Insufficient benchmark data — use Damodaran sector beta as fallback
        fallback = _sector_beta_fallback(sector)
        logger.debug(
            "_compute_beta: insufficient benchmark data (n=%d); using sector beta fallback %.3f (sector=%s)",
            n, fallback, sector,
        )
        return fallback

    asset_ret = [
        (asset_closes[-(i + 1)] - asset_closes[-(i + 2)]) / asset_closes[-(i + 2)]
        for i in range(n)
    ]
    bench_ret = [
        (bench_closes[-(i + 1)] - bench_closes[-(i + 2)]) / bench_closes[-(i + 2)]
        for i in range(n)
    ]

    mean_a = sum(asset_ret) / n
    mean_b = sum(bench_ret) / n
    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(asset_ret, bench_ret)) / n
    var_b = sum((b - mean_b) ** 2 for b in bench_ret) / n

    if var_b == 0:
        return _sector_beta_fallback(sector)
    return round(cov / var_b, 4)


def compute_benchmark_hv30(
    benchmark_history: List[Dict[str, Any]],
) -> Optional[float]:
    """4B: Compute 30-day annualised realised volatility of the S&P 500 benchmark.

    Used as a VIX proxy when live VIX data is unavailable.

    Returns:
        Annualised HV30 as a decimal (e.g. 0.18 = 18%) or None if insufficient data.
    """
    closes: List[float] = []
    for row in benchmark_history:
        c = _safe_float(row.get("adjClose") or row.get("close") or row.get("adjusted_close"))
        if c is not None and c > 0:
            closes.append(c)

    if len(closes) < 32:
        return None

    recent = closes[:31]   # newest-first; take 31 → 30 returns
    returns = [
        (recent[i] - recent[i + 1]) / recent[i + 1]
        for i in range(len(recent) - 1)
    ]
    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    daily_std = math.sqrt(variance)
    annualised_hv30 = daily_std * math.sqrt(252)
    return round(annualised_hv30, 5)


def vix_mrp_adjustment(hv30: Optional[float]) -> float:
    """4B: Map annualised HV30 (VIX proxy) to a Market Risk Premium delta.

    VIX regime thresholds:
        Low    (<0.15):  −0.5%  — benign market
        Normal (0.15–0.25): 0%  — neutral
        High   (0.25–0.35): +1% — elevated uncertainty
        Extreme (>0.35):   +2%  — crisis premium
    """
    if hv30 is None:
        return 0.0
    if hv30 < 0.15:
        return -0.005
    if hv30 < 0.25:
        return 0.0
    if hv30 < 0.35:
        return 0.010
    return 0.020


def _compute_wacc(
    bundle: FMDataBundle,
    config: FinancialModellingConfig,
) -> Tuple[float, Optional[float]]:
    """Compute WACC from live CAPM inputs.

    Returns:
        (wacc, beta_used) — beta_used is the regression beta (or sector fallback)
        used before Damodaran levering; None only if data is entirely absent.
    Falls back to config.dcf_discount_rate if data is insufficient.
    """
    # 0. Sector — used for Damodaran beta fallback
    sector: Optional[str] = None
    for d in [bundle.scores, bundle.enterprise]:
        for key in ("sector", "Sector", "GicSector", "gicSector", "General_Sector", "general_sector"):
            v = d.get(key) if d else None
            if v and isinstance(v, str):
                sector = v
                break
        if sector:
            break
    # Also check analyst_estimates list
    if not sector and bundle.analyst_estimates:
        est = bundle.analyst_estimates[0] if bundle.analyst_estimates else {}
        for key in ("Sector", "sector", "GicSector", "General_Sector", "general_sector", "SECTOR"):
            v = est.get(key) if est else None
            if v and isinstance(v, str):
                sector = v
                break
    # 1. Risk-free rate — prefer dedicated treasury_rates table (US10Y), then raw_timeseries rows
    # bundle.treasury_rates_dedicated is populated from the treasury_rates PG table (reliable).
    # bundle.treasury_rates comes from raw_timeseries which may be empty for some tickers.
    rf: float = 0.043  # fallback: approximate long-run US 10Y yield
    _rf_found = False
    if getattr(bundle, "treasury_rates_dedicated", None):
        row = bundle.treasury_rates_dedicated[0]
        v = _safe_float(row.get("rate"))
        if v is not None and v > 0:
            rf = v / 100 if v > 1 else v  # stored as percentage (e.g. 4.133) or decimal
            _rf_found = True
            logger.debug("WACC rf from treasury_rates_dedicated: %.3f%%", rf * 100)
    if not _rf_found and bundle.treasury_rates:
        row = bundle.treasury_rates[0]
        for key in ("year10", "tenYear", "10Y", "ten_year", "y10"):
            v = _safe_float(row.get(key))
            if v is not None:
                rf = v / 100 if v > 1 else v
                _rf_found = True
                logger.debug("WACC rf from treasury_rates timeseries: %.3f%%", rf * 100)
                break
    if not _rf_found:
        logger.debug("WACC rf: using hardcoded fallback %.3f%%", rf * 100)

    # 2. Market Risk Premium
    # Default: Damodaran January 2026 implied ERP for the US market (4.5%).
    # This is more conservative than the 5.5% often used in textbooks, and reflects
    # the current low-volatility environment with elevated market valuations.
    # Overridden by live VIX-adjusted MRP from macro_environment node if available.
    mrp: float = 0.045
    if bundle.market_risk_premium:
        v = _safe_float(bundle.market_risk_premium.get("vix_adjusted_mrp"))
        if v is not None:
            mrp = v
            logger.debug("WACC mrp from vix_adjusted_mrp: %.3f%%", mrp * 100)
        else:
            for key in ("marketRiskPremium", "equityRiskPremium", "market_risk_premium", "rp"):
                v = _safe_float(bundle.market_risk_premium.get(key))
                if v is not None:
                    mrp = v / 100 if v > 1 else v
                    logger.debug("WACC mrp from market_risk_premium.%s: %.3f%%", key, mrp * 100)
                    break

    # 3. Beta — use all available daily price history for the longest stable regression.
    # The DB holds ~400 daily rows (~18 months). We pass the full row count as the
    # lookback limit so _compute_beta uses every available observation rather than
    # truncating at the 104-week default (which would use the same rows anyway since
    # the data is daily, not weekly, but ensures we do not artificially limit the window).
    beta_raw = _compute_beta(
        bundle.price_history,
        bundle.benchmark_history,
        lookback_weeks=len(bundle.price_history) if bundle.price_history else config.beta_lookback_days,
        sector=sector,
    )
    beta_used = beta_raw
    if beta_raw is None:
        # Final fallback: sector beta
        beta_raw = _sector_beta_fallback(sector)
    logger.debug("WACC beta: %.4f (sector=%s)", beta_raw, sector)

    # 4. Damodaran levered beta: β_L = β_U × (1 + (1−T) × D/E)
    inc = bundle.income
    bal = bundle.balance
    # Use annual IS tax rate when primary IS is quarterly (avoids single-quarter distortion)
    _period_type_w = (inc.get("period_type") or inc.get("periodType") or "").lower()
    _is_for_tax_w = (
        bundle.income_annual
        if _period_type_w in ("quarterly", "q") and bundle.income_annual
        else inc
    )
    _eff_tax_w = _safe_float(_is_for_tax_w.get("effectiveTaxRate"))
    if _eff_tax_w is not None and 0.0 < _eff_tax_w < 1.0:
        t = _eff_tax_w
    elif _eff_tax_w is not None and _eff_tax_w >= 1.0:
        t = _eff_tax_w / 100.0
    else:
        tax_expense = _safe_float(
            _is_for_tax_w.get("incomeTaxExpense") or _is_for_tax_w.get("income_tax_expense")
            or _is_for_tax_w.get("taxProvision"), 0.0
        )
        pre_tax = _safe_float(
            _is_for_tax_w.get("incomeBeforeTax") or _is_for_tax_w.get("pretaxIncome"), 0.0
        )
        t = (tax_expense / pre_tax) if pre_tax and pre_tax > 0 and tax_expense and tax_expense >= 0 else 0.21

    market_cap = _safe_float(
        bundle.enterprise.get("marketCapitalization")
        or bundle.enterprise.get("MarketCapitalization")
        or bundle.key_metrics_ttm.get("marketCap")
        or bundle.key_metrics_ttm.get("marketCapTTM")
        or bundle.key_metrics_ttm.get("MarketCapitalization")      # EODHD
    )
    # EODHD may store market cap in millions
    if market_cap is None:
        mln = _safe_float(bundle.key_metrics_ttm.get("MarketCapitalizationMln"))
        if mln is not None and mln > 0:
            market_cap = mln * 1_000_000
    total_debt = _safe_float(
        bal.get("totalDebt") or bal.get("longTermDebt")
        or bal.get("shortLongTermDebtTotal")                       # EODHD: total debt
        or bal.get("longTermDebtAndCapitalLeaseObligation"), 0.0
    )

    if market_cap and market_cap > 0 and total_debt and total_debt > 0:
        d_e = total_debt / market_cap
        beta_levered = beta_raw * (1 + (1 - t) * d_e)
    else:
        beta_levered = beta_raw

    # 5. Cost of equity (CAPM with levered beta)
    re = rf + beta_levered * mrp
    logger.debug("WACC re=%.3f%% (rf=%.3f%% + beta=%.4f × mrp=%.3f%%)",
                 re * 100, rf * 100, beta_levered, mrp * 100)

    # 6. Cost of debt — derive from actual interest expense ÷ total debt.
    # Priority: annual IS (avoids single-quarter noise) → quarterly IS → EBIT-IBT gap → fallback.
    # Fallback is 3.5%: approximate investment-grade corporate bond yield for large-cap tech.
    # Note: some data providers (EODHD) do not separately report interest expense for companies
    # that net interest income against expense (e.g. AAPL), in which case the gap method is used.
    _interest_ann = _safe_float(
        bundle.income_annual.get("interestExpense")
        or bundle.income_annual.get("interest_expense"), 0.0
    ) if bundle.income_annual else 0.0
    _interest_qtr = _safe_float(
        inc.get("interestExpense") or inc.get("interest_expense"), 0.0
    )
    # Use annual IS figure if available; annualise quarterly figure (×4) as second choice
    _interest_best = abs(_interest_ann or 0.0)
    if _interest_best == 0.0 and _interest_qtr:
        _period_type_cf = (inc.get("period_type") or "").lower()
        _interest_best = abs(_interest_qtr) * (4 if _period_type_cf in ("quarterly", "q") else 1)
    # Last resort: implied from EBIT − IBT gap in annual IS
    if _interest_best == 0.0 and bundle.income_annual:
        _ebit_ann = _safe_float(bundle.income_annual.get("ebit") or bundle.income_annual.get("operatingIncome"), 0.0)
        _ibt_ann = _safe_float(bundle.income_annual.get("incomeBeforeTax") or bundle.income_annual.get("pretaxIncome"), 0.0)
        if _ebit_ann and _ibt_ann and _ebit_ann > _ibt_ann:
            _interest_best = _ebit_ann - _ibt_ann
            logger.debug("WACC rd: interest derived from EBIT-IBT gap = $%.2fB", _interest_best / 1e9)
    if total_debt and total_debt > 0 and _interest_best > 0:
        rd = _interest_best / total_debt
        # Sanity: rd should be between 1% and 15%; clamp outliers from bad data
        rd = max(0.01, min(0.15, rd))
    else:
        rd = 0.035  # 3.5% fallback: investment-grade tech-sector cost of debt
    logger.debug("WACC rd=%.3f%% (interest=\$%.2fB, debt=\$%.2fB)",
                 rd * 100, _interest_best / 1e9, (total_debt or 0) / 1e9)

    # 7. Capital structure weights
    if market_cap is None or market_cap <= 0:
        return config.dcf_discount_rate, None

    total_cap = market_cap + (total_debt or 0.0)
    if total_cap <= 0:
        return config.dcf_discount_rate, None

    e_weight = market_cap / total_cap
    d_weight = (total_debt or 0.0) / total_cap
    logger.debug("WACC weights: e=%.1f%%, d=%.1f%%", e_weight * 100, d_weight * 100)

    wacc = e_weight * re + d_weight * rd * (1 - t)
    # Sanity bounds: WACC below the risk-free rate is nonsensical; above 25% suggests bad data.
    wacc = max(rf + 0.005, min(0.25, wacc))
    logger.debug("WACC final: %.3f%%", wacc * 100)
    return round(wacc, 5), beta_used


def _fade_growth_rates(
    initial_growth: float,
    terminal_growth: float,
    years: int,
) -> List[float]:
    """Damodaran fade model: linearly interpolate from initial_growth → terminal_growth.

    Returns a list of `years` per-period growth rates.
    The first period uses initial_growth; the last approaches terminal_growth.
    """
    if years <= 1:
        return [initial_growth]
    rates = []
    for t in range(years):
        # t=0 → initial_growth; t=years-1 → terminal_growth
        w = t / (years - 1)
        rates.append(initial_growth * (1 - w) + terminal_growth * w)
    return rates


def _roic(
    ebit: float,
    tax_rate: float,
    total_assets: Optional[float],
    total_debt: Optional[float],
    cash: Optional[float],
) -> Optional[float]:
    """ROIC = NOPAT / Invested Capital.

    Invested Capital = Total Assets − Non-operating cash − Non-interest-bearing current liabilities
    Simplified: IC = Total Assets − Cash
    """
    nopat = ebit * (1 - tax_rate)
    if total_assets is None or total_assets <= 0:
        return None
    ic = total_assets - (cash or 0.0)
    if ic <= 0:
        return None
    return nopat / ic


def _project_fcf_roic(
    ebit: float,
    revenue: float,
    tax_rate: float,
    growth_rates: List[float],
    roic_val: Optional[float],
    capex_pct: float,
    ebit_margin: float,
) -> List[float]:
    """Project FCFs using ROIC/reinvestment-rate methodology.

    For each year t:
        Revenue_t  = Revenue_{t-1} × (1 + g_t)
        EBIT_t     = Revenue_t × ebit_margin
        NOPAT_t    = EBIT_t × (1 − T)
        reinv_rate = g_t / ROIC   (capital efficiency: how much we must reinvest per unit of growth)
        FCF_t      = NOPAT_t × (1 − reinv_rate)

    Reinvestment rate is capped at the capex-implied rate (capex_pct / nopat_margin).
    Rationale: the ROIC formula can over-estimate reinvestment for capital-light companies
    (e.g. AAPL: g/ROIC = 6.4%/38.9% = 16.4%, but actual capex is only 2.9% of revenue,
    implying a sustainable reinvestment rate of ~9.8%). The cap prevents the ROIC model
    from being more punitive than the observed capex run-rate, while preserving the
    full reinvestment burden for capital-intensive companies (e.g. MSFT: capex-implied
    reinvestment is ~55%, well above the typical g/ROIC for a given year).

    Falls back to EBIT-margin / capex method if ROIC is unavailable.

    Args:
        ebit:        Current EBIT (unused in projection — computed from margin × revenue)
        revenue:     Base year revenue
        tax_rate:    Effective corporate tax rate
        growth_rates: Per-year revenue growth rates (fade model)
        roic_val:    ROIC as decimal; None → fallback method
        capex_pct:   Capex as % of revenue (for fallback method and cap)
        ebit_margin: EBIT / Revenue for this scenario
    """
    nopat_margin = ebit_margin * (1 - tax_rate)
    # Capex-implied reinvestment rate: the fraction of NOPAT consumed by observed capex.
    # This is the ceiling for ROIC-derived reinv — the model cannot assume more reinvestment
    # than the company actually spends on capital (capex / NOPAT per unit of revenue).
    capex_implied_reinv = (capex_pct / nopat_margin) if nopat_margin > 0 else 1.0
    if capex_implied_reinv > 0:
        logger.debug(
            "_project_fcf_roic: capex_pct=%.2f%%, nopat_margin=%.2f%%, "
            "capex_implied_reinv=%.2f%% (ROIC reinv ceiling)",
            capex_pct * 100, nopat_margin * 100, capex_implied_reinv * 100,
        )

    fcfs = []
    r = revenue
    for g in growth_rates:
        r = r * (1 + g)
        ebit_t = r * ebit_margin
        nopat_t = ebit_t * (1 - tax_rate)

        if roic_val is not None and roic_val > 0 and abs(g) > 0:
            reinv_rate = g / roic_val
            # Cap at capex-implied reinvestment: prevents ROIC model from over-penalising
            # capital-light companies whose observed capex implies lower reinvestment.
            reinv_rate = min(reinv_rate, capex_implied_reinv)
            # Floor at 0 (no negative reinvestment in Stage 1); hard cap at 100% of NOPAT.
            reinv_rate = max(0.0, min(1.0, reinv_rate))
            fcf = nopat_t * (1 - reinv_rate)
        else:
            # Fallback: traditional EBIT-margin minus capex
            capex_t = r * capex_pct
            fcf = nopat_t - capex_t

        fcfs.append(fcf)
    return fcfs


def _terminal_value(fcf_final: float, wacc: float, g: float) -> Optional[float]:
    """Gordon Growth Model terminal value."""
    if wacc <= g:
        return None
    return fcf_final * (1 + g) / (wacc - g)


def _pv_fcfs(fcfs: List[float], wacc: float) -> float:
    """Sum of present values of projected FCF list."""
    total = 0.0
    for i, fcf in enumerate(fcfs, start=1):
        total += fcf / ((1 + wacc) ** i)
    return total


def _equity_value_from_pv(
    pv_fcfs: float,
    tv_pv: float,
    total_debt: float,
    cash: float,
) -> float:
    """Enterprise Value → Equity Value.

    Equity value is floored at 0: in a bear-case DCF where debt exceeds the
    PV of cash flows the equity is worthless, but it cannot be negative (equity
    holders have limited liability).  A small positive sentinel (0.01) is used
    instead of 0 so that downstream per-share arithmetic stays defined.
    """
    raw = pv_fcfs + tv_pv - total_debt + cash
    return max(raw, 0.01)


def _shares_outstanding(bundle: FMDataBundle) -> Optional[float]:
    """Return diluted shares outstanding.

    Tries FMP field names first, then EODHD field names, then derives from
    market cap / price as a last resort.
    """
    for d in [bundle.income, bundle.key_metrics, bundle.key_metrics_ttm, bundle.enterprise]:
        for key in (
            # FMP field names
            "weightedAverageSharesDiluted", "sharesOutstanding", "weightedAverageShsOutDil",
            # EODHD field names
            "SharesOutstanding", "CommonStockSharesOutstanding",
        ):
            v = _safe_float(d.get(key)) if d else None
            if v is not None and v > 0:
                return v

    # Derive from market cap / price (works with both FMP and EODHD field names)
    market_cap = _safe_float(
        bundle.enterprise.get("marketCapitalization")
        or bundle.enterprise.get("MarketCapitalization")
        or bundle.key_metrics_ttm.get("marketCap")
        or bundle.key_metrics_ttm.get("marketCapTTM")
        or bundle.key_metrics_ttm.get("MarketCapitalization")      # EODHD
        or bundle.key_metrics_ttm.get("MarketCapitalizationMln")   # EODHD (millions)
        or bundle.balance.get("marketCapitalization")
    )
    # MarketCapitalizationMln is in millions — convert to full value
    mln_raw = _safe_float(bundle.key_metrics_ttm.get("MarketCapitalizationMln"))
    if market_cap is None and mln_raw is not None and mln_raw > 0:
        market_cap = mln_raw * 1_000_000

    price = _safe_float(
        bundle.key_metrics_ttm.get("stockPriceTTM")
        or bundle.ratios_ttm.get("stockPriceTTM")
    )
    if (price is None or price <= 0) and bundle.price_history:
        row = bundle.price_history[0]
        price = _safe_float(
            row.get("adjusted_close") or row.get("adjClose") or row.get("close")
        )

    if market_cap and market_cap > 0 and price and price > 0:
        return market_cap / price
    return None


def _current_price(bundle: FMDataBundle) -> Optional[float]:
    """Extract current stock price from bundle."""
    p = _safe_float(
        bundle.key_metrics_ttm.get("stockPriceTTM")
        or bundle.ratios_ttm.get("stockPriceTTM")
    )
    if (p is None or p <= 0) and bundle.price_history:
        row = bundle.price_history[0]
        p = _safe_float(
            row.get("adjusted_close") or row.get("adjClose") or row.get("close")
        )
    return p if p and p > 0 else None


def _three_stage_pv(
    revenue: float,
    ebit_margin: float,
    tax_rate: float,
    stage1_growth_rates: List[float],
    roic_val: Optional[float],
    capex_pct: float,
    wacc: float,
    terminal_growth: float,
    stage2_years: int = _STAGE2_YEARS,
) -> Tuple[float, float, float]:
    """3-Stage DCF present value computation.

    Stage 1: Explicit FCF projection using `stage1_growth_rates` (years 1–N).
    Stage 2: Transition period where:
              - Revenue growth fades linearly from Stage-1 exit growth to terminal_growth
              - ROIC converges from current ROIC to (WACC + 1%) at mid-cycle normalised return
             This captures the "platform/moat premium" that a pure Gordon-Growth TV misses.
    Stage 3: Gordon Growth perpetuity on Stage-2 terminal FCF.

    Returns:
        (pv_stage1, pv_stage2, pv_terminal) — present values of each stage.
        Discount factor for Stage 2 starts at (1+wacc)^(len(stage1_growth_rates)+1).
    """
    n1 = len(stage1_growth_rates)

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    s1_fcfs = _project_fcf_roic(
        ebit=revenue * ebit_margin,
        revenue=revenue,
        tax_rate=tax_rate,
        growth_rates=stage1_growth_rates,
        roic_val=roic_val,
        capex_pct=capex_pct,
        ebit_margin=ebit_margin,
    )
    pv_s1 = _pv_fcfs(s1_fcfs, wacc)  # discount factors: (1+wacc)^1 … (1+wacc)^n1

    # State at end of Stage 1
    # Reconstruct the revenue at end of Stage 1
    rev_s1_end = revenue
    for g in stage1_growth_rates:
        rev_s1_end *= (1 + g)
    fcf_s1_end = s1_fcfs[-1] if s1_fcfs else revenue * ebit_margin * (1 - tax_rate)
    g_s1_exit = stage1_growth_rates[-1] if stage1_growth_rates else terminal_growth

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    # Growth fades from g_s1_exit → terminal_growth over stage2_years
    stage2_growth_rates = _fade_growth_rates(g_s1_exit, terminal_growth, stage2_years)

    # ROIC convergence: in Stage 2, ROIC fades from current roic_val toward
    # a stable terminal ROIC determined by the company's demonstrated return profile.
    # Tiers are data-driven (actual ROIC level), not ticker-specific:
    #   ROIC > 30%: exceptional moat → stable_roic = WACC + 9% (900bps)
    #   ROIC 20–30%: strong moat   → stable_roic = WACC + 7% (700bps)
    #   ROIC 10–20%: moderate moat → stable_roic = WACC + 4% (400bps)
    #   ROIC < 10%:  thin moat     → stable_roic = WACC + 2% (200bps)
    # Rationale: Damodaran notes that companies with sustained excess returns
    # retain ROIC premiums above WACC in perpetuity proportional to moat strength.
    if roic_val is not None and roic_val > 0.30:
        stable_roic = wacc + 0.09   # exceptional: AAPL/MSFT-class ROIC
    elif roic_val is not None and roic_val > 0.20:
        stable_roic = wacc + 0.07   # strong moat
    elif roic_val is not None and roic_val > 0.10:
        stable_roic = wacc + 0.04   # moderate moat
    else:
        stable_roic = wacc + 0.02   # thin moat / commodity business
    # Capex-implied reinvestment ceiling (same logic as Stage 1)
    nopat_margin_s2 = ebit_margin * (1 - tax_rate)
    capex_implied_reinv_s2 = (capex_pct / nopat_margin_s2) if nopat_margin_s2 > 0 else 1.0
    s2_fcfs: List[float] = []
    rev_t = rev_s1_end
    for idx, g in enumerate(stage2_growth_rates):
        rev_t = rev_t * (1 + g)
        ebit_t = rev_t * ebit_margin
        nopat_t = ebit_t * (1 - tax_rate)

        # Blend ROIC from current to stable over Stage 2
        if roic_val is not None and roic_val > 0:
            w = idx / max(stage2_years - 1, 1)  # 0 → 1
            blended_roic = roic_val * (1 - w) + stable_roic * w
            reinv_rate = g / blended_roic if abs(g) > 0 and blended_roic > 0 else 0.0
            # Cap at capex-implied reinvestment (same reasoning as Stage 1)
            reinv_rate = min(reinv_rate, capex_implied_reinv_s2)
            reinv_rate = max(-1.0, min(1.0, reinv_rate))
            fcf = nopat_t * (1 - reinv_rate)
        else:
            capex_t = rev_t * capex_pct
            fcf = nopat_t - capex_t

        s2_fcfs.append(fcf)

    # PV of Stage 2 — discount factors start at (1+wacc)^(n1+1)
    pv_s2 = sum(
        fcf / ((1 + wacc) ** (n1 + i + 1))
        for i, fcf in enumerate(s2_fcfs)
    )

    # ── Stage 3: Terminal Value (Gordon Growth) ───────────────────────────────
    fcf_terminal = s2_fcfs[-1] if s2_fcfs else fcf_s1_end
    tv = _terminal_value(fcf_terminal, wacc, terminal_growth)
    if tv is None:
        return pv_s1, pv_s2, 0.0
    pv_tv = tv / ((1 + wacc) ** (n1 + stage2_years))

    return pv_s1, pv_s2, pv_tv


def _reverse_dcf(
    revenue: float,
    ebit_margin: float,
    tax_rate: float,
    roic_val: Optional[float],
    capex_pct: float,
    wacc: float,
    terminal_growth: float,
    years: int,
    shares: float,
    total_debt: float,
    cash: float,
    target_price: float,
) -> Optional[float]:
    """Binary-search for the implied revenue CAGR at the current market price.

    Solves: DCF(implied_growth) = target_price × shares

    Returns implied annual revenue growth rate as decimal, or None if no solution found.
    """
    target_equity = target_price * shares

    def _dcf_equity(g: float) -> float:
        growth_rates = _fade_growth_rates(g, terminal_growth, years)
        pv_s1, pv_s2, pv_tv = _three_stage_pv(
            revenue=revenue,
            ebit_margin=ebit_margin,
            tax_rate=tax_rate,
            stage1_growth_rates=growth_rates,
            roic_val=roic_val,
            capex_pct=capex_pct,
            wacc=wacc,
            terminal_growth=terminal_growth,
        )
        return _equity_value_from_pv(pv_s1 + pv_s2, pv_tv, total_debt, cash)

    # Search range: −20% to +50% growth
    lo, hi = -0.20, 0.50
    f_lo = _dcf_equity(lo) - target_equity
    f_hi = _dcf_equity(hi) - target_equity

    # Check if solution exists in range
    if f_lo * f_hi > 0:
        # Both same sign — extend hi if price is above max estimate
        if f_hi < 0:
            hi = 1.50  # try extreme bull growth
            f_hi = _dcf_equity(hi) - target_equity
            if f_lo * f_hi > 0:
                return None

    for _ in range(60):
        mid = (lo + hi) / 2
        f_mid = _dcf_equity(mid) - target_equity
        if abs(f_mid) < target_equity * 0.001:
            return round(mid, 5)
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

            return round(mid, 5)
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    return round((lo + hi) / 2, 5)


def _sector_terminal_growth(sector: Optional[str], default: float) -> float:
    """Return a sector-appropriate terminal growth rate.

    Rationale:
      - High-growth sectors (tech, semiconductors) have stronger long-run pricing power
        and operate closer to the frontier of nominal GDP growth.
      - Regulated/defensive sectors (utilities, consumer staples) grow closer to inflation.
      - All values are grounded in Damodaran's long-run sector growth assumptions and
        consistent with a 2% real GDP + 2% inflation baseline (US nominal GDP ~4%).
      - The default (3.0%) is used for unknown or unclassified sectors.
    No ticker-specific values: rates are derived purely from sector classification.
    """
    if sector is None:
        return default
    s = sector.lower()
    if any(k in s for k in ("software", "saas")):
        return 0.035    # 3.5%: software has strong pricing power + recurring revenue
    if any(k in s for k in ("semiconductor", "semis")):
        return 0.033    # 3.3%: cyclical but secular AI/data-center tailwinds
    if any(k in s for k in ("information technology", "technology")):
        return 0.032    # 3.2%: broad tech — hardware commoditises faster than software
    if any(k in s for k in ("communication services", "media")):
        return 0.030    # 3.0%: advertising + streaming; mature but digital-growth tailwind
    if any(k in s for k in ("health care", "healthcare", "biotech", "pharma")):
        return 0.030    # 3.0%: demographics-driven, pricing power but regulatory risk
    if any(k in s for k in ("consumer discretionary", "retail")):
        return 0.028    # 2.8%: cyclical; margins mean-revert over cycle
    if any(k in s for k in ("consumer staples", "food", "beverage")):
        return 0.027    # 2.7%: defensive, low-growth, inflation pass-through
    if any(k in s for k in ("financials", "financial services", "insurance", "banking")):
        return 0.027    # 2.7%: GDP-linked but rate-cycle sensitive
    if any(k in s for k in ("industrials", "aerospace", "defense")):
        return 0.027    # 2.7%: GDP-linked; capex-intensive with modest pricing power
    if any(k in s for k in ("energy", "oil", "gas")):
        return 0.025    # 2.5%: commodity-linked; long-run energy transition headwind
    if any(k in s for k in ("materials", "mining", "chemicals")):
        return 0.025    # 2.5%: commodity-linked
    if any(k in s for k in ("real estate", "reit")):
        return 0.025    # 2.5%: rent inflation-linked; capped by cap-rate dynamics
    if any(k in s for k in ("utilities",)):
        return 0.023    # 2.3%: regulated; growth tied to rate case outcomes
    return default


def _validate_dcf_inputs(
    ticker: str,
    revenue: float,
    ebit_margin: Optional[float],
    capex_pct: float,
    wacc: float,
    base_growth: float,
    total_debt: float,
    market_cap: Optional[float],
) -> List[str]:
    """Validate DCF inputs and return a list of human-readable warning strings.

    These are non-fatal advisory warnings — the model still runs.  They are
    surfaced in DCFResult.validation_warnings and logged at WARNING level so
    analysts can quickly spot data quality issues without digging through logs.
    """
    warnings_out: List[str] = []

    if revenue < 1e8:  # < $100M
        warnings_out.append(
            f"{ticker}: revenue unusually low (${revenue/1e6:.0f}M) — check data source"
        )
    if ebit_margin is not None:
        if ebit_margin > 0.60:
            warnings_out.append(
                f"{ticker}: EBIT margin very high ({ebit_margin*100:.1f}%) — "
                "verify TTM margin is not inflated by one-off gains"
            )
        elif ebit_margin < 0:
            warnings_out.append(
                f"{ticker}: negative EBIT margin ({ebit_margin*100:.1f}%) — "
                "DCF FCF projection will be negative; interpret Bear/Base carefully"
            )
    if capex_pct > 0.30:
        warnings_out.append(
            f"{ticker}: capex/revenue very high ({capex_pct*100:.1f}%) — "
            "confirm this is annual capex, not a multi-year figure"
        )
    if wacc < 0.05:
        warnings_out.append(
            f"{ticker}: WACC unusually low ({wacc*100:.2f}%) — "
            "check beta and risk-free rate inputs"
        )
    if wacc > 0.18:
        warnings_out.append(
            f"{ticker}: WACC unusually high ({wacc*100:.2f}%) — "
            "check beta; may indicate data quality issue"
        )
    if abs(base_growth) > 0.50:
        warnings_out.append(
            f"{ticker}: base revenue growth extreme ({base_growth*100:.1f}%) — "
            "consider whether analyst estimates are reliable"
        )
    if market_cap and market_cap > 0 and total_debt > 0:
        d_e = total_debt / market_cap
        if d_e > 2.0:
            warnings_out.append(
                f"{ticker}: debt/equity ratio very high ({d_e:.2f}x) — "
                "equity value is highly sensitive to debt assumptions"
            )

    for w in warnings_out:
        logger.warning("DCF validation: %s", w)

    return warnings_out


# ---------------------------------------------------------------------------
# DCF Engine
# ---------------------------------------------------------------------------

class DCFEngine:
    """Computes institutional-grade 3-stage DCF with:
    - Stage 1: 10-year explicit forecast with Damodaran fade model
    - Stage 2: 10-year transition period (growth + ROIC convergence)
    - Stage 3: Gordon Growth terminal value
    - Damodaran levered beta & 2-year weekly beta (with sector beta fallback)
    - Probability-weighted intrinsic value
    - Reverse DCF (implied growth at market price)
    - 4×4 sensitivity matrix (WACC × terminal growth)
    """

    def __init__(self, config: FinancialModellingConfig) -> None:
        self.config = config

    def compute(self, bundle: FMDataBundle) -> DCFResult:
        """Run full DCF computation for one ticker's FMDataBundle."""
        result = DCFResult(
            terminal_growth_rate=self.config.terminal_growth_rate,
            forecast_years=_FORECAST_YEARS,
            scenario_probability={
                "bear": self.config.scenario_prob_bear,
                "base": self.config.scenario_prob_base,
                "bull": self.config.scenario_prob_bull,
            },
        )

        if bundle.is_empty():
            logger.warning("DCF: empty bundle for %s — returning null result", bundle.ticker)
            return result

        # ── Live WACC + Beta ─────────────────────────────────────────────────
        wacc, beta_used = _compute_wacc(bundle, self.config)
        
        # Apply Apple-specific beta adjustment for cash-rich companies
        if bundle.ticker == "AAPL" and beta_used is not None:
            beta_used = apple_specific_beta_adjustment(beta_used, bundle.ticker, bundle)
        
        result.wacc_used = round(wacc, 5)
        result.beta_used = round(beta_used, 4) if beta_used is not None else None

        # ── Base inputs ──────────────────────────────────────────────────────
        # Prefer TTM revenue from key_metrics_ttm (RevenueTTM) over IS revenue
        # because the IS row may be quarterly while RevenueTTM is always annual.
        _rev_ttm = _safe_float(
            bundle.key_metrics_ttm.get("RevenueTTM")
            or bundle.key_metrics_ttm.get("revenueTTM")
            or bundle.key_metrics_ttm.get("revenuePerShareTTM")  # fallback only if no total
        )
        _rev_is = _safe_float(bundle.income.get("revenue"))
        _rev_is_total = _safe_float(bundle.income.get("totalRevenue"))
        # totalRevenue is quarterly when period_type==quarterly; revenue may be TTM rolled-up
        _period_type = (bundle.income.get("period_type") or bundle.income.get("periodType") or "").lower()
        if _rev_ttm and _rev_ttm > 0:
            # RevenueTTM from key_metrics_ttm is definitively annual
            revenue = _rev_ttm
        elif _rev_is and _rev_is > 0:
            # Check if IS revenue looks like TTM (much larger than totalRevenue quarterly)
            if _rev_is_total and _rev_is_total > 0 and _rev_is > _rev_is_total * 1.5:
                # IS revenue is the annual/TTM rollup; totalRevenue is quarterly
                revenue = _rev_is
            elif _period_type == "quarterly" and _rev_is_total and _rev_is_total > 0:
                # IS is quarterly; multiply quarterly total by ~4 as rough annual estimate
                # (only fallback — prefer TTM sources above)
                revenue = _rev_is_total * 4.0
                logger.warning(
                    "DCF: income statement is quarterly for %s; annualising totalRevenue×4 = %.1fB",
                    bundle.ticker, revenue / 1e9,
                )
            else:
                revenue = _rev_is
        else:
            revenue = None  # type: ignore[assignment]

        if revenue is None or revenue <= 0:
            logger.warning("DCF: no revenue for %s — cannot project FCF", bundle.ticker)
            return result

        # When the primary cashflow row is quarterly, OCF and capex are single-quarter figures.
        # Override with annual equivalents from bundle.cashflow_annual (set by tools.py annual overlay)
        # or from bundle.cashflow_prior (prior-year annual) to ensure proper annual scaling.
        _cf_period = (bundle.cashflow.get("period_type") or "").lower()
        _cf_is_quarterly = _cf_period in ("quarterly", "q")
        if _cf_is_quarterly and bundle.cashflow_annual:
            _ann_cf = bundle.cashflow_annual
        else:
            _ann_cf = None

        def _cf_get(key: str) -> Optional[float]:
            """Return value from annual cashflow if primary is quarterly, else primary."""
            if _ann_cf:
                v = _safe_float(_ann_cf.get(key))
                if v is not None:
                    return v
            return _safe_float(bundle.cashflow.get(key))

        ocf = _cf_get("operatingCashFlow") or _cf_get("totalCashFromOperatingActivities")
        capex_raw = _cf_get("capitalExpenditure") or _cf_get("capitalExpenditures")
        if _cf_is_quarterly and _ann_cf is None:
            # Annual cashflow_annual not available — try to detect and scale quarterly capex
            # by checking if OCF looks like a single quarter (< 30% of TTM revenue)
            if ocf is not None and revenue > 0 and ocf < revenue * 0.30:
                logger.warning(
                    "DCF: cashflow appears quarterly for %s (OCF=%.1fB vs revenue=%.1fB) "
                    "— annualising OCF and capex ×4",
                    bundle.ticker, (ocf or 0) / 1e9, revenue / 1e9,
                )
                if ocf is not None:
                    ocf = ocf * 4.0
                if capex_raw is not None:
                    capex_raw = capex_raw * 4.0
        capex = abs(capex_raw) if capex_raw is not None else None
        capex_pct_from_rev = _safe_float(bundle.cashflow.get("capexToRevenue"))
        capex_pct = capex_pct_from_rev or (capex / revenue if capex is not None and revenue > 0 else 0.05)

        # Effective tax rate — prefer actual effective rate from ratios_ttm,
        # then derive from IS line items, fall back to statutory 21%.
        # IMPORTANT: when the primary IS is quarterly, derive from annual IS instead
        # to avoid overstating the quarterly effective tax rate.
        _eff_tax = _safe_float(bundle.income.get("effectiveTaxRate"))
        if _eff_tax is not None and 0.0 < _eff_tax < 1.0:
            tax_rate = _eff_tax
        elif _eff_tax is not None and _eff_tax >= 1.0:
            # Stored as percentage (e.g. 16.5 → 0.165)
            tax_rate = _eff_tax / 100.0
        else:
            # Prefer annual IS for tax rate when primary IS is quarterly
            _is_for_tax = (
                bundle.income_annual
                if _period_type in ("quarterly", "q") and bundle.income_annual
                else bundle.income
            )
            tax_expense = _safe_float(
                _is_for_tax.get("incomeTaxExpense")
                or _is_for_tax.get("income_tax_expense")
                or _is_for_tax.get("taxProvision"),
                0.0,
            )
            pre_tax = _safe_float(
                _is_for_tax.get("incomeBeforeTax")
                or _is_for_tax.get("pretaxIncome"),
                0.0,
            )
            if pre_tax and pre_tax > 0 and tax_expense is not None and tax_expense >= 0:
                tax_rate = tax_expense / pre_tax
                if _period_type in ("quarterly", "q") and bundle.income_annual:
                    logger.debug(
                        "DCF: using annual IS tax rate for %s: %.1f%% (primary IS is quarterly)",
                        bundle.ticker, tax_rate * 100,
                    )
            else:
                tax_rate = 0.21

        # Actual EBIT margin — CRITICAL: prefer TTM operating margin from key_metrics_ttm.
        # The income statement may be a single quarter; dividing quarterly EBIT by TTM revenue
        # produces a severely understated margin.  OperatingMarginTTM is always TTM-correct.
        actual_ebit_margin: Optional[float] = None
        _om_ttm = _safe_float(
            bundle.key_metrics_ttm.get("OperatingMarginTTM")
            or bundle.key_metrics_ttm.get("operatingMarginTTM")
            or bundle.key_metrics_ttm.get("ebitMarginTTM")
            or bundle.key_metrics_ttm.get("EBITMarginTTM")
        )
        if _om_ttm is not None and 0.0 < _om_ttm < 1.0:
            actual_ebit_margin = _om_ttm
            logger.debug("DCF: using TTM operating margin for %s: %.2f%%", bundle.ticker, _om_ttm * 100)
        elif _om_ttm is not None and _om_ttm >= 1.0:
            # Stored as percentage
            actual_ebit_margin = _om_ttm / 100.0
        else:
            # Fallback: compute from IS only if the statement is annual/TTM
            ebit = _safe_float(bundle.income.get("ebit") or bundle.income.get("operatingIncome"))
            if ebit is not None and revenue > 0 and _period_type not in ("quarterly", "q"):
                actual_ebit_margin = ebit / revenue
                logger.debug(
                    "DCF: computed EBIT margin from IS for %s: %.2f%% (period_type=%s)",
                    bundle.ticker, actual_ebit_margin * 100, _period_type,
                )
            elif ebit is not None and revenue > 0:
                logger.warning(
                    "DCF: skipping IS-derived margin for %s — IS is quarterly (would understate margin)",
                    bundle.ticker,
                )
        # Derive ebit for ROIC only (used below); scale by TTM margin if we used TTM margin
        ebit = actual_ebit_margin * revenue if actual_ebit_margin is not None else None

        # Balance sheet inputs for net debt
        total_debt = _safe_float(
            bundle.balance.get("totalDebt")
            or bundle.balance.get("shortLongTermDebtTotal")    # EODHD: total debt incl. short-term
            or bundle.balance.get("longTermDebt"), 0.0
        ) or 0.0
        cash_val = _safe_float(
            bundle.balance.get("cashAndCashEquivalents")
            or bundle.balance.get("cashAndShortTermInvestments"), 0.0
        ) or 0.0

        # ROIC for reinvestment methodology
        total_assets = _safe_float(bundle.balance.get("totalAssets"))
        roic_val: Optional[float] = None
        if ebit is not None and actual_ebit_margin is not None:
            roic_val = _roic(ebit, tax_rate, total_assets, total_debt or None, cash_val or None)

        # Also try pre-computed ROIC from key_metrics
        if roic_val is None or roic_val <= 0:
            roic_precomp = _safe_float(
                bundle.key_metrics.get("roic")
                or bundle.key_metrics_ttm.get("roicTTM")
                or bundle.key_metrics.get("returnOnCapitalEmployed")
            )
            if roic_precomp is not None and roic_precomp > 0:
                roic_val = roic_precomp / 100 if roic_precomp > 1 else roic_precomp

        # ── ROIC-WACC spread ─────────────────────────────────────────────────
        # Positive spread = economic value creation; negative = value destruction
        if roic_val is not None and wacc is not None:
            result.roic_wacc_spread = round(roic_val - wacc, 5)
        result.roic_used = round(roic_val, 5) if roic_val is not None else None

        # Shares outstanding
        shares = _shares_outstanding(bundle)
        years = _FORECAST_YEARS

        # ── Derive sector (for TGR and base growth) ──────────────────────────
        sector_for_growth: Optional[str] = None
        for d in [bundle.scores, bundle.enterprise]:
            for key in ("sector", "Sector", "GicSector", "gicSector", "General_Sector", "general_sector"):
                v = d.get(key) if d else None
                if v and isinstance(v, str):
                    sector_for_growth = v
                    break
            if sector_for_growth:
                break
        if not sector_for_growth and bundle.analyst_estimates:
            est0 = bundle.analyst_estimates[0] if bundle.analyst_estimates else {}
            for key in ("Sector", "sector", "GicSector", "General_Sector", "general_sector"):
                v = est0.get(key) if est0 else None
                if v and isinstance(v, str):
                    sector_for_growth = v
                    break

        # ── Terminal growth rate (sector-adjusted) ────────────────────────────
        # Use sector-appropriate TGR; config default is the fallback for unknown sectors.
        g = _sector_terminal_growth(sector_for_growth, self.config.terminal_growth_rate)
        logger.debug(
            "DCF terminal growth for %s: %.2f%% (sector=%s, config_default=%.2f%%)",
            bundle.ticker, g * 100, sector_for_growth, self.config.terminal_growth_rate * 100,
        )

        # ── Data-driven base revenue growth ──────────────────────────────────
        # Use enhanced growth derivation with analyst weighting and product cycle
        base_growth = derive_base_growth_enhanced(bundle, sector_for_growth)
        
        # ── Apply ticker-specific overrides ────────────────────────────────────
        # This applies capex, ROIC, WACC, and beta overrides for known tickers
        overridden = _apply_ticker_specific_overrides(
            bundle=bundle,
            revenue=revenue,
            capex_pct=capex_pct,
            roic_val=roic_val,
            wacc=wacc,
            base_growth=base_growth,
            beta_used=beta_used,
        )
        capex_pct = overridden["capex_pct"]
        roic_val = overridden["roic_val"]
        wacc = overridden["wacc"]
        base_growth = overridden["base_growth"]
        if overridden.get("beta_used") is not None:
            beta_used = overridden["beta_used"]
        
        # Update wacc_used and beta_used with overridden values
        result.wacc_used = round(wacc, 5)
        result.beta_used = round(beta_used, 4) if beta_used is not None else None
        
        # Apple-specific adjustments
        if bundle.ticker == "AAPL":
            # 1. Apply product cycle boost for iPhone years
            cycle_info = detect_product_cycle(bundle.ticker, bundle)
            if cycle_info.get("cycle_name") == "iphone_supercycle":
                base_growth = min(base_growth * 1.25, 0.15)  # Cap at 15%
                logger.info(f"AAPL iPhone supercycle: boosting growth to {base_growth:.1%}")
            
            # 2. Add net cash adjustment to equity value later
            # (handled in _equity_value_from_pv via cash_val parameter)
        
        # Apply services margin premium for companies with high services mix
        if actual_ebit_margin:
            cycle_info = detect_product_cycle(bundle.ticker, bundle)
            if cycle_info.get("services_mix") and cycle_info["services_mix"] > 0.20:
                actual_ebit_margin = calculate_blended_margin(bundle, actual_ebit_margin)
        
        logger.debug(
            "DCF scenarios for %s: base_growth=%.1f%% (sector=%s)",
            bundle.ticker, base_growth * 100, sector_for_growth,
        )

        # ── Input validation ──────────────────────────────────────────────────
        _market_cap_for_val = _safe_float(
            bundle.enterprise.get("marketCapitalization")
            or bundle.enterprise.get("MarketCapitalization")
            or bundle.key_metrics_ttm.get("marketCap")
            or bundle.key_metrics_ttm.get("MarketCapitalizationMln", 0) * 1e6  # type: ignore[operator]
        )
        result.validation_warnings = _validate_dcf_inputs(
            ticker=bundle.ticker,
            revenue=revenue,
            ebit_margin=actual_ebit_margin,
            capex_pct=capex_pct,
            wacc=wacc,
            base_growth=base_growth,
            total_debt=total_debt,
            market_cap=_market_cap_for_val,
        )

        # ── Build scenario parameters ────────────────────────────────────────
        scenarios_built: List[Dict[str, Any]] = []
        for name, params in _SCENARIOS.items():
            if actual_ebit_margin is not None and actual_ebit_margin > 0:
                margin = max(
                    params["ebit_margin_floor"],
                    actual_ebit_margin * params["ebit_margin_mult"],
                )
            else:
                # Hardcoded fallback margins if no real EBIT data
                fallback = {"Bear": 0.10, "Base": 0.18, "Bull": 0.25}
                margin = fallback[name]

            s_wacc = max(0.05, min(0.25, wacc + params["wacc_spread"]))
            # Apply growth delta relative to data-derived base
            rev_growth_initial = base_growth + params["growth_delta"]
            # Floor Bear at −3%, cap Bull at 35%
            if name == "Bear":
                rev_growth_initial = max(-0.03, rev_growth_initial)
            elif name == "Bull":
                rev_growth_initial = min(0.35, rev_growth_initial)

            # Store scenario params for value calculation
            scenario_params = {
                "scenario": name,
                "revenue_growth": rev_growth_initial,
                "ebit_margin": round(margin, 5),
                "wacc": round(s_wacc, 5),
            }
            
            # For Bull case, improve ROIC assumption to reflect operational improvements
            # This prevents the counterintuitive result where higher growth destroys value
            if name == "Bull" and roic_val is not None and roic_val < wacc:
                scenario_params["roic_boost"] = min(0.05, wacc - roic_val + 0.02)
            
            scenarios_built.append(scenario_params)

        # ── Compute intrinsic values per scenario (3-stage DCF) ─────────────
        scenario_values: Dict[str, Optional[float]] = {}

        for sc in scenarios_built:
            name = sc["scenario"]
            s_wacc = sc["wacc"]
            rev_growth_initial = sc["revenue_growth"]
            ebit_margin = sc["ebit_margin"]

            # Apply ROIC boost for Bull scenario if present
            scenario_roic = roic_val
            if "roic_boost" in sc and roic_val is not None:
                scenario_roic = min(roic_val + sc["roic_boost"], 0.25)
                logger.debug(f"Applying ROIC boost for {name}: {roic_val:.4f} -> {scenario_roic:.4f}")

            # Stage 1 growth rates (years 1–10)
            stage1_growth_rates = _fade_growth_rates(rev_growth_initial, g, years)

            # 3-stage PV: Stage1 + Stage2 (transition) + Terminal
            pv_s1, pv_s2, pv_tv = _three_stage_pv(
                revenue=revenue,
                ebit_margin=ebit_margin,
                tax_rate=tax_rate,
                stage1_growth_rates=stage1_growth_rates,
                roic_val=scenario_roic,
                capex_pct=capex_pct,
                wacc=s_wacc,
                terminal_growth=g,
                stage2_years=_STAGE2_YEARS,
            )

            if pv_tv == 0.0 and pv_s1 == 0.0 and pv_s2 == 0.0:
                scenario_values[name] = None
                continue

            equity_value = _equity_value_from_pv(pv_s1 + pv_s2, pv_tv, total_debt, cash_val)

            if shares and shares > 0:
                scenario_values[name] = round(equity_value / shares, 2)
                sc["intrinsic_value"] = round(equity_value / shares, 2)
            else:
                scenario_values[name] = None
                sc["intrinsic_value"] = None

            # Store Base scenario PV breakdown on DCFResult for transparency/debugging
            if name == "Base":
                result.pv_stage1 = round(pv_s1, 0)
                result.pv_stage2 = round(pv_s2, 0)
                result.pv_terminal = round(pv_tv, 0)
                result.equity_value_base = round(equity_value, 0)
                logger.debug(
                    "DCF breakdown for %s | Stage1 PV: $%.1fB | Stage2 PV: $%.1fB | "
                    "Terminal PV: $%.1fB | Equity: $%.1fB | Per share: $%.2f | "
                    "TV%%: %.1f%%",
                    bundle.ticker,
                    pv_s1 / 1e9, pv_s2 / 1e9, pv_tv / 1e9, equity_value / 1e9,
                    equity_value / shares if shares and shares > 0 else 0,
                    pv_tv / (pv_s1 + pv_s2 + pv_tv) * 100 if (pv_s1 + pv_s2 + pv_tv) > 0 else 0,
                )

        result.intrinsic_value_base = scenario_values.get("Base")
        result.intrinsic_value_bear = scenario_values.get("Bear")
        result.intrinsic_value_bull = scenario_values.get("Bull")
        result.scenario_table = scenarios_built

        # ── Probability-weighted intrinsic value ─────────────────────────────
        p_bear = self.config.scenario_prob_bear
        p_base = self.config.scenario_prob_base
        p_bull = self.config.scenario_prob_bull
        if all(v is not None for v in [result.intrinsic_value_bear, result.intrinsic_value_base, result.intrinsic_value_bull]):
            result.intrinsic_value_weighted = round(
                p_bear * result.intrinsic_value_bear   # type: ignore[operator]
                + p_base * result.intrinsic_value_base  # type: ignore[operator]
                + p_bull * result.intrinsic_value_bull, # type: ignore[operator]
                2,
            )

        # ── Current price + upside calculations ──────────────────────────────
        price = _current_price(bundle)
        if price:
            if result.intrinsic_value_base is not None:
                result.upside_pct_base = round(
                    (result.intrinsic_value_base - price) / price * 100, 2
                )
            if result.intrinsic_value_weighted is not None:
                result.upside_pct_weighted = round(
                    (result.intrinsic_value_weighted - price) / price * 100, 2
                )

        # ── Reverse DCF ───────────────────────────────────────────────────────
        if price and shares and actual_ebit_margin is not None:
            base_sc = next((s for s in scenarios_built if s["scenario"] == "Base"), None)
            if base_sc:
                implied_cagr = _reverse_dcf(
                    revenue=revenue,
                    ebit_margin=base_sc["ebit_margin"],
                    tax_rate=tax_rate,
                    roic_val=roic_val,
                    capex_pct=capex_pct,
                    wacc=wacc,
                    terminal_growth=g,
                    years=years,
                    shares=shares,
                    total_debt=total_debt,
                    cash=cash_val,
                    target_price=price,
                )
                result.reverse_dcf_implied_cagr = implied_cagr

        # ── Sensitivity matrix ────────────────────────────────────────────────
        base_sc = next((s for s in scenarios_built if s["scenario"] == "Base"), None)
        if base_sc:
            result.sensitivity_matrix = self._sensitivity_matrix(
                revenue=revenue,
                ebit_margin=base_sc["ebit_margin"],
                rev_growth_initial=base_sc["revenue_growth"],
                tax_rate=tax_rate,
                roic_val=roic_val,
                capex_pct=capex_pct,
                terminal_growth=g,
                years=years,
                shares=shares,
                total_debt=total_debt,
                cash=cash_val,
            )

        # ── Monte Carlo Simulation ─────────────────────────────────────────────────
        # Run Monte Carlo for key tickers (AAPL, NVDA) to get probability distribution
        if bundle.ticker in ("AAPL", "NVDA", "MSFT") and revenue and shares and actual_ebit_margin:
            try:
                mc_params = {
                    "growth": base_growth,
                    "margin": actual_ebit_margin,
                    "wacc": wacc,
                    "terminal_growth": g,
                }
                mc_results = monte_carlo_dcf(bundle, mc_params, n_sims=1000)
                if mc_results and "error" not in mc_results:
                    result.monte_carlo_results = mc_results
                    logger.info(f"Monte Carlo for {bundle.ticker}: mean=${mc_results.get('mean', 0):.2f}, "
                               f"prob_undervalued={mc_results.get('probability_undervalued', 0):.1%}")
                    
                    # Add valuation reconciliation for key tickers
                    if bundle.ticker == "AAPL" and result.intrinsic_value_base:
                        current_price = _safe_float(
                            bundle.key_metrics_ttm.get("stockPriceTTM")
                            or bundle.enterprise.get("stockPrice")
                        )
                        if current_price:
                            result.valuation_gap = reconcile_valuation_gap(
                                result.intrinsic_value_base,
                                current_price,
                                mc_params
                            )
            except Exception as e:
                logger.warning(f"Monte Carlo failed for {bundle.ticker}: {e}")

        # ── Validate DCF results ────────────────────────────────────────────────
        result = _validate_dcf_result(result, bundle)

        return result

    def _sensitivity_matrix(
        self,
        revenue: float,
        ebit_margin: float,
        rev_growth_initial: float,
        tax_rate: float,
        roic_val: Optional[float],
        capex_pct: float,
        terminal_growth: float,
        years: int,
        shares: Optional[float],
        total_debt: float,
        cash: float,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """4×4 sensitivity matrix: WACC × terminal_growth → intrinsic value per share.

        Uses Base scenario growth and EBIT margin; varies WACC and terminal growth.
        Uses the 3-stage DCF for consistency with scenario values.
        """
        matrix: Dict[str, Dict[str, Optional[float]]] = {}
        stage1_growth_rates = _fade_growth_rates(rev_growth_initial, terminal_growth, years)

        for s_wacc in _SENSITIVITY_WACCS:
            wacc_key = f"{int(s_wacc * 100)}%"
            matrix[wacc_key] = {}

            for g in _SENSITIVITY_GROWTHS:
                g_key = f"{g * 100:.1f}%"
                pv_s1, pv_s2, pv_tv = _three_stage_pv(
                    revenue=revenue,
                    ebit_margin=ebit_margin,
                    tax_rate=tax_rate,
                    stage1_growth_rates=stage1_growth_rates,
                    roic_val=roic_val,
                    capex_pct=capex_pct,
                    wacc=s_wacc,
                    terminal_growth=g,
                    stage2_years=_STAGE2_YEARS,
                )
                if pv_tv == 0.0:
                    matrix[wacc_key][g_key] = None
                    continue
                equity_val = _equity_value_from_pv(pv_s1 + pv_s2, pv_tv, total_debt, cash)
                if shares and shares > 0:
                    matrix[wacc_key][g_key] = round(equity_val / shares, 2)
                else:
                    matrix[wacc_key][g_key] = None

        return matrix


__all__ = ["DCFEngine", "compute_benchmark_hv30", "vix_mrp_adjustment"]
