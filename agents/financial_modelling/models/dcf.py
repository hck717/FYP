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
from typing import Any, Dict, List, Optional, Tuple

from ..config import FinancialModellingConfig
from ..schema import DCFResult, FMDataBundle

logger = logging.getLogger(__name__)

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
_SENSITIVITY_GROWTHS = [0.015, 0.020, 0.025, 0.030]

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

def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


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


def _derive_base_growth(bundle: "FMDataBundle", sector: Optional[str] = None) -> float:
    """Derive a data-driven base revenue growth rate for scenario construction.

    Priority:
      1. Analyst estimates: implied growth = (estimatedRevenue / currentRevenue) − 1
      2. Key metrics TTM: revenueGrowthTTM
      3. Key metrics annual: revenueGrowth / revenue3YGrowth
      4. Sector default (Damodaran consensus)
      5. 8% generic fallback

    The result is clamped to [−5%, 35%] to prevent outliers.
    """
    current_revenue = _safe_float(bundle.income.get("revenue"))

    # ── 1. Analyst estimates: forward revenue ─────────────────────────────────
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
            if -0.40 < implied < 1.0:      # sanity-check: discard extreme outliers
                logger.debug("_derive_base_growth: analyst implied %.1f%% for %s", implied * 100, bundle.ticker)
                return max(-0.05, min(0.35, implied))

    # ── 2. TTM revenue growth from key_metrics_ttm ───────────────────────────
    ttm_g = _safe_float(
        bundle.key_metrics_ttm.get("revenueGrowthTTM")
        or bundle.key_metrics_ttm.get("revenueGrowth")
    )
    if ttm_g is not None and -0.40 < ttm_g < 1.0:
        # Value may be stored as decimal (0.12) or percent (12.0)
        if abs(ttm_g) > 1:
            ttm_g /= 100.0
        logger.debug("_derive_base_growth: TTM %.1f%% for %s", ttm_g * 100, bundle.ticker)
        return max(-0.05, min(0.35, ttm_g))

    # ── 3. Annual revenue growth from key_metrics ────────────────────────────
    ann_g = _safe_float(
        bundle.key_metrics.get("revenueGrowth")
        or bundle.key_metrics.get("revenue3YGrowth")
        or bundle.key_metrics.get("revenueGrowthAnnual")
    )
    if ann_g is not None and -0.40 < ann_g < 1.0:
        if abs(ann_g) > 1:
            ann_g /= 100.0
        logger.debug("_derive_base_growth: annual key_metrics %.1f%% for %s", ann_g * 100, bundle.ticker)
        return max(-0.05, min(0.35, ann_g))

    # ── 4. Sector default ────────────────────────────────────────────────────
    g = _sector_growth_fallback(sector)
    logger.debug("_derive_base_growth: sector fallback %.1f%% for %s (sector=%s)", g * 100, bundle.ticker, sector)
    return g


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
    # 1. Risk-free rate
    rf: float = 0.043
    if bundle.treasury_rates:
        row = bundle.treasury_rates[0]
        for key in ("year10", "tenYear", "10Y", "ten_year", "y10"):
            v = _safe_float(row.get(key))
            if v is not None:
                rf = v / 100 if v > 1 else v
                break

    # 2. Market Risk Premium (VIX-adjusted if available)
    mrp: float = 0.055
    if bundle.market_risk_premium:
        v = _safe_float(bundle.market_risk_premium.get("vix_adjusted_mrp"))
        if v is not None:
            mrp = v
        else:
            for key in ("marketRiskPremium", "equityRiskPremium", "market_risk_premium", "rp"):
                v = _safe_float(bundle.market_risk_premium.get(key))
                if v is not None:
                    mrp = v / 100 if v > 1 else v
                    break

    # 3. Beta — 2-year weekly (104 data points); falls back to sector beta
    beta_raw = _compute_beta(
        bundle.price_history,
        bundle.benchmark_history,
        lookback_weeks=config.beta_lookback_days,  # repurposed field: weeks for weekly data
        sector=sector,
    )
    beta_used = beta_raw
    if beta_raw is None:
        # Final fallback: market neutral
        beta_raw = _sector_beta_fallback(sector)

    # 4. Damodaran levered beta: β_L = β_U × (1 + (1−T) × D/E)
    inc = bundle.income
    bal = bundle.balance
    # Prefer ratios_ttm effective tax rate, then derive from IS, else 21%
    _eff_tax_w = _safe_float(inc.get("effectiveTaxRate"))
    if _eff_tax_w is not None and 0.0 < _eff_tax_w < 1.0:
        t = _eff_tax_w
    elif _eff_tax_w is not None and _eff_tax_w >= 1.0:
        t = _eff_tax_w / 100.0
    else:
        tax_expense = _safe_float(inc.get("incomeTaxExpense") or inc.get("income_tax_expense"), 0.0)
        pre_tax = _safe_float(inc.get("incomeBeforeTax") or inc.get("pretaxIncome"), 0.0)
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

    # 6. Cost of debt
    _raw_interest = _safe_float(
        inc.get("interestExpense") or inc.get("interest_expense"), 0.0
    )
    interest_expense = abs(_raw_interest if _raw_interest is not None else 0.0)
    rd = (interest_expense / total_debt) if total_debt and total_debt > 0 else 0.04

    # 7. Capital structure weights
    if market_cap is None or market_cap <= 0:
        return config.dcf_discount_rate, None

    total_cap = market_cap + (total_debt or 0.0)
    if total_cap <= 0:
        return config.dcf_discount_rate, None

    e_weight = market_cap / total_cap
    d_weight = (total_debt or 0.0) / total_cap

    wacc = e_weight * re + d_weight * rd * (1 - t)
    wacc = max(0.05, min(0.25, wacc))
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

    Falls back to EBIT-margin / capex method if ROIC is unavailable or reinv_rate > 1.

    Args:
        ebit:        Current EBIT (unused in projection — computed from margin × revenue)
        revenue:     Base year revenue
        tax_rate:    Effective corporate tax rate
        growth_rates: Per-year revenue growth rates (fade model)
        roic_val:    ROIC as decimal; None → fallback method
        capex_pct:   Capex as % of revenue (for fallback method)
        ebit_margin: EBIT / Revenue for this scenario
    """
    fcfs = []
    r = revenue
    for g in growth_rates:
        r = r * (1 + g)
        ebit_t = r * ebit_margin
        nopat_t = ebit_t * (1 - tax_rate)

        if roic_val is not None and roic_val > 0 and abs(g) > 0:
            reinv_rate = g / roic_val
            # Cap reinvestment rate: cannot reinvest more than 100% of NOPAT
            reinv_rate = max(-1.0, min(1.0, reinv_rate))
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
    """Enterprise Value → Equity Value."""
    return pv_fcfs + tv_pv - total_debt + cash


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
    # (wacc + 0.02) — a mature company earns a modest premium over cost of capital.
    # This is the Damodaran "stable growth" ROIC assumption.
    stable_roic = wacc + 0.05  # mature wide-moat ROIC ≈ WACC + 500bps (Damodaran: durable advantage)
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

    return round((lo + hi) / 2, 5)


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
        result.wacc_used = round(wacc, 5)
        result.beta_used = round(beta_used, 4) if beta_used is not None else None

        # ── Base inputs ──────────────────────────────────────────────────────
        revenue = _safe_float(bundle.income.get("revenue"))
        if revenue is None or revenue <= 0:
            logger.warning("DCF: no revenue for %s — cannot project FCF", bundle.ticker)
            return result

        ocf = _safe_float(bundle.cashflow.get("operatingCashFlow"))
        capex_raw = _safe_float(bundle.cashflow.get("capitalExpenditure"))
        capex = abs(capex_raw) if capex_raw is not None else None
        capex_pct_from_rev = _safe_float(bundle.cashflow.get("capexToRevenue"))
        capex_pct = capex_pct_from_rev or (capex / revenue if capex is not None and revenue > 0 else 0.05)

        # Effective tax rate — prefer actual effective rate from ratios_ttm,
        # then derive from IS line items, fall back to statutory 21%.
        _eff_tax = _safe_float(bundle.income.get("effectiveTaxRate"))
        if _eff_tax is not None and 0.0 < _eff_tax < 1.0:
            tax_rate = _eff_tax
        elif _eff_tax is not None and _eff_tax >= 1.0:
            # Stored as percentage (e.g. 16.5 → 0.165)
            tax_rate = _eff_tax / 100.0
        else:
            tax_expense = _safe_float(bundle.income.get("incomeTaxExpense") or bundle.income.get("income_tax_expense"), 0.0)
            pre_tax = _safe_float(bundle.income.get("incomeBeforeTax") or bundle.income.get("pretaxIncome"), 0.0)
            tax_rate = (tax_expense / pre_tax) if pre_tax and pre_tax > 0 and tax_expense and tax_expense >= 0 else 0.21

        # Actual EBIT margin
        ebit = _safe_float(bundle.income.get("ebit") or bundle.income.get("operatingIncome"))
        actual_ebit_margin: Optional[float] = None
        if ebit is not None and revenue > 0:
            actual_ebit_margin = ebit / revenue

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

        # Shares outstanding
        shares = _shares_outstanding(bundle)
        g = self.config.terminal_growth_rate
        years = _FORECAST_YEARS

        # ── Derive sector (for base growth fallback) ─────────────────────────
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

        # ── Data-driven base revenue growth ──────────────────────────────────
        # Derive from analyst consensus or TTM growth; fall back to sector default.
        base_growth = _derive_base_growth(bundle, sector_for_growth)
        logger.debug(
            "DCF scenarios for %s: base_growth=%.1f%% (sector=%s)",
            bundle.ticker, base_growth * 100, sector_for_growth,
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

            scenarios_built.append({
                "scenario": name,
                "revenue_growth": rev_growth_initial,
                "ebit_margin": round(margin, 5),
                "wacc": round(s_wacc, 5),
            })

        # ── Compute intrinsic values per scenario (3-stage DCF) ─────────────
        scenario_values: Dict[str, Optional[float]] = {}

        for sc in scenarios_built:
            name = sc["scenario"]
            s_wacc = sc["wacc"]
            rev_growth_initial = sc["revenue_growth"]
            ebit_margin = sc["ebit_margin"]

            # Stage 1 growth rates (years 1–10)
            stage1_growth_rates = _fade_growth_rates(rev_growth_initial, g, years)

            # 3-stage PV: Stage1 + Stage2 (transition) + Terminal
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
