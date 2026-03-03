"""DCF Engine: WACC calculation, FCF projections, Terminal Value, scenario table.

All computations are deterministic Python — the LLM is never called here.

WACC formula:
    WACC = (E/V) * Re + (D/V) * Rd * (1 - T)
    Re (Cost of Equity) = Rf + β × (Rm - Rf)   [CAPM]
    Rf  = 10Y Treasury yield  (raw_timeseries)
    Rm-Rf = Market Risk Premium               (raw_fundamentals)
    β   = 60-day rolling beta vs. S&P 500     (market_eod_us)
    Rd  = Interest Expense / Total Debt       (raw_fundamentals)
    T   = Effective Tax Rate                  (raw_fundamentals)

Terminal Value (Gordon Growth Model):
    TV = FCF_n × (1 + g) / (WACC - g)

Scenarios:
    Bear: revenue_growth=-5%, ebit_margin=10%, wacc=12%
    Base: revenue_growth=+8%, ebit_margin=18%, wacc=10%
    Bull: revenue_growth=+20%, ebit_margin=25%, wacc=9%

Sensitivity matrix:
    WACC ∈ {8%, 9%, 10%, 11%}  ×  g ∈ {1.5%, 2.0%, 2.5%, 3.0%}
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from ..config import FinancialModellingConfig
from ..schema import DCFResult, FMDataBundle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scenario parameters (from README spec)
# ---------------------------------------------------------------------------

_SCENARIOS: Dict[str, Dict[str, float]] = {
    "Bear": {"revenue_growth": -0.05, "ebit_margin": 0.10, "wacc": 0.12},
    "Base": {"revenue_growth":  0.08, "ebit_margin": 0.18, "wacc": 0.10},
    "Bull": {"revenue_growth":  0.20, "ebit_margin": 0.25, "wacc": 0.09},
}

_SENSITIVITY_WACCS = [0.08, 0.09, 0.10, 0.11]
_SENSITIVITY_GROWTHS = [0.015, 0.020, 0.025, 0.030]


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


def _compute_beta(
    price_history: List[Dict[str, Any]],
    benchmark_history: List[Dict[str, Any]],
    lookback: int = 60,
) -> Optional[float]:
    """Compute rolling 60-day beta of ticker vs. S&P 500.

    Both series must have a 'close' (or 'adjClose') and 'date' key.
    Returns None if insufficient data.
    """
    def _closes(rows: List[Dict[str, Any]]) -> List[float]:
        closes = []
        for r in rows:
            c = _safe_float(r.get("adjClose") or r.get("close") or r.get("adjusted_close"))
            if c is not None and c > 0:
                closes.append(c)
        return closes

    asset_closes = _closes(price_history[-lookback - 1:] if len(price_history) > lookback + 1 else price_history)
    bench_closes = _closes(benchmark_history[-lookback - 1:] if len(benchmark_history) > lookback + 1 else benchmark_history)

    n = min(len(asset_closes), len(bench_closes)) - 1
    if n < 10:
        return None

    # Align by taking last n returns from each
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
        return None
    return round(cov / var_b, 4)


def _compute_wacc(
    bundle: FMDataBundle,
    config: FinancialModellingConfig,
) -> float:
    """Compute WACC from live CAPM inputs.

    Falls back to config.dcf_discount_rate if data is insufficient.
    """
    # 1. Risk-free rate: 10Y Treasury from raw_timeseries
    rf: float = 0.043  # sensible default (4.3%)
    if bundle.treasury_rates:
        row = bundle.treasury_rates[0]
        # FMP treasury_rates payload keys: year10, tenYear, "10Y", etc.
        for key in ("year10", "tenYear", "10Y", "ten_year", "y10"):
            v = _safe_float(row.get(key))
            if v is not None:
                # Could be in % (e.g. 4.3) or decimal (e.g. 0.043)
                rf = v / 100 if v > 1 else v
                break

    # 2. Market Risk Premium from raw_fundamentals
    mrp: float = 0.055  # sensible default (5.5%)
    if bundle.market_risk_premium:
        for key in ("marketRiskPremium", "equityRiskPremium", "market_risk_premium", "rp"):
            v = _safe_float(bundle.market_risk_premium.get(key))
            if v is not None:
                mrp = v / 100 if v > 1 else v
                break

    # 3. Beta: 60-day rolling vs S&P 500
    beta = _compute_beta(
        bundle.price_history,
        bundle.benchmark_history,
        lookback=config.beta_lookback_days,
    )
    if beta is None:
        beta = 1.0  # market-neutral fallback

    # 4. Cost of Equity (CAPM)
    re = rf + beta * mrp

    # 5. Cost of Debt: Interest Expense / Total Debt
    inc = bundle.income
    bal = bundle.balance
    _raw_interest = _safe_float(
        inc.get("interestExpense") or inc.get("interest_expense"), 0.0
    )
    interest_expense = abs(_raw_interest if _raw_interest is not None else 0.0)
    total_debt = _safe_float(
        bal.get("totalDebt") or bal.get("longTermDebt") or bal.get("longTermDebtAndCapitalLeaseObligation"), 0.0
    )
    rd = (interest_expense / total_debt) if total_debt and total_debt > 0 else 0.04

    # 6. Tax rate: effective = income_tax_expense / pre_tax_income
    tax_expense = _safe_float(inc.get("incomeTaxExpense") or inc.get("income_tax_expense"), 0.0)
    pre_tax = _safe_float(inc.get("incomeBeforeTax") or inc.get("pretaxIncome"), 0.0)
    t = (tax_expense / pre_tax) if pre_tax and pre_tax > 0 and tax_expense and tax_expense >= 0 else 0.21

    # 7. Capital structure weights
    market_cap = _safe_float(bundle.enterprise.get("marketCapitalization") or bundle.key_metrics_ttm.get("marketCapTTM"))
    if market_cap is None or market_cap <= 0:
        # Fall back to config default
        return config.dcf_discount_rate

    total_cap = market_cap + (total_debt or 0.0)
    if total_cap <= 0:
        return config.dcf_discount_rate

    e_weight = market_cap / total_cap
    d_weight = (total_debt or 0.0) / total_cap

    wacc = e_weight * re + d_weight * rd * (1 - t)

    # Sanity clamp: WACC between 5% and 25%
    wacc = max(0.05, min(0.25, wacc))
    return round(wacc, 5)


def _project_fcf(
    base_fcf: float,
    revenue_growth: float,
    ebit_margin: float,
    capex_pct: float,
    revenue: float,
    years: int,
) -> List[float]:
    """Project free cash flows for `years` periods.

    Simplified model:
      Revenue_t = Revenue_{t-1} * (1 + revenue_growth)
      EBIT_t    = Revenue_t * ebit_margin
      FCF_t     = EBIT_t * (1 - tax) - Capex_t
              where Capex_t = Revenue_t * capex_pct
    Uses 21% effective tax as default.
    """
    TAX = 0.21
    fcfs = []
    r = revenue
    for _ in range(years):
        r = r * (1 + revenue_growth)
        ebit = r * ebit_margin
        capex = r * capex_pct
        nopat = ebit * (1 - TAX)
        fcf = nopat - capex
        fcfs.append(fcf)
    return fcfs


def _terminal_value(fcf_final: float, wacc: float, g: float) -> Optional[float]:
    """Gordon Growth Model terminal value."""
    if wacc <= g:
        return None  # growth ≥ discount — undefined
    return fcf_final * (1 + g) / (wacc - g)


def _pv_fcfs(fcfs: List[float], wacc: float) -> float:
    """Sum of present values of projected FCF list."""
    total = 0.0
    for i, fcf in enumerate(fcfs, start=1):
        total += fcf / ((1 + wacc) ** i)
    return total


def _shares_outstanding(bundle: FMDataBundle) -> Optional[float]:
    """Return diluted shares outstanding."""
    for d in [bundle.income, bundle.key_metrics, bundle.key_metrics_ttm, bundle.enterprise]:
        for key in ("weightedAverageSharesDiluted", "sharesOutstanding", "weightedAverageShsOutDil"):
            v = _safe_float(d.get(key)) if d else None
            if v is not None and v > 0:
                return v

    # Derive from market cap / current price
    market_cap = _safe_float(
        bundle.enterprise.get("marketCapitalization")
        or bundle.key_metrics_ttm.get("marketCap")
        or bundle.balance.get("marketCapitalization")
    )

    # Current price: try TTM fields first, then most recent price history row
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


# ---------------------------------------------------------------------------
# DCF Engine
# ---------------------------------------------------------------------------

class DCFEngine:
    """Computes DCF intrinsic value with WACC, scenario table, and sensitivity matrix."""

    def __init__(self, config: FinancialModellingConfig) -> None:
        self.config = config

    def compute(self, bundle: FMDataBundle) -> DCFResult:
        """Run full DCF computation for one ticker's FMDataBundle."""
        result = DCFResult(
            terminal_growth_rate=self.config.dcf_terminal_growth_rate,
            scenario_probability={
                "bear": self.config.scenario_prob_bear,
                "base": self.config.scenario_prob_base,
                "bull": self.config.scenario_prob_bull,
            },
        )

        if bundle.is_empty():
            logger.warning("DCF: empty bundle for %s — returning null result", bundle.ticker)
            return result

        # ── Live WACC ────────────────────────────────────────────────────────
        wacc = _compute_wacc(bundle, self.config)
        result.wacc_used = round(wacc, 5)

        # ── Base inputs ──────────────────────────────────────────────────────
        revenue = _safe_float(bundle.income.get("revenue"))
        ocf = _safe_float(bundle.cashflow.get("operatingCashFlow"))
        capex_raw = _safe_float(bundle.cashflow.get("capitalExpenditure"))
        capex = abs(capex_raw) if capex_raw is not None else None
        net_income = _safe_float(bundle.income.get("netIncome"))

        if revenue is None or revenue <= 0:
            logger.warning("DCF: no revenue for %s — cannot project FCF", bundle.ticker)
            return result

        # FCF base
        if ocf is not None and capex is not None:
            base_fcf = ocf - capex
        elif net_income is not None:
            base_fcf = net_income * 0.9  # rough proxy
        else:
            logger.warning("DCF: insufficient cash flow data for %s", bundle.ticker)
            return result

        capex_pct = (capex / revenue) if capex is not None and revenue > 0 else 0.05

        # Shares outstanding for per-share intrinsic values
        shares = _shares_outstanding(bundle)

        years = self.config.dcf_forecast_years
        g = self.config.dcf_terminal_growth_rate

        scenario_values: Dict[str, Optional[float]] = {}

        for scenario_name, params in _SCENARIOS.items():
            s_wacc = params["wacc"]
            rev_growth = params["revenue_growth"]
            ebit_margin = params["ebit_margin"]

            fcfs = _project_fcf(base_fcf, rev_growth, ebit_margin, capex_pct, revenue, years)
            if not fcfs:
                continue

            pv_sum = _pv_fcfs(fcfs, s_wacc)
            tv = _terminal_value(fcfs[-1], s_wacc, g)
            if tv is None:
                continue

            pv_tv = tv / ((1 + s_wacc) ** years)
            enterprise_value = pv_sum + pv_tv

            # Net debt adjustment
            total_debt = _safe_float(
                bundle.balance.get("totalDebt")
                or bundle.balance.get("longTermDebt"), 0.0
            )
            cash = _safe_float(
                bundle.balance.get("cashAndCashEquivalents")
                or bundle.balance.get("cashAndShortTermInvestments"), 0.0
            )
            equity_value = enterprise_value - (total_debt or 0.0) + (cash or 0.0)

            if shares and shares > 0:
                intrinsic_per_share = equity_value / shares
                scenario_values[scenario_name] = round(intrinsic_per_share, 2)
            else:
                scenario_values[scenario_name] = None

        result.intrinsic_value_base = scenario_values.get("Base")
        result.intrinsic_value_bear = scenario_values.get("Bear")
        result.intrinsic_value_bull = scenario_values.get("Bull")

        # Current price for upside calculation
        current_price = _safe_float(
            bundle.key_metrics_ttm.get("stockPriceTTM")
            or bundle.ratios_ttm.get("stockPriceTTM")
        )
        if (current_price is None or current_price <= 0) and bundle.price_history:
            row = bundle.price_history[0]
            current_price = _safe_float(
                row.get("adjusted_close") or row.get("adjClose") or row.get("close")
            )
        if current_price and current_price > 0 and result.intrinsic_value_base is not None:
            result.upside_pct_base = round(
                (result.intrinsic_value_base - current_price) / current_price * 100, 2
            )

        # ── Sensitivity matrix ────────────────────────────────────────────────
        result.sensitivity_matrix = self._sensitivity_matrix(
            base_fcf, capex_pct, revenue, shares, years
        )

        return result

    def _sensitivity_matrix(
        self,
        base_fcf: float,
        capex_pct: float,
        revenue: float,
        shares: Optional[float],
        years: int,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """4×4 sensitivity matrix: WACC × terminal_growth → intrinsic value per share."""
        matrix: Dict[str, Dict[str, Optional[float]]] = {}

        # Use Base scenario growth/margin for sensitivity (WACC and g vary)
        rev_growth = _SCENARIOS["Base"]["revenue_growth"]
        ebit_margin = _SCENARIOS["Base"]["ebit_margin"]

        for wacc in _SENSITIVITY_WACCS:
            wacc_key = f"{int(wacc * 100)}%"
            matrix[wacc_key] = {}
            fcfs = _project_fcf(base_fcf, rev_growth, ebit_margin, capex_pct, revenue, years)
            pv_sum = _pv_fcfs(fcfs, wacc)

            for g in _SENSITIVITY_GROWTHS:
                g_key = f"{g * 100:.1f}%"
                tv = _terminal_value(fcfs[-1], wacc, g)
                if tv is None:
                    matrix[wacc_key][g_key] = None
                    continue
                pv_tv = tv / ((1 + wacc) ** years)
                equity_val = pv_sum + pv_tv
                if shares and shares > 0:
                    matrix[wacc_key][g_key] = round(equity_val / shares, 2)
                else:
                    matrix[wacc_key][g_key] = None

        return matrix


__all__ = ["DCFEngine"]
