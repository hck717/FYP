"""Quality factor calculations: ROE, ROIC, Piotroski F-Score, Beneish M-Score.

All calculations are derived from the FinancialsBundle (PostgreSQL data).
The FMP financial_scores payload may contain pre-computed scores for cross-validation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..schema import FinancialsBundle, QualityFactors
from ..tools import _safe_div, _safe_float

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ROE and ROIC
# ---------------------------------------------------------------------------

def _compute_roe(inc: Dict, bal: Dict, rt: Dict, km_ttm: Dict) -> Optional[float]:
    """Return on Equity = Net Income / Average Shareholders' Equity."""
    # Prefer pre-computed TTM value (FMP key: returnOnEquityTTM in key_metrics_ttm)
    roe_raw = (
        rt.get("returnOnEquityTTM")
        or km_ttm.get("returnOnEquityTTM")
        or km_ttm.get("roeTTM")
    )
    if roe_raw is not None:
        v = _safe_float(roe_raw)
        return round(v, 4) if v is not None else None

    # Derive
    net_income = _safe_float(inc.get("netIncome"))
    equity = _safe_float(
        bal.get("totalStockholdersEquity")
        or bal.get("totalShareholdersEquity")
        or bal.get("stockholdersEquity")
    )
    result = _safe_div(net_income, equity)
    return round(result, 4) if result is not None else None


def _compute_roic(inc: Dict, bal: Dict, cf: Dict, rt: Dict, km_ttm: Dict) -> Optional[float]:
    """Return on Invested Capital = NOPAT / Invested Capital.

    NOPAT = EBIT * (1 - tax_rate)
    Invested Capital = Total Assets - Current Liabilities - Excess Cash
    """
    # Prefer pre-computed TTM
    roic_raw = (
        km_ttm.get("returnOnInvestedCapitalTTM")
        or km_ttm.get("roicTTM")
        or rt.get("returnOnCapitalEmployedTTM")
    )
    if roic_raw is not None:
        v = _safe_float(roic_raw)
        return round(v, 4) if v is not None else None

    # Derive
    ebit = _safe_float(inc.get("operatingIncome") or inc.get("ebit"))
    income_before_tax = _safe_float(inc.get("incomeBeforeTax"))
    income_tax = _safe_float(inc.get("incomeTaxExpense"))
    total_assets = _safe_float(bal.get("totalAssets"))
    current_liabilities = _safe_float(
        bal.get("totalCurrentLiabilities")
        or bal.get("currentLiabilities")
    )
    long_term_debt = _safe_float(
        bal.get("longTermDebt")
        or bal.get("longTermDebtAndCapitalLeaseObligation")
    )
    equity = _safe_float(
        bal.get("totalStockholdersEquity")
        or bal.get("totalShareholdersEquity")
    )

    if ebit is None:
        return None

    # Tax rate
    tax_rate = 0.21  # default US corporate
    if income_before_tax and income_tax:
        computed_rate = income_tax / income_before_tax
        if 0 < computed_rate < 0.6:
            tax_rate = computed_rate

    nopat = ebit * (1 - tax_rate)

    # Invested capital = LT debt + equity (book-value approach)
    if long_term_debt is not None and equity is not None:
        invested_capital = long_term_debt + equity
    elif total_assets is not None and current_liabilities is not None:
        invested_capital = total_assets - current_liabilities
    else:
        return None

    roic = _safe_div(nopat, invested_capital)
    return round(roic, 4) if roic is not None else None


# ---------------------------------------------------------------------------
# Piotroski F-Score (9 signals)
# ---------------------------------------------------------------------------

def compute_piotroski(
    inc: Dict,
    bal: Dict,
    cf: Dict,
    inc_prev: Optional[Dict] = None,
    bal_prev: Optional[Dict] = None,
) -> Optional[int]:
    """Compute the Piotroski F-Score (0–9).

    Requires at least current year data. Prior year data enables 4 additional signals.
    Returns None if insufficient data.
    """
    score = 0
    signals_computed = 0

    def _get(d: Dict, *keys) -> Optional[float]:
        for k in keys:
            v = _safe_float(d.get(k))
            if v is not None:
                return v
        return None

    total_assets = _get(bal, "totalAssets")
    if total_assets is None or total_assets <= 0:
        return None  # Cannot compute without total assets

    # Signal 1: ROA > 0 (Net Income / Total Assets > 0)
    net_income = _get(inc, "netIncome")
    if net_income is not None:
        roa = net_income / total_assets
        if roa > 0:
            score += 1
        signals_computed += 1

    # Signal 2: Operating Cash Flow > 0
    ocf = _get(cf, "operatingCashFlow", "netCashProvidedByOperatingActivities")
    if ocf is not None:
        if ocf > 0:
            score += 1
        signals_computed += 1

    # Signal 3: ROA improving YoY (requires prev year)
    if inc_prev and bal_prev:
        net_income_prev = _get(inc_prev, "netIncome")
        total_assets_prev = _get(bal_prev, "totalAssets")
        if (net_income is not None and total_assets is not None and
                net_income_prev is not None and total_assets_prev and total_assets_prev > 0):
            roa_prev = net_income_prev / total_assets_prev
            roa_curr = net_income / total_assets
            if roa_curr > roa_prev:
                score += 1
            signals_computed += 1

    # Signal 4: Accruals = OCF/Total_Assets > ROA (cash earnings quality)
    if ocf is not None and net_income is not None:
        accruals_roa = ocf / total_assets
        roa_val = net_income / total_assets
        if accruals_roa > roa_val:
            score += 1
        signals_computed += 1

    # Signal 5: Leverage decreasing YoY (requires prev year)
    long_term_debt = _get(bal, "longTermDebt", "longTermDebtAndCapitalLeaseObligation")
    if bal_prev and long_term_debt is not None:
        total_assets_prev = _get(bal_prev, "totalAssets")
        long_term_debt_prev = _get(bal_prev, "longTermDebt", "longTermDebtAndCapitalLeaseObligation")
        if (total_assets_prev and total_assets_prev > 0 and
                long_term_debt_prev is not None):
            leverage_curr = long_term_debt / total_assets
            leverage_prev = long_term_debt_prev / total_assets_prev
            if leverage_curr < leverage_prev:
                score += 1
            signals_computed += 1

    # Signal 6: Current ratio improving YoY (requires prev year)
    current_assets = _get(bal, "totalCurrentAssets", "currentAssets")
    current_liabilities = _get(bal, "totalCurrentLiabilities", "currentLiabilities")
    if bal_prev and current_assets is not None and current_liabilities and current_liabilities > 0:
        ca_prev = _get(bal_prev, "totalCurrentAssets", "currentAssets")
        cl_prev = _get(bal_prev, "totalCurrentLiabilities", "currentLiabilities")
        if ca_prev is not None and cl_prev and cl_prev > 0:
            cr_curr = current_assets / current_liabilities
            cr_prev = ca_prev / cl_prev
            if cr_curr > cr_prev:
                score += 1
            signals_computed += 1

    # Signal 7: No dilution (shares outstanding not increasing)
    shares = _get(inc, "weightedAverageShsOut", "weightedAverageSharesOutstanding")
    if bal_prev and shares is not None:
        shares_prev = _get(
            inc_prev or {},
            "weightedAverageShsOut",
            "weightedAverageSharesOutstanding",
        )
        if shares_prev is not None:
            if shares <= shares_prev:
                score += 1
            signals_computed += 1

    # Signal 8: Gross margin improving YoY (requires prev year)
    gross_profit = _get(inc, "grossProfit")
    revenue = _get(inc, "revenue")
    if inc_prev and gross_profit is not None and revenue and revenue > 0:
        gm_curr = gross_profit / revenue
        gp_prev = _get(inc_prev, "grossProfit")
        rev_prev = _get(inc_prev, "revenue")
        if gp_prev is not None and rev_prev and rev_prev > 0:
            gm_prev = gp_prev / rev_prev
            if gm_curr > gm_prev:
                score += 1
            signals_computed += 1

    # Signal 9: Asset turnover improving YoY (requires prev year)
    if inc_prev and bal_prev and revenue is not None and total_assets > 0:
        rev_prev = _get(inc_prev, "revenue")
        total_assets_prev = _get(bal_prev, "totalAssets")
        if rev_prev is not None and total_assets_prev and total_assets_prev > 0:
            at_curr = revenue / total_assets
            at_prev = rev_prev / total_assets_prev
            if at_curr > at_prev:
                score += 1
            signals_computed += 1

    # Require at least 5 signals computed for a meaningful score
    if signals_computed < 5:
        return None

    return score


# ---------------------------------------------------------------------------
# Beneish M-Score (8 variables, earnings manipulation detector)
# ---------------------------------------------------------------------------

def compute_beneish_m_score(
    inc: Dict,
    bal: Dict,
    cf: Dict,
    inc_prev: Dict,
    bal_prev: Dict,
) -> Optional[float]:
    """Compute the Beneish M-Score.

    M-Score > -2.22 indicates potential earnings manipulation.
    Requires both current and prior-year data.
    """
    if not inc_prev or not bal_prev:
        return None

    def _g(d: Dict, *keys) -> Optional[float]:
        for k in keys:
            v = _safe_float(d.get(k))
            if v is not None:
                return v
        return None

    # Current year
    revenue = _g(inc, "revenue")
    gross_profit = _g(inc, "grossProfit")
    accounts_receivable = _g(bal, "netReceivables", "accountsReceivable")
    total_assets = _g(bal, "totalAssets")
    pp_and_e = _g(bal, "propertyPlantEquipmentNet", "netPPE")
    dep_amort = _g(cf, "depreciationAndAmortization")
    sga = _g(inc, "sellingGeneralAndAdministrativeExpenses", "generalAndAdministrativeExpenses")
    long_term_debt = _g(bal, "longTermDebt", "longTermDebtAndCapitalLeaseObligation")
    current_liabilities = _g(bal, "totalCurrentLiabilities")
    current_assets = _g(bal, "totalCurrentAssets")
    net_income = _g(inc, "netIncome")
    ocf = _g(cf, "operatingCashFlow")

    # Prior year
    revenue_p = _g(inc_prev, "revenue")
    gross_profit_p = _g(inc_prev, "grossProfit")
    ar_p = _g(bal_prev, "netReceivables", "accountsReceivable")
    total_assets_p = _g(bal_prev, "totalAssets")
    pp_and_e_p = _g(bal_prev, "propertyPlantEquipmentNet", "netPPE")
    dep_amort_p = _g(cf, "depreciationAndAmortization")  # use current as proxy if missing
    sga_p = _g(inc_prev, "sellingGeneralAndAdministrativeExpenses", "generalAndAdministrativeExpenses")
    long_term_debt_p = _g(bal_prev, "longTermDebt", "longTermDebtAndCapitalLeaseObligation")
    current_liabilities_p = _g(bal_prev, "totalCurrentLiabilities")
    current_assets_p = _g(bal_prev, "totalCurrentAssets")

    # Guard: need at least revenue + AR + total assets for current and prior
    if None in (revenue, ar_p, total_assets, total_assets_p, revenue_p):
        return None

    # DSRI: Days Sales in Receivables Index
    ar_curr = accounts_receivable or 0.0
    dsri = None
    if ar_p and revenue_p and revenue and revenue_p > 0:
        try:
            dsri = (ar_curr / revenue) / (ar_p / revenue_p)
        except ZeroDivisionError:
            pass

    # GMI: Gross Margin Index
    gmi = None
    if gross_profit and gross_profit_p and revenue and revenue_p and revenue > 0 and revenue_p > 0:
        gm_curr = gross_profit / revenue
        gm_prev = gross_profit_p / revenue_p
        if gm_curr > 0:
            gmi = gm_prev / gm_curr

    # AQI: Asset Quality Index
    aqi = None
    if total_assets and total_assets_p and pp_and_e and pp_and_e_p and current_assets and current_assets_p:
        non_current_non_ppe = total_assets - (current_assets or 0) - (pp_and_e or 0)
        non_current_non_ppe_p = total_assets_p - (current_assets_p or 0) - (pp_and_e_p or 0)
        if total_assets > 0 and total_assets_p > 0:
            aq_curr = non_current_non_ppe / total_assets
            aq_prev = non_current_non_ppe_p / total_assets_p
            if aq_prev > 0:
                aqi = aq_curr / aq_prev

    # SGI: Sales Growth Index
    sgi = None
    if revenue and revenue_p and revenue_p > 0:
        sgi = revenue / revenue_p

    # DEPI: Depreciation Index
    depi = None
    if dep_amort and dep_amort_p and pp_and_e and pp_and_e_p:
        if dep_amort and (dep_amort + pp_and_e) > 0:
            depr_rate_curr = dep_amort / (dep_amort + pp_and_e)
        else:
            depr_rate_curr = None
        if dep_amort_p and (dep_amort_p + pp_and_e_p) > 0:
            depr_rate_prev = dep_amort_p / (dep_amort_p + pp_and_e_p)
        else:
            depr_rate_prev = None
        if depr_rate_curr and depr_rate_prev and depr_rate_curr > 0:
            depi = depr_rate_prev / depr_rate_curr

    # SGAI: SGA Expense Index
    sgai = None
    if sga and sga_p and revenue and revenue_p and revenue > 0 and revenue_p > 0:
        sgai = (sga / revenue) / (sga_p / revenue_p)

    # LVGI: Leverage Index
    lvgi = None
    if long_term_debt is not None and long_term_debt_p is not None and current_liabilities and current_liabilities_p:
        if total_assets and total_assets_p and total_assets > 0 and total_assets_p > 0:
            lev_curr = (long_term_debt + current_liabilities) / total_assets
            lev_prev = (long_term_debt_p + current_liabilities_p) / total_assets_p
            if lev_prev > 0:
                lvgi = lev_curr / lev_prev

    # TATA: Total Accruals to Total Assets
    tata = None
    if net_income is not None and ocf is not None and total_assets and total_assets > 0:
        tata = (net_income - ocf) / total_assets

    # Count available variables
    vars_available = sum(v is not None for v in [dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata])
    if vars_available < 5:
        return None

    # M-Score formula (Beneish 1999)
    m_score = (
        -4.840
        + 0.920 * (dsri or 1.0)
        + 0.528 * (gmi or 1.0)
        + 0.404 * (aqi or 1.0)
        + 0.892 * (sgi or 1.0)
        + 0.115 * (depi or 1.0)
        - 0.172 * (sgai or 1.0)
        + 4.679 * (tata or 0.0)
        - 0.327 * (lvgi or 1.0)
    )
    return round(m_score, 4)


# ---------------------------------------------------------------------------
# Key Metrics helpers
# ---------------------------------------------------------------------------

def compute_key_metrics_quality(
    bundle: FinancialsBundle,
) -> Dict[str, Optional[float]]:
    """Compute gross_margin, ebit_margin, fcf_conversion, dso, current_ratio, debt_to_equity.

    Sources (in priority order):
      1. key_metrics_ttm / ratios_ttm (EODHD PascalCase fields)
      2. valuation_metrics table (profit_margin, operating_margin as fallback)
    current_ratio and debt_to_equity are not in any allowed source — remain null.
    """
    km_ttm = bundle.key_metrics_ttm or {}
    rt = bundle.ratios_ttm or {}
    vm = bundle.valuation_metrics or {}

    def _g(d, *keys):
        for k in keys:
            v = _safe_float(d.get(k))
            if v is not None:
                return v
        return None

    # Gross margin — derive from GrossProfitTTM / RevenueTTM (key_metrics_ttm)
    gm = _g(rt, "grossProfitMarginTTM", "GrossProfitMarginTTM")
    if gm is None:
        gm = _g(km_ttm, "grossProfitMarginTTM", "GrossProfitMarginTTM")
    if gm is None:
        gp = _g(km_ttm, "GrossProfitTTM", "grossProfitTTM")
        rev = _g(km_ttm, "RevenueTTM", "revenueTTM")
        gm = _safe_div(gp, rev)
    # valuation_metrics doesn't have gross_margin directly

    # EBIT margin — EODHD OperatingMarginTTM (PascalCase)
    ebit_margin = (
        _g(km_ttm, "OperatingMarginTTM", "operatingMarginTTM")
        or _g(rt, "operatingProfitMarginTTM", "OperatingProfitMarginTTM")
        or _g(vm, "operating_margin")  # valuation_metrics fallback
    )

    # FCF conversion — requires cash flow data; not available from allowed sources
    fcf_conv: Optional[float] = None

    # DSO — daysOfSalesOutstandingTTM not in EODHD key_metrics_ttm payload
    dso: Optional[float] = None

    # Current ratio — not available in any allowed data source
    cr: Optional[float] = None

    # Debt to equity — not available in any allowed data source
    dte: Optional[float] = None

    return {
        "gross_margin": round(gm, 4) if gm is not None else None,
        "ebit_margin": round(ebit_margin, 4) if ebit_margin is not None else None,
        "fcf_conversion": fcf_conv,
        "dso_days": dso,
        "current_ratio": cr,
        "debt_to_equity": dte,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_quality_factors(
    bundle: FinancialsBundle,
    inc_prev: Optional[Dict] = None,
    bal_prev: Optional[Dict] = None,
    cf_prev: Optional[Dict] = None,
    beneish_threshold: float = -2.22,
) -> QualityFactors:
    """Compute quality factors from the FinancialsBundle.

    RESTRICTED: Uses only allowed data types (key_metrics_ttm, ratios_ttm,
    valuation_metrics). Financial statements are excluded.

    ROE is sourced from key_metrics_ttm (ReturnOnEquityTTM) or valuation_metrics.roe.
    ROIC is not directly available — remains null.
    Piotroski/Beneish/Altman require financial statements — remain null.
    """
    rt = bundle.ratios_ttm or {}
    km_ttm = bundle.key_metrics_ttm or {}
    vm = bundle.valuation_metrics or {}

    # ROE: EODHD payloads use PascalCase (ReturnOnEquityTTM)
    roe_raw = (
        km_ttm.get("ReturnOnEquityTTM") or km_ttm.get("returnOnEquityTTM")
        or rt.get("ReturnOnEquityTTM") or rt.get("returnOnEquityTTM")
        or km_ttm.get("roeTTM")
        or vm.get("roe")
    )
    _roe = _safe_float(roe_raw)
    roe = round(_roe, 4) if _roe is not None else None

    # ROIC: not in key_metrics_ttm or valuation_metrics — remains null
    roic: Optional[float] = None

    # Piotroski and Beneish scores require financial statements - not available
    piotroski: Optional[int] = None
    beneish: Optional[float] = None
    altman_z: Optional[float] = None
    manipulation_risk: Optional[str] = None

    logger.info("[QualityFactors] roe=%.4f roic=None (not available from allowed data)", roe or 0)

    return QualityFactors(
        roe=roe,
        roic=roic,
        piotroski_f_score=piotroski,
        beneish_m_score=beneish,
        manipulation_risk=manipulation_risk,
        altman_z_score=altman_z,
    )


__all__ = [
    "compute_quality_factors",
    "compute_piotroski",
    "compute_beneish_m_score",
    "compute_key_metrics_quality",
]
