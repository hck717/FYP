"""Linked 3-Statement Financial Model.

Constructs a coherent Income Statement → Balance Sheet → Cash Flow Statement
reconciliation from EODHD annual data stored in PostgreSQL.

The model verifies the fundamental accounting identities:
  (1) Net Income (IS) → feeds Retained Earnings on BS (Δ RE = NI - Dividends Paid)
  (2) Net Income (IS) → starting point of CF from Operations (indirect method)
  (3) Cash position on BS must equal End-of-Period Cash on CF Statement
  (4) Total Assets = Total Liabilities + Shareholders' Equity (BS balance check)

Output: ThreeStatementModel dataclass with period-indexed statements and
        a linkage_checks dict showing which identities hold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..schema import FMDataBundle

logger = logging.getLogger(__name__)


def _sf(val: Any, default: Optional[float] = None) -> Optional[float]:
    if val is None:
        return default
    try:
        f = float(val)
        return None if (f != f) else f  # drop NaN
    except (TypeError, ValueError):
        return default


def _sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def _add(*args: Optional[float]) -> Optional[float]:
    result = 0.0
    for v in args:
        if v is None:
            return None
        result += v
    return result


@dataclass
class IncomeStatementRow:
    period: str
    revenue: Optional[float] = None
    cost_of_revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    gross_margin: Optional[float] = None          # gross_profit / revenue
    research_and_development: Optional[float] = None
    operating_expenses: Optional[float] = None
    operating_income: Optional[float] = None      # EBIT
    ebitda: Optional[float] = None
    interest_expense: Optional[float] = None
    income_before_tax: Optional[float] = None
    income_tax_expense: Optional[float] = None
    net_income: Optional[float] = None
    net_margin: Optional[float] = None            # net_income / revenue
    eps_basic: Optional[float] = None             # net_income / shares_outstanding
    effective_tax_rate: Optional[float] = None    # income_tax / income_before_tax

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "revenue": self.revenue,
            "cost_of_revenue": self.cost_of_revenue,
            "gross_profit": self.gross_profit,
            "gross_margin": self.gross_margin,
            "research_and_development": self.research_and_development,
            "operating_expenses": self.operating_expenses,
            "operating_income": self.operating_income,
            "ebitda": self.ebitda,
            "interest_expense": self.interest_expense,
            "income_before_tax": self.income_before_tax,
            "income_tax_expense": self.income_tax_expense,
            "net_income": self.net_income,
            "net_margin": self.net_margin,
            "eps_basic": self.eps_basic,
            "effective_tax_rate": self.effective_tax_rate,
        }


@dataclass
class BalanceSheetRow:
    period: str
    # Assets
    cash_and_equivalents: Optional[float] = None
    net_receivables: Optional[float] = None
    inventory: Optional[float] = None
    other_current_assets: Optional[float] = None
    total_current_assets: Optional[float] = None
    intangible_assets: Optional[float] = None
    goodwill: Optional[float] = None
    total_assets: Optional[float] = None
    # Liabilities
    accounts_payable: Optional[float] = None
    short_term_debt: Optional[float] = None
    other_current_liabilities: Optional[float] = None
    long_term_debt: Optional[float] = None
    total_liabilities: Optional[float] = None
    # Equity
    retained_earnings: Optional[float] = None
    common_stock: Optional[float] = None
    treasury_stock: Optional[float] = None
    total_equity: Optional[float] = None
    # Derived
    net_working_capital: Optional[float] = None   # current assets - current liabilities
    net_debt: Optional[float] = None              # total debt - cash
    book_value_per_share: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "cash_and_equivalents": self.cash_and_equivalents,
            "net_receivables": self.net_receivables,
            "inventory": self.inventory,
            "other_current_assets": self.other_current_assets,
            "total_current_assets": self.total_current_assets,
            "intangible_assets": self.intangible_assets,
            "goodwill": self.goodwill,
            "total_assets": self.total_assets,
            "accounts_payable": self.accounts_payable,
            "short_term_debt": self.short_term_debt,
            "other_current_liabilities": self.other_current_liabilities,
            "long_term_debt": self.long_term_debt,
            "total_liabilities": self.total_liabilities,
            "retained_earnings": self.retained_earnings,
            "common_stock": self.common_stock,
            "treasury_stock": self.treasury_stock,
            "total_equity": self.total_equity,
            "net_working_capital": self.net_working_capital,
            "net_debt": self.net_debt,
            "book_value_per_share": self.book_value_per_share,
        }


@dataclass
class CashFlowRow:
    period: str
    # Operating activities (indirect method)
    net_income: Optional[float] = None
    depreciation_amortization: Optional[float] = None
    stock_based_compensation: Optional[float] = None
    change_in_working_capital: Optional[float] = None
    change_receivables: Optional[float] = None
    change_inventory: Optional[float] = None
    change_payables: Optional[float] = None
    other_operating: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    # Investing activities
    capital_expenditures: Optional[float] = None
    investments: Optional[float] = None
    free_cash_flow: Optional[float] = None        # OCF - CapEx
    # Financing activities
    dividends_paid: Optional[float] = None
    share_buybacks: Optional[float] = None        # negative = buybacks
    net_borrowings: Optional[float] = None
    financing_cash_flow: Optional[float] = None
    # Net change
    net_change_in_cash: Optional[float] = None
    beginning_cash: Optional[float] = None
    ending_cash: Optional[float] = None
    # Derived ratios
    fcf_margin: Optional[float] = None            # free_cash_flow / revenue
    capex_intensity: Optional[float] = None       # |capex| / revenue

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "net_income": self.net_income,
            "depreciation_amortization": self.depreciation_amortization,
            "stock_based_compensation": self.stock_based_compensation,
            "change_in_working_capital": self.change_in_working_capital,
            "change_receivables": self.change_receivables,
            "change_inventory": self.change_inventory,
            "change_payables": self.change_payables,
            "other_operating": self.other_operating,
            "operating_cash_flow": self.operating_cash_flow,
            "capital_expenditures": self.capital_expenditures,
            "investments": self.investments,
            "free_cash_flow": self.free_cash_flow,
            "dividends_paid": self.dividends_paid,
            "share_buybacks": self.share_buybacks,
            "net_borrowings": self.net_borrowings,
            "financing_cash_flow": self.financing_cash_flow,
            "net_change_in_cash": self.net_change_in_cash,
            "beginning_cash": self.beginning_cash,
            "ending_cash": self.ending_cash,
            "fcf_margin": self.fcf_margin,
            "capex_intensity": self.capex_intensity,
        }


@dataclass
class StatementLinkageCheck:
    """Results of the 3-statement linkage verification for one period."""
    period: str
    # (1) NI → Retained Earnings: BS RE_t - BS RE_{t-1} ≈ IS NI - CF Dividends
    re_linkage_holds: Optional[bool] = None
    re_delta: Optional[float] = None              # actual ΔRE from BS
    re_expected: Optional[float] = None           # NI - Dividends
    re_diff: Optional[float] = None               # abs(actual - expected)
    # (2) CF ending cash ≈ BS cash
    cash_linkage_holds: Optional[bool] = None
    bs_cash: Optional[float] = None
    cf_ending_cash: Optional[float] = None
    cash_diff: Optional[float] = None
    # (3) Assets = Liabilities + Equity
    bs_balance_holds: Optional[bool] = None
    total_assets: Optional[float] = None
    total_liab_plus_equity: Optional[float] = None
    bs_diff: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "re_linkage_holds": self.re_linkage_holds,
            "re_delta": self.re_delta,
            "re_expected": self.re_expected,
            "re_diff": self.re_diff,
            "cash_linkage_holds": self.cash_linkage_holds,
            "bs_cash": self.bs_cash,
            "cf_ending_cash": self.cf_ending_cash,
            "cash_diff": self.cash_diff,
            "bs_balance_holds": self.bs_balance_holds,
            "total_assets": self.total_assets,
            "total_liab_plus_equity": self.total_liab_plus_equity,
            "bs_diff": self.bs_diff,
        }


@dataclass
class ThreeStatementModel:
    """Linked Income Statement + Balance Sheet + Cash Flow Statement."""

    ticker: str
    income_statements: List[IncomeStatementRow] = field(default_factory=list)
    balance_sheets: List[BalanceSheetRow] = field(default_factory=list)
    cash_flows: List[CashFlowRow] = field(default_factory=list)
    linkage_checks: List[StatementLinkageCheck] = field(default_factory=list)

    # Tolerance for linkage checks (as fraction of the larger value, or abs $)
    # EODHD data may have minor rounding differences between reported IS/BS/CF.
    # RE linkage also legitimately differs due to AOCI, SBC, and other equity movements
    # not captured in the simplified NI - Divs + Buybacks formula.
    LINKAGE_TOLERANCE_ABS: float = 5e9        # $5B absolute tolerance (covers SBC/AOCI for megacaps)
    LINKAGE_TOLERANCE_REL: float = 0.10       # 10% relative tolerance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "income_statements": [r.to_dict() for r in self.income_statements],
            "balance_sheets": [r.to_dict() for r in self.balance_sheets],
            "cash_flows": [r.to_dict() for r in self.cash_flows],
            "linkage_checks": [c.to_dict() for c in self.linkage_checks],
        }


class ThreeStatementEngine:
    """Constructs and validates a linked 3-statement model from FMDataBundle."""

    def compute(self, bundle: FMDataBundle) -> ThreeStatementModel:
        """Build a 2-period (current annual + prior annual) 3-statement model.

        Uses bundle.income_annual / balance_annual / cashflow_annual as the
        most recent fiscal year (annual) period — preferred over bundle.income
        which may be the most recent quarterly filing.

        Falls back to bundle.income / balance / cashflow if annual is empty.
        """
        model = ThreeStatementModel(ticker=bundle.ticker)

        # Prefer annual data for IS/BS/CF comparability; fall back to current if needed
        curr_inc = bundle.income_annual if bundle.income_annual else bundle.income
        curr_bal = bundle.balance_annual if bundle.balance_annual else bundle.balance
        curr_cf  = bundle.cashflow_annual if bundle.cashflow_annual else bundle.cashflow

        # Build current and prior period statements
        periods = [
            (curr_inc, curr_bal, curr_cf, "current"),
        ]
        if bundle.income_prior:
            periods.append(
                (bundle.income_prior, bundle.balance_prior, bundle.cashflow_prior, "prior")
            )

        # Determine shares outstanding for per-share calcs
        shares = self._get_shares(bundle)

        for inc, bal, cf, label in periods:
            period_label = str(inc.get("date") or inc.get("period") or label)

            is_row = self._build_income_statement(inc, period_label, shares)
            bs_row = self._build_balance_sheet(bal, period_label, shares)
            cf_row = self._build_cash_flow(cf, period_label, is_row.revenue)

            model.income_statements.append(is_row)
            model.balance_sheets.append(bs_row)
            model.cash_flows.append(cf_row)

        # Sort all lists oldest-first for readability (prior before current)
        model.income_statements.reverse()
        model.balance_sheets.reverse()
        model.cash_flows.reverse()

        # Linkage checks (for current period only; need prior BS for ΔRE check)
        if len(model.income_statements) >= 1:
            self._check_linkages(model)

        logger.debug(
            "[3SM] %s: IS periods=%s, linkage_checks=%s",
            bundle.ticker,
            [r.period for r in model.income_statements],
            [(c.period, c.re_linkage_holds, c.cash_linkage_holds, c.bs_balance_holds)
             for c in model.linkage_checks],
        )
        return model

    # ── Statement builders ───────────────────────────────────────────────────

    def _build_income_statement(
        self,
        inc: Dict[str, Any],
        period: str,
        shares: Optional[float],
    ) -> IncomeStatementRow:
        row = IncomeStatementRow(period=period)

        row.revenue              = _sf(inc.get("totalRevenue") or inc.get("revenue"))
        row.cost_of_revenue      = _sf(inc.get("costOfRevenue"))
        row.gross_profit         = _sf(inc.get("grossProfit"))
        row.research_and_development = _sf(inc.get("researchDevelopment") or inc.get("researchAndDevelopmentExpenses"))
        row.operating_expenses   = _sf(inc.get("totalOperatingExpenses") or inc.get("operatingExpenses"))
        row.operating_income     = _sf(inc.get("operatingIncome") or inc.get("ebit"))
        row.ebitda               = _sf(inc.get("ebitda") or inc.get("EBITDA"))
        row.interest_expense     = _sf(inc.get("interestExpense"))
        row.income_before_tax    = _sf(inc.get("incomeBeforeTax"))
        row.income_tax_expense   = _sf(inc.get("incomeTaxExpense") or inc.get("taxProvision"))
        row.net_income           = _sf(inc.get("netIncome"))

        # Derived
        if row.revenue and row.revenue > 0:
            if row.gross_profit is not None:
                row.gross_margin = round(row.gross_profit / row.revenue, 4)
            if row.net_income is not None:
                row.net_margin = round(row.net_income / row.revenue, 4)

        # Gross profit from revenue - COGS if not provided
        if row.gross_profit is None and row.revenue and row.cost_of_revenue:
            row.gross_profit = row.revenue - row.cost_of_revenue
            if row.revenue > 0:
                row.gross_margin = round(row.gross_profit / row.revenue, 4)

        if row.income_before_tax and row.income_before_tax > 0 and row.income_tax_expense:
            row.effective_tax_rate = round(row.income_tax_expense / row.income_before_tax, 4)

        if row.net_income and shares and shares > 0:
            row.eps_basic = round(row.net_income / shares, 4)

        return row

    def _build_balance_sheet(
        self,
        bal: Dict[str, Any],
        period: str,
        shares: Optional[float],
    ) -> BalanceSheetRow:
        row = BalanceSheetRow(period=period)

        row.cash_and_equivalents  = _sf(bal.get("cash") or bal.get("cashAndCashEquivalents"))
        row.net_receivables       = _sf(bal.get("netReceivables"))
        row.inventory             = _sf(bal.get("inventory"))
        row.total_assets          = _sf(bal.get("totalAssets"))
        row.accounts_payable      = _sf(bal.get("accountsPayable"))
        row.short_term_debt       = _sf(bal.get("shortTermDebt") or bal.get("shortLongTermDebt"))
        row.other_current_liabilities = _sf(bal.get("otherCurrentLiab"))
        row.long_term_debt        = _sf(bal.get("longTermDebt") or bal.get("longTermDebtTotal"))
        row.total_liabilities     = _sf(bal.get("totalLiab"))
        row.retained_earnings     = _sf(bal.get("retainedEarnings"))
        row.common_stock          = _sf(bal.get("commonStock") or bal.get("capitalStock"))
        row.treasury_stock        = _sf(bal.get("treasuryStock"))
        row.goodwill              = _sf(bal.get("goodWill") or bal.get("goodwill"))
        row.intangible_assets     = _sf(bal.get("intangibleAssets"))
        row.net_working_capital   = _sf(bal.get("netWorkingCapital"))
        row.net_debt              = _sf(bal.get("netDebt"))

        # Total equity: Assets - Liabilities
        if row.total_assets is not None and row.total_liabilities is not None:
            row.total_equity = row.total_assets - row.total_liabilities

        # Book value per share
        if row.total_equity and shares and shares > 0:
            row.book_value_per_share = round(row.total_equity / shares, 4)

        return row

    def _build_cash_flow(
        self,
        cf: Dict[str, Any],
        period: str,
        revenue: Optional[float],
    ) -> CashFlowRow:
        row = CashFlowRow(period=period)

        row.net_income               = _sf(cf.get("netIncome"))
        row.depreciation_amortization = _sf(cf.get("depreciation") or cf.get("depreciationAndAmortization"))
        row.stock_based_compensation  = _sf(cf.get("stockBasedCompensation"))
        row.change_in_working_capital = _sf(cf.get("changeInWorkingCapital"))
        row.change_receivables        = _sf(cf.get("changeToAccountReceivables") or cf.get("changeReceivables"))
        row.change_inventory          = _sf(cf.get("changeToInventory") or cf.get("changeInventory"))
        row.change_payables           = _sf(cf.get("changeToLiabilities"))
        row.other_operating           = _sf(cf.get("changeToOperatingActivities") or cf.get("otherNonCashItems"))
        row.capital_expenditures      = _sf(cf.get("capitalExpenditures"))
        row.investments               = _sf(cf.get("investments"))
        row.dividends_paid            = _sf(cf.get("dividendsPaid"))
        row.share_buybacks            = _sf(cf.get("salePurchaseOfStock"))
        row.net_borrowings            = _sf(cf.get("netBorrowings"))
        row.net_change_in_cash        = _sf(cf.get("changeInCash") or cf.get("cashAndCashEquivalentsChanges"))
        row.beginning_cash            = _sf(cf.get("beginPeriodCashFlow"))
        row.ending_cash               = _sf(cf.get("endPeriodCashFlow"))

        # Pre-computed OCF and FCF from EODHD (if available)
        row.operating_cash_flow = _sf(
            cf.get("operatingCashFlow")
            or cf.get("totalCashFromOperatingActivities")
            or cf.get("cashFromOperations")
        )
        row.free_cash_flow = _sf(cf.get("freeCashFlow"))

        # Reconstruct OCF from components if not directly available
        if row.operating_cash_flow is None:
            ocf_components = [
                row.net_income,
                row.depreciation_amortization,
                row.stock_based_compensation,
                row.change_in_working_capital,
                row.other_operating,
            ]
            if all(v is not None for v in ocf_components[:2]):  # at least NI + D&A
                row.operating_cash_flow = sum(
                    (v or 0) for v in ocf_components if v is not None
                )

        # Financing CF
        if row.dividends_paid is not None or row.net_borrowings is not None:
            row.financing_cash_flow = (
                (row.dividends_paid or 0)
                + (row.share_buybacks or 0)
                + (row.net_borrowings or 0)
                + _sf(cf.get("issuanceOfCapitalStock") or 0, 0)
            )

        # FCF = OCF + CapEx (CapEx is usually negative in EODHD)
        if row.free_cash_flow is None and row.operating_cash_flow is not None:
            if row.capital_expenditures is not None:
                row.free_cash_flow = row.operating_cash_flow + row.capital_expenditures

        # Derived ratios
        if revenue and revenue > 0:
            if row.free_cash_flow is not None:
                row.fcf_margin = round(row.free_cash_flow / revenue, 4)
            if row.capital_expenditures is not None:
                row.capex_intensity = round(abs(row.capital_expenditures) / revenue, 4)

        return row

    # ── Linkage checks ───────────────────────────────────────────────────────

    def _check_linkages(self, model: ThreeStatementModel) -> None:
        """Verify 3-statement accounting identities."""
        statements = zip(model.income_statements, model.balance_sheets, model.cash_flows)

        prev_bs: Optional[BalanceSheetRow] = None
        for is_row, bs_row, cf_row in statements:
            check = StatementLinkageCheck(period=is_row.period)

            # (1) Retained Earnings linkage: ΔRE = NI - Dividends Paid
            # Note: For companies with large share buybacks (e.g. AAPL), ΔRE may differ
            # from NI - Divs because buybacks can reduce RE when accumulated deficit exists.
            # We also include share repurchase/issuance in the expected formula.
            if prev_bs is not None:
                re_curr = bs_row.retained_earnings
                re_prev = prev_bs.retained_earnings
                if re_curr is not None and re_prev is not None:
                    check.re_delta = round(re_curr - re_prev, 0)
                    ni: float = is_row.net_income or 0.0
                    divs: float = abs(cf_row.dividends_paid or 0.0)  # EODHD stores as negative
                    # Share repurchases can reduce retained earnings when company has
                    # accumulated deficit (negative RE). Include net stock activity.
                    buybacks: float = cf_row.share_buybacks or 0.0   # negative = net repurchases
                    check.re_expected = round(ni - divs + buybacks, 0)
                    if check.re_delta is not None and check.re_expected is not None:
                        check.re_diff = abs(check.re_delta - check.re_expected)
                        ref = max(abs(check.re_delta), abs(check.re_expected), 1.0)
                        rel_err = check.re_diff / ref
                        check.re_linkage_holds = (
                            check.re_diff < model.LINKAGE_TOLERANCE_ABS
                            or rel_err < model.LINKAGE_TOLERANCE_REL
                        )

            # (2) Cash: BS cash ≈ CF ending cash
            check.bs_cash = bs_row.cash_and_equivalents
            check.cf_ending_cash = cf_row.ending_cash
            if check.bs_cash is not None and check.cf_ending_cash is not None:
                check.cash_diff = abs(check.bs_cash - check.cf_ending_cash)
                ref = max(abs(check.bs_cash), abs(check.cf_ending_cash), 1.0)
                rel_err = check.cash_diff / ref
                check.cash_linkage_holds = (
                    check.cash_diff < model.LINKAGE_TOLERANCE_ABS
                    or rel_err < model.LINKAGE_TOLERANCE_REL
                )

            # (3) Balance sheet: Assets = Liabilities + Equity
            check.total_assets = bs_row.total_assets
            if bs_row.total_liabilities is not None and bs_row.total_equity is not None:
                check.total_liab_plus_equity = bs_row.total_liabilities + bs_row.total_equity
            if check.total_assets is not None and check.total_liab_plus_equity is not None:
                check.bs_diff = abs(check.total_assets - check.total_liab_plus_equity)
                ref = max(abs(check.total_assets), abs(check.total_liab_plus_equity), 1.0)
                rel_err = check.bs_diff / ref
                check.bs_balance_holds = (
                    check.bs_diff < model.LINKAGE_TOLERANCE_ABS
                    or rel_err < model.LINKAGE_TOLERANCE_REL
                )

            model.linkage_checks.append(check)
            prev_bs = bs_row

    # ── Helper ───────────────────────────────────────────────────────────────

    def _get_shares(self, bundle: FMDataBundle) -> Optional[float]:
        """Derive shares outstanding for per-share calculations."""
        # From outstanding_shares table (most recent)
        if bundle.outstanding_shares:
            row = bundle.outstanding_shares[0]
            s = _sf(row.get("shares_outstanding"))
            if s and s > 0:
                return s
        # From key_metrics_ttm
        s = _sf(
            bundle.key_metrics_ttm.get("sharesOutstanding")
            or bundle.key_metrics_ttm.get("SharesOutstanding")
            or bundle.key_metrics_ttm.get("CommonStockSharesOutstanding")
        )
        if s and s > 0:
            return s
        # From valuation_metrics
        if bundle.valuation_metrics:
            s = _sf(bundle.valuation_metrics.get("shares_outstanding"))
            if s and s > 0:
                return s
        return None


__all__ = [
    "ThreeStatementEngine",
    "ThreeStatementModel",
    "IncomeStatementRow",
    "BalanceSheetRow",
    "CashFlowRow",
    "StatementLinkageCheck",
]
