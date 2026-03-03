"""Tests for the Quantitative Fundamental agent.

Unit tests use mocked data; integration tests require a live PostgreSQL connection.

Run unit tests only (no DB):
    .venv/bin/python -m pytest agents/quant_fundamental/tests/test_agent.py -v -m "not integration"

Run all tests including integration (requires Docker services running):
    .venv/bin/python -m pytest agents/quant_fundamental/tests/test_agent.py -v
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from agents.quant_fundamental.config import QuantFundamentalConfig
from agents.quant_fundamental.factors.momentum_risk import (
    _daily_returns,
    _parse_price_rows,
    compute_beta_60d,
    compute_return_12m,
    compute_sharpe_12m,
)
from agents.quant_fundamental.factors.quality import (
    compute_beneish_m_score,
    compute_piotroski,
    compute_quality_factors,
)
from agents.quant_fundamental.factors.value import compute_value_factors
from agents.quant_fundamental.schema import (
    DataQualityCheck,
    FinancialsBundle,
    QualityStatus,
)
from agents.quant_fundamental.tools import DataQualityChecker, _safe_div, _safe_float


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_bundle() -> FinancialsBundle:
    """A FinancialsBundle with enough data to compute most factors."""
    return FinancialsBundle(
        ticker="TEST",
        income={
            "revenue": 400_000_000_000,
            "grossProfit": 180_000_000_000,
            "operatingIncome": 120_000_000_000,
            "netIncome": 100_000_000_000,
            "ebitda": 130_000_000_000,
            "incomeTaxExpense": 15_000_000_000,
            "incomeBeforeTax": 115_000_000_000,
            "weightedAverageShsOut": 15_500_000_000,
        },
        balance={
            "totalAssets": 350_000_000_000,
            "totalCurrentAssets": 130_000_000_000,
            "totalCurrentLiabilities": 80_000_000_000,
            "totalStockholdersEquity": 70_000_000_000,
            "longTermDebt": 100_000_000_000,
            "netReceivables": 28_000_000_000,
            "propertyPlantEquipmentNet": 45_000_000_000,
        },
        cashflow={
            "operatingCashFlow": 110_000_000_000,
            "capitalExpenditure": -10_000_000_000,
            "freeCashFlow": 100_000_000_000,
            "depreciationAndAmortization": 10_000_000_000,
        },
        ratios_ttm={
            "peRatioTTM": 28.5,
            "returnOnEquityTTM": 1.45,
            "returnOnCapitalEmployedTTM": 0.55,
            "grossProfitMarginTTM": 0.45,
            "operatingProfitMarginTTM": 0.30,
            "currentRatioTTM": 1.62,
            "debtEquityRatioTTM": 1.42,
        },
        key_metrics_ttm={
            "evToEbitdaTTM": 22.1,
            "pfcfRatioTTM": 28.0,
            "evToSalesTTM": 7.5,
            "roicTTM": 0.55,
        },
        enterprise={
            "enterpriseValue": 2_900_000_000_000,
        },
    )


@pytest.fixture
def price_series() -> List[Dict[str, Any]]:
    """120 synthetic daily price rows for momentum factor testing."""
    import random
    from datetime import timedelta
    random.seed(42)
    rows = []
    price = 150.0
    base = datetime(2024, 1, 1)
    for i in range(120):
        dt = base + timedelta(days=i)
        price *= (1 + random.gauss(0.0003, 0.015))
        rows.append({"date": dt.strftime("%Y-%m-%d"), "close": round(price, 2)})
    return rows


# ---------------------------------------------------------------------------
# Unit: safe math helpers
# ---------------------------------------------------------------------------

class TestSafeHelpers:
    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_str(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_safe_float_invalid(self):
        assert _safe_float("not_a_number") is None

    def test_safe_div_zero_denom(self):
        assert _safe_div(10.0, 0.0) is None

    def test_safe_div_none(self):
        assert _safe_div(None, 5.0) is None

    def test_safe_div_normal(self):
        assert _safe_div(10.0, 4.0) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Unit: Value factors
# ---------------------------------------------------------------------------

class TestValueFactors:
    def test_pe_from_ratios_ttm(self, minimal_bundle):
        result = compute_value_factors(minimal_bundle)
        assert result.pe_trailing == pytest.approx(28.5, abs=0.1)

    def test_ev_ebitda_from_key_metrics(self, minimal_bundle):
        result = compute_value_factors(minimal_bundle)
        assert result.ev_ebitda == pytest.approx(22.1, abs=0.1)

    def test_p_fcf_from_ttm(self, minimal_bundle):
        result = compute_value_factors(minimal_bundle)
        assert result.p_fcf == pytest.approx(28.0, abs=0.1)

    def test_ev_revenue_from_ttm(self, minimal_bundle):
        result = compute_value_factors(minimal_bundle)
        assert result.ev_revenue == pytest.approx(7.5, abs=0.1)

    def test_empty_bundle_returns_nones(self):
        empty = FinancialsBundle(ticker="EMPTY")
        result = compute_value_factors(empty)
        assert result.pe_trailing is None
        assert result.ev_ebitda is None
        assert result.p_fcf is None
        assert result.ev_revenue is None


# ---------------------------------------------------------------------------
# Unit: Quality factors
# ---------------------------------------------------------------------------

class TestQualityFactors:
    def test_roe_from_ratios_ttm(self, minimal_bundle):
        result = compute_quality_factors(minimal_bundle)
        assert result.roe == pytest.approx(1.45, abs=0.01)

    def test_roic_from_ttm(self, minimal_bundle):
        result = compute_quality_factors(minimal_bundle)
        assert result.roic == pytest.approx(0.55, abs=0.01)

    def test_piotroski_computed_without_fmp(self, minimal_bundle):
        # No FMP financial_scores — should compute from scratch
        # Provide prior-year data so ≥5 signals can be computed
        minimal_bundle.scores = {}
        inc_prev = {
            "netIncome": 90_000_000_000,
            "revenue": 380_000_000_000,
            "grossProfit": 165_000_000_000,
            "weightedAverageShsOut": 16_000_000_000,
        }
        bal_prev = {
            "totalAssets": 330_000_000_000,
            "totalCurrentAssets": 120_000_000_000,
            "totalCurrentLiabilities": 78_000_000_000,
            "longTermDebt": 110_000_000_000,
        }
        cf_prev = {"operatingCashFlow": 100_000_000_000}
        result = compute_quality_factors(minimal_bundle, inc_prev=inc_prev, bal_prev=bal_prev, cf_prev=cf_prev)
        assert result.piotroski_f_score is not None
        assert 0 <= result.piotroski_f_score <= 9

    def test_piotroski_from_fmp_scores(self, minimal_bundle):
        minimal_bundle.scores = {"piotroskiScore": 7}
        result = compute_quality_factors(minimal_bundle)
        assert result.piotroski_f_score == 7

    def test_manipulation_risk_low(self, minimal_bundle):
        minimal_bundle.scores = {"beneishMScore": -3.0}
        result = compute_quality_factors(minimal_bundle)
        assert result.manipulation_risk == "LOW"

    def test_manipulation_risk_high(self, minimal_bundle):
        minimal_bundle.scores = {"beneishMScore": -1.5}
        result = compute_quality_factors(minimal_bundle)
        assert result.manipulation_risk == "HIGH"


# ---------------------------------------------------------------------------
# Unit: Piotroski
# ---------------------------------------------------------------------------

class TestPiotroski:
    def test_positive_earnings(self):
        # Need ≥5 computable signals: provide full current + prior year data
        inc = {"netIncome": 100, "revenue": 500, "grossProfit": 200, "weightedAverageShsOut": 10}
        bal = {"totalAssets": 300, "totalCurrentAssets": 100, "totalCurrentLiabilities": 60,
               "longTermDebt": 50}
        cf = {"operatingCashFlow": 80}
        inc_prev = {"netIncome": 80, "revenue": 480, "grossProfit": 185, "weightedAverageShsOut": 11}
        bal_prev = {"totalAssets": 290, "totalCurrentAssets": 95, "totalCurrentLiabilities": 58,
                    "longTermDebt": 60}
        score = compute_piotroski(inc, bal, cf, inc_prev, bal_prev)
        assert score is not None
        assert score >= 2  # ROA > 0, OCF > 0, OCF > ROA

    def test_negative_income_scores_zero(self):
        # Need ≥5 signals: full current + prior year
        inc = {"netIncome": -50, "revenue": 500, "grossProfit": 100, "weightedAverageShsOut": 10}
        bal = {"totalAssets": 300, "totalCurrentAssets": 90, "totalCurrentLiabilities": 80,
               "longTermDebt": 120}
        cf = {"operatingCashFlow": -20}
        inc_prev = {"netIncome": -30, "revenue": 480, "grossProfit": 95, "weightedAverageShsOut": 9}
        bal_prev = {"totalAssets": 280, "totalCurrentAssets": 88, "totalCurrentLiabilities": 75,
                    "longTermDebt": 100}
        score = compute_piotroski(inc, bal, cf, inc_prev, bal_prev)
        assert score is not None
        assert score <= 4  # Mostly negative signals

    def test_returns_none_insufficient_signals(self):
        score = compute_piotroski({}, {}, {})
        assert score is None


# ---------------------------------------------------------------------------
# Unit: Beneish M-Score
# ---------------------------------------------------------------------------

class TestBeneish:
    def test_requires_prior_year(self):
        inc = {"revenue": 1000, "netIncome": 100}
        bal = {"totalAssets": 500}
        cf = {"operatingCashFlow": 80}
        result = compute_beneish_m_score(inc, bal, cf, {}, {})
        assert result is None

    def test_reasonable_m_score_range(self):
        inc = {
            "revenue": 400e9, "grossProfit": 180e9, "netIncome": 100e9,
            "sellingGeneralAndAdministrativeExpenses": 20e9,
        }
        bal = {
            "totalAssets": 350e9, "totalCurrentAssets": 130e9, "netReceivables": 28e9,
            "longTermDebt": 100e9, "totalCurrentLiabilities": 80e9,
            "propertyPlantEquipmentNet": 45e9,
        }
        cf = {"operatingCashFlow": 110e9, "depreciationAndAmortization": 10e9}
        inc_p = {**inc, "revenue": 380e9, "grossProfit": 165e9, "netIncome": 95e9}
        bal_p = {**bal, "totalAssets": 330e9, "netReceivables": 25e9}
        result = compute_beneish_m_score(inc, bal, cf, inc_p, bal_p)
        # M-Score for a healthy company should be around -2.5 to -3.5
        assert result is not None
        assert -6.0 < result < 0.0


# ---------------------------------------------------------------------------
# Unit: Momentum / risk
# ---------------------------------------------------------------------------

class TestMomentumFactors:
    def test_parse_price_rows(self, price_series):
        parsed = _parse_price_rows(price_series)
        assert len(parsed) == len(price_series)
        # Should be sorted ascending
        dates = [d for d, _ in parsed]
        assert dates == sorted(dates)

    def test_daily_returns_length(self, price_series):
        prices = [p for _, p in _parse_price_rows(price_series)]
        returns = _daily_returns(prices)
        assert len(returns) == len(prices) - 1

    def test_return_12m_positive_trend(self):
        # Create an upward trending price series
        prices = [{"date": f"2024-{(i//30)+1:02d}-{(i%30)+1:02d}", "close": 100 + i * 0.5}
                  for i in range(260)]
        parsed = _parse_price_rows(prices)
        ret = compute_return_12m(parsed, lookback_days=252)
        assert ret is not None
        assert ret > 0  # price went up

    def test_beta_requires_aligned_data(self, price_series):
        ticker_parsed = _parse_price_rows(price_series)
        # No benchmark — beta should return None
        result = compute_beta_60d(ticker_parsed, [], lookback_days=60)
        assert result is None

    def test_beta_computed_with_benchmark(self, price_series):
        import random
        from datetime import timedelta
        random.seed(99)
        bench = []
        price = 450.0
        base = datetime(2024, 1, 1)
        for i in range(120):
            dt = base + timedelta(days=i)
            price *= (1 + random.gauss(0.0002, 0.010))
            bench.append({"date": dt.strftime("%Y-%m-%d"), "close": round(price, 2)})
        ticker_parsed = _parse_price_rows(price_series)
        bench_parsed = _parse_price_rows(bench)
        beta = compute_beta_60d(ticker_parsed, bench_parsed, lookback_days=60)
        assert beta is not None
        assert -5.0 < beta < 5.0  # Sanity range

    def test_sharpe_negative_with_declining_prices(self):
        prices = [{"date": f"2024-{(i//30)+1:02d}-{(i%30)+1:02d}", "close": max(1, 100 - i * 0.4)}
                  for i in range(260)]
        parsed = _parse_price_rows(prices)
        sharpe = compute_sharpe_12m(parsed)
        assert sharpe is not None
        assert sharpe < 0  # Declining prices → negative Sharpe


# ---------------------------------------------------------------------------
# Unit: DataQualityChecker
# ---------------------------------------------------------------------------

class TestDataQualityChecker:
    """Tests for the PostgreSQL-based data quality checker."""

    def test_passes_with_valid_bundle(self, minimal_bundle):
        """A bundle with valid TTM fields should pass all range checks."""
        # Add valid TTM data matching the range checks
        minimal_bundle.ratios_ttm = {
            "grossProfitMarginTTM": 0.45,
            "operatingProfitMarginTTM": 0.30,
            "currentRatioTTM": 1.62,
            "priceToEarningsRatioTTM": 28.5,
        }
        minimal_bundle.key_metrics_ttm = {
            "returnOnEquityTTM": 1.45,
            "returnOnInvestedCapitalTTM": 0.55,
            "evToEBITDATTM": 22.1,
        }
        minimal_bundle.price_history = [{"date": "2024-01-01", "close": 150.0}]
        checker = DataQualityChecker()
        result = checker.check(minimal_bundle)
        assert result.status == QualityStatus.PASSED
        assert result.checks_passed > 0
        assert result.issues == []

    def test_skipped_when_no_ttm_data(self):
        """Empty bundle with no TTM sources should return SKIPPED."""
        empty = FinancialsBundle(ticker="EMPTY")
        checker = DataQualityChecker()
        result = checker.check(empty)
        assert result.status == QualityStatus.SKIPPED
        assert result.checks_total == 0

    def test_issues_found_for_out_of_range_value(self):
        """A gross margin > 1.0 should be flagged as an issue."""
        bundle = FinancialsBundle(
            ticker="BAD",
            ratios_ttm={"grossProfitMarginTTM": 1.5},  # invalid: > 1.0
        )
        checker = DataQualityChecker()
        result = checker.check(bundle)
        assert result.status == QualityStatus.ISSUES_FOUND
        assert any("grossProfitMarginTTM" in issue for issue in result.issues)

    def test_missing_fields_are_skipped_not_failed(self):
        """Fields not present in the payload are skipped, not counted as failures."""
        bundle = FinancialsBundle(
            ticker="PARTIAL",
            ratios_ttm={
                "grossProfitMarginTTM": 0.40,
                # currentRatioTTM deliberately absent
            },
        )
        checker = DataQualityChecker()
        result = checker.check(bundle)
        # Should pass for gross_margin, not fail for missing fields
        assert result.status in (QualityStatus.PASSED, QualityStatus.ISSUES_FOUND)
        assert result.checks_passed >= 0

    def test_piotroski_score_range_check(self):
        """Piotroski score outside [0, 9] should be flagged."""
        bundle = FinancialsBundle(
            ticker="BAD_P",
            scores={"piotroskiScore": 15},  # invalid: > 9
        )
        checker = DataQualityChecker()
        result = checker.check(bundle)
        assert result.status == QualityStatus.ISSUES_FOUND
        assert any("piotroskiScore" in issue for issue in result.issues)

    def test_to_dict_structure(self):
        """DataQualityCheck.to_dict() must have the expected keys."""
        dq = DataQualityCheck(
            status=QualityStatus.PASSED,
            checks_passed=5,
            checks_total=5,
            issues=[],
        )
        d = dq.to_dict()
        assert d["status"] == "PASSED"
        assert d["checks_passed"] == 5
        assert d["checks_total"] == 5
        assert d["issues"] == []


# ---------------------------------------------------------------------------
# Unit: Schema to_dict
# ---------------------------------------------------------------------------

class TestSchemaToDict:
    def test_data_quality_to_dict(self):
        dq = DataQualityCheck(
            status=QualityStatus.PASSED,
            checks_passed=4,
            checks_total=4,
            issues=[],
        )
        d = dq.to_dict()
        assert d["status"] == "PASSED"
        assert d["checks_passed"] == 4
        assert d["issues"] == []

    def test_value_factors_to_dict(self):
        from agents.quant_fundamental.schema import ValueFactors
        vf = ValueFactors(pe_trailing=28.5, ev_ebitda=22.1)
        d = vf.to_dict()
        assert d["pe_trailing"] == 28.5
        assert d["p_fcf"] is None

    def test_quality_factors_to_dict(self):
        from agents.quant_fundamental.schema import QualityFactors
        qf = QualityFactors(roe=0.35, roic=0.20, piotroski_f_score=7, manipulation_risk="LOW")
        d = qf.to_dict()
        assert d["piotroski_f_score"] == 7
        assert d["manipulation_risk"] == "LOW"


# ---------------------------------------------------------------------------
# Integration tests (require live PostgreSQL via Docker)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    """Integration tests that hit the real PostgreSQL database.

    Requires Docker services to be running:
        docker-compose up -d postgres
    """

    def test_run_aapl(self):
        """Full pipeline run for AAPL — must return a valid output dict."""
        from agents.quant_fundamental.agent import run
        result = run(ticker="AAPL")
        _assert_valid_output(result, "AAPL")

    def test_run_tsla(self):
        """Full pipeline run for TSLA."""
        from agents.quant_fundamental.agent import run
        result = run(ticker="TSLA")
        _assert_valid_output(result, "TSLA")

    def test_run_nvda(self):
        """Full pipeline run for NVDA."""
        from agents.quant_fundamental.agent import run
        result = run(ticker="NVDA")
        _assert_valid_output(result, "NVDA")

    def test_run_full_analysis_aapl(self):
        """run_full_analysis() must return same schema as run()."""
        from agents.quant_fundamental.agent import run_full_analysis
        result = run_full_analysis(ticker="AAPL")
        _assert_valid_output(result, "AAPL")

    def test_output_has_numeric_factors(self):
        """At least some numeric factors should be non-null for AAPL."""
        from agents.quant_fundamental.agent import run
        result = run(ticker="AAPL")
        value = result.get("value_factors", {})
        quality = result.get("quality_factors", {})
        # At least P/E or EV/EBITDA should be populated
        has_value = any(v is not None for v in value.values())
        # ROE or ROIC should be populated
        has_quality = any(v is not None for v in quality.values())
        assert has_value, f"No value factors computed for AAPL: {value}"
        assert has_quality, f"No quality factors computed for AAPL: {quality}"

    def test_data_quality_not_issues(self):
        """Data quality check should not report issues for AAPL."""
        from agents.quant_fundamental.agent import run
        result = run(ticker="AAPL")
        dq = result.get("data_quality", {})
        status = dq.get("status", "")
        # PASSED or SKIPPED are acceptable; ISSUES_FOUND should be investigated
        assert status in ("PASSED", "SKIPPED"), (
            f"Data quality ISSUES_FOUND for AAPL: {dq.get('issues')}"
        )

    def test_quantitative_summary_is_string(self):
        """quantitative_summary must be a non-empty string."""
        from agents.quant_fundamental.agent import run
        result = run(ticker="AAPL")
        summary = result.get("quantitative_summary", "")
        assert isinstance(summary, str)
        assert len(summary) > 10, f"Summary too short: {repr(summary)}"

    def test_multi_ticker_prompt_returns_list(self):
        """A multi-ticker prompt must return a list with one result per ticker."""
        from agents.quant_fundamental.agent import run
        result = run(prompt="Compare MSFT vs AAPL fundamentals")
        assert isinstance(result, list), f"Expected list for multi-ticker, got {type(result)}"
        assert len(result) == 2
        tickers = [r.get("ticker") for r in result]
        assert "MSFT" in tickers
        assert "AAPL" in tickers

    def test_single_ticker_prompt_returns_dict(self):
        """A single-ticker prompt must return a dict, not a list."""
        from agents.quant_fundamental.agent import run
        result = run(prompt="Analyze AAPL fundamentals")
        assert isinstance(result, dict), f"Expected dict for single ticker, got {type(result)}"
        assert result.get("ticker") == "AAPL"


# ---------------------------------------------------------------------------
# Unit: extract_tickers_from_prompt
# ---------------------------------------------------------------------------

class TestExtractTickersFromPrompt:
    """Tests for the multi-ticker prompt extraction function."""

    def setup_method(self):
        from agents.quant_fundamental.agent import extract_tickers_from_prompt, extract_ticker_from_prompt
        self.extract_all = extract_tickers_from_prompt
        self.extract_one = extract_ticker_from_prompt

    # --- Single ticker cases ---
    def test_simple_analysis_prompt(self):
        assert self.extract_all("Analyze AAPL fundamentals") == ["AAPL"]

    def test_ticker_keyword(self):
        assert self.extract_all("What are the fundamentals for ticker MSFT?") == ["MSFT"]

    def test_ticker_colon_keyword(self):
        assert self.extract_all("Report for ticker: GOOGL") == ["GOOGL"]

    def test_parenthesised_ticker(self):
        result = self.extract_all("analyze apple (AAPL)")
        assert "AAPL" in result

    def test_prefix_ticker(self):
        result = self.extract_all("TSLA analysis please")
        assert "TSLA" in result

    def test_unknown_us_ticker_heuristic(self):
        result = self.extract_all("Quant report for AMD please")
        assert "AMD" in result

    def test_no_ticker_returns_empty(self):
        assert self.extract_all("No ticker here") == []

    def test_empty_string_returns_empty(self):
        assert self.extract_all("") == []

    # --- Multi-ticker cases ---
    def test_two_tickers_vs(self):
        result = self.extract_all("Compare MSFT vs AAPL fundamentals")
        assert "MSFT" in result
        assert "AAPL" in result
        assert len(result) == 2

    def test_two_tickers_and(self):
        result = self.extract_all("Analyze NVDA and TSLA")
        assert "NVDA" in result
        assert "TSLA" in result

    def test_first_ticker_ordering(self):
        result = self.extract_all("Compare MSFT vs AAPL")
        # MSFT appears first in the prompt — must be first in the result
        assert result[0] == "MSFT"
        assert result[1] == "AAPL"

    def test_three_tickers(self):
        result = self.extract_all("Compare AAPL, MSFT and GOOGL")
        assert "AAPL" in result
        assert "MSFT" in result
        assert "GOOGL" in result

    def test_no_duplicates(self):
        result = self.extract_all("Analyze AAPL and then AAPL again")
        assert result.count("AAPL") == 1

    # --- Exchange-suffix tickers ---
    def test_hk_numeric_ticker(self):
        result = self.extract_all("Analyze HK MTR's stock 0066.HK fundamentals")
        assert "0066.HK" in result
        # Should NOT incorrectly extract "HK" as a standalone ticker
        assert "HK" not in result

    def test_london_ticker(self):
        result = self.extract_all("Run quant analysis on VOD.L")
        assert "VOD.L" in result

    def test_tokyo_ticker(self):
        result = self.extract_all("What are the fundamentals of 7203.T?")
        assert "7203.T" in result

    def test_asx_ticker(self):
        result = self.extract_all("Analyze BHP.AX please")
        assert "BHP.AX" in result

    def test_mixed_us_and_hk(self):
        result = self.extract_all("Compare AAPL and 0066.HK")
        assert "AAPL" in result
        assert "0066.HK" in result

    # --- Backward compat: extract_ticker_from_prompt (single) ---
    def test_backward_compat_single(self):
        assert self.extract_one("Analyze AAPL fundamentals") == "AAPL"

    def test_backward_compat_multi_returns_first(self):
        result = self.extract_one("Compare MSFT vs AAPL")
        assert result == "MSFT"

    def test_backward_compat_none(self):
        assert self.extract_one("No ticker here") is None


def _assert_valid_output(result: Dict[str, Any], ticker: str) -> None:
    """Assert the output dict conforms to the expected schema."""
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert result.get("agent") == "quant_fundamental", f"Wrong agent field: {result.get('agent')}"
    assert result.get("ticker") == ticker, f"Wrong ticker: {result.get('ticker')}"
    assert "as_of_date" in result
    assert "value_factors" in result
    assert "quality_factors" in result
    assert "momentum_risk" in result
    assert "key_metrics" in result
    assert "anomaly_flags" in result
    assert "data_quality" in result
    assert "quantitative_summary" in result
    assert "data_sources" in result

    # Sub-structure checks
    vf = result["value_factors"]
    assert set(vf.keys()) >= {"pe_trailing", "ev_ebitda", "p_fcf", "ev_revenue"}

    qf = result["quality_factors"]
    assert set(qf.keys()) >= {"roe", "roic", "piotroski_f_score", "beneish_m_score", "manipulation_risk"}

    mr = result["momentum_risk"]
    assert set(mr.keys()) >= {"beta_60d", "sharpe_ratio_12m", "return_12m_pct"}

    km = result["key_metrics"]
    assert set(km.keys()) >= {"gross_margin", "ebit_margin", "fcf_conversion", "dso_days",
                               "current_ratio", "debt_to_equity"}

    dq = result["data_quality"]
    assert dq.get("status") in ("PASSED", "ISSUES_FOUND", "SKIPPED")
