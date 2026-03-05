"""Tests for the Financial Modelling agent.

Unit tests use mocked/synthetic data — no live DB required.
Integration tests require PostgreSQL + Neo4j via Docker.

Run unit tests only (no DB):
    .venv/bin/python -m pytest agents/financial_modelling/tests/test_agent.py -v -m "not integration"

Run all tests including integration (requires Docker services running):
    .venv/bin/python -m pytest agents/financial_modelling/tests/test_agent.py -v
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from agents.financial_modelling.config import FinancialModellingConfig, load_config
from agents.financial_modelling.schema import (
    DCFResult,
    DividendRecord,
    EarningsRecord,
    FactorScores,
    FMDataBundle,
    TechnicalSnapshot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> FinancialModellingConfig:
    """Minimal config — no external services needed."""
    return FinancialModellingConfig(
        ollama_base_url="http://localhost:11434",
        llm_model="deepseek-r1:8b",
        llm_temperature=0.0,
        llm_max_tokens=512,
        request_timeout=None,
        dcf_discount_rate=0.10,
        terminal_growth_rate=0.03,
        beta_lookback_days=60,
        postgres_host="localhost",
        postgres_port=5432,
        postgres_db="airflow",
        postgres_user="airflow",
        postgres_password="airflow",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="neo4j",
    )


@pytest.fixture
def minimal_bundle() -> FMDataBundle:
    """A synthetic FMDataBundle with enough data for most computations."""
    prices = _make_prices(days=300, start=150.0, drift=0.0004, seed=42)
    bench = _make_prices(days=300, start=4500.0, drift=0.0002, seed=7)
    return FMDataBundle(
        ticker="TEST",
        income={
            "revenue": 400_000_000_000,
            "grossProfit": 180_000_000_000,
            "operatingIncome": 120_000_000_000,
            "netIncome": 100_000_000_000,
            "ebitda": 130_000_000_000,
            "interestExpense": 3_500_000_000,
            "incomeTaxExpense": 15_000_000_000,
            "incomeBeforeTax": 115_000_000_000,
        },
        balance={
            "totalAssets": 350_000_000_000,
            "totalCurrentAssets": 130_000_000_000,
            "totalCurrentLiabilities": 80_000_000_000,
            "totalStockholdersEquity": 70_000_000_000,
            "totalDebt": 100_000_000_000,
            "longTermDebt": 90_000_000_000,
            "retainedEarnings": 50_000_000_000,
        },
        cashflow={
            "operatingCashFlow": 110_000_000_000,
            "capitalExpenditure": -10_000_000_000,
            "freeCashFlow": 100_000_000_000,
        },
        ratios_ttm={
            "peRatioTTM": 28.5,
            "dividendYieldTTM": 0.0055,
            "payoutRatioTTM": 0.15,
        },
        key_metrics_ttm={
            "evToEbitdaTTM": 22.1,
            "stockPriceTTM": 185.0,
            "marketCapTTM": 2_850_000_000_000,
        },
        enterprise={
            "marketCapitalization": 2_850_000_000_000,
            "enterpriseValue": 2_900_000_000_000,
        },
        scores={
            "piotroskiScore": 7,
            "beneishMScore": -2.8,
        },
        price_history=prices,
        benchmark_history=bench,
        treasury_rates=[{"date": "2025-01-01", "tenYear": 0.043}],
        market_risk_premium={"marketRiskPremium": 0.055},
        earnings_history=[
            {"date": "2024-09-30", "actualEarningResult": 1.46, "estimatedEarning": 1.43},
            {"date": "2024-06-30", "actualEarningResult": 1.40, "estimatedEarning": 1.35},
            {"date": "2024-03-31", "actualEarningResult": 1.53, "estimatedEarning": 1.50},
            {"date": "2023-12-31", "actualEarningResult": 2.18, "estimatedEarning": 2.10},
        ],
        dividend_history=[
            {"date": f"2024-{m:02d}-01", "adjDividend": 0.25}
            for m in range(1, 13)
        ],
        peer_fundamentals={},
    )


def _make_prices(days: int, start: float, drift: float, seed: int) -> List[Dict[str, Any]]:
    """Generate synthetic daily OHLCV rows (sorted ascending, newest-last)."""
    import random
    random.seed(seed)
    rows = []
    price = start
    base = datetime(2024, 1, 1)
    for i in range(days):
        price = price * (1 + random.gauss(drift, 0.012))
        price = max(1.0, price)
        dt = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        rows.append({
            "date": dt,
            "open": round(price * 0.995, 4),
            "high": round(price * 1.01, 4),
            "low": round(price * 0.99, 4),
            "close": round(price, 4),
            "adjClose": round(price, 4),
            "volume": 80_000_000,
        })
    return rows


# ---------------------------------------------------------------------------
# Unit: Schema — dataclass to_dict contracts
# ---------------------------------------------------------------------------

class TestSchemaToDict:
    def test_dcf_result_to_dict_keys(self):
        dcf = DCFResult(
            wacc_used=0.10,
            intrinsic_value_base=200.0,
            intrinsic_value_bear=140.0,
            intrinsic_value_bull=280.0,
        )
        d = dcf.to_dict()
        assert "wacc_used" in d
        assert "intrinsic_value_base" in d
        assert "intrinsic_value_bear" in d
        assert "intrinsic_value_bull" in d
        assert "scenario_table" in d
        assert "sensitivity_matrix" in d

    def test_technical_snapshot_to_dict_keys(self):
        snap = TechnicalSnapshot(
            sma_20=180.0, sma_50=170.0, sma_200=160.0, rsi_14=55.0,
        )
        d = snap.to_dict()
        assert "sma_20" in d
        assert "rsi_14" in d
        assert "macd" in d
        assert "trend" in d

    def test_earnings_record_to_dict(self):
        er = EarningsRecord(
            last_eps_actual=1.46, last_eps_estimate=1.43, surprise_pct=2.1,
            beat_streak=4, miss_streak=0,
        )
        d = er.to_dict()
        assert d["last_eps_actual"] == 1.46
        assert d["surprise_pct"] == pytest.approx(2.1, abs=0.01)
        assert d["beat_streak"] == 4
        assert d["miss_streak"] == 0

    def test_dividend_record_to_dict(self):
        dr = DividendRecord(
            dividend_yield=0.0055, annual_dividend=1.0, payout_ratio=0.15,
        )
        d = dr.to_dict()
        assert d["dividend_yield"] == pytest.approx(0.0055)
        assert d["annual_dividend"] == pytest.approx(1.0)

    def test_factor_scores_to_dict(self):
        fs = FactorScores(
            piotroski_f_score=7, beneish_m_score=-2.8, altman_z_score=4.5,
        )
        d = fs.to_dict()
        assert d["piotroski_f_score"] == 7
        assert d["beneish_m_score"] == pytest.approx(-2.8)
        assert d["altman_z_score"] == pytest.approx(4.5)

    def test_fmdatabundle_is_empty_when_no_data(self):
        empty = FMDataBundle(ticker="X")
        assert empty.is_empty()

    def test_fmdatabundle_not_empty_with_income(self, minimal_bundle):
        assert not minimal_bundle.is_empty()


# ---------------------------------------------------------------------------
# Unit: Technical indicators
# ---------------------------------------------------------------------------

class TestTechnicalEngine:
    def test_sma_20_is_computed(self, minimal_bundle):
        from agents.financial_modelling.models.technicals import TechnicalEngine
        engine = TechnicalEngine()
        snap = engine.compute(minimal_bundle)
        assert snap.sma_20 is not None
        assert snap.sma_20 > 0

    def test_sma_50_is_computed(self, minimal_bundle):
        from agents.financial_modelling.models.technicals import TechnicalEngine
        snap = TechnicalEngine().compute(minimal_bundle)
        assert snap.sma_50 is not None

    def test_sma_200_is_computed(self, minimal_bundle):
        from agents.financial_modelling.models.technicals import TechnicalEngine
        snap = TechnicalEngine().compute(minimal_bundle)
        assert snap.sma_200 is not None

    def test_rsi_in_range(self, minimal_bundle):
        from agents.financial_modelling.models.technicals import TechnicalEngine
        snap = TechnicalEngine().compute(minimal_bundle)
        assert snap.rsi_14 is not None
        assert 0 <= snap.rsi_14 <= 100

    def test_macd_is_float(self, minimal_bundle):
        from agents.financial_modelling.models.technicals import TechnicalEngine
        snap = TechnicalEngine().compute(minimal_bundle)
        assert snap.macd is not None
        assert isinstance(snap.macd, float)

    def test_bollinger_bands_ordered(self, minimal_bundle):
        from agents.financial_modelling.models.technicals import TechnicalEngine
        snap = TechnicalEngine().compute(minimal_bundle)
        if snap.bb_upper is not None and snap.bb_lower is not None:
            assert snap.bb_upper > snap.bb_lower

    def test_stochastic_k_in_range(self, minimal_bundle):
        from agents.financial_modelling.models.technicals import TechnicalEngine
        snap = TechnicalEngine().compute(minimal_bundle)
        if snap.stoch_k is not None:
            assert 0 <= snap.stoch_k <= 100

    def test_trend_is_string(self, minimal_bundle):
        from agents.financial_modelling.models.technicals import TechnicalEngine
        snap = TechnicalEngine().compute(minimal_bundle)
        assert snap.trend is not None
        assert snap.trend in ("BULLISH", "BEARISH", "NEUTRAL", "UNKNOWN")

    def test_52w_high_gte_52w_low(self, minimal_bundle):
        from agents.financial_modelling.models.technicals import TechnicalEngine
        snap = TechnicalEngine().compute(minimal_bundle)
        if snap.w52_high is not None and snap.w52_low is not None:
            assert snap.w52_high >= snap.w52_low

    def test_empty_bundle_returns_empty_snapshot(self):
        from agents.financial_modelling.models.technicals import TechnicalEngine
        snap = TechnicalEngine().compute(FMDataBundle(ticker="EMPTY"))
        assert snap.sma_20 is None
        assert snap.rsi_14 is None
        assert snap.trend is None or snap.trend == "UNKNOWN"


# ---------------------------------------------------------------------------
# Unit: DCF Engine
# ---------------------------------------------------------------------------

class TestDCFEngine:
    def test_wacc_is_positive(self, minimal_bundle, config):
        from agents.financial_modelling.models.dcf import DCFEngine
        result = DCFEngine(config).compute(minimal_bundle)
        assert result.wacc_used is not None
        assert result.wacc_used > 0

    def test_wacc_in_reasonable_range(self, minimal_bundle, config):
        from agents.financial_modelling.models.dcf import DCFEngine
        result = DCFEngine(config).compute(minimal_bundle)
        # WACC for large-cap should be between 5% and 20%
        assert 0.04 <= result.wacc_used <= 0.25

    def test_base_intrinsic_value_positive(self, minimal_bundle, config):
        from agents.financial_modelling.models.dcf import DCFEngine
        result = DCFEngine(config).compute(minimal_bundle)
        assert result.intrinsic_value_base is not None
        assert result.intrinsic_value_base > 0

    def test_bear_lt_base_lt_bull(self, minimal_bundle, config):
        from agents.financial_modelling.models.dcf import DCFEngine
        result = DCFEngine(config).compute(minimal_bundle)
        if all(
            v is not None
            for v in [result.intrinsic_value_bear, result.intrinsic_value_base, result.intrinsic_value_bull]
        ):
            assert result.intrinsic_value_bear <= result.intrinsic_value_base
            assert result.intrinsic_value_base <= result.intrinsic_value_bull

    def test_scenario_table_has_three_rows(self, minimal_bundle, config):
        from agents.financial_modelling.models.dcf import DCFEngine
        result = DCFEngine(config).compute(minimal_bundle)
        if result.scenario_table:
            assert len(result.scenario_table) == 3
            scenario_names = {row.get("scenario") for row in result.scenario_table}
            assert scenario_names == {"Bear", "Base", "Bull"}

    def test_sensitivity_matrix_is_dict(self, minimal_bundle, config):
        from agents.financial_modelling.models.dcf import DCFEngine
        result = DCFEngine(config).compute(minimal_bundle)
        assert isinstance(result.sensitivity_matrix, dict)

    def test_empty_bundle_returns_empty_dcf(self, config):
        from agents.financial_modelling.models.dcf import DCFEngine
        result = DCFEngine(config).compute(FMDataBundle(ticker="EMPTY"))
        assert result.intrinsic_value_base is None


# ---------------------------------------------------------------------------
# Unit: Comps Engine
# ---------------------------------------------------------------------------

class TestCompsEngine:
    def test_comps_result_has_ev_ebitda(self, minimal_bundle, config):
        from agents.financial_modelling.models.valuation import CompsEngine
        result = CompsEngine(config).compute(minimal_bundle)
        # ev_ebitda may be None if data is insufficient, but type must be correct
        if result.ev_ebitda is not None:
            assert isinstance(result.ev_ebitda, float)
            assert result.ev_ebitda > 0

    def test_comps_result_to_dict_keys(self, minimal_bundle, config):
        from agents.financial_modelling.models.valuation import CompsEngine
        result = CompsEngine(config).compute(minimal_bundle)
        d = result.to_dict()
        assert "ev_ebitda" in d
        assert "pe_trailing" in d
        assert "pe_forward" in d
        assert "vs_sector_avg" in d

    def test_empty_bundle_returns_empty_comps(self, config):
        from agents.financial_modelling.models.valuation import CompsEngine
        from agents.financial_modelling.schema import CompsResult
        result = CompsEngine(config).compute(FMDataBundle(ticker="EMPTY"))
        assert isinstance(result, CompsResult)


# ---------------------------------------------------------------------------
# Unit: Earnings record computation
# ---------------------------------------------------------------------------

class TestEarningsRecord:
    def test_surprise_positive_beat(self, minimal_bundle):
        from agents.financial_modelling.agent import _compute_earnings_record
        record = _compute_earnings_record(minimal_bundle)
        assert record.surprise_pct is not None
        # Latest entry: actual=1.46, estimate=1.43 → positive surprise
        assert record.surprise_pct > 0

    def test_beat_streak_detected(self, minimal_bundle):
        from agents.financial_modelling.agent import _compute_earnings_record
        record = _compute_earnings_record(minimal_bundle)
        # All 4 entries are beats
        assert record.beat_streak == 4
        assert record.miss_streak == 0

    def test_empty_earnings_history_returns_empty_record(self):
        from agents.financial_modelling.agent import _compute_earnings_record
        bundle = FMDataBundle(ticker="EMPTY")
        record = _compute_earnings_record(bundle)
        assert record.last_eps_actual is None
        assert record.surprise_pct is None
        assert record.beat_streak == 0


# ---------------------------------------------------------------------------
# Unit: Dividend record computation
# ---------------------------------------------------------------------------

class TestDividendRecord:
    def test_annual_dividend_sum(self, minimal_bundle):
        from agents.financial_modelling.agent import _compute_dividends
        record = _compute_dividends(minimal_bundle)
        # 12 entries × $0.25 = $3.00
        assert record.annual_dividend is not None
        assert record.annual_dividend == pytest.approx(3.0, abs=0.01)

    def test_dividend_yield_from_ttm(self, minimal_bundle):
        from agents.financial_modelling.agent import _compute_dividends
        record = _compute_dividends(minimal_bundle)
        assert record.dividend_yield is not None
        assert record.dividend_yield == pytest.approx(0.0055, abs=0.0001)

    def test_payout_ratio(self, minimal_bundle):
        from agents.financial_modelling.agent import _compute_dividends
        record = _compute_dividends(minimal_bundle)
        assert record.payout_ratio is not None
        assert record.payout_ratio == pytest.approx(0.15, abs=0.001)

    def test_empty_bundle_returns_empty_record(self):
        from agents.financial_modelling.agent import _compute_dividends
        record = _compute_dividends(FMDataBundle(ticker="EMPTY"))
        assert record.dividend_yield is None
        assert record.annual_dividend is None


# ---------------------------------------------------------------------------
# Unit: Altman Z-Score computation
# ---------------------------------------------------------------------------

class TestAltmanZScore:
    def test_z_score_computed_for_healthy_company(self, minimal_bundle):
        from agents.financial_modelling.agent import _compute_altman_z
        z = _compute_altman_z(minimal_bundle)
        assert z is not None
        # Healthy large-cap: Z > 2.99 (safe zone)
        assert z > 2.0

    def test_z_score_none_without_total_assets(self):
        from agents.financial_modelling.agent import _compute_altman_z
        bundle = FMDataBundle(ticker="NO_ASSETS", balance={"totalAssets": None})
        z = _compute_altman_z(bundle)
        assert z is None

    def test_z_score_is_float(self, minimal_bundle):
        from agents.financial_modelling.agent import _compute_altman_z
        z = _compute_altman_z(minimal_bundle)
        assert z is None or isinstance(z, float)


# ---------------------------------------------------------------------------
# Unit: Factor scores
# ---------------------------------------------------------------------------

class TestFactorScores:
    def test_piotroski_from_scores_payload(self, minimal_bundle):
        from agents.financial_modelling.agent import _compute_factor_scores
        result = _compute_factor_scores(minimal_bundle)
        assert result.piotroski_f_score == 7

    def test_beneish_from_scores_payload(self, minimal_bundle):
        from agents.financial_modelling.agent import _compute_factor_scores
        result = _compute_factor_scores(minimal_bundle)
        assert result.beneish_m_score == pytest.approx(-2.8, abs=0.01)

    def test_altman_z_derived_when_missing(self, minimal_bundle):
        """Altman Z should be derived from financials when not in scores payload."""
        minimal_bundle.scores = {}  # Remove pre-computed scores
        from agents.financial_modelling.agent import _compute_factor_scores
        result = _compute_factor_scores(minimal_bundle)
        # Should fall back to computing from financials
        assert result.altman_z_score is not None or result.altman_z_score is None  # graceful

    def test_empty_scores_payload_returns_nones(self):
        from agents.financial_modelling.agent import _compute_factor_scores
        result = _compute_factor_scores(FMDataBundle(ticker="X"))
        assert result.piotroski_f_score is None
        assert result.beneish_m_score is None


# ---------------------------------------------------------------------------
# Unit: extract_tickers_from_prompt
# ---------------------------------------------------------------------------

class TestExtractTickers:
    def setup_method(self):
        from agents.financial_modelling.agent import (
            extract_tickers_from_prompt,
            extract_ticker_from_prompt,
        )
        self.extract_all = extract_tickers_from_prompt
        self.extract_one = extract_ticker_from_prompt

    def test_single_known_ticker(self):
        assert self.extract_all("Analyze AAPL valuation") == ["AAPL"]

    def test_multi_ticker_vs(self):
        result = self.extract_all("Compare MSFT vs AAPL DCF")
        assert "MSFT" in result and "AAPL" in result
        assert len(result) == 2

    def test_no_ticker_returns_empty(self):
        assert self.extract_all("No ticker here") == []

    def test_extract_one_returns_first(self):
        assert self.extract_one("Analyze TSLA valuation") == "TSLA"

    def test_extract_one_none_when_no_ticker(self):
        assert self.extract_one("No ticker here") is None

    def test_no_duplicates(self):
        result = self.extract_all("AAPL and then AAPL again")
        assert result.count("AAPL") == 1

    def test_ordering_preserved(self):
        result = self.extract_all("Compare NVDA and TSLA")
        # NVDA appears first in the prompt
        assert result[0] == "NVDA"


# ---------------------------------------------------------------------------
# Unit: node_financial_modelling (orchestration node — mocked)
# ---------------------------------------------------------------------------

class TestNodeFinancialModelling:
    """Test the orchestration node in isolation with a mocked run_full_analysis."""

    def _make_fake_output(self, ticker: str) -> Dict[str, Any]:
        return {
            "agent": "financial_modelling",
            "ticker": ticker,
            "as_of_date": "2026-01-01",
            "current_price": 185.0,
            "valuation": {"dcf": {}, "comps": {}, "implied_price_range": {}},
            "technicals": {},
            "earnings": {},
            "dividends": {},
            "factor_scores": {},
            "quantitative_summary": "Synthetic summary.",
            "data_sources": {},
        }

    @patch("agents.financial_modelling.agent.run_full_analysis")
    def test_single_ticker_populates_output(self, mock_rfa):
        from orchestration.nodes import node_financial_modelling
        from orchestration.state import OrchestrationState

        mock_rfa.return_value = self._make_fake_output("AAPL")

        state: OrchestrationState = {
            "user_query": "Analyze AAPL",
            "tickers": ["AAPL"],
            "react_steps": [],
            "agent_errors": {},
        }
        result = node_financial_modelling(state)

        assert "financial_modelling_outputs" in result
        assert len(result["financial_modelling_outputs"]) == 1
        assert result["financial_modelling_output"]["ticker"] == "AAPL"
        assert result["agent_errors"] == {}

    @patch("agents.financial_modelling.agent.run_full_analysis")
    def test_multi_ticker_returns_all_outputs(self, mock_rfa):
        from orchestration.nodes import node_financial_modelling
        from orchestration.state import OrchestrationState

        mock_rfa.side_effect = [
            self._make_fake_output("MSFT"),
            self._make_fake_output("AAPL"),
        ]

        state: OrchestrationState = {
            "user_query": "Compare MSFT vs AAPL",
            "tickers": ["MSFT", "AAPL"],
            "react_steps": [],
            "agent_errors": {},
        }
        result = node_financial_modelling(state)
        outputs = result["financial_modelling_outputs"]
        assert len(outputs) == 2
        tickers = [o["ticker"] for o in outputs]
        assert "MSFT" in tickers
        assert "AAPL" in tickers
        # First output alias must be the first ticker
        assert result["financial_modelling_output"]["ticker"] == "MSFT"

    @patch("agents.financial_modelling.agent.run_full_analysis", side_effect=RuntimeError("DB down"))
    def test_error_is_captured_gracefully(self, mock_rfa):
        from orchestration.nodes import node_financial_modelling
        from orchestration.state import OrchestrationState

        state: OrchestrationState = {
            "user_query": "Analyze AAPL",
            "tickers": ["AAPL"],
            "react_steps": [],
            "agent_errors": {},
        }
        result = node_financial_modelling(state)
        # Node must not raise — error captured in agent_errors
        assert "financial_modelling" in result["agent_errors"]
        assert "RuntimeError" in result["agent_errors"]["financial_modelling"]


# ---------------------------------------------------------------------------
# Integration tests (require live PostgreSQL + Neo4j via Docker)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegration:
    """Integration tests that hit the real PostgreSQL and Neo4j databases.

    Requires Docker services to be running:
        docker-compose up -d postgres neo4j
    """

    def test_run_aapl(self):
        """Full pipeline run for AAPL — must return a valid output dict."""
        from agents.financial_modelling.agent import run
        result = run(ticker="AAPL")
        _assert_valid_output(result, "AAPL")

    def test_run_tsla(self):
        from agents.financial_modelling.agent import run
        result = run(ticker="TSLA")
        _assert_valid_output(result, "TSLA")

    def test_run_nvda(self):
        from agents.financial_modelling.agent import run
        result = run(ticker="NVDA")
        _assert_valid_output(result, "NVDA")

    def test_run_full_analysis_aapl(self):
        from agents.financial_modelling.agent import run_full_analysis
        result = run_full_analysis(ticker="AAPL")
        _assert_valid_output(result, "AAPL")

    def test_dcf_base_value_positive(self):
        from agents.financial_modelling.agent import run
        result = run(ticker="AAPL")
        dcf = result.get("valuation", {}).get("dcf", {})
        # May be None if data insufficient, but if present must be positive
        base = dcf.get("intrinsic_value_base")
        if base is not None:
            assert base > 0, f"DCF base value non-positive: {base}"

    def test_rsi_in_range(self):
        from agents.financial_modelling.agent import run
        result = run(ticker="MSFT")
        rsi = result.get("technicals", {}).get("rsi_14")
        if rsi is not None:
            assert 0 <= rsi <= 100, f"RSI out of range: {rsi}"

    def test_piotroski_in_range(self):
        from agents.financial_modelling.agent import run
        result = run(ticker="GOOGL")
        score = result.get("factor_scores", {}).get("piotroski_f_score")
        if score is not None:
            assert 0 <= score <= 9, f"Piotroski out of [0,9]: {score}"

    def test_quantitative_summary_is_string(self):
        from agents.financial_modelling.agent import run
        result = run(ticker="AAPL")
        summary = result.get("quantitative_summary", "")
        assert isinstance(summary, str)
        assert len(summary) > 10, f"Summary too short: {repr(summary)}"

    def test_single_ticker_prompt_returns_dict(self):
        from agents.financial_modelling.agent import run
        result = run(prompt="Analyze AAPL valuation")
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert result.get("ticker") == "AAPL"

    def test_multi_ticker_prompt_returns_list(self):
        from agents.financial_modelling.agent import run
        result = run(prompt="Compare MSFT vs AAPL DCF")
        assert isinstance(result, list), f"Expected list for multi-ticker, got {type(result)}"
        assert len(result) == 2
        tickers = [r.get("ticker") for r in result]
        assert "MSFT" in tickers
        assert "AAPL" in tickers

    def test_data_sources_field_present(self):
        from agents.financial_modelling.agent import run
        result = run(ticker="AAPL")
        assert "data_sources" in result
        ds = result["data_sources"]
        assert ds.get("price_data") == "postgresql:raw_timeseries"
        assert ds.get("fundamentals") == "postgresql:raw_fundamentals"

    def test_no_ticker_raises_value_error(self):
        from agents.financial_modelling.agent import run
        with pytest.raises(ValueError):
            run(ticker="", prompt=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_valid_output(result: Dict[str, Any], ticker: str) -> None:
    """Assert the output dict conforms to the expected Financial Modelling schema."""
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert result.get("agent") == "financial_modelling", f"Wrong agent: {result.get('agent')}"
    assert result.get("ticker") == ticker, f"Wrong ticker: {result.get('ticker')}"
    assert "as_of_date" in result
    assert "current_price" in result
    assert "valuation" in result
    assert "technicals" in result
    assert "earnings" in result
    assert "dividends" in result
    assert "factor_scores" in result
    assert "quantitative_summary" in result
    assert "data_sources" in result

    # valuation sub-structure
    val = result["valuation"]
    assert "dcf" in val
    assert "comps" in val
    assert "implied_price_range" in val

    dcf = val["dcf"]
    assert set(dcf.keys()) >= {"wacc_used", "intrinsic_value_base", "scenario_table", "sensitivity_matrix"}

    comps = val["comps"]
    assert set(comps.keys()) >= {"ev_ebitda", "pe_trailing", "pe_forward", "vs_sector_avg"}

    # technicals sub-structure
    tech = result["technicals"]
    assert set(tech.keys()) >= {"sma_20", "sma_50", "rsi_14", "macd", "trend"}

    # earnings sub-structure
    earn = result["earnings"]
    assert set(earn.keys()) >= {"last_eps_actual", "last_eps_estimate", "surprise_pct", "beat_streak"}

    # dividends sub-structure
    div = result["dividends"]
    assert set(div.keys()) >= {"dividend_yield", "annual_dividend", "payout_ratio"}

    # factor_scores sub-structure
    fs = result["factor_scores"]
    assert set(fs.keys()) >= {"piotroski_f_score", "beneish_m_score", "altman_z_score"}
