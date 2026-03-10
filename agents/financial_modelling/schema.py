"""Shared dataclasses for the Financial Modelling agent.

All dataclasses represent Python-computed results — never LLM-generated numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# DCF / Valuation
# ---------------------------------------------------------------------------

@dataclass
class DCFResult:
    """Output of the DCF engine for one ticker."""

    intrinsic_value_base: Optional[float] = None
    intrinsic_value_bear: Optional[float] = None
    intrinsic_value_bull: Optional[float] = None
    intrinsic_value_weighted: Optional[float] = None  # probability-weighted: P(b)*bear + P(B)*base + P(u)*bull
    upside_pct_base: Optional[float] = None           # (base - price) / price * 100
    upside_pct_weighted: Optional[float] = None       # (weighted - price) / price * 100
    wacc_used: Optional[float] = None
    terminal_growth_rate: Optional[float] = None
    forecast_years: int = 10
    scenario_probability: Dict[str, float] = field(
        default_factory=lambda: {"bear": 0.25, "base": 0.55, "bull": 0.20}
    )
    # scenario_table: list of {"scenario", "revenue_growth", "ebit_margin", "wacc", "intrinsic_value"}
    scenario_table: List[Dict[str, Any]] = field(default_factory=list)
    # sensitivity_matrix[wacc_str][growth_str] = implied_value
    sensitivity_matrix: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    # Reverse DCF: implied revenue CAGR at current market price
    reverse_dcf_implied_cagr: Optional[float] = None
    # Beta and WACC components for transparency
    beta_used: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intrinsic_value_base": self.intrinsic_value_base,
            "intrinsic_value_bear": self.intrinsic_value_bear,
            "intrinsic_value_bull": self.intrinsic_value_bull,
            "intrinsic_value_weighted": self.intrinsic_value_weighted,
            "upside_pct_base": self.upside_pct_base,
            "upside_pct_weighted": self.upside_pct_weighted,
            "wacc_used": self.wacc_used,
            "terminal_growth_rate": self.terminal_growth_rate,
            "forecast_years": self.forecast_years,
            "scenario_probability": self.scenario_probability,
            "scenario_table": self.scenario_table,
            "sensitivity_matrix": self.sensitivity_matrix,
            "reverse_dcf_implied_cagr": self.reverse_dcf_implied_cagr,
            "beta_used": self.beta_used,
        }


@dataclass
class CompsResult:
    """Output of the Comparable Company Analysis."""

    ev_ebitda: Optional[float] = None
    ev_ebit: Optional[float] = None            # EV / EBIT — pure operating multiple
    pe_trailing: Optional[float] = None
    pe_forward: Optional[float] = None
    ps_ttm: Optional[float] = None
    ev_revenue: Optional[float] = None
    p_fcf: Optional[float] = None              # Market cap / Free cash flow to firm
    peg_ratio: Optional[float] = None          # P/E ÷ (EPS growth rate × 100)
    vs_sector_avg: Optional[str] = None        # e.g. "premium +18%" or "discount -5%"
    peer_group: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ev_ebitda": self.ev_ebitda,
            "ev_ebit": self.ev_ebit,
            "pe_trailing": self.pe_trailing,
            "pe_forward": self.pe_forward,
            "ps_ttm": self.ps_ttm,
            "ev_revenue": self.ev_revenue,
            "p_fcf": self.p_fcf,
            "peg_ratio": self.peg_ratio,
            "vs_sector_avg": self.vs_sector_avg,
            "peer_group": self.peer_group,
        }


@dataclass
class ValuationResult:
    """Combined DCF + Comps + implied price range."""

    dcf: DCFResult = field(default_factory=DCFResult)
    comps: CompsResult = field(default_factory=CompsResult)
    implied_price_range: Dict[str, Optional[float]] = field(
        default_factory=lambda: {"low": None, "mid": None, "high": None}
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dcf": self.dcf.to_dict(),
            "comps": self.comps.to_dict(),
            "implied_price_range": self.implied_price_range,
        }


# ---------------------------------------------------------------------------
# Technicals
# ---------------------------------------------------------------------------

@dataclass
class TechnicalSnapshot:
    """Output of the technical analysis module.

    Per data boundary enforcement, Financial Modelling Agent only computes
    beta for WACC. All other technical indicators are excluded (reserved for
    Fundamental Math Agent).

    Attribute naming:
      Canonical names (used internally and by the pipeline):
        beta, macd_histogram, bollinger_upper, bollinger_lower,
        stochastic_k, stochastic_d, high_52w, low_52w

      Alias properties (expected by tests and external consumers):
        macd         → macd_histogram
        bb_upper     → bollinger_upper
        bb_lower     → bollinger_lower
        stoch_k      → stochastic_k
        stoch_d      → stochastic_d
        w52_high     → high_52w
        w52_low      → low_52w
    """

    trend: Optional[str] = None               # "bullish" | "bearish" | "neutral"
    beta: Optional[float] = None              # Beta for WACC calculation (2-year weekly rolling)
    rsi_14: Optional[float] = None
    macd_signal: Optional[str] = None         # "buy" | "sell" | "neutral" (direction string)
    macd_signal_line: Optional[float] = None  # MACD signal line value (9-period EMA of MACD)
    macd_histogram: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    sma_50_above_200: Optional[bool] = None
    golden_cross: Optional[bool] = None
    death_cross: Optional[bool] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_position: Optional[str] = None  # "above_upper" | "upper" | "mid" | "lower" | "below_lower"
    atr_14: Optional[float] = None
    hv_30: Optional[float] = None             # annualised 30-day historical volatility
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None
    support: Optional[float] = None
    resistance: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None

    # ── Alias properties ─────────────────────────────────────────────────────
    # These expose alternate attribute names expected by tests and callers.

    @property
    def macd(self) -> Optional[float]:
        return self.macd_histogram

    @macd.setter
    def macd(self, value: Optional[float]) -> None:
        self.macd_histogram = value

    @property
    def bb_upper(self) -> Optional[float]:
        return self.bollinger_upper

    @bb_upper.setter
    def bb_upper(self, value: Optional[float]) -> None:
        self.bollinger_upper = value

    @property
    def bb_lower(self) -> Optional[float]:
        return self.bollinger_lower

    @bb_lower.setter
    def bb_lower(self, value: Optional[float]) -> None:
        self.bollinger_lower = value

    @property
    def stoch_k(self) -> Optional[float]:
        return self.stochastic_k

    @stoch_k.setter
    def stoch_k(self, value: Optional[float]) -> None:
        self.stochastic_k = value

    @property
    def stoch_d(self) -> Optional[float]:
        return self.stochastic_d

    @stoch_d.setter
    def stoch_d(self, value: Optional[float]) -> None:
        self.stochastic_d = value

    @property
    def w52_high(self) -> Optional[float]:
        return self.high_52w

    @w52_high.setter
    def w52_high(self, value: Optional[float]) -> None:
        self.high_52w = value

    @property
    def w52_low(self) -> Optional[float]:
        return self.low_52w

    @w52_low.setter
    def w52_low(self, value: Optional[float]) -> None:
        self.low_52w = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trend": self.trend,
            "beta": self.beta,
            "rsi_14": self.rsi_14,
            "macd": self.macd_histogram,
            "macd_signal": self.macd_signal,
            "macd_signal_line": self.macd_signal_line,
            "macd_histogram": self.macd_histogram,
            "sma_20": self.sma_20,
            "sma_50": self.sma_50,
            "sma_200": self.sma_200,
            "ema_12": self.ema_12,
            "ema_26": self.ema_26,
            "sma_50_above_200": self.sma_50_above_200,
            "golden_cross": self.golden_cross,
            "death_cross": self.death_cross,
            "bollinger_upper": self.bollinger_upper,
            "bollinger_lower": self.bollinger_lower,
            "bollinger_position": self.bollinger_position,
            "atr_14": self.atr_14,
            "hv_30": self.hv_30,
            "stochastic_k": self.stochastic_k,
            "stochastic_d": self.stochastic_d,
            "support": self.support,
            "resistance": self.resistance,
            "52w_high": self.high_52w,
            "52w_low": self.low_52w,
        }


# ---------------------------------------------------------------------------
# Earnings
# ---------------------------------------------------------------------------

@dataclass
class EarningsRecord:
    """Earnings history and EPS surprise tracking."""

    last_eps_actual: Optional[float] = None
    last_eps_estimate: Optional[float] = None
    surprise_pct: Optional[float] = None
    beat_streak: int = 0
    miss_streak: int = 0
    next_earnings_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_eps_actual": self.last_eps_actual,
            "last_eps_estimate": self.last_eps_estimate,
            "surprise_pct": self.surprise_pct,
            "beat_streak": self.beat_streak,
            "miss_streak": self.miss_streak,
            "next_earnings_date": self.next_earnings_date,
        }


# ---------------------------------------------------------------------------
# Dividends
# ---------------------------------------------------------------------------

@dataclass
class DividendRecord:
    """Dividend and payout data."""

    dividend_yield: Optional[float] = None
    annual_dividend: Optional[float] = None
    payout_ratio: Optional[float] = None
    dividend_growth_5y_cagr: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dividend_yield": self.dividend_yield,
            "annual_dividend": self.annual_dividend,
            "payout_ratio": self.payout_ratio,
            "dividend_growth_5y_cagr": self.dividend_growth_5y_cagr,
        }


# ---------------------------------------------------------------------------
# Factor scores
# ---------------------------------------------------------------------------

@dataclass
class FactorScores:
    """Composite financial health factor scores."""

    piotroski_f_score: Optional[int] = None
    beneish_m_score: Optional[float] = None
    altman_z_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "piotroski_f_score": self.piotroski_f_score,
            "beneish_m_score": self.beneish_m_score,
            "altman_z_score": self.altman_z_score,
        }


# ---------------------------------------------------------------------------
# Data bundle passed between pipeline nodes
# ---------------------------------------------------------------------------

@dataclass
class FMDataBundle:
    """Raw data fetched from PostgreSQL for a single ticker."""

    ticker: str

    # From raw_fundamentals (current period — most recent row, may be quarterly)
    income: Dict[str, Any] = field(default_factory=dict)
    balance: Dict[str, Any] = field(default_factory=dict)
    cashflow: Dict[str, Any] = field(default_factory=dict)

    # Most recent annual (yearly) period — used for 3-statement model and annual analysis
    income_annual: Dict[str, Any] = field(default_factory=dict)
    balance_annual: Dict[str, Any] = field(default_factory=dict)
    cashflow_annual: Dict[str, Any] = field(default_factory=dict)

    # Prior-year annual period (for YoY Piotroski signals)
    income_prior: Dict[str, Any] = field(default_factory=dict)
    balance_prior: Dict[str, Any] = field(default_factory=dict)
    cashflow_prior: Dict[str, Any] = field(default_factory=dict)
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    key_metrics_ttm: Dict[str, Any] = field(default_factory=dict)
    ratios: Dict[str, Any] = field(default_factory=dict)
    ratios_ttm: Dict[str, Any] = field(default_factory=dict)
    enterprise: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, Any] = field(default_factory=dict)
    earnings_history: List[Dict[str, Any]] = field(default_factory=list)
    analyst_estimates: List[Dict[str, Any]] = field(default_factory=list)
    dividend_history: List[Dict[str, Any]] = field(default_factory=list)
    market_risk_premium: Dict[str, Any] = field(default_factory=dict)

    # From raw_timeseries
    price_history: List[Dict[str, Any]] = field(default_factory=list)   # ticker EOD prices
    treasury_rates: List[Dict[str, Any]] = field(default_factory=list)  # 10Y yield rows

    # From market_eod_us
    benchmark_history: List[Dict[str, Any]] = field(default_factory=list)   # S&P 500 EOD
    peer_histories: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)  # ticker → prices

    # Peer fundamentals for Comps
    peer_fundamentals: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Revenue segmentation (FMP)
    revenue_segments: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # ── New EODHD-sourced fields ──────────────────────────────────────────────

    # Row 10: Dividend History / Stock Splits — splits part
    splits_history: List[Dict[str, Any]] = field(default_factory=list)

    # Row 11: Treasury Rates / Macro Indicators — from global_macro_indicators table
    macro_indicators: List[Dict[str, Any]] = field(default_factory=list)

    # Row 12: Economic Events Data API — from economic_events table
    economic_events: List[Dict[str, Any]] = field(default_factory=list)

    # Row 13: Bonds Data (Yields, Pricing) — from corporate_bond_yields table
    bond_yields: List[Dict[str, Any]] = field(default_factory=list)

    # Row 14: Forex Historical Rates (EOD) — from forex_rates table
    forex_rates: List[Dict[str, Any]] = field(default_factory=list)

    # Row 18: Financial Statements (IS/BS/CF) — from financial_statements table
    # Keyed by statement_type: "Income_Statement", "Balance_Sheet", "Cash_Flow"
    financial_statements: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Row 19: Valuation Metrics — from valuation_metrics table
    valuation_metrics: Dict[str, Any] = field(default_factory=dict)

    # Row 22: Outstanding Shares History — from outstanding_shares table
    outstanding_shares: List[Dict[str, Any]] = field(default_factory=list)

    # Row 11 (dedicated treasury_rates table) — supersedes raw_timeseries treasury_rates
    treasury_rates_dedicated: List[Dict[str, Any]] = field(default_factory=list)

    # Row 23: Earnings Surprises — from earnings_surprises table (521 rows across 5 tickers)
    # Each dict has: ticker, period_date, eps_actual, eps_estimate, eps_surprise_pct,
    #                revenue_actual, revenue_estimate, revenue_surprise_pct, before_after_market
    earnings_surprises: List[Dict[str, Any]] = field(default_factory=list)

    # Row 10 (dedicated dividends table) — from dividends_history table (394 rows)
    # Each dict has: ticker, amount, ex_date, pay_date, record_date
    dividends_dedicated: List[Dict[str, Any]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any([
            self.income, self.balance, self.cashflow,
            self.key_metrics, self.enterprise, self.scores,
        ])


__all__ = [
    "DCFResult",
    "CompsResult",
    "ValuationResult",
    "TechnicalSnapshot",
    "EarningsRecord",
    "DividendRecord",
    "FactorScores",
    "FMDataBundle",
]
