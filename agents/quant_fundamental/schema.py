"""Shared dataclasses / enums for the Quantitative Fundamental agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class QualityStatus(str, Enum):
    PASSED = "PASSED"
    ISSUES_FOUND = "ISSUES_FOUND"
    SKIPPED = "SKIPPED"


class ManipulationRisk(str, Enum):
    HIGH = "HIGH"
    LOW = "LOW"


@dataclass
class DataQualityCheck:
    """Result of PostgreSQL-based data quality validation.

    Checks that required TTM fields were successfully read from PostgreSQL
    and fall within plausible economic ranges. This is a single-path check —
    it does not re-compute any metric; it only validates presence and range.
    """

    status: QualityStatus = QualityStatus.PASSED
    checks_passed: int = 0
    checks_total: int = 0
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
            "issues": self.issues,
        }


@dataclass
class AnomalyFlag:
    """A metric that deviated more than z_threshold standard deviations from its rolling mean."""

    metric: str
    z_score: float
    current_value: float
    rolling_mean: float
    rolling_std: float
    direction: str  # "above" | "below"
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "z_score": round(self.z_score, 2),
            "current_value": self.current_value,
            "3y_mean": round(self.rolling_mean, 4),
            "direction": self.direction,
            "interpretation": self.interpretation,
        }


@dataclass
class ValueFactors:
    pe_trailing: Optional[float] = None
    ev_ebitda: Optional[float] = None
    p_fcf: Optional[float] = None
    ev_revenue: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pe_trailing": self.pe_trailing,
            "ev_ebitda": self.ev_ebitda,
            "p_fcf": self.p_fcf,
            "ev_revenue": self.ev_revenue,
        }


@dataclass
class QualityFactors:
    roe: Optional[float] = None
    roic: Optional[float] = None
    piotroski_f_score: Optional[int] = None
    beneish_m_score: Optional[float] = None
    manipulation_risk: Optional[str] = None  # "HIGH" | "LOW"
    altman_z_score: Optional[float] = None  # Altman Z-Score from FMP financial_scores

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roe": self.roe,
            "roic": self.roic,
            "piotroski_f_score": self.piotroski_f_score,
            "beneish_m_score": self.beneish_m_score,
            "manipulation_risk": self.manipulation_risk,
            "altman_z_score": self.altman_z_score,
        }


@dataclass
class MomentumRiskFactors:
    beta_60d: Optional[float] = None
    sharpe_ratio_12m: Optional[float] = None
    return_12m_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beta_60d": self.beta_60d,
            "sharpe_ratio_12m": self.sharpe_ratio_12m,
            "return_12m_pct": self.return_12m_pct,
        }


@dataclass
class KeyMetrics:
    gross_margin: Optional[float] = None
    ebit_margin: Optional[float] = None
    fcf_conversion: Optional[float] = None
    dso_days: Optional[float] = None
    current_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gross_margin": self.gross_margin,
            "ebit_margin": self.ebit_margin,
            "fcf_conversion": self.fcf_conversion,
            "dso_days": self.dso_days,
            "current_ratio": self.current_ratio,
            "debt_to_equity": self.debt_to_equity,
        }


@dataclass
class QuarterlyPeriod:
    """A single quarter's key income statement line items for QoQ/YoY trend analysis."""

    period: str                        # e.g. "Q3 2024" or "2024-09-30"
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    eps_diluted: Optional[float] = None
    gross_margin: Optional[float] = None   # derived in Python: gross_profit / revenue
    ebit_margin: Optional[float] = None    # derived in Python: operating_income / revenue

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "revenue": self.revenue,
            "gross_profit": self.gross_profit,
            "operating_income": self.operating_income,
            "net_income": self.net_income,
            "eps_diluted": self.eps_diluted,
            "gross_margin": round(self.gross_margin, 4) if self.gross_margin is not None else None,
            "ebit_margin": round(self.ebit_margin, 4) if self.ebit_margin is not None else None,
        }


@dataclass
class FinancialsBundle:
    """Raw financial data bundle extracted from PostgreSQL raw_fundamentals."""

    ticker: str
    income: Dict[str, Any] = field(default_factory=dict)      # income_statement payload
    balance: Dict[str, Any] = field(default_factory=dict)     # balance_sheet payload
    cashflow: Dict[str, Any] = field(default_factory=dict)    # cash_flow payload
    ratios: Dict[str, Any] = field(default_factory=dict)      # financial_ratios payload
    ratios_ttm: Dict[str, Any] = field(default_factory=dict)  # ratios_ttm payload
    key_metrics: Dict[str, Any] = field(default_factory=dict) # key_metrics payload
    key_metrics_ttm: Dict[str, Any] = field(default_factory=dict)
    enterprise: Dict[str, Any] = field(default_factory=dict)  # enterprise_values payload
    scores: Dict[str, Any] = field(default_factory=dict)      # financial_scores payload
    shares_float: Dict[str, Any] = field(default_factory=dict)  # shares_float payload (FMP)
    price_history: List[Dict[str, Any]] = field(default_factory=list)  # raw_timeseries rows
    benchmark_history: List[Dict[str, Any]] = field(default_factory=list)  # market_eod_us rows
    # Last 4 quarters of key income statement line items — populated by fetch_financials node
    quarterly_trends: List[QuarterlyPeriod] = field(default_factory=list)

    # ── EODHD data (Row 1): Historical EOD prices — already in price_history above
    # ── EODHD data (Row 2): Intraday / delayed live quotes (intraday_1m)
    intraday_quotes: List[Dict[str, Any]] = field(default_factory=list)
    # ── EODHD data (Row 7): Beta & Technicals indicators (raw_timeseries)
    technicals: List[Dict[str, Any]] = field(default_factory=list)
    # ── EODHD data (Row 8): Screener / bulk market snapshot (market_screener)
    screener_snapshot: List[Dict[str, Any]] = field(default_factory=list)
    # ── EODHD data (Row 9): Basic Fundamentals — EPS, key metrics (raw_fundamentals)
    basic_fundamentals: Dict[str, Any] = field(default_factory=dict)
    # ── EODHD data (Row 19): Valuation Metrics from dedicated table
    valuation_metrics: Dict[str, Any] = field(default_factory=dict)
    # ── EODHD data (Row 20): Short Interest & Shares Stats (short_interest table)
    short_interest: Dict[str, Any] = field(default_factory=dict)
    # ── EODHD data (Row 21): Earnings History & Surprises (earnings_surprises table)
    earnings_surprises: List[Dict[str, Any]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any([
            self.key_metrics_ttm, self.ratios_ttm,
            self.price_history, self.technicals,
            self.basic_fundamentals,
        ])


__all__ = [
    "QualityStatus",
    "ManipulationRisk",
    "DataQualityCheck",
    "AnomalyFlag",
    "ValueFactors",
    "QualityFactors",
    "MomentumRiskFactors",
    "KeyMetrics",
    "QuarterlyPeriod",
    "FinancialsBundle",
]
