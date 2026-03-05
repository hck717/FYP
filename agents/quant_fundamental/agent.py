"""Quantitative Fundamental Agent — LangGraph plan-and-execute pipeline.

Architecture (8-node pipeline):

    ticker
        │
        ▼
    fetch_financials          ←  PostgreSQL: ratios_ttm, key_metrics_ttm, scores, price history
        │
        ▼
    chain_of_table_reasoning  ←  SELECT → FILTER → CALCULATE → RANK → IDENTIFY (5-step)
        │
        ▼
    data_quality_check        ←  PostgreSQL field-presence + range validation (single-path)
        │
        ▼
    calculate_value_factors   ←  P/E, EV/EBITDA, P/FCF, EV/Revenue
        │
        ▼
    calculate_quality_factors ←  ROE, ROIC, Piotroski F-Score, Beneish M-Score
        │
        ▼
    calculate_momentum_risk   ←  Beta-60d, Sharpe-12m, Return-12m
        │
        ▼
    flag_anomalies            ←  Z-score vs. 3-year rolling baseline
        │
        ▼
    format_json_output        ←  Assemble output; LLM writes quantitative_summary only
        │
       END → return to Supervisor

Usage (CLI):
    python -m agents.quant_fundamental.agent --ticker AAPL
    python -m agents.quant_fundamental.agent --ticker AAPL --log-level DEBUG
    python -m agents.quant_fundamental.agent --prompt "Compare MSFT vs AAPL fundamentals"
    python -m agents.quant_fundamental.agent --prompt "Analyze 0066.HK fundamentals"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict, cast

from langgraph.graph import END, StateGraph

from .config import QuantFundamentalConfig, load_config
from .factors import (
    compute_value_factors,
    compute_quality_factors,
    compute_key_metrics_quality,
    compute_momentum_risk,
)
from .llm import QuantLLMClient
from .schema import (
    AnomalyFlag,
    DataQualityCheck,
    FinancialsBundle,
    KeyMetrics,
    MomentumRiskFactors,
    QualityFactors,
    QualityStatus,
    QuarterlyPeriod,
    ValueFactors,
)
from .tools import QuantFundamentalToolkit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """Shared mutable state threaded through every node."""

    # Input
    ticker: str

    # Availability profile (optional, injected by orchestration planner)
    availability_profile: Optional[Dict[str, bool]]

    # Loaded data
    bundle: Optional[FinancialsBundle]
    fetch_error: Optional[str]

    # Chain-of-table intermediate
    cot_summary: Optional[Dict[str, Any]]

    # Data quality
    data_quality: Optional[DataQualityCheck]

    # Computed factors
    value_factors: Optional[ValueFactors]
    quality_factors: Optional[QualityFactors]
    momentum_risk: Optional[MomentumRiskFactors]
    key_metrics: Optional[KeyMetrics]

    # Prior-year data for YoY signals
    inc_prev: Optional[Dict[str, Any]]
    bal_prev: Optional[Dict[str, Any]]
    cf_prev: Optional[Dict[str, Any]]

    # Last 4 quarters of income statement data for QoQ/YoY trend analysis
    quarterly_trends: List[QuarterlyPeriod]

    # Anomalies
    anomaly_flags: List[Dict[str, Any]]

    # Final output
    output: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Node 1: fetch_financials
# ---------------------------------------------------------------------------

def _node_fetch_financials(
    state: AgentState,
    toolkit: QuantFundamentalToolkit,
) -> AgentState:
    """Fetch all financial data from PostgreSQL into a FinancialsBundle.

    Short-circuits immediately with an empty bundle when the availability
    profile indicates that no PG fundamentals exist for this ticker.
    """
    ticker = state["ticker"]

    # Availability-aware early exit: skip the DB round-trip when we know there's nothing
    profile: Optional[Dict[str, bool]] = state.get("availability_profile")
    if profile is not None and not profile.get("has_any_quantitative", True):
        logger.info(
            "[QF] Skipping fetch_financials for ticker=%s — no PG fundamentals or timeseries data.",
            ticker,
        )
        return {
            **state,
            "bundle": FinancialsBundle(ticker=ticker),
            "inc_prev": {},
            "bal_prev": {},
            "cf_prev": {},
            "quarterly_trends": [],
            "fetch_error": "No data: availability profile reports no PG fundamentals for this ticker.",
        }

    try:
        bundle = toolkit.fetch_financials(ticker)
        # Also fetch prior-year period for YoY factor signals
        inc_list = toolkit.pg.fetch_latest_fundamental(ticker, "income_statement", limit=5)
        bal_list = toolkit.pg.fetch_latest_fundamental(ticker, "balance_sheet", limit=5)
        cf_list = toolkit.pg.fetch_latest_fundamental(ticker, "cash_flow", limit=5)

        def _second_period(rows):
            """Return the second-most-recent period payload (prior year), or {}."""
            all_periods: List[Dict] = []
            for r in rows:
                p = r.get("payload", [])
                if isinstance(p, list):
                    all_periods.extend(p)
                elif isinstance(p, dict):
                    all_periods.append(p)
            return all_periods[1] if len(all_periods) > 1 else {}

        inc_prev = _second_period(inc_list)
        bal_prev = _second_period(bal_list)
        cf_prev = _second_period(cf_list)

        # Build quarterly_trends: last 4 periods from income_statement rows.
        # Each DB row's payload may be a list (multi-period) or a single dict.
        # We collect all periods newest-first and take the first 4.
        quarterly_trends: List[QuarterlyPeriod] = []
        all_inc_periods: List[Dict] = []
        for r in inc_list:
            p = r.get("payload", [])
            if isinstance(p, list):
                all_inc_periods.extend(p)
            elif isinstance(p, dict) and p:
                all_inc_periods.append(p)

        def _fv(d: Dict, *keys) -> Optional[float]:
            for k in keys:
                v = d.get(k)
                if v is not None:
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        continue
            return None

        for period_data in all_inc_periods[:4]:
            if not isinstance(period_data, dict):
                continue
            rev = _fv(period_data, "revenue")
            gp = _fv(period_data, "grossProfit")
            oi = _fv(period_data, "operatingIncome", "ebit")
            ni = _fv(period_data, "netIncome")
            eps = _fv(period_data, "epsdiluted", "eps", "epsActual")
            period_label = (
                period_data.get("period")
                or period_data.get("calendarYear")
                or period_data.get("date", "")
            )
            gm = round(gp / rev, 4) if rev and rev > 0 and gp is not None else None
            em = round(oi / rev, 4) if rev and rev > 0 and oi is not None else None
            quarterly_trends.append(QuarterlyPeriod(
                period=str(period_label),
                revenue=rev,
                gross_profit=gp,
                operating_income=oi,
                net_income=ni,
                eps_diluted=eps,
                gross_margin=gm,
                ebit_margin=em,
            ))

        # Attach quarterly_trends to the bundle so downstream nodes can use it
        bundle.quarterly_trends = quarterly_trends

        logger.debug(
            "fetch_financials: bundle loaded for %s (income keys=%d, prices=%d, quarterly_periods=%d)",
            ticker, len(bundle.income), len(bundle.price_history), len(quarterly_trends),
        )
        return {
            **state,
            "bundle": bundle,
            "inc_prev": inc_prev,
            "bal_prev": bal_prev,
            "cf_prev": cf_prev,
            "quarterly_trends": quarterly_trends,
            "fetch_error": None,
        }
    except Exception as exc:
        logger.error("fetch_financials failed for %s: %s", ticker, exc, exc_info=True)
        return {**state, "bundle": FinancialsBundle(ticker=ticker), "quarterly_trends": [], "fetch_error": str(exc)}


# ---------------------------------------------------------------------------
# Node 2: chain_of_table_reasoning
# ---------------------------------------------------------------------------

def _node_chain_of_table_reasoning(state: AgentState) -> AgentState:
    """5-step Chain-of-Table pipeline: SELECT → FILTER → CALCULATE → RANK → IDENTIFY.

    This node builds a structured intermediate summary dict that subsequent
    nodes can use for quality checking and factor computation. It does NOT call
    the LLM — all steps are pure Python data transformations.
    """
    bundle: Optional[FinancialsBundle] = state.get("bundle")
    if bundle is None or bundle.is_empty():
        return {**state, "cot_summary": {"status": "NO_DATA"}}

    ticker = bundle.ticker

    # Step 1: SELECT — identify which data tables are populated
    available_tables = []
    if bundle.income:
        available_tables.append("income_statement")
    if bundle.balance:
        available_tables.append("balance_sheet")
    if bundle.cashflow:
        available_tables.append("cash_flow")
    if bundle.ratios_ttm:
        available_tables.append("ratios_ttm")
    if bundle.key_metrics_ttm:
        available_tables.append("key_metrics_ttm")
    if bundle.enterprise:
        available_tables.append("enterprise_values")
    if bundle.scores:
        available_tables.append("financial_scores")
    if bundle.price_history:
        available_tables.append("price_history")
    if bundle.benchmark_history:
        available_tables.append("benchmark_history")

    # Step 2: FILTER — extract key numeric fields
    def _g(d: Dict, *keys):
        for k in keys:
            v = d.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return None

    inc = bundle.income
    bal = bundle.balance
    cf = bundle.cashflow

    key_fields = {
        "revenue": _g(inc, "revenue"),
        "gross_profit": _g(inc, "grossProfit"),
        "operating_income": _g(inc, "operatingIncome", "ebit"),
        "net_income": _g(inc, "netIncome"),
        "total_assets": _g(bal, "totalAssets"),
        "total_equity": _g(bal, "totalStockholdersEquity", "totalShareholdersEquity"),
        "total_debt": _g(bal, "longTermDebt", "longTermDebtAndCapitalLeaseObligation"),
        "ocf": _g(cf, "operatingCashFlow"),
        "capex": _g(cf, "capitalExpenditure"),
        "price_points": len(bundle.price_history),
    }

    # Step 3: CALCULATE — derived ratios for ranking
    revenue = key_fields.get("revenue")
    gross_profit = key_fields.get("gross_profit")
    operating_income = key_fields.get("operating_income")
    net_income = key_fields.get("net_income")
    total_assets = key_fields.get("total_assets")
    total_equity = key_fields.get("total_equity")
    ocf = key_fields.get("ocf")
    capex = key_fields.get("capex")

    derived = {}
    if revenue and revenue > 0:
        if gross_profit is not None:
            derived["gross_margin"] = round(gross_profit / revenue, 4)
        if operating_income is not None:
            derived["ebit_margin"] = round(operating_income / revenue, 4)
    if total_assets and total_assets > 0 and net_income is not None:
        derived["roa"] = round(net_income / total_assets, 4)
    if total_equity and total_equity > 0 and net_income is not None:
        derived["roe"] = round(net_income / total_equity, 4)
    if ocf is not None and capex is not None:
        derived["fcf"] = round(ocf - abs(capex), 0)
        if net_income and net_income != 0:
            derived["fcf_conversion"] = round(derived["fcf"] / net_income, 4)

    # Step 4: RANK — classify quality tier
    quality_tier = "UNKNOWN"
    roe = derived.get("roe")
    ebit_margin = derived.get("ebit_margin")
    if roe is not None and ebit_margin is not None:
        if roe > 0.20 and ebit_margin > 0.15:
            quality_tier = "HIGH"
        elif roe > 0.10 and ebit_margin > 0.05:
            quality_tier = "MEDIUM"
        elif roe is not None:
            quality_tier = "LOW"

    # Step 5: IDENTIFY — flag potential data issues
    data_issues = []
    if not bundle.income:
        data_issues.append("income_statement missing")
    if not bundle.balance:
        data_issues.append("balance_sheet missing")
    if not bundle.price_history:
        data_issues.append("price_history missing — momentum factors unavailable")
    if len(bundle.price_history) < 60:
        data_issues.append(f"limited price history ({len(bundle.price_history)} days) — beta may be unreliable")

    # Step 5b: BUILD QUARTERLY TREND SUMMARY from bundle.quarterly_trends
    # Compute QoQ and YoY (period[0] vs period[4] if available) revenue/margin deltas in Python
    qt_list = bundle.quarterly_trends  # already List[QuarterlyPeriod], newest first
    qt_summary: List[Dict[str, Any]] = [q.to_dict() for q in qt_list]

    qoq_deltas: Dict[str, Any] = {}
    if len(qt_list) >= 2:
        q0, q1 = qt_list[0], qt_list[1]  # most recent vs. prior quarter
        if q0.revenue and q1.revenue and q1.revenue != 0:
            qoq_deltas["revenue_qoq_pct"] = round((q0.revenue - q1.revenue) / abs(q1.revenue) * 100, 2)
        if q0.gross_margin is not None and q1.gross_margin is not None:
            qoq_deltas["gross_margin_qoq_pp"] = round((q0.gross_margin - q1.gross_margin) * 100, 2)
        if q0.ebit_margin is not None and q1.ebit_margin is not None:
            qoq_deltas["ebit_margin_qoq_pp"] = round((q0.ebit_margin - q1.ebit_margin) * 100, 2)
        if q0.net_income is not None and q1.net_income is not None and q1.net_income != 0:
            qoq_deltas["net_income_qoq_pct"] = round((q0.net_income - q1.net_income) / abs(q1.net_income) * 100, 2)
        if q0.eps_diluted is not None and q1.eps_diluted is not None and q1.eps_diluted != 0:
            qoq_deltas["eps_qoq_pct"] = round((q0.eps_diluted - q1.eps_diluted) / abs(q1.eps_diluted) * 100, 2)

    yoy_deltas: Dict[str, Any] = {}
    if len(qt_list) >= 4:
        q_now, q_year_ago = qt_list[0], qt_list[3]  # most recent vs. 4 quarters ago
        if q_now.revenue and q_year_ago.revenue and q_year_ago.revenue != 0:
            yoy_deltas["revenue_yoy_pct"] = round((q_now.revenue - q_year_ago.revenue) / abs(q_year_ago.revenue) * 100, 2)
        if q_now.gross_margin is not None and q_year_ago.gross_margin is not None:
            yoy_deltas["gross_margin_yoy_pp"] = round((q_now.gross_margin - q_year_ago.gross_margin) * 100, 2)
        if q_now.ebit_margin is not None and q_year_ago.ebit_margin is not None:
            yoy_deltas["ebit_margin_yoy_pp"] = round((q_now.ebit_margin - q_year_ago.ebit_margin) * 100, 2)
        if q_now.net_income is not None and q_year_ago.net_income is not None and q_year_ago.net_income != 0:
            yoy_deltas["net_income_yoy_pct"] = round((q_now.net_income - q_year_ago.net_income) / abs(q_year_ago.net_income) * 100, 2)
        if q_now.eps_diluted is not None and q_year_ago.eps_diluted is not None and q_year_ago.eps_diluted != 0:
            yoy_deltas["eps_yoy_pct"] = round((q_now.eps_diluted - q_year_ago.eps_diluted) / abs(q_year_ago.eps_diluted) * 100, 2)

    cot_summary = {
        "status": "OK",
        "ticker": ticker,
        "available_tables": available_tables,
        "key_fields": key_fields,
        "derived_ratios": derived,
        "quality_tier": quality_tier,
        "data_issues": data_issues,
        "quarterly_periods": qt_summary,
        "qoq_deltas": qoq_deltas,
        "yoy_deltas": yoy_deltas,
    }
    logger.debug("chain_of_table: %s quality_tier=%s, tables=%s, quarterly_periods=%d",
                 ticker, quality_tier, available_tables, len(qt_summary))
    return {**state, "cot_summary": cot_summary}


# ---------------------------------------------------------------------------
# Node 3: data_quality_check
# ---------------------------------------------------------------------------

def _node_data_quality_check(
    state: AgentState,
    toolkit: QuantFundamentalToolkit,
) -> AgentState:
    """Validate PostgreSQL data quality: field presence and value range checks.

    This is a single-path check against the data already read from PostgreSQL.
    It verifies that critical TTM fields are present and internally consistent
    (e.g. 0 < gross_margin < 1, pe_trailing > 0). No re-computation is done.
    """
    bundle: Optional[FinancialsBundle] = state.get("bundle")
    cot: Optional[Dict] = state.get("cot_summary", {})

    if bundle is None or cot is None or cot.get("status") == "NO_DATA":
        quality = DataQualityCheck(
            status=QualityStatus.SKIPPED,
            checks_passed=0,
            checks_total=0,
            issues=["Skipped — no financial data available"],
        )
        return {**state, "data_quality": quality}

    quality = toolkit.quality_checker.check(bundle)
    logger.debug(
        "data_quality_check: %s status=%s, passed=%d/%d, issues=%d",
        bundle.ticker,
        quality.status.value,
        quality.checks_passed,
        quality.checks_total,
        len(quality.issues),
    )
    return {**state, "data_quality": quality}


# ---------------------------------------------------------------------------
# Node 4: calculate_value_factors
# ---------------------------------------------------------------------------

def _node_calculate_value_factors(state: AgentState) -> AgentState:
    bundle: Optional[FinancialsBundle] = state.get("bundle")
    if bundle is None:
        return {**state, "value_factors": ValueFactors()}
    value = compute_value_factors(bundle)
    logger.debug("value_factors for %s: %s", bundle.ticker, value.to_dict())
    return {**state, "value_factors": value}


# ---------------------------------------------------------------------------
# Node 5: calculate_quality_factors
# ---------------------------------------------------------------------------

def _node_calculate_quality_factors(
    state: AgentState,
    toolkit: QuantFundamentalToolkit,
) -> AgentState:
    """Compute Piotroski F-Score, Beneish M-Score, Altman Z-Score, ROE, ROIC.

    3A: First attempts to read pre-computed scores from the
    ``mv_daily_factor_scores`` PostgreSQL materialized view (populated nightly
    by a REFRESH job).  Falls back to in-memory computation from the
    FinancialsBundle when the MV is empty or not yet populated.
    """
    bundle: Optional[FinancialsBundle] = state.get("bundle")
    if bundle is None:
        return {**state, "quality_factors": QualityFactors(), "key_metrics": KeyMetrics()}

    ticker = bundle.ticker

    # --- 3A: Try materialized view first ---
    mv_row = toolkit.pg.fetch_factor_scores_from_mv(ticker)
    if mv_row is not None:
        logger.info("[QF] Using mv_daily_factor_scores for ticker=%s (as_of=%s)", ticker, mv_row.get("as_of_date"))
        _piotroski_raw = _float_or_none(mv_row.get("piotroski_score"))
        quality = QualityFactors(
            piotroski_f_score=int(_piotroski_raw) if _piotroski_raw is not None else None,
            beneish_m_score=_float_or_none(mv_row.get("beneish_m_score")),
            altman_z_score=_float_or_none(mv_row.get("altman_z_score")),
            roe=_float_or_none(mv_row.get("roe_ttm")),
            roic=_float_or_none(mv_row.get("roic_ttm")),
        )
        km_dict = {
            "gross_margin":   _float_or_none(mv_row.get("gross_margin_ttm")),
            "ebit_margin":    _float_or_none(mv_row.get("net_margin_ttm")),   # best EODHD proxy for ebit_margin
            "debt_to_equity": _float_or_none(mv_row.get("debt_to_equity_ttm")),
            "current_ratio":  _float_or_none(mv_row.get("current_ratio_ttm")),
        }
        key_metrics = KeyMetrics(**{k: v for k, v in km_dict.items() if v is not None})
        logger.debug("quality_factors from MV for %s: %s", ticker, quality.to_dict())
        return {**state, "quality_factors": quality, "key_metrics": key_metrics}

    # --- Fallback: compute in-memory from FinancialsBundle ---
    logger.info("[QF] MV empty for ticker=%s — computing quality factors in-memory", ticker)
    inc_prev = state.get("inc_prev") or {}
    bal_prev = state.get("bal_prev") or {}
    cf_prev = state.get("cf_prev") or {}

    quality = compute_quality_factors(
        bundle,
        inc_prev=inc_prev,
        bal_prev=bal_prev,
        cf_prev=cf_prev,
    )
    km_dict = compute_key_metrics_quality(bundle)
    key_metrics = KeyMetrics(**km_dict)

    logger.debug("quality_factors (in-memory) for %s: %s", ticker, quality.to_dict())
    return {**state, "quality_factors": quality, "key_metrics": key_metrics}


def _float_or_none(val: Any) -> Optional[float]:
    """Safe cast to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Node 6: calculate_momentum_risk
# ---------------------------------------------------------------------------

def _node_calculate_momentum_risk(
    state: AgentState,
    config: QuantFundamentalConfig,
) -> AgentState:
    bundle: Optional[FinancialsBundle] = state.get("bundle")
    if bundle is None:
        return {**state, "momentum_risk": MomentumRiskFactors()}

    momentum = compute_momentum_risk(
        bundle,
        beta_lookback=config.beta_lookback_days,
        sharpe_lookback=config.sharpe_lookback_days,
    )
    logger.debug("momentum_risk for %s: %s", bundle.ticker, momentum.to_dict())
    return {**state, "momentum_risk": momentum}


# ---------------------------------------------------------------------------
# Node 7: flag_anomalies
# ---------------------------------------------------------------------------

def _node_flag_anomalies(
    state: AgentState,
    toolkit: QuantFundamentalToolkit,
) -> AgentState:
    """Z-score anomaly detection over a rolling multi-period window."""
    bundle: Optional[FinancialsBundle] = state.get("bundle")
    quality: Optional[QualityFactors] = state.get("quality_factors")
    value: Optional[ValueFactors] = state.get("value_factors")
    km: Optional[KeyMetrics] = state.get("key_metrics")

    anomaly_flags: List[Dict[str, Any]] = []

    if bundle is None:
        return {**state, "anomaly_flags": anomaly_flags}

    # Build rolling history from prior-year income statements
    ticker = bundle.ticker
    try:
        inc_rows = toolkit.pg.fetch_latest_fundamental(ticker, "income_statement", limit=12)
        # Collect revenue values across periods for baseline
        rev_history: List[float] = []
        gm_history: List[float] = []
        margin_history: List[float] = []
        for row in inc_rows:
            payload = row.get("payload", [])
            periods = payload if isinstance(payload, list) else [payload]
            for p in periods:
                if not isinstance(p, dict):
                    continue
                rev = p.get("revenue")
                gp = p.get("grossProfit")
                oi = p.get("operatingIncome")
                if rev and float(rev) > 0:
                    rev_history.append(float(rev))
                    if gp is not None:
                        gm_history.append(float(gp) / float(rev))
                    if oi is not None:
                        margin_history.append(float(oi) / float(rev))
    except Exception as exc:
        logger.debug("flag_anomalies: history fetch failed for %s: %s", ticker, exc)
        rev_history = []
        gm_history = []
        margin_history = []

    # Check gross margin anomaly
    if km and km.gross_margin is not None and len(gm_history) >= 4:
        flag = toolkit.anomaly_detector.flag_metric(
            "gross_margin",
            km.gross_margin,
            gm_history,
            interpretation=(
                f"Gross margin of {km.gross_margin:.1%} is "
                f"{'above' if km.gross_margin > (sum(gm_history)/len(gm_history)) else 'below'} "
                f"the {len(gm_history)}-period average."
            ),
        )
        if flag:
            anomaly_flags.append(flag)

    # Check EBIT margin anomaly
    if km and km.ebit_margin is not None and len(margin_history) >= 4:
        flag = toolkit.anomaly_detector.flag_metric(
            "ebit_margin",
            km.ebit_margin,
            margin_history,
            interpretation=(
                f"EBIT margin of {km.ebit_margin:.1%} vs. "
                f"{len(margin_history)}-period history."
            ),
        )
        if flag:
            anomaly_flags.append(flag)

    # Check ROE anomaly using prior periods
    if quality and quality.roe is not None:
        try:
            roe_history: List[float] = []
            rt_rows = toolkit.pg.fetch_latest_fundamental(ticker, "ratios_ttm", limit=4)
            for row in rt_rows:
                p = row.get("payload", {})
                if isinstance(p, list):
                    p = p[0] if p else {}
                v = p.get("returnOnEquityTTM") or p.get("roeTTM")
                if v is not None:
                    try:
                        roe_history.append(float(v))
                    except (TypeError, ValueError):
                        pass
            if len(roe_history) >= 3:
                flag = toolkit.anomaly_detector.flag_metric(
                    "roe",
                    quality.roe,
                    roe_history,
                )
                if flag:
                    anomaly_flags.append(flag)
        except Exception:
            pass

    logger.debug("flag_anomalies: %d flags for %s", len(anomaly_flags), ticker)
    return {**state, "anomaly_flags": anomaly_flags}


# ---------------------------------------------------------------------------
# Node 8: format_json_output
# ---------------------------------------------------------------------------

def _node_format_json_output(
    state: AgentState,
    llm: QuantLLMClient,
) -> AgentState:
    """Assemble the final JSON output and call LLM for the narrative summary."""
    ticker = state.get("ticker", "UNKNOWN")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    bundle: Optional[FinancialsBundle] = state.get("bundle")
    value: ValueFactors = state.get("value_factors") or ValueFactors()
    quality: QualityFactors = state.get("quality_factors") or QualityFactors()
    momentum: MomentumRiskFactors = state.get("momentum_risk") or MomentumRiskFactors()
    km: KeyMetrics = state.get("key_metrics") or KeyMetrics()
    data_quality: DataQualityCheck = state.get("data_quality") or DataQualityCheck(
        status=QualityStatus.SKIPPED, checks_passed=0, checks_total=0, issues=[]
    )
    anomaly_flags: List[Dict] = state.get("anomaly_flags") or []
    fetch_error = state.get("fetch_error")
    quarterly_trends: List[QuarterlyPeriod] = state.get("quarterly_trends") or []
    cot: Dict[str, Any] = state.get("cot_summary") or {}
    qoq_deltas: Dict[str, Any] = cot.get("qoq_deltas") or {}
    yoy_deltas: Dict[str, Any] = cot.get("yoy_deltas") or {}

    # Serialize quarterly trend periods for output / LLM prompt
    qt_serialized = [q.to_dict() for q in quarterly_trends]

    # ---------------------------------------------------------------------------
    # CoT Self-Validation: check internal consistency across factors and flag tensions
    # ---------------------------------------------------------------------------
    cot_validation_notes: List[str] = []

    pe = value.pe_trailing
    f_score = quality.piotroski_f_score
    m_score = quality.beneish_m_score
    roe = quality.roe
    roic = quality.roic
    sharpe = momentum.sharpe_ratio_12m
    ret_12m = momentum.return_12m_pct
    beta = momentum.beta_60d
    gross_margin = km.gross_margin
    ebit_margin = km.ebit_margin
    fcf_conv = km.fcf_conversion
    de = km.debt_to_equity

    # 1. Valuation vs Quality tension
    if pe is not None and f_score is not None:
        if pe > 50 and f_score <= 4:
            cot_validation_notes.append(
                f"TENSION: Elevated P/E ({pe:.1f}x) paired with weak Piotroski F-Score ({f_score}/9) — "
                f"premium valuation is not supported by fundamental quality signals."
            )
        elif pe < 15 and f_score >= 7:
            cot_validation_notes.append(
                f"OPPORTUNITY: Low P/E ({pe:.1f}x) with strong Piotroski F-Score ({f_score}/9) — "
                f"potential value opportunity if the discount is not justified by structural risk."
            )

    # 2. Beneish M-Score vs Piotroski F-Score tension
    if m_score is not None and f_score is not None:
        if m_score > -1.78 and f_score >= 7:
            cot_validation_notes.append(
                f"CAUTION: Elevated Beneish M-Score ({m_score:.2f}, above -1.78 threshold) despite strong "
                f"F-Score ({f_score}/9) — earnings quality risk may be understated; F-Score may be "
                f"inflated by accounting choices flagged by the M-Score."
            )

    # 3. Return vs Sharpe tension
    if ret_12m is not None and sharpe is not None:
        if ret_12m > 30 and sharpe < 0.5:
            cot_validation_notes.append(
                f"RISK-ADJUSTED CONCERN: Strong 12m return ({ret_12m:.1f}%) but low Sharpe ratio "
                f"({sharpe:.2f}) — returns were achieved with high volatility, not disciplined risk management."
            )
        elif ret_12m < -10 and sharpe > 0.5:
            cot_validation_notes.append(
                f"MIXED SIGNAL: Negative 12m return ({ret_12m:.1f}%) with Sharpe ({sharpe:.2f}) above 0.5 — "
                f"the drawdown was orderly relative to volatility; consider whether losses are transient."
            )

    # 4. High beta vs defensive valuation
    if beta is not None and pe is not None:
        if beta > 1.5 and pe < 20:
            cot_validation_notes.append(
                f"NOTE: High beta ({beta:.2f}) with low trailing P/E ({pe:.1f}x) — market may be "
                f"discounting cyclical/execution risk that the multiple alone does not reveal."
            )

    # 5. FCF conversion vs EBIT margin
    if fcf_conv is not None and ebit_margin is not None:
        if fcf_conv < 0.7 and ebit_margin > 0.15:
            cot_validation_notes.append(
                f"QUALITY CONCERN: High EBIT margin ({ebit_margin:.1%}) but low FCF conversion "
                f"({fcf_conv:.2f}x) — reported profitability is not translating to cash. "
                f"Check for elevated capex, working capital build, or non-cash revenue recognition."
            )
        elif fcf_conv > 1.3 and ebit_margin < 0.10:
            cot_validation_notes.append(
                f"POSITIVE QUALITY FLAG: Low EBIT margin ({ebit_margin:.1%}) but high FCF conversion "
                f"({fcf_conv:.2f}x) — cash generation exceeds reported earnings; "
                f"non-cash charges may be depressing GAAP margins."
            )

    # 6. ROIC vs WACC implied tension
    if roic is not None:
        if roic < 0.05:
            cot_validation_notes.append(
                f"VALUE DESTRUCTION RISK: ROIC of {roic:.1%} is below the typical 8–10% WACC benchmark — "
                f"the business is currently earning sub-cost returns on invested capital."
            )

    # 7. QoQ revenue deceleration flag
    if qoq_deltas.get("revenue_qoq_pct") is not None and yoy_deltas.get("revenue_yoy_pct") is not None:
        qoq_rev = qoq_deltas["revenue_qoq_pct"]
        yoy_rev = yoy_deltas["revenue_yoy_pct"]
        if qoq_rev < -5 and yoy_rev > 0:
            cot_validation_notes.append(
                f"DECELERATION SIGNAL: Revenue growth positive YoY ({yoy_rev:+.1f}%) but "
                f"declined QoQ ({qoq_rev:+.1f}%) — sequential momentum is fading even if annual comps remain positive."
            )

    # 8. Debt vs profitability stress
    if de is not None and ebit_margin is not None:
        if de > 2.0 and ebit_margin < 0.10:
            cot_validation_notes.append(
                f"LEVERAGE STRESS: D/E ratio of {de:.1f}x with thin EBIT margin ({ebit_margin:.1%}) — "
                f"limited operational buffer to service debt obligations under adverse conditions."
            )

    logger.debug("CoT validation: %d notes for %s", len(cot_validation_notes), ticker)

    # Build the factor table passed to the LLM for narrative generation
    factor_table = {
        "ticker": ticker,
        "as_of_date": today,
        "value_factors": value.to_dict(),
        "quality_factors": quality.to_dict(),
        "momentum_risk": momentum.to_dict(),
        "key_metrics": km.to_dict(),
        "anomaly_flags": anomaly_flags,
        "data_quality": data_quality.to_dict(),
        "quarterly_trends": qt_serialized,
        "qoq_deltas": qoq_deltas,
        "yoy_deltas": yoy_deltas,
        "cot_validation_notes": cot_validation_notes,
    }

    # LLM writes ONLY the quantitative_summary narrative
    quantitative_summary = llm.generate_summary(factor_table)

    # Data sources provenance
    data_sources: Dict[str, Any] = {
        "fundamentals": "postgresql:raw_fundamentals",
        "price_history": "postgresql:raw_timeseries",
        "benchmark": "postgresql:market_eod_us",
        "llm_scope": "quantitative_summary narrative only",
        "llm_model": llm.config.llm_model,
    }
    if fetch_error:
        data_sources["fetch_error"] = fetch_error

    # Period info
    time_range = "TTM"
    if bundle and bundle.income:
        period = bundle.income.get("period") or bundle.income.get("calendarYear")
        if period:
            time_range = f"TTM (latest: {period})"

    output: Dict[str, Any] = {
        "agent": "quant_fundamental",
        "ticker": ticker,
        "as_of_date": today,
        "time_range": time_range,
        "value_factors": value.to_dict(),
        "quality_factors": quality.to_dict(),
        "momentum_risk": momentum.to_dict(),
        "key_metrics": km.to_dict(),
        "anomaly_flags": anomaly_flags,
        "data_quality": data_quality.to_dict(),
        "quarterly_trends": qt_serialized,
        "qoq_deltas": qoq_deltas,
        "yoy_deltas": yoy_deltas,
        "cot_validation_notes": cot_validation_notes,
        "quantitative_summary": quantitative_summary,
        "data_sources": data_sources,
    }

    return {**state, "output": output}


# ---------------------------------------------------------------------------
# Node 3B: execute_python — code-generating ReAct
# ---------------------------------------------------------------------------

def _node_execute_python(
    state: AgentState,
    toolkit: QuantFundamentalToolkit,
) -> AgentState:
    """Execute an LLM-generated pandas/Python snippet against the FinancialsBundle.

    3B: Code-Generating ReAct — when the user asks for a non-standard metric
    the ``format_json_output`` node can embed a ``custom_metric_code`` key in
    the output dict containing a Python snippet.  This node detects that key,
    executes the snippet in the sandboxed ``execute_python_on_bundle`` helper,
    and appends the result to ``output["custom_metric_result"]``.

    The code snippet MUST assign its final value to the variable ``result``.
    Example snippet (written by the LLM):
        import pandas as pd
        prices = [row['close'] for row in bundle.price_history]
        s = pd.Series(prices)
        result = float(s.pct_change(30).iloc[-1]) if len(s) > 30 else None
    """
    output: Optional[Dict[str, Any]] = state.get("output")
    bundle: Optional[FinancialsBundle] = state.get("bundle")

    if output is None or bundle is None:
        return state

    code = output.get("custom_metric_code")
    if not code or not isinstance(code, str):
        return state

    logger.info("[QF] execute_python: running custom metric code for ticker=%s", bundle.ticker)
    exec_result = toolkit.execute_python(code, bundle)

    updated_output = dict(output)
    if exec_result.get("success"):
        updated_output["custom_metric_result"] = exec_result.get("result")
        updated_output["custom_metric_stdout"] = exec_result.get("stdout", "")
        logger.info("[QF] execute_python result: %s", exec_result.get("result"))
    else:
        updated_output["custom_metric_error"] = exec_result.get("error")
        logger.warning("[QF] execute_python failed: %s", exec_result.get("error"))

    return {**state, "output": updated_output}

def build_graph(
    toolkit: QuantFundamentalToolkit,
    llm: QuantLLMClient,
    config: QuantFundamentalConfig,
) -> Any:
    """Assemble and compile the 8-node LangGraph pipeline."""
    graph = StateGraph(AgentState)

    # Register nodes (close over toolkit/llm/config)
    graph.add_node(
        "fetch_financials",
        lambda state: _node_fetch_financials(cast(AgentState, state), toolkit),
    )
    graph.add_node(
        "chain_of_table_reasoning",
        lambda state: _node_chain_of_table_reasoning(cast(AgentState, state)),
    )
    graph.add_node(
        "data_quality_check",
        lambda state: _node_data_quality_check(cast(AgentState, state), toolkit),
    )
    graph.add_node(
        "calculate_value_factors",
        lambda state: _node_calculate_value_factors(cast(AgentState, state)),
    )
    graph.add_node(
        "calculate_quality_factors",
        lambda state: _node_calculate_quality_factors(cast(AgentState, state), toolkit),
    )
    graph.add_node(
        "calculate_momentum_risk",
        lambda state: _node_calculate_momentum_risk(cast(AgentState, state), config),
    )
    graph.add_node(
        "flag_anomalies",
        lambda state: _node_flag_anomalies(cast(AgentState, state), toolkit),
    )
    graph.add_node(
        "format_json_output",
        lambda state: _node_format_json_output(cast(AgentState, state), llm),
    )
    graph.add_node(
        "execute_python",
        lambda state: _node_execute_python(cast(AgentState, state), toolkit),
    )

    # Entry point
    graph.set_entry_point("fetch_financials")

    # Linear pipeline — no branching through most nodes
    graph.add_edge("fetch_financials", "chain_of_table_reasoning")
    graph.add_edge("chain_of_table_reasoning", "data_quality_check")
    graph.add_edge("data_quality_check", "calculate_value_factors")
    graph.add_edge("calculate_value_factors", "calculate_quality_factors")
    graph.add_edge("calculate_quality_factors", "calculate_momentum_risk")
    graph.add_edge("calculate_momentum_risk", "flag_anomalies")
    graph.add_edge("flag_anomalies", "format_json_output")
    # 3B: Conditional edge — run execute_python only when custom_metric_code present
    graph.add_conditional_edges(
        "format_json_output",
        lambda state: (
            "execute_python"
            if (state.get("output") or {}).get("custom_metric_code")
            else END
        ),
        {"execute_python": "execute_python", END: END},
    )
    graph.add_edge("execute_python", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Prompt parsing — planner-agent compatibility
# ---------------------------------------------------------------------------

# Common US equity tickers (extend as needed)
_KNOWN_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "TSLA", "META", "BRK.B",
    "UNH", "JNJ", "XOM", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    "COST", "PEP", "KO", "AVGO", "LLY", "WMT", "BAC", "MCD", "NFLX", "ADBE",
    "CRM", "TMO", "ACN", "CSCO", "ABT", "NKE", "ORCL", "DHR", "QCOM", "TXN",
    "LIN", "PM", "NEE", "RTX", "INTU", "AMGN", "BMY", "SPGI", "MS", "GS",
    "AMD", "INTC", "SBUX", "NOW", "PLD", "ISRG", "AXP", "CAT", "DE", "MDT",
    "IBM", "PYPL", "ZM", "SHOP", "SQ", "UBER", "LYFT", "SNAP", "TWTR", "PINS",
    "COIN", "HOOD", "RBLX", "U", "RIVN", "LCID", "F", "GM", "BA", "DIS",
}

# Matches exchange-suffixed tickers like 0066.HK, VOD.L, 7203.T, BHP.AX
# as well as plain US tickers like AAPL, BRK.B, GOOGL
_EXCHANGE_SUFFIX_PATTERN = re.compile(
    r"\b(\d{1,6}\.[A-Z]{1,4})\b"          # numeric root + exchange suffix e.g. 0066.HK
    r"|"
    r"\b([A-Z]{1,5}\.[A-Z]{1,4})\b"        # alpha root + exchange suffix e.g. VOD.L, BHP.AX
)

# Plain US ticker pattern (no dot suffix), used for heuristic sweep
_US_TICKER_PATTERN = re.compile(r"\b([A-Z]{1,5})\b")

# English stoplist — words that match the ticker pattern but are not symbols
_STOP: set = {
    "A", "I", "THE", "FOR", "ON", "IN", "OF", "AT", "BY", "TO", "DO",
    "RUN", "GET", "USE", "AND", "OR", "NOT", "ALL", "ANY", "CAN", "IS",
    "ARE", "WAS", "BE", "AN", "AS", "IF", "IT", "US", "NO", "YES", "ME",
    "MY", "UP", "OUT", "NEW", "FY", "TTM", "YTD", "EPS", "ROE", "DCF",
    "PE", "PB", "EV", "EBIT", "ROA", "ROIC", "FCF", "IPO", "CEO", "CFO",
    "LLM", "API", "SQL", "DB", "URL", "ID", "OK", "NA", "NaN",
    # Common English words that look like tickers
    "HERE", "THERE", "WHAT", "WHEN", "WHERE", "WHO", "WHY", "HOW",
    "THIS", "THAT", "WITH", "FROM", "HAVE", "HAS", "HAD", "WILL",
    "WOULD", "COULD", "SHOULD", "PLEASE", "JUST", "ALSO", "ONLY",
    "SOME", "MORE", "MOST", "OVER", "THAN", "INTO", "ABOUT", "AFTER",
    "BEFORE", "THEN", "WHICH", "THEIR", "THEM", "THEY", "YOUR",
    "REPORT", "STOCK", "ANALYSIS", "DATA", "FULL", "LAST", "NEXT",
    "COMPARE", "BETWEEN", "VERSUS", "VS", "AND",
    # Exchange suffix components that are not tickers themselves
    "HK", "US", "L", "T", "AX", "PA", "DE", "TO", "SS",
}


def extract_tickers_from_prompt(prompt: str) -> List[str]:
    """Extract ALL ticker symbols from a natural-language planner prompt.

    Supports:
      - Multiple tickers: "Compare MSFT vs AAPL fundamentals" → ["MSFT", "AAPL"]
      - Exchange-suffixed tickers: "Analyze 0066.HK" → ["0066.HK"]
      - Plain US tickers: "Analyze AAPL fundamentals" → ["AAPL"]
      - Explicit keyword: "ticker: MSFT" → ["MSFT"]
      - Parenthesised: "Apple (AAPL)" → ["AAPL"]

    Returns a deduplicated list of tickers in order of first appearance.
    Returns an empty list if no tickers are found.
    """
    if not prompt:
        return []

    text = prompt.strip()
    # Collect (position, ticker) pairs; deduplicate by ticker symbol
    candidates: List[tuple] = []  # (position, ticker_str)
    seen: set = set()

    def _add(pos: int, t: str) -> None:
        t = t.upper()
        if t not in seen:
            seen.add(t)
            candidates.append((pos, t))

    # 1. Exchange-suffixed tickers (highest precision) — position-aware
    for m in _EXCHANGE_SUFFIX_PATTERN.finditer(text):
        candidate = (m.group(1) or m.group(2)).upper()
        root = candidate.split(".")[0]
        if root.upper() not in _STOP:
            _add(m.start(), candidate)

    # 2. Explicit "ticker XXXX" / "ticker: XXXX" keyword (all-uppercase only)
    for m in re.finditer(r"\bticker[:\s]+([A-Za-z0-9.]{1,10})\b", text, re.IGNORECASE):
        candidate = m.group(1)
        if candidate == candidate.upper():
            _add(m.start(1), candidate)

    # 3. Parenthesised tickers e.g. "(AAPL)" or "[TSLA]"
    for m in re.finditer(r"[\(\[]([A-Z0-9]{1,6}(?:\.[A-Z]{1,4})?)[\)\]]", text):
        _add(m.start(1), m.group(1))

    # 4. Known-ticker set scan — record position of first occurrence in text
    upper = text.upper()
    for ticker in _KNOWN_TICKERS:
        m = re.search(r"\b" + re.escape(ticker) + r"\b", upper)
        if m:
            _add(m.start(), ticker)

    # 5. Heuristic: uppercase words 2-5 chars that look like US tickers.
    #    Only applied as last-resort fallback when no tickers found above.
    if not candidates:
        for m in _US_TICKER_PATTERN.finditer(text):
            cand = m.group(1)
            if cand.upper() not in _STOP and len(cand) >= 2:
                _add(m.start(), cand)

    # Return tickers sorted by their position of first appearance in the text
    candidates.sort(key=lambda x: x[0])
    return [t for _, t in candidates]


def extract_ticker_from_prompt(prompt: str) -> Optional[str]:
    """Extract the first ticker symbol from a natural-language prompt.

    Backward-compatible single-ticker wrapper around ``extract_tickers_from_prompt``.
    For multi-ticker prompts, returns only the first ticker found.

    Returns the ticker in uppercase, or None if nothing is found.
    """
    tickers = extract_tickers_from_prompt(prompt)
    return tickers[0] if tickers else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _run_single(
    ticker: str,
    config: QuantFundamentalConfig,
    toolkit: QuantFundamentalToolkit,
    llm: QuantLLMClient,
    availability_profile: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Run the pipeline for one resolved ticker, reusing shared toolkit/llm."""
    compiled = build_graph(toolkit, llm, config)
    initial_state: AgentState = {
        "ticker": ticker,
        "anomaly_flags": [],
        "availability_profile": availability_profile,
    }
    final_state = compiled.invoke(initial_state)
    return final_state.get("output") or {}


def run(
    ticker: str = "",
    config: Optional[QuantFundamentalConfig] = None,
    *,
    prompt: Optional[str] = None,
    availability_profile: Optional[Dict[str, bool]] = None,
) -> Any:
    """Run the Quantitative Fundamental pipeline for one or more tickers.

    Accepts either a ``ticker`` symbol directly, or a natural-language ``prompt``
    (for planner-agent compatibility). If both are provided, ``ticker`` takes
    precedence.

    - Single ticker input (``--ticker AAPL`` or single-ticker prompt):
      Returns a single ``Dict[str, Any]`` conforming to the output schema.
    - Multi-ticker prompt (``"Compare MSFT vs AAPL"``):
      Returns a ``List[Dict[str, Any]]``, one result per ticker.

    Args:
        ticker:               Ticker symbol (e.g. "AAPL"). Takes precedence over prompt.
        config:               Optional config override; defaults to env-var-based config.
        prompt:               Optional natural-language prompt, e.g. "Compare MSFT vs AAPL".
        availability_profile: Optional per-ticker data profile from data_availability module.

    Returns:
        Single dict for one ticker, list of dicts for multiple tickers.

    Raises:
        ValueError: If no ticker can be determined from ticker or prompt.
    """
    cfg = config or load_config()

    # Resolve ticker(s)
    if ticker.strip():
        # Explicit --ticker always means exactly one
        resolved_tickers = [ticker.strip().upper()]
    elif prompt:
        resolved_tickers = extract_tickers_from_prompt(prompt)
    else:
        resolved_tickers = []

    if not resolved_tickers:
        raise ValueError(
            "No ticker provided. Pass --ticker SYMBOL or "
            "--prompt 'Analyze SYMBOL fundamentals'."
        )

    toolkit = QuantFundamentalToolkit(cfg)
    llm = QuantLLMClient(cfg)

    try:
        results = [
            _run_single(t, cfg, toolkit, llm, availability_profile=availability_profile)
            for t in resolved_tickers
        ]
    finally:
        toolkit.close()

    # Single-ticker: return dict directly (backward-compatible)
    # Multi-ticker: return list
    if len(results) == 1:
        return results[0]
    return results


def run_full_analysis(
    ticker: str = "",
    config: Optional[QuantFundamentalConfig] = None,
    *,
    prompt: Optional[str] = None,
    availability_profile: Optional[Dict[str, bool]] = None,
) -> Any:
    """Run a complete quantitative factor analysis for the Synthesizer / Planner.

    This is the canonical entry point for the Supervisor/Synthesizer/Planner to call.
    Accepts either a direct ``ticker`` symbol or a natural-language ``prompt``
    (the ticker is parsed from the prompt automatically).

    For multi-ticker prompts returns a list; single-ticker returns a dict.

    Args:
        ticker:               Ticker symbol (e.g. "AAPL"). Takes precedence over prompt.
        config:               Optional config override.
        prompt:               Natural-language instruction from a planner agent, e.g.
                              "Compare MSFT vs AAPL fundamentals."
        availability_profile: Optional per-ticker data profile from data_availability module.

    Returns:
        Dict for a single ticker, List[Dict] for multiple tickers.

    Example (direct ticker)::

        from agents.quant_fundamental.agent import run_full_analysis
        report = run_full_analysis(ticker="AAPL")

    Example (multi-ticker prompt)::

        reports = run_full_analysis(prompt="Compare MSFT vs AAPL")
        # reports[0]["ticker"] → "MSFT"
        # reports[1]["ticker"] → "AAPL"
    """
    if not ticker and not prompt:
        raise ValueError("Either ticker or prompt is required for run_full_analysis().")
    return run(ticker=ticker, config=config, prompt=prompt, availability_profile=availability_profile)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant_fundamental",
        description=(
            "Run the Quantitative Fundamental pipeline for one or more tickers.\n\n"
            "Two input modes:\n"
            "  Direct:      --ticker AAPL\n"
            "  Prompt:      --prompt 'Analyze AAPL fundamentals'\n"
            "  Multi-tick:  --prompt 'Compare MSFT vs AAPL fundamentals'\n"
            "  Intl ticker: --prompt 'Analyze 0066.HK fundamentals'\n\n"
            "Multi-ticker prompts return a JSON array; single-ticker returns a JSON object.\n"
            "The --prompt mode is compatible with planner-agent invocations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--ticker",
        type=str,
        metavar="SYMBOL",
        help="Ticker symbol (e.g. AAPL). Direct mode.",
    )
    input_group.add_argument(
        "--prompt",
        type=str,
        metavar="TEXT",
        help=(
            "Natural-language prompt (planner-agent mode). "
            "Tickers are extracted automatically. "
            "Supports multiple tickers, e.g. 'Compare MSFT vs AAPL', "
            "and exchange-suffixed tickers like '0066.HK' or 'VOD.L'."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: true).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )
    return parser


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        result = run(
            ticker=args.ticker or "",
            prompt=args.prompt or None,
        )
        indent = 2 if args.pretty else None
        # Multi-ticker prompt returns a list; single returns a dict.
        # Always serialise as JSON (list or dict).
        print(json.dumps(result, indent=indent, default=str))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.error("Agent run failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
