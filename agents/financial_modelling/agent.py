"""Financial Modelling Agent — LangGraph computation pipeline.

Architecture (8-node pipeline):

    ticker
        │
        ▼
    fetch_price_history      ←  PostgreSQL: raw_timeseries (EOD prices, Treasury rates)
        │                       market_eod_us (S&P 500 benchmark)
        ▼
    fetch_fundamentals       ←  PostgreSQL: raw_fundamentals
        │                       (income, balance, cashflow, ratios, scores,
        │                        earnings, dividends, analyst estimates)
        ▼
    fetch_earnings_history   ←  PostgreSQL: earnings_history payload
        │                       → EPS actual vs. estimate, surprise %, beat/miss streak
        ▼
    calculate_technicals     ←  Python: SMA/EMA, RSI, MACD, Bollinger, ATR, HV30, Stochastic
        │
        ▼
    run_dcf_model            ←  Python: WACC (CAPM), FCF projections, Terminal Value
        │                       → Bear / Base / Bull scenarios + sensitivity matrix
        ▼
    run_comparable_analysis  ←  Python + Neo4j peer group + PG peer multiples
        │                       → EV/EBITDA, P/E, P/S, EV/Revenue vs. sector
        ▼
    assess_analyst_estimates ←  PostgreSQL: analyst_estimates + dividends
        │                       → EPS consensus, dividend yield, Piotroski/Beneish/Altman
        ▼
    format_json_output       ←  Structured JSON assembly; LLM writes quantitative_summary only
        │
       END → return to Supervisor

Usage (CLI):
    python -m agents.financial_modelling.agent --ticker AAPL
    python -m agents.financial_modelling.agent --ticker TSLA --log-level DEBUG
    python -m agents.financial_modelling.agent --prompt "Analyze NVDA valuation"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict, cast

import requests
from langgraph.graph import END, StateGraph

from .config import FinancialModellingConfig, load_config
from .models.dcf import DCFEngine, compute_benchmark_hv30, vix_mrp_adjustment
from .models.technicals import TechnicalEngine
from .models.valuation import CompsEngine
from .prompts import build_system_prompt
from .schema import (
    DCFResult,
    DividendRecord,
    EarningsRecord,
    FactorScores,
    FMDataBundle,
    TechnicalSnapshot,
    ValuationResult,
)
from .tools import FMToolkit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """Mutable state threaded through every pipeline node."""

    # Input
    ticker: str

    # Data bundle (assembled by fetch nodes)
    bundle: Optional[FMDataBundle]
    fetch_error: Optional[str]

    # Computed results
    technicals: Optional[TechnicalSnapshot]
    dcf_result: Optional[DCFResult]
    moe_consensus: Optional[Dict[str, Any]]   # 4A: MoE DCF consensus
    macro_environment: Optional[Dict[str, Any]]  # 4B: macro regime snapshot
    comps_result: Optional[Any]   # CompsResult
    earnings: Optional[EarningsRecord]
    dividends: Optional[DividendRecord]
    factor_scores: Optional[FactorScores]
    current_price: Optional[float]

    # Final output
    output: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Node 1: fetch_price_history
# ---------------------------------------------------------------------------

def _node_fetch_price_history(
    state: AgentState,
    toolkit: FMToolkit,
) -> AgentState:
    """Fetch EOD price history, Treasury rates, and benchmark from PostgreSQL."""
    ticker = state.get("ticker", "")
    try:
        # Price history, treasury rates, and benchmark are all fetched by FMDataFetcher.fetch()
        # We start with an empty bundle here and the next node fills fundamentals.
        # For efficiency we do a partial fetch first.
        bundle = FMDataBundle(ticker=ticker)
        pg = toolkit.pg

        # EOD prices — prefer weekly (~54 rows, 1 year) for indicator depth;
        # fall back to daily (~23 rows, 1 month) if weekly unavailable.
        ts_rows = pg.fetch_timeseries(ticker, "historical_prices_weekly", limit=300)
        if not ts_rows:
            ts_rows = pg.fetch_timeseries(ticker, "historical_prices_eod", limit=400)
        if not ts_rows:
            ts_rows = pg.fetch_timeseries(ticker, "historical_price", limit=400)
        bundle.price_history = toolkit.fetcher._merge_ts_date(ts_rows)

        # Treasury rates
        tr_rows = pg.fetch_timeseries("TREASURY", "treasury_rates", limit=5)
        if not tr_rows:
            tr_rows = pg.fetch_timeseries("TNX", "treasury_rates", limit=5)
        bundle.treasury_rates = toolkit.fetcher._merge_ts_date(tr_rows)

        # Benchmark (market_eod_us → S&P 500)
        mkt_rows = pg.fetch_market_eod(limit=400)
        bundle.benchmark_history = toolkit.fetcher._merge_ts_date(mkt_rows)

        logger.debug(
            "fetch_price_history: %s prices=%d treasury=%d benchmark=%d",
            ticker, len(bundle.price_history), len(bundle.treasury_rates), len(bundle.benchmark_history),
        )
        return {**state, "bundle": bundle, "fetch_error": None}

    except Exception as exc:
        logger.error("fetch_price_history failed for %s: %s", ticker, exc, exc_info=True)
        return {**state, "bundle": FMDataBundle(ticker=ticker), "fetch_error": str(exc)}


# ---------------------------------------------------------------------------
# Node 2: fetch_fundamentals
# ---------------------------------------------------------------------------

def _node_fetch_fundamentals(
    state: AgentState,
    toolkit: FMToolkit,
) -> AgentState:
    """Fetch all fundamentals data from PostgreSQL into the bundle."""
    ticker = state.get("ticker", "")
    bundle: FMDataBundle = state.get("bundle") or FMDataBundle(ticker=ticker)

    try:
        full_bundle = toolkit.fetch_data(ticker)
        # Merge: keep price/treasury/benchmark from node 1, take everything else from full_bundle
        if bundle.price_history:
            full_bundle.price_history = bundle.price_history
        if bundle.treasury_rates:
            full_bundle.treasury_rates = bundle.treasury_rates
        if bundle.benchmark_history:
            full_bundle.benchmark_history = bundle.benchmark_history

        logger.debug(
            "fetch_fundamentals: %s income_keys=%d scores_keys=%d peers=%d",
            ticker, len(full_bundle.income), len(full_bundle.scores),
            len(full_bundle.peer_fundamentals),
        )
        return {**state, "bundle": full_bundle}

    except Exception as exc:
        logger.error("fetch_fundamentals failed for %s: %s", ticker, exc, exc_info=True)
        return {**state, "fetch_error": str(exc)}


# ---------------------------------------------------------------------------
# Node 3: fetch_earnings_history
# ---------------------------------------------------------------------------

def _node_fetch_earnings_history(state: AgentState) -> AgentState:
    """Parse earnings history from bundle and compute EPS surprise + streaks."""
    bundle: Optional[FMDataBundle] = state.get("bundle")
    if bundle is None or bundle.is_empty():
        return {**state, "earnings": EarningsRecord()}

    record = _compute_earnings_record(bundle)
    logger.debug("earnings for %s: %s", bundle.ticker, record.to_dict())
    return {**state, "earnings": record}


def _compute_earnings_record(bundle: FMDataBundle) -> EarningsRecord:
    """Parse earnings_history payload into EarningsRecord."""
    record = EarningsRecord()
    history = bundle.earnings_history  # list of dicts, newest first

    if not history:
        return record

    # Sort by date descending (already should be from DB)
    def _date_key(row: Dict[str, Any]) -> str:
        return str(row.get("date") or row.get("reportDate") or row.get("period") or "")

    try:
        history = sorted(history, key=_date_key, reverse=True)
    except Exception:
        pass

    def _sf(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # Most recent entry
    latest = history[0] if history else {}
    record.last_eps_actual = _sf(
        latest.get("actualEarningResult") or latest.get("actual") or latest.get("eps")
    )
    record.last_eps_estimate = _sf(
        latest.get("estimatedEarning") or latest.get("estimate") or latest.get("epsEstimated")
    )
    if record.last_eps_actual is not None and record.last_eps_estimate is not None:
        if record.last_eps_estimate != 0:
            surprise = (record.last_eps_actual - record.last_eps_estimate) / abs(record.last_eps_estimate) * 100
            record.surprise_pct = round(surprise, 2)

    # Beat / miss streak
    beat_streak = 0
    miss_streak = 0
    current_beat_streak = 0
    current_miss_streak = 0
    in_beat = True
    in_miss = True

    for row in history:
        actual = _sf(row.get("actualEarningResult") or row.get("actual") or row.get("eps"))
        estimate = _sf(row.get("estimatedEarning") or row.get("estimate") or row.get("epsEstimated"))
        if actual is None or estimate is None:
            break
        beat = actual >= estimate
        if in_beat and beat:
            current_beat_streak += 1
        else:
            in_beat = False
        if in_miss and not beat:
            current_miss_streak += 1
        else:
            in_miss = False

    record.beat_streak = current_beat_streak
    record.miss_streak = current_miss_streak

    # Next earnings date from analyst estimates
    for est in (bundle.analyst_estimates or []):
        d = est.get("date") or est.get("reportDate") or est.get("nextEarningsDate")
        if d:
            record.next_earnings_date = str(d)
            break

    return record


# ---------------------------------------------------------------------------
# Node 4: calculate_technicals
# ---------------------------------------------------------------------------

def _node_calculate_technicals(state: AgentState) -> AgentState:
    """Run technical indicator computation."""
    bundle: Optional[FMDataBundle] = state.get("bundle")
    if bundle is None:
        return {**state, "technicals": TechnicalSnapshot()}

    engine = TechnicalEngine()
    snap = engine.compute(bundle)
    logger.debug("technicals for %s: trend=%s rsi=%s", bundle.ticker, snap.trend, snap.rsi_14)
    return {**state, "technicals": snap}


# ---------------------------------------------------------------------------
# Node 4B: macro_environment — Dynamic WACC / VIX-adjusted MRP
# ---------------------------------------------------------------------------

def _node_macro_environment(state: AgentState) -> AgentState:
    """4B: Enrich bundle with VIX-adjusted Market Risk Premium.

    Reads 10Y Treasury yield from ``bundle.treasury_rates`` (already fetched by
    ``_node_fetch_price_history``) and estimates VIX regime from the 30-day
    realised volatility of the S&P 500 benchmark history (HV30 proxy).

    Writes ``vix_adjusted_mrp`` into ``bundle.market_risk_premium`` so that
    ``_compute_wacc()`` in ``dcf.py`` picks it up automatically.

    Also populates ``state["macro_environment"]`` for downstream transparency.
    """
    bundle: Optional[FMDataBundle] = state.get("bundle")
    if bundle is None:
        return state

    # ── 10Y Treasury (already in bundle) ─────────────────────────────────
    rf: float = 0.043
    if bundle.treasury_rates:
        row = bundle.treasury_rates[0]
        for key in ("year10", "tenYear", "10Y", "ten_year", "y10"):
            v = row.get(key)
            if v is not None:
                try:
                    fv = float(v)
                    rf = fv / 100 if fv > 1 else fv
                    break
                except (TypeError, ValueError):
                    pass

    # ── HV30 VIX proxy ───────────────────────────────────────────────────
    hv30: Optional[float] = compute_benchmark_hv30(bundle.benchmark_history)

    # ── Base MRP from fundamentals ────────────────────────────────────────
    base_mrp: float = 0.055
    if bundle.market_risk_premium:
        for key in ("marketRiskPremium", "equityRiskPremium", "market_risk_premium", "rp"):
            raw = bundle.market_risk_premium.get(key)
            if raw is not None:
                try:
                    fv = float(raw)
                    base_mrp = fv / 100 if fv > 1 else fv
                    break
                except (TypeError, ValueError):
                    pass

    # ── Apply VIX regime adjustment ───────────────────────────────────────
    delta = vix_mrp_adjustment(hv30)
    adjusted_mrp = round(max(0.02, min(0.15, base_mrp + delta)), 5)

    # Inject into bundle so _compute_wacc() uses the adjusted value
    bundle.market_risk_premium["vix_adjusted_mrp"] = adjusted_mrp

    # Determine VIX regime label
    if hv30 is None:
        regime = "unknown"
    elif hv30 < 0.15:
        regime = "low"
    elif hv30 < 0.25:
        regime = "normal"
    elif hv30 < 0.35:
        regime = "high"
    else:
        regime = "extreme"

    macro_env: Dict[str, Any] = {
        "risk_free_rate_10y": round(rf, 5),
        "benchmark_hv30": hv30,
        "vix_regime": regime,
        "base_mrp": round(base_mrp, 5),
        "mrp_delta": round(delta, 5),
        "vix_adjusted_mrp": adjusted_mrp,
    }
    logger.info(
        "[Macro] rf=%.3f hv30=%s regime=%s mrp=%.4f → %.4f",
        rf, hv30, regime, base_mrp, adjusted_mrp,
    )
    return {**state, "bundle": bundle, "macro_environment": macro_env}


# ---------------------------------------------------------------------------
# Node 5: run_dcf_model
# ---------------------------------------------------------------------------

def _node_run_dcf_model(
    state: AgentState,
    config: FinancialModellingConfig,
) -> AgentState:
    """Run DCF engine: WACC, FCF projections, scenario table, sensitivity matrix."""
    bundle: Optional[FMDataBundle] = state.get("bundle")
    if bundle is None:
        return {**state, "dcf_result": DCFResult()}

    engine = DCFEngine(config)
    result = engine.compute(bundle)
    logger.debug(
        "dcf for %s: base=%s wacc=%s",
        bundle.ticker, result.intrinsic_value_base, result.wacc_used,
    )
    return {**state, "dcf_result": result}


# ---------------------------------------------------------------------------
# Node 4A: moe_consensus — Mixture-of-Experts DCF
# ---------------------------------------------------------------------------

# MoE persona parameter adjustments applied on top of the DCF engine's base result
_MOE_PERSONAS: Dict[str, Dict[str, float]] = {
    "Optimist": {
        "terminal_growth_rate_delta": +0.005,   # +0.5% TGR
        "discount_rate_delta":        -0.01,    # -1% WACC
        "revenue_growth_multiplier":   1.25,    # 25% optimistic revenue uplift
    },
    "Pessimist": {
        "terminal_growth_rate_delta": -0.005,   # -0.5% TGR
        "discount_rate_delta":        +0.015,   # +1.5% WACC
        "revenue_growth_multiplier":   0.70,    # 30% revenue haircut
    },
    "Realist": {
        "terminal_growth_rate_delta":  0.0,
        "discount_rate_delta":         0.0,
        "revenue_growth_multiplier":   1.0,
    },
}


def _moe_persona_valuation(
    persona_name: str,
    persona_params: Dict[str, float],
    base_result: DCFResult,
    config: FinancialModellingConfig,
) -> Dict[str, Any]:
    """Compute an LLM-narrated persona interpretation of the DCF result.

    Adjusts Bear/Base/Bull intrinsic values by the persona's multipliers,
    then calls the Ollama LLM to produce a 2-3 sentence persona narrative.
    Returns {"persona": ..., "bear": ..., "base": ..., "bull": ..., "narrative": ...}
    """
    tgr_delta = persona_params.get("terminal_growth_rate_delta", 0.0)
    dr_delta = persona_params.get("discount_rate_delta", 0.0)
    rev_mult = persona_params.get("revenue_growth_multiplier", 1.0)

    # Discount-rate adjustment affects intrinsic values inversely
    # (higher WACC → lower value, lower WACC → higher value)
    dr_impact = 1.0 - (dr_delta * 8)  # approx: 1% WACC change ≈ 8% value change
    dr_impact = max(0.5, min(2.0, dr_impact))

    # Terminal growth rate adjustment: more positive TGR → higher terminal value
    tgr_impact = 1.0 + (tgr_delta * 15)  # approx: 0.5% TGR change ≈ 7.5% value change
    tgr_impact = max(0.5, min(2.0, tgr_impact))

    combined = dr_impact * tgr_impact * rev_mult

    def _adj(val: Optional[float]) -> Optional[float]:
        if val is None:
            return None
        return round(val * combined, 2)

    bear = _adj(base_result.intrinsic_value_bear)
    base = _adj(base_result.intrinsic_value_base)
    bull = _adj(base_result.intrinsic_value_bull)

    # Persona LLM narrative
    prompt = (
        f"You are the {persona_name} analyst in a 3-member DCF review committee.\n"
        f"The base DCF model outputs: Bear=${base_result.intrinsic_value_bear}, "
        f"Base=${base_result.intrinsic_value_base}, Bull=${base_result.intrinsic_value_bull}, "
        f"WACC={base_result.wacc_used}.\n"
        f"Your {persona_name} adjustments yield: Bear=${bear}, Base=${base}, Bull=${bull}.\n"
        f"In exactly 2 sentences, justify your scenario adjustments as the {persona_name} analyst. "
        f"Be specific about the economic assumptions driving your view."
    )
    narrative = f"[{persona_name} narrative unavailable]"
    try:
        resp = requests.post(
            f"{config.ollama_base_url}/api/generate",
            json={
                "model": config.llm_model,
                "prompt": prompt,
                "temperature": 0.5,
                "num_predict": 200,
                "stream": False,
                "think": False,
            },
            timeout=config.request_timeout or None,
        )
        resp.raise_for_status()
        narrative = _clean_response(resp.json().get("response", "")).strip() or narrative
    except Exception as exc:
        logger.warning("[MoE] %s persona LLM call failed: %s", persona_name, exc)

    return {
        "persona": persona_name,
        "bear": bear,
        "base": base,
        "bull": bull,
        "wacc_adjustment": round(dr_delta, 4),
        "tgr_adjustment": round(tgr_delta, 4),
        "narrative": narrative,
    }


def _node_moe_consensus(
    state: AgentState,
    config: FinancialModellingConfig,
) -> AgentState:
    """4A: Mixture-of-Experts DCF consensus.

    Runs 3 parallel LLM persona threads (Optimist, Pessimist, Realist), each
    adjusting terminal growth rate and discount rate.  Synthesizes a final
    Bear/Base/Bull consensus range using probability-weighted averaging.

    Adds ``moe_consensus`` to state:
        {
          "personas": [{"persona": ..., "bear": ..., "base": ..., "bull": ..., "narrative": ...}, ...],
          "consensus_bear": <float>,
          "consensus_base": <float>,
          "consensus_bull": <float>,
          "consensus_narrative": <str>,
        }
    """
    base_result: Optional[DCFResult] = state.get("dcf_result")
    if base_result is None or base_result.intrinsic_value_base is None:
        logger.info("[MoE] Skipping MoE consensus — no DCF base result")
        return state

    persona_results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="moe") as executor:
        futures = {
            executor.submit(
                _moe_persona_valuation, name, params, base_result, config
            ): name
            for name, params in _MOE_PERSONAS.items()
        }
        for future in as_completed(futures):
            persona_name = futures[future]
            try:
                persona_results.append(future.result())
            except Exception as exc:
                logger.warning("[MoE] Persona '%s' failed: %s", persona_name, exc)

    if not persona_results:
        return state

    # Probability-weighted consensus (equal weight = 1/3 each)
    def _weighted_avg(key: str) -> Optional[float]:
        vals = [p[key] for p in persona_results if p.get(key) is not None]
        if not vals:
            return None
        return round(sum(vals) / len(vals), 2)

    consensus_bear = _weighted_avg("bear")
    consensus_base = _weighted_avg("base")
    consensus_bull = _weighted_avg("bull")

    # Final synthesis narrative via LLM
    narratives_block = "\n".join(
        f"  {p['persona']}: {p['narrative']}" for p in sorted(persona_results, key=lambda x: x["persona"])
    )
    synthesis_prompt = (
        f"Three analysts have produced DCF scenario estimates:\n{narratives_block}\n\n"
        f"Synthesise their views into a 3-sentence Bear/Base/Bull consensus:\n"
        f"  Bear consensus: ${consensus_bear}\n"
        f"  Base consensus: ${consensus_base}\n"
        f"  Bull consensus: ${consensus_bull}\n"
        f"Highlight the key points of disagreement and the most likely outcome."
    )
    consensus_narrative = "MoE consensus synthesis unavailable."
    try:
        resp = requests.post(
            f"{config.ollama_base_url}/api/generate",
            json={
                "model": config.llm_model,
                "prompt": synthesis_prompt,
                "temperature": 0.3,
                "num_predict": 300,
                "stream": False,
                "think": False,
            },
            timeout=config.request_timeout or None,
        )
        resp.raise_for_status()
        consensus_narrative = _clean_response(resp.json().get("response", "")).strip() or consensus_narrative
    except Exception as exc:
        logger.warning("[MoE] Consensus synthesis LLM call failed: %s", exc)

    moe_consensus: Dict[str, Any] = {
        "personas": sorted(persona_results, key=lambda x: x["persona"]),
        "consensus_bear": consensus_bear,
        "consensus_base": consensus_base,
        "consensus_bull": consensus_bull,
        "consensus_narrative": consensus_narrative,
    }
    logger.info(
        "[MoE] Consensus: bear=%s base=%s bull=%s",
        consensus_bear, consensus_base, consensus_bull,
    )
    return {**state, "moe_consensus": moe_consensus}

def _node_run_comparable_analysis(
    state: AgentState,
    config: FinancialModellingConfig,
) -> AgentState:
    """Run Comps engine: peer multiples vs. target ticker."""
    bundle: Optional[FMDataBundle] = state.get("bundle")
    if bundle is None:
        from .schema import CompsResult
        return {**state, "comps_result": CompsResult()}

    engine = CompsEngine(config)
    result = engine.compute(bundle)
    logger.debug(
        "comps for %s: ev_ebitda=%s vs_sector=%s peers=%s",
        bundle.ticker, result.ev_ebitda, result.vs_sector_avg, result.peer_group,
    )
    return {**state, "comps_result": result}


# ---------------------------------------------------------------------------
# Node 7: assess_analyst_estimates (dividends + factor scores)
# ---------------------------------------------------------------------------

def _node_assess_analyst_estimates(state: AgentState) -> AgentState:
    """Compute dividend metrics and factor scores (Piotroski, Beneish, Altman)."""
    bundle: Optional[FMDataBundle] = state.get("bundle")
    if bundle is None:
        return {
            **state,
            "dividends": DividendRecord(),
            "factor_scores": FactorScores(),
            "current_price": None,
        }

    dividends = _compute_dividends(bundle)
    scores = _compute_factor_scores(bundle)
    current_price = _get_current_price(bundle)

    logger.debug(
        "dividends for %s: yield=%s | factors: piotroski=%s beneish=%s altman=%s",
        bundle.ticker, dividends.dividend_yield,
        scores.piotroski_f_score, scores.beneish_m_score, scores.altman_z_score,
    )
    return {
        **state,
        "dividends": dividends,
        "factor_scores": scores,
        "current_price": current_price,
    }


def _sf(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _serialise_price_history(
    rows: List[Dict[str, Any]], limit: int = 260
) -> List[Dict[str, Any]]:
    """Serialise OHLCV rows for Streamlit charts.

    Accepts rows in any key convention (FMP, yfinance, etc.) and returns a
    clean list of ``{date, open, high, low, close, volume}`` dicts sorted
    oldest-first, capped at *limit* rows.
    """
    out: List[Dict[str, Any]] = []
    for row in rows:
        date = str(
            row.get("date") or row.get("Date") or row.get("timestamp") or ""
        )
        if not date:
            continue
        close = _sf(
            row.get("adjClose")
            or row.get("adjusted_close")
            or row.get("close")
            or row.get("Close")
        )
        if close is None:
            continue
        out.append(
            {
                "date": date,
                "open": _sf(row.get("open") or row.get("Open")) or close,
                "high": _sf(row.get("high") or row.get("High")) or close,
                "low": _sf(row.get("low") or row.get("Low")) or close,
                "close": close,
                "volume": _sf(row.get("volume") or row.get("Volume")) or 0.0,
            }
        )
    # DB rows are typically newest-first; reverse for charting (oldest→newest)
    out.reverse()
    return out[-limit:]


def _compute_dividends(bundle: FMDataBundle) -> DividendRecord:
    record = DividendRecord()
    km_ttm = bundle.key_metrics_ttm
    ratios_ttm = bundle.ratios_ttm

    record.dividend_yield = _sf(
        km_ttm.get("dividendYieldTTM") or ratios_ttm.get("dividendYieldTTM")
        or km_ttm.get("dividendYield")
    )
    record.payout_ratio = _sf(
        ratios_ttm.get("dividendPayoutRatioTTM") or ratios_ttm.get("payoutRatioTTM")
        or km_ttm.get("payoutRatioTTM")
    )

    # Annual dividend from history
    history = bundle.dividend_history
    if history:
        annual_total = 0.0
        recent_year = None
        for div in history[:12]:  # last 12 entries (quarterly/monthly)
            amount = _sf(div.get("dividend") or div.get("adjDividend") or div.get("amount"))
            if amount is not None:
                annual_total += amount
        record.annual_dividend = round(annual_total, 4) if annual_total > 0 else None

        # 5-year CAGR
        if len(history) >= 20:
            recent = [
                _sf(d.get("dividend") or d.get("adjDividend") or d.get("amount"))
                for d in history[:4]
            ]
            old = [
                _sf(d.get("dividend") or d.get("adjDividend") or d.get("amount"))
                for d in history[-4:]
            ]
            recent_ann = sum(v for v in recent if v is not None)
            old_ann = sum(v for v in old if v is not None)
            if old_ann > 0 and recent_ann > 0:
                cagr = (recent_ann / old_ann) ** (1 / 5) - 1
                record.dividend_growth_5y_cagr = round(cagr, 4)

    return record


def _int_safe(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _compute_altman_z(bundle: FMDataBundle) -> Optional[float]:
    """Altman Z-Score: Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5."""
    inc = bundle.income
    bal = bundle.balance
    ent = bundle.enterprise

    total_assets = _sf(bal.get("totalAssets"))
    if not total_assets or total_assets <= 0:
        return None

    current_assets = _sf(bal.get("totalCurrentAssets"))
    current_liabilities = _sf(bal.get("totalCurrentLiabilities"))
    retained_earnings = _sf(bal.get("retainedEarnings"))
    ebit = _sf(inc.get("operatingIncome") or inc.get("ebit"))
    market_cap = _sf(ent.get("marketCapitalization") or bundle.key_metrics_ttm.get("marketCapTTM"))
    total_liabilities = _sf(
        bal.get("totalLiabilities")
        or bal.get("totalDebt")  # proxy: debt is the dominant liability for Z-score
    )
    revenue = _sf(inc.get("revenue"))

    wc = (current_assets or 0) - (current_liabilities or 0)
    # If current assets/liabilities not available, use workingCapital directly
    if current_assets is None and current_liabilities is None:
        wc = _sf(bal.get("workingCapital")) or 0

    x1 = wc / total_assets if total_assets else None
    x2 = (retained_earnings or 0) / total_assets if total_assets else None
    x3 = (ebit or 0) / total_assets if total_assets else None
    x4 = (market_cap or 0) / (total_liabilities or 1) if total_liabilities and total_liabilities > 0 else None
    x5 = (revenue or 0) / total_assets if total_assets else None

    if any(v is None for v in [x1, x2, x3, x4, x5]):
        return None

    # All values are confirmed non-None above; assert to satisfy type checker
    assert x1 is not None and x2 is not None and x3 is not None
    assert x4 is not None and x5 is not None
    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
    return round(z, 3)


def _compute_beneish_m(bundle: FMDataBundle) -> Optional[float]:
    """Simplified Beneish M-Score using available single-period data.

    Uses a reduced form based on the accruals component and profit-margin proxy
    since prior-year comparatives are not available in the current DB schema.

    Simplified model:
        M ≈ -6.065 + 0.823*ACCRUALS_INDEX + 4.679*TOTAL_ACCRUALS_TO_ASSETS
    where:
        ACCRUALS_INDEX = (net_income - OCF) / total_assets  (accruals ratio)
        TOTAL_ACCRUALS_TO_ASSETS = (net_income - OCF) / avg_total_assets

    A single-period approximation:
        OCF ≈ ebit * (1 - eff_tax_rate) * income_quality_ratio
    Falls back to None if insufficient data.
    """
    inc = bundle.income
    bal = bundle.balance
    km_ttm = bundle.key_metrics_ttm
    ratios_ttm = bundle.ratios_ttm

    # Get base inputs
    total_assets = _sf(bal.get("totalAssets"))
    net_income = _sf(inc.get("netIncome"))
    revenue = _sf(inc.get("revenue"))
    ebit = _sf(inc.get("ebit") or inc.get("operatingIncome"))

    if not total_assets or total_assets <= 0:
        return None

    # Estimate OCF from income_quality_ratio (OCF/NetIncome) if available
    income_quality = _sf(km_ttm.get("incomeQualityTTM"))  # OCF/NetIncome
    ocf = None
    if net_income is not None and income_quality is not None:
        ocf = net_income * income_quality

    # If no net_income, try to derive from profit margin × revenue
    if net_income is None and revenue is not None:
        pm = _sf(ratios_ttm.get("netProfitMarginTTM"))
        if pm is not None:
            net_income = revenue * pm

    if net_income is None or ocf is None:
        return None

    # Accruals = net_income - OCF
    accruals = net_income - ocf
    accruals_to_assets = accruals / total_assets

    # Simplified M-Score using accruals only (intercept calibrated to Beneish mean)
    # Full model needs YoY changes (DSRI, GMI, etc.) — not available with single-period data.
    # This reduced form is directionally correct: high accruals → higher M-Score.
    m_score = -6.065 + 4.679 * accruals_to_assets
    return round(m_score, 3)


def _compute_factor_scores(bundle: FMDataBundle) -> FactorScores:
    scores_data = bundle.scores
    result = FactorScores()

    if scores_data:
        result.piotroski_f_score = _int_safe(
            scores_data.get("piotroskiScore") or scores_data.get("piotroski")
        )
        result.beneish_m_score = _sf(
            scores_data.get("beneishMScore") or scores_data.get("beneish_m_score")
        )
        result.altman_z_score = _sf(
            scores_data.get("altmanZScore") or scores_data.get("altman_z_score")
        )

    # Derive Altman Z-Score from financial statements if not pre-computed
    if result.altman_z_score is None:
        result.altman_z_score = _compute_altman_z(bundle)

    # Derive Beneish M-Score if not pre-computed in DB.
    # Simplified accruals-based model (YoY deltas unavailable with single-period data).
    if result.beneish_m_score is None:
        result.beneish_m_score = _compute_beneish_m(bundle)

    return result


def _get_current_price(bundle: FMDataBundle) -> Optional[float]:
    """Extract current (latest) price from various available sources."""
    # Try TTM metrics first
    p = _sf(
        bundle.key_metrics_ttm.get("stockPriceTTM")
        or bundle.ratios_ttm.get("stockPriceTTM")
    )
    if p and p > 0:
        return round(p, 4)
    # Fall back to most recent price history entry (newest-first)
    if bundle.price_history:
        row = bundle.price_history[0]
        c = _sf(
            row.get("adjClose") or row.get("adjusted_close")
            or row.get("close") or row.get("Close")
        )
        if c and c > 0:
            return round(c, 4)
    return None


# ---------------------------------------------------------------------------
# LLM client (mirror of quant_fundamental/llm.py pattern)
# ---------------------------------------------------------------------------

def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_markdown_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text.strip())
    return text.strip()


def _clean_response(text: str) -> str:
    cleaned = _strip_think_tags(text)
    cleaned = _strip_markdown_fences(cleaned)
    if cleaned.strip().startswith("{"):
        try:
            obj = json.loads(cleaned)
            for key in ("quantitative_summary", "summary", "narrative", "analysis"):
                val = obj.get(key, "")
                if val and isinstance(val, str) and len(val) > 20:
                    return val.strip()
        except (json.JSONDecodeError, AttributeError):
            pass
    return cleaned.strip()


def _generate_summary(config: FinancialModellingConfig, factor_table: Dict[str, Any]) -> str:
    """Call Ollama to produce the quantitative_summary narrative string."""
    system = build_system_prompt(from_planner=False)
    lines = []
    for section, val in factor_table.items():
        if isinstance(val, dict):
            for k, v in val.items():
                lines.append(f"  {k}: {v}" if v is not None else f"  {k}: null")
        elif isinstance(val, list):
            lines.append(f"  {section}: {json.dumps(val, default=str)}" if val else f"  {section}: (none)")
        else:
            lines.append(f"  {section}: {val}")

    prompt = (
        f"{system}\n\n"
        f"--- FACTOR TABLE (do NOT recalculate these values) ---\n"
        f"{chr(10).join(lines)}\n"
        f"--- END FACTOR TABLE ---\n\n"
        f"Write your 10-15 sentence quantitative narrative now:"
    )

    payload = {
        "model": config.llm_model,
        "prompt": prompt,
        "temperature": config.llm_temperature,
        "num_predict": config.llm_max_tokens,
        "stream": False,
        "think": False,
    }
    try:
        resp = requests.post(
            f"{config.ollama_base_url}/api/generate",
            json=payload,
            timeout=config.request_timeout,
        )
        resp.raise_for_status()
        content = resp.json().get("response", "")
        if not content.strip():
            return "Quantitative summary unavailable (LLM returned empty response)."
        cleaned = _clean_response(content)
        return cleaned if cleaned and len(cleaned) > 20 else "Quantitative summary unavailable."
    except requests.exceptions.ConnectionError:
        logger.warning("Ollama not reachable — skipping LLM summary")
        return "Quantitative summary unavailable (LLM offline)."
    except Exception as exc:
        logger.warning("LLM summary generation failed: %s", exc)
        return f"Quantitative summary unavailable ({type(exc).__name__})."


# ---------------------------------------------------------------------------
# Node 8: format_json_output
# ---------------------------------------------------------------------------

def _node_format_json_output(
    state: AgentState,
    config: FinancialModellingConfig,
) -> AgentState:
    """Assemble final JSON output and call LLM for narrative summary."""
    ticker = state.get("ticker", "UNKNOWN")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    bundle: Optional[FMDataBundle] = state.get("bundle")
    technicals: TechnicalSnapshot = state.get("technicals") or TechnicalSnapshot()
    dcf: DCFResult = state.get("dcf_result") or DCFResult()
    moe_consensus: Optional[Dict[str, Any]] = state.get("moe_consensus")   # 4A
    macro_env: Optional[Dict[str, Any]] = state.get("macro_environment")   # 4B
    comps = state.get("comps_result")
    earnings: EarningsRecord = state.get("earnings") or EarningsRecord()
    dividends: DividendRecord = state.get("dividends") or DividendRecord()
    factor_scores: FactorScores = state.get("factor_scores") or FactorScores()
    current_price: Optional[float] = state.get("current_price")
    fetch_error = state.get("fetch_error")

    # Lazy import to avoid circular
    from .schema import CompsResult, ValuationResult
    if comps is None:
        comps = CompsResult()

    # Implied price range from DCF scenarios
    implied_range = {
        "low": dcf.intrinsic_value_bear,
        "mid": dcf.intrinsic_value_base,
        "high": dcf.intrinsic_value_bull,
    }

    valuation = {
        "dcf": dcf.to_dict(),
        "comps": comps.to_dict(),
        "implied_price_range": implied_range,
    }

    # Factor table for LLM — all pre-computed numbers, no calculations by LLM
    factor_table = {
        "ticker": ticker,
        "as_of_date": today,
        "current_price": current_price,
        "dcf": dcf.to_dict(),
        "moe_consensus": moe_consensus,             # 4A: MoE DCF consensus
        "comps": comps.to_dict(),
        "technicals": technicals.to_dict(),
        "earnings": earnings.to_dict(),
        "dividends": dividends.to_dict(),
        "factor_scores": factor_scores.to_dict(),
    }

    quantitative_summary = _generate_summary(config, factor_table)

    data_sources: Dict[str, Any] = {
        "price_data": "postgresql:raw_timeseries",
        "fundamentals": "postgresql:raw_fundamentals",
        "peer_group": "neo4j:COMPETES_WITH",
        "treasury_rates": "postgresql:raw_timeseries (FMP treasury endpoint)",
        "market_risk_premium": "postgresql:raw_fundamentals (FMP market risk premium endpoint)",
        "llm_scope": "quantitative_summary narrative only",
        "llm_model": config.llm_model,
    }
    if fetch_error:
        data_sources["fetch_error"] = fetch_error

    output: Dict[str, Any] = {
        "agent": "financial_modelling",
        "ticker": ticker,
        "as_of_date": today,
        "current_price": current_price,
        "valuation": valuation,
        "moe_consensus": moe_consensus,             # 4A: MoE DCF consensus
        "macro_environment": macro_env,             # 4B: VIX-adjusted MRP snapshot
        "technicals": technicals.to_dict(),
        "earnings": earnings.to_dict(),
        "dividends": dividends.to_dict(),
        "factor_scores": factor_scores.to_dict(),
        "quantitative_summary": quantitative_summary,
        "data_sources": data_sources,
        # Price series forwarded for Streamlit charts (serialised OHLCV, oldest-first)
        "price_history": _serialise_price_history(
            bundle.price_history if bundle else []
        ),
        "benchmark_history": _serialise_price_history(
            bundle.benchmark_history if bundle else []
        ),
    }

    return {**state, "output": output}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(toolkit: FMToolkit, config: FinancialModellingConfig) -> Any:
    """Assemble and compile the 8-node LangGraph pipeline."""
    graph = StateGraph(AgentState)

    graph.add_node(
        "fetch_price_history",
        lambda state: _node_fetch_price_history(cast(AgentState, state), toolkit),
    )
    graph.add_node(
        "fetch_fundamentals",
        lambda state: _node_fetch_fundamentals(cast(AgentState, state), toolkit),
    )
    graph.add_node(
        "fetch_earnings_history",
        lambda state: _node_fetch_earnings_history(cast(AgentState, state)),
    )
    graph.add_node(
        "calculate_technicals",
        lambda state: _node_calculate_technicals(cast(AgentState, state)),
    )
    graph.add_node(
        "macro_environment",
        lambda state: _node_macro_environment(cast(AgentState, state)),
    )
    graph.add_node(
        "run_dcf_model",
        lambda state: _node_run_dcf_model(cast(AgentState, state), config),
    )
    graph.add_node(
        "moe_consensus",
        lambda state: _node_moe_consensus(cast(AgentState, state), config),
    )
    graph.add_node(
        "run_comparable_analysis",
        lambda state: _node_run_comparable_analysis(cast(AgentState, state), config),
    )
    graph.add_node(
        "assess_analyst_estimates",
        lambda state: _node_assess_analyst_estimates(cast(AgentState, state)),
    )
    graph.add_node(
        "format_json_output",
        lambda state: _node_format_json_output(cast(AgentState, state), config),
    )

    graph.set_entry_point("fetch_price_history")
    graph.add_edge("fetch_price_history", "fetch_fundamentals")
    graph.add_edge("fetch_fundamentals", "fetch_earnings_history")
    graph.add_edge("fetch_earnings_history", "calculate_technicals")
    graph.add_edge("calculate_technicals", "macro_environment")  # 4B: macro before DCF
    graph.add_edge("macro_environment", "run_dcf_model")
    graph.add_edge("run_dcf_model", "moe_consensus")          # 4A: MoE after DCF
    graph.add_edge("moe_consensus", "run_comparable_analysis")
    graph.add_edge("run_comparable_analysis", "assess_analyst_estimates")
    graph.add_edge("assess_analyst_estimates", "format_json_output")
    graph.add_edge("format_json_output", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Ticker extraction (mirrors quant_fundamental pattern)
# ---------------------------------------------------------------------------

_KNOWN_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "TSLA", "META", "BRK.B",
    "UNH", "JNJ", "XOM", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
    "COST", "PEP", "KO", "AVGO", "LLY", "WMT", "BAC", "MCD", "NFLX", "ADBE",
    "CRM", "TMO", "ACN", "CSCO", "ABT", "NKE", "ORCL", "DHR", "QCOM", "TXN",
    "LIN", "PM", "NEE", "RTX", "INTU", "AMGN", "BMY", "SPGI", "MS", "GS",
    "AMD", "INTC", "SBUX", "NOW", "PLD", "ISRG", "AXP", "CAT", "DE", "MDT",
    "IBM", "PYPL", "ZM", "SHOP", "SQ", "UBER", "LYFT", "SNAP", "COIN", "HOOD",
    "RBLX", "U", "RIVN", "LCID", "F", "GM", "BA", "DIS",
}

_EXCHANGE_SUFFIX_PATTERN = re.compile(
    r"\b(\d{1,6}\.[A-Z]{1,4})\b|"
    r"\b([A-Z]{1,5}\.[A-Z]{1,4})\b"
)
_US_TICKER_PATTERN = re.compile(r"\b([A-Z]{1,5})\b")
_STOP: set = {
    "A", "I", "THE", "FOR", "ON", "IN", "OF", "AT", "BY", "TO", "DO",
    "RUN", "GET", "USE", "AND", "OR", "NOT", "ALL", "ANY", "CAN", "IS",
    "ARE", "WAS", "BE", "AN", "AS", "IF", "IT", "US", "NO", "YES", "ME",
    "MY", "UP", "OUT", "NEW", "FY", "TTM", "YTD", "EPS", "ROE", "DCF",
    "PE", "PB", "EV", "EBIT", "ROA", "ROIC", "FCF", "IPO", "CEO", "CFO",
    "LLM", "API", "SQL", "DB", "URL", "ID", "OK", "NA", "NaN",
    "HERE", "THERE", "WHAT", "WHEN", "WHERE", "WHO", "WHY", "HOW",
    "THIS", "THAT", "WITH", "FROM", "HAVE", "HAS", "HAD", "WILL",
    "WOULD", "COULD", "SHOULD", "PLEASE", "JUST", "ALSO", "ONLY",
    "SOME", "MORE", "MOST", "OVER", "THAN", "INTO", "ABOUT", "AFTER",
    "BEFORE", "THEN", "WHICH", "THEIR", "THEM", "THEY", "YOUR",
    "REPORT", "STOCK", "ANALYSIS", "DATA", "FULL", "LAST", "NEXT",
    "COMPARE", "BETWEEN", "VERSUS", "VS", "HK", "US", "L", "T", "AX",
    "PA", "DE", "TO", "SS", "WACC", "SMA", "EMA", "RSI", "MACD", "ATR",
}


def extract_tickers_from_prompt(prompt: str) -> List[str]:
    if not prompt:
        return []
    text = prompt.strip()
    candidates: List[tuple] = []
    seen: set = set()

    def _add(pos: int, t: str) -> None:
        t = t.upper()
        if t not in seen:
            seen.add(t)
            candidates.append((pos, t))

    for m in _EXCHANGE_SUFFIX_PATTERN.finditer(text):
        candidate = (m.group(1) or m.group(2)).upper()
        root = candidate.split(".")[0]
        if root.upper() not in _STOP:
            _add(m.start(), candidate)

    for m in re.finditer(r"\bticker[:\s]+([A-Za-z0-9.]{1,10})\b", text, re.IGNORECASE):
        candidate = m.group(1)
        if candidate == candidate.upper():
            _add(m.start(1), candidate)

    for m in re.finditer(r"[\(\[]([A-Z0-9]{1,6}(?:\.[A-Z]{1,4})?)[\)\]]", text):
        _add(m.start(1), m.group(1))

    upper = text.upper()
    for ticker in _KNOWN_TICKERS:
        m = re.search(r"\b" + re.escape(ticker) + r"\b", upper)
        if m:
            _add(m.start(), ticker)

    if not candidates:
        for m in _US_TICKER_PATTERN.finditer(text):
            cand = m.group(1)
            if cand.upper() not in _STOP and len(cand) >= 2:
                _add(m.start(), cand)

    candidates.sort(key=lambda x: x[0])
    return [t for _, t in candidates]


def extract_ticker_from_prompt(prompt: str) -> Optional[str]:
    tickers = extract_tickers_from_prompt(prompt)
    return tickers[0] if tickers else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _run_single(
    ticker: str,
    config: FinancialModellingConfig,
    toolkit: FMToolkit,
) -> Dict[str, Any]:
    """Run the pipeline for one resolved ticker."""
    compiled = build_graph(toolkit, config)
    initial_state: AgentState = {"ticker": ticker}
    final_state = compiled.invoke(initial_state)
    return final_state.get("output") or {}


def run(
    ticker: str = "",
    config: Optional[FinancialModellingConfig] = None,
    *,
    prompt: Optional[str] = None,
) -> Any:
    """Run the Financial Modelling pipeline for one or more tickers.

    Args:
        ticker: Ticker symbol (takes precedence over prompt).
        config: Optional config override.
        prompt: Natural-language prompt; ticker parsed automatically.

    Returns:
        Single dict for one ticker, list of dicts for multiple tickers.
    """
    cfg = config or load_config()

    if ticker.strip():
        resolved_tickers = [ticker.strip().upper()]
    elif prompt:
        resolved_tickers = extract_tickers_from_prompt(prompt)
    else:
        resolved_tickers = []

    if not resolved_tickers:
        raise ValueError(
            "No ticker provided. Pass ticker='AAPL' or prompt='Analyze AAPL valuation'."
        )

    toolkit = FMToolkit(cfg)
    try:
        results = [_run_single(t, cfg, toolkit) for t in resolved_tickers]
    finally:
        toolkit.close()

    return results[0] if len(results) == 1 else results


def run_full_analysis(
    ticker: str = "",
    config: Optional[FinancialModellingConfig] = None,
    *,
    prompt: Optional[str] = None,
) -> Any:
    """Run a complete financial modelling valuation dossier.

    This is the canonical entry point for the Supervisor/Planner.

    Args:
        ticker: Ticker symbol (e.g. "AAPL"). Takes precedence over prompt.
        config: Optional config override.
        prompt: Natural-language instruction (ticker parsed automatically).

    Returns:
        Dict for single ticker, List[Dict] for multiple tickers.

    Example::

        from agents.financial_modelling.agent import run_full_analysis
        report = run_full_analysis(ticker="AAPL")
        report["valuation"]["dcf"]["intrinsic_value_base"]  # → float
        report["technicals"]["rsi_14"]                       # → float
    """
    if not ticker and not prompt:
        raise ValueError("Either ticker or prompt is required for run_full_analysis().")
    return run(ticker=ticker, config=config, prompt=prompt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="financial_modelling",
        description=(
            "Run the Financial Modelling pipeline for one or more tickers.\n\n"
            "Input modes:\n"
            "  Direct:  --ticker AAPL\n"
            "  Prompt:  --prompt 'Analyze AAPL valuation'\n"
            "  Multi:   --prompt 'Compare MSFT vs AAPL DCF'\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", type=str, metavar="SYMBOL")
    group.add_argument("--prompt", type=str, metavar="TEXT")
    parser.add_argument("--pretty", action="store_true", default=True)
    parser.add_argument(
        "--log-level", type=str, default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
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
        print(json.dumps(result, indent=indent, default=str))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.error("Agent run failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
