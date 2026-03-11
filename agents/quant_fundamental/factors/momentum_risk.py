"""Momentum and risk factor calculations: Beta (60-day), Sharpe Ratio (12m), 12m return,
SMA-50, SMA-200, golden/death cross, and volume analysis.

All calculations use price data from raw_timeseries (ticker) and market_eod_us (S&P 500 benchmark).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..schema import FinancialsBundle, MomentumRiskFactors

logger = logging.getLogger(__name__)

# Annualisation constant
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE_ANNUAL = 0.05  # 5% — approximate US risk-free rate


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------

def _parse_price_rows(
    rows: List[Dict[str, Any]],
) -> List[Tuple[datetime, float]]:
    """Parse a list of payload dicts into sorted (date, close) tuples, oldest first.

    Handles both raw_timeseries payloads (with 'close' or 'adjClose') and
    market_eod_us payloads (with 'close').
    """
    parsed: List[Tuple[datetime, float]] = []
    for row in rows:
        # payload may be the row itself or a nested dict
        p = row if isinstance(row, dict) else {}
        date_str = (
            p.get("date")
            or p.get("ts_date")
            or p.get("timestamp")
        )
        close = (
            p.get("adjClose")
            or p.get("adjusted_close")
            or p.get("close")
            or p.get("adj_close")
        )
        if date_str is None or close is None:
            continue
        try:
            if isinstance(date_str, datetime):
                dt = date_str
            else:
                # Support "2024-01-15" and "2024-01-15T00:00:00"
                dt = datetime.fromisoformat(str(date_str).split("T")[0])
            price = float(close)
            if price > 0:
                parsed.append((dt, price))
        except (ValueError, TypeError):
            continue

    # Sort ascending (oldest first)
    parsed.sort(key=lambda x: x[0])
    return parsed


def _parse_price_volume_rows(
    rows: List[Dict[str, Any]],
) -> List[Tuple[datetime, float, Optional[float]]]:
    """Parse price rows into sorted (date, close, volume) tuples, oldest first.

    Volume may be None if not present in the payload.
    """
    parsed: List[Tuple[datetime, float, Optional[float]]] = []
    for row in rows:
        p = row if isinstance(row, dict) else {}
        date_str = p.get("date") or p.get("ts_date") or p.get("timestamp")
        close = (
            p.get("adjClose")
            or p.get("adjusted_close")
            or p.get("close")
            or p.get("adj_close")
        )
        volume = p.get("volume")
        if date_str is None or close is None:
            continue
        try:
            if isinstance(date_str, datetime):
                dt = date_str
            else:
                dt = datetime.fromisoformat(str(date_str).split("T")[0])
            price = float(close)
            vol = float(volume) if volume is not None else None
            if price > 0:
                parsed.append((dt, price, vol))
        except (ValueError, TypeError):
            continue
    parsed.sort(key=lambda x: x[0])
    return parsed


def _daily_returns(prices: List[float]) -> List[float]:
    """Compute log daily returns from a list of prices."""
    if len(prices) < 2:
        return []
    returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] > 0 and prices[i] > 0:
            returns.append(math.log(prices[i] / prices[i - 1]))
    return returns


def _align_series(
    ticker_series: List[Tuple[datetime, float]],
    bench_series: List[Tuple[datetime, float]],
) -> Tuple[List[float], List[float]]:
    """Inner-join ticker and benchmark prices on trading date.

    Returns two aligned lists of prices (same length, same dates).
    """
    bench_map: Dict[str, float] = {
        dt.strftime("%Y-%m-%d"): price for dt, price in bench_series
    }
    aligned_ticker: List[float] = []
    aligned_bench: List[float] = []

    for dt, price in ticker_series:
        date_key = dt.strftime("%Y-%m-%d")
        if date_key in bench_map:
            aligned_ticker.append(price)
            aligned_bench.append(bench_map[date_key])

    return aligned_ticker, aligned_bench


# ---------------------------------------------------------------------------
# Beta (60-day rolling)
# ---------------------------------------------------------------------------

def compute_beta_60d(
    ticker_prices: List[Tuple[datetime, float]],
    bench_prices: List[Tuple[datetime, float]],
    lookback_days: int = 60,
) -> Optional[float]:
    """Compute 60-trading-day rolling beta vs. benchmark (S&P 500).

    Beta = Cov(r_ticker, r_bench) / Var(r_bench)
    """
    if not ticker_prices or not bench_prices:
        return None

    # Use the most recent lookback_days + 1 price points
    recent_ticker = ticker_prices[-(lookback_days + 1):]
    recent_bench = bench_prices[-(lookback_days + 1):]

    t_aligned, b_aligned = _align_series(recent_ticker, recent_bench)

    if len(t_aligned) < 10:  # Require at least 10 aligned points
        logger.debug("compute_beta_60d: insufficient aligned data (%d points)", len(t_aligned))
        return None

    t_returns = _daily_returns(t_aligned)
    b_returns = _daily_returns(b_aligned)

    n = min(len(t_returns), len(b_returns))
    if n < 10:
        return None

    t_r = t_returns[-n:]
    b_r = b_returns[-n:]

    mean_t = sum(t_r) / n
    mean_b = sum(b_r) / n

    cov = sum((t_r[i] - mean_t) * (b_r[i] - mean_b) for i in range(n)) / n
    var_b = sum((b_r[i] - mean_b) ** 2 for i in range(n)) / n

    if var_b == 0:
        return None

    beta = cov / var_b
    return round(beta, 4)


# ---------------------------------------------------------------------------
# Sharpe Ratio (12-month)
# ---------------------------------------------------------------------------

def compute_sharpe_12m(
    ticker_prices: List[Tuple[datetime, float]],
    lookback_days: int = 252,
    risk_free_annual: float = RISK_FREE_RATE_ANNUAL,
) -> Optional[float]:
    """Compute 12-month (252 trading day) Sharpe ratio.

    Sharpe = (Annualised Return - Risk-Free Rate) / Annualised Volatility
    Daily risk-free rate = (1 + rf_annual)^(1/252) - 1
    """
    if not ticker_prices:
        return None

    recent = ticker_prices[-(lookback_days + 1):]
    prices = [p for _, p in recent]

    if len(prices) < 20:
        return None

    returns = _daily_returns(prices)
    if not returns:
        return None

    n = len(returns)
    mean_r = sum(returns) / n
    variance = sum((r - mean_r) ** 2 for r in returns) / n
    daily_std = math.sqrt(variance) if variance > 0 else 0.0

    if daily_std == 0:
        return None

    # Annualise
    ann_return = mean_r * TRADING_DAYS_PER_YEAR
    ann_std = daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)

    sharpe = (ann_return - risk_free_annual) / ann_std
    return round(sharpe, 4)


# ---------------------------------------------------------------------------
# 12-month price return
# ---------------------------------------------------------------------------

def compute_return_12m(
    ticker_prices: List[Tuple[datetime, float]],
    lookback_days: int = 252,
) -> Optional[float]:
    """Compute the 12-month (252-trading-day) price return as a percentage.

    Uses the most recent price vs. the price ~252 trading days ago.
    """
    if not ticker_prices or len(ticker_prices) < 2:
        return None

    current_price = ticker_prices[-1][1]

    # Find the price closest to 252 trading days ago
    start_price: Optional[float] = None
    if len(ticker_prices) >= lookback_days:
        start_price = ticker_prices[-lookback_days][1]
    else:
        start_price = ticker_prices[0][1]

    if start_price is None or start_price <= 0:
        return None

    pct_return = (current_price - start_price) / start_price * 100.0
    return round(pct_return, 2)


# ---------------------------------------------------------------------------
# SMA-50, SMA-200 and golden/death cross
# ---------------------------------------------------------------------------

def compute_sma(prices: List[float], period: int) -> Optional[float]:
    """Simple moving average of the last `period` prices."""
    if len(prices) < period:
        return None
    window = prices[-period:]
    return round(sum(window) / period, 4)


def compute_sma_cross(
    ticker_prices: List[Tuple[datetime, float]],
) -> Tuple[Optional[float], Optional[float], Optional[bool]]:
    """Compute SMA-50, SMA-200, and golden/death cross from price history.

    Returns:
        (sma_50, sma_200, golden_cross)
        golden_cross = True  → SMA-50 > SMA-200 (bullish)
        golden_cross = False → SMA-50 < SMA-200 (death cross, bearish)
        golden_cross = None  → insufficient data
    """
    prices = [p for _, p in ticker_prices]
    sma50 = compute_sma(prices, 50)
    sma200 = compute_sma(prices, 200)
    if sma50 is not None and sma200 is not None:
        golden = sma50 > sma200
    else:
        golden = None
    return sma50, sma200, golden


# ---------------------------------------------------------------------------
# Volume analysis
# ---------------------------------------------------------------------------

def compute_volume_analysis(
    rows: List[Dict[str, Any]],
    avg_window: int = 20,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute 20-day average volume and latest-day volume ratio.

    Args:
        rows:       Raw price history rows (dicts with 'volume' field).
        avg_window: Window for average volume (default 20 trading days).

    Returns:
        (avg_volume_20d, volume_ratio)
        volume_ratio = latest_volume / avg_volume_20d
    """
    price_vol = _parse_price_volume_rows(rows)
    volumes = [v for _, _, v in price_vol if v is not None and v > 0]
    if len(volumes) < avg_window:
        if not volumes:
            return None, None
        # Use what we have
        avg_window = len(volumes)

    window_vols = volumes[-avg_window:]
    avg_vol = sum(window_vols) / len(window_vols)
    latest_vol = volumes[-1]
    ratio = round(latest_vol / avg_vol, 4) if avg_vol > 0 else None
    return round(avg_vol, 0), ratio


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_momentum_risk(
    bundle: FinancialsBundle,
    beta_lookback: int = 60,
    sharpe_lookback: int = 252,
) -> MomentumRiskFactors:
    """Compute momentum and risk factors from price history in the bundle.

    Args:
        bundle:          The FinancialsBundle loaded from PostgreSQL.
        beta_lookback:   Number of trading days for beta calculation (default 60).
        sharpe_lookback: Number of trading days for Sharpe/return (default 252).

    Returns:
        MomentumRiskFactors dataclass.
    """
    ticker_parsed = _parse_price_rows(bundle.price_history)
    bench_parsed = _parse_price_rows(bundle.benchmark_history)

    if not ticker_parsed:
        logger.warning("compute_momentum_risk: no price history for %s", bundle.ticker)
        return MomentumRiskFactors()

    beta = compute_beta_60d(ticker_parsed, bench_parsed, lookback_days=beta_lookback)
    sharpe = compute_sharpe_12m(ticker_parsed, lookback_days=sharpe_lookback)
    ret_12m = compute_return_12m(ticker_parsed, lookback_days=sharpe_lookback)
    sma50, sma200, golden = compute_sma_cross(ticker_parsed)
    avg_vol, vol_ratio = compute_volume_analysis(bundle.price_history)

    logger.debug(
        "Momentum/risk for %s: beta_60d=%.3f, sharpe_12m=%.3f, return_12m=%.2f%%, "
        "sma50=%.2f, sma200=%.2f, golden_cross=%s, avg_vol=%.0f, vol_ratio=%.2f",
        bundle.ticker,
        beta or float("nan"),
        sharpe or float("nan"),
        ret_12m or float("nan"),
        sma50 or float("nan"),
        sma200 or float("nan"),
        golden,
        avg_vol or float("nan"),
        vol_ratio or float("nan"),
    )

    return MomentumRiskFactors(
        beta_60d=beta,
        sharpe_ratio_12m=sharpe,
        return_12m_pct=ret_12m,
        sma_50=sma50,
        sma_200=sma200,
        golden_cross=golden,
        avg_volume_20d=avg_vol,
        volume_ratio=vol_ratio,
    )


__all__ = [
    "compute_momentum_risk",
    "compute_beta_60d",
    "compute_sharpe_12m",
    "compute_return_12m",
    "compute_sma_cross",
    "compute_volume_analysis",
]
