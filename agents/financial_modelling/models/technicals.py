"""Technical Analysis engine — price-based indicators from allowed EOD/weekly data.

Allowed data sources (per data boundary enforcement):
  - historical_prices_eod  (~2 years max, daily)
  - historical_prices_weekly (~2 years max, weekly)

Computes:
  - Beta (vs S&P 500 benchmark — input to WACC)
  - Trend (based on SMA50/SMA200 relationship)
  - RSI-14, MACD (12/26/9), Bollinger Bands (20,2), ATR-14, HV-30
  - SMA 20/50/200, EMA 12/26
  - Stochastic %K/%D (14,3)
  - Support/Resistance (52-week levels)
  - Golden cross / Death cross flags
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from ..schema import FMDataBundle, TechnicalSnapshot

logger = logging.getLogger(__name__)


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _extract_closes(price_history: List[Dict[str, Any]]) -> List[float]:
    """Extract adjusted close prices (chronological, oldest first)."""
    closes: List[float] = []
    for row in reversed(price_history):
        c = _safe_float(
            row.get("adjClose") or row.get("adjusted_close")
            or row.get("close") or row.get("Close")
        )
        if c is not None and c > 0:
            closes.append(c)
    return closes


def _extract_high_low(price_history: List[Dict[str, Any]]) -> tuple[List[float], List[float]]:
    """Extract high and low prices (chronological, oldest first)."""
    highs: List[float] = []
    lows: List[float] = []
    for row in reversed(price_history):
        h = _safe_float(row.get("high") or row.get("High"))
        l = _safe_float(row.get("low") or row.get("Low"))
        c = _safe_float(
            row.get("adjClose") or row.get("adjusted_close")
            or row.get("close") or row.get("Close")
        )
        # Use close as fallback for high/low if missing
        if h is not None and l is not None:
            highs.append(h)
            lows.append(l)
        elif c is not None:
            highs.append(c)
            lows.append(c)
    return highs, lows


# ── Indicator helpers ────────────────────────────────────────────────────────

def _sma(series: List[float], period: int) -> Optional[float]:
    if len(series) < period:
        return None
    return round(sum(series[-period:]) / period, 4)


def _ema(series: List[float], period: int) -> List[float]:
    """Return full EMA series (same length as input starting from period-1)."""
    if len(series) < period:
        return []
    k = 2 / (period + 1)
    ema_val = sum(series[:period]) / period
    result = [ema_val]
    for price in series[period:]:
        ema_val = price * k + ema_val * (1 - k)
        result.append(ema_val)
    return result


def _rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    # Initial averages
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 2)


def _macd(closes: List[float]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (macd_line, signal_line, histogram)."""
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    if not ema12 or not ema26:
        return None, None, None
    # Align: ema26 starts 26 periods in, ema12 starts 12 periods in
    # Difference series starts where both are valid
    offset = 26 - 12  # ema26 is shorter by 14 values
    if len(ema12) <= offset:
        return None, None, None
    macd_series = [ema12[offset + i] - ema26[i] for i in range(len(ema26))]
    signal_series = _ema(macd_series, 9)
    if not signal_series:
        return None, None, None
    macd_val = round(macd_series[-1], 4)
    signal_val = round(signal_series[-1], 4)
    hist_val = round(macd_val - signal_val, 4)
    return macd_val, signal_val, hist_val


def _bollinger(closes: List[float], period: int = 20, num_std: float = 2.0) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (upper, lower, position_0_to_1)."""
    if len(closes) < period:
        return None, None, None
    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    std = math.sqrt(variance)
    upper = round(mid + num_std * std, 4)
    lower = round(mid - num_std * std, 4)
    current = closes[-1]
    band_width = upper - lower
    position = round((current - lower) / band_width, 4) if band_width > 0 else None
    return upper, lower, position


def _atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """Average True Range."""
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    return round(sum(trs[-period:]) / period, 4)


def _hv30(closes: List[float]) -> Optional[float]:
    """30-day Historical Volatility (annualised)."""
    if len(closes) < 31:
        return None
    log_rets = [math.log(closes[i] / closes[i - 1]) for i in range(len(closes) - 30, len(closes))]
    mean = sum(log_rets) / len(log_rets)
    variance = sum((r - mean) ** 2 for r in log_rets) / len(log_rets)
    return round(math.sqrt(variance * 252), 4)


def _stochastic(
    highs: List[float], lows: List[float], closes: List[float],
    k_period: int = 14, d_period: int = 3,
) -> tuple[Optional[float], Optional[float]]:
    """Stochastic oscillator %K and %D."""
    if len(closes) < k_period:
        return None, None
    k_values = []
    for i in range(k_period - 1, len(closes)):
        window_h = max(highs[i - k_period + 1: i + 1])
        window_l = min(lows[i - k_period + 1: i + 1])
        denom = window_h - window_l
        if denom == 0:
            k_values.append(50.0)
        else:
            k_values.append(100 * (closes[i] - window_l) / denom)
    if not k_values:
        return None, None
    k = round(k_values[-1], 2)
    if len(k_values) >= d_period:
        d = round(sum(k_values[-d_period:]) / d_period, 2)
    else:
        d = None
    return k, d


def compute_beta(
    price_history: List[Dict[str, Any]],
    benchmark_history: List[Dict[str, Any]],
    lookback_weeks: int = 104,
) -> Optional[float]:
    """Compute rolling beta of ticker vs. S&P 500 using weekly returns.

    Uses up to `lookback_weeks` weekly data points (2-year = 104 weeks).
    """
    asset_closes_raw = _extract_closes(price_history)
    bench_closes_raw = _extract_closes(benchmark_history)

    if len(asset_closes_raw) > lookback_weeks * 2:
        asset_closes_raw = asset_closes_raw[::-1][::5][::-1]
    if len(bench_closes_raw) > lookback_weeks * 2:
        bench_closes_raw = bench_closes_raw[::-1][::5][::-1]

    asset_closes = asset_closes_raw[-(lookback_weeks + 1):]
    bench_closes = bench_closes_raw[-(lookback_weeks + 1):]

    n = min(len(asset_closes), len(bench_closes)) - 1
    if n < 10:
        logger.debug("compute_beta: insufficient benchmark data (n=%d)", n)
        return None

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


class TechnicalEngine:
    """Computes technical indicators from allowed EOD/weekly price data.

    Allowed data: historical_prices_eod, historical_prices_weekly (~2 years max).
    """

    def compute(self, bundle: FMDataBundle) -> TechnicalSnapshot:
        snap = TechnicalSnapshot()

        if not bundle.price_history:
            logger.warning("TechnicalEngine: no price history for %s", bundle.ticker)
            return snap

        closes = _extract_closes(bundle.price_history)
        highs, lows = _extract_high_low(bundle.price_history)

        if len(closes) < 2:
            return snap

        # ── Beta (vs benchmark) ──────────────────────────────────────────────
        if bundle.benchmark_history:
            snap.beta = compute_beta(
                bundle.price_history,
                bundle.benchmark_history,
                lookback_weeks=104,
            )

        # ── Moving averages ──────────────────────────────────────────────────
        snap.sma_20 = _sma(closes, 20)
        snap.sma_50 = _sma(closes, 50)
        snap.sma_200 = _sma(closes, 200)

        ema12_series = _ema(closes, 12)
        ema26_series = _ema(closes, 26)
        snap.ema_12 = round(ema12_series[-1], 4) if ema12_series else None
        snap.ema_26 = round(ema26_series[-1], 4) if ema26_series else None

        # ── Trend ────────────────────────────────────────────────────────────
        if snap.sma_50 and snap.sma_200:
            snap.sma_50_above_200 = snap.sma_50 > snap.sma_200
            current = closes[-1]
            prev_sma50 = _sma(closes[:-1], 50) if len(closes) > 50 else None
            prev_sma200 = _sma(closes[:-1], 200) if len(closes) > 200 else None

            # Golden/Death cross detection (today vs yesterday)
            if prev_sma50 is not None and prev_sma200 is not None:
                was_below = prev_sma50 <= prev_sma200
                is_above = snap.sma_50 > snap.sma_200
                snap.golden_cross = was_below and is_above
                snap.death_cross = (not was_below) and (not is_above)

            if snap.sma_50_above_200 and current > snap.sma_50:
                snap.trend = "bullish"
            elif not snap.sma_50_above_200 and current < snap.sma_50:
                snap.trend = "bearish"
            else:
                snap.trend = "neutral"
        elif snap.sma_50:
            snap.trend = "bullish" if closes[-1] > snap.sma_50 else "bearish"

        # ── RSI ──────────────────────────────────────────────────────────────
        snap.rsi_14 = _rsi(closes, 14)

        # ── MACD ─────────────────────────────────────────────────────────────
        macd_val, macd_sig_val, macd_hist_val = _macd(closes)
        snap.macd = macd_val                    # sets macd_histogram via property
        snap.macd_signal_line = macd_sig_val    # float signal line value
        # Derive direction string from histogram
        if macd_hist_val is not None:
            if macd_hist_val > 0:
                snap.macd_signal = "buy"
            elif macd_hist_val < 0:
                snap.macd_signal = "sell"
            else:
                snap.macd_signal = "neutral"

        # ── Bollinger Bands ──────────────────────────────────────────────────
        bb_upper, bb_lower, bb_pos_float = _bollinger(closes)
        snap.bollinger_upper = bb_upper
        snap.bollinger_lower = bb_lower
        if bb_pos_float is not None:
            if bb_pos_float > 1.0:
                snap.bollinger_position = "above_upper"
            elif bb_pos_float >= 0.7:
                snap.bollinger_position = "upper"
            elif bb_pos_float >= 0.3:
                snap.bollinger_position = "mid"
            elif bb_pos_float >= 0.0:
                snap.bollinger_position = "lower"
            else:
                snap.bollinger_position = "below_lower"

        # ── ATR ──────────────────────────────────────────────────────────────
        if highs and lows and len(highs) == len(lows) == len(closes):
            snap.atr_14 = _atr(highs, lows, closes, 14)

        # ── Historical Volatility 30 ─────────────────────────────────────────
        snap.hv_30 = _hv30(closes)

        # ── Stochastic ───────────────────────────────────────────────────────
        if highs and lows and len(highs) == len(lows) == len(closes):
            snap.stochastic_k, snap.stochastic_d = _stochastic(highs, lows, closes)

        # ── 52-week Support / Resistance ─────────────────────────────────────
        # Use up to 252 daily or 52 weekly data points
        lookback = min(252, len(closes))
        if lookback >= 20:
            window_closes = closes[-lookback:]
            window_highs = highs[-lookback:] if highs else window_closes
            window_lows = lows[-lookback:] if lows else window_closes
            snap.support = round(min(window_lows), 4)
            snap.resistance = round(max(window_highs), 4)
            snap.high_52w = snap.resistance
            snap.low_52w = snap.support

        logger.debug(
            "TechnicalEngine: %s trend=%s rsi=%.1f beta=%s macd=%s",
            bundle.ticker, snap.trend, snap.rsi_14 or 0, snap.beta, snap.macd,
        )
        return snap


__all__ = ["TechnicalEngine", "compute_beta"]
