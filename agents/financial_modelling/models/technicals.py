"""Technical Analysis engine.

Computes all technical indicators from split-adjusted EOD price history.
All calculations are deterministic Python — no external TA library required,
no LLM involvement.

Indicators implemented:
  Trend:       SMA 20, SMA 50, SMA 200, EMA 12, EMA 26
               Golden Cross (SMA50 > SMA200), Death Cross (SMA50 < SMA200)
  Momentum:    RSI(14), MACD + Signal + Histogram, Stochastic %K/%D
  Volatility:  Bollinger Bands (20-period, 2σ), ATR(14), HV30 (annualised)
  Range:       52-week high/low, support (20-day low), resistance (20-day high)
  Trend label: "bullish" / "bearish" / "neutral"
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from ..schema import FMDataBundle, TechnicalSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    for row in reversed(price_history):   # price_history is newest-first from DB
        c = _safe_float(
            row.get("adjClose") or row.get("adjusted_close")
            or row.get("close") or row.get("Close")
        )
        if c is not None and c > 0:
            closes.append(c)
    return closes


def _extract_ohlcv(
    price_history: List[Dict[str, Any]]
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Return (opens, highs, lows, closes) in chronological order."""
    opens, highs, lows, closes = [], [], [], []
    for row in reversed(price_history):
        o = _safe_float(row.get("open") or row.get("Open"))
        h = _safe_float(row.get("high") or row.get("High"))
        l = _safe_float(row.get("low") or row.get("Low"))
        c = _safe_float(row.get("adjClose") or row.get("adjusted_close")
                        or row.get("close") or row.get("Close"))
        if h is not None and l is not None and c is not None:
            opens.append(o or c)
            highs.append(h)
            lows.append(l)
            closes.append(c)
    return opens, highs, lows, closes


# ---------------------------------------------------------------------------
# Core indicator functions (pure Python, no pandas/numpy required)
# ---------------------------------------------------------------------------

def _sma(series: List[float], period: int) -> Optional[float]:
    if len(series) < period:
        return None
    return round(sum(series[-period:]) / period, 4)


def _ema(series: List[float], period: int) -> Optional[float]:
    """Compute EMA using standard multiplier. Returns last EMA value."""
    if len(series) < period:
        return None
    k = 2.0 / (period + 1)
    ema = sum(series[:period]) / period
    for price in series[period:]:
        ema = price * k + ema * (1 - k)
    return round(ema, 4)


def _ema_series(series: List[float], period: int) -> List[float]:
    """Return full EMA series (same length as input after warm-up)."""
    if len(series) < period:
        return []
    k = 2.0 / (period + 1)
    ema = sum(series[:period]) / period
    result = [ema]
    for price in series[period:]:
        ema = price * k + ema * (1 - k)
        result.append(ema)
    return result


def _rsi(closes: List[float], period: int = 14) -> Optional[float]:
    """Wilder-smoothed RSI."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - 100.0 / (1.0 + rs), 2)


def _macd(closes: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """MACD(12,26,9). Returns (macd_value, signal_value, histogram)."""
    ema12 = _ema_series(closes, 12)
    ema26 = _ema_series(closes, 26)
    if not ema12 or not ema26:
        return None, None, None

    # Align: ema26 is shorter
    offset = len(ema12) - len(ema26)
    macd_line = [ema12[offset + i] - ema26[i] for i in range(len(ema26))]
    if len(macd_line) < 9:
        return None, None, None

    signal_series = _ema_series(macd_line, 9)
    if not signal_series:
        return None, None, None

    macd_val = macd_line[-1]
    signal_val = signal_series[-1]
    histogram = macd_val - signal_val
    return round(macd_val, 4), round(signal_val, 4), round(histogram, 4)


def _bollinger(closes: List[float], period: int = 20, std_mult: float = 2.0
               ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (upper, middle/SMA, lower) Bollinger Bands."""
    if len(closes) < period:
        return None, None, None
    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    std = math.sqrt(variance)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return round(upper, 4), round(mid, 4), round(lower, 4)


def _atr(highs: List[float], lows: List[float], closes: List[float],
         period: int = 14) -> Optional[float]:
    """Average True Range (Wilder smoothing)."""
    if len(closes) < period + 1:
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
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return round(atr, 4)


def _historical_volatility(closes: List[float], period: int = 30) -> Optional[float]:
    """Annualised historical volatility from log returns over `period` days."""
    if len(closes) < period + 1:
        return None
    window = closes[-(period + 1):]
    log_returns = [math.log(window[i] / window[i - 1]) for i in range(1, len(window))]
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
    daily_vol = math.sqrt(variance)
    annual_vol = daily_vol * math.sqrt(252)
    return round(annual_vol, 4)


def _stochastic(
    highs: List[float], lows: List[float], closes: List[float],
    k_period: int = 14, d_period: int = 3,
) -> Tuple[Optional[float], Optional[float]]:
    """Stochastic Oscillator %K and %D."""
    if len(closes) < k_period:
        return None, None
    k_values = []
    for i in range(k_period - 1, len(closes)):
        h = max(highs[i - k_period + 1: i + 1])
        l = min(lows[i - k_period + 1: i + 1])
        if h == l:
            k_values.append(50.0)
        else:
            k_values.append((closes[i] - l) / (h - l) * 100)
    if not k_values:
        return None, None
    k = round(k_values[-1], 2)
    if len(k_values) >= d_period:
        d = round(sum(k_values[-d_period:]) / d_period, 2)
    else:
        d = k
    return k, d


# ---------------------------------------------------------------------------
# Technical Engine
# ---------------------------------------------------------------------------

class TechnicalEngine:
    """Computes all technical indicators for a given FMDataBundle."""

    def compute(self, bundle: FMDataBundle) -> TechnicalSnapshot:
        snap = TechnicalSnapshot()

        if not bundle.price_history:
            logger.warning("TechnicalEngine: no price history for %s", bundle.ticker)
            return snap

        opens, highs, lows, closes = _extract_ohlcv(bundle.price_history)
        if len(closes) < 20:
            logger.warning(
                "TechnicalEngine: only %d price points for %s — many indicators skipped",
                len(closes), bundle.ticker,
            )
            return snap

        current_price = closes[-1]

        # ── SMAs ──────────────────────────────────────────────────────────────
        snap.sma_20 = _sma(closes, 20)
        snap.sma_50 = _sma(closes, 50)
        snap.sma_200 = _sma(closes, 200)

        # ── EMAs ──────────────────────────────────────────────────────────────
        snap.ema_12 = _ema(closes, 12)
        snap.ema_26 = _ema(closes, 26)

        # ── Golden / Death Cross ──────────────────────────────────────────────
        if snap.sma_50 is not None and snap.sma_200 is not None:
            snap.sma_50_above_200 = snap.sma_50 > snap.sma_200
            # Detect cross: compare current vs. previous period
            if len(closes) >= 201:
                prev_sma50 = _sma(closes[:-1], 50)
                prev_sma200 = _sma(closes[:-1], 200)
                if prev_sma50 is not None and prev_sma200 is not None:
                    was_below = prev_sma50 <= prev_sma200
                    now_above = snap.sma_50 > snap.sma_200
                    snap.golden_cross = was_below and now_above
                    snap.death_cross = (not was_below) and (not now_above)
                else:
                    snap.golden_cross = False
                    snap.death_cross = False
            else:
                snap.golden_cross = False
                snap.death_cross = False

        # ── RSI ───────────────────────────────────────────────────────────────
        snap.rsi_14 = _rsi(closes, 14)

        # ── MACD ──────────────────────────────────────────────────────────────
        macd_val, signal_val, histogram = _macd(closes)
        snap.macd_histogram = histogram
        if macd_val is not None and signal_val is not None:
            if histogram is not None and histogram > 0:
                snap.macd_signal = "buy"
            elif histogram is not None and histogram < 0:
                snap.macd_signal = "sell"
            else:
                snap.macd_signal = "neutral"
        else:
            snap.macd_signal = None

        # ── Bollinger Bands ───────────────────────────────────────────────────
        upper, mid_bb, lower = _bollinger(closes, 20, 2.0)
        snap.bollinger_upper = upper
        snap.bollinger_lower = lower
        if upper is not None and lower is not None and mid_bb is not None:
            if current_price > upper:
                snap.bollinger_position = "above_upper"
            elif current_price >= mid_bb:
                snap.bollinger_position = "upper"
            elif current_price >= lower:
                snap.bollinger_position = "lower"
            else:
                snap.bollinger_position = "below_lower"
            # Refine: "mid" if within 10% of mid
            band_width = upper - lower
            if band_width > 0 and abs(current_price - mid_bb) / band_width < 0.1:
                snap.bollinger_position = "mid"

        # ── ATR ───────────────────────────────────────────────────────────────
        snap.atr_14 = _atr(highs, lows, closes, 14)

        # ── Historical Volatility (30-day annualised) ─────────────────────────
        snap.hv_30 = _historical_volatility(closes, 30)

        # ── Stochastic ────────────────────────────────────────────────────────
        snap.stochastic_k, snap.stochastic_d = _stochastic(highs, lows, closes)

        # ── 52-week high / low ────────────────────────────────────────────────
        year_closes = closes[-252:] if len(closes) >= 252 else closes
        year_highs = highs[-252:] if len(highs) >= 252 else highs
        year_lows = lows[-252:] if len(lows) >= 252 else lows
        snap.high_52w = round(max(year_highs), 4) if year_highs else None
        snap.low_52w = round(min(year_lows), 4) if year_lows else None

        # ── Support / Resistance (20-day rolling) ────────────────────────────
        recent_n = 20
        if len(lows) >= recent_n:
            snap.support = round(min(lows[-recent_n:]), 4)
        if len(highs) >= recent_n:
            snap.resistance = round(max(highs[-recent_n:]), 4)

        # ── Trend label ───────────────────────────────────────────────────────
        snap.trend = self._classify_trend(snap, current_price)

        return snap

    @staticmethod
    def _classify_trend(snap: TechnicalSnapshot, price: float) -> str:
        """Classify overall trend from SMA position and RSI."""
        bullish_signals = 0
        bearish_signals = 0

        # Price above/below SMAs
        if snap.sma_50 is not None:
            if price > snap.sma_50:
                bullish_signals += 1
            else:
                bearish_signals += 1
        if snap.sma_200 is not None:
            if price > snap.sma_200:
                bullish_signals += 1
            else:
                bearish_signals += 1
        if snap.sma_50_above_200 is True:
            bullish_signals += 1
        elif snap.sma_50_above_200 is False:
            bearish_signals += 1

        # RSI
        if snap.rsi_14 is not None:
            if snap.rsi_14 > 55:
                bullish_signals += 1
            elif snap.rsi_14 < 45:
                bearish_signals += 1

        # MACD
        if snap.macd_signal == "buy":
            bullish_signals += 1
        elif snap.macd_signal == "sell":
            bearish_signals += 1

        if bullish_signals > bearish_signals + 1:
            return "BULLISH"
        elif bearish_signals > bullish_signals + 1:
            return "BEARISH"
        return "NEUTRAL"


__all__ = ["TechnicalEngine"]
