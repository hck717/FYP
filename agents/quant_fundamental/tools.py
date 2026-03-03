"""PostgreSQL connectors and toolkit facade for the Quantitative Fundamental agent.

Key responsibilities:
  - PostgresConnector: fetches raw_fundamentals + raw_timeseries + market_eod_us
  - FinancialDataFetcher: assembles FinancialsBundle from PostgreSQL
  - AnomalyDetector: Z-score anomaly detection over rolling window
  - QuantFundamentalToolkit: façade used by agent nodes and health checks
"""

from __future__ import annotations

import json
import logging
import math
from contextlib import closing
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from .config import QuantFundamentalConfig
from .schema import DataQualityCheck, FinancialsBundle, QualityStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe numeric helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert val to float, returning default on failure."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_div(numerator: Any, denominator: Any) -> Optional[float]:
    """Safe division — returns None if denominator is zero/None."""
    n = _safe_float(numerator)
    d = _safe_float(denominator)
    if n is None or d is None or d == 0:
        return None
    return n / d


# ---------------------------------------------------------------------------
# PostgreSQL connector
# ---------------------------------------------------------------------------

class PostgresConnector:
    """Thin wrapper over psycopg2 for the quantitative data schema."""

    def __init__(self, config: QuantFundamentalConfig) -> None:
        self.config = config

    def _connect(self):
        return psycopg2.connect(
            host=self.config.postgres_host,
            port=self.config.postgres_port,
            dbname=self.config.postgres_db,
            user=self.config.postgres_user,
            password=self.config.postgres_password,
        )

    def fetch_latest_fundamental(
        self,
        ticker: str,
        data_name: str,
        limit: int = 1,
    ) -> List[Dict[str, Any]]:
        """Fetch the latest `limit` rows of raw_fundamentals for a ticker + data_name.

        Returns a list of payload dicts, newest first.
        """
        sql = """
        SELECT payload, as_of_date, source
        FROM raw_fundamentals
        WHERE ticker_symbol = %s
          AND data_name = %s
        ORDER BY as_of_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, data_name, limit))
            rows = cur.fetchall()
        results = []
        for row in rows:
            payload = row["payload"]
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    pass
            results.append({
                "payload": payload,
                "as_of_date": str(row["as_of_date"]),
                "source": row["source"],
            })
        return results

    def fetch_timeseries(
        self,
        ticker: str,
        data_name: str,
        limit: int = 400,
    ) -> List[Dict[str, Any]]:
        """Fetch time-series rows for a ticker from raw_timeseries, newest first."""
        sql = """
        SELECT payload, ts_date, source
        FROM raw_timeseries
        WHERE ticker_symbol = %s
          AND data_name = %s
        ORDER BY ts_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, data_name, limit))
            rows = cur.fetchall()
        results = []
        for row in rows:
            payload = row["payload"]
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    pass
            results.append({
                "payload": payload,
                "ts_date": str(row["ts_date"]),
                "source": row["source"],
            })
        return results

    def fetch_market_eod(self, limit: int = 400) -> List[Dict[str, Any]]:
        """Fetch EOD benchmark prices from market_eod_us, newest first."""
        sql = """
        SELECT payload, ts_date
        FROM market_eod_us
        ORDER BY ts_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
        results = []
        for row in rows:
            payload = row["payload"]
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    pass
            results.append({"payload": payload, "ts_date": str(row["ts_date"])})
        return results

    def healthcheck(self) -> bool:
        try:
            conn = self._connect()
            conn.close()
            return True
        except Exception as exc:
            logger.warning("PostgreSQL healthcheck failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Data fetcher — assembles FinancialsBundle from PostgreSQL
# ---------------------------------------------------------------------------

class FinancialDataFetcher:
    """Assembles a FinancialsBundle by querying PostgreSQL."""

    def __init__(self, pg: PostgresConnector) -> None:
        self.pg = pg

    def _latest_payload(self, ticker: str, data_name: str) -> Dict[str, Any]:
        """Return the most recent payload dict for a given data_name, or {}."""
        rows = self.pg.fetch_latest_fundamental(ticker, data_name, limit=1)
        if not rows:
            return {}
        payload = rows[0]["payload"]
        # Payloads may be a list (e.g. income_statement returns a list of periods)
        if isinstance(payload, list):
            return payload[0] if payload else {}
        if isinstance(payload, dict):
            return payload
        return {}

    def _latest_payload_list(self, ticker: str, data_name: str, limit: int = 40) -> List[Dict[str, Any]]:
        """Return a list of recent payloads for multi-period data."""
        rows = self.pg.fetch_latest_fundamental(ticker, data_name, limit=limit)
        results: List[Dict[str, Any]] = []
        for row in rows:
            payload = row["payload"]
            if isinstance(payload, list):
                results.extend(payload)
            elif isinstance(payload, dict):
                results.append(payload)
        return results

    def fetch(self, ticker: str) -> FinancialsBundle:
        """Fetch all data needed for factor computation for a single ticker."""
        bundle = FinancialsBundle(ticker=ticker)

        # Income statement — most recent period
        try:
            rows = self._latest_payload_list(ticker, "income_statement", limit=5)
            bundle.income = rows[0] if rows else {}
        except Exception as exc:
            logger.warning("Failed to fetch income_statement for %s: %s", ticker, exc)

        # Balance sheet
        try:
            rows = self._latest_payload_list(ticker, "balance_sheet", limit=5)
            bundle.balance = rows[0] if rows else {}
        except Exception as exc:
            logger.warning("Failed to fetch balance_sheet for %s: %s", ticker, exc)

        # Cash flow
        try:
            rows = self._latest_payload_list(ticker, "cash_flow", limit=5)
            bundle.cashflow = rows[0] if rows else {}
        except Exception as exc:
            logger.warning("Failed to fetch cash_flow for %s: %s", ticker, exc)

        # Financial ratios
        try:
            rows = self._latest_payload_list(ticker, "financial_ratios", limit=5)
            bundle.ratios = rows[0] if rows else {}
        except Exception as exc:
            logger.warning("Failed to fetch financial_ratios for %s: %s", ticker, exc)

        # TTM ratios
        try:
            bundle.ratios_ttm = self._latest_payload(ticker, "ratios_ttm")
        except Exception as exc:
            logger.warning("Failed to fetch ratios_ttm for %s: %s", ticker, exc)

        # Key metrics
        try:
            rows = self._latest_payload_list(ticker, "key_metrics", limit=5)
            bundle.key_metrics = rows[0] if rows else {}
        except Exception as exc:
            logger.warning("Failed to fetch key_metrics for %s: %s", ticker, exc)

        # TTM key metrics
        try:
            bundle.key_metrics_ttm = self._latest_payload(ticker, "key_metrics_ttm")
        except Exception as exc:
            logger.warning("Failed to fetch key_metrics_ttm for %s: %s", ticker, exc)

        # Enterprise values
        try:
            rows = self._latest_payload_list(ticker, "enterprise_values", limit=5)
            bundle.enterprise = rows[0] if rows else {}
        except Exception as exc:
            logger.warning("Failed to fetch enterprise_values for %s: %s", ticker, exc)

        # Financial scores (Piotroski, Beneish from FMP)
        try:
            bundle.scores = self._latest_payload(ticker, "financial_scores")
        except Exception as exc:
            logger.warning("Failed to fetch financial_scores for %s: %s", ticker, exc)

        # Historical price data
        # Preference order: daily EOD (up to 400 rows) → weekly (54 rows, ~1 year) →
        # legacy name variants.  Weekly is used as a fallback when daily has too few
        # rows (e.g. only 23 rows of recent EOD data in the DB) so that
        # compute_momentum_risk has enough data points for Beta/Sharpe/Return.
        try:
            ts_rows = self.pg.fetch_timeseries(ticker, "historical_prices_eod", limit=400)
            if not ts_rows:
                ts_rows = self.pg.fetch_timeseries(ticker, "historical_prices_weekly", limit=400)
            if not ts_rows:
                ts_rows = self.pg.fetch_timeseries(ticker, "historical_price", limit=400)
            if not ts_rows:
                ts_rows = self.pg.fetch_timeseries(ticker, "eod_price", limit=400)
            # If daily data is sparse (< 60 rows), supplement with weekly data so that
            # momentum / beta calculations have enough price points.
            if len(ts_rows) < 60:
                weekly_rows = self.pg.fetch_timeseries(ticker, "historical_prices_weekly", limit=400)
                if weekly_rows:
                    # Merge: use a date-keyed dict so daily rows take precedence over weekly
                    existing_dates = {
                        str(row.get("ts_date", "")) for row in ts_rows
                    }
                    for row in weekly_rows:
                        if str(row.get("ts_date", "")) not in existing_dates:
                            ts_rows.append(row)
                    logger.info(
                        "Price history for %s supplemented with weekly data — total %d rows",
                        ticker, len(ts_rows),
                    )
            # Merge ts_date into payload so momentum_risk can find a date field
            for row in ts_rows:
                payload = row.get("payload")
                if not payload:
                    continue
                merged = dict(payload) if isinstance(payload, dict) else {}
                if "date" not in merged and "ts_date" not in merged:
                    merged["date"] = str(row.get("ts_date", ""))
                bundle.price_history.append(merged)
        except Exception as exc:
            logger.warning("Failed to fetch price_history for %s: %s", ticker, exc)

        # Benchmark (market EOD) prices
        # market_eod_us may be empty or contain only junk rows.  When it has
        # fewer than 10 usable rows we fall back to a synthetic equal-weighted
        # index built from the weekly price history of all supported tickers
        # EXCEPT the one being analysed.  This is not a true market index but
        # gives compute_beta_60d enough aligned points to produce a valid beta.
        try:
            mkt_rows = self.pg.fetch_market_eod(limit=400)
            for row in mkt_rows:
                payload = row.get("payload")
                if not payload:
                    continue
                merged = dict(payload) if isinstance(payload, dict) else {}
                if "date" not in merged and "ts_date" not in merged:
                    merged["date"] = str(row.get("ts_date", ""))
                bundle.benchmark_history.append(merged)
        except Exception as exc:
            logger.warning("Failed to fetch benchmark_history: %s", exc)

        # Synthetic benchmark fallback when market_eod_us is unusable
        if len(bundle.benchmark_history) < 10:
            logger.warning(
                "market_eod_us has only %d usable rows — building synthetic benchmark "
                "from peer weekly prices for beta calculation.",
                len(bundle.benchmark_history),
            )
            try:
                _PEER_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
                peers = [t for t in _PEER_TICKERS if t != ticker]
                # Collect weekly closes per date across all peers
                date_sums: Dict[str, List[float]] = {}
                for peer in peers:
                    peer_rows = self.pg.fetch_timeseries(peer, "historical_prices_weekly", limit=400)
                    for row in peer_rows:
                        payload = row.get("payload")
                        if not payload:
                            continue
                        p = dict(payload) if isinstance(payload, dict) else {}
                        close = (
                            p.get("adjusted_close") or p.get("adjClose")
                            or p.get("close") or p.get("adj_close")
                        )
                        date_str = p.get("date") or str(row.get("ts_date", ""))
                        if close and date_str:
                            date_key = str(date_str).split("T")[0].split(" ")[0]
                            date_sums.setdefault(date_key, []).append(float(close))
                # Build equal-weighted synthetic index rows
                bundle.benchmark_history = []
                for date_key, closes in sorted(date_sums.items()):
                    avg_close = sum(closes) / len(closes)
                    bundle.benchmark_history.append({
                        "date": date_key,
                        "close": avg_close,
                        "adjusted_close": avg_close,
                    })
                logger.info(
                    "Synthetic benchmark built: %d weekly dates from %d peers",
                    len(bundle.benchmark_history), len(peers),
                )
            except Exception as exc:
                logger.warning("Failed to build synthetic benchmark: %s", exc)

        return bundle


# ---------------------------------------------------------------------------
# Data quality checker — PostgreSQL-based field-presence + range validation
# ---------------------------------------------------------------------------

class DataQualityChecker:
    """Validate that critical TTM fields were read from PostgreSQL and are internally consistent.

    This is a single-path PostgreSQL quality check. It verifies:
    1. Required TTM fields are present and non-null.
    2. Values fall within plausible economic ranges (e.g. 0 < gross_margin < 1).
    3. Internal consistency (e.g. pe_trailing > 0 when set).

    No second computation engine is used — this is NOT a dual-path cross-check.
    """

    # Fields to check: (source_dict_attr, key, min_val, max_val, description)
    _RANGE_CHECKS = [
        ("ratios_ttm", "grossProfitMarginTTM", 0.0, 1.0, "gross margin fraction"),
        ("ratios_ttm", "operatingProfitMarginTTM", -1.0, 1.0, "operating margin fraction"),
        ("ratios_ttm", "currentRatioTTM", 0.0, 50.0, "current ratio"),
        ("ratios_ttm", "priceToEarningsRatioTTM", 0.0, 2000.0, "P/E ratio"),
        ("key_metrics_ttm", "returnOnEquityTTM", -5.0, 50.0, "ROE"),
        ("key_metrics_ttm", "returnOnInvestedCapitalTTM", -2.0, 20.0, "ROIC"),
        ("key_metrics_ttm", "evToEBITDATTM", 0.0, 500.0, "EV/EBITDA"),
    ]

    def check(self, bundle: FinancialsBundle) -> DataQualityCheck:
        """Run presence and range checks on the bundle.

        Returns a DataQualityCheck summary.
        """
        issues: List[str] = []
        checks_passed = 0
        checks_total = 0

        # Presence check: at least one of the primary TTM sources must be populated
        ttm_sources = {
            "ratios_ttm": bundle.ratios_ttm,
            "key_metrics_ttm": bundle.key_metrics_ttm,
            "financial_scores": bundle.scores,
        }
        populated = [k for k, v in ttm_sources.items() if v]
        if not populated:
            return DataQualityCheck(
                status=QualityStatus.SKIPPED,
                checks_passed=0,
                checks_total=0,
                issues=["No TTM data available — all quality checks skipped"],
            )

        # Range checks on available TTM fields
        src_map = {
            "ratios_ttm": bundle.ratios_ttm,
            "key_metrics_ttm": bundle.key_metrics_ttm,
        }
        for src_attr, key, lo, hi, desc in self._RANGE_CHECKS:
            src = src_map.get(src_attr, {})
            if not src:
                continue  # source not populated — skip this check
            raw = src.get(key)
            if raw is None:
                continue  # field not present — not a failure, just skip
            checks_total += 1
            try:
                val = float(raw)
                if lo <= val <= hi:
                    checks_passed += 1
                else:
                    issues.append(
                        f"{key}={val:.4f} out of expected range [{lo}, {hi}] ({desc})"
                    )
            except (TypeError, ValueError):
                issues.append(f"{key} is not numeric: {raw!r}")

        # Piotroski score range check (0–9)
        if bundle.scores:
            raw_p = bundle.scores.get("piotroskiScore")
            if raw_p is not None:
                checks_total += 1
                try:
                    p_val = int(float(raw_p))
                    if 0 <= p_val <= 9:
                        checks_passed += 1
                    else:
                        issues.append(f"piotroskiScore={p_val} out of range [0, 9]")
                except (TypeError, ValueError):
                    issues.append(f"piotroskiScore is not numeric: {raw_p!r}")

        # Price history presence check
        checks_total += 1
        if bundle.price_history:
            checks_passed += 1
        else:
            issues.append("price_history is empty — momentum factors will be null")

        if checks_total == 0:
            status = QualityStatus.SKIPPED
        elif issues:
            status = QualityStatus.ISSUES_FOUND
        else:
            status = QualityStatus.PASSED

        logger.debug(
            "DataQualityChecker: %s status=%s, passed=%d/%d, issues=%d",
            bundle.ticker, status.value, checks_passed, checks_total, len(issues),
        )
        return DataQualityCheck(
            status=status,
            checks_passed=checks_passed,
            checks_total=checks_total,
            issues=issues,
        )


# ---------------------------------------------------------------------------
# Anomaly detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Z-score based anomaly detection over a 3-year rolling window."""

    def __init__(self, config: QuantFundamentalConfig) -> None:
        self.config = config

    def compute_z_score(
        self,
        current: float,
        history: List[float],
    ) -> Optional[float]:
        """Compute the Z-score of current vs. history list.

        Returns None if history has fewer than 4 data points.
        """
        if len(history) < 4:
            return None
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std = math.sqrt(variance)
        if std == 0:
            return None
        return (current - mean) / std

    def flag_metric(
        self,
        metric_name: str,
        current_value: Optional[float],
        history: List[float],
        interpretation: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Return an anomaly flag dict if current_value is outside the Z-score threshold."""
        if current_value is None or not history:
            return None
        z = self.compute_z_score(current_value, history)
        if z is None:
            return None
        if abs(z) >= self.config.anomaly_zscore_threshold:
            mean = sum(history) / len(history)
            return {
                "metric": metric_name,
                "z_score": round(z, 2),
                "current_value": round(current_value, 4),
                "3y_mean": round(mean, 4),
                "direction": "above" if z > 0 else "below",
                "interpretation": interpretation or (
                    f"{metric_name} is {'elevated' if z > 0 else 'depressed'} "
                    f"relative to its 3-year baseline (Z={z:.2f})"
                ),
            }
        return None


# ---------------------------------------------------------------------------
# Toolkit façade
# ---------------------------------------------------------------------------

class QuantFundamentalToolkit:
    """Façade combining PostgreSQL fetcher, data quality checker, and anomaly detector."""

    def __init__(self, config: Optional[QuantFundamentalConfig] = None) -> None:
        self.config = config or QuantFundamentalConfig()
        self.pg = PostgresConnector(self.config)
        self.fetcher = FinancialDataFetcher(self.pg)
        self.quality_checker = DataQualityChecker()
        self.anomaly_detector = AnomalyDetector(self.config)

    def fetch_financials(self, ticker: str) -> FinancialsBundle:
        return self.fetcher.fetch(ticker)

    def healthcheck(self) -> Dict[str, bool]:
        return {"postgres": self.pg.healthcheck()}

    def close(self) -> None:
        pass  # psycopg2 connections are opened/closed per-query


__all__ = [
    "QuantFundamentalToolkit",
    "PostgresConnector",
    "DataQualityChecker",
    "FinancialDataFetcher",
    "AnomalyDetector",
    "_safe_float",
    "_safe_div",
]
