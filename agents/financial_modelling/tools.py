"""PostgreSQL + Neo4j connectors and toolkit facade for the Financial Modelling agent.

Key responsibilities:
  - PostgresConnector: fetches raw_fundamentals, raw_timeseries, market_eod_us
  - Neo4jPeerSelector: fetches peer tickers via COMPETES_WITH / BELONGS_TO edges
  - FMDataFetcher: assembles FMDataBundle from all data sources
  - FMToolkit: façade used by agent nodes
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import closing
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from .config import FinancialModellingConfig
from .schema import FMDataBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Boundary Enforcement - Financial Modelling Agent Whitelist
# ---------------------------------------------------------------------------

ALLOWED_DATA_TYPES: List[str] = [
    "key_metrics_ttm",
    "ratios_ttm",
    "dividends",
    "dividends_history",
    "historical_dividends",
    "stock_splits",
    "splits_history",
    "treasury_rates",
    "macro_indicators",
    "global_macro_indicators",
    "economic_events",
    "bonds_yields",
    "corporate_bond_yields",
    "forex_eod",
    "forex_rates",
    "forex_historical_rates",
    "income_statement",
    "balance_sheet",
    "cash_flow",
    "income_statement_as_reported",
    "balance_sheet_as_reported",
    "cash_flow_as_reported",
    "financial_statements",
    "valuation_multiples",
    "enterprise_values",
    "valuation_metrics",
    "outstanding_shares",
    "outstanding_shares_history",
    "historical_prices_eod",
    "historical_prices_weekly",
    "company_core_info",
    "analyst_estimates",
    "analyst_estimates_eodhd",
    "financial_scores",
    "fundamentals",
    "revenue_product_segmentation",
    "revenue_geographic_segmentation",
]


def validate_data_name(data_name: str) -> None:
    """Validate that a data_name is allowed for Financial Modelling Agent.

    Args:
        data_name: The data name to validate.

    Raises:
        ValueError: If the data type is not in the allowed whitelist.
    """
    if not data_name:
        return
    if not any(
        data_name == a or data_name.startswith(a + "_") or data_name.startswith(a)
        for a in ALLOWED_DATA_TYPES
    ):
        raise ValueError(
            f"Data type not allowed for Financial Modelling Agent: {data_name}. "
            f"Allowed types: {ALLOWED_DATA_TYPES}"
        )


# ---------------------------------------------------------------------------
# Safe numeric helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_div(numerator: Any, denominator: Any) -> Optional[float]:
    n = _safe_float(numerator)
    d = _safe_float(denominator)
    if n is None or d is None or d == 0:
        return None
    return n / d


# ---------------------------------------------------------------------------
# PostgreSQL connector
# ---------------------------------------------------------------------------

class PostgresConnector:
    """Thin psycopg2 wrapper for the financial modelling data schema."""

    def __init__(self, config: FinancialModellingConfig) -> None:
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
        """Fetch latest `limit` rows from raw_fundamentals for ticker + data_name."""
        validate_data_name(data_name)
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

    def fetch_annual_fundamental(
        self,
        ticker: str,
        data_name: str,
        limit: int = 2,
    ) -> List[Dict[str, Any]]:
        """Fetch up to `limit` annual (period_type='yearly') rows, newest first.

        Used to obtain current + prior-year annual data for YoY calculations
        (e.g. Piotroski F-Score).  Falls back to all rows if no yearly rows exist.
        """
        validate_data_name(data_name)
        sql = """
        SELECT payload, as_of_date, source
        FROM raw_fundamentals
        WHERE ticker_symbol = %s
          AND data_name = %s
          AND (payload->>'period_type' = 'yearly' OR payload->>'periodType' = 'annual')
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
        """Fetch time-series rows from raw_timeseries, newest first."""
        validate_data_name(data_name)
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
        """Fetch EOD prices from market_eod_us (S&P universe), newest first."""
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

    def fetch_peer_fundamentals_by_sector(
        self,
        sector: str,
        exclude_ticker: str,
        limit: int = 5,
    ) -> List[str]:
        """Fallback: top-N tickers by market cap in the same GICS sector.

        Queries the 'key_metrics_ttm' data_name (FMP) joined with company_core_info
        for sector classification.  Falls back to income_statement sector field.
        """
        sql = """
        SELECT DISTINCT f.ticker_symbol
        FROM raw_fundamentals f
        JOIN raw_fundamentals km
          ON km.ticker_symbol = f.ticker_symbol
         AND km.data_name = 'key_metrics_ttm'
        WHERE f.data_name = 'company_core_info'
          AND f.ticker_symbol != %s
          AND (
            f.payload->>'sector' = %s
            OR f.payload->>'Sector' = %s
          )
        ORDER BY (km.payload->>'marketCapTTM')::numeric DESC NULLS LAST
        LIMIT %s
        """
        try:
            conn = self._connect()
            with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (exclude_ticker, sector, sector, limit))
                rows = cur.fetchall()
            return [r["ticker_symbol"] for r in rows]
        except Exception as exc:
            logger.warning("fetch_peer_fundamentals_by_sector failed: %s", exc)
            return []

    def fetch_buyback_history(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Fetch share buyback history for the ticker."""
        sql = """
        SELECT payload, as_of_date
        FROM raw_fundamentals
        WHERE ticker_symbol = %s
          AND data_name = 'share_buyback_history'
        ORDER BY as_of_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
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
                "as_of_date": str(row["as_of_date"])
            })
        return results

    def fetch_exec_compensation(self, ticker: str, limit: int = 5) -> List[Dict]:
        """Fetch executive compensation data for the ticker."""
        sql = """
        SELECT payload, as_of_date
        FROM raw_fundamentals
        WHERE ticker_symbol = %s
          AND data_name = 'executive_compensation'
        ORDER BY as_of_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
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
                "as_of_date": str(row["as_of_date"])
            })
        return results

    def fetch_forex_rates(self, ticker: str, limit: int = 365) -> List[Dict]:
        """Fetch forex historical rates for the ticker."""
        sql = """
        SELECT payload, ts_date
        FROM raw_timeseries
        WHERE ticker_symbol = %s
          AND data_name = 'forex_historical_rates'
        ORDER BY ts_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
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
                "ts_date": str(row["ts_date"])
            })
        return results

    def fetch_sector_multiples(self, ticker: str, limit: int = 1) -> List[Dict]:
        """Fetch sector/industry multiples for the ticker."""
        sql = """
        SELECT payload, as_of_date
        FROM raw_fundamentals
        WHERE ticker_symbol = %s
          AND data_name = 'sector_industry_multiples'
        ORDER BY as_of_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
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
                "as_of_date": str(row["as_of_date"])
            })
        return results

    def healthcheck(self) -> bool:
        try:
            conn = self._connect()
            conn.close()
            return True
        except Exception as exc:
            logger.warning("PostgreSQL healthcheck failed: %s", exc)
            return False

    def fetch_economic_events(self, limit: int = 50) -> List[Dict]:
        """Fetch recent economic events (Row 12: Economic Events Data API)."""
        sql = """
        SELECT event_date, country, event_name, actual, forecast, previous, impact, currency
        FROM economic_events
        ORDER BY event_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def fetch_bond_yields(self, limit: int = 10) -> List[Dict]:
        """Fetch recent corporate bond yield rows (Row 13: Bonds Data)."""
        sql = """
        SELECT isin, ticker, issuer_name, coupon_rate, maturity_date,
               yield_to_maturity, current_price, currency, ts_date
        FROM corporate_bond_yields
        ORDER BY ts_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def fetch_treasury_rates_dedicated(self, indicator: str = "US10Y", limit: int = 5) -> List[Dict]:
        """Fetch rows from the dedicated treasury_rates table (Row 11).

        The dedicated table stores pre-extracted rate rows with columns:
        indicator, rate, ts_date.
        """
        sql = """
        SELECT indicator, rate, ts_date
        FROM treasury_rates
        WHERE indicator = %s
        ORDER BY ts_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (indicator, limit))
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def fetch_financial_statements(
        self,
        ticker: str,
        statement_type: Optional[str] = None,
        limit: int = 4,
    ) -> Dict[str, List[Dict]]:
        """Fetch rows from the financial_statements table (Row 18).

        Returns a dict keyed by statement_type:
          {"Income_Statement": [...], "Balance_Sheet": [...], "Cash_Flow": [...]}

        If statement_type is specified, only that type is fetched.
        """
        if statement_type:
            sql = """
            SELECT ticker, statement_type, period_type, report_date, payload
            FROM financial_statements
            WHERE ticker = %s AND statement_type = %s
            ORDER BY report_date DESC
            LIMIT %s
            """
            conn = self._connect()
            with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (ticker, statement_type, limit))
                rows = cur.fetchall()
            result = {}
            for row in rows:
                st = row["statement_type"]
                payload = row["payload"]
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        pass
                entry = {"report_date": str(row["report_date"]), "period_type": row["period_type"],
                         "payload": payload}
                result.setdefault(st, []).append(entry)
            return result
        else:
            sql = """
            SELECT ticker, statement_type, period_type, report_date, payload
            FROM financial_statements
            WHERE ticker = %s
            ORDER BY statement_type, report_date DESC
            """
            conn = self._connect()
            with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (ticker,))
                rows = cur.fetchall()
            result: Dict[str, List[Dict]] = {}
            counts: Dict[str, int] = {}
            for row in rows:
                st = row["statement_type"]
                if counts.get(st, 0) >= limit:
                    continue
                payload = row["payload"]
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        pass
                entry = {"report_date": str(row["report_date"]), "period_type": row["period_type"],
                         "payload": payload}
                result.setdefault(st, []).append(entry)
                counts[st] = counts.get(st, 0) + 1
            return result

    def fetch_earnings_surprises(self, ticker: str, limit: int = 20) -> List[Dict]:
        """Fetch earnings surprise history from the dedicated earnings_surprises table.

        Returns rows sorted by period_date DESC (most recent first).
        Columns: period_date, eps_actual, eps_estimate, eps_surprise_pct,
                 revenue_actual, revenue_estimate, revenue_surprise_pct, before_after_market.
        """
        sql = """
        SELECT ticker, period_date, eps_actual, eps_estimate, eps_surprise_pct,
               revenue_actual, revenue_estimate, revenue_surprise_pct, before_after_market
        FROM earnings_surprises
        WHERE ticker = %s
        ORDER BY period_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
            rows = cur.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            # Convert Decimal → float, date → str
            for k, v in d.items():
                if hasattr(v, '__float__') and not isinstance(v, (str, bool)):
                    d[k] = float(v)
                elif hasattr(v, 'isoformat'):
                    d[k] = str(v)
            result.append(d)
        return result

    def fetch_dividends_dedicated(self, ticker: str, limit: int = 30) -> List[Dict]:
        """Fetch dividend history from the dedicated dividends_history table.

        Returns rows sorted by pay_date DESC (most recent first).
        Columns: ticker, amount, ex_date, pay_date, record_date.
        Only returns rows with amount > 0.
        """
        sql = """
        SELECT ticker, amount, ex_date, pay_date, record_date
        FROM dividends_history
        WHERE ticker = %s AND amount > 0
        ORDER BY pay_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
            rows = cur.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            for k, v in d.items():
                if hasattr(v, '__float__') and not isinstance(v, (str, bool)):
                    d[k] = float(v)
                elif hasattr(v, 'isoformat'):
                    d[k] = str(v)
            result.append(d)
        return result

    def fetch_valuation_metrics(self, ticker: str) -> Optional[Dict]:
        """Fetch latest valuation metrics for the ticker (Row 19).

        Falls back to company_profile_neo4j.json Highlights + Valuation sections when
        the valuation_metrics table row has all-NULL value columns.
        """
        sql = """
        SELECT *
        FROM valuation_metrics
        WHERE ticker = %s
        ORDER BY as_of_date DESC
        LIMIT 1
        """
        try:
            conn = self._connect()
            with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (ticker,))
                row = cur.fetchone()
                if row:
                    result = {k: (float(v) if isinstance(v, (int, float)) and k not in ("ticker", "id") else
                                  str(v) if v is not None else None)
                              for k, v in dict(row).items()}
                    # Check if any actual value fields are non-NULL
                    value_keys = {k for k in result if k not in ("ticker", "id", "as_of_date", "ingested_at")}
                    if any(result.get(k) is not None for k in value_keys):
                        return result
        except Exception as exc:
            logger.debug("[FM] fetch_valuation_metrics primary query failed for %s: %s", ticker, exc)

        # Fallback: company_profile_neo4j.json Highlights + Valuation sections
        try:
            base_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "ingestion", "etl", "agent_data", ticker.upper()
            )
            neo4j_path = os.path.normpath(os.path.join(base_dir, "company_profile_neo4j.json"))
            if os.path.exists(neo4j_path):
                with open(neo4j_path, "r", encoding="utf-8") as f:
                    neo4j_data = json.load(f)
                h = neo4j_data.get("Highlights", {})
                v = neo4j_data.get("Valuation", {})
                if h or v:
                    logger.info("[FM] fetch_valuation_metrics: using company_profile_neo4j.json fallback for %s", ticker)
                    def _fn(x):
                        try:
                            return float(x) if x is not None else None
                        except (TypeError, ValueError):
                            return None
                    return {
                        "ticker": ticker,
                        "trailing_pe":    _fn(v.get("TrailingPE") or h.get("PERatio")),
                        "forward_pe":     _fn(v.get("ForwardPE")),
                        "price_sales_ttm": _fn(v.get("PriceSalesTTM")),
                        "price_book_mrq": _fn(v.get("PriceBookMRQ")),
                        "enterprise_value": _fn(v.get("EnterpriseValue")),
                        "ev_revenue":     _fn(v.get("EnterpriseValueRevenue")),
                        "ev_ebitda":      _fn(v.get("EnterpriseValueEbitda")),
                        "market_cap":     _fn(h.get("MarketCapitalization")),
                        "ebitda":         _fn(h.get("EBITDA")),
                        "pe_ratio":       _fn(h.get("PERatio")),
                        "peg_ratio":      _fn(h.get("PEGRatio")),
                        "eps":            _fn(h.get("EarningsShare")),
                        "profit_margin":  _fn(h.get("ProfitMargin")),
                        "operating_margin": _fn(h.get("OperatingMarginTTM")),
                        "roa":            _fn(h.get("ReturnOnAssetsTTM")),
                        "roe":            _fn(h.get("ReturnOnEquityTTM")),
                        "revenue_ttm":    _fn(h.get("RevenueTTM")),
                        "book_value":     _fn(h.get("BookValue")),
                        "dividend_share": _fn(h.get("DividendShare")),
                        "dividend_yield": _fn(h.get("DividendYield")),
                        "wall_st_target": _fn(h.get("WallStreetTargetPrice")),
                    }
        except Exception as exc:
            logger.debug("[FM] fetch_valuation_metrics neo4j json fallback failed for %s: %s", ticker, exc)
        return None

    def fetch_analyst_ratings(self, ticker: str) -> Optional[Dict]:
        """Fetch latest analyst ratings for the ticker.

        Falls back to raw_fundamentals data_name='analyst_ratings', then to
        company_profile_neo4j.json AnalystRatings section.
        """
        sql = """
        SELECT *
        FROM analyst_ratings
        WHERE ticker = %s
        ORDER BY as_of_date DESC
        LIMIT 1
        """
        try:
            conn = self._connect()
            with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (ticker,))
                row = cur.fetchone()
                if row:
                    return {k: (float(v) if isinstance(v, (int, float)) and k not in ("ticker", "id") else
                                str(v) if v is not None else None)
                            for k, v in dict(row).items()}
        except Exception as exc:
            logger.debug("[FM] fetch_analyst_ratings primary query failed for %s: %s", ticker, exc)

        # Fallback 1: raw_fundamentals data_name='analyst_ratings'
        try:
            rows = self.fetch_latest_fundamental(ticker, "analyst_ratings", limit=1)
            if rows:
                payload = rows[0].get("payload", {})
                if isinstance(payload, str):
                    payload = json.loads(payload)
                if isinstance(payload, dict) and payload:
                    logger.info("[FM] fetch_analyst_ratings: using raw_fundamentals fallback for %s", ticker)
                    def _fn(x):
                        try:
                            return float(x) if x is not None else None
                        except (TypeError, ValueError):
                            return None
                    return {
                        "ticker": ticker,
                        "rating":           _fn(payload.get("Rating")),
                        "target_price":     _fn(payload.get("TargetPrice")),
                        "strong_buy_count": _fn(payload.get("StrongBuy")),
                        "buy_count":        _fn(payload.get("Buy")),
                        "hold_count":       _fn(payload.get("Hold")),
                        "sell_count":       _fn(payload.get("Sell")),
                        "strong_sell_count": _fn(payload.get("StrongSell")),
                    }
        except Exception as exc:
            logger.debug("[FM] fetch_analyst_ratings raw_fundamentals fallback failed for %s: %s", ticker, exc)

        # Fallback 2: company_profile_neo4j.json AnalystRatings section
        try:
            base_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "ingestion", "etl", "agent_data", ticker.upper()
            )
            neo4j_path = os.path.normpath(os.path.join(base_dir, "company_profile_neo4j.json"))
            if os.path.exists(neo4j_path):
                with open(neo4j_path, "r", encoding="utf-8") as f:
                    neo4j_data = json.load(f)
                ar = neo4j_data.get("AnalystRatings", {})
                if ar:
                    logger.info("[FM] fetch_analyst_ratings: using company_profile_neo4j.json fallback for %s", ticker)
                    def _fn(x):
                        try:
                            return float(x) if x is not None else None
                        except (TypeError, ValueError):
                            return None
                    return {
                        "ticker": ticker,
                        "rating":           _fn(ar.get("Rating")),
                        "target_price":     _fn(ar.get("TargetPrice")),
                        "strong_buy_count": _fn(ar.get("StrongBuy")),
                        "buy_count":        _fn(ar.get("Buy")),
                        "hold_count":       _fn(ar.get("Hold")),
                        "sell_count":       _fn(ar.get("Sell")),
                        "strong_sell_count": _fn(ar.get("StrongSell")),
                    }
        except Exception as exc:
            logger.debug("[FM] fetch_analyst_ratings neo4j json fallback failed for %s: %s", ticker, exc)
        return None

    def fetch_company_profile(self, ticker: str) -> Optional[Dict]:
        """Fetch company profile for the ticker.

        Falls back to company_profile_neo4j.json General section when the
        company_profiles table is empty.
        """
        sql = """
        SELECT *
        FROM company_profiles
        WHERE ticker = %s
        LIMIT 1
        """
        try:
            conn = self._connect()
            with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (ticker,))
                row = cur.fetchone()
                if row:
                    return {k: (float(v) if isinstance(v, (int, float)) and k not in ("ticker", "id") else
                                str(v) if v is not None else None)
                            for k, v in dict(row).items()}
        except Exception as exc:
            logger.debug("[FM] fetch_company_profile primary query failed for %s: %s", ticker, exc)

        # Fallback: company_profile_neo4j.json General section
        try:
            base_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "ingestion", "etl", "agent_data", ticker.upper()
            )
            neo4j_path = os.path.normpath(os.path.join(base_dir, "company_profile_neo4j.json"))
            if os.path.exists(neo4j_path):
                with open(neo4j_path, "r", encoding="utf-8") as f:
                    neo4j_data = json.load(f)
                g = neo4j_data.get("General", {})
                if g:
                    logger.info("[FM] fetch_company_profile: using company_profile_neo4j.json fallback for %s", ticker)
                    return {
                        "ticker":              ticker,
                        "name":                g.get("Name"),
                        "exchange":            g.get("Exchange"),
                        "sector":              g.get("Sector"),
                        "industry":            g.get("Industry"),
                        "gic_sector":          g.get("GicSector"),
                        "description":         g.get("Description"),
                        "address":             g.get("Address"),
                        "city":                g.get("AddressData", {}).get("City") if isinstance(g.get("AddressData"), dict) else None,
                        "state":               g.get("AddressData", {}).get("State") if isinstance(g.get("AddressData"), dict) else None,
                        "country":             g.get("CountryName") or (g.get("AddressData", {}).get("Country") if isinstance(g.get("AddressData"), dict) else None),
                        "phone":               g.get("Phone"),
                        "web_url":             g.get("WebURL"),
                        "full_time_employees": g.get("FullTimeEmployees"),
                        "fiscal_year_end":     g.get("FiscalYearEnd"),
                        "ipo_date":            g.get("IPODate"),
                        "currency":            g.get("CurrencyCode"),
                        "isin":                g.get("ISIN"),
                        "cusip":               g.get("CUSIP"),
                        "cik":                 g.get("CIK"),
                        "is_delisted":         False,
                    }
        except Exception as exc:
            logger.debug("[FM] fetch_company_profile neo4j json fallback failed for %s: %s", ticker, exc)
        return None

    def fetch_outstanding_shares(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Fetch outstanding shares history for the ticker (Row 22)."""
        sql = """
        SELECT ticker, period_type, shares_date, shares_outstanding
        FROM outstanding_shares
        WHERE ticker = %s
        ORDER BY shares_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def fetch_splits_history(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Fetch stock split history for the ticker (Row 10: splits part)."""
        sql = """
        SELECT ticker, split_ratio, announce_date, ex_date
        FROM splits_history
        WHERE ticker = %s
        ORDER BY ex_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def fetch_macro_indicators(self, indicator: str, limit: int = 5) -> List[Dict]:
        """Fetch global macro indicator rows (Row 11: Treasury Rates / Macro Indicators).

        Queries the global_macro_indicators table.
        Common indicator values: 'GDP', 'CPI', 'UNEMPLOYMENT', 'US10Y', etc.
        """
        sql = """
        SELECT indicator, ts_date, payload, source
        FROM global_macro_indicators
        WHERE indicator = %s
        ORDER BY ts_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (indicator, limit))
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
                "indicator": row["indicator"],
                "ts_date": str(row["ts_date"]),
                "payload": payload,
                "source": row["source"],
            })
        return results

    def fetch_forex_rates_dedicated(self, forex_pair: str, limit: int = 5) -> List[Dict]:
        """Fetch rows from the dedicated forex_rates table (Row 14: Forex Historical Rates).

        The forex_rates table has columns: forex_pair, rate, ts_date.
        """
        sql = """
        SELECT forex_pair, rate, ts_date
        FROM forex_rates
        WHERE forex_pair = %s
        ORDER BY ts_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (forex_pair, limit))
            rows = cur.fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Neo4j peer selector
# ---------------------------------------------------------------------------

class Neo4jPeerSelector:
    """Selects peer tickers via Neo4j COMPETES_WITH / BELONGS_TO graph edges."""

    def __init__(self, config: FinancialModellingConfig) -> None:
        self.config = config
        self._driver = None

    def _get_driver(self):
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
                self._driver = GraphDatabase.driver(
                    self.config.neo4j_uri,
                    auth=(self.config.neo4j_user, self.config.neo4j_password),
                )
            except ImportError:
                logger.warning("neo4j package not available — peer selection will use PG fallback")
        return self._driver

    def get_peers(self, ticker: str, limit: int = 5) -> List[str]:
        """Return up to `limit` peer tickers from Neo4j COMPETES_WITH edges."""
        driver = self._get_driver()
        if driver is None:
            return []
        query = """
        MATCH (c:Company {ticker: $ticker})-[:COMPETES_WITH|BELONGS_TO*1..2]-(peer:Company)
        WHERE peer.ticker <> $ticker
        WITH DISTINCT peer.ticker AS t
        LIMIT $limit
        RETURN t
        """
        try:
            with driver.session() as session:
                result = session.run(query, ticker=ticker, limit=limit)
                return [record["t"] for record in result if record["t"]]
        except Exception as exc:
            logger.warning("Neo4j peer query failed for %s: %s", ticker, exc)
            return []

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None


# ---------------------------------------------------------------------------
# Data fetcher — assembles FMDataBundle from PostgreSQL
# ---------------------------------------------------------------------------

class FMDataFetcher:
    """Assembles a FMDataBundle by querying PostgreSQL and Neo4j."""

    def __init__(
        self,
        pg: PostgresConnector,
        neo4j: Neo4jPeerSelector,
        config: FinancialModellingConfig,
    ) -> None:
        self.pg = pg
        self.neo4j = neo4j
        self.config = config

    def _latest_payload(self, ticker: str, data_name: str) -> Dict[str, Any]:
        rows = self.pg.fetch_latest_fundamental(ticker, data_name, limit=1)
        if not rows:
            return {}
        payload = rows[0]["payload"]
        if isinstance(payload, list):
            return payload[0] if payload else {}
        if isinstance(payload, dict):
            return payload
        return {}

    def _latest_payload_list(
        self, ticker: str, data_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        rows = self.pg.fetch_latest_fundamental(ticker, data_name, limit=limit)
        results: List[Dict[str, Any]] = []
        for row in rows:
            payload = row["payload"]
            if isinstance(payload, list):
                results.extend(payload)
            elif isinstance(payload, dict):
                results.append(payload)
        return results

    def _financial_stmt_payload(
        self,
        ticker: str,
        statement_type: str,
        period_type: str = "yearly",
    ) -> Dict[str, Any]:
        """Fetch the most recent payload from the financial_statements specialty table.

        Used as a last-resort fallback when raw_fundamentals is empty (i.e. FMP and
        EODHD raw_fundamentals data_names are both absent).

        Args:
            statement_type: 'Income_Statement' | 'Balance_Sheet' | 'Cash_Flow'
            period_type: 'yearly' (default) | 'quarterly'
        """
        try:
            stmts = self.pg.fetch_financial_statements(
                ticker, statement_type=statement_type, limit=1
            )
            rows = stmts.get(statement_type, [])
            if rows and rows[0].get("payload"):
                return rows[0]["payload"]
        except Exception as exc:
            logger.debug(
                "[FM] financial_statements fallback failed for %s/%s: %s",
                ticker, statement_type, exc,
            )
        return {}

    def _merge_ts_date(
        self, rows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge ts_date into payload dict so downstream has a reliable date key."""
        merged = []
        for row in rows:
            payload = row.get("payload")
            if not payload:
                continue
            p = dict(payload) if isinstance(payload, dict) else {}
            if "date" not in p and "ts_date" not in p:
                p["date"] = str(row.get("ts_date", ""))
            merged.append(p)
        return merged

    def fetch(self, ticker: str) -> FMDataBundle:
        """Fetch all data needed for the FM pipeline for one ticker.

        Primary source is FMP (raw_fundamentals); when FMP ingestion is paused
        the method falls back to EODHD data already in the database:

          raw_fundamentals — FMP (primary) or EODHD (fallback):
            FMP data_names:
              "income_statement", "balance_sheet", "cash_flow",
              "income_statement_as_reported", "balance_sheet_as_reported",
              "cash_flow_as_reported", "key_metrics_ttm", "ratios_ttm",
              "financial_scores", "analyst_estimates",
              "revenue_product_segmentation" (FMP), "revenue_geographic_segmentation" (FMP),
              "treasury_rates", "stock_peers"
            EODHD fallback data_names (same table):
              "financial_scores"   : revenue, ebit, totalAssets, totalLiabilities,
                                     workingCapital, retainedEarnings, altmanZScore,
                                     piotroskiScore — used to reconstruct income/balance
              "key_metrics_ttm"    : freeCashFlowToFirmTTM, enterpriseValueTTM,
                                     returnOnEquityTTM, returnOnInvestedCapitalTTM, etc.
              "ratios_ttm"         : priceToEarningsRatioTTM, grossProfitMarginTTM,
                                     effectiveTaxRateTTM, dividendYieldTTM, etc.
              "fundamentals"       : Highlights_EBITDA, Highlights_MarketCapitalization,
                                     Valuation_EnterpriseValue (flattened payload)
              "analyst_estimates_eodhd": EODHD analyst estimates

          raw_timeseries — EODHD:
            "historical_prices_weekly"       : ~54 weekly rows (preferred for indicators)
            "historical_prices_eod"          : daily rows
            "dividends_history"              : quarterly dividend rows (value key)
            "revenue_product_segmentation"   : EODHD product revenue by fiscal year
            "revenue_geographic_segmentation": EODHD geographic revenue by fiscal year
        """
        bundle = FMDataBundle(ticker=ticker)

        # ── Income statement (FMP primary → EODHD raw_fundamentals → financial_statements) ──
        try:
            rows = self._latest_payload_list(ticker, "income_statement", limit=5)
            inc = rows[0] if rows else {}
            if not inc:
                # EODHD fallback: financial_scores has revenue, ebit, netIncome etc.
                inc = self._latest_payload(ticker, "financial_scores")
                if inc:
                    logger.info(
                        "[FM] income_statement: FMP empty for %s — using EODHD financial_scores fallback",
                        ticker,
                    )
            if not inc:
                # Final fallback: financial_statements specialty table (EODHD-sourced)
                inc = self._financial_stmt_payload(ticker, "Income_Statement")
                if inc:
                    # EODHD stores revenue as 'totalRevenue'; normalise to 'revenue'
                    if not inc.get("revenue") and inc.get("totalRevenue"):
                        inc["revenue"] = inc["totalRevenue"]
                    logger.info(
                        "[FM] income_statement: using financial_statements table fallback for %s",
                        ticker,
                    )
            # EBITDA overlay from EODHD fundamentals if still missing
            ebitda_val = inc.get("ebitda")
            if ebitda_val is None:
                fund = self._latest_payload(ticker, "fundamentals")
                ebitda_val = fund.get("Highlights_EBITDA")
            bundle.income = {
                # Period metadata (needed for 3SM period labels)
                "date":                 inc.get("date"),
                "period":               inc.get("period"),
                "period_type":          inc.get("period_type"),
                # Revenue — EODHD uses "totalRevenue"; FMP uses "revenue"
                "revenue":              inc.get("revenue") or inc.get("totalRevenue"),
                "totalRevenue":         inc.get("totalRevenue") or inc.get("revenue"),
                # Cost & margins
                "costOfRevenue":        inc.get("costOfRevenue"),
                "grossProfit":          inc.get("grossProfit"),
                # R&D — EODHD uses "researchDevelopment"; FMP uses "researchAndDevelopmentExpenses"
                "researchDevelopment":  inc.get("researchDevelopment") or inc.get("researchAndDevelopmentExpenses"),
                "researchAndDevelopmentExpenses": inc.get("researchAndDevelopmentExpenses") or inc.get("researchDevelopment"),
                # Operating
                "totalOperatingExpenses": inc.get("totalOperatingExpenses") or inc.get("operatingExpenses"),
                "operatingExpenses":    inc.get("operatingExpenses") or inc.get("totalOperatingExpenses"),
                "ebit":                 inc.get("operatingIncome") or inc.get("ebit"),
                "operatingIncome":      inc.get("operatingIncome") or inc.get("ebit"),
                "ebitda":               ebitda_val,
                "EBITDA":               ebitda_val,
                # Below-line
                "interestExpense":      inc.get("interestExpense"),
                "incomeBeforeTax":      inc.get("incomeBeforeTax"),
                # Tax — EODHD uses "taxProvision"; FMP uses "incomeTaxExpense"
                "incomeTaxExpense":     inc.get("incomeTaxExpense") or inc.get("taxProvision"),
                "taxProvision":         inc.get("taxProvision") or inc.get("incomeTaxExpense"),
                # Bottom line
                "netIncome":            inc.get("netIncome"),
                "depreciationAmortization": inc.get("depreciationAndAmortization"),
                "depreciationAndAmortization": inc.get("depreciationAndAmortization"),
            }
        except Exception as exc:
            logger.warning("income_statement fetch failed for %s: %s", ticker, exc)

        # ── Balance sheet (FMP primary → EODHD raw_fundamentals → financial_statements) ──
        try:
            rows = self._latest_payload_list(ticker, "balance_sheet", limit=5)
            bal = rows[0] if rows else {}
            if not bal:
                # EODHD fallback: financial_scores has totalAssets, totalLiabilities,
                # workingCapital, retainedEarnings
                bal = self._latest_payload(ticker, "financial_scores")
                if bal:
                    logger.info(
                        "[FM] balance_sheet: FMP empty for %s — using EODHD financial_scores fallback",
                        ticker,
                    )
            if not bal:
                # Final fallback: financial_statements specialty table (EODHD-sourced)
                bal = self._financial_stmt_payload(ticker, "Balance_Sheet")
                if bal:
                    logger.info(
                        "[FM] balance_sheet: using financial_statements table fallback for %s",
                        ticker,
                    )
            bundle.balance = {
                # Core totals
                "totalAssets":              bal.get("totalAssets"),
                # EODHD uses "totalLiab"; FMP uses "totalLiabilities" — keep both keys
                "totalLiab":                bal.get("totalLiab") or bal.get("totalLiabilities"),
                "totalLiabilities":         bal.get("totalLiabilities") or bal.get("totalLiab"),
                # EODHD uses "totalStockholderEquity" (singular); FMP uses "totalStockholdersEquity"
                "totalStockholderEquity":   bal.get("totalStockholderEquity") or bal.get("totalStockholdersEquity"),
                "totalStockholdersEquity":  bal.get("totalStockholdersEquity") or bal.get("totalStockholderEquity"),
                # Debt
                "totalDebt":                bal.get("totalDebt") or bal.get("longTermDebt"),
                "longTermDebt":             bal.get("longTermDebt") or bal.get("longTermDebtTotal"),
                "longTermDebtTotal":        bal.get("longTermDebtTotal") or bal.get("longTermDebt"),
                "shortTermDebt":            bal.get("shortTermDebt") or bal.get("shortLongTermDebt"),
                "shortLongTermDebt":        bal.get("shortLongTermDebt") or bal.get("shortTermDebt"),
                "shortLongTermDebtTotal":   bal.get("shortLongTermDebtTotal"),
                # Cash — EODHD uses "cash"; FMP uses "cashAndCashEquivalents"
                "cash":                     bal.get("cash") or bal.get("cashAndCashEquivalents") or bal.get("cashAndEquivalents"),
                "cashAndCashEquivalents":   bal.get("cashAndCashEquivalents") or bal.get("cash") or bal.get("cashAndEquivalents"),
                "cashAndEquivalents":       bal.get("cashAndEquivalents") or bal.get("cash"),
                # Current items
                "totalCurrentAssets":       bal.get("totalCurrentAssets"),
                "totalCurrentLiabilities":  bal.get("totalCurrentLiabilities"),
                "netReceivables":           bal.get("netReceivables"),
                "inventory":                bal.get("inventory"),
                "otherCurrentAssets":       bal.get("otherCurrentAssets"),
                "otherCurrentLiab":         bal.get("otherCurrentLiab"),
                "accountsPayable":          bal.get("accountsPayable"),
                # Non-current items
                "goodWill":                 bal.get("goodWill") or bal.get("goodwill"),
                "intangibleAssets":         bal.get("intangibleAssets"),
                "propertyPlantEquipment":   bal.get("propertyPlantEquipment") or bal.get("propertyPlantAndEquipmentNet"),
                # Equity components
                "retainedEarnings":         bal.get("retainedEarnings"),
                "commonStock":              bal.get("commonStock") or bal.get("capitalStock"),
                "capitalStock":             bal.get("capitalStock") or bal.get("commonStock"),
                "treasuryStock":            bal.get("treasuryStock"),
                "additionalPaidInCapital":  bal.get("additionalPaidInCapital"),
                "otherStockholderEquity":   bal.get("otherStockholderEquity"),
                # Derived / convenience
                "workingCapital":           _safe_float(bal.get("totalCurrentAssets"), 0)
                                            - _safe_float(bal.get("totalCurrentLiabilities"), 0)
                                            if (bal.get("totalCurrentAssets") and bal.get("totalCurrentLiabilities"))
                                            else (bal.get("workingCapital") or bal.get("netWorkingCapital")),
                "netWorkingCapital":        bal.get("netWorkingCapital"),
                "netDebt":                  bal.get("netDebt"),
                "netTangibleAssets":        bal.get("netTangibleAssets"),
                # Period metadata
                "date":                     bal.get("date"),
                "period":                   bal.get("period"),
                "period_type":              bal.get("period_type"),
                "marketCapitalization":     None,  # populated from key_metrics_ttm below
            }
        except Exception as exc:
            logger.warning("balance_sheet fetch failed for %s: %s", ticker, exc)

        # ── Cash flow statement (FMP primary → EODHD key_metrics_ttm → financial_statements) ──
        try:
            rows = self._latest_payload_list(ticker, "cash_flow", limit=5)
            cf = rows[0] if rows else {}
            bundle.cashflow = {
                # Operating — EODHD uses "totalCashFromOperatingActivities"; FMP uses "operatingCashFlow"
                "totalCashFromOperatingActivities": cf.get("totalCashFromOperatingActivities") or cf.get("operatingCashFlow"),
                "operatingCashFlow":                cf.get("operatingCashFlow") or cf.get("totalCashFromOperatingActivities"),
                # Investing
                "capitalExpenditures":              cf.get("capitalExpenditures") or cf.get("capitalExpenditure"),
                "capitalExpenditure":               cf.get("capitalExpenditure") or cf.get("capitalExpenditures"),
                "investments":                      cf.get("investments"),
                "totalCashflowsFromInvestingActivities": cf.get("totalCashflowsFromInvestingActivities"),
                "otherCashflowsFromInvestingActivities": cf.get("otherCashflowsFromInvestingActivities"),
                # Financing
                "totalCashFromFinancingActivities": cf.get("totalCashFromFinancingActivities"),
                "dividendsPaid":                    cf.get("dividendsPaid"),
                "salePurchaseOfStock":              cf.get("salePurchaseOfStock"),
                "issuanceOfCapitalStock":           cf.get("issuanceOfCapitalStock"),
                "netBorrowings":                    cf.get("netBorrowings"),
                "otherCashflowsFromFinancingActivities": cf.get("otherCashflowsFromFinancingActivities"),
                # Net change & cash levels
                "changeInCash":                     cf.get("changeInCash") or cf.get("cashAndCashEquivalentsChanges"),
                "cashAndCashEquivalentsChanges":    cf.get("cashAndCashEquivalentsChanges"),
                "beginPeriodCashFlow":              cf.get("beginPeriodCashFlow"),
                "endPeriodCashFlow":                cf.get("endPeriodCashFlow"),
                "exchangeRateChanges":              cf.get("exchangeRateChanges"),
                # Components of OCF (indirect method)
                "netIncome":                        cf.get("netIncome"),
                "depreciation":                     cf.get("depreciation"),
                "depreciationAndAmortization":      cf.get("depreciationAndAmortization") or cf.get("depreciation"),
                "depreciationAmortization":         cf.get("depreciationAndAmortization") or cf.get("depreciation"),
                "stockBasedCompensation":           cf.get("stockBasedCompensation"),
                "changeInWorkingCapital":           cf.get("changeInWorkingCapital"),
                "changeToAccountReceivables":       cf.get("changeToAccountReceivables") or cf.get("changeReceivables"),
                "changeReceivables":                cf.get("changeReceivables") or cf.get("changeToAccountReceivables"),
                "changeToInventory":                cf.get("changeToInventory") or cf.get("changeInventory"),
                "changeInventory":                  cf.get("changeInventory") or cf.get("changeToInventory"),
                "changeToLiabilities":              cf.get("changeToLiabilities"),
                "changeToNetincome":                cf.get("changeToNetincome"),
                "changeToOperatingActivities":      cf.get("changeToOperatingActivities"),
                "otherNonCashItems":                cf.get("otherNonCashItems"),
                "cashFlowsOtherOperating":          cf.get("cashFlowsOtherOperating"),
                # Pre-computed FCF
                "freeCashFlow":                     cf.get("freeCashFlow"),
                # Period metadata
                "date":                             cf.get("date"),
                "period":                           cf.get("period"),
                "period_type":                      cf.get("period_type"),
            }
            if not cf:
                # EODHD fallback: key_metrics_ttm has freeCashFlowToFirmTTM,
                # freeCashFlowToEquityTTM, capexToRevenueTTM etc.
                km = self._latest_payload(ticker, "key_metrics_ttm")
                if km:
                    logger.info(
                        "[FM] cash_flow: FMP empty for %s — using EODHD key_metrics_ttm fallback",
                        ticker,
                    )
                    fcff = _safe_float(km.get("freeCashFlowToFirmTTM"))
                    fcfe = _safe_float(km.get("freeCashFlowToEquityTTM"))
                    revenue = _safe_float(bundle.income.get("revenue")) if bundle.income else None
                    capex_ratio = _safe_float(km.get("capexToRevenueTTM"))
                    capex_abs = -(capex_ratio * revenue) if (capex_ratio and revenue) else None
                    ocf = _safe_float(km.get("freeCashFlowToFirmTTM"))
                    # OCF ≈ FCFF + capex (capex is negative)
                    if ocf and capex_abs:
                        ocf = ocf + abs(capex_abs)
                    bundle.cashflow = {
                        "operatingCashFlow":        ocf,
                        "capitalExpenditure":       capex_abs,
                        "freeCashFlow":             fcfe,
                        "freeCashFlowToFirm":       fcff,
                        "netIncome":                bundle.income.get("netIncome") if bundle.income else None,
                        "depreciationAmortization": None,
                    }
                else:
                    # Final fallback: financial_statements specialty table (EODHD-sourced)
                    cf_stmt = self._financial_stmt_payload(ticker, "Cash_Flow")
                    if cf_stmt:
                        logger.info(
                            "[FM] cash_flow: using financial_statements table fallback for %s",
                            ticker,
                        )
                        bundle.cashflow = {
                            "operatingCashFlow":        cf_stmt.get("totalCashFromOperatingActivities") or cf_stmt.get("operatingCashFlow"),
                            "capitalExpenditure":       cf_stmt.get("capitalExpenditures") or cf_stmt.get("capitalExpenditure"),
                            "freeCashFlow":             cf_stmt.get("freeCashFlow"),
                            "netIncome":                cf_stmt.get("netIncome"),
                            "depreciationAmortization": cf_stmt.get("depreciation"),
                        }
        except Exception as exc:
            logger.warning("cash_flow fetch failed for %s: %s", ticker, exc)

        # ── Financial scores (FMP Piotroski / Altman / Beneish) ───────────────
        try:
            bundle.scores = self._latest_payload(ticker, "financial_scores")
        except Exception as exc:
            logger.warning("financial_scores fetch failed for %s: %s", ticker, exc)

        # ── Annual data for 3-Statement Model + Piotroski F-Score YoY signals ────
        try:
            ann_inc_rows = self.pg.fetch_annual_fundamental(ticker, "income_statement", limit=2)
            ann_bal_rows = self.pg.fetch_annual_fundamental(ticker, "balance_sheet", limit=2)
            ann_cf_rows  = self.pg.fetch_annual_fundamental(ticker, "cash_flow", limit=2)
            # Index 0 = current annual, index 1 = prior annual
            # bundle.income_annual/balance_annual/cashflow_annual = current annual (for 3SM)
            # bundle.income_prior/balance_prior/cashflow_prior = prior annual (for Piotroski YoY)

            def _map_annual_income(p: Dict) -> Dict:
                ebitda_ann = p.get("ebitda") or p.get("EBITDA")
                return {
                    "date":                   p.get("date"),
                    "period":                 p.get("period"),
                    "period_type":            p.get("period_type"),
                    "revenue":                p.get("revenue") or p.get("totalRevenue"),
                    "totalRevenue":           p.get("totalRevenue") or p.get("revenue"),
                    "costOfRevenue":          p.get("costOfRevenue"),
                    "grossProfit":            p.get("grossProfit"),
                    "researchDevelopment":    p.get("researchDevelopment") or p.get("researchAndDevelopmentExpenses"),
                    "totalOperatingExpenses": p.get("totalOperatingExpenses") or p.get("operatingExpenses"),
                    "ebit":                   p.get("operatingIncome") or p.get("ebit"),
                    "operatingIncome":        p.get("operatingIncome") or p.get("ebit"),
                    "ebitda":                 ebitda_ann,
                    "EBITDA":                 ebitda_ann,
                    "interestExpense":        p.get("interestExpense"),
                    "incomeBeforeTax":        p.get("incomeBeforeTax"),
                    "incomeTaxExpense":       p.get("incomeTaxExpense") or p.get("taxProvision"),
                    "taxProvision":           p.get("taxProvision") or p.get("incomeTaxExpense"),
                    "netIncome":              p.get("netIncome"),
                    "depreciationAndAmortization": p.get("depreciationAndAmortization"),
                    "depreciationAmortization":    p.get("depreciationAndAmortization"),
                }

            def _map_annual_balance(p: Dict) -> Dict:
                return {
                    "date":                    p.get("date"),
                    "period":                  p.get("period"),
                    "period_type":             p.get("period_type"),
                    "totalAssets":             p.get("totalAssets"),
                    "totalLiab":               p.get("totalLiab") or p.get("totalLiabilities"),
                    "totalLiabilities":        p.get("totalLiabilities") or p.get("totalLiab"),
                    "totalCurrentAssets":      p.get("totalCurrentAssets"),
                    "totalCurrentLiabilities": p.get("totalCurrentLiabilities"),
                    "longTermDebt":            p.get("longTermDebt") or p.get("longTermDebtTotal"),
                    "longTermDebtTotal":       p.get("longTermDebtTotal") or p.get("longTermDebt"),
                    "shortTermDebt":           p.get("shortTermDebt") or p.get("shortLongTermDebt"),
                    "cash":                    p.get("cash") or p.get("cashAndCashEquivalents") or p.get("cashAndEquivalents"),
                    "cashAndCashEquivalents":  p.get("cashAndCashEquivalents") or p.get("cash"),
                    "netReceivables":          p.get("netReceivables"),
                    "inventory":               p.get("inventory"),
                    "accountsPayable":         p.get("accountsPayable"),
                    "goodWill":                p.get("goodWill") or p.get("goodwill"),
                    "intangibleAssets":        p.get("intangibleAssets"),
                    "retainedEarnings":        p.get("retainedEarnings"),
                    "commonStock":             p.get("commonStock") or p.get("capitalStock"),
                    "treasuryStock":           p.get("treasuryStock"),
                    "totalStockholderEquity":  p.get("totalStockholderEquity") or p.get("totalStockholdersEquity"),
                    "totalStockholdersEquity": p.get("totalStockholdersEquity") or p.get("totalStockholderEquity"),
                    "netWorkingCapital":       p.get("netWorkingCapital"),
                    "netDebt":                 p.get("netDebt"),
                }

            def _map_annual_cashflow(p: Dict) -> Dict:
                return {
                    "date":                     p.get("date"),
                    "period":                   p.get("period"),
                    "period_type":              p.get("period_type"),
                    "totalCashFromOperatingActivities": p.get("totalCashFromOperatingActivities") or p.get("operatingCashFlow"),
                    "operatingCashFlow":        p.get("operatingCashFlow") or p.get("totalCashFromOperatingActivities"),
                    "capitalExpenditures":      p.get("capitalExpenditures") or p.get("capitalExpenditure"),
                    "capitalExpenditure":       p.get("capitalExpenditure") or p.get("capitalExpenditures"),
                    "investments":              p.get("investments"),
                    "totalCashflowsFromInvestingActivities": p.get("totalCashflowsFromInvestingActivities"),
                    "totalCashFromFinancingActivities": p.get("totalCashFromFinancingActivities"),
                    "dividendsPaid":            p.get("dividendsPaid"),
                    "salePurchaseOfStock":      p.get("salePurchaseOfStock"),
                    "issuanceOfCapitalStock":   p.get("issuanceOfCapitalStock"),
                    "netBorrowings":            p.get("netBorrowings"),
                    "changeInCash":             p.get("changeInCash") or p.get("cashAndCashEquivalentsChanges"),
                    "cashAndCashEquivalentsChanges": p.get("cashAndCashEquivalentsChanges"),
                    "beginPeriodCashFlow":      p.get("beginPeriodCashFlow"),
                    "endPeriodCashFlow":        p.get("endPeriodCashFlow"),
                    "freeCashFlow":             p.get("freeCashFlow"),
                    "netIncome":                p.get("netIncome"),
                    "depreciation":             p.get("depreciation"),
                    "depreciationAndAmortization": p.get("depreciationAndAmortization") or p.get("depreciation"),
                    "depreciationAmortization": p.get("depreciationAndAmortization") or p.get("depreciation"),
                    "stockBasedCompensation":   p.get("stockBasedCompensation"),
                    "changeInWorkingCapital":   p.get("changeInWorkingCapital"),
                    "changeToAccountReceivables": p.get("changeToAccountReceivables"),
                    "changeToInventory":        p.get("changeToInventory"),
                    "changeToLiabilities":      p.get("changeToLiabilities"),
                    "changeToOperatingActivities": p.get("changeToOperatingActivities"),
                    "otherNonCashItems":        p.get("otherNonCashItems"),
                    "cashFlowsOtherOperating":  p.get("cashFlowsOtherOperating"),
                    "exchangeRateChanges":       p.get("exchangeRateChanges"),
                }

            if ann_inc_rows:
                bundle.income_annual = _map_annual_income(ann_inc_rows[0]["payload"])
            if ann_bal_rows:
                bundle.balance_annual = _map_annual_balance(ann_bal_rows[0]["payload"])
            if ann_cf_rows:
                bundle.cashflow_annual = _map_annual_cashflow(ann_cf_rows[0]["payload"])

            if len(ann_inc_rows) >= 2:
                bundle.income_prior = _map_annual_income(ann_inc_rows[1]["payload"])
            if len(ann_bal_rows) >= 2:
                bundle.balance_prior = _map_annual_balance(ann_bal_rows[1]["payload"])
            if len(ann_cf_rows) >= 2:
                bundle.cashflow_prior = _map_annual_cashflow(ann_cf_rows[1]["payload"])
            # Also ensure current-year annual data is in bundle for signals that need it.
            # If the current income row was quarterly, overlay OCF from annual cash_flow.
            if ann_cf_rows:
                curr_cf = ann_cf_rows[0]["payload"]
                if not bundle.cashflow.get("operatingCashFlow"):
                    ocf = curr_cf.get("operatingCashFlow") or curr_cf.get("totalCashFromOperatingActivities")
                    if ocf:
                        bundle.cashflow["operatingCashFlow"] = ocf
                if not bundle.cashflow.get("freeCashFlow"):
                    fcf = curr_cf.get("freeCashFlow")
                    if fcf:
                        bundle.cashflow["freeCashFlow"] = fcf
                if not bundle.cashflow.get("capitalExpenditure"):
                    capex = curr_cf.get("capitalExpenditure") or curr_cf.get("capitalExpenditures")
                    if capex:
                        bundle.cashflow["capitalExpenditure"] = capex
        except Exception as exc:
            logger.warning("prior-year annual data fetch failed for %s: %s", ticker, exc)

        # ── key_metrics_ttm (FMP primary / EODHD fallback) ───────────────────
        try:
            bundle.key_metrics_ttm = self._latest_payload(ticker, "key_metrics_ttm")
            if bundle.key_metrics_ttm:
                ev_ttm = _safe_float(
                    # FMP field names
                    bundle.key_metrics_ttm.get("enterpriseValueTTM")
                    # EODHD field names (enterprise_values table)
                )
                mkt_cap = _safe_float(
                    # FMP field names
                    bundle.key_metrics_ttm.get("marketCap")
                    or bundle.key_metrics_ttm.get("marketCapTTM")
                    # EODHD Highlights field names
                    or bundle.key_metrics_ttm.get("MarketCapitalization")
                )
                # EODHD also stores market cap in millions (MarketCapitalizationMln)
                if mkt_cap is None:
                    mln = _safe_float(bundle.key_metrics_ttm.get("MarketCapitalizationMln"))
                    if mln is not None and mln > 0:
                        mkt_cap = mln * 1_000_000

                bundle.enterprise["enterpriseValueTTM"]   = ev_ttm
                bundle.enterprise["marketCapitalization"] = mkt_cap

                # Backfill market cap into balance sheet
                if mkt_cap and bundle.balance is not None:
                    bundle.balance["marketCapitalization"] = mkt_cap

                # Net debt bridge (EV - Market Cap)
                if ev_ttm and mkt_cap and ev_ttm > 0 and mkt_cap > 0:
                    net_debt = ev_ttm - mkt_cap
                    if bundle.balance is not None:
                        if net_debt >= 0:
                            if not bundle.balance.get("totalDebt"):
                                bundle.balance["totalDebt"] = net_debt
                            if not bundle.balance.get("cashAndCashEquivalents"):
                                bundle.balance["cashAndCashEquivalents"] = 0.0
                        else:
                            if not bundle.balance.get("totalDebt"):
                                bundle.balance["totalDebt"] = 0.0
                            if not bundle.balance.get("cashAndCashEquivalents"):
                                bundle.balance["cashAndCashEquivalents"] = abs(net_debt)

                # Supplement cashflow with FCFF if cash_flow statement was sparse
                fcff = _safe_float(bundle.key_metrics_ttm.get("freeCashFlowToFirmTTM"))
                if fcff is not None and fcff > 0 and not bundle.cashflow.get("freeCashFlow"):
                    capex_to_rev = _safe_float(bundle.key_metrics_ttm.get("capexToRevenueTTM"))
                    revenue = _safe_float(bundle.income.get("revenue")) if bundle.income else None
                    capex_abs = (capex_to_rev * revenue) if (capex_to_rev and revenue) else 0.0
                    bundle.cashflow["freeCashFlowToFirm"] = fcff
                    if not bundle.cashflow.get("operatingCashFlow"):
                        bundle.cashflow["operatingCashFlow"] = fcff + abs(capex_abs)
                    if not bundle.cashflow.get("capitalExpenditure"):
                        bundle.cashflow["capitalExpenditure"] = -abs(capex_abs)

                # Override revenue with RevenueTTM (authoritative TTM annual figure).
                # The income_statement table may contain a quarterly row as its most
                # recent entry (e.g. Q4 = $143B) which is far smaller than the true
                # annual TTM ($435B for AAPL).  RevenueTTM from key_metrics_ttm is
                # always the trailing-twelve-months annual total — use it whenever
                # available so DCF projections are based on annual revenue.
                rev_ttm = _safe_float(bundle.key_metrics_ttm.get("RevenueTTM"))
                if rev_ttm and rev_ttm > 0 and bundle.income is not None:
                    bundle.income["revenue"] = rev_ttm
                    logger.debug(
                        "[FM] Overriding income.revenue with RevenueTTM=%.0f for %s",
                        rev_ttm, ticker,
                    )
                    # Re-derive EBIT margin with updated (annual) revenue
                    ebit = _safe_float(bundle.income.get("ebit"))
                    if ebit and ebit > 0:
                        bundle.income["ebitMargin"] = round(ebit / rev_ttm, 6)
                else:
                    # EBIT margin overlay (fallback when RevenueTTM unavailable)
                    ebit   = _safe_float(bundle.income.get("ebit")) if bundle.income else None
                    rev    = _safe_float(bundle.income.get("revenue")) if bundle.income else None
                    if ebit and rev:
                        bundle.income["ebitMargin"] = round(ebit / rev, 6)
        except Exception as exc:
            logger.warning("key_metrics_ttm fetch failed for %s: %s", ticker, exc)

        # Fallback: synthesize key_metrics_ttm from company_profile_neo4j.json Highlights
        if not bundle.key_metrics_ttm:
            try:
                base_dir = os.path.join(
                    os.path.dirname(__file__), "..", "..", "ingestion", "etl", "agent_data", ticker.upper()
                )
                neo4j_path = os.path.normpath(os.path.join(base_dir, "company_profile_neo4j.json"))
                if os.path.exists(neo4j_path):
                    with open(neo4j_path, "r", encoding="utf-8") as f:
                        neo4j_data = json.load(f)
                    h = neo4j_data.get("Highlights", {})
                    v = neo4j_data.get("Valuation", {})
                    if h:
                        bundle.key_metrics_ttm = {
                            "ReturnOnEquityTTM":  h.get("ReturnOnEquityTTM"),
                            "ReturnOnAssetsTTM":  h.get("ReturnOnAssetsTTM"),
                            "OperatingMarginTTM": h.get("OperatingMarginTTM"),
                            "ProfitMargin":       h.get("ProfitMargin"),
                            "PERatio":            h.get("PERatio"),
                            "RevenueTTM":         h.get("RevenueTTM"),
                            "marketCapTTM":       h.get("MarketCapitalization"),
                            "MarketCapitalization": h.get("MarketCapitalization"),
                            "EarningsShare":      h.get("EarningsShare"),
                            "DividendYield":      h.get("DividendYield"),
                            "EBITDA":             h.get("EBITDA"),
                            "enterpriseValueTTM": v.get("EnterpriseValue") if v else None,
                            "QuarterlyRevenueGrowthYOY":   h.get("QuarterlyRevenueGrowthYOY"),
                            "QuarterlyEarningsGrowthYOY":  h.get("QuarterlyEarningsGrowthYOY"),
                        }
                        logger.info("[FM] key_metrics_ttm: using company_profile_neo4j.json fallback for %s", ticker)
                        # Backfill derived fields from the newly populated key_metrics_ttm
                        mkt_cap = _safe_float(h.get("MarketCapitalization"))
                        ev_ttm  = _safe_float(v.get("EnterpriseValue") if v else None)
                        if mkt_cap:
                            bundle.enterprise["marketCapitalization"] = mkt_cap
                            if bundle.balance is not None:
                                bundle.balance["marketCapitalization"] = mkt_cap
                        if ev_ttm:
                            if not bundle.enterprise.get("enterpriseValueTTM"):
                                bundle.enterprise["enterpriseValueTTM"] = ev_ttm
                        rev_ttm = _safe_float(h.get("RevenueTTM"))
                        if rev_ttm and rev_ttm > 0 and bundle.income is not None:
                            bundle.income["revenue"] = rev_ttm
                            ebit = _safe_float(bundle.income.get("ebit"))
                            if ebit and ebit > 0:
                                bundle.income["ebitMargin"] = round(ebit / rev_ttm, 6)
            except Exception as exc:
                logger.debug("[FM] key_metrics_ttm neo4j json fallback failed for %s: %s", ticker, exc)

        # ── ratios_ttm (FMP) ──────────────────────────────────────────────────
        try:
            bundle.ratios_ttm = self._latest_payload(ticker, "ratios_ttm")
            if bundle.ratios_ttm and bundle.income:
                bundle.income["effectiveTaxRate"] = bundle.ratios_ttm.get("effectiveTaxRateTTM")
        except Exception as exc:
            logger.warning("ratios_ttm fetch failed for %s: %s", ticker, exc)

        # Fallback: synthesize ratios_ttm from company_profile_neo4j.json Highlights + Valuation
        if not bundle.ratios_ttm:
            try:
                base_dir = os.path.join(
                    os.path.dirname(__file__), "..", "..", "ingestion", "etl", "agent_data", ticker.upper()
                )
                neo4j_path = os.path.normpath(os.path.join(base_dir, "company_profile_neo4j.json"))
                if os.path.exists(neo4j_path):
                    with open(neo4j_path, "r", encoding="utf-8") as f:
                        neo4j_data = json.load(f)
                    h = neo4j_data.get("Highlights", {})
                    v = neo4j_data.get("Valuation", {})
                    if h or v:
                        bundle.ratios_ttm = {
                            "ReturnOnEquityTTM":  h.get("ReturnOnEquityTTM"),
                            "ReturnOnAssetsTTM":  h.get("ReturnOnAssetsTTM"),
                            "OperatingMarginTTM": h.get("OperatingMarginTTM"),
                            "ProfitMargin":       h.get("ProfitMargin"),
                            "PERatio":            h.get("PERatio"),
                            "PriceSalesTTM":      v.get("PriceSalesTTM") if v else None,
                            "PriceBookMRQ":       v.get("PriceBookMRQ") if v else None,
                            "TrailingPE":         v.get("TrailingPE") if v else None,
                            "ForwardPE":          v.get("ForwardPE") if v else None,
                            "DividendYield":      h.get("DividendYield"),
                        }
                        logger.info("[FM] ratios_ttm: using company_profile_neo4j.json fallback for %s", ticker)
            except Exception as exc:
                logger.debug("[FM] ratios_ttm neo4j json fallback failed for %s: %s", ticker, exc)

        # ── enterprise_values (EODHD) — EnterpriseValue, TrailingPE, etc. ────
        try:
            ev_payload = self._latest_payload(ticker, "enterprise_values")
            if ev_payload:
                ev = _safe_float(ev_payload.get("EnterpriseValue"))
                if ev and ev > 0:
                    if not bundle.enterprise.get("enterpriseValue"):
                        bundle.enterprise["enterpriseValue"] = ev
                    if not bundle.enterprise.get("enterpriseValueTTM"):
                        bundle.enterprise["enterpriseValueTTM"] = ev
                # Also derive shares from MarketCap / RevenuePerShare if needed
                rev_per_share = _safe_float(
                    bundle.key_metrics_ttm.get("RevenuePerShareTTM")
                    or bundle.ratios_ttm.get("RevenuePerShareTTM")
                )
                rev = _safe_float(bundle.income.get("revenue")) if bundle.income else None
                if rev and rev > 0 and rev_per_share and rev_per_share > 0:
                    implied_shares = rev / rev_per_share
                    if implied_shares > 0 and not bundle.income.get("sharesOutstanding"):
                        bundle.income["sharesOutstanding"] = implied_shares
        except Exception as exc:
            logger.warning("enterprise_values fetch failed for %s: %s", ticker, exc)
        try:
            est_rows = self._latest_payload_list(ticker, "analyst_estimates", limit=8)
            if not est_rows:
                # EODHD fallback: analyst_estimates_eodhd (same raw_fundamentals table)
                est_rows = self._latest_payload_list(ticker, "analyst_estimates_eodhd", limit=8)
                if est_rows:
                    logger.info(
                        "[FM] analyst_estimates: FMP empty for %s — using EODHD analyst_estimates_eodhd fallback",
                        ticker,
                    )
            bundle.analyst_estimates = est_rows
        except Exception as exc:
            logger.warning("analyst_estimates fetch failed for %s: %s", ticker, exc)

        # ── As-reported GAAP financials (FMP) ─────────────────────────────────
        try:
            as_rep_inc = self._latest_payload_list(ticker, "income_statement_as_reported", limit=3)
            as_rep_bal = self._latest_payload_list(ticker, "balance_sheet_as_reported", limit=3)
            as_rep_cf  = self._latest_payload_list(ticker, "cash_flow_as_reported", limit=3)
            # Overlay as-reported values only if reported financials have missing fields
            if as_rep_inc and bundle.income:
                ar = as_rep_inc[0]
                for field in ("netIncome", "incomeTaxExpense", "interestExpense"):
                    if bundle.income.get(field) is None and ar.get(field) is not None:
                        bundle.income[field] = ar[field]
            if as_rep_bal and bundle.balance:
                ar = as_rep_bal[0]
                for field in ("retainedEarnings", "totalDebt", "cashAndCashEquivalents"):
                    if bundle.balance.get(field) is None and ar.get(field) is not None:
                        bundle.balance[field] = ar[field]
            if as_rep_cf and bundle.cashflow:
                ar = as_rep_cf[0]
                for field in ("operatingCashFlow", "capitalExpenditure", "freeCashFlow"):
                    if bundle.cashflow.get(field) is None and ar.get(field) is not None:
                        bundle.cashflow[field] = ar[field]
        except Exception as exc:
            logger.warning("as_reported financials fetch failed for %s: %s", ticker, exc)

        # ── Revenue segmentation (FMP primary → EODHD raw_timeseries fallback) ─
        try:
            prod_segs  = self._latest_payload_list(ticker, "revenue_product_segmentation", limit=5)
            geo_segs   = self._latest_payload_list(ticker, "revenue_geographic_segmentation", limit=5)

            # EODHD stores revenue segmentation in raw_timeseries, not raw_fundamentals
            if not prod_segs:
                prod_ts = self.pg.fetch_timeseries(ticker, "revenue_product_segmentation", limit=10)
                prod_segs = self._merge_ts_date(prod_ts)
                if prod_segs:
                    logger.info(
                        "[FM] revenue_product_segmentation: FMP empty for %s — using EODHD timeseries fallback (%d rows)",
                        ticker, len(prod_segs),
                    )
            if not geo_segs:
                geo_ts = self.pg.fetch_timeseries(ticker, "revenue_geographic_segmentation", limit=10)
                geo_segs = self._merge_ts_date(geo_ts)
                if geo_segs:
                    logger.info(
                        "[FM] revenue_geographic_segmentation: FMP empty for %s — using EODHD timeseries fallback (%d rows)",
                        ticker, len(geo_segs),
                    )
            bundle.revenue_segments = {
                "product":    prod_segs,
                "geographic": geo_segs,
            }
        except Exception as exc:
            logger.warning("revenue_segmentation fetch failed for %s: %s", ticker, exc)

        # ── Dividend history from raw_timeseries (EODHD) ─────────────────────
        try:
            div_ts_rows = self.pg.fetch_timeseries(ticker, "dividends_history", limit=30)
            if not div_ts_rows:
                # FMP fallback: historical_dividends in raw_fundamentals
                fmp_div = self.pg.fetch_latest_fundamental(ticker, "historical_dividends", limit=1)
                if fmp_div:
                    payload = fmp_div[0]["payload"]
                    if isinstance(payload, list):
                        div_ts_rows = [{"payload": r, "ts_date": r.get("date", "")} for r in payload[:30]]
                    elif isinstance(payload, dict) and "historical" in payload:
                        div_ts_rows = [
                            {"payload": r, "ts_date": r.get("date", "")}
                            for r in payload["historical"][:30]
                        ]
            if div_ts_rows:
                merged = self._merge_ts_date(div_ts_rows)
                for row in merged:
                    if "value" in row and "dividend" not in row:
                        row["dividend"] = row["value"]
                bundle.dividend_history = merged
        except Exception as exc:
            logger.warning("dividend_history fetch failed for %s: %s", ticker, exc)

        # ── EOD price history (EODHD weekly preferred) ────────────────────────
        try:
            ts_rows = self.pg.fetch_timeseries(ticker, "historical_prices_weekly", limit=300)
            if not ts_rows:
                ts_rows = self.pg.fetch_timeseries(ticker, "historical_prices_eod", limit=400)
            bundle.price_history = self._merge_ts_date(ts_rows)
        except Exception as exc:
            logger.warning("price_history fetch failed for %s: %s", ticker, exc)

        # ── Treasury rates (FMP, stored per-ticker) ───────────────────────────
        try:
            tr_rows = self.pg.fetch_timeseries(ticker, "treasury_rates", limit=5)
            if not tr_rows:
                tr_fund = self.pg.fetch_latest_fundamental(ticker, "treasury_rates", limit=5)
                if tr_fund:
                    tr_rows = [
                        {"payload": r["payload"], "ts_date": r.get("as_of_date", "")}
                        for r in tr_fund
                    ]
            if not tr_rows:
                # Legacy fallback keys
                tr_rows = self.pg.fetch_timeseries("TREASURY", "treasury_rates", limit=5)
            if not tr_rows:
                tr_rows = self.pg.fetch_timeseries("TNX", "treasury_rates", limit=5)
            bundle.treasury_rates = self._merge_ts_date(tr_rows)
        except Exception as exc:
            logger.warning("treasury_rates fetch failed: %s", exc)

        # ── Benchmark (S&P 500) history from market_eod_us ────────────────────
        try:
            mkt_rows = self.pg.fetch_market_eod(limit=400)
            bundle.benchmark_history = self._merge_ts_date(mkt_rows)
        except Exception as exc:
            logger.warning("benchmark_history fetch failed: %s", exc)

        # ── Peer group ────────────────────────────────────────────────────────
        peers = self._resolve_peers(ticker)

        # Peer price histories
        for peer in peers:
            try:
                ts_rows = self.pg.fetch_timeseries(peer, "historical_prices_weekly", limit=100)
                if not ts_rows:
                    ts_rows = self.pg.fetch_timeseries(peer, "historical_prices_eod", limit=100)
                if ts_rows:
                    bundle.peer_histories[peer] = self._merge_ts_date(ts_rows)
            except Exception as exc:
                logger.debug("peer price history fetch failed for %s: %s", peer, exc)

        # Peer fundamentals for Comps multiples (all FMP data_names)
        for peer in peers:
            try:
                peer_km_ttm    = self._latest_payload(peer, "key_metrics_ttm")
                peer_ratios_ttm = self._latest_payload(peer, "ratios_ttm")
                peer_scores    = self._latest_payload(peer, "financial_scores")
                peer_inc_rows  = self._latest_payload_list(peer, "income_statement", limit=3)
                peer_inc       = peer_inc_rows[0] if peer_inc_rows else {}
                peer_income = {
                    "revenue": peer_inc.get("revenue") or (peer_scores.get("revenue") if peer_scores else None),
                    "ebit":    peer_inc.get("operatingIncome") or (peer_scores.get("ebit") if peer_scores else None),
                    "ebitda":  peer_inc.get("ebitda"),
                }
                peer_ent = {}
                if peer_km_ttm:
                    peer_ent = {
                        "enterpriseValueTTM":   peer_km_ttm.get("enterpriseValueTTM"),
                        "marketCapitalization": peer_km_ttm.get("marketCap"),
                    }
                bundle.peer_fundamentals[peer] = {
                    "key_metrics_ttm": peer_km_ttm,
                    "ratios_ttm":      peer_ratios_ttm,
                    "enterprise":      peer_ent,
                    "income":          peer_income,
                }
            except Exception as exc:
                logger.debug("peer fundamentals fetch failed for %s: %s", peer, exc)

        # ── Economic events (Row 12) ──────────────────────────────────────────
        try:
            bundle.economic_events = self.pg.fetch_economic_events(limit=50)
        except Exception as exc:
            logger.warning("economic_events fetch failed: %s", exc)

        # ── Corporate bond yields (Row 13) ────────────────────────────────────
        try:
            bundle.bond_yields = self.pg.fetch_bond_yields(limit=10)
        except Exception as exc:
            logger.warning("bond_yields fetch failed: %s", exc)

        # ── Forex rates from dedicated table (Row 14) ─────────────────────────
        try:
            # Fetch EURUSD and USDJPY as representative pairs
            for pair in ("EURUSD", "USDJPY", "GBPUSD"):
                rows = self.pg.fetch_forex_rates_dedicated(pair, limit=5)
                bundle.forex_rates.extend(rows)
        except Exception as exc:
            logger.warning("forex_rates_dedicated fetch failed: %s", exc)

        # ── Financial statements from dedicated table (Row 18) ────────────────
        try:
            stmts = self.pg.fetch_financial_statements(ticker, limit=4)
            if stmts:
                bundle.financial_statements = stmts
        except Exception as exc:
            logger.warning("financial_statements fetch failed for %s: %s", ticker, exc)

        # ── Valuation metrics from dedicated table (Row 19) ───────────────────
        try:
            vm = self.pg.fetch_valuation_metrics(ticker)
            if vm:
                bundle.valuation_metrics = vm
        except Exception as exc:
            logger.warning("valuation_metrics fetch failed for %s: %s", ticker, exc)

        # ── Earnings surprises from dedicated table (Row 23) ──────────────────
        try:
            es_rows = self.pg.fetch_earnings_surprises(ticker, limit=20)
            if es_rows:
                bundle.earnings_surprises = es_rows
                logger.debug("[FM] earnings_surprises: %d rows for %s", len(es_rows), ticker)
        except Exception as exc:
            logger.warning("earnings_surprises fetch failed for %s: %s", ticker, exc)

        # ── Dedicated dividends table (Row 10 — primary source) ───────────────
        try:
            div_rows = self.pg.fetch_dividends_dedicated(ticker, limit=30)
            if div_rows:
                bundle.dividends_dedicated = div_rows
                logger.debug("[FM] dividends_dedicated: %d rows for %s", len(div_rows), ticker)
        except Exception as exc:
            logger.warning("dividends_dedicated fetch failed for %s: %s", ticker, exc)

        # ── Outstanding shares history (Row 22) ───────────────────────────────
        try:
            bundle.outstanding_shares = self.pg.fetch_outstanding_shares(ticker, limit=10)
        except Exception as exc:
            logger.warning("outstanding_shares fetch failed for %s: %s", ticker, exc)

        # ── Splits history (Row 10: splits part) ──────────────────────────────
        try:
            bundle.splits_history = self.pg.fetch_splits_history(ticker, limit=10)
        except Exception as exc:
            logger.warning("splits_history fetch failed for %s: %s", ticker, exc)

        # ── Macro indicators (Row 11: dedicated treasury_rates + global_macro) ─
        try:
            # Dedicated treasury_rates table (US10Y as primary WACC input)
            bundle.treasury_rates_dedicated = self.pg.fetch_treasury_rates_dedicated("US10Y", limit=5)
            # Global macro indicators: GDP, CPI, Unemployment
            for indicator in ("GDP", "CPI", "UNEMPLOYMENT"):
                rows = self.pg.fetch_macro_indicators(indicator, limit=5)
                bundle.macro_indicators.extend(rows)
        except Exception as exc:
            logger.warning("macro_indicators fetch failed: %s", exc)

        return bundle

    # Static peer map: Neo4j has no COMPETES_WITH/BELONGS_TO edges in this deployment.
    # This map covers all 5 supported tickers with their 4 closest peers.
    _STATIC_PEERS: Dict[str, List[str]] = {
        "AAPL":  ["MSFT", "GOOGL", "NVDA", "TSLA"],
        "MSFT":  ["AAPL", "GOOGL", "NVDA", "TSLA"],
        "GOOGL": ["AAPL", "MSFT", "NVDA", "TSLA"],
        "TSLA":  ["AAPL", "MSFT", "NVDA", "GOOGL"],
        "NVDA":  ["AAPL", "MSFT", "GOOGL", "TSLA"],
    }

    def _resolve_peers(self, ticker: str) -> List[str]:
        """Get peer tickers: static map first, then Neo4j, then same-sector PG query."""
        n = self.config.comps_sector_peers

        # 1. Static peer map (always available for the 5 supported tickers)
        static = self._STATIC_PEERS.get(ticker.upper(), [])
        if static:
            return static[:n]

        # 2. Neo4j (may be empty if COMPETES_WITH edges don't exist)
        peers = self.neo4j.get_peers(ticker, limit=n)
        if len(peers) >= n:
            return peers[:n]

        # 3. Same-sector PG fallback using FMP company_core_info
        sector = ""
        try:
            core = self._latest_payload(ticker, "company_core_info")
            sector = core.get("sector", "") or core.get("Sector", "")
        except Exception:
            pass
        if sector:
            pg_peers = self.pg.fetch_peer_fundamentals_by_sector(sector, ticker, limit=n)
            existing = set(peers)
            for p in pg_peers:
                if p not in existing and len(peers) < n:
                    peers.append(p)
                    existing.add(p)
        return peers[:n]


# ---------------------------------------------------------------------------
# Toolkit façade
# ---------------------------------------------------------------------------

class FMToolkit:
    """Façade combining PostgreSQL fetcher and Neo4j peer selector."""

    def __init__(self, config: Optional[FinancialModellingConfig] = None) -> None:
        self.config = config or FinancialModellingConfig()
        self.pg = PostgresConnector(self.config)
        self.neo4j = Neo4jPeerSelector(self.config)
        self.fetcher = FMDataFetcher(self.pg, self.neo4j, self.config)

    def fetch_data(self, ticker: str) -> FMDataBundle:
        return self.fetcher.fetch(ticker)

    def healthcheck(self) -> Dict[str, bool]:
        return {"postgres": self.pg.healthcheck()}

    def fetch_buyback_history(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Fetch share buyback history for the ticker."""
        return self.pg.fetch_buyback_history(ticker, limit)

    def fetch_exec_compensation(self, ticker: str, limit: int = 5) -> List[Dict]:
        """Fetch executive compensation data for the ticker."""
        return self.pg.fetch_exec_compensation(ticker, limit)

    def fetch_forex_rates(self, ticker: str, limit: int = 365) -> List[Dict]:
        """Fetch forex historical rates for the ticker."""
        return self.pg.fetch_forex_rates(ticker, limit)

    def fetch_sector_multiples(self, ticker: str, limit: int = 1) -> List[Dict]:
        """Fetch sector/industry multiples for the ticker."""
        return self.pg.fetch_sector_multiples(ticker, limit)

    def fetch_economic_events(self, limit: int = 50) -> List[Dict]:
        """Fetch recent economic events (Row 12)."""
        return self.pg.fetch_economic_events(limit)

    def fetch_bond_yields(self, limit: int = 10) -> List[Dict]:
        """Fetch recent corporate bond yield rows (Row 13)."""
        return self.pg.fetch_bond_yields(limit)

    def fetch_treasury_rates_dedicated(self, indicator: str = "US10Y", limit: int = 5) -> List[Dict]:
        """Fetch rows from the dedicated treasury_rates table (Row 11)."""
        return self.pg.fetch_treasury_rates_dedicated(indicator, limit)

    def fetch_financial_statements(
        self,
        ticker: str,
        statement_type: Optional[str] = None,
        limit: int = 4,
    ) -> Dict[str, List[Dict]]:
        """Fetch rows from the financial_statements table (Row 18)."""
        return self.pg.fetch_financial_statements(ticker, statement_type, limit)

    def fetch_valuation_metrics(self, ticker: str) -> Optional[Dict]:
        """Fetch latest valuation metrics for the ticker (Row 19)."""
        return self.pg.fetch_valuation_metrics(ticker)

    def fetch_outstanding_shares(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Fetch outstanding shares history for the ticker (Row 22)."""
        return self.pg.fetch_outstanding_shares(ticker, limit)

    def fetch_splits_history(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Fetch stock split history for the ticker (Row 10: splits)."""
        return self.pg.fetch_splits_history(ticker, limit)

    def fetch_earnings_surprises(self, ticker: str, limit: int = 20) -> List[Dict]:
        """Fetch earnings surprise history from the dedicated earnings_surprises table."""
        return self.pg.fetch_earnings_surprises(ticker, limit)

    def fetch_dividends_dedicated(self, ticker: str, limit: int = 30) -> List[Dict]:
        """Fetch dividend history from the dedicated dividends_history table."""
        return self.pg.fetch_dividends_dedicated(ticker, limit)

    def fetch_macro_indicators(self, indicator: str, limit: int = 5) -> List[Dict]:
        """Fetch global macro indicator rows (Row 11)."""
        return self.pg.fetch_macro_indicators(indicator, limit)

    def fetch_forex_rates_dedicated(self, forex_pair: str, limit: int = 5) -> List[Dict]:
        """Fetch rows from the dedicated forex_rates table (Row 14)."""
        return self.pg.fetch_forex_rates_dedicated(forex_pair, limit)

    def close(self) -> None:
        self.neo4j.close()


__all__ = [
    "FMToolkit",
    "PostgresConnector",
    "Neo4jPeerSelector",
    "FMDataFetcher",
    "_safe_float",
    "_safe_div",
]
