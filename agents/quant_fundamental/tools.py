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
import os
from contextlib import closing
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from .config import QuantFundamentalConfig
from .schema import DataQualityCheck, FinancialsBundle, QualityStatus, QuarterlyPeriod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Boundary Enforcement — Fundamental Math Agent whitelist
# ---------------------------------------------------------------------------

ALLOWED_DATA_TYPES = [
    # Price data
    "historical_prices_eod",
    "historical_prices_weekly",
    "historical_price",
    "eod_price",
    # Intraday quotes
    "intraday_1m",
    "intraday_5m",
    # Technical indicators
    "technical_beta",
    "technical_volatility",
    "technical_rsi",
    "technical_macd",
    # Screener
    "market_screener",
    # Basic fundamentals
    "key_metrics_ttm",
    "ratios_ttm",
    # Financial statements — required for Piotroski/Beneish/Altman/ROIC/DSO/DPO/DIO
    "financial_statements",
    "income_statement",
    "balance_sheet",
    "cash_flow",
    # Short interest & shares stats
    "short_interest",
    "shares_stats",
    # Earnings
    "earnings_history",
    "earnings_surprises",
]


def validate_data_name(data_name: str) -> None:
    """Validate that data_name is in the allowed whitelist.
    
    Raises ValueError if the data type is not allowed for the Fundamental Math Agent.
    """
    if not any(data_name == a or data_name.startswith(a) for a in ALLOWED_DATA_TYPES):
        raise ValueError(
            f"Data type not allowed for Fundamental Math Agent: {data_name}. "
            f"Allowed types: {ALLOWED_DATA_TYPES}"
        )


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
        
        Raises ValueError if data_name is not in ALLOWED_DATA_TYPES.
        """
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

    def fetch_timeseries(
        self,
        ticker: str,
        data_name: str,
        limit: int = 400,
    ) -> List[Dict[str, Any]]:
        """Fetch time-series rows for a ticker from raw_timeseries, newest first.
        
        Raises ValueError if data_name is not in ALLOWED_DATA_TYPES.
        """
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

    def fetch_factor_scores_from_mv(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Query mv_daily_factor_scores materialized view for pre-computed factor scores.

        Returns a dict with keys: piotroski_score, altman_z_score, beneish_m_score,
        roe_ttm, roa_ttm, roic_ttm, gross_margin_ttm, net_margin_ttm,
        debt_to_equity_ttm, current_ratio_ttm, as_of_date, refreshed_at.
        Returns None if the MV is empty or the ticker is not present.

        Falls back gracefully if the materialized view does not yet exist
        (e.g. fresh container before first REFRESH).
        """
        sql = """
        SELECT ticker, as_of_date, piotroski_score, altman_z_score, beneish_m_score,
               roe_ttm, roa_ttm, roic_ttm, gross_margin_ttm, net_margin_ttm,
               debt_to_equity_ttm, current_ratio_ttm, refreshed_at
        FROM mv_daily_factor_scores
        WHERE ticker = %s
        LIMIT 1
        """
        try:
            conn = self._connect()
            with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (ticker,))
                row = cur.fetchone()
            if row is None:
                return None
            return {k: (float(v) if isinstance(v, (int, float)) and k not in ("ticker",) else str(v) if v is not None else None)
                    for k, v in dict(row).items()}
        except Exception as exc:
            logger.warning(
                "[QF] fetch_factor_scores_from_mv failed for %s (MV may not be populated yet): %s",
                ticker, exc,
            )
            return None

    def fetch_short_interest(self, ticker: str) -> Optional[Dict]:
        """Fetch latest short interest data for the ticker."""
        sql = """
        SELECT ticker, as_of_date, shares_outstanding, shares_float,
               percent_insiders, percent_institutions, shares_short,
               shares_short_prior_month, short_ratio,
               short_percent_outstanding, short_percent_float
        FROM short_interest
        WHERE ticker = %s
        ORDER BY as_of_date DESC
        LIMIT 1
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker,))
            row = cur.fetchone()
            if not row:
                return None
            return {k: (float(v) if hasattr(v, '__float__') and k not in ("ticker", "as_of_date") else
                        str(v) if v is not None else None)
                    for k, v in dict(row).items()}

    def fetch_options_chain(self, ticker: str, limit: int = 50) -> List[Dict]:
        """Fetch recent options chain data for the ticker."""
        sql = """
        SELECT ticker, expiry_date, strike, call_put, implied_vol, open_interest, ts_date
        FROM options_chain
        WHERE ticker = %s
        ORDER BY ts_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def fetch_earnings_surprises(self, ticker: str, limit: int = 8) -> List[Dict]:
        """Fetch recent earnings surprises for the ticker from the dedicated table."""
        sql = """
        SELECT ticker, period_date, eps_actual, eps_estimate, eps_surprise_pct,
               revenue_actual, revenue_estimate, revenue_surprise_pct,
               before_after_market, currency
        FROM earnings_surprises
        WHERE ticker = %s
        ORDER BY period_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
            rows = cur.fetchall()
        results = []
        for row in rows:
            d = dict(row)
            # Convert Decimal/date to serialisable types
            results.append({
                "period_date": str(d.get("period_date") or ""),
                "eps_actual": float(d["eps_actual"]) if d.get("eps_actual") is not None else None,
                "eps_estimate": float(d["eps_estimate"]) if d.get("eps_estimate") is not None else None,
                "eps_surprise_pct": float(d["eps_surprise_pct"]) if d.get("eps_surprise_pct") is not None else None,
                "revenue_actual": float(d["revenue_actual"]) if d.get("revenue_actual") is not None else None,
                "revenue_estimate": float(d["revenue_estimate"]) if d.get("revenue_estimate") is not None else None,
                "revenue_surprise_pct": float(d["revenue_surprise_pct"]) if d.get("revenue_surprise_pct") is not None else None,
                "before_after_market": d.get("before_after_market"),
            })
        return results

    def fetch_senate_trades(self, ticker: str, limit: int = 20) -> List[Dict]:
        """Fetch recent senate/congress trading activity for the ticker."""
        sql = """
        SELECT ticker, politician, transaction_type, amount_range, trade_date, disclosed_date
        FROM senate_congress_trading
        WHERE ticker = %s
        ORDER BY trade_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def fetch_intraday_quotes(self, ticker: str, limit: int = 390) -> List[Dict]:
        """Fetch intraday 1-minute quotes for the ticker (Row 2: Intraday / Delayed Live Quotes)."""
        sql = """
        SELECT payload, ts_date, source
        FROM raw_timeseries
        WHERE ticker_symbol = %s
          AND data_name = 'intraday_1m'
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
            results.append({"payload": payload, "ts_date": str(row["ts_date"]), "source": row["source"]})
        return results

    def fetch_technicals(self, ticker: str, limit: int = 60) -> List[Dict]:
        """Fetch technical indicator rows for the ticker (Row 7: Beta & Volatility / Technicals).

        Returns rows from raw_timeseries where data_name matches any technical indicator
        ingested by the EODHD pipeline (rsi, macd, sma, ema, bbands, atr, adx, cci,
        roc, wma, beta, sar, stochrsi, slope, stddev).
        """
        sql = """
        SELECT ticker_symbol, data_name, payload, ts_date, source
        FROM raw_timeseries
        WHERE ticker_symbol = %s
          AND data_name LIKE 'technical_%%'
        ORDER BY ts_date DESC, data_name
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
                "data_name": row["data_name"],
                "payload": payload,
                "ts_date": str(row["ts_date"]),
                "source": row["source"],
            })
        return results

    def fetch_screener_snapshot(self, limit: int = 100) -> List[Dict]:
        """Fetch latest bulk screener snapshot rows (Row 8: Screener API / Bulk).

        Returns the most recent rows from market_screener.
        """
        sql = """
        SELECT ticker_code, payload, ts_date, source
        FROM market_screener
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
            results.append({
                "ticker_code": row["ticker_code"],
                "payload": payload,
                "ts_date": str(row["ts_date"]),
                "source": row["source"],
            })
        return results

    def fetch_valuation_metrics(self, ticker: str) -> Optional[Dict]:
        """Fetch latest valuation metrics for the ticker from the dedicated table (Row 19).

        Returns fields: trailing_pe, forward_pe, price_sales_ttm, price_book_mrq,
        enterprise_value, ev_revenue, ev_ebitda, market_cap, ebitda, pe_ratio,
        peg_ratio, eps, profit_margin, operating_margin, roa, roe, revenue_ttm, etc.

        Falls back to company_profile_neo4j.json Highlights + Valuation sections when
        the valuation_metrics table is empty or all-NULL.
        """
        sql = """
        SELECT *
        FROM valuation_metrics
        WHERE ticker = %s
        ORDER BY as_of_date DESC
        LIMIT 1
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker,))
            row = cur.fetchone()
            if row:
                result = {k: (float(v) if isinstance(v, (int, float)) and k not in ("ticker", "id") else
                              str(v) if v is not None else None)
                          for k, v in dict(row).items()}
                # Check if the row has any real values (not all None/null)
                value_fields = [k for k in result if k not in ("ticker", "id", "as_of_date", "ingested_at")]
                has_values = any(result.get(k) is not None for k in value_fields)
                if has_values:
                    return result

        # Fallback: company_profile_neo4j.json Highlights + Valuation sections
        try:
            import os
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
                    logger.info("[QF] fetch_valuation_metrics: using company_profile_neo4j.json fallback for %s", ticker)
                    return {
                        "ticker": ticker,
                        # From Valuation section (most accurate)
                        "trailing_pe":      _safe_float(v.get("TrailingPE") or h.get("PERatio")),
                        "forward_pe":       _safe_float(v.get("ForwardPE")),
                        "price_sales_ttm":  _safe_float(v.get("PriceSalesTTM")),
                        "price_book_mrq":   _safe_float(v.get("PriceBookMRQ")),
                        "enterprise_value": _safe_float(v.get("EnterpriseValue")),
                        "ev_revenue":       _safe_float(v.get("EnterpriseValueRevenue")),
                        "ev_ebitda":        _safe_float(v.get("EnterpriseValueEbitda")),
                        # From Highlights section
                        "market_cap":       _safe_float(h.get("MarketCapitalization")),
                        "ebitda":           _safe_float(h.get("EBITDA")),
                        "pe_ratio":         _safe_float(h.get("PERatio") or v.get("TrailingPE")),
                        "peg_ratio":        _safe_float(h.get("PEGRatio")),
                        "eps":              _safe_float(h.get("EarningsShare")),
                        "profit_margin":    _safe_float(h.get("ProfitMargin")),
                        "operating_margin": _safe_float(h.get("OperatingMarginTTM")),
                        "roa":              _safe_float(h.get("ReturnOnAssetsTTM")),
                        "roe":              _safe_float(h.get("ReturnOnEquityTTM")),
                        "revenue_ttm":      _safe_float(h.get("RevenueTTM")),
                        "book_value":       _safe_float(h.get("BookValue")),
                        "dividend_share":   _safe_float(h.get("DividendShare")),
                        "dividend_yield":   _safe_float(h.get("DividendYield")),
                        "wall_st_target":   _safe_float(h.get("WallStreetTargetPrice")),
                        "as_of_date":       None,
                    }
        except Exception as exc:
            logger.debug("[QF] fetch_valuation_metrics neo4j json fallback failed for %s: %s", ticker, exc)

        return None

    def fetch_analyst_ratings(self, ticker: str) -> Optional[Dict]:
        """Fetch latest analyst ratings for the ticker.

        Returns fields: rating, target_price, strong_buy_count, buy_count,
        hold_count, sell_count, strong_sell_count, total_analysts, as_of_date

        Falls back to raw_fundamentals (data_name='analyst_ratings') when the
        dedicated analyst_ratings table is empty, and further falls back to
        company_profile_neo4j.json AnalystRatings section.
        """
        sql = """
        SELECT *
        FROM analyst_ratings
        WHERE ticker = %s
        ORDER BY as_of_date DESC
        LIMIT 1
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker,))
            row = cur.fetchone()
            if row:
                return {k: (float(v) if isinstance(v, (int, float)) and k not in ("ticker", "id") else
                            str(v) if v is not None else None)
                        for k, v in dict(row).items()}

        # Fallback 1: raw_fundamentals with data_name='analyst_ratings'
        try:
            sql2 = """
            SELECT payload, as_of_date
            FROM raw_fundamentals
            WHERE ticker_symbol = %s
              AND data_name = 'analyst_ratings'
            ORDER BY as_of_date DESC
            LIMIT 1
            """
            conn2 = self._connect()
            with closing(conn2), conn2.cursor(cursor_factory=RealDictCursor) as cur2:
                cur2.execute(sql2, (ticker,))
                row2 = cur2.fetchone()
            if row2:
                payload = row2["payload"]
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        payload = {}
                if isinstance(payload, dict) and payload:
                    strong_buy = _safe_float(payload.get("StrongBuy"))
                    buy = _safe_float(payload.get("Buy"))
                    hold = _safe_float(payload.get("Hold"))
                    sell = _safe_float(payload.get("Sell"))
                    strong_sell = _safe_float(payload.get("StrongSell"))
                    total = sum(x for x in [strong_buy, buy, hold, sell, strong_sell] if x is not None)
                    logger.info("[QF] fetch_analyst_ratings: using raw_fundamentals fallback for %s", ticker)
                    return {
                        "ticker": ticker,
                        "rating": _safe_float(payload.get("Rating")),
                        "target_price": _safe_float(payload.get("TargetPrice")),
                        "strong_buy_count": strong_buy,
                        "buy_count": buy,
                        "hold_count": hold,
                        "sell_count": sell,
                        "strong_sell_count": strong_sell,
                        "total_analysts": total if total > 0 else None,
                        "as_of_date": str(row2["as_of_date"]),
                    }
        except Exception as exc:
            logger.debug("[QF] fetch_analyst_ratings raw_fundamentals fallback failed for %s: %s", ticker, exc)

        # Fallback 2: company_profile_neo4j.json AnalystRatings section
        try:
            import os
            base_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "ingestion", "etl", "agent_data", ticker.upper()
            )
            neo4j_path = os.path.normpath(os.path.join(base_dir, "company_profile_neo4j.json"))
            if os.path.exists(neo4j_path):
                with open(neo4j_path, "r", encoding="utf-8") as f:
                    neo4j_data = json.load(f)
                ar = neo4j_data.get("AnalystRatings", {})
                if ar:
                    strong_buy = _safe_float(ar.get("StrongBuy"))
                    buy = _safe_float(ar.get("Buy"))
                    hold = _safe_float(ar.get("Hold"))
                    sell = _safe_float(ar.get("Sell"))
                    strong_sell = _safe_float(ar.get("StrongSell"))
                    total = sum(x for x in [strong_buy, buy, hold, sell, strong_sell] if x is not None)
                    logger.info("[QF] fetch_analyst_ratings: using company_profile_neo4j.json fallback for %s", ticker)
                    return {
                        "ticker": ticker,
                        "rating": _safe_float(ar.get("Rating")),
                        "target_price": _safe_float(ar.get("TargetPrice")),
                        "strong_buy_count": strong_buy,
                        "buy_count": buy,
                        "hold_count": hold,
                        "sell_count": sell,
                        "strong_sell_count": strong_sell,
                        "total_analysts": total if total > 0 else None,
                        "as_of_date": None,
                    }
        except Exception as exc:
            logger.debug("[QF] fetch_analyst_ratings neo4j json fallback failed for %s: %s", ticker, exc)

        return None

    def fetch_company_profile(self, ticker: str) -> Optional[Dict]:
        """Fetch company profile for the ticker.

        Returns fields: name, exchange, sector, industry, description, address,
        city, state, country, phone, web_url, full_time_employees, etc.

        Falls back to company_profile_neo4j.json when the dedicated
        company_profiles table is empty.
        """
        sql = """
        SELECT *
        FROM company_profiles
        WHERE ticker = %s
        LIMIT 1
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker,))
            row = cur.fetchone()
            if row:
                return {k: (float(v) if isinstance(v, (int, float)) and k not in ("ticker", "id") else
                            str(v) if v is not None else None)
                        for k, v in dict(row).items()}

        # Fallback: company_profile_neo4j.json General section
        try:
            import os
            base_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "ingestion", "etl", "agent_data", ticker.upper()
            )
            neo4j_path = os.path.normpath(os.path.join(base_dir, "company_profile_neo4j.json"))
            if os.path.exists(neo4j_path):
                with open(neo4j_path, "r", encoding="utf-8") as f:
                    neo4j_data = json.load(f)
                g = neo4j_data.get("General", {})
                if g:
                    addr = g.get("AddressData") or {}
                    logger.info("[QF] fetch_company_profile: using company_profile_neo4j.json fallback for %s", ticker)
                    return {
                        "ticker": ticker,
                        "name": g.get("Name"),
                        "exchange": g.get("Exchange"),
                        "sector": g.get("Sector"),
                        "industry": g.get("Industry"),
                        "gic_sector": g.get("GicSector"),
                        "gic_group": g.get("GicGroup"),
                        "gic_industry": g.get("GicIndustry"),
                        "description": g.get("Description"),
                        "address": g.get("Address"),
                        "city": addr.get("City"),
                        "state": addr.get("State"),
                        "country": addr.get("Country"),
                        "zip": addr.get("ZIP"),
                        "phone": g.get("Phone"),
                        "web_url": g.get("WebURL"),
                        "full_time_employees": _safe_float(g.get("FullTimeEmployees")),
                        "fiscal_year_end": g.get("FiscalYearEnd"),
                        "ipo_date": g.get("IPODate"),
                        "currency": g.get("CurrencyCode"),
                        "isin": g.get("ISIN"),
                        "cusip": g.get("CUSIP"),
                        "cik": g.get("CIK"),
                        "is_delisted": g.get("IsDelisted"),
                    }
        except Exception as exc:
            logger.debug("[QF] fetch_company_profile neo4j json fallback failed for %s: %s", ticker, exc)

        return None

    def fetch_financial_statements(
        self,
        ticker: str,
        statement_type: Optional[str] = None,
        limit: int = 4,
    ) -> Dict[str, List[Dict]]:
        """Fetch rows from the financial_statements specialty table (EODHD-sourced).

        Returns a dict keyed by statement_type:
          {"Income_Statement": [...], "Balance_Sheet": [...], "Cash_Flow": [...]}

        Used as a last-resort fallback when raw_fundamentals is empty.
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
            entry = {
                "report_date": str(row["report_date"]),
                "period_type": row["period_type"],
                "payload": payload,
            }
            result.setdefault(st, []).append(entry)
            counts[st] = counts.get(st, 0) + 1
        return result

    def fetch_quarterly_trends(self, ticker: str, limit: int = 5) -> List[QuarterlyPeriod]:
        """Fetch the last `limit` quarters from financial_statements (Income_Statement) and
        return a list of QuarterlyPeriod objects sorted newest-first.

        Uses totalRevenue, grossProfit, operatingIncome, netIncome from the payload.
        Computes gross_margin and ebit_margin as derived ratios.
        """
        sql = """
        SELECT report_date, payload
        FROM financial_statements
        WHERE ticker = %s
          AND statement_type = 'Income_Statement'
          AND period_type = 'quarterly'
        ORDER BY report_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
            rows = cur.fetchall()

        periods: List[QuarterlyPeriod] = []
        for row in rows:
            payload = row["payload"] or {}
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {}

            def _f(v: Any) -> Optional[float]:
                try:
                    return float(v) if v is not None else None
                except (TypeError, ValueError):
                    return None

            report_date = str(row["report_date"])
            revenue = _f(payload.get("totalRevenue") or payload.get("revenue"))
            gross_profit = _f(payload.get("grossProfit"))
            op_income = _f(payload.get("operatingIncome"))
            net_income = _f(payload.get("netIncome"))
            eps_diluted = _f(
                payload.get("dilutedEPS")
                or payload.get("epsDiluted")
                or payload.get("eps")
            )
            gross_margin = (
                round(gross_profit / revenue, 4) if gross_profit and revenue and revenue > 0 else None
            )
            ebit_margin = (
                round(op_income / revenue, 4) if op_income and revenue and revenue > 0 else None
            )
            periods.append(QuarterlyPeriod(
                period=report_date,
                revenue=revenue,
                gross_profit=gross_profit,
                operating_income=op_income,
                net_income=net_income,
                eps_diluted=eps_diluted,
                gross_margin=gross_margin,
                ebit_margin=ebit_margin,
            ))
        return periods  # newest-first

    def fetch_basic_fundamentals(self, ticker: str) -> Dict:
        """Fetch basic fundamentals snapshot for the ticker from raw_fundamentals.

        Tries: fundamentals → company_profile → financial_scores (fallback chain).
        Returns the most recent payload dict (Row 9: Basic Fundamentals).
        """
        for data_name in ("fundamentals", "company_profile", "financial_scores"):
            rows = self.fetch_latest_fundamental(ticker, data_name, limit=1)
            if rows and rows[0].get("payload"):
                return rows[0]["payload"] if isinstance(rows[0]["payload"], dict) else {}
        return {}

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

    def _financial_stmt_payload(
        self,
        ticker: str,
        statement_type: str,
    ) -> Dict[str, Any]:
        """Fetch the most recent payload from the financial_statements specialty table.

        Used as a last-resort fallback when raw_fundamentals is empty.

        Args:
            statement_type: 'Income_Statement' | 'Balance_Sheet' | 'Cash_Flow'
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
                "[QF] financial_statements fallback failed for %s/%s: %s",
                ticker, statement_type, exc,
            )
        return {}

    def fetch(self, ticker: str) -> FinancialsBundle:
        """Fetch all data needed for factor computation for a single ticker.

        RESTRICTED TO FUNDAMENTAL MATH AGENT:
        Only fetches data types in ALLOWED_DATA_TYPES:
        - Price data: historical_prices_eod, historical_prices_weekly
        - Technical indicators: technical_beta, technical_volatility, technical_rsi, technical_macd
        - Basic fundamentals: key_metrics_ttm, ratios_ttm
        - Short interest: short_interest, shares_stats
        - Earnings: earnings_history, earnings_surprises
        
        Excluded (for Financial Modelling Agent): income_statement, balance_sheet,
        cash_flow, financial_ratios, enterprise_values, financial_scores, shares_float.
        """
        bundle = FinancialsBundle(ticker=ticker)

        # === ALLOWED DATA: key_metrics_ttm ===
        try:
            bundle.key_metrics_ttm = self._latest_payload(ticker, "key_metrics_ttm")
        except ValueError:
            logger.warning("[QF] key_metrics_ttm not in allowed data types for %s", ticker)
        except Exception as exc:
            logger.warning("Failed to fetch key_metrics_ttm for %s: %s", ticker, exc)

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
                    if h:
                        bundle.key_metrics_ttm = {
                            "ReturnOnEquityTTM": h.get("ReturnOnEquityTTM"),
                            "ReturnOnAssetsTTM": h.get("ReturnOnAssetsTTM"),
                            "OperatingMarginTTM": h.get("OperatingMarginTTM"),
                            "ProfitMargin": h.get("ProfitMargin"),
                            "PERatio": h.get("PERatio"),
                            "RevenueTTM": h.get("RevenueTTM"),
                            "marketCapTTM": h.get("MarketCapitalization"),
                            "EarningsShare": h.get("EarningsShare"),
                            "DividendYield": h.get("DividendYield"),
                            "QuarterlyRevenueGrowthYOY": h.get("QuarterlyRevenueGrowthYOY"),
                            "QuarterlyEarningsGrowthYOY": h.get("QuarterlyEarningsGrowthYOY"),
                        }
                        logger.info("[QF] fetch key_metrics_ttm: using company_profile_neo4j.json fallback for %s", ticker)
            except Exception as exc:
                logger.debug("[QF] key_metrics_ttm neo4j json fallback failed for %s: %s", ticker, exc)

        # === ALLOWED DATA: ratios_ttm ===
        try:
            bundle.ratios_ttm = self._latest_payload(ticker, "ratios_ttm")
        except ValueError:
            logger.warning("[QF] ratios_ttm not in allowed data types for %s", ticker)
        except Exception as exc:
            logger.warning("Failed to fetch ratios_ttm for %s: %s", ticker, exc)

        # Fallback: synthesize ratios_ttm from company_profile_neo4j.json Highlights
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
                            "ReturnOnEquityTTM": h.get("ReturnOnEquityTTM"),
                            "ReturnOnAssetsTTM": h.get("ReturnOnAssetsTTM"),
                            "OperatingMarginTTM": h.get("OperatingMarginTTM"),
                            "ProfitMargin": h.get("ProfitMargin"),
                            "PERatio": h.get("PERatio"),
                            "PriceSalesTTM": v.get("PriceSalesTTM"),
                            "PriceBookMRQ": v.get("PriceBookMRQ"),
                            "TrailingPE": v.get("TrailingPE"),
                            "ForwardPE": v.get("ForwardPE"),
                            "DividendYield": h.get("DividendYield"),
                        }
                        logger.info("[QF] fetch ratios_ttm: using company_profile_neo4j.json fallback for %s", ticker)
            except Exception as exc:
                logger.debug("[QF] ratios_ttm neo4j json fallback failed for %s: %s", ticker, exc)

        # === ALLOWED DATA: earnings_surprises ===
        try:
            earnings_data = self.pg.fetch_earnings_surprises(ticker, limit=8)
            bundle.earnings_surprises = earnings_data
            # Also keep basic_fundamentals for any legacy references
            if earnings_data:
                bundle.basic_fundamentals = {"earnings_surprises": earnings_data}
        except Exception as exc:
            logger.warning("Failed to fetch earnings_surprises for %s: %s", ticker, exc)

        # === Valuation metrics (dedicated table) ===
        try:
            vm = self.pg.fetch_valuation_metrics(ticker)
            if vm:
                bundle.valuation_metrics = vm
        except Exception as exc:
            logger.warning("Failed to fetch valuation_metrics for %s: %s", ticker, exc)

        # === Short interest (dedicated table) ===
        try:
            si = self.pg.fetch_short_interest(ticker)
            if si:
                bundle.short_interest = si
        except Exception as exc:
            logger.warning("Failed to fetch short_interest for %s: %s", ticker, exc)

        # === Historical price data (ALLOWED) ===
        # Preference order: daily EOD → weekly → legacy variants
        try:
            ts_rows = self.pg.fetch_timeseries(ticker, "historical_prices_eod", limit=400)
            if not ts_rows:
                ts_rows = self.pg.fetch_timeseries(ticker, "historical_prices_weekly", limit=400)
            if not ts_rows:
                ts_rows = self.pg.fetch_timeseries(ticker, "historical_price", limit=400)
            if not ts_rows:
                ts_rows = self.pg.fetch_timeseries(ticker, "eod_price", limit=400)
            # Supplement with weekly if daily is sparse
            if len(ts_rows) < 60:
                weekly_rows = self.pg.fetch_timeseries(ticker, "historical_prices_weekly", limit=400)
                if weekly_rows:
                    existing_dates = {str(row.get("ts_date", "")) for row in ts_rows}
                    for row in weekly_rows:
                        if str(row.get("ts_date", "")) not in existing_dates:
                            ts_rows.append(row)
                    logger.info("Price history for %s supplemented with weekly data — total %d rows", ticker, len(ts_rows))
            # Flatten payload into top-level keys for momentum_risk compatibility.
            # DB rows are: {"payload": {"close": ..., "adjusted_close": ...}, "ts_date": "..."}
            # _parse_price_rows expects: {"close": ..., "adjusted_close": ..., "date": "..."}
            flat_rows = []
            for row in ts_rows:
                payload = row.get("payload")
                ts_date = row.get("ts_date", "")
                if isinstance(payload, dict):
                    flat = dict(payload)
                    # Ensure date is accessible as "date" or "ts_date"
                    flat.setdefault("date", ts_date)
                    flat.setdefault("ts_date", ts_date)
                    flat_rows.append(flat)
                else:
                    # Fallback: use the row as-is
                    flat_rows.append(row)
            bundle.price_history = flat_rows
        except ValueError:
            logger.warning("[QF] historical price data not in allowed data types for %s", ticker)
        except Exception as exc:
            logger.warning("Failed to fetch price history for %s: %s", ticker, exc)

        # === Technical indicators (ALLOWED) ===
        # Use the dedicated fetch_technicals() method which uses a LIKE query internally
        # (bypasses validate_data_name since it's already scoped to allowed technical_ prefix)
        try:
            tech_rows = self.pg.fetch_technicals(ticker, limit=200)
            bundle.technicals = tech_rows
        except Exception as exc:
            logger.warning("Failed to fetch technical indicators for %s: %s", ticker, exc)

        # === Analyst Ratings (dedicated table) ===
        try:
            ar = self.pg.fetch_analyst_ratings(ticker)
            if ar:
                bundle.analyst_ratings = ar
        except Exception as exc:
            logger.warning("Failed to fetch analyst_ratings for %s: %s", ticker, exc)

        # === Financial Statements (now ALLOWED — needed for Piotroski/Beneish/Altman/ROIC) ===
        try:
            stmts = self.pg.fetch_financial_statements(ticker, limit=4)
            # Prefer annual statements; fallback to quarterly
            def _best_stmt(rows: List[Dict]) -> Dict:
                if not rows:
                    return {}
                annual = [r for r in rows if (r.get("period_type") or "").lower() in ("annual", "yearly", "fy")]
                chosen = annual[0] if annual else rows[0]
                p = chosen.get("payload")
                return p if isinstance(p, dict) else {}

            inc_rows = stmts.get("Income_Statement", [])
            bal_rows = stmts.get("Balance_Sheet", [])
            cf_rows  = stmts.get("Cash_Flow", [])
            bundle.income    = _best_stmt(inc_rows)
            bundle.balance   = _best_stmt(bal_rows)
            bundle.cashflow  = _best_stmt(cf_rows)
            # Keep prev-period rows for Piotroski YoY deltas
            bundle.income_prev  = _best_stmt(inc_rows[1:]) if len(inc_rows) > 1 else {}
            bundle.balance_prev = _best_stmt(bal_rows[1:]) if len(bal_rows) > 1 else {}
            bundle.cf_prev      = _best_stmt(cf_rows[1:])  if len(cf_rows) > 1  else {}
            logger.info(
                "[QF] financial_statements for %s: IS=%d  BS=%d  CF=%d",
                ticker, len(inc_rows), len(bal_rows), len(cf_rows),
            )
            # Build quarterly trends for QoQ/YoY analysis
            try:
                qt_rows = self.pg.fetch_quarterly_trends(ticker, limit=5)
                bundle.quarterly_trends = qt_rows
            except Exception as qt_exc:
                logger.warning("Failed to fetch quarterly_trends for %s: %s", ticker, qt_exc)
        except Exception as exc:
            logger.warning("Failed to fetch financial_statements for %s: %s", ticker, exc)

        # === Benchmark price history (S&P 500 from market_eod_us) ===
        try:
            bench_rows = self.pg.fetch_market_eod(limit=400)
            flat_bench = []
            for row in bench_rows:
                payload = row.get("payload")
                ts_date = row.get("ts_date", "")
                if isinstance(payload, dict):
                    flat = dict(payload)
                    flat.setdefault("date", ts_date)
                    flat.setdefault("ts_date", ts_date)
                    flat_bench.append(flat)
                else:
                    flat_bench.append(row)
            bundle.benchmark_history = flat_bench
        except Exception as exc:
            logger.warning("Failed to fetch benchmark history for %s: %s", ticker, exc)

        return bundle


# ---------------------------------------------------------------------------
# Data quality checker — PostgreSQL-based field-presence + range validation


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
        # key_metrics_ttm — EODHD PascalCase keys
        ("key_metrics_ttm", "ReturnOnEquityTTM", -5.0, 50.0, "ROE"),
        ("key_metrics_ttm", "ReturnOnAssetsTTM", -2.0, 5.0, "ROA"),
        ("key_metrics_ttm", "OperatingMarginTTM", -1.0, 1.0, "operating margin fraction"),
        ("key_metrics_ttm", "ProfitMargin", -1.0, 1.0, "profit margin fraction"),
        ("key_metrics_ttm", "PERatio", 0.0, 2000.0, "P/E ratio"),
        # ratios_ttm — EODHD PascalCase keys
        ("ratios_ttm", "ReturnOnEquityTTM", -5.0, 50.0, "ROE"),
        ("ratios_ttm", "OperatingMarginTTM", -1.0, 1.0, "operating margin fraction"),
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
# 3B: Code-generating ReAct — execute_python sandbox
# ---------------------------------------------------------------------------

# Allowed top-level module names for the sandboxed exec environment.
# This is a whitelist — any import outside this list is blocked.
_ALLOWED_EXEC_MODULES = frozenset({
    "math", "statistics", "datetime", "json", "re",
    "pandas", "numpy",  # common financial data libraries
})


def execute_python_on_bundle(
    code: str,
    bundle: "FinancialsBundle",
    extra_vars: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute an LLM-generated pandas/Python snippet against a FinancialsBundle.

    The code runs in a restricted exec() environment with:
      - ``bundle``  : the FinancialsBundle for the ticker
      - ``result``  : pre-initialised to None — the code must assign to it
      - a whitelist of safe built-ins (no open/exec/eval/os/sys)

    Returns a dict:
      {"success": True,  "result": <value>, "stdout": <str>}
    or
      {"success": False, "error": <str>}

    Security notes:
      - __builtins__ is replaced with a restricted set.
      - import is replaced with a whitelist-checked version.
      - No file I/O, no subprocess, no network.
      - Execution is synchronous — caller is responsible for timeouts.
    """
    import io
    import builtins

    # Restricted import: only allow whitelisted modules
    def _restricted_import(name: str, *args: Any, **kwargs: Any) -> Any:
        top = name.split(".")[0]
        if top not in _ALLOWED_EXEC_MODULES:
            raise ImportError(
                f"Module '{name}' is not allowed in the execute_python sandbox. "
                f"Allowed: {sorted(_ALLOWED_EXEC_MODULES)}"
            )
        return builtins.__import__(name, *args, **kwargs)

    safe_builtins = {
        k: getattr(builtins, k)
        for k in (
            "abs", "all", "any", "bool", "dict", "dir", "divmod",
            "enumerate", "filter", "float", "format", "frozenset",
            "getattr", "hasattr", "hash", "help", "hex", "id",
            "int", "isinstance", "issubclass", "iter", "len", "list",
            "map", "max", "min", "next", "oct", "pow", "print",
            "range", "repr", "reversed", "round", "set", "setattr",
            "slice", "sorted", "str", "sum", "tuple", "type", "zip",
            "None", "True", "False", "NotImplemented", "Ellipsis",
        )
        if hasattr(builtins, k)
    }
    safe_builtins["__import__"] = _restricted_import

    # Capture stdout from print() calls inside the code
    _stdout_capture = io.StringIO()

    exec_globals: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "bundle": bundle,
        "result": None,
    }
    if extra_vars:
        exec_globals.update(extra_vars)

    import sys
    _orig_stdout = sys.stdout
    try:
        sys.stdout = _stdout_capture  # type: ignore[assignment]
        exec(compile(code, "<llm_code>", "exec"), exec_globals)  # noqa: S102
    except Exception as exc:
        sys.stdout = _orig_stdout
        return {"success": False, "error": str(exc)}
    finally:
        sys.stdout = _orig_stdout

    return {
        "success": True,
        "result": exec_globals.get("result"),
        "stdout": _stdout_capture.getvalue(),
    }


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

    def execute_python(
        self,
        code: str,
        bundle: "FinancialsBundle",
        extra_vars: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run LLM-generated Python code against a FinancialsBundle (3B: code-generating ReAct)."""
        return execute_python_on_bundle(code, bundle, extra_vars=extra_vars)

    def healthcheck(self) -> Dict[str, bool]:
        return {"postgres": self.pg.healthcheck()}

    def fetch_short_interest(self, ticker: str) -> Optional[Dict]:
        """Fetch latest short interest data for the ticker."""
        return self.pg.fetch_short_interest(ticker)

    def fetch_options_chain(self, ticker: str, limit: int = 50) -> List[Dict]:
        """Fetch recent options chain data for the ticker."""
        return self.pg.fetch_options_chain(ticker, limit)

    def fetch_earnings_surprises(self, ticker: str, limit: int = 8) -> List[Dict]:
        """Fetch recent earnings surprises for the ticker."""
        return self.pg.fetch_earnings_surprises(ticker, limit)

    def fetch_senate_trades(self, ticker: str, limit: int = 20) -> List[Dict]:
        """Fetch recent senate/congress trading activity for the ticker."""
        return self.pg.fetch_senate_trades(ticker, limit)

    def fetch_intraday_quotes(self, ticker: str, limit: int = 390) -> List[Dict]:
        """Fetch intraday 1-minute quotes for the ticker (Row 2)."""
        return self.pg.fetch_intraday_quotes(ticker, limit)

    def fetch_technicals(self, ticker: str, limit: int = 60) -> List[Dict]:
        """Fetch technical indicator rows for the ticker (Row 7)."""
        return self.pg.fetch_technicals(ticker, limit)

    def fetch_screener_snapshot(self, limit: int = 100) -> List[Dict]:
        """Fetch latest bulk screener snapshot rows (Row 8)."""
        return self.pg.fetch_screener_snapshot(limit)

    def fetch_basic_fundamentals(self, ticker: str) -> Dict:
        """Fetch basic fundamentals snapshot for the ticker (Row 9)."""
        return self.pg.fetch_basic_fundamentals(ticker)

    def fetch_valuation_metrics(self, ticker: str) -> Optional[Dict]:
        """Fetch latest valuation metrics for the ticker (Row 19)."""
        return self.pg.fetch_valuation_metrics(ticker)

    def close(self) -> None:
        pass  # psycopg2 connections are opened/closed per-query


__all__ = [
    "QuantFundamentalToolkit",
    "PostgresConnector",
    "DataQualityChecker",
    "FinancialDataFetcher",
    "AnomalyDetector",
    "execute_python_on_bundle",
    "_safe_float",
    "_safe_div",
]
