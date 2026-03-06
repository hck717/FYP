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
from contextlib import closing
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from .config import FinancialModellingConfig
from .schema import FMDataBundle

logger = logging.getLogger(__name__)


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
        """Fetch time-series rows from raw_timeseries, newest first."""
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

    def fetch_valuation_metrics(self, ticker: str) -> Optional[Dict]:
        """Fetch latest valuation metrics for the ticker (Row 19)."""
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
            if not row:
                return None
            return {k: (float(v) if isinstance(v, (int, float)) and k not in ("ticker", "id") else
                        str(v) if v is not None else None)
                    for k, v in dict(row).items()}

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
                "revenue":              inc.get("revenue"),
                "grossProfit":          inc.get("grossProfit"),
                "ebit":                 inc.get("operatingIncome") or inc.get("ebit"),
                "operatingIncome":      inc.get("operatingIncome") or inc.get("ebit"),
                "ebitda":               ebitda_val,
                "netIncome":            inc.get("netIncome"),
                "interestExpense":      inc.get("interestExpense"),
                "incomeTaxExpense":     inc.get("incomeTaxExpense"),
                "incomeBeforeTax":      inc.get("incomeBeforeTax"),
                "depreciationAmortization": inc.get("depreciationAndAmortization"),
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
                "totalAssets":              bal.get("totalAssets"),
                "totalLiabilities":         bal.get("totalLiabilities") or bal.get("totalLiab"),
                "totalDebt":                bal.get("totalDebt") or bal.get("longTermDebt"),
                "longTermDebt":             bal.get("longTermDebt"),
                "cashAndCashEquivalents":   bal.get("cashAndCashEquivalents") or bal.get("cash"),
                "workingCapital":           bal.get("totalCurrentAssets", 0)
                                            - bal.get("totalCurrentLiabilities", 0)
                                            if (bal.get("totalCurrentAssets") and bal.get("totalCurrentLiabilities"))
                                            else bal.get("workingCapital"),
                "retainedEarnings":         bal.get("retainedEarnings"),
                "totalCurrentAssets":       bal.get("totalCurrentAssets"),
                "totalCurrentLiabilities":  bal.get("totalCurrentLiabilities"),
                "totalStockholdersEquity":  bal.get("totalStockholdersEquity"),
                "marketCapitalization":     None,  # populated from key_metrics_ttm below
            }
        except Exception as exc:
            logger.warning("balance_sheet fetch failed for %s: %s", ticker, exc)

        # ── Cash flow statement (FMP primary → EODHD key_metrics_ttm → financial_statements) ──
        try:
            rows = self._latest_payload_list(ticker, "cash_flow", limit=5)
            cf = rows[0] if rows else {}
            bundle.cashflow = {
                "operatingCashFlow":   cf.get("operatingCashFlow"),
                "capitalExpenditure":  cf.get("capitalExpenditure"),
                "freeCashFlow":        cf.get("freeCashFlow"),
                "netIncome":           cf.get("netIncome"),
                "depreciationAmortization": cf.get("depreciationAndAmortization"),
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

        # ── key_metrics_ttm (FMP) — EV, market cap, FCFF, capex/rev ──────────
        try:
            bundle.key_metrics_ttm = self._latest_payload(ticker, "key_metrics_ttm")
            if bundle.key_metrics_ttm:
                ev_ttm  = _safe_float(bundle.key_metrics_ttm.get("enterpriseValueTTM"))
                mkt_cap = _safe_float(bundle.key_metrics_ttm.get("marketCap"))
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

                # EBIT margin overlay
                ebit   = _safe_float(bundle.income.get("ebit")) if bundle.income else None
                rev    = _safe_float(bundle.income.get("revenue")) if bundle.income else None
                if ebit and rev:
                    bundle.income["ebitMargin"] = round(ebit / rev, 6)
        except Exception as exc:
            logger.warning("key_metrics_ttm fetch failed for %s: %s", ticker, exc)

        # ── ratios_ttm (FMP) ──────────────────────────────────────────────────
        try:
            bundle.ratios_ttm = self._latest_payload(ticker, "ratios_ttm")
            if bundle.ratios_ttm and bundle.income:
                bundle.income["effectiveTaxRate"] = bundle.ratios_ttm.get("effectiveTaxRateTTM")
        except Exception as exc:
            logger.warning("ratios_ttm fetch failed for %s: %s", ticker, exc)

        # ── Analyst estimates (FMP primary → EODHD fallback) ─────────────────
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
