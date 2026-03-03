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
        """Fallback: top-N tickers by market cap in the same GICS sector."""
        sql = """
        SELECT DISTINCT ticker_symbol
        FROM raw_fundamentals
        WHERE data_name = 'key_metrics_ttm'
          AND ticker_symbol != %s
          AND payload->>'sector' = %s
        ORDER BY (payload->>'marketCapTTM')::numeric DESC NULLS LAST
        LIMIT %s
        """
        try:
            conn = self._connect()
            with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (exclude_ticker, sector, limit))
                rows = cur.fetchall()
            return [r["ticker_symbol"] for r in rows]
        except Exception as exc:
            logger.warning("fetch_peer_fundamentals_by_sector failed: %s", exc)
            return []

    def healthcheck(self) -> bool:
        try:
            conn = self._connect()
            conn.close()
            return True
        except Exception as exc:
            logger.warning("PostgreSQL healthcheck failed: %s", exc)
            return False


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

        Data layout in the actual PostgreSQL schema:
          raw_fundamentals:
            - "financial_scores"     : revenue, ebit, totalAssets, totalLiabilities,
                                       workingCapital, retainedEarnings, marketCap,
                                       altmanZScore, piotroskiScore
            - "fundamentals"         : Highlights_*/Valuation_* flat keys (company overview)
            - "key_metrics_ttm"      : evToEBITDATTM, evToSalesTTM, marketCap,
                                       enterpriseValueTTM, capexToRevenueTTM, etc.
            - "ratios_ttm"           : priceToEarningsRatioTTM, priceToSalesRatioTTM,
                                       dividendYieldTTM, dividendPayoutRatioTTM, etc.
            - "earnings_history"     : stub metadata only (no EPS data usable)
            - "analyst_estimates_eodhd": same shape as "fundamentals"

          raw_timeseries:
            - "historical_prices_eod"    : ~23 daily rows  (open/high/low/close/volume)
            - "historical_prices_weekly" : ~54 weekly rows
            - "historical_prices_monthly": ~26 monthly rows
            - "dividends_history"        : quarterly dividend rows with "value" key
        """
        bundle = FMDataBundle(ticker=ticker)

        # ── financial_scores: income/balance proxies + Piotroski + Altman ────
        # This table has revenue, ebit, totalAssets, totalLiabilities,
        # workingCapital, retainedEarnings, marketCap — sufficient for DCF + Altman Z.
        try:
            scores_payload = self._latest_payload(ticker, "financial_scores")
            bundle.scores = scores_payload

            # Populate income/balance from financial_scores so DCF + Altman Z work
            if scores_payload:
                bundle.income = {
                    "revenue":        scores_payload.get("revenue"),
                    "ebit":           scores_payload.get("ebit"),
                    "operatingIncome": scores_payload.get("ebit"),
                    # Derive net income: income * profit margin from fundamentals overlay later
                    "netIncome":      None,
                    "interestExpense": None,
                    "incomeTaxExpense": None,
                    "incomeBeforeTax":  None,
                }
                bundle.balance = {
                    "totalAssets":          scores_payload.get("totalAssets"),
                    "totalLiabilities":     scores_payload.get("totalLiabilities"),
                    "workingCapital":       scores_payload.get("workingCapital"),
                    "retainedEarnings":     scores_payload.get("retainedEarnings"),
                    # totalCurrentAssets/Liabilities not stored; approximate from workingCapital
                    "totalCurrentAssets":   None,
                    "totalCurrentLiabilities": None,
                    "totalDebt":            None,
                    "cashAndCashEquivalents": None,
                    "longTermDebt":         None,
                }
        except Exception as exc:
            logger.warning("financial_scores fetch failed for %s: %s", ticker, exc)

        # ── fundamentals (flat): overlay additional income/balance fields ────
        try:
            fund_payload = self._latest_payload(ticker, "fundamentals")
            if fund_payload:
                # Derive net income from revenue × profit margin
                revenue = bundle.income.get("revenue") if bundle.income else None
                profit_margin = _safe_float(fund_payload.get("Highlights_ProfitMargin"))
                if revenue and profit_margin:
                    bundle.income["netIncome"] = revenue * profit_margin

                # Use Highlights_EBITDA to back-derive cash flow proxy
                ebitda = _safe_float(fund_payload.get("Highlights_EBITDA"))
                if ebitda and bundle.income:
                    bundle.income["ebitda"] = ebitda

                # Market cap from fundamentals for WACC capital structure
                mkt_cap = _safe_float(fund_payload.get("Highlights_MarketCapitalization"))
                if mkt_cap and bundle.balance:
                    bundle.balance["marketCapitalization"] = mkt_cap

                # Enterprise value
                ev = _safe_float(fund_payload.get("Valuation_EnterpriseValue"))
                if ev:
                    bundle.enterprise["enterpriseValue"] = ev
                    bundle.enterprise["enterpriseValueTTM"] = ev
                    bundle.enterprise["marketCapitalization"] = mkt_cap

                # Forward P/E proxy for earnings estimate
                fwd_pe = _safe_float(fund_payload.get("Valuation_ForwardPE"))
                if fwd_pe:
                    bundle.enterprise["forwardPE"] = fwd_pe

        except Exception as exc:
            logger.warning("fundamentals fetch failed for %s: %s", ticker, exc)

        # ── key_metrics_ttm ────────────────────────────────────────────────────
        try:
            bundle.key_metrics_ttm = self._latest_payload(ticker, "key_metrics_ttm")
            # Overlay useful fields onto enterprise if not already set
            if bundle.key_metrics_ttm:
                if not bundle.enterprise.get("enterpriseValueTTM"):
                    bundle.enterprise["enterpriseValueTTM"] = bundle.key_metrics_ttm.get("enterpriseValueTTM")
                # marketCap from key_metrics_ttm uses key "marketCap"
                if not bundle.enterprise.get("marketCapitalization"):
                    bundle.enterprise["marketCapitalization"] = bundle.key_metrics_ttm.get("marketCap")
        except Exception as exc:
            logger.warning("key_metrics_ttm fetch failed for %s: %s", ticker, exc)

        # ── ratios_ttm ─────────────────────────────────────────────────────────
        try:
            bundle.ratios_ttm = self._latest_payload(ticker, "ratios_ttm")
            # Pull payout ratio and effective tax rate into income
            if bundle.ratios_ttm and bundle.income:
                bundle.income["effectiveTaxRate"] = bundle.ratios_ttm.get("effectiveTaxRateTTM")
        except Exception as exc:
            logger.warning("ratios_ttm fetch failed for %s: %s", ticker, exc)

        # ── Earnings history: use analyst_estimates_eodhd for EPS data ────────
        # The "earnings_history" data_name only contains stub metadata.
        # Real EPS data is in analyst_estimates_eodhd (same flat shape as fundamentals).
        try:
            est_payload = self._latest_payload(ticker, "analyst_estimates_eodhd")
            if est_payload:
                # Build synthetic EPS entry from available highlights
                eps_est = _safe_float(est_payload.get("Highlights_EPSEstimateCurrentYear"))
                eps_actual = None
                # No historical actual EPS in the DB — leave as empty list
                # but populate analyst_estimates for forward P/E
                bundle.analyst_estimates = [est_payload] if est_payload else []
        except Exception as exc:
            logger.warning("analyst_estimates_eodhd fetch failed for %s: %s", ticker, exc)

        # ── Dividend history from raw_timeseries ──────────────────────────────
        # "dividends_history" lives in raw_timeseries with payload keys:
        # value, unadjustedValue, period, paymentDate, recordDate, declarationDate
        try:
            div_ts_rows = self.pg.fetch_timeseries(ticker, "dividends_history", limit=30)
            if div_ts_rows:
                # Normalise: rename "value" → "dividend" so _compute_dividends can find it
                merged = self._merge_ts_date(div_ts_rows)
                for row in merged:
                    if "value" in row and "dividend" not in row:
                        row["dividend"] = row["value"]
                bundle.dividend_history = merged
        except Exception as exc:
            logger.warning("dividend_history fetch failed for %s: %s", ticker, exc)

        # ── EOD price history (weekly preferred for indicator depth) ──────────
        # Daily: ~23 rows (only ~1 month) — insufficient for SMA 50/200.
        # Weekly: ~54 rows (>1 year) — sufficient for SMA 20/50, EMA 26, RSI, MACD, HV30.
        # We use weekly as primary; fall back to daily if weekly missing.
        try:
            ts_rows = self.pg.fetch_timeseries(ticker, "historical_prices_weekly", limit=300)
            if not ts_rows:
                ts_rows = self.pg.fetch_timeseries(ticker, "historical_prices_eod", limit=400)
            if not ts_rows:
                ts_rows = self.pg.fetch_timeseries(ticker, "historical_price", limit=400)
            bundle.price_history = self._merge_ts_date(ts_rows)
        except Exception as exc:
            logger.warning("price_history fetch failed for %s: %s", ticker, exc)

        # ── Treasury rates ─────────────────────────────────────────────────────
        try:
            tr_rows = self.pg.fetch_timeseries("TREASURY", "treasury_rates", limit=5)
            if not tr_rows:
                tr_rows = self.pg.fetch_timeseries("TNX", "treasury_rates", limit=5)
            bundle.treasury_rates = self._merge_ts_date(tr_rows)
        except Exception as exc:
            logger.warning("treasury_rates fetch failed: %s", exc)

        # ── Benchmark (S&P 500) history from market_eod_us ───────────────────
        try:
            mkt_rows = self.pg.fetch_market_eod(limit=400)
            bundle.benchmark_history = self._merge_ts_date(mkt_rows)
        except Exception as exc:
            logger.warning("benchmark_history fetch failed: %s", exc)

        # ── Peer group: static fallback (Neo4j has no COMPETES_WITH edges) ────
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

        # Peer fundamentals for Comps multiples
        for peer in peers:
            try:
                peer_km_ttm = self._latest_payload(peer, "key_metrics_ttm")
                peer_ratios_ttm = self._latest_payload(peer, "ratios_ttm")
                peer_scores = self._latest_payload(peer, "financial_scores")
                peer_income = {}
                if peer_scores:
                    peer_income = {
                        "revenue": peer_scores.get("revenue"),
                        "ebit": peer_scores.get("ebit"),
                        "ebitda": None,
                    }
                peer_ent = {}
                if peer_km_ttm:
                    peer_ent = {
                        "enterpriseValueTTM": peer_km_ttm.get("enterpriseValueTTM"),
                        "marketCapitalization": peer_km_ttm.get("marketCap"),
                    }
                bundle.peer_fundamentals[peer] = {
                    "key_metrics_ttm": peer_km_ttm,
                    "ratios_ttm": peer_ratios_ttm,
                    "enterprise": peer_ent,
                    "income": peer_income,
                }
            except Exception as exc:
                logger.debug("peer fundamentals fetch failed for %s: %s", peer, exc)

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

        # 3. Same-sector PG fallback
        sector = ""
        try:
            fund = self._latest_payload(ticker, "fundamentals")
            sector = fund.get("Sector", "") or fund.get("sector", "")
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
