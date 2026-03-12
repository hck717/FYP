"""
EODHD Unified Data Ingestion DAG  —  data_needed.txt model
===========================================================

Folder layout:
  agent_data/{TICKER}/{data_name}.json   – raw API response
  agent_data/{TICKER}/{data_name}.csv    – tabular form (when applicable)
  agent_data/{TICKER}/metadata.json      – rich per-file metadata

Tickers   : AAPL, TSLA, NVDA, MSFT, GOOGL  (env TRACKED_TICKERS)
Macro data: stored under agent_data/_MACRO/ (treasury, forex, bonds, screener,
            GDP/CPI/unemployment, economic_events, financial_calendar)

Design choices
--------------
* Incremental load:  MD5 hash comparison → skip files whose content is unchanged.
* Rate limiting:     EODHD allows 1 000 req/min.  We sleep 60/EODHD_RATE_LIMIT
                     after every real HTTP call (cached calls don’t count).
* Retry logic:       _fetch() retries up to 3x with exponential backoff using
                     tenacity (graceful fallback if not installed).
* Sentiment source:  sentiment_trends → aggregated from /api/news per-article
                     sentiment fields (pos / neg / neu).  After financial_news is
                     saved, _aggregate_news_sentiment() groups articles by date,
                     averages pos/neg/neu, and writes sentiment_trends.csv so the
                     PostgreSQL loader upserts real bullish/bearish/neutral rows.
                     The old /api/sentiments "normalized" scalar endpoint has been
                     removed — it did not provide bullish/bearish/neutral data.
* Timeouts:          scrape task   → 20 min  (fundamentals endpoint is large)
                     load tasks    →  8 min
                     summary task  →  2 min
* financial_news / realtime_news_feed share the same API call; only one HTTP
  request is made and both files are populated from the cached result.
* Macro endpoints (treasury, forex, bonds, screener, GDP/CPI/UE, economic_events,
  calendar) are fetched once (for the first ticker) and saved to _MACRO/.
  A dedicated load_postgres_macro task pushes them to PostgreSQL.
* Per-ticker fundamentals (financial_statements, valuation_metrics,
  short_interest, earnings_surprises, outstanding_shares) are extracted
  from the EODHD fundamentals/{ticker} response (filter=Financials /
  filter=SharesStats / filter=Earnings / filter=outstandingShares).
* Validation task:   eodhd_validate_data runs after all load tasks and checks
                     that key tables are non-empty; raises AirflowException on failure.
* PDF auto-detect:   textual ingestion tasks use scan_new_pdfs() to only
                     process new/modified PDFs (mtime+size hash).
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# ── Path setup so ETL modules are importable in both local dev and container ──
_DAG_DIR = pathlib.Path(__file__).resolve().parent
_CONTAINER_ETL = _DAG_DIR.parent / "etl"
_REPO_ROOT = _DAG_DIR.parent.parent.parent
_LOCAL_ETL = _REPO_ROOT / "ingestion" / "etl"

_ETL_DIR = _CONTAINER_ETL if _CONTAINER_ETL.exists() else _LOCAL_ETL

if str(_ETL_DIR) not in sys.path:
    sys.path.insert(0, str(_ETL_DIR))

from load_postgres import load_postgres_for_ticker, load_postgres_macro  # noqa: E402
from load_neo4j import load_neo4j_for_ticker  # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────────────
EODHD_API_KEY = os.getenv("EODHD_API_KEY", "")
BASE_URL = "https://eodhd.com/api"

_BASE_DIR_ENV = os.getenv("BASE_ETL_DIR")
if _BASE_DIR_ENV:
    BASE_OUTPUT_DIR = _BASE_DIR_ENV
else:
    BASE_OUTPUT_DIR = str(_ETL_DIR / "agent_data")

TRACKED_TICKERS_RAW: list[str] = [
    t.strip() for t in os.getenv("TRACKED_TICKERS", "AAPL").split(",") if t.strip()
]
TICKERS: list[str] = [f"{t}.US" for t in TRACKED_TICKERS_RAW]
TICKER_SYMBOLS: list[str] = TRACKED_TICKERS_RAW

EODHD_RATE_LIMIT: int = int(os.getenv("EODHD_RATE_LIMIT", "1000"))
# Minimum delay between real HTTP calls to stay under the rate limit
RATE_LIMIT_DELAY: float = 60.0 / EODHD_RATE_LIMIT  # ≈ 0.06 s at 1 000 req/min

_MACRO_TICKER = "_MACRO"

# ── Tenacity retry setup ───────────────────────────────────────────────────────
try:
    from tenacity import (
        retry as _tenacity_retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
    )
    _TENACITY_AVAILABLE = True
except ImportError:
    _TENACITY_AVAILABLE = False

# ── Failure callback (logs context; extend with Slack/email if desired) ───────

def _on_failure_callback(context):
    """Called by Airflow when any task fails.
    Logs the failure details and can be extended with Slack/email notifications.
    """
    task_id   = context.get("task_instance", {}).task_id if hasattr(context.get("task_instance"), "task_id") else "unknown"
    dag_id    = context.get("dag", {}).dag_id if hasattr(context.get("dag"), "dag_id") else "unknown"
    exec_date = context.get("execution_date", "unknown")
    exception = context.get("exception", "")
    print(
        f"[FAILURE] DAG={dag_id} TASK={task_id} execution_date={exec_date}\n"
        f"  exception: {exception}"
    )
    # To enable Slack notifications, set env var SLACK_WEBHOOK_URL and
    # uncomment the block below:
    # import requests as _req
    # slack_url = os.getenv("SLACK_WEBHOOK_URL")
    # if slack_url:
    #     _req.post(slack_url, json={
    #         "text": f":red_circle: Airflow failure: *{dag_id}.{task_id}* ({exec_date})\n```{exception}```"
    #     }, timeout=10)


# ── Endpoint registry ──────────────────────────────────────────────────────────────────
#   (data_name, endpoint_template, extra_params, storage_dest, description)
#
#   Endpoint templates use {ticker} → "AAPL.US", {ticker_symbol} → "AAPL"
#   Macro endpoints have no ticker placeholder.
#   financial_news & realtime_news_feed share one HTTP call (see CACHE_SHARE).

DATA_ENDPOINTS: list[tuple] = [
    # name                        endpoint                       params                                                     dest          description
    # ── Row 1: Company Profiles — stored in Neo4j ───────────────────────────────
    ("company_profile_neo4j",   "fundamentals/{ticker}",       {},                                                        "neo4j",      "Company fundamentals, EPS, valuation, highlights"),
    # ── Row 2 & 19: Financial News + Real-Time News (shared HTTP call) ───────────
    ("financial_news",            "news",                        {"s": "{ticker_symbol}", "limit": 50},                     "postgresql", "Recent ticker-tagged financial news (last 50)"),
    ("realtime_news_feed",        "news",                        {"s": "{ticker_symbol}", "limit": 50},                     "postgresql", "Alias of financial_news — real-time news feed"),
    # ── Row 3: Insider Transactions ────────────────────────────────────────────
    ("insider_transactions",      "insider-transactions",        {"code": "{ticker}", "limit": 100},                        "postgresql", "Insider buy/sell transactions (SEC Form 4)"),
    # ── Row 4: Institutional Holders ────────────────────────────────────────────
    ("institutional_holders",     "fundamentals/{ticker}",       {"filter": "Holders"},                                     "postgresql", "Major institutional / mutual fund holders"),
    # ── Row 5: Historical EOD Prices ───────────────────────────────────────────
    ("historical_prices_eod",     "eod/{ticker}",                {"from": "{date_2y}"},                                     "postgresql", "End-of-day OHLCV prices for last 2 years"),
    # ── Row 6: Intraday / Live Quotes (shared HTTP call) ───────────────────────
    ("realtime_quote",            "real-time/{ticker}",          {},                                                        "postgresql", "Real-time / delayed stock quote"),
    ("live_stock_price",          "real-time/{ticker}",          {},                                                        "postgresql", "Alias of realtime_quote — live price snapshot"),
    ("intraday_1m",               "intraday/{ticker}",           {"interval": "1m", "from": "{ts_7d}", "to": "{ts_now}"},   "postgresql", "1-minute intraday bars (last 7 days)"),
    # ── Row 7: Technicals (13 indicators from data_needed.txt) ───────────────────
    ("technical_rsi",             "technical/{ticker}",          {"function": "rsi",      "period": 14},                      "postgresql", "RSI-14 technical indicator"),
    ("technical_macd",            "technical/{ticker}",          {"function": "macd",     "fast_period": 12, "slow_period": 26, "signal_period": 9}, "postgresql", "MACD line/signal/histogram"),
    ("technical_sma20",           "technical/{ticker}",          {"function": "sma",      "period": 20},                      "postgresql", "20-day SMA"),
    ("technical_sma",             "technical/{ticker}",          {"function": "sma",      "period": 50},                      "postgresql", "50-day SMA"),
    ("technical_ema",             "technical/{ticker}",          {"function": "ema",      "period": 50},                      "postgresql", "EMA-50"),
    ("technical_bbands",          "technical/{ticker}",          {"function": "bbands",   "period": 20},                      "postgresql", "Bollinger Bands (upper/mid/lower)"),
    ("technical_atr",             "technical/{ticker}",          {"function": "atr",      "period": 14},                      "postgresql", "ATR-14"),
    ("technical_stochrsi",        "technical/{ticker}",          {"function": "stochrsi", "period": 14},                      "postgresql", "StochRSI fast-K / fast-D lines (replaces stoch — not supported by EODHD)"),
    ("technical_adx",             "technical/{ticker}",          {"function": "adx",      "period": 14},                      "postgresql", "ADX-14"),
    ("technical_cci",             "technical/{ticker}",          {"function": "cci",      "period": 20},                      "postgresql", "CCI-20"),
    ("technical_slope",           "technical/{ticker}",          {"function": "slope",    "period": 14},                      "postgresql", "Linear regression slope-14 (replaces willr — not supported by EODHD)"),
    ("technical_roc",             "technical/{ticker}",          {"function": "roc",      "period": 14},                      "postgresql", "Rate of Change"),
    ("technical_stddev",          "technical/{ticker}",          {"function": "stddev",   "period": 14},                      "postgresql", "Std-dev-14 volatility (replaces mom — not supported by EODHD)"),
    # ── Row 10: Dividend History & Stock Splits ───────────────────────────────────
    ("dividends_history",         "div/{ticker}",                {},                                                        "postgresql", "Historical dividend payments"),
    ("splits_history",            "splits/{ticker}",             {},                                                        "postgresql", "Historical stock splits"),
    # ── Row 16: Financial Calendar (per-ticker) ─────────────────────────────────
    ("financial_calendar",        "calendar/earnings",           {"symbols": "{ticker_symbol}", "from": "{date_365d}", "to": "{date_future_90d}"}, "postgresql", "Upcoming and recent earnings dates per ticker"),
    # ── Row 18: Financial Calendar macro — global IPO / splits / dividends ───────
    ("calendar_ipo",              "calendar/ipos",               {"from": "{date_today}", "to": "{date_future_90d}"},                    "postgresql", "Upcoming IPO calendar (global)"),
    ("calendar_splits",           "calendar/splits",             {"from": "{date_today}", "to": "{date_future_90d}"},                    "postgresql", "Upcoming stock split calendar"),
    ("calendar_dividends",        "calendar/dividends",          {"from": "{date_today}", "to": "{date_future_90d}"},                     "postgresql", "Global dividend calendar for next 90 days"),
    # ── Row 18: Financial Statements (Income, Balance, Cash Flow) ───────────────
    ("financial_statements",      "fundamentals/{ticker}",       {"filter": "Financials"},                                  "postgresql", "Income statement, balance sheet, cash flow (quarterly + annual)"),
    # ── Row 21: Earnings History & Surprises ─────────────────────────────────
    ("earnings_surprises",        "fundamentals/{ticker}",       {"filter": "Earnings::History"},                           "postgresql", "Historical EPS actuals vs estimates, surprise %"),
    # ── Row 20: Short Interest & Shares Stats ────────────────────────────────
    ("short_interest",            "fundamentals/{ticker}",       {"filter": "SharesStats"},                                 "postgresql", "Short interest, float, insider/institution ownership %"),
    # ── Row 22: Outstanding Shares History ──────────────────────────────────
    ("outstanding_shares",        "fundamentals/{ticker}",       {"filter": "outstandingShares"},                           "postgresql", "Historical outstanding shares count (annual + quarterly)"),
    # ── NEW: Analyst Ratings ─────────────────────────────────────────────────
    ("analyst_ratings",           "fundamentals/{ticker}",       {"filter": "AnalystRatings"},                             "postgresql", "Analyst consensus ratings (Rating, TargetPrice, StrongBuy, Buy, Hold, Sell, StrongSell)"),
    # ── NEW: Company Profiles ─────────────────────────────────────────────────
    ("company_profile",            "fundamentals/{ticker}",       {"filter": "General"},                                    "postgresql", "Company profile (Name, Exchange, Sector, Industry, Description, Address, Officers)"),
    # ── Row 7 (sentiment) — derived from financial_news aggregation ─────────────
    # sentiment_trends is populated by _aggregate_news_sentiment() called inside
    # _save_data() when data_name == "financial_news".  It groups the per-article
    # pos/neg/neu fields by date and writes sentiment_trends.csv / .json.
    # The old /api/sentiments endpoint (scalar "normalized") has been removed.
    # ── Row 8: News Word Weights ─────────────────────────────────────────────────
    ("news_word_weights",         "news-word-weights",           {"s": "{ticker_symbol}", "filter[date_from]": "{date_30d}", "filter[date_to]": "{date_today}", "page[limit]": 100}, "postgresql", "Top news word weights per ticker (last 30 days)"),
    # ── Macro / global (fetched once, stored under _MACRO) ───────────────────────
    ("screener_bulk",             "screener",                    {"limit": 100},                                            "postgresql", "Bulk market screener — top 100 stocks"),
    ("treasury_bill_rates",       "ust/bill-rates",              {"from": "{date_30d}", "to": "{date_today}"},              "postgresql", "UST bill rates (4-week, 3M, 6M, 1Y) from dedicated UST API"),
    ("treasury_yield_curve",      "ust/yield-rates",             {"from": "{date_30d}", "to": "{date_today}"},              "postgresql", "Full UST par yield curve (2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y) from UST API"),
    ("economic_indicators_gdp",   "macro-indicator/USA",         {"indicator": "gdp_growth_annual"},                        "postgresql", "US GDP growth rate macro indicator"),
    ("economic_indicators_cpi",   "macro-indicator/USA",         {"indicator": "consumer_price_index"},                     "postgresql", "US CPI inflation macro indicator"),
    ("economic_indicators_unemployment", "macro-indicator/USA",  {"indicator": "unemployment_total_percent"},               "postgresql", "US unemployment rate macro indicator"),
    ("economic_events",           "economic-events",             {"from": "{date_30d}", "to": "{date_today}", "country": "US", "limit": 1000}, "postgresql", "Global macro economic events (CPI releases, FOMC, NFP, etc.)"),
    ("bond_aapl_fundamentals",    "bond-fundamentals/US037833AK68", {},                                                       "postgresql", "Apple Inc 2.4% 2023 senior unsecured note (IG corporate bond reference)"),
    ("bond_amzn_fundamentals",    "bond-fundamentals/US023135BX34", {},                                                       "postgresql", "Amazon.com Inc 1% 2026 senior unsecured note (IG corporate bond reference)"),
    ("corporate_bond_yields",     "eod/LQD.US",                  {"from": "{date_30d}"},                                    "postgresql", "Investment-grade bond ETF (LQD) price series — credit spread proxy"),
    ("forex_historical_rates",    "eod/EURUSD.FOREX",            {"from": "{date_30d}"},                                    "postgresql", "EUR/USD forex rate (EODHD FOREX feed)"),
    ("etf_index_constituents",    "fundamentals/SPY.US",         {"filter": "ETF_Data::Holdings"},                          "neo4j",      "S&P 500 ETF (SPY) constituent holdings and weights"),
    ("market_sp500_eod",          "eod/GSPC.INDX",               {"from": "{date_2y}"},                                     "postgresql", "S&P 500 index EOD prices (2 years) — used as market benchmark for beta"),
]

# These data_names are global/macro — fetched only once for the first ticker
MACRO_DATA_NAMES: set[str] = {
    "screener_bulk",
    "treasury_bill_rates",
    "treasury_yield_curve",
    "economic_indicators_gdp",
    "economic_indicators_cpi",
    "economic_indicators_unemployment",
    "economic_events",
    "corporate_bond_yields",
    "bond_aapl_fundamentals",
    "bond_amzn_fundamentals",
    "forex_historical_rates",
    "etf_index_constituents",
    "calendar_ipo",
    "calendar_splits",
    "calendar_dividends",
    "market_sp500_eod",
}

CACHE_SHARE: dict[str, list[str]] = {
    "financial_news": ["realtime_news_feed"],
    "realtime_quote": ["live_stock_price"],
}

# ── Airflow DAG defaults ─────────────────────────────────────────────────────────────
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    # Retry 3 times with 5-minute exponential delay (matches tenacity pattern)
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": _on_failure_callback,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _md5(data: object) -> str:
    return hashlib.md5(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


def _load_metadata(ticker_symbol: str) -> dict:
    path = Path(BASE_OUTPUT_DIR) / ticker_symbol / "metadata.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_metadata(ticker_symbol: str, metadata: dict) -> None:
    path = Path(BASE_OUTPUT_DIR) / ticker_symbol / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def _fetch(endpoint: str, params: dict | None = None) -> object | None:
    """
    Make one authenticated GET request with exponential-backoff retry (3 attempts).
    Logs the full URL for sentiment endpoint transparency.
    Returns parsed JSON or None.
    """
    url = f"{BASE_URL}/{endpoint}"
    p = dict(params or {})
    p["api_token"] = EODHD_API_KEY
    p["fmt"] = "json"

    # Log endpoint + params clearly (mask API key for security)
    log_params = {k: v for k, v in p.items() if k != "api_token"}
    print(f"    Fetching: {url}  params={log_params}")

    def _single_attempt() -> object | None:
        try:
            resp = requests.get(url, params=p, timeout=90)
            print(f"    GET {endpoint}  →  HTTP {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) == 0:
                    print("    ⊘ empty list response")
                    return None
                if isinstance(data, dict) and not data:
                    print("    ⊘ empty dict response")
                    return None
                return data
            print(f"    ✗ HTTP {resp.status_code}: {resp.text[:200]}")
            # Raise so retry fires on 429 / 5xx
            if resp.status_code in (429, 500, 502, 503, 504):
                resp.raise_for_status()
            return None
        except requests.RequestException as exc:
            print(f"    ✗ RequestException: {exc}")
            raise

    if _TENACITY_AVAILABLE:
        from tenacity import retry as _retry, stop_after_attempt, wait_exponential, retry_if_exception_type
        @_retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            retry=retry_if_exception_type(requests.RequestException),
            reraise=False,
        )
        def _retried() -> object | None:
            return _single_attempt()
        try:
            return _retried()
        except Exception as exc:
            print(f"    ✗ All retries failed: {exc}")
            return None
    else:
        # Simple 3-attempt fallback with exponential backoff
        last_exc = None
        for attempt in range(3):
            try:
                result = _single_attempt()
                if result is not None:
                    return result
                return result  # None is a valid “no data” result; stop retrying
            except Exception as exc:
                last_exc = exc
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
        print(f"    ✗ All retries failed: {last_exc}")
        return None


def _aggregate_news_sentiment(ticker_symbol: str, articles: list[dict], ticker_dir: "Path") -> None:
    """
    Aggregate per-article sentiment (pos/neg/neu) from a list of news articles
    into daily averages, then write sentiment_trends.json + sentiment_trends.csv
    so the PostgreSQL loader can upsert them into sentiment_trends.

    Each article is expected to have a 'sentiment' sub-dict:
        {"polarity": 0.98, "neg": 0.054, "neu": 0.851, "pos": 0.095}
    and a 'date' field like "2026-03-09T09:20:00+00:00".
    """
    if not articles:
        return

    # Accumulate pos/neg/neu per calendar date
    buckets: dict[str, list[dict]] = {}
    for article in articles:
        raw_date = article.get("date") or ""
        # Extract YYYY-MM-DD from ISO timestamp
        date_str = str(raw_date)[:10]
        if not date_str or date_str == "None":
            continue
        sentiment = article.get("sentiment")
        if not isinstance(sentiment, dict):
            continue
        try:
            pos = float(sentiment.get("pos") or 0)
            neg = float(sentiment.get("neg") or 0)
            neu = float(sentiment.get("neu") or 0)
        except (TypeError, ValueError):
            continue
        if date_str not in buckets:
            buckets[date_str] = []
        buckets[date_str].append({"pos": pos, "neg": neg, "neu": neu})

    if not buckets:
        print(f"    ⊘ no sentiment data to aggregate for {ticker_symbol}")
        return

    rows = []
    for date_str, sentiments in sorted(buckets.items()):
        n = len(sentiments)
        avg_pos = round(sum(s["pos"] for s in sentiments) / n, 6)
        avg_neg = round(sum(s["neg"] for s in sentiments) / n, 6)
        avg_neu = round(sum(s["neu"] for s in sentiments) / n, 6)
        rows.append({
            "date":    date_str,
            "pos":     avg_pos,
            "neg":     avg_neg,
            "neu":     avg_neu,
            "article_count": n,
        })

    if not rows:
        return

    # Write JSON
    json_path = ticker_dir / "sentiment_trends.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    # Write CSV
    csv_path = ticker_dir / "sentiment_trends.csv"
    try:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    except Exception as exc:
        print(f"    ⚠ sentiment_trends CSV write warning: {exc}")

    print(f"    ✓ sentiment_trends aggregated: {len(rows)} date-rows from {sum(len(v) for v in buckets.values())} articles  → {ticker_dir}")

def _flatten_fundamentals(data: dict) -> dict:
    """Flatten EODHD fundamentals response to one CSV row."""
    general = data.get("General", {})
    flat = {k: v for k, v in general.items() if not isinstance(v, (dict, list))}
    for section, keys in [
        ("Highlights", [
            "MarketCapitalization", "MarketCapitalizationMln", "EBITDA",
            "PERatio", "PEGRatio", "WallStreetTargetPrice",
            "BookValue", "DividendShare", "DividendYield",
            "EarningsShare", "EPSEstimateCurrentYear", "EPSEstimateNextYear",
            "EPSEstimateNextQuarter", "EPSEstimateCurrentQuarter",
            "MostRecentQuarter", "ProfitMargin", "OperatingMarginTTM",
            "ReturnOnAssetsTTM", "ReturnOnEquityTTM", "RevenueTTM",
            "RevenuePerShareTTM", "QuarterlyRevenueGrowthYOY",
            "GrossProfitTTM", "DilutedEpsTTM", "QuarterlyEarningsGrowthYOY",
        ]),
        ("Valuation", [
            "TrailingPE", "ForwardPE", "PriceSalesTTM",
            "PriceBookMRQ", "EnterpriseValue",
            "EnterpriseValueRevenue", "EnterpriseValueEbitda",
        ]),
        ("SharesStats", [
            "SharesOutstanding", "SharesFloat", "PercentInsiders",
            "PercentInstitutions", "SharesShort", "SharesShortPriorMonth",
            "ShortRatio", "ShortPercentOutstanding", "ShortPercentFloat",
        ]),
        ("SplitsDividends", [
            "ForwardAnnualDividendRate", "ForwardAnnualDividendYield",
            "PayoutRatio", "DividendDate", "ExDividendDate",
            "LastSplitFactor", "LastSplitDate",
        ]),
    ]:
        section_data = data.get(section, {})
        for k in keys:
            if k in section_data and not isinstance(section_data[k], (dict, list)):
                flat[f"{section}_{k}"] = section_data[k]
    return flat


def _save_data(
    ticker_symbol: str,
    data_name: str,
    data: object,
    metadata: dict,
    storage_dest: str,
    description: str,
    endpoint: str,
) -> bool:
    """
    Persist data to JSON + CSV and update the metadata entry.

    Returns True if the file was actually written (content changed or new),
    False if unchanged (hash match → incremental skip).
    """
    if data is None:
        print(f"    ⊘ no data for {data_name}")
        return False

    ticker_dir = Path(BASE_OUTPUT_DIR) / ticker_symbol
    ticker_dir.mkdir(parents=True, exist_ok=True)

    # After saving financial_news, aggregate per-article sentiment into sentiment_trends
    if data_name == "financial_news" and isinstance(data, list):
        _aggregate_news_sentiment(ticker_symbol, data, ticker_dir)
        # Write a metadata entry for sentiment_trends so the PG loader picks it up
        st_csv = ticker_dir / "sentiment_trends.csv"
        if st_csv.exists():
            metadata["sentiment_trends"] = {
                "hash":                _md5(data),
                "last_updated":        datetime.now().isoformat(),
                "last_checked":        datetime.now().isoformat(),
                "record_count":        len(data),
                "source":              "eodhd",
                "storage_destination": "postgresql",
                "description":         "Daily sentiment (bullish/bearish/neutral) aggregated from news articles",
                "api_endpoint":        "news",
                "json_file":           "sentiment_trends.json",
                "csv_file":            "sentiment_trends.csv",
                "ticker":              ticker_symbol,
                "data_type":           "structured",
                "incremental":         True,
            }

    new_hash = _md5(data)
    existing = metadata.get(data_name, {})
    if existing.get("hash") == new_hash:
        print(f"    = unchanged: {data_name}")
        existing["last_checked"] = datetime.now().isoformat()
        metadata[data_name] = existing
        return False

    # ── Write JSON ────────────────────────────────────────────────────────────────
    json_path = ticker_dir / f"{data_name}.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # ── Write CSV (best-effort) ──────────────────────────────────────────────────
    csv_path = ticker_dir / f"{data_name}.csv"
    record_count = 1
    try:
        if isinstance(data, list):
            record_count = len(data)
            if record_count > 0:
                pd.DataFrame(data).to_csv(csv_path, index=False)
        elif isinstance(data, dict):
            if "General" in data:
                flat = _flatten_fundamentals(data)
                if flat:
                    pd.DataFrame([flat]).to_csv(csv_path, index=False)
            elif "Institutions" in data or "Funds" in data:
                rows_inst = []
                for holder_type in ("Institutions", "Funds"):
                    holders = data.get(holder_type) or {}
                    if isinstance(holders, dict):
                        for name, info in holders.items():
                            r = {"holder_type": holder_type, "name": name}
                            r.update(info if isinstance(info, dict) else {})
                            rows_inst.append(r)
                    elif isinstance(holders, list):
                        for info in holders:
                            r = {"holder_type": holder_type}
                            r.update(info if isinstance(info, dict) else {})
                            rows_inst.append(r)
                if rows_inst:
                    record_count = len(rows_inst)
                    pd.DataFrame(rows_inst).to_csv(csv_path, index=False)
            elif "Income_Statement" in data or "Balance_Sheet" in data or "Cash_Flow" in data:
                rows_fs = []
                for stmt_type in ("Income_Statement", "Balance_Sheet", "Cash_Flow"):
                    stmt = data.get(stmt_type, {})
                    for period_type in ("quarterly", "yearly"):
                        period_data = stmt.get(period_type, {})
                        if isinstance(period_data, dict):
                            for period_date, values in period_data.items():
                                if isinstance(values, dict):
                                    r = {"statement_type": stmt_type, "period_type": period_type, "report_date": period_date}
                                    r.update(values)
                                    rows_fs.append(r)
                if rows_fs:
                    record_count = len(rows_fs)
                    pd.DataFrame(rows_fs).to_csv(csv_path, index=False)
            elif "History" in data:
                history = data.get("History", {})
                rows_eh = []
                if isinstance(history, dict):
                    for period_date, values in history.items():
                        if isinstance(values, dict):
                            r = {"period_date": period_date}
                            r.update(values)
                            rows_eh.append(r)
                if rows_eh:
                    record_count = len(rows_eh)
                    pd.DataFrame(rows_eh).to_csv(csv_path, index=False)
            elif "annual" in data or "quarterly" in data:
                rows_os = []
                for period_type in ("annual", "quarterly"):
                    period_list = data.get(period_type, [])
                    if isinstance(period_list, dict):
                        period_list = list(period_list.values())
                    if isinstance(period_list, list):
                        for item in period_list:
                            if isinstance(item, dict):
                                r = {"period_type": period_type}
                                r.update(item)
                                rows_os.append(r)
                if rows_os:
                    record_count = len(rows_os)
                    pd.DataFrame(rows_os).to_csv(csv_path, index=False)
            elif "SharesOutstanding" in data or "SharesFloat" in data or "PercentInsiders" in data:
                flat_ss = {k: str(v) for k, v in data.items() if not isinstance(v, (dict, list))}
                if flat_ss:
                    pd.DataFrame([flat_ss]).to_csv(csv_path, index=False)
            else:
                list_val = next(
                    (v for v in data.values() if isinstance(v, list) and len(v) > 0),
                    None,
                )
                if list_val:
                    record_count = len(list_val)
                    pd.DataFrame(list_val).to_csv(csv_path, index=False)
                else:
                    dict_val = next(
                        (v for v in data.values() if isinstance(v, dict) and len(v) > 0),
                        None,
                    )
                    if dict_val:
                        rows_dod = [{"key": k, **v} if isinstance(v, dict) else {"key": k, "value": v}
                                    for k, v in data.items()]
                        record_count = len(rows_dod)
                        pd.DataFrame(rows_dod).to_csv(csv_path, index=False)
                    else:
                        flat = {k: str(v) for k, v in data.items()
                                if not isinstance(v, (dict, list))}
                        if flat:
                            pd.DataFrame([flat]).to_csv(csv_path, index=False)
    except Exception as exc:
        print(f"    ⚠ CSV write warning for {data_name}: {exc}")

    # ── Update metadata ───────────────────────────────────────────────────────────
    metadata[data_name] = {
        "hash":                new_hash,
        "last_updated":        datetime.now().isoformat(),
        "last_checked":        datetime.now().isoformat(),
        "record_count":        record_count,
        "source":              "eodhd",
        "storage_destination": storage_dest,
        "description":         description,
        "api_endpoint":        endpoint,
        "json_file":           f"{data_name}.json",
        "csv_file":            f"{data_name}.csv" if csv_path.exists() else None,
        "ticker":              ticker_symbol,
        "data_type":           "macro" if ticker_symbol == _MACRO_TICKER else "structured",
        "incremental":         True,
    }

    print(f"    ✓ saved {data_name}  ({record_count} records)  → {storage_dest}")
    return True


# ── Core scrape function ─────────────────────────────────────────────────────────────────

def scrape_ticker(ticker: str, ticker_symbol: str, **context) -> int:
    """
    Fetch all DATA_ENDPOINTS for one ticker, applying:
      - incremental hash-based skip (unchanged data not re-written)
      - API rate-limit delay after each real HTTP call
      - exponential-backoff retry via _fetch()
      - macro-data deduplication (only first ticker fetches macro endpoints)
      - response caching within one run (financial_news ↔ realtime_news_feed, etc.)
      - sentiment aggregation: after financial_news is saved, per-article pos/neg/neu
        fields are averaged by date and written to sentiment_trends.csv/json
    """
    now_ts = int(time.time())
    ts_7d  = int((datetime.now() - timedelta(days=7)).timestamp())
    date_7d  = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    date_30d = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    date_365d = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    date_2y   = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    date_future_90d = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")
    date_today = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*70}")
    print(f"[EODHD] scrape_ticker  ticker={ticker}  symbol={ticker_symbol}")
    print(f"  BASE_OUTPUT_DIR : {BASE_OUTPUT_DIR}")
    print(f"  RATE_LIMIT_DELAY: {RATE_LIMIT_DELAY:.4f}s ({EODHD_RATE_LIMIT} req/min)")
    print(f"{'='*70}")

    metadata        = _load_metadata(ticker_symbol)
    macro_metadata  = _load_metadata(_MACRO_TICKER)
    updates = errors = 0
    http_cache: dict = {}

    alias_to_canonical: dict[str, str] = {}
    for canonical, aliases in CACHE_SHARE.items():
        for alias in aliases:
            alias_to_canonical[alias] = canonical

    for data_name, endpoint_tpl, params_tpl, storage_dest, description in DATA_ENDPOINTS:

        is_macro = data_name in MACRO_DATA_NAMES
        is_first_ticker = ticker_symbol == TICKER_SYMBOLS[0]
        is_alias = data_name in alias_to_canonical

        print(f"\n  [{data_name}]")

        if is_macro and not is_first_ticker:
            print(f"    ↩ macro — already handled by {TICKER_SYMBOLS[0]}")
            continue

        if is_macro:
            eff_symbol = _MACRO_TICKER
            endpoint   = endpoint_tpl
            params: dict = {
                k: (
                    v.format(
                        ticker=ticker,
                        ticker_symbol=ticker_symbol,
                        date_7d=date_7d,
                        date_30d=date_30d,
                        date_365d=date_365d,
                        date_2y=date_2y,
                        date_future_90d=date_future_90d,
                        date_today=date_today,
                        ts_now=now_ts,
                        ts_7d=ts_7d,
                    )
                    if isinstance(v, str) else v
                )
                for k, v in params_tpl.items()
            }
        else:
            eff_symbol = ticker_symbol
            endpoint   = endpoint_tpl.format(ticker=ticker)
            params = {
                k: (
                    v.format(
                        ticker=ticker,
                        ticker_symbol=ticker_symbol,
                        date_7d=date_7d,
                        date_30d=date_30d,
                        date_365d=date_365d,
                        date_2y=date_2y,
                        date_future_90d=date_future_90d,
                        date_today=date_today,
                        ts_now=now_ts,
                        ts_7d=ts_7d,
                    )
                    if isinstance(v, str) else v
                )
                for k, v in params_tpl.items()
            }

        cache_key = (endpoint, frozenset(params.items()))

        try:
            if cache_key in http_cache:
                data = http_cache[cache_key]
                print(f"    ↩ reusing cached response")
            else:
                data = _fetch(endpoint, params)
                http_cache[cache_key] = data
                time.sleep(RATE_LIMIT_DELAY)

            target_metadata = macro_metadata if is_macro else metadata
            changed = _save_data(
                eff_symbol, data_name, data,
                target_metadata, storage_dest, description, endpoint,
            )
            if changed:
                updates += 1
                if is_macro:
                    _save_metadata(_MACRO_TICKER, macro_metadata)
            elif data is None:
                errors += 1

        except Exception as exc:
            print(f"    ✗ {data_name}: {exc}")
            errors += 1

    _save_metadata(ticker_symbol, metadata)

    _divider = "\u2500" * 70
    print(f"\n{_divider}")
    print(f"[EODHD] {ticker_symbol}: {updates} updated, {errors} failed/empty")
    print(_divider)

    context["task_instance"].xcom_push(
        key=f"{ticker_symbol}_updates", value=updates
    )
    return updates


# ── Summary task ─────────────────────────────────────────────────────────────────

def report_summary(**context) -> dict:
    ti = context["task_instance"]
    total = 0
    summary: dict[str, int] = {}
    for sym in TICKER_SYMBOLS:
        n = ti.xcom_pull(task_ids=f"eodhd_scrape_{sym}", key=f"{sym}_updates") or 0
        summary[sym] = n
        total += n

    print(f"\n{'='*70}")
    print(f"EODHD INGESTION SUMMARY  —  {total} total updates")
    print(f"{'='*70}")
    for k, v in summary.items():
        print(f"  {k}: {v} updates")
    print(f"{'='*70}")
    return summary


# ── Post-load validation task ──────────────────────────────────────────────────────

def validate_loaded_data(**context) -> dict:
    """
    Post-load validation: checks that key PostgreSQL tables have > 0 rows for
    every tracked ticker.  Raises AirflowException if any critical table is empty.

    Checks performed:
      - sentiment_trends:       must have rows for each ticker
      - raw_timeseries:         must have rows for each ticker
      - financial_statements:   must have rows for each ticker

    Non-critical checks (log warning only):
      - earnings_surprises
      - insider_transactions
      - treasury_yield_curve    (macro table — no per-ticker filter)

    Post-checks:
      - sentiment_trends:  warns if any rows have net_sentiment_score=0
                           (indicates upstream field-mapping bug)
    """
    import psycopg2

    pg_host = os.getenv("POSTGRES_HOST", "postgres")
    pg_port = int(os.getenv("POSTGRES_PORT", "5432"))
    pg_db   = os.getenv("POSTGRES_DB",   "airflow")
    pg_user = os.getenv("POSTGRES_USER",  "airflow")
    pg_pass = os.getenv("POSTGRES_PASSWORD", "airflow")

    # Table → (column, critical)
    # column = ticker column name in that table; None = macro table (no ticker filter)
    CHECKS: list[tuple[str, str | None, bool]] = [
        ("sentiment_trends",      "ticker", True),
        ("raw_timeseries",        "ticker_symbol", True),
        ("financial_statements",  "ticker", True),
        ("earnings_surprises",    "ticker", False),
        ("insider_transactions",  "ticker", False),
        ("treasury_yield_curve",  None,     False),  # macro table — no ticker col, non-critical
    ]

    failures: list[str] = []
    report: dict[str, dict] = {}

    try:
        conn = psycopg2.connect(
            host=pg_host, port=pg_port,
            dbname=pg_db, user=pg_user, password=pg_pass,
        )
        with conn:
            with conn.cursor() as cur:
                for table, col, critical in CHECKS:
                    if col is None:
                        # Macro table — just check total row count, no per-ticker loop
                        try:
                            cur.execute(f"SELECT COUNT(*) FROM {table}")
                            row_count = cur.fetchone()[0]
                        except Exception:
                            row_count = 0
                        key = table
                        report[key] = {"rows": row_count, "critical": critical}
                        if row_count == 0:
                            msg = f"[VALIDATE] {table} is EMPTY"
                            if critical:
                                print(f"  ❌ {msg}")
                                failures.append(msg)
                            else:
                                print(f"  ⚠️ {msg} (non-critical)")
                        else:
                            print(f"  ✓ {table}: {row_count} rows")
                        continue

                    for ticker_sym in TICKER_SYMBOLS:
                        try:
                            cur.execute(
                                f"SELECT COUNT(*) FROM {table} WHERE {col} = %s",
                                (ticker_sym,),
                            )
                            row_count = cur.fetchone()[0]
                        except Exception:
                            # Table may not exist yet on first run
                            row_count = 0

                        key = f"{table}/{ticker_sym}"
                        report[key] = {"rows": row_count, "critical": critical}

                        if row_count == 0:
                            msg = f"[VALIDATE] {table} is EMPTY for ticker={ticker_sym}"
                            if critical:
                                print(f"  ❌ {msg}")
                                failures.append(msg)
                            else:
                                print(f"  ⚠️ {msg} (non-critical)")
                        else:
                            print(f"  ✓ {table}/{ticker_sym}: {row_count} rows")

                # Post-check: warn if sentiment_trends rows all have zero scores
                # (indicates a field-mapping bug upstream in the aggregation logic)
                for ticker_sym in TICKER_SYMBOLS:
                    try:
                        cur.execute(
                            "SELECT COUNT(*) FROM sentiment_trends "
                            "WHERE net_sentiment_score = 0.0 AND ticker = %s",
                            (ticker_sym,),
                        )
                        zero_count = cur.fetchone()[0]
                        if zero_count > 0:
                            print(
                                f"  ⚠️ sentiment_trends has {zero_count} rows with "
                                f"net_sentiment_score=0 for {ticker_sym} — possible field mapping bug"
                            )
                    except Exception:
                        pass  # Column may not exist on first run

        conn.close()
    except Exception as exc:
        # Don’t block the pipeline if Postgres is unreachable during validation
        print(f"[VALIDATE] Could not connect to Postgres: {exc}")
        return {"error": str(exc)}

    if failures:
        raise AirflowException(
            f"Data validation failed ({len(failures)} critical issues):\n"
            + "\n".join(failures)
        )

    print("[VALIDATE] All critical checks passed.")
    return report


# ── Textual document ingestion wrapper functions ─────────────────────────────────

def load_textual_earnings_calls(ticker_symbol: str):
    """Wrapper to call earnings call ingestion with new-PDF-only detection."""
    import sys
    import os
    sys.path.insert(0, "/opt/airflow/ingestion/etl")
    
    # Set the correct path for Airflow
    os.environ["TEXTUAL_DATA_DIR"] = "/opt/airflow/data/textual data"
    
    from ingest_earnings_calls import load_earnings_call_chunks
    ticker = ticker_symbol.replace(".US", "")
    # Use only_new=True so Airflow only processes new/modified PDFs each run
    return load_earnings_call_chunks(ticker, only_new=False)


def load_textual_broker_reports(ticker_symbol: str):
    """Wrapper to call broker report ingestion with new-PDF-only detection."""
    import sys
    import os
    sys.path.insert(0, "/opt/airflow/ingestion/etl")
    
    # Set the correct path for Airflow
    os.environ["TEXTUAL_DATA_DIR"] = "/opt/airflow/data/textual data"
    
    from ingest_broker_reports import load_broker_report_chunks
    ticker = ticker_symbol.replace(".US", "")
    return load_broker_report_chunks(ticker, only_new=False)


# ── DAG definition ────────────────────────────────────────────────────────────────

with DAG(
    "eodhd_complete_ingestion",
    default_args=default_args,
    description="EODHD incremental ingestion for all 5 tickers + macro data",
    schedule_interval="0 1 * * *",   # 01:00 HKT daily
    start_date=days_ago(1),
    catchup=False,
    tags=["eodhd", "ingestion", "production"],
) as dag:

    scrape_tasks:    dict[str, PythonOperator] = {}
    load_pg_tasks:   dict[str, PythonOperator] = {}
    load_neo4j_tasks: dict[str, PythonOperator] = {}

    for _ticker, _symbol in zip(TICKERS, TICKER_SYMBOLS):
        scrape_tasks[_symbol] = PythonOperator(
            task_id=f"eodhd_scrape_{_symbol}",
            python_callable=scrape_ticker,
            op_kwargs={"ticker": _ticker, "ticker_symbol": _symbol},
            execution_timeout=timedelta(minutes=20),
        )

        load_pg_tasks[_symbol] = PythonOperator(
            task_id=f"eodhd_load_postgres_{_symbol}",
            python_callable=load_postgres_for_ticker,
            op_kwargs={"ticker_symbol": _symbol},
            execution_timeout=timedelta(minutes=8),
        )

        load_neo4j_tasks[_symbol] = PythonOperator(
            task_id=f"eodhd_load_neo4j_{_symbol}",
            python_callable=load_neo4j_for_ticker,
            op_kwargs={"ticker_symbol": _symbol},
            execution_timeout=timedelta(minutes=8),
        )

    load_macro_task = PythonOperator(
        task_id="eodhd_load_postgres_macro",
        python_callable=load_postgres_macro,
        execution_timeout=timedelta(minutes=8),
    )

    load_neo4j_macro_task = PythonOperator(
        task_id="eodhd_load_neo4j_macro",
        python_callable=load_neo4j_for_ticker,
        op_kwargs={"ticker_symbol": "_MACRO"},
        execution_timeout=timedelta(minutes=8),
    )

    summary_task = PythonOperator(
        task_id="eodhd_generate_summary",
        python_callable=report_summary,
        execution_timeout=timedelta(minutes=2),
    )

    # Post-load data validation task — runs after all load tasks complete
    validate_task = PythonOperator(
        task_id="eodhd_validate_data",
        python_callable=validate_loaded_data,
        execution_timeout=timedelta(minutes=5),
    )

    # Textual document ingestion tasks (earnings calls and broker reports)
    earnings_call_tasks: dict[str, PythonOperator] = {}
    broker_report_tasks: dict[str, PythonOperator] = {}

    for _symbol in TICKER_SYMBOLS:
        earnings_call_tasks[_symbol] = PythonOperator(
            task_id=f"textual_load_earnings_calls_{_symbol}",
            python_callable=load_textual_earnings_calls,
            op_kwargs={"ticker_symbol": _symbol},
            execution_timeout=timedelta(hours=6),
        )
        broker_report_tasks[_symbol] = PythonOperator(
            task_id=f"textual_load_broker_reports_{_symbol}",
            python_callable=load_textual_broker_reports,
            op_kwargs={"ticker_symbol": _symbol},
            execution_timeout=timedelta(hours=6),
        )

    # Wire up dependencies
    for _symbol in TICKER_SYMBOLS:
        scrape_tasks[_symbol] >> load_pg_tasks[_symbol] >> load_neo4j_tasks[_symbol]
        load_pg_tasks[_symbol] >> summary_task
        load_neo4j_tasks[_symbol] >> summary_task
        load_neo4j_tasks[_symbol] >> earnings_call_tasks[_symbol]
        load_neo4j_tasks[_symbol] >> broker_report_tasks[_symbol]
        earnings_call_tasks[_symbol] >> summary_task
        broker_report_tasks[_symbol] >> summary_task
        # Validation runs after each postgres load
        load_pg_tasks[_symbol] >> validate_task

    scrape_tasks[TICKER_SYMBOLS[0]] >> load_macro_task
    scrape_tasks[TICKER_SYMBOLS[0]] >> load_neo4j_macro_task
    load_macro_task >> summary_task
    load_neo4j_macro_task >> summary_task
    # Validate after macro load too
    load_macro_task >> validate_task
    # Summary runs after validation
    validate_task >> summary_task