"""
Step 1: Load insider transactions and news articles from PostgreSQL/Neo4j.

Data source
-----------
PostgreSQL tables:
  - insider_transactions: ticker, insider_name, insider_title, transaction_type, shares, price, transaction_date, value
  - news_articles: ticker, title, content, source, article_date, tags, sentiment

Neo4j relationships:
  (Company)-[:HAS_INSIDER_TRADE]->(Insider)
  (Company)-[:HAS_NEWS]->(NewsArticle)

This module retrieves data from PostgreSQL (primary) or Neo4j (fallback) and
returns as langchain_core.documents.Document objects.

Run standalone::

    python agent_step1_pg.py          # default AAPL
    python agent_step1_pg.py NVDA
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timedelta

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── Load .env ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_env_path = _REPO_ROOT / ".env"
if _env_path.exists():
    try:
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())
    except Exception:
        pass

# ── PostgreSQL config ─────────────────────────────────────────────────────────
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB = os.getenv("POSTGRES_DB", "airflow")
PG_USER = os.getenv("POSTGRES_USER", "airflow")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "airflow")

# ── Neo4j config (fallback) ───────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "SecureNeo4jPass2025!")

# ── Data retention (6 months) ─────────────────────────────────────────────────
DATA_RETENTION_MONTHS = 6
INSIDER_LIMIT = 100  # Max insider transactions per ticker
NEWS_LIMIT = 50      # Max news articles per ticker


def _get_pg_connection():
    """Create PostgreSQL connection."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD,
        )
        return conn
    except Exception as e:
        logger.warning(f"PostgreSQL connection failed: {e}")
        return None


def _get_neo4j_driver():
    """Create Neo4j driver (fallback)."""
    try:
        from neo4j import GraphDatabase
        return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except Exception as e:
        logger.warning(f"Neo4j connection failed: {e}")
        return None


def _fetch_insider_transactions_pg(ticker: str) -> List[Document]:
    """Fetch insider transactions from PostgreSQL."""
    conn = _get_pg_connection()
    if not conn:
        logger.warning(f"No PostgreSQL connection for insider transactions ({ticker})")
        return []

    try:
        cursor = conn.cursor()
        cutoff_date = datetime.now() - timedelta(days=DATA_RETENTION_MONTHS * 30)

        query = """
            SELECT 
                ticker, insider_name, transaction_type, 
                shares, price, transaction_date
            FROM insider_transactions
            WHERE ticker = %s AND transaction_date >= %s
            ORDER BY transaction_date DESC
            LIMIT %s
        """

        cursor.execute(query, (ticker, cutoff_date, INSIDER_LIMIT))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        documents = []
        for row in rows:
            (tick, insider_name, trans_type, shares, price, trans_date) = row
            
            # Skip if shares is NULL (no transaction data)
            if shares is None:
                continue
            
            # Calculate transaction value (handle None price)
            try:
                value = float(shares) * float(price) if price else 0
            except (ValueError, TypeError):
                value = 0
            
            # Build document text
            price_str = f"${price:.2f}" if price else "N/A"
            text = (
                f"Insider: {insider_name}\n"
                f"Transaction Type: {trans_type}\n"
                f"Shares: {shares}\n"
                f"Price: {price_str}\n"
                f"Value: ${value:,.0f}\n"
                f"Date: {trans_date}"
            )

            doc = Document(
                page_content=text,
                metadata={
                    "ticker": tick,
                    "source": "insider_transactions",
                    "insider_name": insider_name,
                    "transaction_type": trans_type,
                    "shares": shares,
                    "price": float(price) if price else None,
                    "transaction_date": str(trans_date),
                    "value": value,
                    "doc_name": f"{insider_name} - {trans_type} {shares} shares",
                    "page": 0,
                },
            )
            documents.append(doc)

        logger.info(f"Fetched {len(documents)} insider transactions for {ticker}")
        return documents

    except Exception as e:
        logger.error(f"Error fetching insider transactions: {e}")
        return []


def _fetch_news_articles_pg(ticker: str) -> List[Document]:
    """Fetch news articles from PostgreSQL."""
    conn = _get_pg_connection()
    if not conn:
        logger.warning(f"No PostgreSQL connection for news articles ({ticker})")
        return []

    try:
        cursor = conn.cursor()
        cutoff_date = datetime.now() - timedelta(days=DATA_RETENTION_MONTHS * 30)

        query = """
            SELECT 
                ticker, title, content, link, article_date, tags, sentiment_polarity
            FROM news_articles
            WHERE ticker = %s AND article_date >= %s
            ORDER BY article_date DESC
            LIMIT %s
        """

        cursor.execute(query, (ticker, cutoff_date, NEWS_LIMIT))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        documents = []
        for row in rows:
            (tick, title, content, link, article_date, tags, sentiment) = row

            # Build document text
            text = (
                f"Title: {title}\n"
                f"Date: {article_date}\n"
                f"Tags: {tags}\n"
                f"Sentiment: {sentiment}\n\n"
                f"Content:\n{content[:1000]}"  # Truncate content to 1000 chars
            )

            doc = Document(
                page_content=text,
                metadata={
                    "ticker": tick,
                    "source": "news_articles",
                    "title": title,
                    "article_date": str(article_date),
                    "tags": str(tags),
                    "sentiment": sentiment,
                    "link": link,
                    "doc_name": f"News - {article_date.strftime('%Y-%m-%d')}",
                    "page": 0,
                },
            )
            documents.append(doc)

        logger.info(f"Fetched {len(documents)} news articles for {ticker}")
        return documents

    except Exception as e:
        logger.error(f"Error fetching news articles: {e}")
        return []


def fetch_insider_and_news_data(ticker: str) -> Tuple[List[Document], List[Document], dict]:
    """
    Fetch insider transactions and news articles for a given ticker.

    Returns
    -------
    (insider_docs, news_docs, metadata)
        insider_docs: List of Document objects for insider transactions
        news_docs: List of Document objects for news articles
        metadata: Dict with counts and date range
    """
    logger.info(f"Fetching insider & news data for {ticker}...")

    insider_docs = _fetch_insider_transactions_pg(ticker)
    news_docs = _fetch_news_articles_pg(ticker)

    # Calculate date range
    all_dates = []
    for doc in insider_docs + news_docs:
        if "transaction_date" in doc.metadata:
            all_dates.append(doc.metadata["transaction_date"])
        elif "article_date" in doc.metadata:
            all_dates.append(doc.metadata["article_date"])

    date_range = f"{min(all_dates)} to {max(all_dates)}" if all_dates else "No data"

    metadata = {
        "insider_count": len(insider_docs),
        "news_count": len(news_docs),
        "date_range": date_range,
        "data_source": "pg",
    }

    return insider_docs, news_docs, metadata


if __name__ == "__main__":
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    insider_docs, news_docs, meta = fetch_insider_and_news_data(ticker)

    print(f"\n{'='*60}")
    print(f"  Insider & News Data: {ticker}")
    print(f"{'='*60}\n")
    print(f"Insider Transactions: {meta['insider_count']}")
    for doc in insider_docs[:3]:
        print(f"  - {doc.metadata['doc_name']}")
    print(f"\nNews Articles: {meta['news_count']}")
    for doc in news_docs[:3]:
        print(f"  - {doc.metadata['doc_name']}")
    print(f"\nDate Range: {meta['date_range']}")
