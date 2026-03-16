# agents/web_search/tools.py
import os
import json
import logging
import re
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import closing
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root — works regardless of where script is called from
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PostgreSQL connector for web_search agent
# ---------------------------------------------------------------------------
class PostgresConnector:
    """Thin wrapper over psycopg2 for fetching financial calendar and earnings data."""

    def __init__(self) -> None:
        self.postgres_host = os.getenv("POSTGRES_HOST", "localhost")
        self.postgres_port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.postgres_db = os.getenv("POSTGRES_DB", "financial_db")
        self.postgres_user = os.getenv("POSTGRES_USER", "postgres")
        self.postgres_password = os.getenv("POSTGRES_PASSWORD", "")

    def _connect(self):
        return psycopg2.connect(
            host=self.postgres_host,
            port=self.postgres_port,
            dbname=self.postgres_db,
            user=self.postgres_user,
            password=self.postgres_password,
        )

    def fetch_financial_calendar(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Fetch upcoming financial calendar events for the ticker."""
        sql = """
        SELECT ticker, event_type, event_date, eps_estimate, revenue_estimate
        FROM financial_calendar
        WHERE ticker = %s
        ORDER BY event_date DESC
        LIMIT %s
        """
        conn = self._connect()
        with closing(conn), conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (ticker, limit))
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def fetch_earnings_surprises_context(self, ticker: str, limit: int = 5) -> List[Dict]:
        """Fetch recent earnings surprises for context in web search."""
        sql = """
        SELECT payload, as_of_date
        FROM raw_fundamentals
        WHERE ticker_symbol = %s
          AND data_name = 'earnings_surprises_history'
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


# ---------------------------------------------------------------------------
# Perplexity API integration
# ---------------------------------------------------------------------------

# Perplexity endpoint is fixed — no env var needed for URL
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
DEFAULT_MODEL      = os.getenv("WEB_SEARCH_MODEL", "sonar-pro")

# Key is read lazily at call time so that setting os.environ["PERPLEXITY_API_KEY"]
# after module import (e.g. from the Streamlit UI) is honoured.
def _get_perplexity_key() -> str:
    return (
        os.getenv("PERPLEXITY_API_KEY", "")
        or os.getenv("PERPLEXITY_KEY", "")
    ).strip()


import time

def perplexity_chat_completions(
    messages: List[Dict],
    model: str = DEFAULT_MODEL,
    recency_filter: str = "week",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> Dict:
    api_key = _get_perplexity_key()
    if not api_key:
        raise EnvironmentError(
            "PERPLEXITY_API_KEY not found. Set it in .env or enter it in the "
            "Streamlit sidebar under 'Web Search (Optional)'."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "return_citations": True,
        "search_recency_filter": recency_filter,
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"[tools] Attempt {attempt}/{max_retries} POST {PERPLEXITY_API_URL} model={model}")
            resp = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            content = ""
            if "choices" in data:
                content = data["choices"][0]["message"]["content"]
            elif "content" in data:
                content = data["content"]

            citations = data.get("citations", [])
            return {"content": content, "citations": citations}

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in (502, 503, 504) and attempt < max_retries:
                wait = 2 ** attempt   # 2s, 4s, 8s
                logger.warning(f"[tools] {status} error, retrying in {wait}s... (attempt {attempt}/{max_retries})")
                time.sleep(wait)
                last_error = e
                continue
            raise   # non-retryable or final attempt

    if last_error is not None:
        raise last_error
    raise RuntimeError("perplexity_chat_completions: exhausted retries with no error captured")



def extract_json_from_response(content: str) -> Optional[dict]:
    """
    Safely extract and parse JSON from the model's text response.
    Handles both raw JSON and markdown-fenced ```json ... ``` blocks.
    """
    # Try markdown fenced block first: ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON (entire response is a JSON object)
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass

    logger.warning("[tools] Could not extract structured JSON from agent response.")
    return None
