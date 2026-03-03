# agents/web_search/tools.py
import os
import json
import logging
import re
import requests
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load .env from repo root — works regardless of where script is called from
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

# Perplexity endpoint is fixed — no env var needed for URL
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "").strip()
DEFAULT_MODEL      = os.getenv("WEB_SEARCH_MODEL", "sonar-pro")


import time

def perplexity_chat_completions(
    messages: List[Dict],
    model: str = DEFAULT_MODEL,
    recency_filter: str = "week",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> Dict:
    if not PERPLEXITY_API_KEY:
        raise EnvironmentError("PERPLEXITY_API_KEY not found.")

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
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
