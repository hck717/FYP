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


def perplexity_chat_completions(
    messages: List[Dict],
    model: str = DEFAULT_MODEL,
    recency_filter: str = "week",
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout_s: int = 90,
) -> Dict:
    """
    Calls Perplexity Sonar API — OpenAI-compatible /chat/completions endpoint.
    Endpoint: https://api.perplexity.ai/chat/completions
    Reads PERPLEXITY_API_KEY from .env automatically.

    Args:
        messages:        Full message list including system prompt.
        model:           Perplexity model. Default: sonar-pro.
        recency_filter:  "day" | "week" | "month" | "year"
        temperature:     Keep low (0.1) for factual finance output.
        max_tokens:      Max output tokens.
        timeout_s:       Request timeout in seconds.

    Returns:
        {"content": "<assistant text>", "citations": [<url strings>]}
    """
    if not PERPLEXITY_API_KEY:
        raise EnvironmentError(
            "PERPLEXITY_API_KEY not found. Add it to your .env file."
        )

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

    logger.debug(f"[tools] POST {PERPLEXITY_API_URL} model={model} recency={recency_filter}")

    resp = requests.post(
        PERPLEXITY_API_URL,
        headers=headers,
        json=payload,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()

    # Parse OpenAI-compatible response
    content: str = ""
    if "choices" in data:
        content = data["choices"][0]["message"]["content"]
    elif "content" in data:
        content = data["content"]

    citations: List[str] = data.get("citations", [])
    return {"content": content, "citations": citations}


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
