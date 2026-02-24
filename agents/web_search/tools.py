# agents/web_search/tools.py
import os
import json
import logging
import re
import requests
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

# tools.py — remove POE_API_URL entirely
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "").strip()
DEFAULT_MODEL      = os.getenv("WEB_SEARCH_MODEL", "sonar")



def poe_chat_completions(
    messages: List[Dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout_s: int = 60,
) -> Dict:
    """
    Calls Poe API — OpenAI-compatible /chat/completions endpoint.
    Endpoint: https://api.poe.com/v1/chat/completions
    Reads POE_API_KEY from .env automatically.
    Returns: {"content": "<text>", "citations": []}
    """
    if not POE_API_KEY:
        raise EnvironmentError("POE_API_KEY not found. Add it to your .env file.")

    headers = {
        "Authorization": f"Bearer {POE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.debug(f"[tools] POST {POE_API_URL} model={model}")

    resp = requests.post(POE_API_URL, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()

    # OpenAI-compatible response shape
    content: str = ""
    if "choices" in data:
        content = data["choices"][0]["message"]["content"]
    elif "content" in data:
        content = data["content"]

    citations: List[str] = data.get("citations", [])
    return {"content": content, "citations": citations}


def extract_json_from_response(content: str) -> Optional[dict]:
    """Extract JSON from raw or markdown-fenced response."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass
    logger.warning("Could not extract structured JSON from agent response.")
    return None
