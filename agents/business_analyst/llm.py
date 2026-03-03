"""LLM interaction utilities for the Business Analyst agent."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

import requests

from .config import BusinessAnalystConfig
from .prompts import SYSTEM_PROMPT, JSON_SCHEMA_PROMPT, REWRITE_PROMPT
from .schema import SentimentSnapshot

logger = logging.getLogger(__name__)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks produced by deepseek-r1."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_markdown_fences(text: str) -> str:
    """Strip leading/trailing markdown code fences (```json ... ``` or ``` ... ```)."""
    # Remove opening fence: ```json or ```
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
    # Remove closing fence
    text = re.sub(r"\n?```\s*$", "", text.strip())
    return text.strip()


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract the first top-level JSON object from raw LLM text.

    Strategy (in order):
    1. Strip <think> blocks.
    2. Try parsing the cleaned string directly.
    3. Find first ``{`` … matching ``}`` balanced brace substring and parse that.
    4. Raise JSONDecodeError so callers can surface the failure clearly.
    """
    cleaned = _strip_think_tags(text)
    cleaned = _strip_markdown_fences(cleaned)
    # Direct parse (ideal case)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Find first balanced JSON object
    start = cleaned.find("{")
    if start != -1:
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(cleaned[start:], start=start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise json.JSONDecodeError("No valid JSON object found in LLM response", cleaned, 0)


class LLMClient:
    def __init__(self, config: BusinessAnalystConfig) -> None:
        self.config = config

    def generate(self, query: str, ticker: Optional[str], context: str, sentiment: Optional[SentimentSnapshot] = None) -> Dict[str, Any]:
        # Extract real chunk IDs from context so we can inject them into the prompt
        # to prevent the LLM from constructing plausible-looking but wrong IDs.
        real_chunk_ids = re.findall(r'chunk_id: (qdrant::[^\]]+|neo4j::[^\]]+)', context)
        payload = {
            "model": self.config.llm_model,
            "prompt": self._build_prompt(query, ticker, context, sentiment, real_chunk_ids),
            "temperature": self.config.llm_temperature,
            "num_predict": self.config.llm_max_tokens,
            "stream": False,
            "think": False,   # suppress <think> chain-of-thought blocks (deepseek-r1:8b)
        }
        resp = requests.post(
            f"{self.config.ollama_base_url}/api/generate",
            json=payload,
            timeout=self.config.request_timeout,
        )
        resp.raise_for_status()
        content = resp.json().get("response", "")
        try:
            return _extract_json(content)
        except json.JSONDecodeError as exc:
            logger.error("LLM JSON decode failed: %s\nRaw response snippet: %.500s", exc, content)
            raise

    def rewrite_query(self, query: str) -> str:
        payload = {
            "model": self.config.llm_model,
            "prompt": REWRITE_PROMPT.replace("{{query}}", query),
            "temperature": 0.2,
            "num_predict": 200,
            "stream": False,
        }
        resp = requests.post(
            f"{self.config.ollama_base_url}/api/generate",
            json=payload,
            timeout=self.config.request_timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", query)
        return _strip_think_tags(raw).strip() or query

    def _build_prompt(self, query: str, ticker: Optional[str], context: str, sentiment: Optional[SentimentSnapshot] = None, real_chunk_ids: Optional[list] = None) -> str:
        schema = JSON_SCHEMA_PROMPT.replace("{{ticker}}", ticker or "null").replace("{{today}}", self._today())
        # Pre-fill exact sentiment numbers so the LLM cannot hallucinate them.
        # The schema template has placeholder zeros; we replace them with actual values
        # when sentiment data is available, so the LLM just copies them.
        if sentiment:
            schema = schema.replace(
                '"bullish_pct": 0,\n    "bearish_pct": 0,\n    "neutral_pct": 0,',
                f'"bullish_pct": {sentiment.bullish_pct},\n    "bearish_pct": {sentiment.bearish_pct},\n    "neutral_pct": {sentiment.neutral_pct},',
            )
            schema = schema.replace(
                '"trend": "improving|deteriorating|stable|unknown",',
                f'"trend": "{sentiment.trend}",',
            )
        # Inject the exact retrieved chunk IDs to prevent the LLM from constructing
        # plausible-looking but incorrect citation slugs.
        if real_chunk_ids:
            ids_list = "\n".join(f"  - {cid}" for cid in real_chunk_ids)
            chunk_id_note = (
                f"\nIMPORTANT — Valid chunk IDs for citations (copy EXACTLY, do not invent others):\n{ids_list}\n"
            )
        else:
            chunk_id_note = ""
        return f"{SYSTEM_PROMPT}\nContext:\n{context}\n{chunk_id_note}\nQuestion: {query}\n\n{schema}"

    @staticmethod
    def _today() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


__all__ = ["LLMClient"]
