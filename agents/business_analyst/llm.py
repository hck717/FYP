"""LLM interaction utilities for the Business Analyst agent."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import requests

from .config import BusinessAnalystConfig
from .prompts import SYSTEM_PROMPT, JSON_SCHEMA_PROMPT, REWRITE_PROMPT
from .schema import SentimentSnapshot

logger = logging.getLogger(__name__)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks produced by deepseek-r1.

    Handles two cases:
    1. Properly closed blocks: <think>...</think>  — stripped entirely.
    2. Unterminated blocks: <think>... (no closing tag) — everything from
       <think> to end-of-string is stripped, because the model output
       was truncated inside the think block and contains no real JSON.
    """
    # Remove properly closed blocks first
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove any remaining unterminated opening tag and everything after it
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def _strip_markdown_fences(text: str) -> str:
    """Strip leading/trailing markdown code fences (```json ... ``` or ``` ... ```)."""
    # Remove opening fence: ```json or ```
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
    # Remove closing fence
    text = re.sub(r"\n?```\s*$", "", text.strip())
    return text.strip()


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract the first top-level JSON object from a plain text string.

    Strategy:
    1. Try parsing the text directly (after stripping markdown fences).
    2. Find first ``{`` … matching ``}`` balanced brace substring and parse that.
    3. Raise JSONDecodeError if no valid JSON object found.
    """
    cleaned = _strip_markdown_fences(text)
    # Direct parse (ideal case)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Find first balanced JSON object using brace matching
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

        # Balanced-brace scan failed — likely truncated JSON.
        # Attempt repair: close all unclosed { and [ with matching } and ].
        fragment = cleaned[start:]
        repaired = _repair_truncated_json(fragment)
        if repaired is not None:
            return repaired

    raise json.JSONDecodeError("No valid JSON object found in text", cleaned, 0)


def _repair_truncated_json(fragment: str) -> Optional[Dict[str, Any]]:
    """Attempt to close a truncated JSON object by appending missing brackets.

    Walks the fragment tracking brace/bracket depth and string state, then
    appends the appropriate closing characters in reverse stack order.
    Returns the parsed dict on success, or None if repair fails.
    """
    stack: list = []
    in_str = False
    escape = False
    for ch in fragment:
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
        if ch in ("{", "["):
            stack.append("}" if ch == "{" else "]")
        elif ch in ("}", "]") and stack:
            stack.pop()

    if not stack:
        return None  # already balanced — nothing to repair

    # Close any open string first, then close all open containers
    closer = ('"' if in_str else "") + "".join(reversed(stack))
    candidate = fragment.rstrip().rstrip(",") + closer
    try:
        result = json.loads(candidate)
        if isinstance(result, dict):
            logger.warning("JSON repair succeeded — %d characters appended", len(closer))
            return result
    except json.JSONDecodeError:
        pass
    return None


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract the first top-level JSON object from raw LLM text.

    Strategy (in order):
    1. Strip <think> blocks; if content remains, parse it.
    2. If stripping think-blocks left nothing (model embedded JSON inside <think>),
       fall back to searching the raw text (pre-strip) for a JSON object.
    3. Raise JSONDecodeError so callers can surface the failure clearly.
    """
    stripped = _strip_think_tags(text).strip()

    # If content remains after stripping think-blocks, try that first
    if stripped:
        try:
            return _extract_json_from_text(stripped)
        except json.JSONDecodeError:
            pass

    # Fallback: the model may have placed JSON inside <think>…</think> or the
    # stripped result is empty — search the full raw text for a JSON object.
    if text.strip():
        try:
            return _extract_json_from_text(text)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No valid JSON object found in LLM response", stripped or text, 0)


class LLMClient:
    def __init__(self, config: BusinessAnalystConfig) -> None:
        self.config = config

    def generate(self, query: str, ticker: Optional[str], context: str, sentiment: Optional[SentimentSnapshot] = None) -> Dict[str, Any]:
        # Extract real chunk IDs from context so we can inject them into the prompt
        # to prevent the LLM from constructing plausible-looking but wrong IDs.
        real_chunk_ids = re.findall(r'chunk_id: (qdrant::[^\]]+|neo4j::[^\]]+)', context)
        prompt = self._build_prompt(query, ticker, context, sentiment, real_chunk_ids)
        payload = {
            "model": self.config.llm_model,
            "prompt": prompt,
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

        # If the model returned an empty response (can happen when think=False
        # conflicts with the model's generation mode), retry without that flag.
        if not content.strip():
            logger.warning("LLM returned empty response (think=False may have suppressed output) — retrying without think flag")
            retry_payload = {k: v for k, v in payload.items() if k != "think"}
            retry_resp = requests.post(
                f"{self.config.ollama_base_url}/api/generate",
                json=retry_payload,
                timeout=self.config.request_timeout,
            )
            retry_resp.raise_for_status()
            content = retry_resp.json().get("response", "")

        # First parse attempt
        try:
            return _extract_json(content)
        except json.JSONDecodeError:
            logger.warning(
                "LLM JSON decode failed on first attempt — retrying with format:json.\n"
                "Raw response snippet: %.300s",
                content,
            )

        # --- Retry with Ollama's native JSON mode ---
        # This forces the model to emit a valid JSON object regardless of its
        # chain-of-thought tendencies.  We send the same prompt but add a
        # concise reminder at the top so the model knows what schema to fill.
        json_mode_prompt = (
            "OUTPUT ONLY a valid JSON object. No markdown, no prose, no <think> blocks. "
            "Start your response with '{' and end with '}'.\n\n"
            + prompt
        )
        json_payload = {
            "model": self.config.llm_model,
            "prompt": json_mode_prompt,
            "temperature": 0.0,          # deterministic for the retry
            "num_predict": self.config.llm_max_tokens,
            "stream": False,
            "format": "json",            # Ollama native JSON mode
        }
        try:
            json_resp = requests.post(
                f"{self.config.ollama_base_url}/api/generate",
                json=json_payload,
                timeout=self.config.request_timeout,
            )
            json_resp.raise_for_status()
            retry_content = json_resp.json().get("response", "")
            result = _extract_json(retry_content)
            logger.info("LLM JSON retry (format:json) succeeded.")
            return result
        except json.JSONDecodeError as exc:
            logger.error(
                "LLM JSON decode failed after format:json retry: %s\nRaw response snippet: %.500s",
                exc,
                retry_content if "retry_content" in dir() else "",
            )
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
