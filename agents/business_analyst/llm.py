"""LLM interaction utilities for the Business Analyst agent.

LLM backend: DeepSeek API (deepseek-v4-pro) via the openai-compatible SDK.
Ollama is retained for embeddings only (nomic-embed-text:v1.5); it is NOT used
for any generation in this module.

One API call per agent run:
  - generate() -> single chat.completions.create() call; no retry loops.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import BusinessAnalystConfig
from .prompts import SYSTEM_PROMPT, JSON_SCHEMA_PROMPT
from .schema import SentimentSnapshot

logger = logging.getLogger(__name__)

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _sanitise_json_string_newlines(text: str) -> str:
    """Replace literal newlines inside JSON string values with \\n escape sequences.

    LLMs (especially deepseek-r1) frequently emit multi-paragraph string values
    with bare newline characters, which is invalid JSON (RFC 7159 §7 requires
    that control characters including U+000A be escaped as \\n inside strings).

    Strategy: scan the text tracking whether we are inside a quoted string.
    When inside a string, replace bare CR/LF with their JSON escape equivalents.
    """
    result: list[str] = []
    in_str = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            result.append(ch)
            continue
        if ch == "\\" and in_str:
            escape = True
            result.append(ch)
            continue
        if ch == '"':
            in_str = not in_str
            result.append(ch)
            continue
        if in_str and ch == "\n":
            result.append("\\n")
            continue
        if in_str and ch == "\r":
            result.append("\\r")
            continue
        result.append(ch)
    return "".join(result)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks produced by deepseek-r1.

    With deepseek-reasoner the reasoning is returned in a separate
    ``reasoning_content`` field, so think tags should not appear in the
    main ``content`` field.  This function is kept as a safety net.

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
    2. Sanitise literal newlines inside strings, then retry.
    3. Find first ``{`` … matching ``}`` balanced brace substring and parse that
       (with and without newline sanitisation).
    4. Raise JSONDecodeError if no valid JSON object found.
    """
    cleaned = _strip_markdown_fences(text)
    # Direct parse (ideal case)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Sanitise embedded newlines in strings and retry — LLMs often violate RFC 7159
    sanitised = _sanitise_json_string_newlines(cleaned)
    try:
        return json.loads(sanitised)
    except json.JSONDecodeError:
        pass
    # Find first balanced JSON object using brace matching (on sanitised text)
    for attempt_text in (sanitised, cleaned):
        start = attempt_text.find("{")
        if start != -1:
            depth = 0
            in_str = False
            escape = False
            for i, ch in enumerate(attempt_text[start:], start=start):
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
                            return json.loads(attempt_text[start : i + 1])
                        except json.JSONDecodeError:
                            break

            # Balanced-brace scan failed — likely truncated JSON.
            # Attempt repair: close all unclosed { and [ with matching } and ].
            fragment = attempt_text[start:]
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
        self._client = OpenAI(
            api_key=config.deepseek_api_key,
            base_url=_DEEPSEEK_BASE_URL,
            timeout=float(config.request_timeout) if config.request_timeout else 120.0,
        )

    def generate(self, query: str, ticker: Optional[str], context: str, sentiment: Optional[SentimentSnapshot] = None) -> Dict[str, Any]:
        """Generate a structured analysis JSON via a single DeepSeek API call.

        No retry loops, no format:json fallback — one call per agent run.
        The system message carries SYSTEM_PROMPT; the user message carries
        the context, chunk-ID guard, question, and JSON schema.
        """
        # Extract real chunk IDs from context so we can inject them into the prompt
        # to prevent the LLM from constructing plausible-looking but wrong IDs.
        real_chunk_ids = re.findall(
            r'chunk_id:\s*((?:neo4j|pgvec|qdrant)::[^\]\n]+|[A-Z]{1,6}::[^\]\n]+)',
            context,
        )
        # Deduplicate while preserving order
        seen: set = set()
        real_chunk_ids = [c for c in real_chunk_ids if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]

        system_msg, user_msg = self._build_messages(query, ticker, context, sentiment, real_chunk_ids)

        response = self._client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature,
            reasoning_effort="high",  # Enable thinking mode for v4-pro/reasoner models
        )

        message = response.choices[0].message
        content = message.content or ""

        # Log reasoning token count (deepseek-reasoner returns reasoning separately)
        reasoning = getattr(message, "reasoning_content", None)
        logger.debug(
            "DeepSeek generate stats: model=%s, input_tokens=%s, output_tokens=%s, reasoning_chars=%s",
            self.config.llm_model,
            getattr(response.usage, "prompt_tokens", "?"),
            getattr(response.usage, "completion_tokens", "?"),
            len(reasoning) if reasoning else 0,
        )
        if reasoning:
            logger.debug("DeepSeek reasoning snippet (first 500 chars): %.500s", reasoning)

        try:
            return _extract_json(content)
        except json.JSONDecodeError as exc:
            logger.error(
                "LLM JSON decode failed: %s\nRaw response snippet: %.500s",
                exc,
                content,
            )
            logger.debug("LLM full raw response:\n%s", content)
            raise

    def _build_messages(
        self,
        query: str,
        ticker: Optional[str],
        context: str,
        sentiment: Optional[SentimentSnapshot] = None,
        real_chunk_ids: Optional[list] = None,
    ) -> tuple[str, str]:
        """Return (system_message, user_message) for the DeepSeek chat API."""
        schema = JSON_SCHEMA_PROMPT.replace("{{ticker}}", ticker or "null").replace("{{today}}", self._today())

        # Pre-fill exact sentiment numbers so the LLM cannot hallucinate them.
        if sentiment:
            schema = schema.replace(
                '"bullish_pct": 0,\n    "bearish_pct": 0,\n    "neutral_pct": 0,',
                f'"bullish_pct": {sentiment.bullish_pct},\n    "bearish_pct": {sentiment.bearish_pct},\n    "neutral_pct": {sentiment.neutral_pct},',
            )
            schema = schema.replace(
                '"trend": "improving|deteriorating|stable|unknown",',
                f'"trend": "{sentiment.trend}",',
            )

        # Inject exact retrieved chunk IDs to prevent hallucinated citation slugs.
        if real_chunk_ids:
            ids_list = "\n".join(f"  - {cid}" for cid in real_chunk_ids)
            chunk_id_note = (
                f"\nIMPORTANT — Valid chunk IDs for citations (copy EXACTLY, do not invent others):\n{ids_list}\n"
            )
        else:
            chunk_id_note = ""

        user_msg = f"Context:\n{context}\n{chunk_id_note}\nQuestion: {query}\n\n{schema}"
        return SYSTEM_PROMPT, user_msg

    @staticmethod
    def _today() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


__all__ = ["LLMClient"]
