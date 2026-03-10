"""LLM client for the Quantitative Fundamental agent.

Uses DeepSeek API (OpenAI-compatible) for generating quantitative summaries.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict

import requests

from .config import QuantFundamentalConfig
from .prompts import build_system_prompt

logger = logging.getLogger(__name__)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (deepseek-reasoner)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_markdown_fences(text: str) -> str:
    """Strip leading/trailing markdown code fences."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text.strip())
    return text.strip()


def _clean_response(text: str) -> str:
    """Clean raw LLM response to a plain narrative string."""
    cleaned = _strip_think_tags(text)
    cleaned = _strip_markdown_fences(cleaned)
    if cleaned.strip().startswith("{"):
        try:
            obj = json.loads(cleaned)
            for key in ("quantitative_summary", "summary", "narrative", "analysis"):
                val = obj.get(key, "")
                if val and isinstance(val, str) and len(val) > 20:
                    return val.strip()
        except (json.JSONDecodeError, AttributeError):
            pass
    return cleaned.strip()


class QuantLLMClient:
    """Calls DeepSeek API to produce the quantitative_summary narrative."""

    def __init__(self, config: QuantFundamentalConfig) -> None:
        self.config = config
        self.api_key = DEEPSEEK_API_KEY or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = DEEPSEEK_BASE_URL

    def generate_summary(
        self,
        factor_table: Dict[str, Any],
        *,
        from_planner: bool = False,
    ) -> str:
        """Generate the quantitative_summary string from the pre-computed factor table.

        Args:
            factor_table: The fully computed factor dict (value, quality, momentum, etc.)
            from_planner: If True, prepend the planner-override preamble to the system
                          prompt so the model ignores any task-routing instructions the
                          planner may have injected into the context.

        Returns:
            A 3-5 sentence narrative string, or a fallback message on LLM failure.
        """
        if not self.api_key:
            logger.warning("DeepSeek API key not set — skipping LLM summary")
            return "Quantitative summary unavailable (API key not configured)."

        prompt = self._build_prompt(factor_table, from_planner=from_planner)

        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.llm_model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "Write the quantitative narrative based on the factor table."},
                    ],
                    "temperature": self.config.llm_temperature,
                    "max_tokens": self.config.llm_max_tokens,
                },
                timeout=self.config.request_timeout or 120,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content.strip():
                return "Quantitative summary unavailable (LLM returned empty response)."
            cleaned = _clean_response(content)
            if cleaned and len(cleaned) > 20:
                return cleaned
            return "Quantitative summary unavailable (LLM returned unusable content)."
        except requests.exceptions.ConnectionError:
            logger.warning("DeepSeek API not reachable — skipping LLM summary")
            return "Quantitative summary unavailable (API offline)."
        except Exception as exc:
            logger.warning("LLM summary generation failed: %s", exc)
            return f"Quantitative summary unavailable ({type(exc).__name__})."

    def _build_prompt(
        self,
        factor_table: Dict[str, Any],
        *,
        from_planner: bool = False,
    ) -> str:
        """Build the prompt: system instructions + factor table."""
        system = build_system_prompt(from_planner=from_planner)
        lines = []
        for section, val in factor_table.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    if v is not None:
                        lines.append(f"  {k}: {v}")
                    else:
                        lines.append(f"  {k}: null")
            elif isinstance(val, list):
                if val:
                    lines.append(f"  {section}: {json.dumps(val, default=str)}")
                else:
                    lines.append(f"  {section}: (none)")
            else:
                lines.append(f"  {section}: {val}")

        factor_text = "\n".join(lines)
        return (
            f"{system}\n\n"
            f"--- FACTOR TABLE (do NOT recalculate these values) ---\n"
            f"{factor_text}\n"
            f"--- END FACTOR TABLE ---\n\n"
            f"Write your 8-12 sentence quantitative narrative now:"
        )


__all__ = ["QuantLLMClient"]

