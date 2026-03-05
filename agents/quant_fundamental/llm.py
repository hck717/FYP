"""LLM client for the Quantitative Fundamental agent.

The LLM is used ONLY to produce the `quantitative_summary` narrative field.
All numeric fields in the output are computed deterministically in Python from PostgreSQL.

The prompt targets plain-text output — no JSON wrapping. This avoids the fragile
JSON extraction path and produces more natural, readable narratives.

deepseek-r1 / Ollama notes
--------------------------
- Ollama >= 0.6.0 supports `"think": false` in the API payload to suppress the
  chain-of-thought reasoning block. We always send this flag.
- As a belt-and-suspenders measure, _clean_response() also strips any residual
  <think>...</think> tags from the response text.
- `request_timeout` defaults to None (no timeout) — let Ollama run to completion.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

import requests

from .config import QuantFundamentalConfig
from .prompts import build_system_prompt

logger = logging.getLogger(__name__)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (deepseek-r1 / ollama)."""
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
    # If the LLM returns JSON despite instructions, extract the summary value
    if cleaned.strip().startswith("{"):
        try:
            obj = json.loads(cleaned)
            # Try common summary keys
            for key in ("quantitative_summary", "summary", "narrative", "analysis"):
                val = obj.get(key, "")
                if val and isinstance(val, str) and len(val) > 20:
                    return val.strip()
        except (json.JSONDecodeError, AttributeError):
            pass
    return cleaned.strip()


class QuantLLMClient:
    """Calls Ollama to produce the quantitative_summary narrative."""

    def __init__(self, config: QuantFundamentalConfig) -> None:
        self.config = config

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
        prompt = self._build_prompt(factor_table, from_planner=from_planner)
        payload = {
            "model": self.config.llm_model,
            "prompt": prompt,
            "temperature": self.config.llm_temperature,
            "num_predict": self.config.llm_max_tokens,
            "stream": False,
            "think": False,  # Suppress <think> chain-of-thought (deepseek-r1 / Ollama)
        }
        try:
            resp = requests.post(
                f"{self.config.ollama_base_url}/api/generate",
                json=payload,
                timeout=self.config.request_timeout,
            )
            resp.raise_for_status()
            content = resp.json().get("response", "")
            if not content.strip():
                return "Quantitative summary unavailable (LLM returned empty response)."
            cleaned = _clean_response(content)
            if cleaned and len(cleaned) > 20:
                return cleaned
            return "Quantitative summary unavailable (LLM returned unusable content)."
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama not reachable — skipping LLM summary")
            return "Quantitative summary unavailable (LLM offline)."
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
        # Format the factor table as readable key=value lines to reduce token count
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

