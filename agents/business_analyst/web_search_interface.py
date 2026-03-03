"""Web Search Agent interface for the Business Analyst CRAG fallback.

When CRAG confidence < 0.5 (INCORRECT), the Business Analyst agent calls this
module to trigger the Web Search Agent. This module provides a clean boundary:

- If the Web Search Agent exists at agents/web_search, it is imported and called.
- If it does not yet exist (pre-Week 7 in the roadmap), a graceful stub is returned
  so the Business Analyst pipeline can still complete with a partial result.

The Supervisor/Synthesizer will see `fallback_triggered: true` in the output and
know to supplement this agent's response with live web data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def web_search_fallback(
    query: str,
    ticker: Optional[str],
    config: Any = None,
) -> Dict[str, Any]:
    """Call the Web Search Agent and return its result dict.

    Returns a dict with at minimum:
        {
            "summary": str,           # 1-3 sentence summary of findings
            "key_risks": [...],        # list of risk dicts (may be empty)
            "sources": [...],          # list of source URLs / article titles
            "agent": "web_search",
        }

    Falls back to a stub result if the web_search agent module is unavailable.
    """
    try:
        # Attempt live import of the Web Search Agent (available from Week 7 onward)
        from agents.web_search.agent import run as web_run  # type: ignore[import]

        logger.info("Calling Web Search Agent for ticker=%s query=%r", ticker, query)
        result = web_run(task=query, ticker=ticker, config=config)
        return result if isinstance(result, dict) else {"summary": str(result), "key_risks": [], "sources": []}

    except ImportError:
        # Web Search Agent not yet implemented — return a structured stub
        logger.warning(
            "Web Search Agent not available (agents/web_search/agent.py missing). "
            "Returning stub fallback for ticker=%s.",
            ticker,
        )
        return _stub_result(query, ticker)
    except Exception as exc:
        logger.error("Web Search Agent call failed: %s", exc, exc_info=True)
        return _stub_result(query, ticker, error=str(exc))


def _stub_result(
    query: str,
    ticker: Optional[str],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Minimal stub returned when the Web Search Agent is unavailable."""
    label = ticker or "the requested company"
    reason = f" (error: {error})" if error else " (agent not yet implemented)"
    return {
        "agent": "web_search",
        "ticker": ticker,
        "summary": (
            f"INSUFFICIENT_DATA: Web Search Agent fallback was triggered for {label}{reason}. "
            f"Query: {query!r}. Live web data unavailable — please check manually."
        ),
        "key_risks": [],
        "sources": [],
        "fallback_stub": True,
    }


__all__ = ["web_search_fallback"]
