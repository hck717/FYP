"""Web Search Agent interface for Business Analyst CRAG escalation.

This path must call the real Web Search Agent (Perplexity-backed) directly.
No local stub fallback is used, so failures are surfaced explicitly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _transform_web_search_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """Transform WebSearchOutput to Business Analyst expected format.
    
    Business Analyst expects:
        - summary: A text summary (from sentiment_rationale or first breaking_news title)
        - key_risks: List of risk dicts with 'risk', 'source', 'severity' keys
        - sources: List of source URLs
    
    WebSearchOutput actually returns:
        - sentiment_rationale, unknown_risk_flags, breaking_news, 
          competitor_signals, raw_citations
    
    This transform maps between the two schemas while preserving raw fields
    for orchestrator citations.
    """
    summary = result.get("sentiment_rationale", "")
    if not summary and result.get("breaking_news"):
        first_news = result["breaking_news"][0]
        summary = first_news.get("title", "") or ""

    key_risks = []
    for flag in (result.get("unknown_risk_flags") or []):
        if isinstance(flag, dict):
            key_risks.append({
                "risk": flag.get("risk", ""),
                "source": flag.get("source_url", ""),
                "severity": flag.get("severity", "MEDIUM"),
            })

    sources = []
    for news in (result.get("breaking_news") or []):
        if isinstance(news, dict) and news.get("url"):
            sources.append(news["url"])
    for sig in (result.get("competitor_signals") or []):
        if isinstance(sig, dict) and sig.get("source_url"):
            sources.append(sig["source_url"])
    for url in (result.get("raw_citations") or []):
        if url and url not in sources:
            sources.append(url)

    return {
        "agent": "web_search",
        "ticker": result.get("ticker"),
        "summary": summary,
        "key_risks": key_risks,
        "sources": sources[:10],
        "breaking_news": result.get("breaking_news", []),
        "sentiment_signal": result.get("sentiment_signal"),
        "sentiment_rationale": result.get("sentiment_rationale"),
        "unknown_risk_flags": result.get("unknown_risk_flags", []),
        "competitor_signals": result.get("competitor_signals", []),
        "raw_citations": result.get("raw_citations", []),
    }


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

    Calls the real web_search agent directly.
    """
    try:
        # Attempt live import of the Web Search Agent
        from agents.web_search.agent import run_web_search_agent, WebSearchInput  # type: ignore[import]

        logger.info("Calling Web Search Agent for ticker=%s query=%r", ticker, query)
        agent_input = WebSearchInput(
            query=query,
            ticker=ticker,
            recency_filter="week",
            model=None,
        )
        result = run_web_search_agent(agent_input)
        if result:
            return _transform_web_search_output(dict(result))
        return {"summary": "", "key_risks": [], "sources": []}

    except ImportError as exc:
        logger.error("Web Search Agent import failed (hard failure): %s", exc)
        raise RuntimeError(
            "Web Search Agent is unavailable; direct Perplexity-backed search is required."
        ) from exc
    except Exception as exc:
        logger.error("Web Search Agent call failed (hard failure): %s", exc, exc_info=True)
        raise


__all__ = ["web_search_fallback"]
