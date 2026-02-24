# agents/web_search/tests/test_agent.py
"""
Unit tests for the Web Search Agent.
All Perplexity API calls are mocked -- no API key or network needed.

Run:
    pytest agents/web_search/tests/test_agent.py -v
"""
import json
from unittest.mock import patch

from agents.web_search.agent import run_web_search_agent
from agents.web_search.tools import extract_json_from_response


# ── Shared mock payload ────────────────────────────────────────────────────────
MOCK_OK_RESPONSE = {
    "content": json.dumps({
        "agent": "web_search",
        "ticker": "NVDA",
        "query_date": "2026-02-24",
        "breaking_news": [{
            "title": "NVDA Reports Record Q4 2026 Revenue",
            "url": "https://reuters.com/nvda-q4-2026",
            "published_date": "2026-02-24",
            "source_tier": 2,
            "relevance_score": 0.95,
            "verified": True,
        }],
        "sentiment_signal": "BULLISH",
        "sentiment_rationale": "Record revenue beat driven by datacenter demand. reuters.com 2026-02-24.",
        "unknown_risk_flags": [],
        "competitor_signals": [],
        "supervisor_escalation": {
            "action": "CONFIRMATORY_SIGNAL",
            "rationale": "Aligns with Quant agent bullish view.",
            "conflict_with_agent": None,
        },
        "fallback_triggered": False,
        "confidence": 0.92,
    }),
    "citations": ["https://reuters.com/nvda-q4-2026"],
}


# ── Test 1: Happy path ─────────────────────────────────────────────────────────
@patch("agents.web_search.agent.perplexity_chat_completions", return_value=MOCK_OK_RESPONSE)
def test_agent_returns_structured_json(mock_call):
    """Agent parses valid JSON response correctly."""
    result = run_web_search_agent({
        "query": "NVDA latest earnings results",
        "ticker": "NVDA",
        "recency_filter": "week",
        "model": "sonar-pro",
    })
    assert result["agent"] == "web_search"
    assert result["ticker"] == "NVDA"
    assert result["sentiment_signal"] == "BULLISH"
    assert result["fallback_triggered"] is False
    assert result["error"] is None
    assert len(result["breaking_news"]) == 1
    assert result["raw_citations"] == ["https://reuters.com/nvda-q4-2026"]


# ── Test 2: API failure ────────────────────────────────────────────────────────
@patch("agents.web_search.agent.perplexity_chat_completions",
       side_effect=Exception("API down"))
def test_agent_fallback_on_api_error(mock_call):
    """Agent degrades gracefully when Perplexity API throws."""
    result = run_web_search_agent({
        "query": "TSLA news",
        "ticker": "TSLA",
        "recency_filter": "week",
        "model": "sonar-pro",
    })
    assert result["fallback_triggered"] is True
    assert result["confidence"] == 0.2
    assert result["error"] == "API down"
    assert result["breaking_news"] == []


# ── Test 3: Invalid JSON from model ───────────────────────────────────────────
@patch("agents.web_search.agent.perplexity_chat_completions", return_value={
    "content": "Sorry, I cannot provide that information.",
    "citations": [],
})
def test_agent_handles_invalid_json(mock_call):
    """Agent handles model returning non-JSON output."""
    result = run_web_search_agent({
        "query": "AAPL news",
        "ticker": "AAPL",
        "recency_filter": "week",
        "model": "sonar-pro",
    })
    assert result["fallback_triggered"] is True
    assert result["error"] == "JSON parse failure"
    assert result["confidence"] == 0.3


# ── Test 4: extract_json_from_response util ────────────────────────────────────
def test_extract_raw_json():
    """Extracts plain JSON string."""
    raw = json.dumps({"agent": "web_search", "ticker": "AAPL"})
    result = extract_json_from_response(raw)
    assert result["ticker"] == "AAPL"


def test_extract_fenced_json():
    """Extracts JSON inside markdown code fence."""
    content = '```json\n{"agent": "web_search", "ticker": "MSFT"}\n```'
    result = extract_json_from_response(content)
    assert result["ticker"] == "MSFT"


def test_extract_invalid_returns_none():
    """Returns None for plain text (not JSON)."""
    result = extract_json_from_response("This is just plain text.")
    assert result is None
