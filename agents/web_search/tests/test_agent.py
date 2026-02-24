import json
from unittest.mock import patch

from agents.web_search.agent import run_web_search_agent

MOCK_OK = {
    "content": json.dumps({
        "agent": "web_search",
        "ticker": "NVDA",
        "query_date": "2026-02-24",
        "breaking_news": [],
        "sentiment_signal": "NEUTRAL",
        "sentiment_rationale": "No material updates found. https://sec.gov ... 2026-02-24",
        "unknown_risk_flags": [],
        "competitor_signals": [],
        "supervisor_escalation": {"action": "STANDALONE", "rationale": "No conflicts", "conflict_with_agent": None},
        "fallback_triggered": False,
        "confidence": 0.7
    }),
    "citations": ["https://www.sec.gov/"]
}

@patch("agents.web_search.agent.poe_chat_completions", return_value=MOCK_OK)
def test_agent_returns_structured_json(mock_call):
    out = run_web_search_agent({
        "query": "NVDA latest regulatory risk",
        "ticker": "NVDA",
        "recency_filter": "week",
        "model": "deepseek-v3.2-exp"
    })
    assert out["agent"] == "web_search"
    assert out["ticker"] == "NVDA"
    assert out["error"] is None

@patch("agents.web_search.agent.poe_chat_completions", side_effect=Exception("API down"))
def test_agent_fallback_on_api_error(mock_call):
    out = run_web_search_agent({
        "query": "TSLA news",
        "ticker": "TSLA",
        "recency_filter": "week",
        "model": "deepseek-v3.2-exp"
    })
    assert out["fallback_triggered"] is True
    assert out["error"] is not None
