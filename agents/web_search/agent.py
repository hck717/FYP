import logging
from datetime import datetime, timezone
from typing import Optional, TypedDict, List, Dict

from agents.web_search.prompts import SYSTEM_PROMPT
from agents.web_search.tools import poe_chat_completions, extract_json_from_response

logger = logging.getLogger(__name__)
TODAY_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%d")


class WebSearchInput(TypedDict):
    query: str
    ticker: Optional[str]
    recency_filter: Optional[str]   # reserved for future (if Poe supports)
    model: Optional[str]


class WebSearchOutput(TypedDict):
    agent: str
    ticker: Optional[str]
    query_date: str
    breaking_news: List[Dict]
    sentiment_signal: str
    sentiment_rationale: str
    unknown_risk_flags: List[Dict]
    competitor_signals: List[Dict]
    supervisor_escalation: Dict
    fallback_triggered: bool
    confidence: float
    raw_citations: List[str]
    error: Optional[str]


def run_web_search_agent(state: WebSearchInput) -> WebSearchOutput:
    query = state.get("query", "")
    ticker = state.get("ticker")
    model = state.get("model") or "deepseek-v3.2-exp"

    # Enforce strict JSON output in the user instruction (so model cannot drift)
    schema_instruction = f"""
Return ONLY a valid JSON object matching this schema (no markdown, no commentary):
{{
  "agent": "web_search",
  "ticker": "{ticker}" or null,
  "query_date": "{TODAY_UTC}",
  "breaking_news": [{{"title":"...","url":"...","published_date":"...","source_tier":1,"relevance_score":0.0,"verified":true}}],
  "sentiment_signal": "BULLISH|BEARISH|NEUTRAL|MIXED",
  "sentiment_rationale": "1 sentence with URL + date",
  "unknown_risk_flags": [{{"risk":"...","source_url":"...","severity":"HIGH|MEDIUM|LOW"}}],
  "competitor_signals": [{{"company":"...","signal":"...","source_url":"..."}}],
  "supervisor_escalation": {{"action":"CONFLICT_SIGNAL|CONFIRMATORY_SIGNAL|STANDALONE","rationale":"...","conflict_with_agent":null}},
  "fallback_triggered": false,
  "confidence": 0.0
}}
"""

    user_msg = f"Ticker: {ticker}\nQuestion: {query}\n\n{schema_instruction}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        resp = poe_chat_completions(messages=messages, model=model, temperature=0.1, max_tokens=4096)
        content = resp.get("content", "")
        citations = resp.get("citations", [])
    except Exception as e:
        return WebSearchOutput(
            agent="web_search",
            ticker=ticker,
            query_date=TODAY_UTC,
            breaking_news=[],
            sentiment_signal="NEUTRAL",
            sentiment_rationale="INSUFFICIENT_DATA: Poe API error",
            unknown_risk_flags=[],
            competitor_signals=[],
            supervisor_escalation={"action": "STANDALONE", "rationale": "API failure", "conflict_with_agent": None},
            fallback_triggered=True,
            confidence=0.2,
            raw_citations=[],
            error=str(e),
        )

    structured = extract_json_from_response(content)
    if structured is None:
        return WebSearchOutput(
            agent="web_search",
            ticker=ticker,
            query_date=TODAY_UTC,
            breaking_news=[],
            sentiment_signal="NEUTRAL",
            sentiment_rationale="INSUFFICIENT_DATA: model did not return valid JSON",
            unknown_risk_flags=[],
            competitor_signals=[],
            supervisor_escalation={"action": "STANDALONE", "rationale": "JSON parse failure", "conflict_with_agent": None},
            fallback_triggered=True,
            confidence=0.3,
            raw_citations=citations,
            error="JSON parse failure",
        )

    structured.setdefault("agent", "web_search")
    structured.setdefault("ticker", ticker)
    structured.setdefault("query_date", TODAY_UTC)
    structured.setdefault("breaking_news", [])
    structured.setdefault("unknown_risk_flags", [])
    structured.setdefault("competitor_signals", [])
    structured.setdefault("fallback_triggered", False)
    structured.setdefault("confidence", 0.7)

    return WebSearchOutput(
        **structured,
        raw_citations=citations,
        error=None
    )


def web_search_node(state: dict) -> dict:
    """
    LangGraph node wrapper: merges output into shared state.
    """
    agent_input: WebSearchInput = {
        "query": state.get("query", ""),
        "ticker": state.get("ticker"),
        "recency_filter": state.get("recency_filter"),
        "model": state.get("web_search_model"),
    }
    out = run_web_search_agent(agent_input)
    return {"web_search_output": dict(out)}
