# agents/web_search/agent.py
"""
Web Search Agent -- "The News Desk"
Agent 7 of 7 | FYP: The Agentic Investment Analyst

LangGraph-compatible node. Called by Supervisor via:
    from agents.web_search.agent import web_search_node

Implements:
  - Step-Back Prompting
  - HyDE (Hypothetical Document Embeddings) query enrichment
  - Perplexity Sonar API (primary)
  - Structured JSON output for Supervisor consumption
"""
import logging
from datetime import datetime, timezone
from typing import Optional, TypedDict, List, Dict

from agents.web_search.prompts import SYSTEM_PROMPT
from agents.web_search.tools import perplexity_chat_completions, extract_json_from_response

logger = logging.getLogger(__name__)
TODAY_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ── LangGraph State Schema ─────────────────────────────────────────────────────
class WebSearchInput(TypedDict):
    """Input state passed by the Supervisor agent."""
    query: str                        # User's original question
    ticker: Optional[str]             # Ticker symbol resolved by Supervisor
    recency_filter: Optional[str]     # "day" | "week" | "month" -- default: "week"
    model: Optional[str]              # Override model -- default: sonar-pro


class WebSearchOutput(TypedDict):
    """Output state returned to the Supervisor agent."""
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


# ── Helpers ────────────────────────────────────────────────────────────────────
def _build_error_output(
    ticker: Optional[str],
    error_msg: str,
    citations: Optional[List[str]] = None,
    confidence: float = 0.2,
    parse_failure: bool = False,
) -> WebSearchOutput:
    """Builds a graceful degraded output on any failure."""
    rationale = (
        "INSUFFICIENT_DATA: model did not return valid JSON"
        if parse_failure
        else "INSUFFICIENT_DATA: Perplexity API error"
    )
    return WebSearchOutput(
        agent="web_search",
        ticker=ticker,
        query_date=TODAY_UTC,
        breaking_news=[],
        sentiment_signal="NEUTRAL",
        sentiment_rationale=rationale,
        unknown_risk_flags=[],
        competitor_signals=[],
        supervisor_escalation={
            "action": "STANDALONE",
            "rationale": "API or parse failure -- output not reliable",
            "conflict_with_agent": None,
        },
        fallback_triggered=True,
        confidence=confidence,
        raw_citations=citations or [],
        error=error_msg,
    )


# ── Core Agent ─────────────────────────────────────────────────────────────────
def run_web_search_agent(state: WebSearchInput) -> WebSearchOutput:
    """
    Main agent execution function.
    LangGraph calls this as a node in the Supervisor's parallel execution graph.

    Flow:
      1. Build enriched user message with HyDE context + schema instruction
      2. Call Perplexity Sonar API
      3. Parse + validate structured JSON output
      4. Return WebSearchOutput to Supervisor
    """
    query         = state.get("query", "")
    ticker        = state.get("ticker")
    recency       = state.get("recency_filter") or "week"
    model         = state.get("model") or "sonar-pro"

    logger.info(
        f"[WebSearchAgent] query='{query}' ticker={ticker} "
        f"recency={recency} model={model}"
    )

    # ── Step 1: Build enriched user message ───────────────────────────────────
    # HyDE: hypothetical ideal article primes the model toward the right
    # semantic space before the real search query.
    ticker_str = ticker or "the target company"
    hyde_context = (
        f"Hypothetical context: A breaking news article from a Tier-1 financial source "
        f"published today ({TODAY_UTC}) reports on {ticker_str} regarding: {query}. "
        f"The article cites SEC filings, management commentary, and analyst reactions."
    )

    schema_instruction = f"""
Return ONLY a valid JSON object (no markdown, no commentary outside the JSON):
{{
  "agent": "web_search",
  "ticker": "{ticker}",
  "query_date": "{TODAY_UTC}",
  "breaking_news": [
    {{"title": "...", "url": "...", "published_date": "YYYY-MM-DD",
      "source_tier": 1, "relevance_score": 0.0, "verified": true}}
  ],
  "sentiment_signal": "BULLISH|BEARISH|NEUTRAL|MIXED",
  "sentiment_rationale": "1 sentence with source URL + date",
  "unknown_risk_flags": [
    {{"risk": "...", "source_url": "...", "severity": "HIGH|MEDIUM|LOW"}}
  ],
  "competitor_signals": [
    {{"company": "...", "signal": "...", "source_url": "..."}}
  ],
  "supervisor_escalation": {{
    "action": "CONFLICT_SIGNAL|CONFIRMATORY_SIGNAL|STANDALONE",
    "rationale": "...",
    "conflict_with_agent": null
  }},
  "fallback_triggered": false,
  "confidence": 0.0
}}
"""

    user_msg = (
        f"Target ticker: {ticker or 'N/A'}\n"
        f"Research question: {query}\n\n"
        f"[HyDE Context]\n{hyde_context}\n\n"
        f"{schema_instruction}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    # ── Step 2: Call Perplexity ────────────────────────────────────────────────
    try:
        resp = perplexity_chat_completions(
            messages=messages,
            model=model,
            recency_filter=recency,
            temperature=0.1,
            max_tokens=4096,
        )
        content   = resp.get("content", "")
        citations = resp.get("citations", [])
    except Exception as e:
        logger.error(f"[WebSearchAgent] Perplexity API failed: {e}")
        return _build_error_output(ticker, str(e))

    # ── Step 3: Parse structured JSON ─────────────────────────────────────────
    structured = extract_json_from_response(content)
    if structured is None:
        logger.error("[WebSearchAgent] JSON parse failure.")
        return _build_error_output(
            ticker, "JSON parse failure",
            citations=citations, confidence=0.3, parse_failure=True
        )

    # ── Step 4: Fill safe defaults + attach citations ──────────────────────────
    structured.setdefault("agent", "web_search")
    structured.setdefault("ticker", ticker)
    structured.setdefault("query_date", TODAY_UTC)
    structured.setdefault("breaking_news", [])
    structured.setdefault("unknown_risk_flags", [])
    structured.setdefault("competitor_signals", [])
    structured.setdefault("fallback_triggered", False)
    structured.setdefault("confidence", 0.7)
    structured.setdefault("sentiment_rationale", "")
    structured["raw_citations"] = citations
    structured["error"] = None

    logger.info(
        f"[WebSearchAgent] Done. sentiment={structured.get('sentiment_signal')} "
        f"news_count={len(structured.get('breaking_news', []))} "
        f"confidence={structured.get('confidence')}"
    )

    return WebSearchOutput(**structured)


# ── LangGraph Node Entrypoint ──────────────────────────────────────────────────
def web_search_node(state: dict) -> dict:
    """
    LangGraph node wrapper.
    Supervisor registers this as: graph.add_node("web_search", web_search_node)
    """
    agent_input = WebSearchInput(
        query=state.get("query", ""),
        ticker=state.get("ticker"),
        recency_filter=state.get("recency_filter", "week"),
        model=state.get("web_search_model", "sonar-pro"),
    )
    out = run_web_search_agent(agent_input)
    return {"web_search_output": dict(out)}
