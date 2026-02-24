# agents/business_analyst/agent.py
"""
Business Analyst Agent -- "The Qualitative Analyst"
Agent 2 of 7 | FYP: The Agentic Investment Analyst

LangGraph-compatible node. Called by Supervisor via:
    from agents.business_analyst.agent import business_analyst_node

Implements:
  - CRAG (Corrective RAG): CORRECT / AMBIGUOUS / INCORRECT routing
  - Hybrid retrieval: Neo4j vector + Cypher graph + BM25 + Cross-Encoder rerank
  - PostgreSQL sentiment injection
  - Qdrant news recency layer
  - Structured JSON output for Supervisor consumption
  - Automatic Web Search Agent fallback on INCORRECT (<0.5 confidence)
"""
import logging
from datetime import datetime, timezone
from typing import Optional, TypedDict, List, Dict

import requests

from agents.business_analyst.prompts import (
    SYSTEM_PROMPT, ANALYSIS_PROMPT, QUERY_REWRITE_PROMPT
)
from agents.business_analyst.tools import (
    fetch_sentiment,
    fetch_company_profile,
    hybrid_retrieve,
    qdrant_news_search,
    crag_evaluate,
    extract_json_from_response,
)

logger = logging.getLogger(__name__)
TODAY_UTC = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# ── Config ─────────────────────────────────────────────────────────────────────
import os
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
BA_MODEL        = os.getenv("BUSINESS_ANALYST_MODEL", "deepseek-v3.2-exp")
MAX_RETRIES     = 2   # CRAG correction attempts


# ── LangGraph State Schema ─────────────────────────────────────────────────────
class BusinessAnalystInput(TypedDict):
    """Input state passed by the Supervisor agent."""
    task: str               # Natural language question
    ticker: Optional[str]   # Ticker resolved by Supervisor (e.g. "AAPL")


class BusinessAnalystOutput(TypedDict):
    """Output state returned to the Supervisor agent."""
    agent: str
    ticker: Optional[str]
    query_date: str
    company_overview: Dict
    sentiment: Dict
    competitive_moat: Dict
    key_risks: List[Dict]
    missing_context: List[Dict]
    crag_status: str
    confidence: float
    fallback_triggered: bool
    qualitative_summary: str
    error: Optional[str]


# ── Helpers ────────────────────────────────────────────────────────────────────
def _build_error_output(
    ticker: Optional[str],
    error_msg: str,
    crag_status: str = "INCORRECT",
    confidence: float = 0.1,
) -> BusinessAnalystOutput:
    """Graceful degraded output on any failure."""
    return BusinessAnalystOutput(
        agent="business_analyst",
        ticker=ticker,
        query_date=TODAY_UTC,
        company_overview={},
        sentiment={},
        competitive_moat={"rating": None, "key_strengths": [], "sources": []},
        key_risks=[],
        missing_context=[{"gap": f"Agent error: {error_msg}", "severity": "HIGH"}],
        crag_status=crag_status,
        confidence=confidence,
        fallback_triggered=True,
        qualitative_summary="INSUFFICIENT_DATA: agent error",
        error=error_msg,
    )


def _call_ollama(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """
    Call local Ollama with DeepSeek-V3.2-Exp.
    Returns raw content string.
    """
    payload = {
        "model": BA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 2000,
            "num_ctx": 8192,
        },
    }
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    # Strip DeepSeek <think>...</think> reasoning tags
    import re
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return content


def _rewrite_query(original: str, ticker: str, context_hint: str) -> str:
    """Use LLM to rewrite an AMBIGUOUS query for better retrieval."""
    prompt = QUERY_REWRITE_PROMPT.format(
        original_query=original,
        ticker=ticker,
        context_hint=context_hint[:400],
    )
    try:
        rewritten = _call_ollama(
            prompt,
            system="You are a search query optimisation expert. Return only the rewritten query."
        )
        rewritten = rewritten.split("\n")[0].strip()
        return rewritten if len(rewritten) > 5 else original
    except Exception as e:
        logger.warning(f"[BA] Query rewrite failed: {e}")
        return original


def _format_context(docs: List[str], label: str = "") -> str:
    """Format retrieved documents for prompt injection."""
    if not docs:
        return "No context available."
    lines = []
    for i, doc in enumerate(docs, 1):
        lines.append(f"[{label}{i}] {doc[:600]}")
    return "\n\n".join(lines)


# ── Core Agent ─────────────────────────────────────────────────────────────────
def run_business_analyst_agent(
    state: BusinessAnalystInput,
) -> BusinessAnalystOutput:
    """
    Main CRAG pipeline.

    Flow:
      1. Fetch sentiment (PostgreSQL) + company profile (Neo4j)
      2. Hybrid retrieval: Neo4j vector + Cypher + BM25 + Cross-Encoder
      3. CRAG evaluation: CORRECT / AMBIGUOUS / INCORRECT
         - CORRECT  (>0.7) → generate directly
         - AMBIGUOUS (0.5-0.7) → rewrite query → retry retrieval
         - INCORRECT (<0.5) → return fallback_triggered=True (Supervisor calls Web Search)
      4. Generate structured JSON via Ollama DeepSeek-V3.2-Exp
      5. Parse + validate output
    """
    task   = state.get("task", "")
    ticker = state.get("ticker") or ""

    logger.info(f"[BA] task='{task}' ticker={ticker}")

    # ── Step 1: Pre-fetch context ──────────────────────────────────────────────
    sentiment_data  = fetch_sentiment(ticker) if ticker else {}
    company_profile = fetch_company_profile(ticker) if ticker else {}

    sentiment_context = (
        f"Bullish: {sentiment_data.get('bullish_pct')}% | "
        f"Bearish: {sentiment_data.get('bearish_pct')}% | "
        f"Neutral: {sentiment_data.get('neutral_pct')}% | "
        f"Trend: {sentiment_data.get('trend')}"
        if sentiment_data else "Sentiment data not available."
    )

    # ── Step 2: Initial hybrid retrieval ──────────────────────────────────────
    docs, score = hybrid_retrieve(task, ticker)
    crag_status = crag_evaluate(score)
    logger.info(f"[BA] Initial CRAG: status={crag_status} score={score:.3f}")

    # ── Step 3: CRAG correction loop ──────────────────────────────────────────
    active_query = task

    if crag_status == "AMBIGUOUS":
        context_hint = docs[0] if docs else ""
        rewritten    = _rewrite_query(task, ticker, context_hint)
        logger.info(f"[BA] AMBIGUOUS → rewritten query: '{rewritten}'")

        docs_retry, score_retry = hybrid_retrieve(rewritten, ticker)
        crag_retry = crag_evaluate(score_retry)
        logger.info(f"[BA] Retry CRAG: status={crag_retry} score={score_retry:.3f}")

        if score_retry >= score:
            docs, score, crag_status, active_query = (
                docs_retry, score_retry, crag_retry, rewritten
            )

    if crag_status == "INCORRECT":
        logger.warning(f"[BA] INCORRECT confidence ({score:.3f}) — returning fallback signal")
        return BusinessAnalystOutput(
            agent="business_analyst",
            ticker=ticker or None,
            query_date=TODAY_UTC,
            company_overview=company_profile,
            sentiment=sentiment_data,
            competitive_moat={"rating": None, "key_strengths": [], "sources": []},
            key_risks=[],
            missing_context=[{
                "gap": (
                    f"Graph context insufficient for '{ticker}' "
                    f"(CRAG score {score:.2f} < 0.5). "
                    "Ticker may not be ingested yet — Web Search fallback recommended."
                ),
                "severity": "HIGH",
            }],
            crag_status="INCORRECT",
            confidence=round(score, 3),
            fallback_triggered=True,
            qualitative_summary="INSUFFICIENT_DATA: graph context below confidence threshold.",
            error=None,
        )

    # ── Step 4: Assemble prompt context ───────────────────────────────────────
    news_docs     = qdrant_news_search(active_query, ticker, k=5)
    graph_context = _format_context(docs, label="GRAPH")
    news_context  = _format_context(news_docs, label="NEWS")

    analysis_prompt = ANALYSIS_PROMPT.format(
        query=active_query,
        ticker=ticker,
        sentiment_context=sentiment_context,
        graph_context=graph_context,
        news_context=news_context,
    )

    # ── Step 5: Generate via Ollama ────────────────────────────────────────────
    try:
        raw_response = _call_ollama(analysis_prompt)
    except Exception as e:
        logger.error(f"[BA] Ollama call failed: {e}")
        return _build_error_output(ticker, str(e))

    # ── Step 6: Parse structured JSON ─────────────────────────────────────────
    structured = extract_json_from_response(raw_response)
    if structured is None:
        logger.error("[BA] JSON parse failure")
        return _build_error_output(
            ticker, "JSON parse failure — model did not return valid JSON",
            crag_status=crag_status, confidence=score
        )

    # ── Step 7: Fill safe defaults ─────────────────────────────────────────────
    structured.setdefault("agent",         "business_analyst")
    structured.setdefault("ticker",         ticker or None)
    structured.setdefault("query_date",     TODAY_UTC)
    structured.setdefault("company_overview", company_profile)
    structured.setdefault("sentiment",       sentiment_data)
    structured.setdefault("competitive_moat", {"rating": None, "key_strengths": [], "sources": []})
    structured.setdefault("key_risks",       [])
    structured.setdefault("missing_context", [])
    structured.setdefault("fallback_triggered", False)
    structured["crag_status"] = crag_status
    structured["confidence"]  = round(score, 3)
    structured["error"]       = None

    logger.info(
        f"[BA] Done. crag={crag_status} confidence={score:.3f} "
        f"risks={len(structured.get('key_risks', []))}"
    )
    return BusinessAnalystOutput(**structured)


# ── LangGraph Node Entrypoint ──────────────────────────────────────────────────
def business_analyst_node(state: dict) -> dict:
    """
    LangGraph node wrapper.
    Supervisor registers this as:
        graph.add_node("business_analyst", business_analyst_node)
    """
    agent_input = BusinessAnalystInput(
        task=state.get("query", ""),
        ticker=state.get("ticker"),
    )
    out = run_business_analyst_agent(agent_input)
    return {"business_analyst_output": dict(out)}


# ── CLI quick-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json, sys
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Business Analyst Agent quick test")
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol")
    parser.add_argument("--task",   default="What is the competitive moat and key risks?",
                        help="Analysis task")
    args = parser.parse_args()

    result = run_business_analyst_agent(
        BusinessAnalystInput(task=args.task, ticker=args.ticker)
    )
    print(json.dumps(result, indent=2))
