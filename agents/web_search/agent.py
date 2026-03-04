# agents/web_search/agent.py
"""
Web Search Agent -- "The News Desk"
Agent 7 of 7 | FYP: The Agentic Investment Analyst

LangGraph-compatible node. Called by Supervisor via:
    from agents.web_search.agent import web_search_node

Implements:
  - Step-Back Prompting
  - 5A: Directed HyDE (intent-aware hypothetical document generation)
  - 5B: Iterative Fact-Checking (secondary Perplexity query for severe risk flags)
  - Perplexity Sonar API (primary)
  - Structured JSON output for Supervisor consumption
"""
import logging
import re
from datetime import datetime, timezone
from typing import Optional, TypedDict, List, Dict, Any

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


# ── 5A: Directed HyDE ──────────────────────────────────────────────────────────

# Intent keywords → (intent_label, HyDE template)
_INTENT_PATTERNS: List[tuple] = [
    (
        r"\b(earnings?|EPS|revenue beat|revenue miss|quarterly results?|Q[1-4] results?|guidance)\b",
        "earnings_call",
        (
            "Hypothetical context: A Tier-1 financial media outlet (Bloomberg / Reuters / WSJ) "
            "published an earnings call transcript summary for {ticker} on {date}. "
            "The report details Q{quarter} EPS actual vs. consensus estimate, management guidance "
            "for the next quarter, and analyst reactions. Key commentary covered: revenue drivers, "
            "margin outlook, and forward guidance."
        ),
    ),
    (
        r"\b(lawsuit|litigation|legal action|SEC probe|DOJ investigation|antitrust|settlement|class action)\b",
        "legal_brief",
        (
            "Hypothetical context: A Bloomberg Law / Reuters Legal brief published on {date} "
            "covers a significant legal development involving {ticker}. "
            "The article references court filings, regulatory letters, and quotes from "
            "plaintiff/defence counsel. Key issues: alleged violations, potential financial exposure, "
            "and timeline for resolution."
        ),
    ),
    (
        r"\b(merger|acquisition|M&A|takeover|buyout|deal|bid|acquires?|acquired)\b",
        "ma_announcement",
        (
            "Hypothetical context: A Reuters / FT breaking M&A announcement on {date} reports "
            "that {ticker} is involved in a major transaction. "
            "The article details deal structure (cash/stock), enterprise value, strategic rationale, "
            "expected synergies, regulatory hurdles, and analyst commentary on valuation."
        ),
    ),
    (
        r"\b(FDA|regulatory approval|approval|clinical trial|drug approval|PDUFA|NDA|BLA)\b",
        "regulatory_filing",
        (
            "Hypothetical context: A regulatory newswire dispatch published on {date} covers "
            "a key regulatory milestone for {ticker}. "
            "The report details agency decision, approval/rejection rationale, label restrictions, "
            "commercial launch timeline, and market size estimates from sell-side analysts."
        ),
    ),
    (
        r"\b(product launch|new product|release|unveil|announce)\b",
        "product_launch",
        (
            "Hypothetical context: A product launch press release covered by TechCrunch / "
            "The Verge / Bloomberg Technology on {date} describes a major new offering "
            "from {ticker}. Key details: product specifications, pricing, target market, "
            "differentiation vs. competitors, and analyst take on revenue potential."
        ),
    ),
]

_GENERIC_HYDE_TEMPLATE = (
    "Hypothetical context: A breaking news article from a Tier-1 financial source "
    "(Bloomberg, Reuters, or WSJ) published today ({date}) reports on {ticker} regarding: {query}. "
    "The article cites SEC filings, management commentary, and analyst reactions."
)


def _classify_intent(query: str) -> str:
    """Return intent label for the query (used for logging/transparency)."""
    q = query or ""
    for pattern, label, _ in _INTENT_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return label
    return "general"


def _build_directed_hyde(
    query: str,
    ticker: Optional[str],
    date: str,
) -> str:
    """5A: Generate an intent-aware HyDE context string.

    Classifies the query into one of 5 intents (earnings_call, legal_brief,
    ma_announcement, regulatory_filing, product_launch) and uses a tailored
    hypothetical document template for each.  Falls back to the generic HyDE
    template for unclassified queries.

    Returns the formatted HyDE context string.
    """
    ticker_str = ticker or "the target company"
    q = query or ""
    # Approximate current quarter from month
    from datetime import datetime as _dt
    month = _dt.now().month
    quarter = (month - 1) // 3 + 1

    for pattern, intent_label, template in _INTENT_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            logger.debug("[HyDE] Intent classified as '%s'", intent_label)
            return template.format(
                ticker=ticker_str,
                date=date,
                query=q,
                quarter=quarter,
            )

    logger.debug("[HyDE] Intent classified as 'general' — using generic template")
    return _GENERIC_HYDE_TEMPLATE.format(
        ticker=ticker_str,
        date=date,
        query=q,
    )


# ── 5B: Iterative Fact-Checking ────────────────────────────────────────────────

_TIER1_SOURCES = {
    "bloomberg.com", "reuters.com", "wsj.com", "ft.com", "sec.gov",
    "barrons.com", "cnbc.com", "marketwatch.com", "apnews.com",
}

_VERIFY_PROMPT_TEMPLATE = (
    "You are a financial fact-checker. Verify the following claim about {ticker}:\n\n"
    '"{claim}"\n\n'
    "Search for corroborating evidence from Tier-1 financial sources "
    "(Bloomberg, Reuters, WSJ, FT, SEC.gov). "
    "Return ONLY a JSON object:\n"
    '{{"verified": true/false, "corroborating_source": "<URL or null>", '
    '"source_tier": 1/2/3, "verification_note": "<1 sentence>"}}'
)


def _is_tier1_source(url: str) -> bool:
    """Check if a URL belongs to a known Tier-1 source."""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower().lstrip("www.")
        return any(t1 in domain for t1 in _TIER1_SOURCES)
    except Exception:
        return False


def _verify_claim(
    claim: str,
    ticker: Optional[str],
    model: str,
    recency: str,
) -> Dict[str, Any]:
    """5B: Issue a secondary Perplexity query to verify a single risk claim.

    Returns a dict with keys: verified (bool), corroborating_source, source_tier,
    verification_note.
    """
    ticker_str = ticker or "the company"
    prompt = _VERIFY_PROMPT_TEMPLATE.format(ticker=ticker_str, claim=claim)
    messages = [
        {"role": "system", "content": "You are a financial fact-checker. Be concise and objective."},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = perplexity_chat_completions(
            messages=messages,
            model=model,
            recency_filter=recency,
            temperature=0.0,
            max_tokens=512,
        )
        result = extract_json_from_response(resp.get("content", ""))
        if result is None:
            return {"verified": False, "corroborating_source": None, "source_tier": 3, "verification_note": "Could not parse verification response."}
        return result
    except Exception as exc:
        logger.warning("[FactCheck] Verification call failed: %s", exc)
        return {"verified": False, "corroborating_source": None, "source_tier": 3, "verification_note": f"Verification call failed: {exc}"}


def _iterative_fact_check(
    risk_flags: List[Dict],
    ticker: Optional[str],
    model: str,
    recency: str,
) -> List[Dict]:
    """5B: Verify the most severe risk flag via secondary Perplexity query.

    Strategy:
      - Find the single HIGH-severity risk flag (or first flag if none are HIGH).
      - Issue a secondary Perplexity query to corroborate it.
      - If not corroborated by a Tier-1 source, drop the flag from the list
        and log the decision.
      - Return the (possibly pruned) risk_flags list.
    """
    if not risk_flags:
        return risk_flags

    # Find most severe flag to verify
    high_flags = [f for f in risk_flags if str(f.get("severity", "")).upper() == "HIGH"]
    target_flag = high_flags[0] if high_flags else risk_flags[0]
    claim = str(target_flag.get("risk", ""))

    if not claim:
        return risk_flags

    logger.info("[FactCheck] Verifying claim: '%s'", claim[:80])
    verification = _verify_claim(claim, ticker, model, recency)

    verified = verification.get("verified", False)
    corr_source = verification.get("corroborating_source") or ""
    tier = int(verification.get("source_tier", 3))

    # Accept if: verified=True AND (source is Tier-1 domain OR tier reported as 1)
    is_corroborated = verified and (tier <= 1 or _is_tier1_source(corr_source))

    if not is_corroborated:
        logger.info(
            "[FactCheck] Dropping unverified claim (verified=%s tier=%s source=%s): '%s'",
            verified, tier, corr_source or "none", claim[:80],
        )
        risk_flags = [f for f in risk_flags if f is not target_flag]
    else:
        logger.info(
            "[FactCheck] Claim corroborated by Tier-%s source: %s",
            tier, corr_source,
        )
        # Annotate flag with verification info
        target_flag["fact_checked"] = True
        target_flag["verification_source"] = corr_source

    return risk_flags


# ── Core Agent ─────────────────────────────────────────────────────────────────
def run_web_search_agent(state: WebSearchInput) -> WebSearchOutput:
    """
    Main agent execution function.
    LangGraph calls this as a node in the Supervisor's parallel execution graph.

    Flow:
      1. Classify query intent (5A)
      2. Build directed HyDE context (5A) + schema instruction
      3. Call Perplexity Sonar API
      4. Parse + validate structured JSON output
      5. Iterative fact-check most severe risk flag (5B)
      6. Return WebSearchOutput to Supervisor
    """
    query         = state.get("query", "")
    ticker        = state.get("ticker")
    recency       = state.get("recency_filter") or "week"
    model         = state.get("model") or "sonar-pro"

    logger.info(
        f"[WebSearchAgent] query='{query}' ticker={ticker} "
        f"recency={recency} model={model}"
    )

    # ── Step 1 & 2: Build directed HyDE context (5A) ─────────────────────────
    ticker_str = ticker or "the target company"
    intent = _classify_intent(query)
    hyde_context = _build_directed_hyde(query, ticker, TODAY_UTC)
    logger.info("[WebSearchAgent] HyDE intent=%s", intent)

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
        f"[Directed HyDE Context — intent: {intent}]\n{hyde_context}\n\n"
        f"{schema_instruction}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    # ── Step 3: Call Perplexity ────────────────────────────────────────────────
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

    # ── Step 4: Parse structured JSON ─────────────────────────────────────────
    structured = extract_json_from_response(content)
    if structured is None:
        logger.error("[WebSearchAgent] JSON parse failure.")
        return _build_error_output(
            ticker, "JSON parse failure",
            citations=citations, confidence=0.3, parse_failure=True
        )

    # ── Step 5: Iterative fact-check (5B) ─────────────────────────────────────
    risk_flags = structured.get("unknown_risk_flags", [])
    if risk_flags:
        try:
            risk_flags = _iterative_fact_check(risk_flags, ticker, model, recency)
            structured["unknown_risk_flags"] = risk_flags
        except Exception as exc:
            logger.warning("[WebSearchAgent] Fact-check step failed (non-fatal): %s", exc)

    # ── Step 6: Fill safe defaults + attach citations ──────────────────────────
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
        f"confidence={structured.get('confidence')} "
        f"risk_flags_after_fc={len(structured.get('unknown_risk_flags', []))}"
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
