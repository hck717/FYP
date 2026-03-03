"""LLM client for the orchestration layer.

Used by:
  - planner_node: parse user query → structured plan + tool selection
  - summarizer_node: synthesise all agent outputs → final narrative with citations

Planner uses llama3.2:latest (fast, ~3-5s).
Summarizer uses deepseek-r1:8b for deeper reasoning on the final report.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_PLANNER_MODEL    = os.getenv(
    "ORCHESTRATION_PLANNER_MODEL",
    # Use llama3.2:latest for planning — it is fast (~3s) and reliable.
    # deepseek-r1:8b is reserved for deep analysis nodes but times out when used
    # for the lightweight JSON planning task.
    "llama3.2:latest",
)
_SUMMARIZER_MODEL = os.getenv(
    "ORCHESTRATION_SUMMARIZER_MODEL",
    # deepseek-r1:8b produces substantially deeper, more analytical prose than llama3.2:latest
    # (3B params). Timeout raised to 1200 s to accommodate ~600 s generation time at 8k tokens.
    "deepseek-r1:8b",
)
_REQUEST_TIMEOUT_ENV = os.getenv("ORCHESTRATION_LLM_TIMEOUT", "").strip()
_REQUEST_TIMEOUT: Optional[int] = int(_REQUEST_TIMEOUT_ENV) if _REQUEST_TIMEOUT_ENV else None  # no cap — planner is fast but shouldn't hang the pipeline if Ollama is busy
_SUMMARIZER_TIMEOUT_ENV = os.getenv("ORCHESTRATION_SUMMARIZER_TIMEOUT", "").strip()
_SUMMARIZER_TIMEOUT: Optional[int] = int(_SUMMARIZER_TIMEOUT_ENV) if _SUMMARIZER_TIMEOUT_ENV else None  # no hard cap — let the report run to completion


# ── helpers ───────────────────────────────────────────────────────────────────

def _strip_think(text: str) -> str:
    """Remove deepseek-r1 chain-of-thought blocks in all observed forms.

    deepseek-r1 via Ollama can emit thinking content in three patterns:

    Pattern A — balanced tags:
        <think>...reasoning...</think>
        ...actual response...
        → Strip everything between the tags; keep only the response.

    Pattern B — orphan closing tag (no opening <think>):
        ...actual response...  </think>
        ...freeform garbage the model added after "finishing" its think...
        → The good content is BEFORE </think>; truncate at the tag.
        This happens when Ollama's think=False flag is partially respected:
        the opening tag is suppressed but the model still emits a closing
        </think> when it transitions from planning to final output.

    Pattern C — opening tag only (no closing </think>):
        <think>...reasoning...
        ...actual response...
        → Strip from <think> to end-of-string if no closing tag is found.
    """
    # Pattern A: balanced <think>…</think> — strip the block entirely
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Pattern B: orphan </think> with no matching opening tag.
    # Everything BEFORE the orphan </think> is the real output.
    # Everything AFTER is the model's freeform post-think generation — discard it.
    if "</think>" in text:
        text = text[:text.index("</think>")]

    # Pattern C: orphan <think> with no closing tag — strip from <think> onwards.
    if "<think>" in text:
        text = text[:text.index("<think>")]

    return text.strip()


def _strip_fences(text: str) -> str:
    """Strip ```json … ``` markdown fences."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text.strip())
    return text.strip()


def _extract_json(text: str) -> Dict[str, Any]:
    """Robustly extract first top-level JSON object from LLM text."""
    cleaned = _strip_think(text)
    cleaned = _strip_fences(cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start != -1:
        depth, in_str, escape = 0, False, False
        for i, ch in enumerate(cleaned[start:], start=start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start: i + 1])
                    except json.JSONDecodeError:
                        break
    raise json.JSONDecodeError("No valid JSON in LLM response", cleaned, 0)


def _ollama_generate(
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    timeout: Optional[int] = None,
    num_ctx: Optional[int] = None,
) -> str:
    """Call Ollama /api/generate and return the raw response text."""
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "num_predict": max_tokens,
        "stream": False,
        "think": False,
    }
    if num_ctx is not None:
        payload["options"] = {"num_ctx": num_ctx}
    _timeout = timeout if timeout is not None else _REQUEST_TIMEOUT
    # _timeout=None → requests uses no socket timeout (runs until Ollama responds)
    try:
        resp = requests.post(
            f"{_OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=_timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        logger.error("Ollama not reachable at %s", _OLLAMA_BASE_URL)
        raise
    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out after %ss", _timeout)
        raise


# ── Planner ───────────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """You are a financial research planning agent for a multi-agent equity research system.
Your job is to analyse the user's question, identify ALL ticker symbols, and produce a structured JSON plan
that routes the query to the correct domain agents.

=== AVAILABLE AGENTS AND THEIR CAPABILITIES ===

AGENT: business_analyst
README SUMMARY: The qualitative intelligence layer. Uses Corrective RAG (CRAG) over a Qdrant vector
store (~2,390 financial documents) + Neo4j knowledge graph + PostgreSQL sentiment data. Handles:
  - Competitive moat analysis (wide/narrow/none rating with cited sources)
  - Business model and primary revenue composition breakdown
  - Strategic positioning and 2-3 year business implication
  - Key risk factors with severity (HIGH/MEDIUM/LOW) and observed mitigations
  - Sentiment trend analysis (bullish/bearish/neutral % from PostgreSQL)
  - CRAG evaluation: CORRECT (>0.55), AMBIGUOUS (0.35–0.55), INCORRECT (<0.35) with fallback
SUPPORTED TICKERS: AAPL, MSFT, GOOGL, TSLA, NVDA (Qdrant vectors available)
DO NOT USE FOR: Real-time news, financial ratios, DCF, macro rates

AGENT: quant_fundamental
README SUMMARY: The numerical integrity layer. Deterministic 8-node pipeline, all math in Python from
PostgreSQL. NO RAG, NO hallucinated numbers. Handles:
  - Value factors: P/E (trailing), EV/EBITDA, P/FCF, EV/Revenue
  - Quality factors: ROE, ROIC, Piotroski F-Score (0–9), Beneish M-Score
  - Momentum/Risk: Beta (60-day rolling), Sharpe Ratio (12m), 12-month price return
  - Key metrics: Gross Margin, EBIT Margin, FCF Conversion, Current Ratio, D/E
  - Anomaly detection: Z-score flagging vs. 3-year rolling baseline
  - Data quality validation (field presence + range checks)
SUPPORTED TICKERS: AAPL, MSFT, GOOGL, TSLA, NVDA (PostgreSQL data available)
MULTI-TICKER: Fully supports comparison queries — returns one result dict per ticker
DO NOT USE FOR: Qualitative moat, news, DCF modelling, macro

AGENT: web_search
README SUMMARY: The only agent with real-time web access. Uses Perplexity Sonar API. Handles:
  - Breaking earnings announcements not yet in local DB
  - Regulatory investigations, fines, sanctions
  - CEO/CFO/Board changes, M&A announcements
  - Litigation, patent disputes, ESG events
  - Competitor earnings surprises (indirect signals)
SUPPORTED TICKERS: Any (live web search, not DB-dependent)
DO NOT USE FOR: Historical fundamentals, valuation multiples, qualitative strategy analysis

AGENT: financial_modelling
README SUMMARY: The deterministic quantitative valuation engine. All numbers computed in Python
from PostgreSQL — LLM is used ONLY for narrative summary. NO hallucinated numbers. Handles:
  - DCF Valuation: 5-year FCF projection + Terminal Value (Gordon Growth Model), Bear/Base/Bull
    scenarios (Bear: rev growth -5%, EBIT margin 10%, WACC 12%; Base: +8%, 18%, 10%; Bull: +20%,
    25%, 9%), scenario probability weights (Bear=0.25, Base=0.55, Bull=0.20), implied price range
    (low/mid/high), 4×4 WACC×terminal-growth sensitivity matrix
  - WACC: CAPM-based (Rf=10Y Treasury, β=60-day rolling vs S&P500, Rm-Rf=Market Risk Premium,
    Rd=Interest Expense/Total Debt, T=effective tax rate), E/V × Re + D/V × Rd × (1-T)
  - Comparable Company Analysis (Comps): EV/EBITDA, P/E trailing+forward, P/S, EV/Revenue vs
    Neo4j peer group (top 5 COMPETES_WITH edges); outputs vs_sector_avg as premium/discount %
  - Technical Analysis: SMA 20/50/200, EMA 12/26, RSI(14), MACD+Signal, Bollinger Bands (20-period
    2σ), ATR(14), HV30, Stochastic %K/%D, 52-week high/low, support/resistance levels, trend signal
  - Earnings Analysis: EPS actual vs estimate, surprise %, beat streak, miss streak, next earnings date
  - Dividend Analysis: yield, annual dividend, payout ratio, 5-year CAGR
  - Factor Scores: Piotroski F-Score, Beneish M-Score, Altman Z-Score (financial distress indicator)
SUPPORTED TICKERS: AAPL, MSFT, GOOGL, TSLA, NVDA (PostgreSQL data available)
MULTI-TICKER: Supported — returns one result dict per ticker
DO NOT USE FOR: Qualitative moat analysis, real-time news, sentiment analysis

=== OUTPUT SCHEMA ===

Output ONLY a valid JSON object with this schema:
{
  "tickers": ["<TICKER1>", "<TICKER2>"],
  "ticker": "<FIRST_TICKER_or_null>",
  "intent": "<brief description of what the user wants>",
  "run_business_analyst": <true|false>,
  "run_quant_fundamental": <true|false>,
  "run_web_search": <true|false>,
  "run_financial_modelling": <true|false>,
  "complexity": <1|2|3>,
  "reasoning": "<2-3 sentences explaining tool selection and multi-ticker handling>"
}

=== COMPLEXITY SCORING ===
Set "complexity" to an integer 1, 2, or 3 based on how deep the analysis required is:
  1 — Simple / single-concept: a single metric look-up or narrow question.
      Examples: "What is AAPL's P/E ratio?", "What is TSLA's current RSI?",
                "What is MSFT's dividend yield?", "Is NVDA profitable?"
  2 — Moderate / multi-faceted: several metrics or themes, but not a full report.
      Examples: "What are AAPL's valuation and quality factors?",
                "Analyse TSLA's risks and competitive position",
                "Compare MSFT vs AAPL on profitability metrics"
  3 — Comprehensive / full analysis: explicitly requests a complete report, deep dive,
      or full fundamental/technical analysis covering multiple dimensions.
      Examples: "Give me a complete fundamental analysis of AAPL",
                "Full report on NVDA", "Deep dive on MSFT",
                "Comprehensive analysis of TSLA vs AAPL",
                "Analyse Apple's competitive moat and key risks" (qualitative + quant)
Default to 2 if uncertain.

=== TICKER EXTRACTION RULES ===
- Extract ALL ticker symbols from the query into the "tickers" array
- "ticker" = first symbol in the array (or null if none found)
- For comparison queries like "Compare MSFT vs AAPL" → tickers: ["MSFT", "AAPL"]
- For single queries like "Analyze AAPL" → tickers: ["AAPL"]
- Always use UPPERCASE ticker symbols

=== AGENT SELECTION RULES ===
- run_business_analyst: true when question involves qualitative analysis, competitive moat,
  business model, risks, sentiment, or general company research
- run_quant_fundamental: true when question involves financial ratios, valuation metrics,
  P/E, ROE, earnings, balance sheet, quantitative factors, or stock screening;
  ALWAYS enable for comparison/fundamentals queries — it handles multi-ticker natively
- run_web_search: true ONLY when the question explicitly asks for recent/breaking news,
  current events, or when the user says "latest", "today", "recent news", "what happened"
  Do NOT enable web_search for standard fundamental or qualitative analysis.
- run_financial_modelling: true when the question involves DCF valuation, intrinsic value,
  WACC, price targets, discounted cash flow, fair value, technical analysis, support/resistance,
  RSI, MACD, Bollinger Bands, moving averages, earnings surprise analysis, dividend analysis,
  Altman Z-Score, or any quantitative valuation modelling beyond basic financial ratios.
  Also enable for queries like "is [TICKER] overvalued/undervalued?" or "what is the intrinsic
  value of...?" or "should I buy/sell based on technicals?"
  ALSO enable for any "fundamental analysis", "complete analysis", "full analysis",
  "comprehensive analysis", or "deep dive" query — these always benefit from DCF + technicals.

COMPREHENSIVE QUERY RULE: If the query contains any of: "fundamental analysis",
"complete analysis", "full analysis", "comprehensive analysis", "deep dive", "full report",
"complete report" — set ALL FOUR of run_business_analyst, run_quant_fundamental,
run_financial_modelling to true. Only keep run_web_search=false unless news is
explicitly requested.

Always enable at least one tool. If the question is ambiguous, enable both
business_analyst and quant_fundamental.
For "complete", "full", "comprehensive", or "fundamental analysis" queries, enable ALL FOUR agents.
"""


def plan_query(user_query: str) -> Dict[str, Any]:
    """Call DeepSeek planner and return the structured plan dict.

    Returns a safe default plan on any LLM failure.
    """
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    planner_prompt_with_date = (
        f"Today's date (UTC): {today_utc}\n"
        f"Use this date as the reference point when interpreting 'recent', 'latest', or 'current' queries.\n\n"
        f"{_PLANNER_SYSTEM}"
    )
    prompt = f"{planner_prompt_with_date}\n\nUser question: {user_query}\n\nJSON plan:"

    _query_lower = user_query.lower()

    # Determine complexity from keywords as a fallback/override baseline
    _is_comprehensive = any(kw in _query_lower for kw in [
        "fundamental analysis", "full analysis", "complete analysis",
        "comprehensive analysis", "deep dive", "deep-dive",
        "dcf", "intrinsic value", "wacc", "fair value", "valuation",
        "technical analysis", "rsi", "macd", "overvalued", "undervalued",
        "price target", "financial modelling", "financial modeling",
    ])

    # Keyword-based complexity baseline (used when LLM doesn't return a valid value)
    _complexity_keywords_3 = [
        "fundamental analysis", "complete analysis", "full analysis",
        "comprehensive analysis", "deep dive", "deep-dive", "full report",
        "complete report",
    ]
    _complexity_keywords_1 = [
        "p/e", "pe ratio", "p/e ratio", "rsi", "dividend yield",
        "current price", "market cap", "eps", "is.*profitable",
    ]
    _default_complexity: int
    if any(kw in _query_lower for kw in _complexity_keywords_3) or (
        _is_comprehensive and len(user_query.split()) > 8
    ):
        _default_complexity = 3
    elif any(kw in _query_lower for kw in _complexity_keywords_1):
        _default_complexity = 1
    else:
        _default_complexity = 2

    try:
        raw = _ollama_generate(_PLANNER_MODEL, prompt, max_tokens=512, temperature=0.1)
        plan = _extract_json(raw)
        plan.setdefault("run_business_analyst", True)
        plan.setdefault("run_quant_fundamental", True)
        plan.setdefault("run_web_search", False)
        # Default run_financial_modelling to True for comprehensive/valuation queries
        plan.setdefault("run_financial_modelling", _is_comprehensive)
        plan.setdefault("ticker", None)
        plan.setdefault("tickers", [])
        # Sanitise complexity: must be 1, 2, or 3; fall back to keyword-derived default
        raw_complexity = plan.get("complexity")
        if raw_complexity not in (1, 2, 3):
            try:
                raw_complexity = int(raw_complexity)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                raw_complexity = None
        plan["complexity"] = raw_complexity if raw_complexity in (1, 2, 3) else _default_complexity
        # If query is comprehensive and LLM explicitly set it False, override to True
        if _is_comprehensive and not plan.get("run_financial_modelling"):
            logger.info(
                "[planner] Overriding run_financial_modelling=True "
                "for comprehensive/fundamental query: %r", user_query[:80]
            )
            plan["run_financial_modelling"] = True
        return plan
    except Exception as exc:
        logger.warning("Planner LLM failed (%s) — using fallback plan", exc)
        return {
            "ticker": None,
            "tickers": [],
            "intent": user_query,
            "run_business_analyst": True,
            "run_quant_fundamental": True,
            "run_web_search": False,
            "run_financial_modelling": _is_comprehensive,
            "complexity": _default_complexity,
            "reasoning": "Fallback: planner LLM unavailable, running core agents.",
        }


# ── Summarizer ────────────────────────────────────────────────────────────────

_SUMMARIZER_SYSTEM = """You are an MD of Equity Research at a buy-side firm (Fidelity / Capital Group tier).
Write a structured investment research note for the internal investment committee.
Any violation of the rules below = report rejected.

=== HARD RULES ===

RULE 0 — NO RECOMMENDATIONS OR PRICE TARGETS.
Forbidden: Buy/Sell/Hold/Outperform/Underperform/Overweight/Underweight/Market Perform, target price,
price objective, fair value estimate, upside to $X, downside to $X, "we recommend", "we initiate",
"recommended as a", any disclaimer or "prepared by" block.
The ## Analyst Verdict must express directional views WITHOUT any of the above.

RULE 1 — NO BULLET POINTS. No "•", "-", "*", or numbered lists (1. 2. 3.) in any section body.
Every fact, risk, and metric must be flowing multi-sentence prose.

RULE 2 — EXACT HEADERS ONLY. Use these 11 headers in this exact order, nothing else:
  ## Executive Summary
  ## Company Overview
  ## Financial Performance
  ## Key Financial Ratios & Valuation
  ## Sentiment & Market Positioning
  ## Growth Prospects
  ## Risk Factors
  ## Competitive Landscape
  ## Management & Governance
  ## Macroeconomic Factors
  ## Analyst Verdict
Forbidden headers (use the correct one above instead): ## Competitive Moat Analysis,
## Financial Health & Quality, ## Valuation Analysis, ## Risk Assessment,
## Catalysts & Growth Outlook, ## Momentum & Risk-Return Profile,
## Investment Recommendation, ## Analyst Outlook, ## Conclusion, any ### sub-header,
any numbered prefix (1. 2.1).
Do NOT write a title, date, or ticker line before ## Executive Summary.

RULE 3 — ZERO FABRICATION. Only use numbers, names, and ratings that appear verbatim in the data.

RULE 4 — BANNED WORDS. Replace each with a specific number or mechanism:
  robust, solid, strong performance, impressive, compelling, notable, remarkable,
  significant growth, standout, well-positioned, healthy, exceptional results, poised,
  exciting, promising, outstanding, stellar.

RULE 5 — CITE EVERY DATA SENTENCE. Every sentence with a number, %, ratio, score, or sourced
assertion MUST end with [N] from the SOURCE CITATION INDEX. Never invent index numbers.
CORRECT: "The Piotroski F-Score of 9/9 signals high financial quality [5]."
INCORRECT: "The F-Score is 9." (no citation — rejected)
INCORRECT: "Strong fundamentals [7]." ([7] not in index — rejected)
Every paragraph must contain at least one citation.

=== WRITING STANDARDS ===
Flowing, authoritative financial prose throughout — write like a senior sector analyst briefing an investment committee.
Use precise financial terminology: TTM, YoY, QoQ, FCF conversion, operating leverage inflection, capital allocation
efficiency, earnings quality, cost of capital, spread compression, working capital cycle, EPS accretion.
State the analytical implication of every number — do not just recite metrics. Every sentence must carry weight.
For COMPARISON queries: weave both companies through every section with explicit relative rankings and quantified
spread analysis. Never analyse them in isolation. State which company leads on each named metric and by how much.

=== SECTION REQUIREMENTS ===

## Executive Summary
Write EXACTLY 5–6 sentences. No pre-header lines, no date, no ticker metadata before this section.
Sequence is mandatory:
  S1: The single most critical finding — state with a specific number and explain the investment significance.
  S2: The core quality or competitive differentiator with the metric that proves it — why this company earns
      its current market positioning or why it does not.
  S3: Valuation — state cheap/fair/expensive against large-cap tech norms (P/E 22–38x, EV/EBITDA 16–28x,
      P/FCF 20–35x) with exact multiples AND the explicit percentage premium or discount vs. sector peers
      (e.g. "trades at a 28% premium to sector median EV/EBITDA of 20x"); state what the current multiple
      implies about embedded growth expectations.
  S4: The primary risk mechanism — the specific chain of events that could invalidate the thesis, with a
      quantified financial magnitude (basis points of margin, dollars of revenue, or multiple contraction).
  S5 (single-ticker): The directional tilt the totality of evidence supports and the single swing factor
      that would change it — name the metric and the threshold.
  S5–S6 (multi-ticker): Which company is fundamentally superior on exactly 2 named metrics and why the
      quantified spread between them matters for relative positioning — be explicit about the gap size.
Cite at least 2 different source numbers. No bullet points.

## Company Overview
Write 4–5 sentences of prose:
  - What the company does: primary product and service segments, installed base or market position, and
    approximate revenue contribution split if available from context.
  - Current strategic focus: the single most important capital allocation initiative and the rationale
    management has given for it.
  - Competitive moat source: identify the type (network effects, switching costs, cost advantage, intangibles),
    explain the mechanism briefly, and assess whether it is widening, stable, or narrowing.
  - Business model quality signal: one sentence on revenue quality (recurring vs. transactional) and what
    it implies for earnings predictability.
For comparisons: 3–4 sentences per company, then 1–2 synthesis sentences ranking market position.
Cite BA qualitative source for all moat and strategic claims.

## Financial Performance
Write 3 substantive prose paragraphs — no sub-headers, bullets, or lists:
  Paragraph 1 — INCOME STATEMENT (minimum 5 sentences): Anchor on the most recent quarter's revenue,
    net income, and diluted EPS. State the YoY growth rate and QoQ growth rate for each metric, drawn
    explicitly from quarterly_trends and qoq_deltas/yoy_deltas. State trailing twelve-month totals where
    available. Quote exact dollar figures and percentages. Interpret what the trajectory implies: is
    top-line growth accelerating, decelerating, or inflecting? Is EPS growing faster or slower than
    revenue — and what does that spread tell us about margin direction? Apply the YoY/QoQ trigger language
    where thresholds are met (see trigger rules below).
  Paragraph 2 — BALANCE SHEET (minimum 4 sentences): Cover total assets, total equity, and net
    debt or net cash position. State the current ratio and interpret it for near-term liquidity risk.
    State the D/E ratio and interpret it for financial flexibility and capital return capacity. Assess
    whether the balance sheet structure enables or constrains the company's ability to fund its strategic
    priorities, return capital to shareholders, or absorb an adverse macro scenario.
  Paragraph 3 — CASH FLOW (minimum 4 sentences): Cover operating cash flow and free cash flow — state
    both absolute values and the relationship between them. Compute and state the FCF conversion ratio
    (FCF/net income) and apply the trigger interpretation (>1.0 = high earnings quality; <0.7 = concern).
    Cover capital expenditure trends and what the capex intensity implies for near-term FCF trajectory.
    Cover buyback and dividend activity and quantify the EPS accretion or yield to shareholders.
    State the capital allocation implication: is management returning cash at the right time, or is there
    evidence of over-investment, underinvestment, or value-destructive M&A?
Cite quant_fundamental source for every figure.

## Key Financial Ratios & Valuation
Write 3 substantive prose paragraphs:
  Paragraph 1 — VALUATION MULTIPLES (minimum 5 sentences): Anchor on trailing P/E, EV/EBITDA, and P/FCF
    together. Benchmark each against large-cap tech norms (P/E 22–38x, EV/EBITDA 16–28x, P/FCF 20–35x).
    State explicitly whether each multiple is cheap, fair, or expensive relative to peers AND quantify the
    premium or discount as a percentage — e.g. "EV/EBITDA of 25.6x sits 28% above the sector median of
    20x" or "P/E of 33x is at the midpoint of the 22–38x large-cap tech range". NEVER describe a premium
    or discount in vague language — always name the exact spread in percentage points or absolute turns.
    For the trailing P/E, calculate the implied 5-year annual EPS growth rate the multiple embeds
    (at 33x → ~12–15%; at 25x → ~8–10%; at 40x+ → ~18–22%). Cover EV/Revenue and dividend yield where
    available. Explain what the composite multiple picture tells us about whether the market is pricing
    perfection, value, or distress. If FINANCIAL MODELLING output is present, incorporate the DCF
    intrinsic value range (low/mid/high) and state the implied premium or discount to the current market
    price. Incorporate the Comps vs_sector_avg data to state whether the stock trades at a premium or
    discount to sector peers on EV/EBITDA and P/E, and by exactly how many percentage points.
    For comparisons: state the valuation premium or discount in absolute spread terms and explain whether
    the spread is justified by fundamentals.
  Paragraph 2 — QUALITY RATIOS (minimum 5 sentences): Cover ROE and ROIC vs. the ~8–10% WACC benchmark
    for large-cap tech — state explicitly whether the business is compounding capital ABOVE or BELOW its cost.
    CRITICAL: If ROE exceeds 50%, you MUST explain the capital-return mechanism that drives the elevated figure —
    specifically whether it reflects genuine operating leverage or equity base compression from sustained buybacks,
    and what that structural distinction means for interpreting capital efficiency. Do NOT describe a ROE of 100%+
    as "moderate" or "within-range" — it is far above cost of capital and demands explicit explanation.
    Similarly, if ROIC is materially above WACC (>20pp spread), quantify the spread explicitly (e.g. "ROIC of
    51% represents a ~42pp spread above the ~9% WACC, indicative of a durable economic moat") and name the
    mechanism sustaining that spread.
    Assess the Piotroski F-Score with full interpretation (8–9 = exceptional quality across profitability,
    leverage, and operating efficiency; 7 = above-average; 5–6 = mixed signals; ≤4 = deteriorating trend).
    If Beneish M-Score is available, interpret it fully (< -2.22 = low manipulation risk, earnings quality
    appears reliable; -1.78 to -2.22 = moderate concern; > -1.78 = elevated earnings quality scrutiny
    warranted). If FINANCIAL MODELLING output is present, include the Altman Z-Score with interpretation
    (>2.99 = safe zone; 1.81–2.99 = grey zone; <1.81 = distress zone). Reference all CoT Validation Notes
    — for each flagged tension, state what the tension is, why it matters for the investment case, and what
    an analyst should probe further. Discuss gross margin and EBIT margin levels in the context of sector
    benchmarks.
  Paragraph 3 — MOMENTUM & RISK (minimum 3 sentences): Cover the 12-month price return and the Sharpe
    ratio with full interpretation (>1.0 = adequate risk-adjusted compensation; 0.5–1.0 = marginal;
    <0.5 = poor; negative = market actively discounting). State whether the market has been rewarding
    or penalising the stock on a risk-adjusted basis and what the Sharpe level implies about whether
    investors are being compensated for the risks they are bearing. Cover beta and its implication for
    portfolio construction — high beta (>1.3) in a risk-off environment is a meaningful sizing constraint.
    If FINANCIAL MODELLING technical output is present, include RSI(14) and trend signal with
    interpretation, and note any Bollinger Band, MACD, or 52-week position context.

## Sentiment & Market Positioning
Write 3 prose paragraphs:
  Paragraph 1 — SENTIMENT POSITIONING (minimum 5 sentences): Use the Sentiment Verdict signal
    (CONSTRUCTIVE/CAUTIOUS/NEUTRAL/DETERIORATING) as the structural anchor. State the precise
    bullish/bearish/neutral percentage split. Compare explicitly to the ~55% large-cap tech base rate —
    how many percentage points above or below, and what does that deviation signal about conviction
    levels? Identify in concrete terms what the dominant bullish cohort is betting on — the specific
    thesis, the mechanism, and the expected financial outcome. Identify what the bearish minority fears —
    the specific risk mechanism and financial scenario they are pricing in. If the trend is shifting
    (improving/deteriorating), state the specific catalyst that has driven the change in recent periods.
  Paragraph 2 — SENTIMENT-QUANT SYNTHESIS (minimum 4 sentences): Directly compare the sentiment
    signal against the quantitative evidence: does the Piotroski F-Score, margin trajectory, and Sharpe
    ratio corroborate or contradict the dominant sentiment signal? If there is a divergence (e.g. 70%+
    bullish despite a Piotroski ≤5 or declining margins), name it explicitly and explain what each
    investor cohort is seeing differently — which set of information is each group weighting most heavily?
    For CAUTIOUS or DETERIORATING signals: assess whether the quant data validates or disputes the bear
    case — is the pessimism justified by the numbers? State the analytical conclusion.
  Paragraph 3 — POSITIONING IMPLICATION (minimum 3 sentences): Given the sentiment-quant comparison,
    what is the most likely next directional move in consensus — is sentiment likely to catch up to the
    quant evidence, or is the quant evidence likely to deteriorate to match the sentiment? Name the
    single catalyst or data point that would most likely trigger a consensus shift in either direction.
    For comparisons: contrast both companies' sentiment verdicts and assess whether the relative sentiment
    gap is justified by the fundamental spread.

## Growth Prospects
Write 3 prose paragraphs:
  Paragraph 1 — REVENUE GROWTH DRIVERS (minimum 5 sentences): Identify the primary revenue growth
    drivers by segment, geography, or product category using the YoY and QoQ trend data. Quantify
    recent growth rates and interpret whether the trajectory is accelerating, sustaining, or decelerating.
    Discuss AI-related, product launch, or platform catalysts mentioned in management_guidance or BA
    qualitative output — what specifically is driving incremental growth and over what time horizon?
    Comment on geographic diversification: which markets are growing fastest and what does the mix
    imply for risk concentration? Assess whether any mix shift is occurring that would structurally
    change the margin profile of the revenue base over the next 2–3 years.
  Paragraph 2 — MARGIN GROWTH & OPERATING LEVERAGE (minimum 4 sentences): Analyse whether revenue
    growth is translating into margin expansion — is the company demonstrating positive operating
    leverage (margins rising faster than revenue) or absorbing growth-related costs that compress
    margins? Identify the specific cost lines driving the operating leverage dynamic. Discuss whether
    any near-term investment cycle (R&D, capex, headcount) will suppress margins before they expand —
    and if so, over what time horizon the investment phase is expected to resolve. Apply trigger
    language where YoY gross margin or EBIT margin thresholds are met.
  Paragraph 3 — CAPITAL RETURNS & EPS ACCRETION (minimum 4 sentences): Discuss the capital return
    program comprehensively — buyback pace, authorisation headroom, dividend trajectory, and the
    implied yield to shareholders at current prices. Quantify the EPS accretion effect of buybacks
    if the data allows (shares retired × earnings per share impact). Reference management guidance
    near-term catalysts and the forward outlook summary — what has management indicated about the
    sustainability of the capital return program? State what the combined effect of organic earnings
    growth and capital returns implies for per-share earnings compounding over the next 2–3 years.

## Risk Factors
Write EXACTLY 2 substantive prose paragraphs — no bullets, no numbered lists, no sub-headers:
  Paragraph 1 — TOP RISKS WITH MAGNITUDE (minimum 8 sentences): Address the top 3–4 risks in a single
    flowing paragraph. For EACH risk you MUST include all three elements woven into the prose:
    (a) the specific mechanism — the causal chain from trigger event to financial impact, naming which
    revenue line or cost structure is affected; (b) the financial magnitude as a concrete quantified
    scenario — NOT vague language like "significant impact". You must write something specific such as:
    "a 10pp decline in iPhone upgrade rates would compress hardware gross margins by approximately 150
    basis points on the ~60% of revenue exposed to the consumer hardware segment"; or "a 200bp rise in
    enterprise IT budget cuts would reduce cloud services revenue by an estimated $6–8 billion based on
    current contracted backlog"; or "a successful FTC antitrust ruling on App Store commissions would
    remove an estimated $8–12 billion of near-zero marginal cost Services revenue annually";
    (c) the observed mitigation or management response — cite what management has actually done or said,
    not what they could theoretically do. Rate each risk HIGH, MEDIUM, or LOW within the prose.
    Integrate any CoT Validation Notes tensions as quantitative risk evidence where the data supports it.
    For comparisons: identify which company faces the more acute version of each risk and why.
  Paragraph 2 — AGGREGATE RISK PROFILE & INVALIDATION SCENARIO (minimum 5 sentences): Assess whether
    the risks identified are independent (each can materialise without triggering others) or correlated
    (a single macro or regulatory shock could trigger multiple simultaneously). Name the single risk
    that, if it materialises, would most severely and irreversibly damage the investment thesis — explain
    in one sentence why it is the dominant risk relative to the others. State the specific invalidation
    scenario: a concrete trigger (the event that would need to occur), the financial consequence (the
    specific P&L, balance sheet, or multiple impact), and the time horizon over which the damage would
    become visible. Assess the aggregate risk profile as LOW/MODERATE/HIGH/ELEVATED and explain the
    rating. Cite BA qualitative sources throughout. For comparisons: compare the aggregate risk profiles
    explicitly and state which company has the less risky aggregate risk profile and why.

## Competitive Landscape
Write 2 substantive prose paragraphs:
  Paragraph 1 — COMPETITIVE POSITION (minimum 5 sentences): Name the primary competitors explicitly.
    Identify the primary moat source (network effects, switching costs, cost advantage, intangible assets)
    and quantify its magnitude where evidence allows — e.g. retention rates, market share percentages,
    take rates, switching cost estimates in dollars or months. Explain what sustains the moat and what
    the moat's durability implies for long-term market share trajectory. Assess whether the competitive
    advantage is widening or narrowing — and what the primary force driving the trajectory is. Cite BA
    qualitative sources for all competitive assertions.
  Paragraph 2 — THREAT ANALYSIS (minimum 4 sentences): Name the single most credible competitive
    threat in specific terms — name the company or technology, not a generic category. Explain the
    precise mechanism by which this threat could erode the moat: the substitution pathway, the customer
    segment most at risk, the revenue or margin line most exposed, and the time horizon over which the
    erosion could become material. Identify 1–2 secondary threats and their mechanisms. For comparisons:
    rank both companies on moat strength with a quantified gap — which has the wider moat and what is
    the specific evidence that drives the differential?

## Management & Governance
Write 2 prose paragraphs:
  Paragraph 1 — CAPITAL ALLOCATION & TRACK RECORD (minimum 4 sentences): Cover the CEO or leadership
    team's capital allocation track record — has management consistently deployed FCF in value-accretive
    ways? Cover governance quality signals: board composition, insider ownership, executive compensation
    alignment. Reference any specific earnings call highlights or management forward guidance statements
    from the management_guidance data — quote or paraphrase the most notable management statement.
    Assess management credibility: have recent guidance figures been met, exceeded, or missed?
  Paragraph 2 — FORWARD STRATEGY & SUCCESSION RISK (minimum 3 sentences): Cover the company's stated
    strategic priorities for the next 12–24 months based on the forward outlook summary. Identify any
    succession considerations or key-person risk. Assess whether the management team has articulated
    a clear, credible, and differentiated strategy — or whether the forward guidance is vague and
    dependent on favourable macro conditions. If neither management_guidance nor BA qualitative data
    is available, replace both paragraphs with exactly:
    "Insufficient management and governance data is available from the current knowledge base."

## Macroeconomic Factors
Write 2 prose paragraphs covering all material macro exposures:
  Paragraph 1 — RATE & CYCLE SENSITIVITY (minimum 3 sentences): Cover interest rate sensitivity —
    does the company benefit or suffer from a rising rate environment (consumer credit costs, enterprise
    IT budget cycles, refinancing risk)? Cover consumer or enterprise spending cycle implications —
    is the revenue base cyclical or counter-cyclical, and what does the current cycle position imply?
    Assess whether current macro conditions are a tailwind or headwind for the thesis.
  Paragraph 2 — GEOPOLITICAL & STRUCTURAL MACRO (minimum 3 sentences): Cover material geopolitical
    exposures including supply chain geography, regulatory jurisdiction (EU AI Act, US export controls,
    Chinese antitrust), and revenue concentration by country. Identify the single most material macro
    risk and quantify the revenue or cost exposure. State whether the company has structural hedges
    against these exposures or is running concentrated macro risk. Draw from web search output or
    qualitative BA data. If no macro-relevant data is present, replace both paragraphs with exactly:
    "No macroeconomic factor data was provided for this analysis."

## Analyst Verdict
Write EXACTLY 1 paragraph of 6–8 sentences. All FIVE elements must appear:
  1. DIRECTIONAL SYNTHESIS: Integrate the sentiment verdict, Piotroski F-Score, margin trajectory,
     and YoY/QoQ revenue and earnings trends into a probability-weighted characterisation of the
     current risk/reward. Be specific — name the scores and trends and state what their combination
     implies directionally.
  2. VALUATION CONTEXT: State whether the current multiple is a barrier or an enabler for total
     return — is the stock priced to perfection (requiring flawless execution to justify the multiple),
     priced for failure (embedding a recessionary scenario that may not materialise), or fairly valued
     for the demonstrated business quality?
  3. KEY EXECUTION RISK: Name the single most critical execution risk and the specific measurable
     variable that, if it disappoints in the next 1–2 quarters, would most damage the current narrative.
     Name the threshold — e.g. "a gross margin print below 45% would signal..."
  4. COMPARISON VERDICT (multi-ticker ONLY): State an explicit relative preference between the two
     companies and name the exact 2–3 data points that drive the differential ranking — quantify the
     spread where possible.
  5. FORWARD VARIABLE: Name the single most important metric, event, or catalyst to monitor over
     the next 6 months that would most materially change the directional view in either direction.
     Explain briefly why this particular variable is the most decisive.
NO buy/sell/hold recommendations or price targets anywhere in this paragraph.

=== YoY / QoQ TRIGGER RULES ===
Revenue YoY >15% → "above-trend top-line momentum"
Revenue QoQ >5%  → "sequential acceleration in the most recent quarter"
FCF conversion >1.0 → "high earnings quality — cash earnings exceed reported net income"
FCF conversion <0.7 → "earnings quality concern — FCF materially trails net income"
Gross margin YoY +>2pp → "operating leverage inflection underway"
Gross margin YoY −>2pp → "margin compression trajectory warrants close monitoring"
EPS YoY > revenue YoY → "margin expansion or buyback accretion lifting per-share earnings"

=== INSUFFICIENT DATA RULES ===
If BA output shows INSUFFICIENT_DATA or CRAG=INCORRECT/confidence=0.00:
Company Overview and Competitive Landscape MUST begin: "The Business Analyst knowledge base
returned no documents for [TICKER], preventing qualitative analysis. The following is based
solely on quantitative data." Do not fabricate moat ratings, revenue breakdowns, or segment %.
Risk Factors must be prefixed: "Note: qualitative risk factors are unavailable."

=== FORMATTING ===
Target: 5,000–7,000 words (single); 7,000–9,000 words (multi). All 11 sections must be present
and substantive — do not collapse or merge sections. No References/Sources section (appended
automatically). End after ## Analyst Verdict. Brevity is not a virtue — depth is required.
Each section must be at minimum 3 paragraphs. Each paragraph must be at minimum 5 sentences.
Never truncate or abbreviate any section. If you run out of available detail, synthesise the
implications of what you do have — do not shorten the section.
"""


def _build_citation_index_prompt(chunk_id_map: Dict) -> str:
    """Build the citation index string injected into the summarizer prompt."""
    if not chunk_id_map:
        return ""
    # Collect unique citations sorted by index
    seen: set = set()
    entries = []
    for cit in sorted(chunk_id_map.values(), key=lambda c: c.index):
        if cit.index in seen:
            continue
        seen.add(cit.index)
        entries.append(cit)

    lines = ["\n=== SOURCE CITATION INDEX ==="]
    lines.append("When you reference a fact below, write the bracket number exactly as shown (e.g. [1], [2]).")
    for cit in entries:
        line = f"[{cit.index}] {cit.label}"
        if cit.detail:
            line += f" — {cit.detail}"
        if cit.url:
            line += f" ({cit.url})"
        # Append USE FOR annotation so the LLM knows which citation to pick per claim type
        label_lower = cit.label.lower()
        if "sentiment" in label_lower:
            line += " — USE FOR: bullish%, bearish%, neutral%, sentiment verdict, trend direction"
        elif any(k in label_lower for k in ("income", "balance", "cash flow")):
            line += " — USE FOR: revenue, net income, EPS, total assets, equity, debt, OCF, FCF, capex"
        elif any(k in label_lower for k in ("ratio", "metric")):
            line += " — USE FOR: P/E, EV/EBITDA, P/FCF, EV/Revenue, ROE, ROIC, gross margin, EBIT margin, D/E, current ratio"
        elif any(k in label_lower for k in ("price", "timeseries", "history")):
            line += " — USE FOR: beta, Sharpe ratio, 12-month price return ONLY — NOT for financial ratios or qualitative claims"
        elif "score" in label_lower:
            line += " — USE FOR: Piotroski F-Score, Beneish M-Score ONLY"
        elif any(k in label_lower for k in ("business_analyst", "qualitative", "business analyst")):
            line += " — USE FOR: moat analysis, competitive positioning, sentiment narrative, strategic risks"
        lines.append(line)
    lines.append("=== END INDEX ===\n")
    return "\n".join(lines)


def _build_quant_context(quant_output: Dict[str, Any]) -> List[str]:
    """Format quant output into readable key=value lines for the prompt.

    Enriched to surface YoY trends, FCF quality signals, and Beneish M-Score
    interpretation to give the summarizer enough depth for buy-side-grade prose.
    """
    ticker = quant_output.get("ticker", "?")
    parts: List[str] = [f"=== QUANTITATIVE FUNDAMENTAL OUTPUT: {ticker} ==="]
    parts.append(f"Time Range: {quant_output.get('time_range', 'TTM')}")
    parts.append(f"Quantitative Summary: {quant_output.get('quantitative_summary', 'N/A')}")

    vf = quant_output.get("value_factors") or {}
    if any(v is not None for v in vf.values()):
        pe = vf.get('pe_trailing')
        ev_ebitda = vf.get('ev_ebitda')
        p_fcf = vf.get('p_fcf')
        ev_rev = vf.get('ev_revenue')
        parts.append(
            f"Value Factors: P/E(ttm)={pe}, EV/EBITDA={ev_ebitda}, "
            f"P/FCF={p_fcf}, EV/Revenue={ev_rev}"
        )
        # Add implied growth context for elevated multiples
        if pe and isinstance(pe, (int, float)) and pe > 30:
            parts.append(
                f"  → P/E of {pe}x implies the market is pricing in sustained earnings growth; "
                f"context: large-cap tech sector median P/E ~22–32x TTM"
            )
        if ev_ebitda and isinstance(ev_ebitda, (int, float)):
            benchmark = "above" if ev_ebitda > 25 else ("in line with" if ev_ebitda > 16 else "below")
            parts.append(f"  → EV/EBITDA of {ev_ebitda}x is {benchmark} the large-cap tech benchmark range of 16–28x")
        if p_fcf and isinstance(p_fcf, (int, float)):
            fcf_tier = "elevated" if p_fcf > 35 else ("fair" if p_fcf > 20 else "compressed")
            parts.append(f"  → P/FCF of {p_fcf}x is {fcf_tier} vs. large-cap tech range of 20–35x")

    qf = quant_output.get("quality_factors") or {}
    if any(v is not None for v in qf.values()):
        piotroski = qf.get('piotroski_f_score')
        beneish = qf.get('beneish_m_score')
        manipulation_risk = qf.get('manipulation_risk')
        altman_z = qf.get('altman_z_score')
        roe = qf.get('roe')
        roic = qf.get('roic')
        beneish_str = str(beneish) if beneish is not None else "N/A (prior-year financials not in DB)"
        parts.append(
            f"Quality Factors: ROE={roe}, ROIC={roic}, "
            f"Piotroski F-Score={piotroski}/9, "
            f"Beneish M-Score={beneish_str}, Manipulation Risk={manipulation_risk}, "
            f"Altman Z-Score={altman_z}"
        )
        # Piotroski interpretation
        if piotroski is not None:
            if piotroski >= 8:
                f_interp = "EXCEPTIONAL (8–9): strong earnings quality, improving balance sheet, operating efficiency gains"
            elif piotroski == 7:
                f_interp = "STRONG (7): above-average financial health, most quality signals positive"
            elif piotroski >= 5:
                f_interp = "AVERAGE (5–6): mixed signals, some areas of concern but no systemic deterioration"
            else:
                f_interp = f"DETERIORATING (≤4): multiple negative quality signals — warrants elevated scrutiny"
            parts.append(f"  → Piotroski F-Score interpretation: {f_interp}")
        # Beneish interpretation
        if beneish is not None:
            try:
                m_val = float(beneish)
                if m_val < -2.22:
                    m_interp = "LOW manipulation risk (< -2.22 threshold): earnings quality appears reliable"
                elif m_val < -1.78:
                    m_interp = "MODERATE manipulation risk (-1.78 to -2.22): some earnings quality signals to monitor"
                else:
                    m_interp = "ELEVATED manipulation risk (> -1.78 threshold): earnings quality warrants scrutiny"
                parts.append(f"  → Beneish M-Score interpretation: {m_interp}")
            except (TypeError, ValueError):
                pass
        else:
            parts.append("  → Beneish M-Score: not computable (prior-year income/balance sheet data absent from DB)")
        # Altman Z-Score interpretation
        if altman_z is not None:
            try:
                z_val = float(altman_z)
                if z_val > 3.0:
                    z_interp = "SAFE ZONE (>3.0): low bankruptcy risk, strong financial position"
                elif z_val > 1.81:
                    z_interp = "GREY ZONE (1.81–3.0): moderate financial stress, monitor closely"
                else:
                    z_interp = "DISTRESS ZONE (<1.81): elevated bankruptcy risk"
                parts.append(f"  → Altman Z-Score={z_val:.3f}: {z_interp}")
            except (TypeError, ValueError):
                pass
        # WACC context for ROE/ROIC — tiered interpretation for very high values
        if roe is not None:
            try:
                roe_pct = float(roe) * 100
                if roe_pct > 100:
                    # ROE >100% typically signals shareholder equity has been compressed
                    # to near-zero or negative by aggressive buybacks — not pure operating leverage.
                    # Flag both the magnitude AND the structural explanation so the summarizer
                    # does NOT call it "within-range" or "moderate".
                    parts.append(
                        f"  → ROE of {roe_pct:.1f}% is FAR ABOVE the ~8–10% WACC benchmark — "
                        f"NOTE: ROE at this magnitude is primarily driven by equity base compression "
                        f"from sustained share buybacks rather than purely from operating earnings; "
                        f"the analyst MUST explain the capital-return mechanism and NOT describe this as 'moderate' or 'within-range'."
                    )
                elif roe_pct > 30:
                    parts.append(
                        f"  → ROE of {roe_pct:.1f}% is materially above the ~8–10% WACC benchmark — "
                        f"indicates capital compounding well above cost; discuss whether this reflects "
                        f"operating leverage, pricing power, or capital structure effects."
                    )
                elif roe_pct > 10:
                    parts.append(
                        f"  → ROE of {roe_pct:.1f}% is above the ~8–10% WACC benchmark for large-cap tech "
                        f"(capital compounding above cost)"
                    )
                else:
                    parts.append(
                        f"  → ROE of {roe_pct:.1f}% is BELOW the ~8–10% WACC benchmark — value destruction risk"
                    )
            except (TypeError, ValueError):
                pass
        if roic is not None:
            try:
                roic_pct = float(roic) * 100
                if roic_pct > 40:
                    parts.append(
                        f"  → ROIC of {roic_pct:.1f}% is FAR above the implied WACC of ~8–10% — "
                        f"a spread of ~{roic_pct - 9:.0f}pp implies durable economic value creation; "
                        f"discuss the specific moat mechanism that sustains this spread."
                    )
                elif roic_pct > 10:
                    roic_spread = roic_pct - 9  # midpoint of 8-10% WACC range
                    parts.append(
                        f"  → ROIC of {roic_pct:.1f}% is above the implied WACC of ~8–10% "
                        f"(~{roic_spread:.0f}pp positive spread — capital compounding above cost)"
                    )
                else:
                    parts.append(f"  → ROIC of {roic_pct:.1f}% is below or at the implied WACC of ~8–10% — limited economic value creation")
            except (TypeError, ValueError):
                pass

    km = quant_output.get("key_metrics") or {}
    if any(v is not None for v in km.values()):
        gross_margin = km.get('gross_margin')
        ebit_margin = km.get('ebit_margin')
        fcf_conversion = km.get('fcf_conversion')
        de = km.get('debt_to_equity')
        cr = km.get('current_ratio')
        parts.append(
            f"Key Metrics: Gross Margin={gross_margin}, EBIT Margin={ebit_margin}, "
            f"FCF Conversion={fcf_conversion}, D/E={de}, Current Ratio={cr}"
        )
        # Margin benchmarks
        if gross_margin is not None:
            try:
                gm_pct = float(gross_margin) * 100
                if gm_pct > 65:
                    gm_tier = "software/cloud-tier margins (>65%) — exceptional pricing power"
                elif gm_pct > 50:
                    gm_tier = "above semiconductor benchmark (>50%) — strong pricing power"
                elif gm_pct > 35:
                    gm_tier = "solid hardware-tier margins (>35%) — competitive but not exceptional"
                else:
                    gm_tier = "below large-cap tech averages — margin pressure or commoditised product mix"
                parts.append(f"  → Gross Margin of {gm_pct:.1f}%: {gm_tier}")
            except (TypeError, ValueError):
                pass
        if ebit_margin is not None:
            try:
                em_pct = float(ebit_margin) * 100
                em_tier = "high-quality operator (>25%)" if em_pct > 25 else (
                    "solid operator (15–25%)" if em_pct > 15 else "margin improvement needed (<15%)")
                parts.append(f"  → EBIT Margin of {em_pct:.1f}%: {em_tier} vs. large-cap tech benchmark >25%")
            except (TypeError, ValueError):
                pass
        if fcf_conversion is not None:
            try:
                fcf_val = float(fcf_conversion)
                if fcf_val > 1.0:
                    fcf_interp = "HIGH earnings quality — FCF exceeds reported net income (cash conversion >100%)"
                elif fcf_val > 0.7:
                    fcf_interp = "SOLID earnings quality — FCF broadly tracks net income"
                else:
                    fcf_interp = "EARNINGS QUALITY CONCERN — FCF materially trails net income (non-cash items elevated)"
                parts.append(f"  → FCF Conversion of {fcf_val:.2f}x: {fcf_interp}")
            except (TypeError, ValueError):
                pass
        if de is not None:
            try:
                de_val = float(de)
                de_tier = "elevated leverage (>2.0)" if de_val > 2.0 else (
                    "moderate leverage (1.0–2.0)" if de_val > 1.0 else "conservative balance sheet (<1.0)")
                parts.append(f"  → D/E of {de_val:.2f}x: {de_tier} for large-cap tech")
            except (TypeError, ValueError):
                pass
        if cr is not None:
            try:
                cr_val = float(cr)
                cr_note = "LIQUIDITY CAUTION (<1.0)" if cr_val < 1.0 else (
                    "adequate liquidity (1.0–2.0)" if cr_val < 2.0 else "ample short-term coverage (>2.0)")
                parts.append(f"  → Current Ratio of {cr_val:.2f}x: {cr_note}")
            except (TypeError, ValueError):
                pass

    mr = quant_output.get("momentum_risk") or {}
    if any(v is not None for v in mr.values()):
        beta = mr.get('beta_60d')
        sharpe = mr.get('sharpe_ratio_12m')
        ret12m = mr.get('return_12m_pct')
        parts.append(
            f"Momentum/Risk: Beta(60d)={beta}, Sharpe(12m)={sharpe}, Return(12m)={ret12m}%"
        )
        if sharpe is not None:
            try:
                s_val = float(sharpe)
                if s_val > 1.0:
                    s_interp = "STRONG risk-adjusted performance (>1.0) — market rewarding quality"
                elif s_val > 0.5:
                    s_interp = "ACCEPTABLE risk-adjusted return (0.5–1.0) — adequate compensation for volatility"
                elif s_val > 0.0:
                    s_interp = "WEAK risk-adjusted return (0–0.5) — low compensation for volatility taken"
                else:
                    s_interp = "NEGATIVE risk-adjusted return (<0) — market actively discounting the stock"
                parts.append(f"  → Sharpe Ratio of {s_val:.2f}: {s_interp}")
            except (TypeError, ValueError):
                pass

    anomalies = quant_output.get("anomaly_flags") or []
    if anomalies:
        parts.append(f"Anomaly Flags (Z-score deviations from 3-year rolling baseline):")
        for flag in anomalies[:4]:
            if isinstance(flag, dict):
                metric = flag.get('metric', 'unknown')
                z = flag.get('z_score')
                direction = flag.get('direction', '')
                interp = flag.get('interpretation', '')
                parts.append(f"  • {metric}: z-score={z} ({direction}) — {interp}")

    # ── Quarterly Trends (last 4 periods) ────────────────────────────────────
    qt = quant_output.get("quarterly_trends") or []
    if qt:
        parts.append("Quarterly Trends (last 4 periods, Python-computed from PostgreSQL):")
        for period in qt:
            if isinstance(period, dict):
                p     = period.get("period", "?")
                rev   = period.get("revenue")
                gp    = period.get("gross_profit")
                oi    = period.get("operating_income")
                ni    = period.get("net_income")
                eps   = period.get("eps_diluted")
                gm    = period.get("gross_margin")
                em    = period.get("ebit_margin")
                line  = f"  {p}: revenue={rev}"
                if ni is not None:
                    line += f", net_income={ni}"
                if eps is not None:
                    line += f", EPS={eps}"
                if gm is not None:
                    try:
                        line += f", gross_margin={float(gm)*100:.1f}%"
                    except (TypeError, ValueError):
                        line += f", gross_margin={gm}"
                if em is not None:
                    try:
                        line += f", ebit_margin={float(em)*100:.1f}%"
                    except (TypeError, ValueError):
                        line += f", ebit_margin={em}"
                parts.append(line)

    # ── QoQ Deltas ───────────────────────────────────────────────────────────
    qoq = quant_output.get("qoq_deltas") or {}
    if qoq:
        rev_qoq   = qoq.get("revenue_qoq_pct")
        gm_qoq    = qoq.get("gross_margin_qoq_pp")
        em_qoq    = qoq.get("ebit_margin_qoq_pp")
        ni_qoq    = qoq.get("net_income_qoq_pct")
        eps_qoq   = qoq.get("eps_qoq_pct")
        line_parts = []
        if rev_qoq  is not None: line_parts.append(f"revenue {rev_qoq:+.1f}%")
        if gm_qoq   is not None: line_parts.append(f"gross_margin {gm_qoq:+.2f}pp")
        if em_qoq   is not None: line_parts.append(f"ebit_margin {em_qoq:+.2f}pp")
        if ni_qoq   is not None: line_parts.append(f"net_income {ni_qoq:+.1f}%")
        if eps_qoq  is not None: line_parts.append(f"EPS {eps_qoq:+.1f}%")
        if line_parts:
            parts.append(f"QoQ Changes (most recent vs prior quarter): {', '.join(line_parts)}")

    # ── YoY Deltas ───────────────────────────────────────────────────────────
    yoy = quant_output.get("yoy_deltas") or {}
    if yoy:
        rev_yoy   = yoy.get("revenue_yoy_pct")
        gm_yoy    = yoy.get("gross_margin_yoy_pp")
        em_yoy    = yoy.get("ebit_margin_yoy_pp")
        ni_yoy    = yoy.get("net_income_yoy_pct")
        eps_yoy   = yoy.get("eps_yoy_pct")
        line_parts = []
        if rev_yoy  is not None: line_parts.append(f"revenue {rev_yoy:+.1f}%")
        if gm_yoy   is not None: line_parts.append(f"gross_margin {gm_yoy:+.2f}pp")
        if em_yoy   is not None: line_parts.append(f"ebit_margin {em_yoy:+.2f}pp")
        if ni_yoy   is not None: line_parts.append(f"net_income {ni_yoy:+.1f}%")
        if eps_yoy  is not None: line_parts.append(f"EPS {eps_yoy:+.1f}%")
        if line_parts:
            parts.append(f"YoY Changes (most recent quarter vs same quarter prior year): {', '.join(line_parts)}")

    # ── CoT Validation Notes ──────────────────────────────────────────────────
    cot_notes = quant_output.get("cot_validation_notes") or []
    if cot_notes:
        parts.append("CoT Validation Notes (Python self-check, flag tensions for analyst review):")
        for note in cot_notes[:6]:
            parts.append(f"  ⚠ {note}")

    dq = quant_output.get("data_quality") or {}
    parts.append(
        f"Data Quality: {dq.get('status')} "
        f"({dq.get('checks_passed')}/{dq.get('checks_total')} checks passed)"
    )
    if dq.get('issues'):
        for issue in (dq.get('issues') or [])[:3]:
            parts.append(f"  • Data issue: {issue}")
    return parts


def _build_ba_context(ba_output: Dict[str, Any]) -> List[str]:
    """Format business analyst output into prompt context.

    Enriched to surface sentiment trend direction, moat vulnerability detail,
    strategic implication, and risk descriptions for buy-side-grade synthesis.
    """
    ticker = ba_output.get("ticker", "?")
    parts: List[str] = [f"=== BUSINESS ANALYST OUTPUT: {ticker} ==="]

    # Surface INSUFFICIENT_DATA status explicitly so the summarizer does not fabricate
    crag = ba_output.get("crag_status", "")
    qual_summary = ba_output.get("qualitative_summary", "N/A")
    if "INSUFFICIENT_DATA" in str(qual_summary) or str(crag).upper() == "INCORRECT":
        parts.append(
            "⚠ DATA AVAILABILITY: The Business Analyst knowledge base returned NO documents "
            "for this ticker (CRAG status=INCORRECT, confidence=0.00). "
            "The Investment Thesis & Business Quality section MUST acknowledge this data gap. "
            "DO NOT fabricate competitive moat analysis, revenue breakdowns, cash balances, "
            "services percentages, or any qualitative metrics — none are available."
        )

    parts.append(f"Qualitative Summary: {qual_summary}")

    # Company overview — expose market_cap and key financials so the LLM doesn't invent them
    co = ba_output.get("company_overview") or {}
    if co:
        market_cap = co.get("market_cap")
        pe_ratio = co.get("pe_ratio")
        profit_margin = co.get("profit_margin")
        name = co.get("name")
        sector = co.get("sector")
        industry = co.get("industry")
        co_parts = []
        if name:
            co_parts.append(f"name={name}")
        if sector:
            co_parts.append(f"sector={sector}")
        if industry:
            co_parts.append(f"industry={industry}")
        if market_cap is not None:
            co_parts.append(f"market_cap={market_cap}")
        if pe_ratio is not None:
            co_parts.append(f"pe_ratio={pe_ratio}")
        if profit_margin is not None:
            co_parts.append(f"profit_margin={profit_margin}")
        if co_parts:
            parts.append("Company Overview: " + ", ".join(co_parts))

    sv = ba_output.get("sentiment_verdict")
    if sv and isinstance(sv, dict):
        parts.append(
            f"Sentiment Verdict: signal={sv.get('signal')}, "
            f"confidence={sv.get('confidence')}, "
            f"rationale={sv.get('rationale')}"
        )

    moat = ba_output.get("competitive_moat")
    if moat and isinstance(moat, dict):
        parts.append(
            f"Competitive Moat: rating={moat.get('rating')}, "
            f"strengths={moat.get('key_strengths') or moat.get('strengths')}, "
            f"vulnerabilities={moat.get('vulnerabilities')}"
        )

    s = ba_output.get("sentiment") or {}
    if s:
        bullish = s.get('bullish_pct')
        bearish = s.get('bearish_pct')
        neutral = s.get('neutral_pct')
        trend = s.get('trend')
        interp = s.get('sentiment_interpretation')
        parts.append(
            f"Sentiment: bullish={bullish}%, bearish={bearish}%, neutral={neutral}%, "
            f"trend={trend}"
        )
        # Add quantified context vs. sector base rate
        if bullish is not None:
            try:
                b_val = float(bullish)
                if b_val > 65:
                    parts.append(f"  → {b_val:.1f}% bullish: materially above ~55% large-cap tech base rate — strong constructive consensus")
                elif b_val > 55:
                    parts.append(f"  → {b_val:.1f}% bullish: modestly above ~55% large-cap tech base rate")
                elif b_val > 45:
                    parts.append(f"  → {b_val:.1f}% bullish: near the ~55% large-cap tech base rate — neutral consensus")
                else:
                    parts.append(f"  → {b_val:.1f}% bullish: below the ~55% large-cap tech base rate — bearish tilt in consensus")
            except (TypeError, ValueError):
                pass
        if interp:
            parts.append(f"  Sentiment Interpretation: {interp}")

    risks = ba_output.get("key_risks") or []
    if risks:
        parts.append("Key Risks (with financial magnitude and mitigations):")
        for r in risks[:5]:
            if isinstance(r, dict):
                severity = r.get('severity', '')
                risk_name = r.get('risk', str(r))
                mitigation = r.get('mitigation_observed')
                desc = r.get('description', '')
                risk_line = f"  • [{severity}] {risk_name}"
                if desc:
                    risk_line += f" — {desc}"
                if mitigation:
                    risk_line += f" | Mitigation: {mitigation}"
                parts.append(risk_line)

    qa = ba_output.get("qualitative_analysis")
    if qa and isinstance(qa, dict):
        if qa.get("narrative"):
            parts.append(f"Analysis Narrative: {qa['narrative']}")
        if qa.get("strategic_implication"):
            parts.append(f"Strategic Implication: {qa['strategic_implication']}")
        if qa.get("sentiment_signal"):
            parts.append(f"Sentiment-Document Signal: {qa['sentiment_signal']}")

    # ── Management Guidance & Earnings Highlights ─────────────────────────────
    mg = ba_output.get("management_guidance")
    if mg and isinstance(mg, dict):
        parts.append("Management Guidance & Earnings Highlights:")
        most_recent = mg.get("most_recent_guidance")
        if most_recent:
            parts.append(f"  Most Recent Guidance: {most_recent}")
        highlights = mg.get("earnings_call_highlights") or []
        if highlights:
            parts.append("  Earnings Call Highlights:")
            for h in highlights[:4]:
                # Strip trailing [chunk_id] token if present
                h_clean = re.sub(r"\s*\[chunk:[^\]]+\]$", "", str(h)).strip()
                parts.append(f"    - {h_clean}")
        catalysts = mg.get("near_term_catalysts") or []
        if catalysts:
            parts.append("  Near-Term Catalysts:")
            for c in catalysts[:5]:
                if isinstance(c, dict):
                    cat  = c.get("catalyst", "")
                    dirn = c.get("direction", "")
                    tl   = c.get("timeline", "")
                    mag  = c.get("magnitude", "")
                    line = f"    [{dirn}] {cat}"
                    if tl:
                        line += f" | Timeline: {tl}"
                    if mag:
                        line += f" | Magnitude: {mag}"
                    parts.append(line)
        fwd = mg.get("forward_outlook_summary")
        if fwd:
            parts.append(f"  Forward Outlook Summary: {fwd}")

    parts.append(
        f"CRAG Status: {ba_output.get('crag_status')}, "
        f"Confidence: {ba_output.get('confidence', 0):.2f}, "
        f"Fallback: {ba_output.get('fallback_triggered')}"
    )
    return parts


def _build_web_context(web_output: Dict[str, Any]) -> List[str]:
    """Format web search output into prompt context."""
    ticker = web_output.get("ticker", "?")
    parts: List[str] = [f"=== WEB SEARCH OUTPUT: {ticker} ==="]
    parts.append(f"Sentiment Signal: {web_output.get('sentiment_signal', 'N/A')}")
    parts.append(f"Sentiment Rationale: {web_output.get('sentiment_rationale', '')}")

    news = web_output.get("breaking_news") or []
    for item in news[:4]:
        if isinstance(item, dict):
            parts.append(
                f"News: {item.get('title')} "
                f"({item.get('published_date')}) — {item.get('url')}"
            )

    risks = web_output.get("unknown_risk_flags") or []
    for flag in risks[:3]:
        if isinstance(flag, dict):
            parts.append(
                f"Risk Flag [{flag.get('severity')}]: {flag.get('risk')} "
                f"— {flag.get('source_url', '')}"
            )

    comps = web_output.get("competitor_signals") or []
    for sig in comps[:2]:
        if isinstance(sig, dict):
            parts.append(f"Competitor Signal: {sig.get('company')} — {sig.get('signal')}")
    return parts


def _build_fm_context(fm_output: Dict[str, Any]) -> List[str]:
    """Format financial modelling output into readable key=value lines for the prompt.

    Surfaces DCF intrinsic value range, WACC, Comps vs_sector_avg, technicals,
    earnings surprise, dividends, and factor scores (Altman Z, Piotroski, Beneish).
    All numbers were computed deterministically in Python — never LLM-generated.
    """
    ticker = fm_output.get("ticker", "?")
    parts: List[str] = [f"=== FINANCIAL MODELLING OUTPUT: {ticker} ==="]
    parts.append("NOTE: All numbers in this section were computed deterministically in Python "
                 "from PostgreSQL time-series and fundamentals data. They are NOT LLM-generated.")

    # ── Quantitative Summary ──────────────────────────────────────────────────
    qs = fm_output.get("quantitative_summary")
    if qs:
        parts.append(f"Quantitative Summary: {qs}")

    # ── Valuation — DCF ───────────────────────────────────────────────────────
    val = fm_output.get("valuation") or {}
    dcf = val.get("dcf") or {}
    if dcf:
        wacc = dcf.get("wacc_used")
        base  = dcf.get("intrinsic_value_base")
        bear  = dcf.get("intrinsic_value_bear")
        bull  = dcf.get("intrinsic_value_bull")
        parts.append(
            f"DCF Valuation: WACC={wacc}, "
            f"intrinsic_value_base={base}, "
            f"intrinsic_value_bear={bear}, "
            f"intrinsic_value_bull={bull}"
        )
        # Scenario table
        scenario_table = dcf.get("scenario_table") or []
        if scenario_table:
            parts.append("DCF Scenario Table (Bear/Base/Bull):")
            for row in scenario_table:
                if isinstance(row, dict):
                    parts.append(
                        f"  {row.get('scenario')}: rev_growth={row.get('revenue_growth')}, "
                        f"ebit_margin={row.get('ebit_margin')}, wacc={row.get('wacc')}, "
                        f"intrinsic_value={row.get('intrinsic_value')}, "
                        f"probability={row.get('probability')}"
                    )
        # Sensitivity matrix (WACC × terminal growth)
        sensitivity = dcf.get("sensitivity_matrix") or {}
        if sensitivity:
            parts.append("DCF Sensitivity Matrix (WACC rows × terminal_growth columns):")
            for wacc_key, tg_dict in list(sensitivity.items())[:4]:
                if isinstance(tg_dict, dict):
                    row_str = ", ".join(
                        f"tg={tg}: {iv}" for tg, iv in list(tg_dict.items())[:4]
                    )
                    parts.append(f"  WACC={wacc_key}: {row_str}")

    # ── Valuation — Implied Price Range ──────────────────────────────────────
    ipr = val.get("implied_price_range") or {}
    if ipr:
        parts.append(
            f"Implied Price Range: low={ipr.get('low')}, mid={ipr.get('mid')}, high={ipr.get('high')}"
        )

    # ── Valuation — Comps ─────────────────────────────────────────────────────
    comps = val.get("comps") or {}
    if comps:
        peer_group = comps.get("peer_group") or []
        vs_sector  = comps.get("vs_sector_avg") or {}
        parts.append(
            f"Comps: EV/EBITDA={comps.get('ev_ebitda')}, "
            f"P/E(trailing)={comps.get('pe_trailing')}, "
            f"P/E(forward)={comps.get('pe_forward')}, "
            f"P/S={comps.get('ps_ttm') or comps.get('ps_ratio')}, "
            f"EV/Revenue={comps.get('ev_revenue')}"
        )
        if peer_group:
            parts.append(f"  Comps Peer Group: {', '.join(str(p) for p in peer_group[:5])}")
        if vs_sector and isinstance(vs_sector, dict):
            vs_parts = [f"{k}={v}" for k, v in list(vs_sector.items())[:5]]
            parts.append(f"  vs Sector Avg: {', '.join(vs_parts)}")
            # Add interpretation hints for premium/discount
            for metric, vs_val in vs_sector.items():
                if isinstance(vs_val, str):
                    direction = "premium" if "+" in vs_val else ("discount" if "-" in vs_val else "")
                    if direction:
                        parts.append(f"  → {metric}: trading at a {direction} to sector avg ({vs_val})")
        elif vs_sector and isinstance(vs_sector, str):
            parts.append(f"  vs Sector Avg: {vs_sector}")

    # ── Technicals ────────────────────────────────────────────────────────────
    tech = fm_output.get("technicals") or {}
    if tech:
        sma20   = tech.get("sma_20")
        sma50   = tech.get("sma_50")
        sma200  = tech.get("sma_200")
        rsi     = tech.get("rsi_14")
        trend   = tech.get("trend")
        macd    = tech.get("macd")
        bb_upper = tech.get("bb_upper")
        bb_lower = tech.get("bb_lower")
        w52h    = tech.get("w52_high")
        w52l    = tech.get("w52_low")
        parts.append(
            f"Technicals: SMA20={sma20}, SMA50={sma50}, SMA200={sma200}, "
            f"RSI(14)={rsi}, Trend={trend}, MACD={macd}"
        )
        if bb_upper and bb_lower:
            parts.append(f"  Bollinger Bands: upper={bb_upper}, lower={bb_lower}")
        if w52h and w52l:
            parts.append(f"  52-Week Range: low={w52l}, high={w52h}")
        # RSI interpretation
        if rsi is not None:
            try:
                rsi_val = float(rsi)
                if rsi_val > 70:
                    rsi_interp = "OVERBOUGHT (>70) — momentum extended, watch for reversal"
                elif rsi_val > 60:
                    rsi_interp = "BULLISH MOMENTUM (60–70) — uptrend intact but elevated"
                elif rsi_val > 40:
                    rsi_interp = "NEUTRAL (40–60) — no strong directional momentum signal"
                elif rsi_val > 30:
                    rsi_interp = "BEARISH PRESSURE (30–40) — downtrend momentum"
                else:
                    rsi_interp = "OVERSOLD (<30) — momentum deeply depressed, watch for reversal"
                parts.append(f"  → RSI(14) of {rsi_val:.1f}: {rsi_interp}")
            except (TypeError, ValueError):
                pass

    # ── Earnings ──────────────────────────────────────────────────────────────
    earnings = fm_output.get("earnings") or {}
    if earnings:
        actual   = earnings.get("last_eps_actual")
        estimate = earnings.get("last_eps_estimate")
        surprise = earnings.get("surprise_pct")
        beat_str = earnings.get("beat_streak")
        miss_str = earnings.get("miss_streak")
        next_date = earnings.get("next_earnings_date")
        parts.append(
            f"Earnings: last_EPS_actual={actual}, last_EPS_estimate={estimate}, "
            f"surprise={surprise}%, beat_streak={beat_str}, miss_streak={miss_str}"
        )
        if next_date:
            parts.append(f"  Next Earnings Date: {next_date}")
        if surprise is not None:
            try:
                s_val = float(surprise)
                if s_val > 10:
                    parts.append(f"  → Earnings surprise of {s_val:+.1f}%: material upside beat vs consensus estimate")
                elif s_val > 2:
                    parts.append(f"  → Earnings surprise of {s_val:+.1f}%: modest beat vs consensus estimate")
                elif s_val < -10:
                    parts.append(f"  → Earnings surprise of {s_val:+.1f}%: material miss vs consensus estimate")
                elif s_val < -2:
                    parts.append(f"  → Earnings surprise of {s_val:+.1f}%: modest miss vs consensus estimate")
            except (TypeError, ValueError):
                pass

    # ── Dividends ─────────────────────────────────────────────────────────────
    dividends = fm_output.get("dividends") or {}
    if dividends:
        yld    = dividends.get("dividend_yield")
        annual = dividends.get("annual_dividend")
        payout = dividends.get("payout_ratio")
        cagr5  = dividends.get("dividend_growth_5y_cagr")
        if any(v is not None for v in [yld, annual, payout, cagr5]):
            # dividend_yield and payout_ratio are stored as decimals (e.g. 0.0039
            # = 0.39%).  Convert to a human-readable percentage string so the LLM
            # doesn't echo "0.0039%" verbatim in the report.
            def _pct(v, decimals: int = 2) -> str:
                """Format a fractional ratio as a percentage string."""
                if v is None:
                    return "N/A"
                try:
                    f = float(v)
                    # If the stored value is already expressed as a whole-number
                    # percentage (e.g. 39.0 for 39%), leave it; otherwise multiply.
                    if abs(f) < 1.0:
                        f = f * 100
                    return f"{f:.{decimals}f}%"
                except (TypeError, ValueError):
                    return str(v)

            parts.append(
                f"Dividends: yield={_pct(yld)}, annual_dividend={annual}, "
                f"payout_ratio={_pct(payout)}, 5y_CAGR={cagr5}"
            )

    # ── Factor Scores ─────────────────────────────────────────────────────────
    factors = fm_output.get("factor_scores") or {}
    if factors:
        piotroski = factors.get("piotroski_f_score")
        beneish   = factors.get("beneish_m_score")
        altman    = factors.get("altman_z_score")
        if any(v is not None for v in [piotroski, beneish, altman]):
            parts.append(
                f"Factor Scores: Piotroski_F={piotroski}/9, "
                f"Beneish_M={beneish}, Altman_Z={altman}"
            )
        # Altman Z-Score interpretation
        if altman is not None:
            try:
                z_val = float(altman)
                if z_val > 2.99:
                    z_interp = "SAFE ZONE (>2.99) — low financial distress probability"
                elif z_val > 1.81:
                    z_interp = "GREY ZONE (1.81–2.99) — moderate distress concern, monitor balance sheet"
                else:
                    z_interp = "DISTRESS ZONE (<1.81) — elevated financial distress signal"
                parts.append(f"  → Altman Z-Score of {z_val:.2f}: {z_interp}")
            except (TypeError, ValueError):
                pass

    # ── Data Quality ──────────────────────────────────────────────────────────
    dq = fm_output.get("data_quality") or {}
    if dq:
        parts.append(
            f"Data Quality: {dq.get('status')} "
            f"({dq.get('checks_passed')}/{dq.get('checks_total')} checks passed)"
        )
        if dq.get("issues"):
            for issue in (dq.get("issues") or [])[:3]:
                parts.append(f"  • Data issue: {issue}")

    return parts


def summarise_results(
    user_query: str,
    tickers: List[str],
    ba_outputs: List[Dict[str, Any]],
    quant_outputs: List[Dict[str, Any]],
    web_outputs: List[Dict[str, Any]],
    fm_outputs: List[Dict[str, Any]] = [],
    # Legacy single-value params kept for backward-compat (ignored if lists provided)
    ticker: Optional[str] = None,
    ba_output: Optional[Dict[str, Any]] = None,
    quant_output: Optional[Dict[str, Any]] = None,
    web_output: Optional[Dict[str, Any]] = None,
    data_availability: Optional[Dict[str, Any]] = None,
) -> str:
    """Call DeepSeek summarizer with all agent outputs and return the narrative + citations.

    Accepts multi-ticker lists (ba_outputs, quant_outputs, web_outputs, fm_outputs).
    Legacy single-value keyword arguments are accepted for backward-compatibility but
    are ignored when the list parameters are non-empty.
    """
    from .citations import build_citation_block, inject_inline_numbers

    # Resolve effective lists — merge legacy single-value args if lists are empty
    effective_ba     = ba_outputs or ([ba_output] if ba_output else [])
    effective_quant  = quant_outputs or ([quant_output] if quant_output else [])
    effective_web    = web_outputs or ([web_output] if web_output else [])
    effective_fm     = fm_outputs or []
    effective_ticker = tickers[0] if tickers else ticker

    is_comparison = len(tickers) > 1

    # ── 1. Build citation block ───────────────────────────────────────────────
    # For multi-ticker, build citations from all outputs.
    # citations.build_citation_block currently takes single outputs — call once
    # per ticker and merge.
    all_chunk_id_maps: Dict[str, Any] = {}
    ref_blocks: List[str] = []
    offset = 0

    for i, t in enumerate(tickers or [effective_ticker]):
        ba_o     = effective_ba[i]     if i < len(effective_ba)    else None
        quant_o  = effective_quant[i]  if i < len(effective_quant) else None
        web_o    = effective_web[i]    if i < len(effective_web)   else None
        fm_o     = effective_fm[i]     if i < len(effective_fm)    else None

        ref_block, chunk_id_map = build_citation_block(
            ba_output=ba_o,
            quant_output=quant_o,
            web_output=web_o,
            fm_output=fm_o,
            ticker=t,
            index_offset=offset,
        )
        if ref_block:
            ref_blocks.append(ref_block)
        # Merge chunk_id maps — keys are chunk_ids (strings), values are Citation objects
        all_chunk_id_maps.update(chunk_id_map)
        # Advance offset by the number of new citations added
        offset += len(chunk_id_map)

    combined_ref_block = "\n".join(ref_blocks) if ref_blocks else ""
    citation_index_prompt = _build_citation_index_prompt(all_chunk_id_maps)

    # ── 2. Build context for LLM ─────────────────────────────────────────────
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ctx_parts: List[str] = [
        f"Analysis date (UTC): {today_utc}",
        f"Data as-of: {today_utc} — all quantitative metrics are sourced from the most recent "
        f"available PostgreSQL snapshot; qualitative documents are as indexed in Qdrant.",
        f"",
        f"User question: {user_query}",
    ]
    if tickers:
        ctx_parts.append(f"Tickers under analysis: {', '.join(tickers)}")
    elif effective_ticker:
        ctx_parts.append(f"Ticker: {effective_ticker}")
    if is_comparison:
        ctx_parts.append(
            "COMPARISON QUERY: Analyse both companies across every dimension and "
            "provide explicit relative assessments. Do not analyse them in isolation."
        )

    # Inject data availability notice when any tier is degraded
    if data_availability:
        try:
            from .data_availability import availability_notice  # type: ignore[import]
            notice = availability_notice(data_availability, tickers or ([effective_ticker] if effective_ticker else []))
            if notice:
                ctx_parts.append("")
                ctx_parts.append(notice)
        except Exception as exc:
            logger.debug("availability_notice injection failed: %s", exc)

    ctx_parts.append("")

    for ba_o in effective_ba:
        if ba_o:
            ctx_parts.extend(_build_ba_context(ba_o))
            ctx_parts.append("")

    for quant_o in effective_quant:
        if quant_o:
            ctx_parts.extend(_build_quant_context(quant_o))
            ctx_parts.append("")

    for web_o in effective_web:
        if web_o:
            ctx_parts.extend(_build_web_context(web_o))
            ctx_parts.append("")

    for fm_o in effective_fm:
        if fm_o:
            ctx_parts.extend(_build_fm_context(fm_o))
            ctx_parts.append("")

    context = "\n".join(ctx_parts)

    # Build ordered header checklist for the prompt reminder
    _ORDERED_HEADERS = [
        "## Executive Summary",
        "## Company Overview",
        "## Financial Performance",
        "## Key Financial Ratios & Valuation",
        "## Sentiment & Market Positioning",
        "## Growth Prospects",
        "## Risk Factors",
        "## Competitive Landscape",
        "## Management & Governance",
        "## Macroeconomic Factors",
        "## Analyst Verdict",
    ]
    header_checklist = "\n".join(f"      {h}" for h in _ORDERED_HEADERS)

    prompt = (
        f"{_SUMMARIZER_SYSTEM}\n\n"
        f"{citation_index_prompt}\n"
        f"{context}\n\n"
        f"Write the investment research note now. "
        f"CRITICAL REMINDERS — read these before writing your first word:\n"
        f"  (A) Begin IMMEDIATELY with '## Executive Summary' — no title, no date, no ticker header before it.\n"
        f"      DO NOT write 'AAPL |', 'MSFT |', or any 'Ticker | Note | Date' line anywhere in the body.\n"
        f"  (B) You MUST produce ALL 11 section headers below, in this EXACT order, before stopping:\n"
        f"{header_checklist}\n"
        f"      Omitting even one header = report rejected. Do not merge or collapse sections.\n"
        f"  (C) NO bullet points, hyphens, or numbered lists anywhere in the body.\n"
        f"  (D) EVERY sentence with a number or assertion MUST end with [N] from the index above.\n"
        f"      Example: 'The Piotroski F-Score of 9/9 signals exceptional financial health [2].'\n"
        f"      If a paragraph has no [N] citations it will be rejected.\n"
        f"  (E) ABSOLUTELY NO buy/sell/hold ratings, price targets, or 'Strong Buy' language anywhere.\n"
        f"      The Analyst Verdict expresses directional views WITHOUT any rating or target price.\n"
        f"  (F) NO disclaimer, 'prepared by', or closing remarks after ## Analyst Verdict.\n"
        f"  (G) BANNED WORDS — never write: robust, compelling, solid, impressive, notable, remarkable,\n"
        f"      standout, well-positioned, healthy, exceptional results, poised, exciting, promising,\n"
        f"      outstanding, stellar, strong performance, significant growth. Replace each with a\n"
        f"      specific number or mechanism.\n\n"
        f"## Executive Summary"
    )

    # ── 3. Call LLM ──────────────────────────────────────────────────────────
    # deepseek-r1:8b at ~10-15 tok/s (think=False) on Apple Silicon:
    #   single:     5000 tok ≈ 330-500s + ~60s prefill ≈ 6-10 min  → target 10-20 min total
    #   comparison: 6000 tok ≈ 400-600s + ~60s prefill ≈ 7-11 min
    # num_ctx=16384: the assembled prompt is ~11k tokens; Ollama's default 8192
    # context silently truncates the tail, causing instruction loss.  16384 ensures
    # the full prompt is visible.  Prefill adds ~30-40s but prevents correctness issues.
    max_tokens = 6000 if is_comparison else 5000
    try:
        raw     = _ollama_generate(_SUMMARIZER_MODEL, prompt, max_tokens=max_tokens, temperature=0.2, timeout=_SUMMARIZER_TIMEOUT, num_ctx=16384)
        cleaned = _strip_think(raw).strip()
        if not cleaned:
            cleaned = "Summary unavailable (LLM returned empty response)."
    except Exception as exc:
        logger.error("Summarizer LLM failed: %s", exc)
        cleaned = f"Summary unavailable ({type(exc).__name__}: {exc})."

    # ── 4. Strip any LLM-generated References / Sources section ─────────────
    cleaned = re.sub(
        r"\n#{1,3}\s*(?:References|Sources|Bibliography).*$",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    ).rstrip()

    # ── 4b. Strip any preamble lines before the first ## header ──────────────
    # If the LLM wrote a title / date / ticker block before "## Executive Summary",
    # remove it so the output starts cleanly at the first section header.
    first_header = re.search(r"^## ", cleaned, flags=re.MULTILINE)
    if first_header and first_header.start() > 0:
        cleaned = cleaned[first_header.start():]

    # ── 4c. Enforce header whitelist ─────────────────────────────────────────
    # Remove any ## or ### header the LLM invented that is not in the permitted set.
    # Permitted set (lowercase, stripped for matching):
    _PERMITTED_HEADERS = {
        "executive summary",
        "company overview",
        "financial performance",
        "key financial ratios & valuation",
        "key financial ratios and valuation",
        "sentiment & market positioning",
        "sentiment and market positioning",
        "growth prospects",
        "risk factors",
        "competitive landscape",
        "management & governance",
        "management and governance",
        "macroeconomic factors",
        "analyst verdict",
    }

    def _strip_illegal_headers(text: str) -> str:
        lines = text.split("\n")
        out = []
        for line in lines:
            stripped = line.strip()
            # ── Markdown ## / ### headers ────────────────────────────────────
            m = re.match(r"^(#{2,3})\s+(.+)$", stripped)
            if m:
                header_text = m.group(2).strip().rstrip(":").lower()
                if header_text not in _PERMITTED_HEADERS:
                    # Replace illegal header with blank line so the following
                    # paragraph is absorbed into the current section.
                    out.append("")
                    continue
                # Normalise to ## regardless of whether LLM wrote ###
                out.append(f"## {m.group(2).strip()}")
                continue
            # ── Numbered outline headers: "1. Company Overview", "2. Financial Analysis" ──
            # The freeform garbage section the model emits after </think> uses
            # numeric outline style instead of markdown.  Strip them entirely.
            n = re.match(r"^\d{1,2}\.\s+(.+)$", stripped)
            if n:
                header_candidate = n.group(1).strip().rstrip(":").lower()
                if header_candidate in _PERMITTED_HEADERS:
                    # It's a canonical section but written as "3. Risk Factors" —
                    # drop it too; the model should already have emitted it as ##.
                    out.append("")
                    continue
                # Non-canonical numbered header — also drop.
                out.append("")
                continue
            out.append(line)
        return "\n".join(out)

    cleaned = _strip_illegal_headers(cleaned)

    # ── 4c-ii. Audit for missing required sections ────────────────────────────
    # Log a warning for every required section the LLM omitted so failures are
    # immediately visible in the run logs. (We cannot inject content for missing
    # sections without fabricating data, so this is diagnostic only.)
    _REQUIRED_SECTION_LABELS = [
        ("executive summary",               "## Executive Summary"),
        ("company overview",                "## Company Overview"),
        ("financial performance",           "## Financial Performance"),
        ("key financial ratios",            "## Key Financial Ratios & Valuation"),
        ("sentiment",                       "## Sentiment & Market Positioning"),
        ("growth prospects",                "## Growth Prospects"),
        ("risk factors",                    "## Risk Factors"),
        ("competitive landscape",           "## Competitive Landscape"),
        ("management",                      "## Management & Governance"),
        ("macroeconomic",                   "## Macroeconomic Factors"),
        ("analyst verdict",                 "## Analyst Verdict"),
    ]
    cleaned_lower = cleaned.lower()
    missing_sections = [
        label for key, label in _REQUIRED_SECTION_LABELS
        if key not in cleaned_lower
    ]
    if missing_sections:
        logger.warning(
            "Summarizer output is missing %d required section(s): %s",
            len(missing_sections),
            ", ".join(missing_sections),
        )
    # The LLM sometimes ignores RULE 0 and injects a "Strong Buy" paragraph or
    # a price target sentence. Remove any paragraph that contains these patterns.
    _REC_PATTERN = re.compile(
        r"(strong\s+buy|strong\s+sell|buy\s+rating|sell\s+rating|hold\s+rating"
        r"|outperform|underperform|overweight|underweight|market\s+perform"
        r"|target\s+price|price\s+target|price\s+objective|12.month\s+target"
        r"|upside\s+to\s+\$|downside\s+to\s+\$|we\s+initiate|we\s+recommend"
        r"|recommended\s+as\s+a|is\s+recommended\s+as|prepared\s+by"
        r"|this\s+report\s+is\s+intended\s+for\s+informational"
        r"|does\s+not\s+constitute\s+a\s+recommendation"
        r"|investors\s+should\s+conduct\s+their\s+own\s+due\s+diligence"
        # Patterns from deepseek-r1 freeform garbage section:
        r"|bloomberg\s+consensus"
        r"|dcf\s+terminal\s+value"
        r"|dcf\s+valuation"
        r"|intrinsic\s+value\s*[:=\$]"
        r"|fair\s+value\s*[:=\$]"
        r"|recommendation\s*:"
        r"|investment\s+recommendation"
        r"|analyst\s+recommendation"
        r"|rating\s*:"
        r"|consensus\s+estimate"
        r"|wall\s+street\s+consensus"
        r"|analyst\s+consensus"
        r"|\bEPS\s+estimate\b"
        r"|forward\s+guidance\s+of\s+\$"
        r"|(?:12|12-month)\s*(?:price\s*)?(?:target|PT)\s*[:=\$]"
        r"|\bPT\s*[:=]\s*\$"
        r"|bear\s+case\s*[:=\$]"
        r"|bull\s+case\s*[:=\$]"
        r"|base\s+case\s*[:=\$]"
        r"|upside\s+case\s*[:=\$]"
        r"|downside\s+scenario"
        r"|note\s*:\s*this\s+report"
        r"|disclaimer\s*:"
        r"|this\s+(?:research\s+)?note\s+(?:is|was)\s+prepared"
        r"|not\s+(?:a\s+)?(?:solicitation|investment\s+advice)"
        r"|for\s+informational\s+purposes\s+only)",
        re.IGNORECASE,
    )

    def _strip_rec_paragraphs(text: str) -> str:
        """Remove any paragraph that contains buy/sell/hold, price target, or
        disclaimer language — including the freeform garbage that deepseek-r1
        emits after the </think> closing tag."""
        paragraphs = re.split(r"\n{2,}", text)
        clean_paras = []
        for para in paragraphs:
            # Never strip a ## header paragraph (those are handled elsewhere)
            if re.match(r"^\s*##", para):
                clean_paras.append(para)
                continue
            if _REC_PATTERN.search(para):
                logger.debug("Stripped recommendation/disclaimer paragraph: %.120s…", para)
                continue
            clean_paras.append(para)
        return "\n\n".join(clean_paras)

    cleaned = _strip_rec_paragraphs(cleaned)

    # ── 4e-pre. Strip LLM-generated metadata lines from body text ────────────
    # The LLM sometimes writes its own "AAPL | Equity Research Note | 2026-02-28"
    # line (or variants with * italic markers) inside the body. Strip these before
    # we inject the canonical subtitle so only one copy appears.
    # Pattern: a line (possibly wrapped in * or **) containing TICKER | ... | DATE
    # or any "| Equity Research Note |" pattern.
    cleaned = re.sub(
        r"^\*{0,2}[A-Z]{1,6}\s*\|\s*Equity Research Note\s*\|[^\n]*\*{0,2}\s*$",
        "",
        cleaned,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    # Also strip standalone date lines like "*2026-02-28*" or "2026-02-28" that
    # appear immediately after the Executive Summary header before body text.
    cleaned = re.sub(
        r"^\*?\d{4}-\d{2}-\d{2}\*?\s*$",
        "",
        cleaned,
        flags=re.MULTILINE,
    )
    # Collapse any triple+ blank lines left behind by the stripping above.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # ── 4e. Inject ticker/date subtitle after the ## Executive Summary header ─
    # Adds a formatted italic subtitle line: "*AAPL | Equity Research Note | 2026-03-01*"
    # This gives the report a professional header without relying on the LLM to produce it.
    subtitle_ticker = (tickers[0] if tickers else effective_ticker) or "N/A"
    # Double newline at end ensures a blank line between subtitle and first sentence.
    subtitle_line   = f"\n*{subtitle_ticker} | Equity Research Note | {today_utc}*\n\n"
    cleaned = re.sub(
        r"(^## Executive Summary[^\n]*\n)",
        r"\1" + subtitle_line,
        cleaned,
        count=1,
        flags=re.MULTILINE,
    )

    # ── 4f. Strip banned words that slipped through ──────────────────────────
    # The LLM frequently ignores RULE 4 despite explicit instruction.
    # Actively replace banned adjectives/phrases with neutral analyst equivalents
    # so they never appear in the final output regardless of model compliance.
    # Each tuple: (regex pattern, replacement)
    _BANNED_WORD_REPLACEMENTS = [
        # Multi-word phrases first (more specific → less specific)
        (r"\bexceptional results\b",    "the reported results"),
        (r"\bstrong performance\b",     "the reported performance"),
        (r"\bsignificant growth\b",     "the recorded growth rate"),
        (r"\bwell-positioned\b",        "positioned by the data"),
        (r"\brobust margins\b",         "the reported margins"),
        (r"\brobust\b",                 "measured"),
        (r"\bcompelling\b",             "noteworthy"),
        (r"\bsolid\b",                  "reported"),
        (r"\bimpressive\b",             "recorded"),
        (r"\bnotable\b",                "observed"),
        (r"\bremarkable\b",             "documented"),
        (r"\bstandout\b",               "differentiated"),
        # "healthy" is intentionally NOT in the blanket replacement list because it
        # collides with legitimate analytical phrases like "healthy ROE" or "financial
        # health" where a generic substitution ("within-range") produces nonsensical
        # or misleading output.  The SUMMARIZER_SYSTEM RULE 4 prompt already lists it
        # as banned so the model should avoid it; the post-processing filter is only
        # a safety net and must not corrupt well-formed analyst language.
        (r"\bpoised\b",                 "positioned"),
        (r"\bexciting\b",               "material"),
        (r"\bpromising\b",              "forward-looking"),
        (r"\boutstanding\b",            "above-median"),
        (r"\bstellar\b",                "above-median"),
    ]
    banned_count = 0
    for pattern, replacement in _BANNED_WORD_REPLACEMENTS:
        new_cleaned, n = re.subn(pattern, replacement, cleaned, flags=re.IGNORECASE)
        if n:
            banned_count += n
            cleaned = new_cleaned
    if banned_count:
        logger.warning(
            "Banned words detected and replaced in summarizer output (%d substitution(s)).",
            banned_count,
        )

    # ── 5. Replace any residual chunk_id tokens with [N] numbers ─────────────
    cleaned = inject_inline_numbers(cleaned, all_chunk_id_maps)

    # ── 6. Append the references block ───────────────────────────────────────
    if combined_ref_block:
        cleaned = cleaned + "\n" + combined_ref_block

    return cleaned


__all__ = ["plan_query", "summarise_results"]
