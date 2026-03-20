"""LLM client for the orchestration layer.

Used by:
  - planner_node: parse user query → structured plan + tool selection
  - summarizer_node: synthesise all agent outputs → final narrative with citations

Planner uses deepseek-chat for planning.
Summarizer uses deepseek-chat for generating the final report.
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

_DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
_DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
_PLANNER_MODEL = os.getenv(
    "ORCHESTRATION_PLANNER_MODEL",
    "deepseek-chat",
)
_SUMMARIZER_MODEL = os.getenv(
    "ORCHESTRATION_SUMMARIZER_MODEL",
    "deepseek-chat",
)
_TRANSLATION_MODEL = os.getenv("ORCHESTRATION_TRANSLATION_MODEL", _SUMMARIZER_MODEL)
_REQUEST_TIMEOUT_ENV = os.getenv("ORCHESTRATION_LLM_TIMEOUT", "").strip()
_REQUEST_TIMEOUT: Optional[int] = int(_REQUEST_TIMEOUT_ENV) if _REQUEST_TIMEOUT_ENV else 60
_SUMMARIZER_TIMEOUT_ENV = os.getenv("ORCHESTRATION_SUMMARIZER_TIMEOUT", "").strip()
_SUMMARIZER_TIMEOUT: Optional[int] = int(_SUMMARIZER_TIMEOUT_ENV) if _SUMMARIZER_TIMEOUT_ENV else 1200


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


def _deepseek_generate(
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    timeout: Optional[int] = None,
    system_prompt: Optional[str] = None,
    return_reasoning: bool = False,
) -> Any:
    """Call DeepSeek API and return the raw response text.

    If ``return_reasoning`` is True, returns a tuple ``(content, reasoning_content)``
    where ``reasoning_content`` is the model's chain-of-thought string (non-empty
    only when using deepseek-reasoner; empty string otherwise).
    """
    headers = {
        "Authorization": f"Bearer {_DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    _timeout = timeout if timeout is not None else _REQUEST_TIMEOUT
    try:
        resp = requests.post(
            f"{_DEEPSEEK_BASE_URL}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=_timeout,
        )
        resp.raise_for_status()
        message = resp.json().get("choices", [{}])[0].get("message", {})
        content: str = message.get("content", "")
        if return_reasoning:
            reasoning: str = message.get("reasoning_content", "") or ""
            return content, reasoning
        return content
    except requests.exceptions.ConnectionError:
        logger.error("DeepSeek API not reachable at %s", _DEEPSEEK_BASE_URL)
        raise
    except requests.exceptions.Timeout:
        logger.error("DeepSeek request timed out after %ss", _timeout)
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

AGENT: stock_research
README SUMMARY: The earnings-call and broker-report intelligence layer. Uses a RAG pipeline over
earnings call transcripts and broker research reports stored in Neo4j (with Ollama embeddings),
falling back to local PDFs. 8-step pipeline: load → quality-parse → broker labelling → transcript
feature extraction → FAISS/vector retrieval → LLM synthesis. Handles:
  - Earnings call transcript analysis: management tone, guidance language, Q&A behaviour,
    forward-looking statements, analyst pushback, key themes across calls
  - Broker report synthesis: consensus ratings, price targets, bull/bear thesis comparison,
    analyst disagreements, upgrade/downgrade drivers
  - Transcript-vs-broker comparison: whether broker sentiment aligns with management commentary
  - Quarter-over-quarter narrative changes in earnings calls
  - Identification of red flags or positive signals in management language
SUPPORTED TICKERS: Any ticker with earnings call transcripts or broker reports ingested into
  Neo4j (AAPL, MSFT, GOOGL, TSLA, NVDA have data); falls back to local PDFs if available
DO NOT USE FOR: Financial ratios, DCF valuation, real-time news, technical analysis, price charts

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
  "run_stock_research": <true|false>,
  "complexity": <1|2|3>,
  "reasoning": "<2-3 sentences explaining tool selection and multi-ticker handling>",
  "chart_hints": ["<chart_type1>", "<chart_type2>"]
}

=== CHART_HINTS RULES ===
"chart_hints" is an array of zero or more chart type strings selected from the list below.
Choose ONLY charts that are directly relevant to what the user asked for.
Do NOT include a chart just because its data might be available — it must match the user's intent.

Available chart types and when to include them:
  "price_history"      — Include when: user asks about share price, price chart, historical prices,
                         candlestick, OHLC, volume, 52-week range, support/resistance levels
  "price_performance"  — Include when: user asks about stock performance vs market, relative
                         performance, alpha, outperformance, vs S&P 500, indexed performance
  "revenue_trend"      — Include when: user asks about revenue growth, sales trend, top line,
                         quarterly revenue, financial results
  "margin_trends"      — Include when: user asks about margin trends, gross margin, EBIT margin,
                         profitability trends, margin expansion/compression
  "ebitda_trend"       — Include when: user asks about EBITDA, EBIT, operating income,
                         operating profit trend
  "eps_trend"          — Include when: user asks about EPS, earnings per share, earnings trend,
                         beat/miss history, quarterly earnings
  "dcf_scenarios"      — Include when: user asks about DCF, intrinsic value, fair value, valuation,
                         overvalued/undervalued, price target, discounted cash flow, Bear/Base/Bull
  "sensitivity_heatmap"— Include when: user asks about WACC sensitivity, scenario analysis, valuation
                         sensitivity, or any comprehensive valuation question
  "football_field"     — Include when: user asks about valuation range, price target range,
                         multiple methodologies, sum-of-parts, Bear/Base/Bull price range
  "dcf_waterfall"      — Include when: user asks about DCF bridge, Bear/Base/Bull scenarios,
                         intrinsic value scenario comparison
  "fcf_trend"          — Include when: user asks about free cash flow, FCF, cash generation,
                         cash conversion, capital returns
  "technicals"         — Include when: user asks about technicals, RSI, MACD, moving averages,
                         support/resistance, Bollinger Bands, trend analysis, price action, "should I buy"
  "sentiment_donut"    — Include when: user asks about sentiment, market opinion, investor mood,
                         bullish/bearish positioning, or qualitative assessment
  "peer_comps"         — Include when: user asks about peer comparison, relative valuation,
                         sector comparison, how X compares to competitors, EV/EBITDA vs peers
  "moe_consensus"      — Include when: user asks about analyst views, price range, consensus,
                         different scenarios, bull/bear case, or any comprehensive valuation
  "factor_scorecard"   — Include when: user asks about quality factors, financial health score,
                         Piotroski, ROE, ROIC, gross margin, Sharpe ratio, or overall stock quality;
                         also include for any comprehensive/fundamental analysis query
  "factor_radar"       — Alias for factor_scorecard — use either name
  "altman_z"           — Include when: user asks about bankruptcy risk, financial distress,
                         Altman Z-Score, solvency, credit risk, or company financial health

For comprehensive/full analysis queries, include ALL relevant chart types.
For narrow queries (e.g. "what is AAPL's RSI?"), include only the directly relevant chart ("technicals").
Default to [] if no data-driven visualisation is clearly warranted.

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
- run_stock_research: true when the question explicitly asks about earnings call transcripts,
  broker reports, analyst reports, management commentary, earnings call Q&A, guidance language,
  broker consensus, price targets from brokers, analyst upgrades/downgrades, or any synthesis
  of what management said vs. what brokers think.
  Also enable for "complete analysis", "full analysis", "comprehensive analysis", or "deep dive"
  queries — these benefit from transcript and broker report context.
  Do NOT enable for standard financial ratios, DCF, real-time news, or technical analysis.

COMPREHENSIVE QUERY RULE: If the query contains any of: "fundamental analysis",
"complete analysis", "full analysis", "comprehensive analysis", "deep dive", "full report",
"complete report" — set ALL FIVE of run_business_analyst, run_quant_fundamental,
run_financial_modelling, run_stock_research to true. Only keep run_web_search=false unless news is
explicitly requested.

Always enable at least one tool. If the question is ambiguous, enable both
business_analyst and quant_fundamental.
For "complete", "full", "comprehensive", or "fundamental analysis" queries, enable ALL FIVE agents.
"""


# ── Semantic Router (1C) ──────────────────────────────────────────────────────
# Cache known plan outputs keyed by their all-MiniLM-L6-v2 embedding.
# On a future query, if cosine similarity > threshold we return the cached plan
# directly without calling the LLM — saving ~3s per repeat query.

SEMANTIC_ROUTER_THRESHOLD: float = float(
    os.getenv("SEMANTIC_ROUTER_THRESHOLD", "0.85")
)
_SEMANTIC_ROUTER_MAX_CACHE: int = int(
    os.getenv("SEMANTIC_ROUTER_MAX_CACHE", "500")
)


def _router_embed(text: str) -> Optional[List[float]]:
    """Embed text with all-MiniLM-L6-v2 for the semantic router.

    Returns None if sentence_transformers is unavailable (graceful degradation).
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        # Cache the model at module level to avoid repeated disk loads
        if not hasattr(_router_embed, "_model"):
            _old_hf = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                _router_embed._model = SentenceTransformer("all-MiniLM-L6-v2")  # type: ignore[attr-defined]
            finally:
                if _old_hf is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = _old_hf
        vec = _router_embed._model.encode(text, normalize_embeddings=True).tolist()  # type: ignore[attr-defined]
        return vec
    except Exception as exc:
        logger.debug("[semantic_router] Embedding unavailable: %s", exc)
        return None


def _router_cosine(a: List[float], b: List[float]) -> float:
    """Dot product of two L2-normalised vectors (= cosine similarity)."""
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class _SemanticRouter:
    """In-process LRU-style cache mapping query embeddings → plan dicts.

    Thread-safe for concurrent requests (uses a list + lock).
    """

    def __init__(self, max_size: int = _SEMANTIC_ROUTER_MAX_CACHE) -> None:
        import threading
        self._entries: List[Any] = []  # List[Tuple[List[float], Dict]]
        self._max_size = max_size
        self._lock = threading.Lock()

    def lookup(self, query: str) -> Optional[Dict[str, Any]]:
        """Return a cached plan if a similar query was seen before."""
        vec = _router_embed(query)
        if vec is None:
            return None
        with self._lock:
            for stored_vec, plan in self._entries:
                if _router_cosine(vec, stored_vec) >= SEMANTIC_ROUTER_THRESHOLD:
                    return dict(plan)  # return a copy
        return None

    def store(self, query: str, plan: Dict[str, Any]) -> None:
        """Persist a (query-embedding, plan) pair for future lookups."""
        vec = _router_embed(query)
        if vec is None:
            return
        with self._lock:
            # Evict oldest entry if at capacity
            if len(self._entries) >= self._max_size:
                self._entries.pop(0)
            self._entries.append((vec, dict(plan)))


_semantic_router = _SemanticRouter()


# ── C2: Dynamic Few-Shot helpers ──────────────────────────────────────────────

def _fetch_top_successful_queries(limit: int = 5) -> List[Dict[str, Any]]:
    """C2: Query PostgreSQL query_logs for recent high-rated queries.

    Returns up to `limit` rows where overall_rating = 1 (positive feedback),
    ordered by recorded_at DESC.  Each row is returned as a dict with keys:
      user_query, agent_outputs (partial), plan (if stored).

    Falls back to an empty list if the DB is unavailable or the table schema
    differs — this keeps the planner functional even without any prior runs.
    """
    try:
        import psycopg2  # type: ignore[import]
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "airflow"),
            user=os.getenv("POSTGRES_USER", "airflow"),
            password=os.getenv("POSTGRES_PASSWORD", "airflow"),
        )
        rows: List[Dict[str, Any]] = []
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_query, agent_outputs, plan
                FROM query_logs
                WHERE overall_rating = 1
                ORDER BY recorded_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            for row in cur.fetchall():
                user_query, agent_outputs_raw, plan_raw = row
                if not user_query:
                    continue
                entry: Dict[str, Any] = {"user_query": str(user_query)}
                # Parse JSON columns if they are strings
                try:
                    entry["agent_outputs"] = (
                        json.loads(agent_outputs_raw)
                        if isinstance(agent_outputs_raw, str)
                        else (agent_outputs_raw or {})
                    )
                except Exception:
                    entry["agent_outputs"] = {}
                try:
                    entry["plan"] = (
                        json.loads(plan_raw)
                        if isinstance(plan_raw, str)
                        else (plan_raw or {})
                    )
                except Exception:
                    entry["plan"] = {}
                rows.append(entry)
        conn.close()
        return rows
    except Exception as exc:
        logger.debug("[planner/c2] Could not fetch few-shot examples: %s", exc)
        return []


def _build_few_shot_block(examples: List[Dict[str, Any]]) -> str:
    """Format successful past queries as a few-shot prompt block.

    Each example shows:
      User question: <query>
      JSON plan:     <plan JSON>

    This primes the LLM to produce the same JSON schema and routing decisions.
    """
    if not examples:
        return ""
    lines = ["\n=== EXAMPLES OF CORRECT PLANS (from recent successful queries) ==="]
    for i, ex in enumerate(examples, start=1):
        plan = ex.get("plan") or {}
        if not plan:
            continue
        # Keep only the schema-relevant fields to avoid polluting the prompt
        clean_plan = {
            k: plan[k] for k in (
                "tickers", "ticker", "intent",
                "run_business_analyst", "run_quant_fundamental",
                "run_web_search", "run_financial_modelling",
                "complexity", "reasoning",
            ) if k in plan
        }
        if not clean_plan:
            continue
        try:
            plan_str = json.dumps(clean_plan, separators=(",", ":"))
        except Exception:
            continue
        lines.append(f"\nExample {i}:")
        lines.append(f"User question: {ex['user_query']}")
        lines.append(f"JSON plan: {plan_str}")
    lines.append("=== END EXAMPLES ===\n")
    return "\n".join(lines)


def _infer_chart_hints(query_lower: str, plan: Dict[str, Any]) -> List[str]:
    """Keyword-based chart_hints inference — used as fallback when LLM omits the field.

    Maps query intent to chart type strings that the Streamlit UI knows how to render.
    Deterministic, never raises.
    """
    hints: List[str] = []
    run_fm = plan.get("run_financial_modelling", False)

    # Price history / OHLCV
    if any(kw in query_lower for kw in [
        "price chart", "candlestick", "ohlc", "historical price", "price history",
        "52 week", "52w", "volume", "share price",
    ]):
        hints.append("price_history")

    # Price performance vs benchmark
    if any(kw in query_lower for kw in [
        "vs s&p", "vs market", "relative performance", "outperform", "alpha",
        "indexed", "benchmark", "performance vs",
    ]):
        hints.append("price_performance")

    # Revenue trend
    if any(kw in query_lower for kw in [
        "revenue", "sales", "top line", "quarterly revenue", "financial results",
    ]):
        hints.append("revenue_trend")

    # Margin trends
    if any(kw in query_lower for kw in [
        "margin", "gross margin", "ebit margin", "profitability trend",
        "margin expansion", "margin compression",
    ]):
        hints.append("margin_trends")

    # EBITDA / EBIT trend
    if any(kw in query_lower for kw in [
        "ebitda", "ebit", "operating income", "operating profit",
    ]):
        hints.append("ebitda_trend")

    # EPS trend
    if any(kw in query_lower for kw in [
        "eps", "earnings per share", "earnings trend", "beat", "miss",
        "quarterly earnings", "income",
    ]):
        hints.append("eps_trend")

    # DCF / valuation
    if run_fm and any(kw in query_lower for kw in [
        "dcf", "intrinsic", "fair value", "overvalued", "undervalued",
        "price target", "valuation", "discounted cash", "bear", "bull",
    ]):
        hints.append("dcf_scenarios")
        hints.append("sensitivity_heatmap")
        hints.append("football_field")

    # Football field (valuation range)
    if any(kw in query_lower for kw in [
        "football field", "valuation range", "price target range", "sum of parts",
    ]):
        if "football_field" not in hints:
            hints.append("football_field")

    # WACC sensitivity — only if explicitly mentioned
    if "wacc" in query_lower or "sensitivity" in query_lower:
        if "sensitivity_heatmap" not in hints:
            hints.append("sensitivity_heatmap")

    # FCF trend
    if any(kw in query_lower for kw in [
        "free cash flow", "fcf", "cash generation", "cash conversion", "capital return",
    ]):
        hints.append("fcf_trend")

    # Technicals
    if any(kw in query_lower for kw in [
        "technical", "rsi", "macd", "sma", "moving average", "bollinger",
        "support", "resistance", "trend", "momentum", "chart",
        "buy", "sell", "entry", "breakout",
    ]):
        hints.append("technicals")

    # Quarterly / revenue trends (backward compat)
    if any(kw in query_lower for kw in [
        "quarterly", "growth", "q1", "q2", "q3", "q4", "profit",
    ]):
        if "revenue_trend" not in hints:
            hints.append("revenue_trend")

    # Sentiment
    if any(kw in query_lower for kw in [
        "sentiment", "bullish", "bearish", "market opinion", "investor",
        "moat", "competitive", "qualitative",
    ]):
        hints.append("sentiment_donut")

    # Peer comps
    if any(kw in query_lower for kw in [
        "compare", "vs ", "versus", "peer", "sector", "competitor", "relative",
        "cheaper", "expensive", "premium", "discount",
    ]):
        hints.append("peer_comps")

    # MoE consensus — always if FM is running
    if run_fm:
        hints.append("moe_consensus")

    # Factor scorecard + Altman Z — always if QF is running
    if plan.get("run_quant_fundamental", False):
        hints.append("factor_scorecard")
        hints.append("altman_z")

    # Comprehensive / full analysis → show everything
    if any(kw in query_lower for kw in [
        "complete analysis", "full analysis", "fundamental analysis",
        "comprehensive", "deep dive", "full report",
    ]):
        hints = list(dict.fromkeys([
            "price_history", "price_performance",
            "revenue_trend", "margin_trends", "ebitda_trend", "eps_trend",
            "football_field", "dcf_scenarios", "sensitivity_heatmap",
            "peer_comps", "fcf_trend",
            "technicals", "sentiment_donut", "factor_scorecard", "altman_z",
            "moe_consensus",
        ]))

    return list(dict.fromkeys(hints))  # deduplicate, preserve order


def plan_query(
    user_query: str,
    worst_case_context: str = "",
) -> Dict[str, Any]:
    """Call DeepSeek planner and return the structured plan dict.

    Before invoking the LLM, a semantic router checks cosine similarity
    (all-MiniLM-L6-v2, 384-dim) against a cache of previously-seen queries.
    If similarity > SEMANTIC_ROUTER_THRESHOLD (default 0.85), the cached
    plan is returned immediately, bypassing the LLM entirely (~50ms vs ~3s).

    C2: Up to 5 recent high-rated query plans are fetched from query_logs and
    injected as few-shot examples into the planner prompt, improving routing
    accuracy over time as user feedback accumulates.

    Returns a safe default plan on any LLM failure.

    The returned plan dict always contains a ``"planner_trace"`` key with the
    model's chain-of-thought reasoning string (empty string if unavailable or
    served from the semantic router cache).
    """
    # --- semantic router (1C) ----------------------------------------------
    cached = _semantic_router.lookup(user_query)
    if cached is not None:
        logger.info("[planner] Semantic router cache hit (similarity>%.2f) — skipping LLM.",
                    SEMANTIC_ROUTER_THRESHOLD)
        cached.setdefault("planner_trace", "")
        return cached

    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    planner_prompt_with_date = (
        f"Today's date (UTC): {today_utc}\n"
        f"Use this date as the reference point when interpreting 'recent', 'latest', or 'current' queries.\n\n"
        f"{_PLANNER_SYSTEM}"
    )

    # C2: fetch dynamic few-shot examples from successful past runs
    few_shot_examples = _fetch_top_successful_queries(limit=5)
    few_shot_block = _build_few_shot_block(few_shot_examples)
    if few_shot_block:
        logger.debug("[planner/c2] Injecting %d few-shot example(s) into planner prompt.",
                     len(few_shot_examples))

    user_content = user_query
    if worst_case_context:
        user_content = (
            f"{user_query}\n\n"
            f"---\n"
            f"{worst_case_context}\n"
            f"---"
        )

    prompt = (
        f"{planner_prompt_with_date}"
        f"{few_shot_block}"
        f"\n\nUser question: {user_content}\n\nJSON plan:"
    )

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
        raw, reasoning = _deepseek_generate(
            _PLANNER_MODEL, prompt, max_tokens=512, temperature=0.1,
            system_prompt=_PLANNER_SYSTEM, return_reasoning=True,
        )
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
        # Attach reasoning trace (empty string if deepseek-chat, non-empty for deepseek-reasoner)
        plan["planner_trace"] = reasoning or ""
        # Store in semantic router cache for future bypass
        _semantic_router.store(user_query, plan)
        # Ensure chart_hints is always present (may be absent in older cached plans)
        if "chart_hints" not in plan:
            plan["chart_hints"] = _infer_chart_hints(_query_lower, plan)
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
            "planner_trace": "",
            "chart_hints": _infer_chart_hints(_query_lower, {
                "run_financial_modelling": _is_comprehensive,
            }),
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

RULE 0b — ALWAYS INCLUDE REFERENCES. A numbered reference list will be automatically appended after your report.
Do NOT write, truncate, or omit the references section. Ensure your report flows naturally into the
references that follow. The reference list is mandatory — your report is incomplete without it.

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

RULE 3 — ZERO FABRICATION. Only use numbers, names, and ratings that appear verbatim in the data block.
Any numeric value (P/E, EV/EBITDA, ROE, ROIC, Piotroski score, Beneish M-Score, Altman Z, RSI, price,
margin, growth rate, etc.) that is NOT explicitly present in the data block is FORBIDDEN.
The system prompt contains illustrative FORMAT EXAMPLES (e.g. "Nx", "Y%") — these are format patterns only,
NOT actual data values. DO NOT use any number from the instructional examples as a data point.
A LOCKED DATA ANCHOR block appears in the data section labelled "=== LOCKED DATA ANCHOR ===".
The values in that block are the sole authoritative source for P/E, EV/EBITDA, ROE, ROIC,
Piotroski F-Score, Beneish M-Score, Altman Z-Score, WACC, and DCF values.
You MUST copy them into the report exactly as stated — no rounding, no substitution from prior knowledge.
If a metric is absent from the data, omit it or state it is unavailable.

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
    premium or discount as a percentage — e.g. "EV/EBITDA of Nx sits Y% above/below the sector median"
    or "P/E of Nx is at the midpoint/high-end/low-end of the 22–38x large-cap tech range". NEVER describe a premium
    or discount in vague language — always name the exact spread in percentage points or absolute turns.
    Use ONLY the exact numbers from the provided data — do NOT substitute illustrative examples for actual figures.
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
     Similarly, if ROIC is materially above WACC (>20pp spread), quantify the spread explicitly using
     the ACTUAL ROIC value from the data (e.g. "ROIC of [actual value]% represents a [actual spread]pp
     spread above the ~[WACC]% WACC, indicative of a durable economic moat") and name the
     mechanism sustaining that spread. NEVER use placeholder or example numbers — use only the exact
     figures from the agent output data.
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
        roe_raw = qf.get('roe')
        roic_raw = qf.get('roic')
        # Format as percentages to avoid LLM misinterpretation of decimal ratios
        roe_fmt = f"{float(roe_raw)*100:.2f}%" if roe_raw is not None else "N/A"
        roic_fmt = f"{float(roic_raw)*100:.2f}%" if roic_raw is not None else "N/A"
        beneish_str = str(beneish) if beneish is not None else "N/A (prior-year financials not in DB)"
        parts.append(
            f"Quality Factors: ROE={roe_fmt} (decimal={roe_raw}), ROIC={roic_fmt} (decimal={roic_raw}), "
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
        if roe_raw is not None:
            try:
                roe_pct = float(roe_raw) * 100
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
        if roic_raw is not None:
            try:
                roic_pct = float(roic_raw) * 100
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
    if not isinstance(s, dict):
        s = {}
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


def _build_data_tables(
    quant_outputs: List[Dict[str, Any]],
    fm_outputs: List[Dict[str, Any]],
    tickers: List[str],
) -> str:
    """Build data tables for the fundamental report.
    
    Creates tables for:
    - Quarterly Revenue (in millions)
    - Annual Revenue (Fiscal Year)
    - Quarterly Operating Earnings (Non-GAAP, per share)
    - Annual Operating Earnings (Non-GAAP, per share)
    - Annual Valuation (Fiscal Year P/E Ratio)
    """
    if not quant_outputs and not fm_outputs:
        return ""
    
    tables: List[str] = []
    
    for i, ticker in enumerate(tickers):
        quant = quant_outputs[i] if i < len(quant_outputs) else {}
        fm = fm_outputs[i] if i < len(fm_outputs) else {}
        
        if not quant and not fm:
            continue
        
        ticker_header = f"### {ticker} Financial Data"
        tables.append(ticker_header)
        
        # Quarterly Revenue table
        qt = quant.get("quarterly_trends") or []
        if qt:
            qt_sorted = sorted(qt, key=lambda r: str(r.get("period", "")))
            q_rev_rows = []
            for row in qt_sorted[-4:]:
                period = str(row.get("period", ""))
                revenue = row.get("revenue")
                if revenue is not None:
                    try:
                        rev_val = float(revenue)
                        q_rev_rows.append(f"| {period} | ${rev_val:,.0f} |")
                    except (TypeError, ValueError):
                        q_rev_rows.append(f"| {period} | N/A |")
            
            if q_rev_rows:
                tables.append("")
                tables.append("**Quarterly Revenue (in millions)**")
                tables.append("| Quarter | Revenue |")
                tables.append("|---------|---------|")
                tables.extend(q_rev_rows)
        
        # Annual Revenue table (from key_metrics or annual data)
        key_metrics = quant.get("key_metrics") or {}
        annual_rev = key_metrics.get("revenue")
        if annual_rev is not None:
            try:
                rev_val = float(annual_rev)
                tables.append("")
                tables.append("**Annual Revenue (Fiscal Year)**")
                tables.append("| Fiscal Year | Revenue (M) |")
                tables.append("|-------------|-------------|")
                tables.append(f"| {key_metrics.get('fiscal_year', 'Latest')} | ${rev_val:,.0f} |")
            except (TypeError, ValueError):
                pass
        
        # Quarterly Operating Earnings (Non-GAAP, per share) - from quarterly_trends
        if qt:
            qt_sorted = sorted(qt, key=lambda r: str(r.get("period", "")))
            q_eps_rows = []
            for row in qt_sorted[-4:]:
                period = str(row.get("period", ""))
                eps = row.get("eps_diluted") or row.get("operating_earnings_per_share")
                if eps is not None:
                    try:
                        eps_val = float(eps)
                        q_eps_rows.append(f"| {period} | ${eps_val:.2f} |")
                    except (TypeError, ValueError):
                        q_eps_rows.append(f"| {period} | N/A |")
            
            if q_eps_rows:
                tables.append("")
                tables.append("**Quarterly Operating Earnings (Non-GAAP, per share)**")
                tables.append("| Quarter | EPS |")
                tables.append("|--------|-----|")
                tables.extend(q_eps_rows)
        
        # Annual Operating Earnings (Non-GAAP, per share)
        annual_eps = key_metrics.get("operating_earnings_per_share") or key_metrics.get("eps_diluted")
        if annual_eps is not None:
            try:
                eps_val = float(annual_eps)
                tables.append("")
                tables.append("**Annual Operating Earnings (Non-GAAP, per share)**")
                tables.append("| Fiscal Year | EPS |")
                tables.append("|-------------|-----|")
                tables.append(f"| {key_metrics.get('fiscal_year', 'Latest')} | ${eps_val:.2f} |")
            except (TypeError, ValueError):
                pass
        
        # Annual Valuation (Fiscal Year P/E Ratio)
        vf = quant.get("value_factors") or {}
        pe_trailing = vf.get("pe_trailing")
        if pe_trailing is not None:
            try:
                pe_val = float(pe_trailing)
                tables.append("")
                tables.append("**Annual Valuation (Fiscal Year P/E Ratio)**")
                tables.append("| Fiscal Year | P/E (TTM) |")
                tables.append("|-------------|-----------|")
                tables.append(f"| {key_metrics.get('fiscal_year', 'Latest')} | {pe_val:.1f}x |")
            except (TypeError, ValueError):
                pass
        
        # Also get P/E from FM agent if available
        comps = fm.get("comps") or {}
        if not pe_trailing:
            pe_fm = comps.get("pe_trailing")
            if pe_fm is not None:
                try:
                    pe_val = float(pe_fm)
                    tables.append("")
                    tables.append("**Annual Valuation (Fiscal Year P/E Ratio)**")
                    tables.append("| Fiscal Year | P/E (TTM) |")
                    tables.append("|-------------|-----------|")
                    tables.append(f"| {key_metrics.get('fiscal_year', 'Latest')} | {pe_val:.1f}x |")
                except (TypeError, ValueError):
                    pass
        
        tables.append("")
    
    return "\n".join(tables)


def _build_stock_research_context(sr_output: Dict[str, Any]) -> List[str]:
    """Format stock-research-agent output into readable lines for the summarizer prompt.

    Surfaces broker consensus, transcript comparison, Q&A behaviour analysis,
    deterministic NLP features, and broker rating distribution.
    All text is LLM-generated by the stock-research pipeline and is clearly
    labelled so the summarizer knows the source.
    """
    ticker = sr_output.get("ticker", "?")
    parts: List[str] = [f"=== STOCK RESEARCH OUTPUT: {ticker} ==="]
    parts.append(
        "NOTE: This section is generated by a separate PDF-based pipeline that reads "
        "broker research reports and earnings call transcripts directly.  All LLM "
        "analyses below include [doc_name p.N] inline citations."
    )

    # ── Error / empty guard ───────────────────────────────────────────────────
    if sr_output.get("error"):
        parts.append(f"Pipeline error: {sr_output['error']}")
        return parts

    # ── Transcript metadata ───────────────────────────────────────────────────
    latest_t   = sr_output.get("latest_transcript", "")
    previous_t = sr_output.get("previous_transcript", "")
    if latest_t or previous_t:
        parts.append(f"Latest transcript: {latest_t} | Previous transcript: {previous_t}")

    # ── Deterministic NLP features ────────────────────────────────────────────
    features = sr_output.get("features") or {}
    latest_feat = features.get("latest") or {}
    prev_feat   = features.get("previous") or {}
    kpi_diff    = features.get("kpi_diff") or {}

    if latest_feat:
        parts.append(
            f"Transcript features (latest): "
            f"kpi_per_1k_words={latest_feat.get('kpi_per_1k_words')}, "
            f"hedge_ratio={latest_feat.get('hedge_ratio')}, "
            f"evasive_count={latest_feat.get('evasive_count')}, "
            f"qa_vs_prep_hedge_delta={latest_feat.get('qa_vs_prep_hedge_delta')}, "
            f"pivot_per_1k_words={latest_feat.get('pivot_per_1k_words')}"
        )
    if prev_feat:
        parts.append(
            f"Transcript features (previous): "
            f"kpi_per_1k_words={prev_feat.get('kpi_per_1k_words')}, "
            f"hedge_ratio={prev_feat.get('hedge_ratio')}"
        )
    if kpi_diff:
        dropped = kpi_diff.get("dropped_kpis") or []
        added   = kpi_diff.get("added_kpis") or []
        if dropped or added:
            parts.append(f"KPI changes: dropped={dropped}, added={added}")

    # ── Broker rating distribution ─────────────────────────────────────────────
    broker_labels_raw = sr_output.get("broker_labels") or {}
    broker_parsed = sr_output.get("broker_parsed") or []

    # broker_labels may be a list (from agent_step4_broker_labels) or a dict.
    # Normalise to list[dict] for counting.
    if isinstance(broker_labels_raw, dict):
        broker_labels_list = list(broker_labels_raw.values())
    elif isinstance(broker_labels_raw, list):
        broker_labels_list = broker_labels_raw
    else:
        broker_labels_list = []

    if broker_labels_list:
        bullish = sum(1 for v in broker_labels_list if (v or {}).get("rating") == "bullish")
        neutral = sum(1 for v in broker_labels_list if (v or {}).get("rating") == "neutral")
        bearish = sum(1 for v in broker_labels_list if (v or {}).get("rating") == "bearish")
        unknown = sum(1 for v in broker_labels_list if (v or {}).get("rating") not in ("bullish","neutral","bearish"))
        parts.append(
            f"Broker rating distribution: {bullish} bullish / {neutral} neutral / "
            f"{bearish} bearish / {unknown} unknown  (from {len(broker_labels_list)} broker reports)"
        )

    # Per-broker price targets and EPS estimates (up to 5 brokers)
    if broker_parsed:
        for bp in broker_parsed[:5]:
            pt   = bp.get("price_target")
            eps  = bp.get("eps_estimates") or {}
            name = bp.get("broker_name", bp.get("institution", "unknown"))
            rating = bp.get("rating", "")
            line = f"  Broker {name} [{rating}]:"
            if pt is not None:
                line += f" price_target={pt}"
            if eps:
                # eps_estimates may be a dict or a list of strings — handle both
                if isinstance(eps, dict):
                    eps_str = ", ".join(f"{k}={v}" for k, v in list(eps.items())[:3])
                elif isinstance(eps, list):
                    eps_str = "; ".join(str(e) for e in eps[:3])
                else:
                    eps_str = str(eps)
                line += f" eps={{{eps_str}}}"
            if line.strip().endswith(":"):
                line += " (no structured data extracted)"
            parts.append(line)

    # ── LLM-generated analyses ────────────────────────────────────────────────
    transcript_cmp = sr_output.get("transcript_comparison", "")
    if transcript_cmp:
        parts.append("Transcript Comparison (LLM, cited):")
        parts.append(transcript_cmp.strip())

    qa_behavior = sr_output.get("qa_behavior", "")
    if qa_behavior:
        parts.append("Q&A Behaviour Analysis (LLM, cited):")
        parts.append(qa_behavior.strip())

    broker_consensus = sr_output.get("broker_consensus", "")
    if broker_consensus:
        parts.append("Broker Consensus (LLM, cited):")
        parts.append(broker_consensus.strip())

    return parts


# ---------------------------------------------------------------------------
# Structured summarisation: JSON-schema-enforced hallucination prevention
# ---------------------------------------------------------------------------

def _build_anchor_dict(
    tickers: List[str],
    quant_outputs: List[Dict[str, Any]],
    fm_outputs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract all authoritative numeric values from agent outputs into a flat dict.

    These are the ONLY numbers the LLM is permitted to use.  Every key maps to
    a value that came directly from the PostgreSQL DB — no training-data numbers.
    """
    anchor: Dict[str, Any] = {}
    for i, t in enumerate(tickers):
        quant_o = quant_outputs[i] if i < len(quant_outputs) else None
        fm_o    = fm_outputs[i]    if i < len(fm_outputs)    else None
        prefix  = t.upper()

        if quant_o:
            vf     = quant_o.get("value_factors")    or {}
            qfact  = quant_o.get("quality_factors")  or {}
            km     = quant_o.get("key_metrics")      or {}
            mr     = quant_o.get("momentum_risk") or quant_o.get("momentum_factors") or {}
            yoy    = quant_o.get("yoy_deltas")        or {}
            qt     = quant_o.get("quarterly_trends")  or []

            # Value factors
            if vf.get("pe_trailing")  is not None: anchor[f"{prefix}_pe_trailing"]  = round(float(vf["pe_trailing"]),  4)
            if vf.get("ev_ebitda")    is not None: anchor[f"{prefix}_ev_ebitda"]    = round(float(vf["ev_ebitda"]),    2)
            if vf.get("p_fcf")        is not None: anchor[f"{prefix}_p_fcf"]        = round(float(vf["p_fcf"]),        2)
            if vf.get("ev_revenue")   is not None: anchor[f"{prefix}_ev_revenue"]   = round(float(vf["ev_revenue"]),   2)

            # Quality factors
            if qfact.get("roe")               is not None: anchor[f"{prefix}_roe_pct"]          = round(float(qfact["roe"]) * 100, 2)
            if qfact.get("roic")              is not None: anchor[f"{prefix}_roic_pct"]         = round(float(qfact["roic"]) * 100, 2)
            if qfact.get("piotroski_f_score") is not None: anchor[f"{prefix}_piotroski"]        = int(qfact["piotroski_f_score"])
            if qfact.get("beneish_m_score")   is not None: anchor[f"{prefix}_beneish_m"]        = round(float(qfact["beneish_m_score"]), 4)
            if qfact.get("altman_z_score")    is not None: anchor[f"{prefix}_altman_z"]         = round(float(qfact["altman_z_score"]),  3)

            # Key metrics
            if km.get("gross_margin")   is not None: anchor[f"{prefix}_gross_margin_pct"]  = round(float(km["gross_margin"]) * 100, 2)
            if km.get("ebit_margin")    is not None: anchor[f"{prefix}_ebit_margin_pct"]   = round(float(km["ebit_margin"])  * 100, 2)
            if km.get("current_ratio")  is not None: anchor[f"{prefix}_current_ratio"]     = round(float(km["current_ratio"]), 2)
            if km.get("debt_to_equity") is not None: anchor[f"{prefix}_debt_to_equity"]    = round(float(km["debt_to_equity"]), 4)

            # Momentum/risk
            if mr.get("beta_60d")         is not None: anchor[f"{prefix}_beta_60d"]          = round(float(mr["beta_60d"]),         4)
            if mr.get("sharpe_ratio_12m") is not None: anchor[f"{prefix}_sharpe_ratio_12m"]  = round(float(mr["sharpe_ratio_12m"]), 4)
            if mr.get("return_12m_pct")   is not None: anchor[f"{prefix}_return_12m_pct"]    = round(float(mr["return_12m_pct"]),   2)
            if mr.get("sma_50")           is not None: anchor[f"{prefix}_sma_50"]            = round(float(mr["sma_50"]),           2)
            if mr.get("sma_200")          is not None: anchor[f"{prefix}_sma_200"]           = round(float(mr["sma_200"]),          2)

            # YoY deltas
            if yoy.get("revenue_yoy_pct")          is not None: anchor[f"{prefix}_revenue_yoy_pct"]          = round(float(yoy["revenue_yoy_pct"]),          2)
            if yoy.get("gross_profit_yoy_pct")      is not None: anchor[f"{prefix}_gross_profit_yoy_pct"]     = round(float(yoy["gross_profit_yoy_pct"]),      2)
            if yoy.get("operating_income_yoy_pct")  is not None: anchor[f"{prefix}_operating_income_yoy_pct"] = round(float(yoy["operating_income_yoy_pct"]),  2)
            if yoy.get("net_income_yoy_pct")        is not None: anchor[f"{prefix}_net_income_yoy_pct"]       = round(float(yoy["net_income_yoy_pct"]),        2)

            # Quarterly trends (TTM and latest quarter)
            if qt:
                q0 = qt[0]
                if q0.get("revenue")    is not None: anchor[f"{prefix}_revenue_latest_q_b"]    = round(float(q0["revenue"])    / 1e9, 2)
                if q0.get("net_income") is not None: anchor[f"{prefix}_net_income_latest_q_b"] = round(float(q0["net_income"]) / 1e9, 2)
                anchor[f"{prefix}_latest_q_period"] = q0.get("period", "")
                if len(qt) >= 4:
                    ttm_r = sum((q.get("revenue")    or 0) for q in qt[:4])
                    ttm_n = sum((q.get("net_income") or 0) for q in qt[:4])
                    if ttm_r: anchor[f"{prefix}_revenue_ttm_b"]    = round(ttm_r / 1e9, 2)
                    if ttm_n: anchor[f"{prefix}_net_income_ttm_b"] = round(ttm_n / 1e9, 2)

        if fm_o:
            dcf   = (fm_o.get("valuation") or {}).get("dcf") or {}
            tsm   = fm_o.get("three_statement_model") or {}
            cf_s  = tsm.get("cash_flows")        or []
            bs_s  = tsm.get("balance_sheets")    or []
            inc_s = tsm.get("income_statements") or []
            divs  = fm_o.get("dividends")        or {}
            earn  = fm_o.get("earnings")         or {}

            # DCF
            if dcf.get("wacc_used")              is not None: anchor[f"{prefix}_wacc_pct"]           = round(float(dcf["wacc_used"]) * 100, 2)
            if dcf.get("intrinsic_value_base")   is not None: anchor[f"{prefix}_dcf_base"]           = round(float(dcf["intrinsic_value_base"]),   2)
            if dcf.get("intrinsic_value_bull")   is not None: anchor[f"{prefix}_dcf_bull"]           = round(float(dcf["intrinsic_value_bull"]),   2)
            if dcf.get("intrinsic_value_bear")   is not None: anchor[f"{prefix}_dcf_bear"]           = round(float(dcf["intrinsic_value_bear"]),   2)
            if dcf.get("intrinsic_value_weighted")is not None: anchor[f"{prefix}_dcf_weighted"]       = round(float(dcf["intrinsic_value_weighted"]), 2)
            if dcf.get("terminal_growth_rate")   is not None: anchor[f"{prefix}_terminal_growth_pct"]= round(float(dcf["terminal_growth_rate"]) * 100, 1)
            if dcf.get("reverse_dcf_implied_cagr")is not None: anchor[f"{prefix}_implied_cagr_pct"]  = round(float(dcf["reverse_dcf_implied_cagr"]) * 100, 1)

            # Cash flows (most recent annual)
            if cf_s:
                cf = cf_s[0]
                anchor[f"{prefix}_cf_period"] = cf.get("period", "")
                if cf.get("operating_cash_flow") is not None: anchor[f"{prefix}_ocf_b"]  = round(float(cf["operating_cash_flow"]) / 1e9, 2)
                if cf.get("free_cash_flow")      is not None: anchor[f"{prefix}_fcf_b"]  = round(float(cf["free_cash_flow"])      / 1e9, 2)
                if cf.get("capital_expenditures")is not None: anchor[f"{prefix}_capex_b"]= round(float(cf["capital_expenditures"]) / 1e9, 2)
                if cf.get("dividends_paid")      is not None: anchor[f"{prefix}_dividends_paid_b"] = round(float(cf["dividends_paid"]) / 1e9, 2)
                if cf.get("share_buybacks")      is not None: anchor[f"{prefix}_buybacks_b"]       = round(abs(float(cf["share_buybacks"])) / 1e9, 2)
                if cf.get("fcf_margin")          is not None: anchor[f"{prefix}_fcf_margin_pct"]   = round(float(cf["fcf_margin"]) * 100, 2)
                if cf.get("cfo_ni_ratio")        is not None: anchor[f"{prefix}_cfo_ni_ratio"]     = round(float(cf["cfo_ni_ratio"]), 4)

            # Balance sheet (most recent annual)
            if bs_s:
                bs = bs_s[0]
                anchor[f"{prefix}_bs_period"] = bs.get("period", "")
                if bs.get("total_assets")        is not None: anchor[f"{prefix}_total_assets_b"]    = round(float(bs["total_assets"])        / 1e9, 2)
                if bs.get("total_liabilities")   is not None: anchor[f"{prefix}_total_liabilities_b"]= round(float(bs["total_liabilities"])   / 1e9, 2)
                if bs.get("total_equity")        is not None: anchor[f"{prefix}_total_equity_b"]    = round(float(bs["total_equity"])        / 1e9, 2)
                if bs.get("cash_and_equivalents")is not None: anchor[f"{prefix}_cash_b"]            = round(float(bs["cash_and_equivalents"])/ 1e9, 2)
                if bs.get("long_term_debt")      is not None: anchor[f"{prefix}_ltd_b"]             = round(float(bs["long_term_debt"])      / 1e9, 2)
                if bs.get("short_term_debt")     is not None: anchor[f"{prefix}_std_b"]             = round(float(bs["short_term_debt"])     / 1e9, 2)
                if bs.get("net_debt")            is not None: anchor[f"{prefix}_net_debt_b"]        = round(float(bs["net_debt"])            / 1e9, 2)
                if bs.get("net_working_capital") is not None: anchor[f"{prefix}_nwc_b"]             = round(float(bs["net_working_capital"]) / 1e9, 2)
                if bs.get("dso")                 is not None: anchor[f"{prefix}_dso_days"]          = round(float(bs["dso"]),  1)
                if bs.get("dpo")                 is not None: anchor[f"{prefix}_dpo_days"]          = round(float(bs["dpo"]),  1)
                if bs.get("cash_conversion_cycle")is not None: anchor[f"{prefix}_ccc_days"]         = round(float(bs["cash_conversion_cycle"]), 1)
                # Derived D/E from balance sheet
                tot_l = bs.get("total_liabilities")
                tot_e = bs.get("total_equity")
                if tot_l is not None and tot_e is not None and float(tot_e) != 0:
                    anchor[f"{prefix}_debt_to_equity_bs"] = round(float(tot_l) / float(tot_e), 4)
                ltd = bs.get("long_term_debt")
                std = bs.get("short_term_debt")
                if ltd is not None and std is not None and tot_e is not None and float(tot_e) != 0:
                    anchor[f"{prefix}_financial_debt_to_equity"] = round((float(ltd) + float(std)) / float(tot_e), 4)

            # Income statement (most recent annual)
            if inc_s:
                inc = inc_s[0]
                anchor[f"{prefix}_inc_period"] = inc.get("period", "")
                if inc.get("revenue")          is not None: anchor[f"{prefix}_annual_revenue_b"]   = round(float(inc["revenue"])          / 1e9, 2)
                if inc.get("net_income")       is not None: anchor[f"{prefix}_annual_net_income_b"]= round(float(inc["net_income"])       / 1e9, 2)
                if inc.get("operating_income") is not None: anchor[f"{prefix}_annual_op_income_b"] = round(float(inc["operating_income"]) / 1e9, 2)
                if inc.get("ebitda")           is not None: anchor[f"{prefix}_annual_ebitda_b"]    = round(float(inc["ebitda"])           / 1e9, 2)
                if inc.get("gross_margin")     is not None: anchor[f"{prefix}_annual_gross_margin_pct"] = round(float(inc["gross_margin"]) * 100, 2)
                if inc.get("net_margin")       is not None: anchor[f"{prefix}_annual_net_margin_pct"]   = round(float(inc["net_margin"])   * 100, 2)

            # Dividends
            if divs.get("dividend_yield")  is not None: anchor[f"{prefix}_dividend_yield_pct"] = round(float(divs["dividend_yield"]) * 100, 2)
            if divs.get("annual_dividend") is not None: anchor[f"{prefix}_annual_dividend"]    = round(float(divs["annual_dividend"]), 2)

            # Current price
            cur_price = fm_o.get("current_price")
            if cur_price is not None: anchor[f"{prefix}_current_price"] = round(float(cur_price), 2)

    return anchor


_JSON_STAGE1_SYSTEM = """You are a financial data extraction assistant.
Your ONLY job is to output a JSON object using EXCLUSIVELY the numbers provided in the ANCHOR DATA below.
DO NOT invent, estimate, interpolate, infer, or recall any numbers from your training data.
If a field's value is not in the ANCHOR DATA, set it to null.
For prose fields, do not introduce any new numeric literals beyond values already present in JSON numeric fields.
Output ONLY valid JSON — no prose outside schema fields, no markdown, no explanation."""


def _build_json_schema(anchor: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    """Build a JSON schema that exactly matches the allowed set of numeric fields."""
    t = ticker.upper()

    def _num(key: str) -> Dict[str, Any]:
        """Schema entry: nullable number, present only if DB has a value."""
        val = anchor.get(key)
        if val is None:
            return {"type": ["number", "null"]}
        return {"type": ["number", "null"], "description": f"EXACTLY {val}"}

    def _str_field() -> Dict[str, Any]:
        return {"type": "string"}

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "executive_summary", "company_overview", "financial_performance",
            "valuation", "sentiment_and_market_positioning", "growth_prospects",
            "risk_factors", "competitive_landscape", "management_and_governance",
            "macroeconomic_factors", "analyst_verdict",
        ],
        "properties": {
            # ── Qualitative text sections ─────────────────────────────────
            "executive_summary":               _str_field(),
            "company_overview":                _str_field(),
            "sentiment_and_market_positioning": _str_field(),
            "growth_prospects":                _str_field(),
            "risk_factors":                    _str_field(),
            "competitive_landscape":           _str_field(),
            "management_and_governance":       _str_field(),
            "macroeconomic_factors":           _str_field(),
            "analyst_verdict":                 _str_field(),

            # ── Financial Performance section ─────────────────────────────
            "financial_performance": {
                "type": "object",
                "additionalProperties": False,
                "required": ["prose"],
                "properties": {
                    "prose":                   _str_field(),
                    "revenue_ttm_b":           _num(f"{t}_revenue_ttm_b"),
                    "revenue_latest_q_b":      _num(f"{t}_revenue_latest_q_b"),
                    "revenue_yoy_pct":         _num(f"{t}_revenue_yoy_pct"),
                    "net_income_ttm_b":        _num(f"{t}_net_income_ttm_b"),
                    "net_income_latest_q_b":   _num(f"{t}_net_income_latest_q_b"),
                    "net_income_yoy_pct":      _num(f"{t}_net_income_yoy_pct"),
                    "gross_profit_yoy_pct":    _num(f"{t}_gross_profit_yoy_pct"),
                    "operating_income_yoy_pct":_num(f"{t}_operating_income_yoy_pct"),
                    "annual_revenue_b":        _num(f"{t}_annual_revenue_b"),
                    "annual_net_income_b":     _num(f"{t}_annual_net_income_b"),
                    "annual_op_income_b":      _num(f"{t}_annual_op_income_b"),
                    "annual_ebitda_b":         _num(f"{t}_annual_ebitda_b"),
                    "annual_gross_margin_pct": _num(f"{t}_annual_gross_margin_pct"),
                    "annual_net_margin_pct":   _num(f"{t}_annual_net_margin_pct"),
                    "gross_margin_pct":        _num(f"{t}_gross_margin_pct"),
                    "ebit_margin_pct":         _num(f"{t}_ebit_margin_pct"),
                    "ocf_b":                   _num(f"{t}_ocf_b"),
                    "fcf_b":                   _num(f"{t}_fcf_b"),
                    "capex_b":                 _num(f"{t}_capex_b"),
                    "fcf_margin_pct":          _num(f"{t}_fcf_margin_pct"),
                    "cfo_ni_ratio":            _num(f"{t}_cfo_ni_ratio"),
                    "dividends_paid_b":        _num(f"{t}_dividends_paid_b"),
                    "buybacks_b":              _num(f"{t}_buybacks_b"),
                    "dividend_yield_pct":      _num(f"{t}_dividend_yield_pct"),
                    "annual_dividend":         _num(f"{t}_annual_dividend"),
                },
            },

            # ── Key Financial Ratios & Valuation section ──────────────────
            "key_financial_ratios_and_valuation": {
                "type": "object",
                "additionalProperties": False,
                "required": ["prose"],
                "properties": {
                    "prose":                  _str_field(),
                    "pe_trailing":            _num(f"{t}_pe_trailing"),
                    "ev_ebitda":              _num(f"{t}_ev_ebitda"),
                    "p_fcf":                  _num(f"{t}_p_fcf"),
                    "ev_revenue":             _num(f"{t}_ev_revenue"),
                    "roe_pct":                _num(f"{t}_roe_pct"),
                    "roic_pct":               _num(f"{t}_roic_pct"),
                    "piotroski":              _num(f"{t}_piotroski"),
                    "beneish_m":              _num(f"{t}_beneish_m"),
                    "altman_z":               _num(f"{t}_altman_z"),
                    "current_ratio":          _num(f"{t}_current_ratio"),
                    "debt_to_equity":         _num(f"{t}_debt_to_equity"),
                    "debt_to_equity_bs":      _num(f"{t}_debt_to_equity_bs"),
                    "financial_debt_to_equity":_num(f"{t}_financial_debt_to_equity"),
                    "total_assets_b":         _num(f"{t}_total_assets_b"),
                    "total_equity_b":         _num(f"{t}_total_equity_b"),
                    "total_liabilities_b":    _num(f"{t}_total_liabilities_b"),
                    "cash_b":                 _num(f"{t}_cash_b"),
                    "ltd_b":                  _num(f"{t}_ltd_b"),
                    "std_b":                  _num(f"{t}_std_b"),
                    "net_debt_b":             _num(f"{t}_net_debt_b"),
                    "nwc_b":                  _num(f"{t}_nwc_b"),
                    "dso_days":               _num(f"{t}_dso_days"),
                    "dpo_days":               _num(f"{t}_dpo_days"),
                    "ccc_days":               _num(f"{t}_ccc_days"),
                    "dcf_base":               _num(f"{t}_dcf_base"),
                    "dcf_bull":               _num(f"{t}_dcf_bull"),
                    "dcf_bear":               _num(f"{t}_dcf_bear"),
                    "dcf_weighted":           _num(f"{t}_dcf_weighted"),
                    "wacc_pct":               _num(f"{t}_wacc_pct"),
                    "terminal_growth_pct":    _num(f"{t}_terminal_growth_pct"),
                    "implied_cagr_pct":       _num(f"{t}_implied_cagr_pct"),
                    "current_price":          _num(f"{t}_current_price"),
                    "beta_60d":               _num(f"{t}_beta_60d"),
                    "sharpe_ratio_12m":       _num(f"{t}_sharpe_ratio_12m"),
                    "return_12m_pct":         _num(f"{t}_return_12m_pct"),
                    "sma_50":                 _num(f"{t}_sma_50"),
                    "sma_200":                _num(f"{t}_sma_200"),
                },
            },
        },
    }
    return schema


def _validate_anchor_values(data: Dict[str, Any], anchor: Dict[str, Any], ticker: str) -> List[str]:
    """Check that every numeric value in data matches the anchor exactly (within 0.01%).

    Returns a list of violation strings for logging.
    """
    from .validation import flatten_json  # type: ignore[import]
    t = ticker.upper()
    violations = []
    flat = flatten_json(data)
    for key, val in flat.items():
        if not isinstance(val, (int, float)):
            continue
        # Map the nested key back to an anchor key
        # e.g. "financial_performance_revenue_ttm_b" → "AAPL_revenue_ttm_b"
        # Try to find any anchor key whose suffix matches the data key
        for anchor_key, anchor_val in anchor.items():
            if not anchor_key.startswith(t + "_"):
                continue
            suffix = anchor_key[len(t) + 1:]  # strip "AAPL_"
            if key.endswith(suffix) and isinstance(anchor_val, (int, float)):
                deviation = abs(float(val) - float(anchor_val)) / (abs(float(anchor_val)) + 1e-9)
                if deviation > 0.0001:  # 0.01% tolerance
                    violations.append(
                        f"{key}: JSON value {val} differs from anchor {anchor_val} ({deviation*100:.3f}%)"
                    )
    return violations


_JSON_STAGE2_SYSTEM = """You are a senior equity research analyst writing investment research notes.
You have access to outputs from 5 specialized agents that you must synthesize into a coherent narrative:
1. BUSINESS ANALYST (ba_outputs): Company overview, competitive positioning, management assessment, industry trends
2. QUANT FUNDAMENTAL (quant_outputs): Valuation metrics, financial ratios, Piotroski/Altman/Beneish scores, price performance
3. WEB SEARCH (web_outputs): Recent news, market sentiment, analyst upgrades/downgrades, macro factors
4. FINANCIAL MODELLING (fm_outputs): DCF valuation, intrinsic value scenarios, margin projections
5. STOCK RESEARCH (sr_outputs): Earnings transcript analysis, broker sentiment, Q&A dynamics, growth catalysts

CRITICAL RULES:
1. Use the EXACT numbers from the JSON — do NOT change, round, or substitute any figure.
2. Every sentence containing a number must end with a citation marker like [1], [2] etc.
3. Use only numbers that appear in the JSON data object. Do NOT add any extra numbers from memory.
3b. If a value is missing/null, explicitly state data is unavailable instead of inventing a proxy number.
4. Write in flowing multi-sentence paragraphs — NO bullet points, NO lists.
5. Use professional financial terminology throughout.
6. BALANCE SHEET: total_assets, total_liabilities, and total_equity are THREE DISTINCT values. Never confuse them.

ANALYSIS DEPTH REQUIREMENTS:
- Go beyond stating facts: explain the BUSINESS IMPLICATIONS of every metric and trend
- For each ratio/metric: What does it mean for the company's financial health? Why should an investor care?
- Connect quantitative metrics to qualitative business outcomes (e.g., "High ROIC of 58% suggests strong competitive moat and efficient capital allocation")
- Discuss causation, not just correlation: Why did margins improve? What drove the change in cash flow?
- Address forward-looking implications: What do current trends suggest for future performance?
- Consider stakeholder perspectives: How do these metrics affect shareholders, creditors, and management decisions?

SYNTHESIS GUIDANCE:
- Weave insights from ALL 5 agents together — don't treat them as siloed sections
- The Web Search provides context; Quant Fundamental provides numbers; Business Analyst provides narrative; Stock Research provides tone; Financial Modelling provides valuation
- When discussing valuation, incorporate both DCF outputs and peer comparisons
- Use broker sentiment from Stock Research to contextualize quantitative findings
- Connect recent news (Web Search) to financial performance (Quant/BA)"""


def summarise_results_structured(
    user_query: str,
    tickers: List[str],
    ba_outputs: List[Dict[str, Any]],
    quant_outputs: List[Dict[str, Any]],
    web_outputs: List[Dict[str, Any]],
    fm_outputs: List[Dict[str, Any]] = [],
    sr_outputs: List[Dict[str, Any]] = [],
    ticker: Optional[str] = None,
    ba_output: Optional[Dict[str, Any]] = None,
    quant_output: Optional[Dict[str, Any]] = None,
    web_output: Optional[Dict[str, Any]] = None,
    data_availability: Optional[Dict[str, Any]] = None,
    output_language: Optional[str] = None,
    _trace_out: Optional[List[str]] = None,
) -> str:
    """Two-pass structured summarisation: JSON schema validation then prose conversion.

    Pass 1: Force the LLM to emit a strictly validated JSON object whose numeric
            fields are drawn exclusively from the DB anchor.  Any value not in the
            anchor must be null.  The JSON schema uses additionalProperties=False so
            the LLM cannot sneak in extra numeric fields.

    Pass 2: A second LLM call converts the validated JSON to flowing prose.  This
            call receives no free-floating numbers in its prompt — only the JSON
            that has already been validated — so it cannot hallucinate new figures.

    Pass 3: Final audit.  Any number that appears in the final prose but was not
            present in the validated JSON is flagged and replaced with a safe
            anchor value (if one can be found) or redacted.

    Falls back to the standard ``summarise_results`` if JSON generation/validation
    fails after retries, preserving existing quality.
    """
    # ── Normalise legacy single-value params to lists ─────────────────────────
    effective_tickers = tickers or ([ticker] if ticker else [])
    effective_quant   = quant_outputs  or ([quant_output]  if quant_output  else [])
    effective_fm      = fm_outputs     or []
    effective_ba      = ba_outputs     or ([ba_output]     if ba_output     else [])
    effective_web     = web_outputs    or []
    effective_sr      = sr_outputs     or []

    if not effective_tickers:
        logger.warning("[structured] No tickers — falling back to summarise_results")
        return summarise_results(
            user_query=user_query, tickers=tickers,
            ba_outputs=ba_outputs, quant_outputs=quant_outputs,
            web_outputs=web_outputs, fm_outputs=fm_outputs,
            sr_outputs=sr_outputs, ticker=ticker, ba_output=ba_output,
            quant_output=quant_output, web_output=web_output,
            data_availability=data_availability, _trace_out=_trace_out,
        )

    # ── Build the authoritative anchor dict ───────────────────────────────────
    anchor = _build_anchor_dict(effective_tickers, effective_quant, effective_fm)
    logger.info("[structured] Anchor built: %d fields for tickers=%s", len(anchor), effective_tickers)

    # For now support single-ticker path (extend later for multi-ticker comparison)
    primary_ticker = effective_tickers[0]
    schema = _build_json_schema(anchor, primary_ticker)

    # ── Gather qualitative context (BA, web, SR) ──────────────────────────────
    qual_parts: List[str] = []
    for ba_o in effective_ba:
        if ba_o:
            qual_parts.extend(_build_ba_context(ba_o))
    for web_o in effective_web:
        if web_o:
            qual_parts.extend(_build_web_context(web_o))
    for sr_o in effective_sr:
        if sr_o:
            qual_parts.extend(_build_stock_research_context(sr_o))
    qualitative_context = "\n".join(qual_parts)[:6000]  # cap at ~6k chars

    # ── Stage 1: JSON generation ──────────────────────────────────────────────
    anchor_json_str = json.dumps(
        {k: v for k, v in anchor.items() if not k.endswith("_period")},
        indent=2
    )
    stage1_prompt = f"""USER QUERY: {user_query}

ANCHOR DATA (use ONLY these numbers — every number you write must appear verbatim here):
{anchor_json_str}

QUALITATIVE CONTEXT FROM RESEARCH AGENTS (use for text/analysis, cite with [N]):
{qualitative_context}

OUTPUT INSTRUCTIONS:
Produce a JSON object that EXACTLY matches this schema:
{json.dumps(schema, indent=2)}

CRITICAL RULES:
1. Every numeric field in financial_performance and key_financial_ratios_and_valuation must be
   copied VERBATIM from the ANCHOR DATA above. Do NOT modify, round, or estimate any number.
2. If a field's value does not appear in the ANCHOR DATA, set it to null.
3. All prose fields must remain consistent with numeric fields in the same JSON object.
   Do not invent extra numeric literals in prose. If required data is unavailable, write "data unavailable".
4. NO extra fields — the schema uses additionalProperties: false.
5. Return ONLY the JSON object. No markdown fences, no explanation, no preamble."""

    stage1_result: Optional[Dict[str, Any]] = None
    for attempt in range(3):
        try:
            raw = _deepseek_generate(
                _SUMMARIZER_MODEL, stage1_prompt,
                max_tokens=8000, temperature=0.1,
                timeout=_SUMMARIZER_TIMEOUT,
                system_prompt=_JSON_STAGE1_SYSTEM,
            )
            raw = _strip_think(raw).strip()
            raw = _strip_fences(raw)
            data = json.loads(raw)
            # Validate required top-level keys exist
            required_keys = {"executive_summary", "company_overview", "financial_performance",
                             "analyst_verdict"}
            if not required_keys.issubset(data.keys()):
                missing = required_keys - data.keys()
                logger.warning("[structured] Stage1 attempt %d missing keys: %s", attempt+1, missing)
                continue
            # Validate numeric fields match anchor
            violations = _validate_anchor_values(data, anchor, primary_ticker)
            if violations:
                logger.warning("[structured] Stage1 attempt %d anchor violations: %s", attempt+1, violations[:5])
                # Fix violations in-place rather than retrying
                _fix_anchor_violations(data, anchor, primary_ticker)
            stage1_result = data
            logger.info("[structured] Stage1 JSON validated OK on attempt %d", attempt+1)
            break
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("[structured] Stage1 attempt %d JSON parse error: %s", attempt+1, e)

    if stage1_result is None:
        logger.error("[structured] Stage1 failed after 3 attempts — falling back to summarise_results")
        return summarise_results(
            user_query=user_query, tickers=tickers,
            ba_outputs=ba_outputs, quant_outputs=quant_outputs,
            web_outputs=web_outputs, fm_outputs=fm_outputs,
            sr_outputs=sr_outputs, ticker=ticker, ba_output=ba_output,
            quant_output=quant_output, web_output=web_output,
            data_availability=data_availability, _trace_out=_trace_out,
        )

    # ── Stage 2: JSON → prose ─────────────────────────────────────────────────
    # Build citation index from web outputs for [N] references
    citation_index = _build_citation_index(effective_web, effective_sr, effective_ba)
    citation_str = "\n".join(f"[{i+1}] {src}" for i, src in enumerate(citation_index[:20]))

    t = primary_ticker.upper()
    period_bs  = anchor.get(f"{t}_bs_period",  "FY2024")
    period_cf  = anchor.get(f"{t}_cf_period",  "FY2024")
    period_inc = anchor.get(f"{t}_inc_period", "FY2024")
    _citation_block = citation_str if citation_str else "[1] Company filings and DB\n[2] Market data"
    _latest_q_period = anchor.get(f"{t}_latest_q_period", "Q1 FY2025")

    stage2_prompt = f"""Convert the following validated JSON into a professional equity research note.

VALIDATED JSON DATA (these numbers are ground-truth — use them exactly):
{json.dumps(stage1_result, indent=2)}

CITATION INDEX (use [N] inline after every factual sentence):
{_citation_block}

PERIOD CONTEXT:
- Balance sheet period: {period_bs}
- Cash flow period: {period_cf}
- Income statement period: {period_inc}
- Latest quarter: {_latest_q_period}

AGENT SOURCES TO SYNTHESIZE:
- BUSINESS ANALYST outputs: company overview, competitive positioning, management quality, industry dynamics
- QUANT FUNDAMENTAL outputs: valuation multiples, financial ratios (ROE, ROIC, Piotroski, Altman Z), price performance
- WEB SEARCH outputs: recent news, analyst actions, macro headwinds/tailwinds, market sentiment
- FINANCIAL MODELLING outputs: DCF intrinsic value, bull/bear/base scenarios, margin forecasts
- STOCK RESEARCH outputs: earnings call tone, broker consensus, Q&A dynamics, forward-looking catalysts

Write EXACTLY these 11 sections in this order, using ## headers:
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

ANALYSIS DEPTH REQUIREMENTS:
- For EVERY metric mentioned, explain the BUSINESS IMPLICATIONS — not just the number
- Answer: What does this mean for the company's competitive position? Why should an investor care?
- Example: Don't just say "ROIC is 58%" — explain "58% ROIC indicates exceptional capital efficiency, suggesting a durable competitive moat and strong pricing power"
- Connect data points: How do earnings trends relate to cash flow? What drove margin expansion?
- Forward-looking: What do current trends imply for next quarter/year?
- Weave ALL 5 agent outputs together — don't siloe them

RULES:
- Every number you write MUST appear in the JSON data above. NO other numbers.
- Do NOT introduce numbers not present in the JSON (no RSI, no current ratio, no market capitalisation unless it's in the JSON).
- If a number is not present in JSON, write a qualitative sentence without a numeric literal.
- Every sentence with a number must end with [N] citation.
- Flowing prose paragraphs only — NO bullet points, NO lists.
- 6-10 sentences per section minimum to ensure depth.
- BANNED WORDS: never use "robust", "strong", "significant", "impressive" — use neutral professional language instead.
- BALANCE SHEET CRITICAL: total_assets ≠ total_liabilities ≠ total_equity. These are THREE DISTINCT values. Never use the total_assets figure when referring to total liabilities, and vice versa.
- Start directly with ## Executive Summary."""

    try:
        prose_raw = _deepseek_generate(
            _SUMMARIZER_MODEL, stage2_prompt,
            max_tokens=8000, temperature=0.1,
            timeout=_SUMMARIZER_TIMEOUT,
            system_prompt=_JSON_STAGE2_SYSTEM,
        )
        prose = _strip_think(prose_raw).strip()
        if not prose:
            raise ValueError("Stage2 returned empty response")
    except Exception as exc:
        logger.error("[structured] Stage2 failed: %s — falling back", exc)
        return summarise_results(
            user_query=user_query, tickers=tickers,
            ba_outputs=ba_outputs, quant_outputs=quant_outputs,
            web_outputs=web_outputs, fm_outputs=fm_outputs,
            sr_outputs=sr_outputs, ticker=ticker, ba_output=ba_output,
            quant_output=quant_output, web_output=web_output,
            data_availability=data_availability, _trace_out=_trace_out,
        )

    # ── Stage 2.5: Fix malformed number artifacts and banned words ────────────
    # Fix double-decimal artefacts like "435.62.62B", "18.72.8%", "18.72.72%"
    # The LLM sometimes duplicates or concatenates decimal parts.
    # Pattern: digits.digits.digits — keep only the first X.YY portion.
    _bad_before = len(re.findall(r'\d+\.\d+\.\d+', prose))
    prose = re.sub(
        r'(\d+\.\d+)\.\d+',
        r'\1',
        prose,
    )
    _bad_after = len(re.findall(r'\d+\.\d+\.\d+', prose))
    logger.warning("[structured] Stage2.5: fixed %d double-decimal artefacts (%d remain)",
                   _bad_before - _bad_after, _bad_after)
    # Replace banned words with neutral professional alternatives
    # Note: "notable" is also banned — use "measured" or "consistent" instead
    _BANNED_REPLACEMENTS = [
        (re.compile(r'\brobust\b',      re.IGNORECASE), 'measured'),
        (re.compile(r'\bstrong\b',      re.IGNORECASE), 'consistent'),
        (re.compile(r'\bnotable\b',     re.IGNORECASE), 'material'),
        (re.compile(r'\bimpressive\b',  re.IGNORECASE), 'elevated'),
        (re.compile(r'\bsignificant\b', re.IGNORECASE), 'material'),
        (re.compile(r'\bstellar\b',     re.IGNORECASE), 'elevated'),
        (re.compile(r'\bexceptional\b', re.IGNORECASE), 'elevated'),
        (re.compile(r'\bsolid\b',       re.IGNORECASE), 'consistent'),
    ]
    for _pat, _repl in _BANNED_REPLACEMENTS:
        prose = _pat.sub(_repl, prose)

    # ── Stage 3: Final number audit ───────────────────────────────────────────
    # Build the full set of numeric strings that are allowed to appear in prose
    from .validation import flatten_json  # type: ignore[import]
    flat_json = flatten_json(stage1_result)
    # All numeric values in the validated JSON (as strings, various decimal forms)
    allowed_nums: set = set()
    for val in flat_json.values():
        if isinstance(val, (int, float)):
            # Allow multiple rounding representations
            fval = float(val)
            allowed_nums.add(str(int(fval)) if fval == int(fval) else None)
            for dp in range(5):
                allowed_nums.add(f"{fval:.{dp}f}")
            allowed_nums.discard(None)
    # Also allow all anchor values
    for val in anchor.values():
        if isinstance(val, (int, float)):
            fval = float(val)
            for dp in range(5):
                allowed_nums.add(f"{fval:.{dp}f}")

    # Find numbers in prose that aren't in allowed set
    prose = _audit_and_replace_numbers(prose, allowed_nums, anchor, primary_ticker)

    # ── Apply FactChecker as final safety net ─────────────────────────────────
    try:
        from .validation import validate_quant_output, FactChecker  # type: ignore[import]
        _fact_checker = FactChecker()
        for i, t_tick in enumerate(effective_tickers):
            quant_o = effective_quant[i] if i < len(effective_quant) else None
            fm_o    = effective_fm[i]    if i < len(effective_fm)    else None
            if quant_o:
                metrics = validate_quant_output(quant_o, fm_output=fm_o)
                if metrics:
                    prose, corrections = _fact_checker.correct_report(prose, metrics)
                    if corrections:
                        logger.info("[structured] FactChecker corrections: %s", corrections)
    except Exception as _fce:
        logger.warning("[structured] FactChecker failed (non-fatal): %s", _fce)

    # ── Stage 3.5: Final double-decimal sweep (after audit+FactChecker) ─────────
    # Stage 3 and FactChecker can each independently reintroduce double-decimals
    # (e.g. matching '435' inside '$435.62B' and replacing with '435.62').
    # Run one last cleanup pass to collapse any X.YY.ZZ patterns.
    _dd_before = len(re.findall(r'\d+\.\d+\.\d+', prose))
    prose = re.sub(r'(\d+\.\d+)\.\d+', r'\1', prose)
    _dd_after = len(re.findall(r'\d+\.\d+\.\d+', prose))
    if _dd_before:
        logger.warning("[structured] Stage3.5: fixed %d double-decimal artefacts after audit (%d remain)",
                       _dd_before - _dd_after, _dd_after)

    # ── Inject full combined citation block (BA + SR + Web + Quant + FM) ─────
    # Keep numbering consistent with build_citation_block ordering.
    from .citations import build_citation_block, inject_inline_numbers  # type: ignore[import]

    all_chunk_id_maps: Dict[str, Any] = {}
    ref_blocks: List[str] = []
    offset = 0
    for i, t in enumerate(effective_tickers or [primary_ticker]):
        ba_o = effective_ba[i] if i < len(effective_ba) else None
        quant_o = effective_quant[i] if i < len(effective_quant) else None
        web_o = effective_web[i] if i < len(effective_web) else None
        fm_o = effective_fm[i] if i < len(effective_fm) else None
        sr_o = effective_sr[i] if i < len(effective_sr) else None

        ref_block, chunk_id_map = build_citation_block(
            ba_output=ba_o,
            quant_output=quant_o,
            web_output=web_o,
            fm_output=fm_o,
            sr_output=sr_o,
            ticker=t,
            index_offset=offset,
        )
        if ref_block:
            ref_blocks.append(ref_block)
        all_chunk_id_maps.update(chunk_id_map)
        offset += len(chunk_id_map)

    prose = inject_inline_numbers(prose, all_chunk_id_maps)
    combined_ref_block = "\n".join(ref_blocks) if ref_blocks else ""
    if combined_ref_block:
        prose = prose.rstrip() + "\n" + combined_ref_block
    else:
        prose = prose.rstrip() + "\n\n---\n### References\n*No external citations available — analysis based on internal data sources.*"

    # ── Inject subtitle ───────────────────────────────────────────────────────
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    subtitle = f"*{primary_ticker} | Equity Research Note | {today}*\n\n"
    prose = re.sub(
        r"(^## Executive Summary\s*\n)",
        r"\1" + subtitle,
        prose,
        count=1,
        flags=re.MULTILINE,
    )

    if _trace_out is not None:
        _trace_out.append("[structured-summariser] Stage1+Stage2 completed")

    # ── Stage 4: Translate to requested language (if any) ─────────────────────
    # Translation is done here (Stage 4) after all number auditing is complete,
    # so the English prose with validated numbers is what gets translated — not raw
    # LLM output.  This replaces the old node_translator DeepSeek call.
    if output_language:
        try:
            prose = translate_text(prose, output_language)
            logger.info("[structured] Stage4: translated to %s (%d chars)", output_language, len(prose))
        except Exception as _te:
            logger.warning("[structured] Stage4 translation failed (returning English): %s", _te)

    logger.info("[structured] Completed. Prose length: %d chars", len(prose))
    return prose


def _fix_anchor_violations(data: Dict[str, Any], anchor: Dict[str, Any], ticker: str) -> None:
    """In-place fix: replace any numeric values in data that deviate from anchor."""
    from .validation import flatten_json  # type: ignore[import]
    t = ticker.upper()

    def _fix_dict(d: Any) -> None:
        if isinstance(d, dict):
            for key in list(d.keys()):
                val = d[key]
                if isinstance(val, (int, float)):
                    # Find matching anchor key
                    for anchor_key, anchor_val in anchor.items():
                        if anchor_key.startswith(t + "_") and isinstance(anchor_val, (int, float)):
                            suffix = anchor_key[len(t) + 1:]
                            if key == suffix:
                                d[key] = anchor_val
                                break
                elif isinstance(val, (dict, list)):
                    _fix_dict(val)
        elif isinstance(d, list):
            for item in d:
                _fix_dict(item)

    _fix_dict(data)


def _build_citation_index(
    web_outputs: List[Dict[str, Any]],
    sr_outputs: List[Dict[str, Any]],
    ba_outputs: List[Dict[str, Any]],
) -> List[str]:
    """Extract a flat list of citation source strings for [1], [2], ... references."""
    sources: List[str] = []
    for ba_o in ba_outputs:
        if not ba_o:
            continue
        for c in (ba_o.get("citations") or []):
            if not isinstance(c, dict):
                continue
            src = c.get("doc_name") or c.get("chunk_id")
            if src and src not in sources:
                sources.append(str(src)[:120])
    for web_o in web_outputs:
        if not web_o:
            continue
        for chunk in (web_o.get("chunks") or []):
            src = chunk.get("source") or chunk.get("url") or chunk.get("title")
            if src and src not in sources:
                sources.append(str(src)[:120])
    for sr_o in sr_outputs:
        if not sr_o:
            continue
        for chunk in (sr_o.get("chunks") or []):
            src = chunk.get("source") or chunk.get("url") or chunk.get("title")
            if src and src not in sources:
                sources.append(str(src)[:120])
    return sources[:20]


def _audit_and_replace_numbers(
    prose: str,
    allowed_nums: set,
    anchor: Dict[str, Any],
    ticker: str,
) -> str:
    """Find numbers in prose that are not in allowed_nums.

    For each such number, try to find the closest anchor value to replace it.
    Numbers that are clearly non-financial (e.g. section numbers, years, citation
    indices) are left untouched.
    """
    # Patterns to skip: citation markers [1], years (19xx/20xx), small integers ≤ 12
    # (months/days), page numbers, percentages that are clearly ordinal
    _SKIP_PATTERN = re.compile(
        r'(?:'
        r'\[\d+\]'                        # citation [N]
        r'|\b(?:19|20)\d{2}\b'            # years 1900-2099
        r'|\b[1-9]\b'                     # single digits
        r'|\b1[0-2]\b'                    # 10-12 (months)
        r')'
    )

    def _try_replace(match: re.Match) -> str:
        num_str = match.group(0)
        # Skip if it's a citation, year, or small integer
        if _SKIP_PATTERN.fullmatch(num_str):
            return num_str
        # Check if it's already allowed
        if num_str in allowed_nums:
            return num_str
        # Try to find the closest anchor value
        try:
            num_val = float(num_str)
        except ValueError:
            return num_str
        # Skip years — likely structural rather than financial claims
        if 1900 <= num_val <= 2100:
            return num_str
        # Find closest anchor value
        t = ticker.upper()
        best_key, best_diff = None, float("inf")
        for anchor_key, anchor_val in anchor.items():
            if not anchor_key.startswith(t + "_") or not isinstance(anchor_val, (int, float)):
                continue
            diff = abs(float(anchor_val) - num_val) / (abs(float(anchor_val)) + 1e-9)
            if diff < best_diff:
                best_diff = diff
                best_key = anchor_key
        # Only replace if very close (within 5%) — otherwise redact
        if best_key and best_diff < 0.05:
            replacement = str(anchor[best_key])
            logger.info("[audit] Replacing stray number %s with anchor %s=%s", num_str, best_key, replacement)
            return replacement
        # No safe anchor match: redact numeric literal to prevent hallucinated figures
        logger.warning("[audit] Stray number %s not in anchor — redacting", num_str)
        return "N/A"

    # Match standalone numbers (not inside words or immediately followed by a decimal point,
    # which would indicate we're matching just the integer part of a larger decimal number
    # like matching '435' in '$435.62B' which would produce '$435.62.62B' after replacement).
    return re.sub(r'(?<![/\w])\d+(?:\.\d+)?(?![/\w%.])', _try_replace, prose)


def summarise_results(
    user_query: str,
    tickers: List[str],
    ba_outputs: List[Dict[str, Any]],
    quant_outputs: List[Dict[str, Any]],
    web_outputs: List[Dict[str, Any]],
    fm_outputs: List[Dict[str, Any]] = [],
    sr_outputs: List[Dict[str, Any]] = [],
    # Legacy single-value params kept for backward-compat (ignored if lists provided)
    ticker: Optional[str] = None,
    ba_output: Optional[Dict[str, Any]] = None,
    quant_output: Optional[Dict[str, Any]] = None,
    web_output: Optional[Dict[str, Any]] = None,
    data_availability: Optional[Dict[str, Any]] = None,
    _trace_out: Optional[List[str]] = None,
) -> str:
    """Call DeepSeek summarizer with all agent outputs and return the narrative + citations.

    Accepts multi-ticker lists (ba_outputs, quant_outputs, web_outputs, fm_outputs).
    Legacy single-value keyword arguments are accepted for backward-compatibility but
    are ignored when the list parameters are non-empty.

    If ``_trace_out`` is provided (a list), the model's reasoning trace string is
    appended to it so the caller can surface it in the UI without changing the
    return type.
    """
    from .citations import build_citation_block, inject_inline_numbers

    # Resolve effective lists — merge legacy single-value args if lists are empty
    effective_ba     = ba_outputs or ([ba_output] if ba_output else [])
    effective_quant  = quant_outputs or ([quant_output] if quant_output else [])
    effective_web    = web_outputs or ([web_output] if web_output else [])
    effective_fm     = fm_outputs or []
    effective_sr     = sr_outputs or []
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
        sr_o     = effective_sr[i]     if i < len(effective_sr)    else None

        ref_block, chunk_id_map = build_citation_block(
            ba_output=ba_o,
            quant_output=quant_o,
            web_output=web_o,
            fm_output=fm_o,
            sr_output=sr_o,
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

    # ── LOCKED DATA ANCHOR: pre-computed from agent outputs ──────────────────
    # These are the ONLY authoritative values for key metrics.
    # The LLM MUST copy them verbatim — do NOT adjust, round, or substitute.
    anchor_lines: List[str] = [
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║  LOCKED DATA ANCHOR — DATABASE-SOURCED VALUES (TTM, as of today)    ║",
        "║  COPY THESE NUMBERS VERBATIM. Do NOT use values from training memory.║",
        "║  Our proprietary DB may differ from textbook or online figures.      ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
    ]
    for i, t in enumerate(tickers or ([effective_ticker] if effective_ticker else [])):
        quant_o = effective_quant[i] if i < len(effective_quant) else None
        fm_o = effective_fm[i] if i < len(effective_fm) else None
        if quant_o:
            vf = quant_o.get("value_factors") or {}
            qfact = quant_o.get("quality_factors") or {}
            pe = vf.get("pe_trailing")
            ev_ebitda = vf.get("ev_ebitda")
            p_fcf = vf.get("p_fcf")
            roe_raw = qfact.get("roe")
            roic_raw = qfact.get("roic")
            piotroski = qfact.get("piotroski_f_score")
            beneish = qfact.get("beneish_m_score")
            altman_z = qfact.get("altman_z_score")
            roe_pct = f"{float(roe_raw)*100:.2f}%" if roe_raw is not None else "N/A"
            roic_pct = f"{float(roic_raw)*100:.2f}%" if roic_raw is not None else "N/A"
            anchor_lines.append(f"  [{t}] P/E (trailing, TTM): EXACTLY {pe}x")
            anchor_lines.append(f"  [{t}] EV/EBITDA: EXACTLY {ev_ebitda}x")
            anchor_lines.append(f"  [{t}] P/FCF: EXACTLY {p_fcf}x")
            anchor_lines.append(f"  [{t}] ROE: EXACTLY {roe_pct}")
            anchor_lines.append(f"  [{t}] ROIC: EXACTLY {roic_pct}")
            anchor_lines.append(
                f"  [{t}] Piotroski F-Score: EXACTLY {piotroski}/9"
                + (" (WARNING: this DB value may differ from published sources — use it anyway)" if piotroski is not None and piotroski <= 3 else "")
            )
            anchor_lines.append(
                f"  [{t}] Beneish M-Score: EXACTLY {beneish}"
                + (" (WARNING: this DB value may differ from other sources — use it anyway)" if beneish is not None and abs(float(beneish)) < 3.5 else "")
            )
            anchor_lines.append(
                f"  [{t}] Altman Z-Score: EXACTLY {altman_z}"
                + (" (WARNING: this DB value may differ from other sources — use it anyway)" if altman_z is not None and float(altman_z) < 5 else "")
            )
            # Revenue / income figures from quarterly trends
            qt = quant_o.get("quarterly_trends") or []
            if qt:
                q_latest = qt[0]
                rev_q = q_latest.get("revenue")
                ni_q  = q_latest.get("net_income")
                period_q = q_latest.get("period", "latest Q")
                if rev_q is not None:
                    anchor_lines.append(f"  [{t}] Latest quarterly revenue ({period_q}): EXACTLY ${rev_q/1e9:.2f}B")
                if ni_q is not None:
                    anchor_lines.append(f"  [{t}] Latest quarterly net income ({period_q}): EXACTLY ${ni_q/1e9:.2f}B")
                if len(qt) >= 4:
                    ttm_rev = sum((q.get("revenue") or 0) for q in qt[:4])
                    ttm_ni  = sum((q.get("net_income") or 0) for q in qt[:4])
                    anchor_lines.append(f"  [{t}] TTM revenue (last 4 quarters): EXACTLY ${ttm_rev/1e9:.2f}B")
                    anchor_lines.append(f"  [{t}] TTM net income (last 4 quarters): EXACTLY ${ttm_ni/1e9:.2f}B")
            # Beta and Sharpe
            mr = quant_o.get("momentum_risk") or quant_o.get("momentum_factors") or {}
            beta_v = mr.get("beta_60d")
            sharpe_v = mr.get("sharpe_ratio_12m")
            if beta_v is not None:
                anchor_lines.append(
                    f"  [{t}] Beta (60-day): EXACTLY {beta_v:.4f}"
                    " (WARNING: may differ from published figures — use DB value)"
                )
            if sharpe_v is not None:
                anchor_lines.append(
                    f"  [{t}] Sharpe ratio (12m): EXACTLY {sharpe_v:.4f}"
                    " (WARNING: may differ from published figures — use DB value)"
                )
        if fm_o:
            dcf = (fm_o.get("valuation") or {}).get("dcf") or {}
            wacc = dcf.get("wacc_used")
            dcf_base = dcf.get("intrinsic_value_base")
            dcf_bull = dcf.get("intrinsic_value_bull")
            dcf_bear = dcf.get("intrinsic_value_bear")
            if wacc is not None:
                anchor_lines.append(f"  [{t}] WACC: EXACTLY {wacc*100:.2f}%")
            if dcf_base is not None:
                anchor_lines.append(f"  [{t}] DCF intrinsic value (base): EXACTLY ${dcf_base:.2f}")
            if dcf_bull is not None:
                anchor_lines.append(f"  [{t}] DCF intrinsic value (bull): EXACTLY ${dcf_bull:.2f}")
            if dcf_bear is not None:
                anchor_lines.append(f"  [{t}] DCF intrinsic value (bear): EXACTLY ${dcf_bear:.2f}")
            # Cash flow and balance sheet ground truth (most recent annual period)
            tsm = fm_o.get("three_statement_model") or {}
            cf_stmts = tsm.get("cash_flows") or []
            bs_stmts = tsm.get("balance_sheets") or []
            inc_stmts = tsm.get("income_statements") or []
            if cf_stmts:
                cf = cf_stmts[0]
                ocf = cf.get("operating_cash_flow")
                fcf = cf.get("free_cash_flow")
                period_cf = cf.get("period", "latest annual")
                if ocf is not None:
                    anchor_lines.append(f"  [{t}] Operating cash flow ({period_cf}): EXACTLY ${ocf/1e9:.2f}B")
                if fcf is not None:
                    anchor_lines.append(f"  [{t}] Free cash flow ({period_cf}): EXACTLY ${fcf/1e9:.2f}B")
            if bs_stmts:
                bs = bs_stmts[0]
                tot_a = bs.get("total_assets")
                tot_l = bs.get("total_liabilities")
                cash_b = bs.get("cash_and_equivalents")
                ltd_b  = bs.get("long_term_debt")
                period_bs = bs.get("period", "latest annual")
                if tot_a is not None:
                    anchor_lines.append(f"  [{t}] Total assets ({period_bs}): EXACTLY ${tot_a/1e9:.2f}B")
                if tot_l is not None:
                    anchor_lines.append(f"  [{t}] Total liabilities ({period_bs}): EXACTLY ${tot_l/1e9:.2f}B")
                if tot_a is not None and tot_l is not None:
                    eq = tot_a - tot_l
                    anchor_lines.append(f"  [{t}] Shareholders equity ({period_bs}): EXACTLY ${eq/1e9:.2f}B")
                if cash_b is not None:
                    anchor_lines.append(f"  [{t}] Cash & equivalents ({period_bs}): EXACTLY ${cash_b/1e9:.2f}B")
            if inc_stmts:
                inc = inc_stmts[0]
                rev_a  = inc.get("revenue")
                ni_a   = inc.get("net_income")
                oi_a   = inc.get("operating_income")
                period_inc = inc.get("period", "latest annual")
                if rev_a is not None:
                    anchor_lines.append(f"  [{t}] Annual revenue ({period_inc}): EXACTLY ${rev_a/1e9:.2f}B")
                if ni_a is not None:
                    anchor_lines.append(f"  [{t}] Annual net income ({period_inc}): EXACTLY ${ni_a/1e9:.2f}B")
    anchor_lines.append("╔══════════════════════════════════════════════════════════════════════╗")
    anchor_lines.append("║  END LOCKED DATA ANCHOR                                              ║")
    anchor_lines.append("╚══════════════════════════════════════════════════════════════════════╝")
    ctx_parts.append("")
    ctx_parts.extend(anchor_lines)
    ctx_parts.append("")

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

    for sr_o in effective_sr:
        if sr_o:
            ctx_parts.extend(_build_stock_research_context(sr_o))
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
        f"      specific number or mechanism.\n"
        f"  (H) LOCKED DATA VALUES — the LOCKED DATA ANCHOR block in the data section lists the\n"
        f"      ONLY valid values for P/E, EV/EBITDA, ROE, ROIC, Piotroski, Beneish, Altman Z,\n"
        f"      WACC, and DCF. Copy those exact numbers — do NOT use any other values for these\n"
        f"      metrics even if you believe you know the 'correct' figure from training knowledge.\n\n"
        f"## Executive Summary"
    )

    # ── 3. Trim prompt to fit within native 8192-token context ───────────────
    # deepseek-r1:8b loaded with default num_ctx=8192. Passing num_ctx=16384
    # via options causes Ollama to RELOAD the model, which OOMs on Apple Silicon
    # and returns HTTP 500.  Instead we trim the context section to ≤ ~20 000
    # chars (≈5 000 tokens at ~4 chars/token), leaving ~3 000 tokens for output.
    #
    # Trimming strategy: keep the system instructions intact (first ~4 000 chars
    # and last ~2 000 chars of the full prompt); cut from the middle context block.
    def _trim_prompt_to_window(text: str, max_chars: int = 20_000) -> str:
        if len(text) <= max_chars:
            return text
        keep_head = (max_chars * 2) // 3   # ~13 333 chars (instructions + early context)
        keep_tail = max_chars - keep_head   # ~6 667 chars (closing instructions)
        return (
            text[:keep_head]
            + "\n\n[...context trimmed to fit 8192-token context window...]\n\n"
            + text[-keep_tail:]
        )

    prompt = _trim_prompt_to_window(prompt)

    # deepseek-reasoner:
    #   single:     3000 tok ≈ 200-300s + ~45s prefill ≈ 4-6 min  → target 8-12 min total
    #   comparison: 3000 tok ≈ 200-300s + ~45s prefill ≈ 4-6 min
    # Raised from 6000 → 8000 to give enough headroom for all 11 required sections
    # (target 5000-7000 words ≈ 7000-9000 tokens at ~1.3 tok/word).
    max_tokens = 8000
    try:
        raw, reasoning = _deepseek_generate(
            _SUMMARIZER_MODEL, prompt, max_tokens=max_tokens, temperature=0.2,
            timeout=_SUMMARIZER_TIMEOUT, system_prompt=_SUMMARIZER_SYSTEM,
            return_reasoning=True,
        )
        if _trace_out is not None:
            _trace_out.append(reasoning or "")
        cleaned = _strip_think(raw).strip()
        if not cleaned:
            cleaned = "Summary unavailable (LLM returned empty response)."
    except Exception as exc:
        logger.error("Summarizer LLM failed: %s", exc)
        cleaned = f"Summary unavailable ({type(exc).__name__}: {exc})."
        if _trace_out is not None:
            _trace_out.append("")

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
        r"|target\s+price\s+of\s+\$|price\s+target\s+of\s+\$|price\s+objective\s+of\s+\$|12.month\s+target"
        r"|upside\s+to\s+\$|downside\s+to\s+\$|we\s+initiate|we\s+recommend"
        r"|recommended\s+as\s+a|is\s+recommended\s+as|prepared\s+by"
        r"|this\s+report\s+is\s+intended\s+for\s+informational"
        r"|does\s+not\s+constitute\s+a\s+recommendation"
        r"|investors\s+should\s+conduct\s+their\s+own\s+due\s+diligence"
        r"|bloomberg\s+consensus"
        r"|recommendation\s*:"
        r"|investment\s+recommendation"
        r"|analyst\s+recommendation"
        r"|consensus\s+estimate"
        r"|wall\s+street\s+consensus"
        r"|analyst\s+consensus"
        r"|\bEPS\s+estimate\b"
        r"|forward\s+guidance\s+of\s+\$"
        r"|(?:12|12-month)\s*(?:price\s*)?(?:target|PT)\s*[:=\$]"
        r"|\bPT\s*[:=]\s*\$"
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

    # ── 4g. Metric value correction pass (via FactChecker) ──────────────────
    # Uses the structured validation.FactChecker to detect and replace any
    # numeric metric values the LLM hallucinated with authoritative DB values.
    try:
        from .validation import validate_quant_output, FactChecker  # type: ignore[import]
        _fact_checker = FactChecker()
        _correction_tickers = tickers or ([effective_ticker] if effective_ticker else [])
        logger.info(
            "[4g] FactChecker metric correction: tickers=%s  quant_count=%d",
            _correction_tickers, len(effective_quant),
        )
        for i, t in enumerate(_correction_tickers):
            quant_o = effective_quant[i] if i < len(effective_quant) else None
            fm_o = effective_fm[i] if i < len(effective_fm) else None
            logger.info("[4g] ticker=%s  quant_o_present=%s  fm_o_present=%s", t, quant_o is not None, fm_o is not None)
            if quant_o:
                metrics = validate_quant_output(quant_o, fm_output=fm_o)
                if metrics:
                    cleaned, corrections = _fact_checker.correct_report(cleaned, metrics)
                    if corrections:
                        logger.info(
                            "[4g] FactChecker corrected %d value(s) for %s: %s",
                            len(corrections), t, "; ".join(corrections),
                        )
                    else:
                        logger.info("[4g] FactChecker: no corrections needed for %s", t)
    except Exception as _fc_exc:
        logger.warning("[4g] FactChecker failed (non-fatal): %s", _fc_exc)

    # ── 5. Replace any residual chunk_id tokens with [N] numbers ─────────────
    cleaned = inject_inline_numbers(cleaned, all_chunk_id_maps)

    # ── 5b. Fix double-decimal artefacts (e.g. "$435.62.62B", "18.72.8%") ────
    # The Stage-2 LLM sometimes duplicates or concatenates decimal parts.
    _bad_count = len(re.findall(r'\d+\.\d+\.\d+', cleaned))
    if _bad_count:
        cleaned = re.sub(r'(\d+\.\d+)\.\d+', r'\1', cleaned)
        logger.warning("[summarise_results] Fixed %d double-decimal artefacts", _bad_count)

    # ── 6. Append the references block ───────────────────────────────────────
    # Always include a references section - even if empty, to ensure completeness
    if combined_ref_block:
        cleaned = cleaned + "\n" + combined_ref_block
    else:
        # Fallback: include at least a placeholder references section
        cleaned = cleaned + "\n\n---\n### References\n*No external citations available — analysis based on internal data sources.*"

    return cleaned


# ── Translation Helper ───────────────────────────────────────────────────────────

def translate_text(text: str, target_language: str) -> str:
    """Translate text to the target language using DeepSeek."""

    system_prompt = (
        "You are a senior translator for financial research. Translate the provided report into "
        f"{target_language} while preserving formatting, headers, numeric citations, and analytical tone."
        " Do not add any commentary or change the numeric citations."
    )

    prompt = """Translate the following financial research report:

""" + text + "\n\nTranslation:"""

    try:
        translated = _deepseek_generate(
            _TRANSLATION_MODEL,
            prompt,
            max_tokens=4096,
            temperature=0.1,
            system_prompt=system_prompt,
        )
        translated = _strip_think(translated)
        translated = _strip_fences(translated)
        return translated.strip() or text
    except Exception as exc:
        logger.warning("Translation via DeepSeek failed: %s", exc)
        return text


__all__ = ["plan_query", "summarise_results", "translate_text"]
