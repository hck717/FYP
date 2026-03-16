"""Business Analyst Agent — Optimized Adaptive Agentic Graph RAG pipeline.

Architecture:

    Query + Ticker
        |
        v
    metadata_precheck         <- Neo4j chunk counts, pgvector status, sentiment availability
        |
        v
    precheck_data_coverage    <- Coverage warnings, sentiment freshness, sentiment-query short-circuit
        |
        v
    fetch_sentiment_data      <- PostgreSQL sentiment snapshot (with 7-day recency gate + NLP fallback)
        |
        v
    complex_retrieval         <- unified multi-stage retrieval (Neo4j + pgvector + BM25 + CE + RRF)
        |
        v
    generate_analysis         <- LLM decides whether context is sufficient
        |
        +-> INSUFFICIENT_DATA -> web_search_fallback
        |                         |
        +-------------------------+
        |
        v
    semantic_cache_check      <- Hit? Return cached -> Miss? Proceed + cache result
        |
        v
    format_json_output        <- Structured JSON for Supervisor
        |
       END -> return to Supervisor

Usage (CLI):
    python -m agents.business_analyst.agent --ticker AAPL
    python -m agents.business_analyst.agent --ticker AAPL --task "What is Apple's competitive moat?"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, cast

# Load .env early so DEEPSEEK_API_KEY and other env vars are available
# before config.py dataclass field factories run.  This is a no-op inside
# Docker (vars are injected by docker-compose) and safe to call multiple times.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed — rely on env vars being set externally

from langgraph.graph import END, StateGraph

# orchestration.data_availability is used by tools.py (guarded there).
# We do NOT import it here at module level — it is not used in this file
# and a hard top-level import would crash when /opt/airflow is not on
# sys.path (e.g. the first docker exec invocation before PYTHONPATH is set).

from .config import BusinessAnalystConfig, load_config
from .llm import LLMClient
from .schema import CRAGStatus, MetadataProfile, RetrievalResult, SentimentSnapshot, serialise_chunk
from .tools import BusinessAnalystToolkit
from .web_search_interface import web_search_fallback as _call_web_search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verbose progress printer
# ---------------------------------------------------------------------------

# Module-level flag; set to True by --verbose CLI flag or verbose=True in run().
_VERBOSE: bool = False

_STEP_COUNTER: int = 0
# Per-run thinking trace — collected across all nodes, embedded in final output
_THINKING_TRACE: List[Dict[str, Any]] = []


def _print_progress(step: str, detail: str = "", *, symbol: str = ">>") -> None:
    """Print a pipeline step to stderr when verbose mode is active.

    Output goes to stderr so it does not pollute the JSON stdout output.
    Format:  [step N] >> STEP_NAME  detail

    Also appends a record to the module-level _THINKING_TRACE list so that the
    Streamlit UI can surface each step without needing verbose mode.
    """
    global _STEP_COUNTER, _THINKING_TRACE
    _STEP_COUNTER += 1
    ts = datetime.now().strftime("%H:%M:%S")
    step_num = _STEP_COUNTER
    # Always record to trace (regardless of _VERBOSE)
    _THINKING_TRACE.append({
        "step": step_num,
        "ts": ts,
        "symbol": symbol.strip(),
        "name": step,
        "detail": detail,
    })
    if not _VERBOSE:
        return
    line = f"[{ts}] [{step_num:02d}] {symbol} {step}"
    if detail:
        line += f"  |  {detail}"
    print(line, file=sys.stderr, flush=True)


def _reset_step_counter() -> None:
    global _STEP_COUNTER, _THINKING_TRACE
    _STEP_COUNTER = 0
    _THINKING_TRACE = []



class AgentState(TypedDict, total=False):
    """Shared mutable state threaded through every node."""

    # Inputs
    task: str
    ticker: Optional[str]

    # Availability profile (optional, injected by orchestration planner)
    availability_profile: Optional[Dict[str, bool]]

    # Metadata pre-check profile (Neo4j counts, pgvector status, sentiment flag)
    metadata_profile: Optional[MetadataProfile]

    # Pre-check layer results (populated by _node_precheck_data_coverage)
    data_coverage_warning: Optional[str]   # human-readable warning if data is thin/stale
    use_sentiment_db: bool                 # True → use PostgreSQL snapshot; skip heavy retrieval for sentiment queries
    sentiment_is_fresh: bool               # True → PostgreSQL sentiment is within 7-day freshness window

    # Enrichment
    sentiment: Optional[SentimentSnapshot]
    company_node: Optional[Dict[str, Any]]   # raw Neo4j Company node properties
    community_summary: Optional[str]         # graph-community summary (2A: Graph RAG)
    retrieval: Optional[RetrievalResult]

    # CRAG evaluation
    crag_status: Optional[CRAGStatus]
    confidence: float

    # Generation
    llm_output: Optional[Dict[str, Any]]
    fallback_triggered: bool
    web_search_result: Optional[Dict[str, Any]]

    # Thinking trace — list of step dicts appended by each node for UI display
    thinking_trace: List[Dict[str, Any]]

    # Final output
    output: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def _node_metadata_precheck(state: AgentState, toolkit: BusinessAnalystToolkit) -> AgentState:
    """Fetch and cache the metadata profile for this ticker.

    Populates ``state["metadata_profile"]`` with Neo4j chunk counts, pgvector
    status, and sentiment availability flags.  The result is used by later
    nodes to make data-aware routing decisions without re-querying.
    """
    ticker = state.get("ticker")
    _print_progress("METADATA PRECHECK", f"ticker={ticker}")
    if not ticker:
        return state
    t0 = time.monotonic()
    profile = toolkit.get_metadata_profile(ticker)
    elapsed = time.monotonic() - t0
    _print_progress(
        "METADATA PRECHECK done",
        f"neo4j_chunks={profile.neo4j_chunk_count}  "
        f"pgvec_chunks={profile.pgvector_chunk_count}  "
        f"sentiment={'yes' if profile.has_sentiment else 'no'}  "
        f"index={'ONLINE' if profile.neo4j_chunk_index_ready else 'OFFLINE'}  "
        f"({elapsed:.2f}s)",
        symbol="  OK",
    )
    state_update = {**state, "metadata_profile": profile}
    # Log any pre-existing data_coverage_warning set by an earlier run (defensive).
    existing_warning: Optional[str] = state.get("data_coverage_warning")
    if existing_warning:
        logger.warning(
            "[MetadataPrecheck] Existing data_coverage_warning for ticker=%s: %s",
            ticker, existing_warning,
        )
    return cast(AgentState, state_update)


def _node_precheck_data_coverage(state: AgentState) -> AgentState:
    """Pre-check data coverage and freshness before the main pipeline.

    Reads the ``metadata_profile`` (populated by ``_node_metadata_precheck``)
    and sets three state flags:

    * ``data_coverage_warning`` — human-readable warning string when chunk
      coverage or sentiment data is thin/stale; ``None`` when everything looks
      healthy.
    * ``use_sentiment_db`` — ``True`` when the PostgreSQL sentiment snapshot is
      within the 7-day freshness window.  The ``_node_fetch_sentiment`` step
      will still hit the DB; this flag lets later routing logic skip heavy RAG
      for sentiment-only queries.
    * ``sentiment_is_fresh`` — mirrors ``use_sentiment_db``; exposed separately
      so the final output and downstream tools can reason about freshness
      without rechecking the raw date string.

    For queries that are primarily sentiment/news-focused AND the DB snapshot
    is fresh, the CRAG evaluation is short-circuited by pre-setting
    ``crag_status=CORRECT`` and ``confidence=1.0``.  This avoids wasting
    bi-encoder recall and cross-encoder rerranking cycles when PostgreSQL
    already provides a definitive answer.

    All warnings are logged at WARNING level so they surface in production
    logs without requiring DEBUG verbosity.
    """
    profile: Optional[MetadataProfile] = state.get("metadata_profile")
    query: str = state.get("task", "")
    ticker: Optional[str] = state.get("ticker")
    _print_progress("PRECHECK DATA COVERAGE", f"ticker={ticker}")

    if profile is None:
        # No profile yet (ticker was None or metadata_precheck failed) — skip.
        _print_progress("PRECHECK DATA COVERAGE", "no profile — skipping", symbol="  --")
        return {**state, "data_coverage_warning": None, "use_sentiment_db": False, "sentiment_is_fresh": False}

    warnings_parts: List[str] = []

    # -----------------------------------------------------------------------
    # 1. Chunk coverage check
    # -----------------------------------------------------------------------
    MIN_NEO4J_CHUNKS = 20
    MIN_PG_CHUNKS = 5

    if profile.neo4j_chunk_count < MIN_NEO4J_CHUNKS:
        msg = (
            f"Neo4j chunk count for {ticker} is low "
            f"({profile.neo4j_chunk_count} < {MIN_NEO4J_CHUNKS}). "
            "Run the FMP ingestion DAG to populate the knowledge base."
        )
        warnings_parts.append(msg)
        logger.warning("[PreCheck] %s", msg)

    if profile.pgvector_chunk_count < MIN_PG_CHUNKS:
        msg = (
            f"pgvector chunk count for {ticker} is low "
            f"({profile.pgvector_chunk_count} < {MIN_PG_CHUNKS}). "
            "Run the pgvector ingestion step to populate embeddings."
        )
        warnings_parts.append(msg)
        logger.warning("[PreCheck] %s", msg)

    # -----------------------------------------------------------------------
    # 2. Sentiment freshness check (7-day window)
    # -----------------------------------------------------------------------
    sentiment_fresh = False
    if profile.sentiment_last_updated:
        try:
            from datetime import date
            last_date = datetime.strptime(profile.sentiment_last_updated, "%Y-%m-%d").date()
            age_days = (datetime.now(timezone.utc).date() - last_date).days
            if age_days <= 7:
                sentiment_fresh = True
                logger.info(
                    "[PreCheck] Sentiment for %s is FRESH (age=%d days, last_updated=%s).",
                    ticker, age_days, profile.sentiment_last_updated,
                )
            else:
                msg = (
                    f"Sentiment data for {ticker} is stale "
                    f"(last_updated={profile.sentiment_last_updated}, age={age_days} days > 7). "
                    "PostgreSQL DB will be skipped; local NLP fallback will be used."
                )
                warnings_parts.append(msg)
                logger.warning("[PreCheck] %s", msg)
        except (ValueError, TypeError) as exc:
            logger.warning(
                "[PreCheck] Could not parse sentiment_last_updated=%r for ticker=%s: %s",
                profile.sentiment_last_updated, ticker, exc,
            )
    elif not profile.has_sentiment:
        msg = (
            f"No sentiment data found for {ticker} in PostgreSQL. "
            "Local NLP fallback will be used during analysis."
        )
        warnings_parts.append(msg)
        logger.warning("[PreCheck] %s", msg)

    data_coverage_warning: Optional[str] = "; ".join(warnings_parts) if warnings_parts else None

    # -----------------------------------------------------------------------
    # 2.5  Ticker / query mismatch validation
    # -----------------------------------------------------------------------
    # Check if the query explicitly mentions a company whose ticker does NOT
    # match the requested ticker.  This catches accidental mismatches like
    # "Tell me about Microsoft" with ticker=AAPL.
    # We only warn — we do NOT block the pipeline, as the user may have
    # intentionally asked about a competitor while scoping to ticker data.
    _TICKER_ALIASES: Dict[str, List[str]] = {
        "AAPL":  ["apple", "apple inc"],
        "MSFT":  ["microsoft", "microsoft corp"],
        "GOOGL": ["google", "alphabet", "alphabet inc"],
        "GOOG":  ["google", "alphabet", "alphabet inc"],
        "AMZN":  ["amazon", "amazon.com"],
        "NVDA":  ["nvidia", "nvidia corp"],
        "META":  ["meta", "meta platforms", "facebook"],
        "TSLA":  ["tesla", "tesla inc", "tesla motors"],
        "NFLX":  ["netflix", "netflix inc"],
        "BABA":  ["alibaba", "alibaba group"],
        "TSM":   ["tsmc", "taiwan semiconductor"],
        "INTC":  ["intel", "intel corp"],
        "AMD":   ["amd", "advanced micro devices"],
        "CRM":   ["salesforce", "salesforce.com"],
        "ORCL":  ["oracle", "oracle corp"],
        "SAP":   ["sap", "sap se"],
        "SONY":  ["sony", "sony group"],
        "TCEHY": ["tencent", "tencent holdings"],
        "BIDU":  ["baidu", "baidu inc"],
        "JD":    ["jd.com", "jingdong"],
    }
    if ticker:
        ticker_upper = ticker.upper()
        query_lower_check = query.lower()
        # Names that belong to OTHER tickers but appear in the query
        foreign_company_mentions: List[str] = []
        for other_ticker, aliases in _TICKER_ALIASES.items():
            if other_ticker == ticker_upper:
                continue  # skip own ticker aliases
            for alias in aliases:
                # Use word-boundary matching to avoid false positives (e.g. "apple" in "pineapple")
                if re.search(r"\b" + re.escape(alias) + r"\b", query_lower_check):
                    foreign_company_mentions.append(f"{alias} ({other_ticker})")
                    break  # one match per ticker is enough
        if foreign_company_mentions:
            mismatch_msg = (
                f"Query mentions {', '.join(foreign_company_mentions)} "
                f"but analysis is scoped to ticker={ticker_upper}. "
                "Results will only reflect data for the requested ticker. "
                "If you intended a different company, re-run with the correct ticker."
            )
            if data_coverage_warning:
                data_coverage_warning = data_coverage_warning + "; " + mismatch_msg
            else:
                data_coverage_warning = mismatch_msg
            logger.warning("[PreCheck] Ticker mismatch: %s", mismatch_msg)

    # -----------------------------------------------------------------------
    # 3. Query-type-aware short-circuit for sentiment-only queries
    # -----------------------------------------------------------------------
    # If the query is primarily about sentiment/news AND the DB snapshot is
    # fresh, pre-set crag_status=CORRECT so the complex retrieval path is
    # skipped entirely for this lightweight query type.
    # IMPORTANT: do NOT short-circuit multi-pillar / comprehensive analysis
    # queries — these contain the word "sentiment" as one of many pillars but
    # require full RAG retrieval to produce meaningful output.
    # Keywords that indicate the query is PURELY about sentiment/market-feel.
    # The short-circuit should ONLY fire when the query is exclusively asking
    # about sentiment numbers/trends — NOT when "sentiment" appears alongside
    # analytical topics like risks, business model, competitive position, etc.
    _PURE_SENTIMENT_KEYWORDS = (
        "how is the market feeling",
        "what does the market think",
        "what is the sentiment",
        "current sentiment",
        "sentiment score",
        "sentiment trend",
        "sentiment breakdown",
        "crowd sentiment",
        "wall street view",
        "investor sentiment",
        "analyst sentiment",
        "news sentiment",
        "overall sentiment",
        "market sentiment",
    )
    # Keywords that signal a BROADER analytical question — presence of any of
    # these means the query is NOT a pure-sentiment query even if it also
    # mentions sentiment in passing.
    _ANALYTICAL_TOPIC_MARKERS = (
        "risk", "business model", "competitive", "moat", "revenue", "growth",
        "margin", "valuation", "earnings", "guidance", "management", "strategy",
        "products", "services", "market share", "regulation", "macro",
        "financial", "profitability", "supply chain", "demand", "esg",
        "opportunities", "threats", "strengths", "weaknesses", "swot",
        "explain", "analyze", "assessment", "overview", "summary",
        "how does", "why does", "what are", "describe",
        "pillar", "all pillars", "comprehensive", "all of the following",
        "covering all", "full analysis", "deeply comprehensive",
    )
    query_lower = query.lower()
    # A query is "pure sentiment" only when it contains a sentiment-specific
    # phrase AND does NOT contain any broader analytical topic markers.
    is_pure_sentiment_query = (
        any(kw in query_lower for kw in _PURE_SENTIMENT_KEYWORDS)
        and not any(m in query_lower for m in _ANALYTICAL_TOPIC_MARKERS)
    )
    is_comprehensive_query = any(m in query_lower for m in _ANALYTICAL_TOPIC_MARKERS)
    # Also treat very long queries as comprehensive (run_full_analysis task is ~4 KB)
    if len(query) > 1000:
        is_comprehensive_query = True

    base: Dict[str, Any] = {
        **cast(dict, state),
        "data_coverage_warning": data_coverage_warning,
        "use_sentiment_db": sentiment_fresh,
        "sentiment_is_fresh": sentiment_fresh,
    }

    if is_pure_sentiment_query and sentiment_fresh and not is_comprehensive_query:
        logger.info(
            "[PreCheck] Sentiment-focused query with fresh DB snapshot for %s — "
            "pre-setting crag_status=CORRECT to skip complex RAG.",
            ticker,
        )
        _print_progress(
            "PRECHECK DATA COVERAGE done",
            f"sentiment_fresh=True  query_type=sentiment-focused  → crag_status=CORRECT  "
            f"warning={'yes' if data_coverage_warning else 'no'}",
            symbol="  OK",
        )
        base = {
            **base,
            "crag_status": CRAGStatus.CORRECT,
            "confidence": 1.0,
        }
    else:
        _print_progress(
            "PRECHECK DATA COVERAGE done",
            f"sentiment_fresh={sentiment_fresh}  "
            f"neo4j={profile.neo4j_chunk_count}  pgvec={profile.pgvector_chunk_count}  "
            f"warning={'yes' if data_coverage_warning else 'no'}",
            symbol="  OK" if not data_coverage_warning else "  !?",
        )

    return cast(AgentState, base)


def _node_fetch_sentiment(state: AgentState, toolkit: BusinessAnalystToolkit) -> AgentState:
    """Fetch bullish/bearish/neutral % from PostgreSQL sentiment_trends.
    Also fetches the Company node properties from Neo4j for company_overview,
    and builds a graph-community summary (2A: Graph RAG).
    """
    ticker = state.get("ticker")
    _print_progress("FETCH SENTIMENT", f"ticker={ticker}")
    t0 = time.monotonic()
    sentiment = toolkit.fetch_sentiment(ticker)
    company_node = toolkit.fetch_company_overview(ticker)
    community_summary = toolkit.fetch_community_summary(ticker)
    elapsed = time.monotonic() - t0
    if sentiment:
        _print_progress(
            "FETCH SENTIMENT done",
            f"bullish={sentiment.bullish_pct:.1f}%  bearish={sentiment.bearish_pct:.1f}%  "
            f"neutral={sentiment.neutral_pct:.1f}%  trend={sentiment.trend}  ({elapsed:.2f}s)",
            symbol="  OK",
        )
    else:
        _print_progress("FETCH SENTIMENT done", f"no sentiment data  ({elapsed:.2f}s)", symbol="  --")
    if community_summary:
        logger.info("[BA] Graph community summary for %s: %s", ticker, community_summary[:120])
    return {**state, "sentiment": sentiment, "company_node": company_node, "community_summary": community_summary}


def _node_complex_retrieval(state: AgentState, toolkit: BusinessAnalystToolkit) -> AgentState:
    """Full multi-stage retrieval for COMPLEX queries.

    Stage 1 — Bi-encoder recall (Neo4j + pgvector, top-100 each)
    Stage 2 — Cross-encoder rerank + graph traversal
    Fusion   — Reciprocal Rank Fusion (RRF)

    Replaces the old ``_node_hybrid_retrieval`` on the complex path.
    """
    query = state.get("task", "")
    ticker = state.get("ticker")
    profile: Optional[Dict[str, bool]] = state.get("availability_profile")

    if profile is not None and not profile.get("has_any_qualitative", True):
        logger.warning(
            "[BA] Availability profile reports no qualitative data for ticker=%s "
            "— attempting retrieval anyway (profile may be stale or incomplete).",
            ticker,
        )

    if ticker and not query.upper().startswith(ticker.upper()):
        query = f"{ticker.upper()}: {query}"

    _print_progress("COMPLEX RETRIEVAL (multi-stage)", f"ticker={ticker}")
    t0 = time.monotonic()
    retrieval = toolkit.retrieve_multi_stage(query, ticker)
    elapsed = time.monotonic() - t0
    n_chunks = len(retrieval.chunks) if retrieval else 0
    n_facts = len(retrieval.graph_facts) if retrieval else 0
    top_score = retrieval.chunks[0].score if (retrieval and retrieval.chunks) else 0.0
    _print_progress(
        "COMPLEX RETRIEVAL done",
        f"chunks={n_chunks}  graph_facts={n_facts}  top_score={top_score:.3f}  ({elapsed:.2f}s)",
        symbol="  OK",
    )
    return {**state, "retrieval": retrieval}


def _node_generate_analysis(state: AgentState, llm: LLMClient) -> AgentState:
    """Generate structured JSON analysis from retrieved context (CORRECT path)."""
    retrieval: Optional[RetrievalResult] = state.get("retrieval")
    sentiment: Optional[SentimentSnapshot] = state.get("sentiment")
    task = state.get("task", "")
    ticker = state.get("ticker", "")
    community_summary: Optional[str] = state.get("community_summary")

    chunks = retrieval.chunks if retrieval else []
    graph_facts = retrieval.graph_facts if retrieval else []

    # Final ticker guard — Neo4j vector search may return chunks with
    # unknown/missing metadata. Re-apply here so the LLM context is guaranteed clean.
    if ticker:
        before = len(chunks)
        from .tools import _chunk_ticker_matches
        chunks = [c for c in chunks if _chunk_ticker_matches(c, ticker)]
        dropped = before - len(chunks)
        if dropped:
            logger.warning(
                "[_node_generate_analysis] Removed %d off-ticker chunk(s) before LLM context "
                "build for ticker=%s.",
                dropped,
                ticker,
            )

    _print_progress(
        "GENERATE ANALYSIS (LLM)",
        f"ticker={ticker}  chunks={len(chunks)}  graph_facts={len(graph_facts)}",
    )

    # --- No-data guard ---
    # If there are no retrieved chunks at all, short-circuit immediately —
    # UNLESS the precheck layer validated the data OR structured data (sentiment,
    # company_node) is available.  In those cases the LLM can produce a meaningful
    # analysis from structured DB data even without document chunks.
    company_node: Optional[Dict[str, Any]] = state.get("company_node")
    precheck_correct = state.get("crag_status") == CRAGStatus.CORRECT
    has_structured_data = sentiment is not None or company_node is not None

    if not chunks and not (precheck_correct or has_structured_data):
        logger.warning(
            "No chunks retrieved for ticker=%s — returning INSUFFICIENT_DATA without LLM call.", ticker
        )
        raw = {
            "qualitative_summary": (
                f"INSUFFICIENT_DATA: No documents found in the knowledge base for ticker {ticker}. "
                "Analysis cannot be performed without source data."
            ),
            "crag_status": CRAGStatus.INCORRECT.value,
            "key_risks": [],
            "missing_context": [
                {
                    "gap": f"No news articles or filings indexed for {ticker} in the vector store.",
                    "severity": "HIGH",
                }
            ],
        }
        return {**state, "llm_output": raw, "fallback_triggered": False}

    if not chunks:
        logger.info(
            "[GenerateAnalysis] No chunks but structured data available (sentiment=%s, company_node=%s) — "
            "proceeding with structured-data-only LLM call for ticker=%s.",
            sentiment is not None,
            company_node is not None,
            ticker,
        )

    # Inject company_node into context if retrieved chunks are thin
    _company_context_prefix: Optional[str] = None
    if company_node and (not chunks or len(chunks) < 3):
        company_text = json.dumps(
            {k: v for k, v in company_node.items()
             if k in ("Name", "Sector", "Industry", "Description", "Country",
                      "MarketCapitalization", "FullTimeEmployees",
                      "Highlights_ProfitMargin", "Highlights_RevenueGrowth",
                      "Highlights_EarningsShare", "Highlights_PERatio",
                      "Highlights_52WeekHigh", "Highlights_52WeekLow",
                      "AnalystRatings_Rating", "AnalystRatings_TargetPrice",
                      "AnalystRatings_StrongBuy", "AnalystRatings_Buy",
                      "AnalystRatings_Hold", "AnalystRatings_Sell",
                      "AnalystRatings_StrongSell")},
            indent=2, default=str,
        )
        _company_context_prefix = (
            f"=== COMPANY OVERVIEW (Neo4j: Company node) ===\n{company_text}\n"
        )

    # Build a rich context block for the LLM — ordered by relevance score
    context_parts: List[str] = []

    # Task question framed prominently so the LLM keeps it front-of-mind
    context_parts.append(f"=== ANALYST QUESTION ===\n{task}\n")

    # Company overview from Neo4j Company node (prepended when chunks are thin/absent)
    if _company_context_prefix:
        context_parts.append(_company_context_prefix)

    # Sentiment data — framed as a signal to interpret, not just numbers
    if sentiment:
        trend_label = sentiment.trend or "unknown"
        context_parts.append(
            f"=== SENTIMENT DATA (PostgreSQL: sentiment_trends) ===\n"
            f"Ticker: {ticker}\n"
            f"Bullish: {sentiment.bullish_pct}%  Bearish: {sentiment.bearish_pct}%  "
            f"Neutral: {sentiment.neutral_pct}%\n"
            f"Trend: {trend_label}\n"
            f"(Interpret what this distribution signals about market perception given the question above)\n"
        )

    # Graph facts from Neo4j (relationships: risks, strategies, competitors)
    if graph_facts:
        facts_json = json.dumps(graph_facts[:25], indent=2)
        context_parts.append(f"=== GRAPH FACTS (Neo4j: Company relationships) ===\n{facts_json}\n")

    # Graph community summary (2A: Graph RAG — relationship-count centrality)
    if community_summary:
        context_parts.append(
            f"=== GRAPH COMMUNITY SUMMARY (Neo4j: relationship centrality) ===\n"
            f"{community_summary}\n"
            f"(Use this to understand the company's most prominent strategic/risk/competitive links)\n"
        )

    # Top retrieved document chunks with explicit chunk IDs for citation
    # 2B: Separate recent vs historical chunks for contrastive analysis
    recent_chunks = [c for c in chunks[:7] if (c.metadata or {}).get("temporal_band") == "recent"]
    historical_chunks = [c for c in chunks[:7] if (c.metadata or {}).get("temporal_band") == "historical"]
    other_chunks = [c for c in chunks[:7] if (c.metadata or {}).get("temporal_band") not in ("recent", "historical")]

    if chunks:
        context_parts.append(f"=== RETRIEVED DOCUMENT CHUNKS (cite these IDs in your analysis) ===")
        if recent_chunks or historical_chunks:
            context_parts.append(
                "(2B: Chunks are tagged by recency. Where both RECENT and HISTORICAL chunks exist, "
                "analyse the *delta* — what has changed and what the trend implies.)"
            )
        for chunk in chunks[:7]:
            band = (chunk.metadata or {}).get("temporal_band", "")
            band_label = f", temporal={band}" if band else ""
            source_name = (chunk.metadata or {}).get("source_name", "")
            source_label = f", source_name={source_name!r}" if source_name else f", source={chunk.source}"
            context_parts.append(
                f"\n[chunk_id: {chunk.chunk_id}] (relevance={chunk.score:.3f}{source_label}{band_label})\n"
                f"{chunk.text[:900]}"  # 900 chars per chunk
            )

    context = "\n".join(context_parts)

    try:
        t0 = time.monotonic()
        raw = llm.generate(
            query=task,
            ticker=ticker,
            context=context,
            sentiment=sentiment,
        )
        elapsed = time.monotonic() - t0
        _print_progress(
            "GENERATE ANALYSIS done",
            f"({elapsed:.2f}s)",
            symbol="  OK",
        )
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        _print_progress("GENERATE ANALYSIS failed", str(exc), symbol="  !!")
        raw = {
            "qualitative_summary": f"GENERATION_ERROR: {exc}",
            "crag_status": state.get("crag_status", CRAGStatus.INCORRECT).value  # type: ignore[union-attr]
        }
    return {**state, "llm_output": raw, "fallback_triggered": False}


def _node_web_search_fallback(state: AgentState, toolkit: BusinessAnalystToolkit) -> AgentState:
    """Trigger Web Search Agent when CRAG confidence is INCORRECT (AMBIGUOUS exhausted or < 0.35)."""
    ticker = state.get("ticker")
    query = state.get("task", "")
    logger.info("CRAG INCORRECT — triggering web search fallback for ticker=%s", ticker)
    _print_progress("WEB SEARCH FALLBACK", f"ticker={ticker}  query={query[:60]!r}", symbol=" !!")

    t0 = time.monotonic()
    result = _call_web_search(query=query, ticker=ticker, config=toolkit.config)
    elapsed = time.monotonic() - t0
    _print_progress("WEB SEARCH FALLBACK done", f"({elapsed:.2f}s)", symbol="  OK")
    return {**state, "web_search_result": result, "fallback_triggered": True}


def _check_citation_grounding(
    llm_output: Dict[str, Any],
    retrieval: Optional[RetrievalResult],
) -> None:
    """Warn if the LLM output contains chunk_ids not present in the retrieved set.

    Hallucinated citations are chunk_id strings in the JSON output that do not
    correspond to any chunk actually returned by the retriever.  We detect them
    with a simple string scan of the serialised output so we catch citations
    buried anywhere in the nested structure.
    """
    if not retrieval or not retrieval.chunks:
        return

    real_ids: set[str] = {c.chunk_id for c in retrieval.chunks if c.chunk_id}
    if not real_ids:
        return

    output_text = json.dumps(llm_output, ensure_ascii=False)
    # Extract cited IDs handling both old and new citation formats:
    #   Old:  [TICKER::section::hash] or [neo4j::...]
    #   New:  [source_name | TICKER::section::hash]
    # Step 1: extract the full bracket content where :: is present
    _bracket_contents = re.findall(r'\[([^\]]+::[^\]]+)\]', output_text)
    cited_ids: set[str] = set()
    for raw in _bracket_contents:
        raw = raw.strip().rstrip("].,;")
        # New format: "source_name | chunk_id"
        if " | " in raw:
            cid = raw.split(" | ", 1)[1].strip().rstrip("].,;")
        else:
            cid = raw
        cited_ids.add(cid)
    # Use prefix matching with Unicode normalization: a cited ID is "grounded" if it
    # matches any real ID after normalising Unicode.
    def _norm(s: str) -> str:
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

    def _is_grounded(cited: str, real_set: set) -> bool:
        cited_n = _norm(cited)
        if cited_n in {_norm(r) for r in real_set}:
            return True
        for real in real_set:
            real_n = _norm(real)
            if real_n.startswith(cited_n) or cited_n.startswith(real_n):
                return True
        return False

    ungrounded = {cid for cid in cited_ids if not _is_grounded(cid, real_ids)}
    # Exclude PostgreSQL-sourced citations like "neo4j::TSLA::sentiment_trends" —
    # these are the LLM citing the sentiment source, not a chunk ID.
    ungrounded = {cid for cid in ungrounded if "sentiment_trends" not in cid and "postgresql" not in cid}
    if ungrounded:
        logger.warning(
            "Citation grounding check: %d ungrounded chunk_id(s) found in LLM output "
            "(not in retrieved set of %d chunks): %s",
            len(ungrounded),
            len(real_ids),
            sorted(ungrounded),
        )
    else:
        logger.debug(
            "Citation grounding check PASSED: all %d cited IDs are grounded in retrieved chunks.",
            len(cited_ids),
        )


def validate_citations(
    output: Dict[str, Any],
    retrieval: Optional[RetrievalResult],
) -> Dict[str, Any]:
    """Return a structured citation grounding report for the final output.

    Scans all text in the output dict for neo4j:: chunk_id patterns and checks
    each against the set of chunk_ids actually returned by the retriever.

    Args:
        output:    The final assembled output dict (post-format_json_output).
        retrieval: The RetrievalResult from the pipeline (may be None).

    Returns:
        A dict with:
          total_cited        — int: number of distinct chunk IDs cited
          grounded           — int: how many match a retrieved chunk
          ungrounded         — int: how many do not
          grounding_rate_pct — float: 100.0 * grounded / total_cited (0 if none)
          ungrounded_ids     — List[str]: sorted list of hallucinated IDs
    """
    real_ids: set[str] = (
        {c.chunk_id for c in retrieval.chunks if c.chunk_id} if retrieval and retrieval.chunks else set()
    )

    output_text = json.dumps(output, ensure_ascii=False)
    # Extract cited IDs handling both old and new citation formats:
    #   Old:  [TICKER::section::hash] or [neo4j::...]
    #   New:  [source_name | TICKER::section::hash]
    _bracket_contents = re.findall(r'\[([^\]]+::[^\]]+)\]', output_text)
    cited_ids: set[str] = set()
    for raw in _bracket_contents:
        raw = raw.strip().rstrip("].,;")
        if " | " in raw:
            cid = raw.split(" | ", 1)[1].strip().rstrip("].,;")
        else:
            cid = raw
        cited_ids.add(cid)
    # Exclude source-level citations like "neo4j::TSLA::sentiment_trends"
    cited_ids = {cid for cid in cited_ids if "sentiment_trends" not in cid and "postgresql" not in cid}

    def _norm(s: str) -> str:
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

    def _is_grounded(cited: str) -> bool:
        cited_n = _norm(cited)
        if cited_n in {_norm(r) for r in real_ids}:
            return True
        for real in real_ids:
            real_n = _norm(real)
            if real_n.startswith(cited_n) or cited_n.startswith(real_n):
                return True
        return False

    ungrounded_ids = sorted(cid for cid in cited_ids if not _is_grounded(cid))
    grounded_count = len(cited_ids) - len(ungrounded_ids)
    total = len(cited_ids)
    return {
        "total_cited": total,
        "grounded": grounded_count,
        "ungrounded": len(ungrounded_ids),
        "grounding_rate_pct": 100.0 * grounded_count / total if total else 100.0,
        "ungrounded_ids": ungrounded_ids,
    }


def validate_citations_from_output(output: Dict[str, Any]) -> Dict[str, Any]:
    """Citation grounding report when the RetrievalResult is no longer available.

    Operates purely on the final JSON output: it treats all chunk_ids found in
    the output as cited, but since we have no retrieved set to check against,
    we cannot distinguish grounded from ungrounded.  This is used by the CLI
    ``--validate-citations`` flag which receives the final JSON only.

    The report still counts cited IDs and flags known-format anomalies
    (e.g. IDs that look invented because they don't follow the established
    ``neo4j::<TICKER>::<slug>`` or ``pgvec::<TICKER>::<uuid>`` patterns).

    Returns:
        Same shape as validate_citations() but with a note that grounding
        cannot be determined without access to the original RetrievalResult.
    """
    return validate_citations(output, retrieval=None)


def _strip_ungrounded_inline_citations(
    obj: Any,
    real_ids: set[str],
) -> Any:
    """Recursively scan obj and replace ungrounded inline citations in string values.

    Any ``[neo4j::...]`` token embedded in a string
    that does not prefix-match a real retrieved chunk_id is replaced with
    ``[source unavailable]``.  This is the last-resort backstop after the prompt
    instructions that tell the LLM not to invent IDs.

    Args:
        obj:      The value to sanitise (dict, list, str, or scalar).
        real_ids: Set of real chunk_ids from the retriever.

    Returns:
        A new object of the same type with ungrounded citations removed.
    """
    if not real_ids:
        return obj

    def _is_grounded(cited: str) -> bool:
        # Normalise Unicode: the LLM sometimes uses curly quotes/apostrophes
        # (e.g. \u2019 RIGHT SINGLE QUOTATION MARK) while the actual Neo4j
        # chunk_id was built from the ASCII-slugified article title (straight
        # apostrophe ' or underscore).  NFKD + ASCII encode normalises them.
        def _norm(s: str) -> str:
            return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

        cited_n = _norm(cited.rstrip("].,;"))
        if cited_n in {_norm(r) for r in real_ids}:
            return True
        for real in real_ids:
            real_n = _norm(real)
            if real_n.startswith(cited_n) or cited_n.startswith(real_n):
                return True
        return False

    def _sanitise_str(s: str) -> str:
        def _replace_match(m: re.Match) -> str:
            raw = m.group(1).strip().rstrip("].,;")
            # Preserve PostgreSQL / sentiment citations — they aren't Neo4j chunks
            if "sentiment_trends" in raw or "postgresql" in raw:
                return m.group(0)
            # New format: [source_name | chunk_id] — extract the chunk_id part
            if " | " in raw:
                cid = raw.split(" | ", 1)[1].strip().rstrip("].,;")
            else:
                cid = raw
            if _is_grounded(cid):
                return m.group(0)
            logger.debug("Stripping ungrounded inline citation from prose: %s", raw)
            return ""

        # Matches:
        #   [SomeName | TICKER::section::hash]   (new format with source_name)
        #   [TICKER::section::hash]              (old bare format)
        #   [neo4j::...], [pgvec::...], [qdrant::...]  (legacy prefixed format)
        # The inner group allows any characters except ] to catch spaces in invented IDs.
        return re.sub(r'\[([^\]]+::[^\]]+)\]', _replace_match, s)
    if isinstance(obj, str):
        # Also clean up any literal "[source unavailable]" the LLM wrote directly
        cleaned = _sanitise_str(obj)
        cleaned = re.sub(r'\[source unavailable\]', '', cleaned).strip()
        return cleaned
    if isinstance(obj, dict):
        return {k: _strip_ungrounded_inline_citations(v, real_ids) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_ungrounded_inline_citations(item, real_ids) for item in obj]
    return obj


def _node_semantic_cache_check(state: AgentState, toolkit: BusinessAnalystToolkit) -> AgentState:
    """Check the semantic cache before format_json_output.

    On a cache hit, the final ``output`` dict is populated directly from the
    cached ``RetrievalResult`` (which was stored alongside the generation
    result) and the pipeline short-circuits to END without calling the LLM
    again.

    NOTE: This node does NOT skip format_json_output — it only flags a hit so
    that format_json_output can detect it.  The actual caching of the *output*
    dict happens inside retrieve_fast / retrieve_multi_stage as a
    ``RetrievalResult`` — the semantic cache operates at the retrieval layer,
    not the generation layer.  This node is therefore a lightweight no-op
    unless a future enhancement caches the full output dict.
    """
    _print_progress("SEMANTIC CACHE CHECK", "cache operates at retrieval layer — pass-through", symbol="  --")
    # Placeholder: semantic caching at the retrieval layer is handled inside
    # toolkit.retrieve_fast() and toolkit.retrieve_multi_stage().  This node
    # is present to make the pipeline architecture explicit and to allow a
    # future extension that caches the final output dict.
    return state


def _node_format_json_output(state: AgentState) -> AgentState:
    """Assemble and validate final structured JSON output for the Supervisor."""
    ticker = state.get("ticker")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    crag_status = state.get("crag_status", CRAGStatus.INCORRECT)
    confidence = state.get("confidence", 0.0)
    fallback_triggered = state.get("fallback_triggered", False)
    sentiment = state.get("sentiment")
    retrieval: Optional[RetrievalResult] = state.get("retrieval")
    llm_output = state.get("llm_output") or {}
    web_result = state.get("web_search_result") or {}

    _print_progress(
        "FORMAT JSON OUTPUT",
        f"ticker={ticker}  crag={crag_status.value if isinstance(crag_status, CRAGStatus) else crag_status}"
        f"  confidence={confidence:.3f}  fallback={fallback_triggered}",
    )

    # Verify citations are grounded in the actual retrieved chunks
    _check_citation_grounding(llm_output, retrieval)

    # Company overview — priority order:
    # 1. Neo4j Company node (authoritative — has real financials scraped from FMP/EOD)
    # 2. LLM-generated (fallback — LLM tends to hallucinate nulls for financials)
    # 3. Extracted from graph_facts (last resort)
    # We ALWAYS prefer the Neo4j node for numeric financial fields; LLM values for
    # those fields are unreliable (often null or hallucinated).
    company_node: Optional[Dict[str, Any]] = state.get("company_node")
    company_overview = _build_company_overview_from_node(company_node) if company_node else None
    if company_overview is None:
        company_overview = llm_output.get("company_overview")
    if company_overview is None and retrieval and retrieval.graph_facts:
        company_overview = _extract_company_overview_from_graph(retrieval.graph_facts, ticker)

    # Sentiment block
    llm_sentiment = llm_output.get("sentiment") or {}
    if sentiment:
        sentiment_block = {
            "bullish_pct": float(sentiment.bullish_pct),
            "bearish_pct": float(sentiment.bearish_pct),
            "neutral_pct": float(sentiment.neutral_pct),
            "trend": sentiment.trend,
            "source": sentiment.source,
            # Carry through LLM's interpretation if it generated one
            "sentiment_interpretation": llm_sentiment.get("sentiment_interpretation"),
        }
    else:
        sentiment_block = llm_sentiment or {
            "bullish_pct": 0.0,
            "bearish_pct": 0.0,
            "neutral_pct": 0.0,
            "trend": "unknown",
            "source": "postgresql:sentiment_trends",
            "sentiment_interpretation": None,
        }

    # Fallback: LLM sometimes puts interpretation inside qualitative_analysis.sentiment_signal
    # or sentiment_verdict.rationale rather than sentiment.sentiment_interpretation.
    # NOTE: this is re-applied after coercion below to catch empty-string-coerced values.
    if not sentiment_block.get("sentiment_interpretation"):
        qa_block = llm_output.get("qualitative_analysis") or {}
        sv_block = llm_output.get("sentiment_verdict") or {}
        _interp = (
            qa_block.get("sentiment_signal")
            or sv_block.get("rationale")
            or sv_block.get("interpretation")
        )
        if _interp:
            sentiment_block = {**sentiment_block, "sentiment_interpretation": _interp}

    # If fallback was triggered, merge web search facts into key_risks / missing_context
    # (Re-read after aliasing pass below)

    # --- Field-name aliasing: the LLM sometimes uses variant key names ---
    # Map common alternatives to the canonical schema field names.
    _KEY_ALIASES: Dict[str, str] = {
        # competitive_moat variants
        "competitive_positioning": "competitive_moat",
        "moat": "competitive_moat",
        "competitive_analysis": "competitive_moat",
        # qualitative_summary variants
        "executive_summary": "qualitative_summary",
        "summary": "qualitative_summary",
        # management_guidance variants  
        "guidance": "management_guidance",
        "forward_outlook": "management_guidance",
        # key_risks variants
        "risks": "key_risks",
        "risk_factors": "key_risks",
    }
    for alias, canonical in _KEY_ALIASES.items():
        if alias in llm_output and canonical not in llm_output:
            llm_output = {**llm_output, canonical: llm_output[alias]}

    # --- Sub-field aliasing inside competitive_moat ---
    # The LLM sometimes uses strengths/weaknesses/overview instead of
    # key_strengths/vulnerabilities/narrative as defined in the schema.
    _cm = llm_output.get("competitive_moat")
    if isinstance(_cm, dict):
        _cm_updated = dict(_cm)
        if "strengths" in _cm_updated and "key_strengths" not in _cm_updated:
            _cm_updated["key_strengths"] = _cm_updated.pop("strengths")
        if "weaknesses" in _cm_updated and "vulnerabilities" not in _cm_updated:
            _cm_updated["vulnerabilities"] = _cm_updated.pop("weaknesses")
        if "overview" in _cm_updated and "narrative" not in _cm_updated:
            _cm_updated["narrative"] = _cm_updated.pop("overview")
        # Coerce empty strings → None in competitive_moat sub-fields
        _cm_updated = {
            k: (None if v == "" or v == [""] else v)
            for k, v in _cm_updated.items()
        }
        llm_output = {**llm_output, "competitive_moat": _cm_updated}

    key_risks = llm_output.get("key_risks") or []
    missing_context = llm_output.get("missing_context") or []

    # Coerce qualitative_summary to string if LLM returned a dict
    qualitative_summary_raw = llm_output.get("qualitative_summary", "")
    qualitative_summary = ""
    if isinstance(qualitative_summary_raw, str):
        qualitative_summary = qualitative_summary_raw
    elif isinstance(qualitative_summary_raw, dict):
        qualitative_summary = str(qualitative_summary_raw)
    qualitative_analysis = llm_output.get("qualitative_analysis")

    if fallback_triggered and web_result:
        web_summary = web_result.get("summary", "")
        if web_summary:
            qualitative_summary = web_summary
            # Surface web findings in qualitative_analysis if LLM didn't produce one
            if qualitative_analysis is None:
                qualitative_analysis = {
                    "narrative": web_summary,
                    "sentiment_signal": "Web search fallback used — sentiment vs. document correlation not available.",
                    "strategic_implication": None,
                    "data_quality_note": "Local vector store context was insufficient (CRAG INCORRECT). Analysis derived from web search fallback.",
                }
        web_risks = web_result.get("key_risks", [])
        key_risks = key_risks + web_risks if web_risks else key_risks
        if not missing_context:
            missing_context = [
                {
                    "gap": "Local graph context insufficient — Web Search Agent fallback used",
                    "severity": "HIGH",
                }
            ]

    # Coerce missing_context entries: LLM sometimes returns flat strings instead
    # of {"gap": ..., "severity": ...} objects.  Normalise to schema.
    missing_context = [
        item if isinstance(item, dict) else {"gap": str(item), "severity": "MEDIUM"}
        for item in missing_context
    ]

    # Coerce key_risks: LLM sometimes writes mitigation_observed as the string "null"
    # instead of JSON null.  Also ensure each entry is a dict (defensive).
    # Additionally, null out any `source` field that isn't a real retrieved chunk_id
    # to prevent hallucinated citations from leaking into the final output.
    real_chunk_ids: set[str] = (
        {c.chunk_id for c in retrieval.chunks if c.chunk_id} if retrieval else set()
    )
    normalised_risks = []
    for risk in key_risks:
        if not isinstance(risk, dict):
            continue
        if risk.get("mitigation_observed") in ("null", "none", "None", "N/A", "n/a"):
            risk = {**risk, "mitigation_observed": None}
        # Null out invented source citations (prefix-match + Unicode normalisation against real IDs)
        source = risk.get("source")
        if source and real_chunk_ids:
            def _norm_s(s: str) -> str:
                return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
            source_n = _norm_s(source)
            grounded = any(
                _norm_s(r).startswith(source_n) or source_n.startswith(_norm_s(r))
                for r in real_chunk_ids
            )
            if not grounded:
                logger.debug("Nulling ungrounded key_risk source: %s", source)
                risk = {**risk, "source": None}
        normalised_risks.append(risk)
    key_risks = normalised_risks

    # Sanitise competitive_moat.sources — plain chunk_id strings (not inline tokens)
    # that the LLM hallucinated must be removed here; _strip_ungrounded_inline_citations
    # only handles the [neo4j::...] inline-token format, not bare string arrays.
    competitive_moat = llm_output.get("competitive_moat")
    if competitive_moat and isinstance(competitive_moat, dict) and real_chunk_ids:
        raw_sources = competitive_moat.get("sources") or []
        if isinstance(raw_sources, list):
            def _norm_id(s: str) -> str:
                return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
            normed_real = {_norm_id(r) for r in real_chunk_ids}
            grounded_sources = []
            for src in raw_sources:
                if not isinstance(src, str):
                    continue
                src_n = _norm_id(src)
                if any(r.startswith(src_n) or src_n.startswith(r) for r in normed_real):
                    grounded_sources.append(src)
                else:
                    logger.debug("Removing ungrounded competitive_moat source: %s", src)
            competitive_moat = {**competitive_moat, "sources": grounded_sources}

    # Repair null/missing competitive_moat.rating when the LLM obeyed the "use null"
    # hard constraint but we still have a valid competitive_moat block.
    # Strategy: (1) scan qualitative_summary for explicit moat-rating keywords,
    # (2) scan qualitative_analysis.narrative, (3) default to "unknown".
    if isinstance(competitive_moat, dict) and not competitive_moat.get("rating"):
        _rating_inferred: Optional[str] = None
        _scan_texts = [
            llm_output.get("qualitative_summary") or "",
            (llm_output.get("qualitative_analysis") or {}).get("narrative") or ""
            if isinstance(llm_output.get("qualitative_analysis"), dict) else "",
        ]
        for _scan_text in _scan_texts:
            if not isinstance(_scan_text, str):
                continue
            _t = _scan_text.lower()
            if "wide moat" in _t or "wide competitive moat" in _t or "rating: wide" in _t or '"wide"' in _t:
                _rating_inferred = "wide"
                break
            if "narrow moat" in _t or "narrow competitive moat" in _t or "rating: narrow" in _t or '"narrow"' in _t:
                _rating_inferred = "narrow"
                break
            if "no moat" in _t or "rating: none" in _t or '"none"' in _t or "moat: none" in _t:
                _rating_inferred = "none"
                break
        if _rating_inferred:
            logger.info(
                "[FormatJSON] Inferred competitive_moat.rating=%r from prose text for ticker=%s",
                _rating_inferred, ticker,
            )
        else:
            _rating_inferred = "unknown"
            logger.info(
                "[FormatJSON] competitive_moat.rating was null/empty for ticker=%s — defaulting to 'unknown'",
                ticker,
            )
        competitive_moat = {**competitive_moat, "rating": _rating_inferred}

    # Fallback: if LLM put moat analysis in qualitative_summary instead of competitive_moat
    # Use llm_output directly so the guard isn't blocked by the INSUFFICIENT_DATA default
    # that may have already been assigned to the local `qualitative_summary` variable.
    _raw_qs = llm_output.get("qualitative_summary") or ""
    if not competitive_moat and _raw_qs and not _raw_qs.startswith("INSUFFICIENT_DATA"):
        competitive_moat = {"narrative": qualitative_summary, "sources": [], "rating": "unknown"}

    # Fallback: when competitive_moat is still null, derive a minimal entry from
    # qualitative_analysis.narrative — the LLM often embeds moat analysis there.
    if not competitive_moat:
        _qa_for_cm = llm_output.get("qualitative_analysis") or {}
        _narrative_for_cm = _qa_for_cm.get("narrative") if isinstance(_qa_for_cm, dict) else None
        if isinstance(_narrative_for_cm, str) and _narrative_for_cm.strip():
            competitive_moat = {
                "narrative": _narrative_for_cm,
                "sources": [],
                "rating": "unknown",
                "data_quality_note": "Derived from qualitative_analysis.narrative — LLM did not populate competitive_moat directly.",
            }

    if not llm_output.get("qualitative_analysis") and qualitative_summary and not qualitative_summary.startswith("INSUFFICIENT_DATA"):
        qualitative_analysis = {"narrative": qualitative_summary, "sentiment_signal": None, "strategic_implication": None, "data_quality_note": "Derived from qualitative_summary fallback"}

    # Additional fallback: derive qualitative_summary from qualitative_analysis.narrative when missing.
    # The LLM sometimes populates qualitative_analysis richly but omits qualitative_summary.
    if not qualitative_summary or qualitative_summary.startswith("INSUFFICIENT_DATA"):
        qa = llm_output.get("qualitative_analysis") or qualitative_analysis
        if isinstance(qa, dict):
            narrative = qa.get("narrative")
            if isinstance(narrative, str) and narrative.strip():
                qualitative_summary = narrative

    # Additional fallback: derive qualitative_summary from management_guidance when missing.
    # The LLM sometimes populates management_guidance richly but leaves qualitative_summary null
    # (common on SIMPLE/NUMERICAL paths where the task is narrow).
    if not qualitative_summary or qualitative_summary.startswith("INSUFFICIENT_DATA"):
        mgmt = llm_output.get("management_guidance")
        if mgmt:
            if isinstance(mgmt, dict):
                # Try known summary fields first, then any string value in the dict
                _SUMMARY_KEYS = (
                    "forward_outlook_summary", "most_recent_guidance",
                    "overview", "summary", "narrative",
                )
                for _key in _SUMMARY_KEYS:
                    val = mgmt.get(_key)
                    if isinstance(val, list) and val:
                        qualitative_summary = " ".join(str(s) for s in val[:2])
                        break
                    elif isinstance(val, str) and val:
                        qualitative_summary = val
                        break
                # Last resort: grab the first non-empty string value from any key
                if not qualitative_summary or qualitative_summary.startswith("INSUFFICIENT_DATA"):
                    for val in mgmt.values():
                        if isinstance(val, str) and val.strip():
                            qualitative_summary = val
                            break
                        if isinstance(val, list) and val:
                            first = next((s for s in val if isinstance(s, str) and s.strip()), None)
                            if first:
                                qualitative_summary = first
                                break
            elif isinstance(mgmt, str) and mgmt:
                qualitative_summary = mgmt

    # Coerce empty strings → None inside management_guidance (LLM often emits "" for unknown fields)
    _mgmt_raw = llm_output.get("management_guidance")
    if isinstance(_mgmt_raw, dict):
        _mgmt_raw = {k: (v if v != "" else None) for k, v in _mgmt_raw.items()}
        llm_output = {**llm_output, "management_guidance": _mgmt_raw}

    # Coerce empty strings → None inside qualitative_analysis sub-fields
    _qa_raw = llm_output.get("qualitative_analysis")
    if isinstance(_qa_raw, dict):
        _qa_raw = {k: (None if v == "" else v) for k, v in _qa_raw.items()}
        llm_output = {**llm_output, "qualitative_analysis": _qa_raw}

    # Fallback: when key_risks is empty but we have missing_context entries, promote
    # HIGH-severity missing_context items to synthetic key_risks so the field is not empty.
    # Also try to extract risks from qualitative_analysis.narrative.
    if not key_risks:
        # Promote HIGH missing_context as synthetic risks
        for gap_entry in missing_context:
            if isinstance(gap_entry, dict) and gap_entry.get("severity") == "HIGH":
                gap_text = gap_entry.get("gap", "")
                if gap_text:
                    key_risks.append({
                        "risk": gap_text,
                        "severity": "HIGH",
                        "mitigation_observed": None,
                        "source": None,
                        "data_quality_note": "Derived from missing_context — LLM did not populate key_risks directly.",
                    })
        # If still empty, create a placeholder from qualitative_analysis.narrative
        if not key_risks:
            _qa_narr = (llm_output.get("qualitative_analysis") or {}).get("narrative") if isinstance(llm_output.get("qualitative_analysis"), dict) else None
            if isinstance(_qa_narr, str) and _qa_narr.strip():
                key_risks.append({
                    "risk": "See qualitative_analysis.narrative for detailed risk discussion.",
                    "severity": "MEDIUM",
                    "mitigation_observed": None,
                    "source": None,
                    "data_quality_note": "LLM embedded risk analysis in qualitative_analysis.narrative rather than key_risks.",
                })

    # Second-pass sentiment_interpretation fallback — now that empty strings are coerced,
    # re-check qualitative_analysis.sentiment_signal and sentiment_verdict.rationale.
    if not sentiment_block.get("sentiment_interpretation"):
        _qa2 = llm_output.get("qualitative_analysis") or {}
        _sv2 = llm_output.get("sentiment_verdict") or {}
        _interp2 = (
            _qa2.get("sentiment_signal")
            or _sv2.get("rationale")
            or _sv2.get("interpretation")
        )
        if _interp2:
            sentiment_block = {**sentiment_block, "sentiment_interpretation": _interp2}

    output: Dict[str, Any] = {
        "agent": "business_analyst",
        "ticker": ticker,
        "query_date": today,
        "company_overview": company_overview,
        "sentiment": sentiment_block,
        "competitive_moat": competitive_moat,
        "qualitative_analysis": llm_output.get("qualitative_analysis"),
        "management_guidance": llm_output.get("management_guidance"),
        "sentiment_verdict": llm_output.get("sentiment_verdict"),
        "key_risks": key_risks,
        "missing_context": missing_context,
        "crag_status": crag_status.value if isinstance(crag_status, CRAGStatus) else str(crag_status),
        "confidence": round(confidence, 4),
        "fallback_triggered": fallback_triggered,
        "qualitative_summary": qualitative_summary or "INSUFFICIENT_DATA: No analysis generated.",
        # Pre-check layer diagnostics (None when data is healthy)
        "data_coverage_warning": state.get("data_coverage_warning"),
        # Thinking trace — pipeline steps for UI display
        "thinking_trace": list(_THINKING_TRACE),
    }

    # Option B backstop: strip any remaining ungrounded inline citations from all
    # prose string fields (narrative, risk, strategic_implication, etc.).
    # The prompt (Option A) should prevent them, but this guarantees clean output.
    if real_chunk_ids:
        logger.debug(
            "Running inline citation post-processor with %d real_ids", len(real_chunk_ids)
        )
        output = _strip_ungrounded_inline_citations(output, real_chunk_ids)  # type: ignore[assignment]
    else:
        logger.debug("Skipping inline citation post-processor: real_chunk_ids is empty")

    # Verbose: citation grounding summary — also surfaced in output for downstream consumers
    citation_report = validate_citations(output, retrieval)
    output["citation_report"] = {
        "total_cited": citation_report["total_cited"],
        "grounded": citation_report["grounded"],
        "ungrounded": citation_report["ungrounded"],
        "grounding_rate_pct": round(citation_report["grounding_rate_pct"], 1),
        "ungrounded_ids": citation_report.get("ungrounded_ids", []),
    }
    _print_progress(
        "CITATION GROUNDING",
        f"cited={citation_report['total_cited']}  grounded={citation_report['grounded']}  "
        f"ungrounded={citation_report['ungrounded']}  rate={citation_report['grounding_rate_pct']:.1f}%"
        + (f"  hallucinated={citation_report['ungrounded_ids']}" if citation_report["ungrounded_ids"] else ""),
        symbol="  OK" if citation_report["ungrounded"] == 0 else " !?",
    )
    _print_progress("FORMAT JSON OUTPUT done", "", symbol="  OK")

    return {**state, "output": output}


def _build_company_overview_from_node(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build company_overview from a raw Neo4j Company node property dict.

    The FMP-scraped Company node stores financials under both CamelCase keys
    (from EOD Historical Data) and snake_case keys (from FMP). We check both.
    """
    name = (
        node.get("companyName")
        or node.get("Name")
        or node.get("name")
    )
    if not name:
        return None

    def _float(val: Any) -> Optional[float]:
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    market_cap = _float(
        node.get("marketCap") or node.get("Highlights_MarketCapitalization")
    )
    pe_ratio = _float(
        node.get("Valuation_TrailingPE") or node.get("Highlights_PERatio")
    )
    profit_margin = _float(
        node.get("Highlights_ProfitMargin")
    )
    sector = node.get("sector") or node.get("Sector") or node.get("GicSector")
    industry = node.get("industry") or node.get("Industry") or node.get("GicIndustry")

    return {
        "name": name,
        "sector": sector,
        "industry": industry,
        "market_cap": market_cap,
        "pe_ratio": pe_ratio,
        "profit_margin": profit_margin,
    }


def _extract_company_overview_from_graph(
    graph_facts: List[Dict[str, Any]], ticker: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of company overview from graph facts."""
    for fact in graph_facts:
        node = fact.get("node") or {}
        name = node.get("name")
        # Accept any node that has a name and at least one additional non-null property
        if name and any(v is not None for k, v in node.items() if k != "name"):
            return {
                "name": name,
                "sector": node.get("sector"),
                "market_cap": node.get("marketCap") or node.get("market_cap"),
                "pe_ratio": node.get("pe_ratio") or node.get("peRatio"),
                "profit_margin": node.get("profit_margin") or node.get("profitMargin"),
            }
    return None


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_after_sentiment(state: AgentState) -> str:
    """Route after fetch_sentiment_data.

    Short-circuit: if precheck_data_coverage pre-set crag_status=CORRECT
    (e.g. fresh sentiment data for a sentiment-focused query), jump straight
    to generate_analysis to avoid running retrieval on pure sentiment queries.
    Otherwise proceed to complex_retrieval.
    """
    if state.get("crag_status") == CRAGStatus.CORRECT:
        logger.info(
            "[Router] crag_status=CORRECT pre-set by precheck — skipping retrieval, "
            "routing directly to generate_analysis."
        )
        return "generate_analysis"
    return "complex_retrieval"


def _route_after_generation(state: AgentState) -> str:
    """Route after generate_analysis.

    If the LLM declared INSUFFICIENT_DATA, trigger the web search fallback.
    Otherwise proceed to semantic cache check for normal output formatting.
    """
    llm_output = state.get("llm_output") or {}
    summary = llm_output.get("qualitative_summary", "") or ""
    if summary.startswith("INSUFFICIENT_DATA"):
        logger.warning("LLM returned INSUFFICIENT_DATA — triggering web fallback.")
        return "web_search_fallback"
    return "semantic_cache_check"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    toolkit: BusinessAnalystToolkit,
    llm: LLMClient,
) -> Any:  # CompiledStateGraph — typed as Any to avoid stub incompatibilities
    """Assemble the Optimized Adaptive Agentic Graph RAG pipeline and return a compiled graph."""

    graph = StateGraph(AgentState)

    # --- Phase 1: Metadata pre-check + data-coverage pre-check + sentiment enrichment ---
    graph.add_node(
        "metadata_precheck",
        lambda state: _node_metadata_precheck(cast(AgentState, state), toolkit),
    )
    graph.add_node(
        "precheck_data_coverage",
        lambda state: _node_precheck_data_coverage(cast(AgentState, state)),
    )
    graph.add_node(
        "fetch_sentiment_data",
        lambda state: _node_fetch_sentiment(cast(AgentState, state), toolkit),
    )

    # --- Phase 2: Unified retrieval (always multi-stage) ---
    graph.add_node(
        "complex_retrieval",
        lambda state: _node_complex_retrieval(cast(AgentState, state), toolkit),
    )

    # --- Phase 3: Generation ---
    graph.add_node(
        "generate_analysis",
        lambda state: _node_generate_analysis(cast(AgentState, state), llm),
    )
    graph.add_node(
        "web_search_fallback",
        lambda state: _node_web_search_fallback(cast(AgentState, state), toolkit),
    )

    # --- Phase 4: Semantic cache check + output formatting ---
    graph.add_node(
        "semantic_cache_check",
        lambda state: _node_semantic_cache_check(cast(AgentState, state), toolkit),
    )
    graph.add_node(
        "format_json_output",
        _node_format_json_output,
    )

    # Entry point
    graph.set_entry_point("metadata_precheck")

    # Phase 1: linear pre-check chain
    graph.add_edge("metadata_precheck", "precheck_data_coverage")
    graph.add_edge("precheck_data_coverage", "fetch_sentiment_data")

    # After sentiment: short-circuit to generate_analysis if precheck set CORRECT,
    # otherwise always go through complex_retrieval.
    graph.add_conditional_edges(
        "fetch_sentiment_data",
        lambda state: _route_after_sentiment(cast(AgentState, state)),
        {
            "generate_analysis": "generate_analysis",
            "complex_retrieval": "complex_retrieval",
        },
    )

    # Retrieval always feeds directly into generation (no CRAG gating)
    graph.add_edge("complex_retrieval", "generate_analysis")

    # After generation: if LLM declares INSUFFICIENT_DATA → web fallback,
    # otherwise proceed to semantic cache check.
    graph.add_conditional_edges(
        "generate_analysis",
        lambda state: _route_after_generation(cast(AgentState, state)),
        {
            "web_search_fallback": "web_search_fallback",
            "semantic_cache_check": "semantic_cache_check",
        },
    )

    # Web fallback rejoins the main path
    graph.add_edge("web_search_fallback", "semantic_cache_check")
    graph.add_edge("semantic_cache_check", "format_json_output")

    # Terminal
    graph.add_edge("format_json_output", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public run function
# ---------------------------------------------------------------------------

def run(
    task: str,
    ticker: Optional[str] = None,
    config: Optional[BusinessAnalystConfig] = None,
    availability_profile: Optional[Dict[str, bool]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the Business Analyst CRAG pipeline and return a structured JSON dict.

    Args:
        task:                 Natural language query (e.g. "What is Apple's competitive moat?").
        ticker:               Optional ticker symbol (e.g. "AAPL").
        config:               Optional config override; defaults to env-var-based config.
        availability_profile: Optional per-ticker data profile from data_availability module.
                              When provided, used to skip dead code paths (e.g. empty vector stores).
        verbose:              When True, print node-by-node pipeline progress to stderr.

    Returns:
        Structured JSON dict conforming to the output schema in README.md.
    """
    global _VERBOSE
    _VERBOSE = verbose
    _reset_step_counter()

    cfg = config or load_config()
    toolkit = BusinessAnalystToolkit(cfg)
    llm = LLMClient(cfg)

    compiled = build_graph(toolkit, llm)

    initial_state: AgentState = {
        "task": task,
        "ticker": ticker,
        "fallback_triggered": False,
        "availability_profile": availability_profile,
    }

    try:
        final_state = compiled.invoke(initial_state)
    finally:
        toolkit.close()

    return final_state.get("output") or {}


def run_full_analysis(
    ticker: str,
    config: Optional[BusinessAnalystConfig] = None,
    availability_profile: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """Run a comprehensive, all-pillars analysis of a company for the Synthesizer.

    Unlike ``run()``, which answers a single analyst question, this function
    issues a broad task that instructs the LLM to populate *every* output
    section (competitive moat, business model, risk factors, strategy,
    sentiment) in one pass.  The result is a complete company dossier that the
    Synthesizer can consume directly for fundamental analysis without needing
    to call the agent multiple times.

    The output schema is identical to ``run()`` — a structured JSON dict
    conforming to the schema in README.md.  The Synthesizer should treat this
    as the canonical qualitative intelligence package for the ticker.

    Args:
        ticker:               Ticker symbol (e.g. ``"AAPL"``).  Required.
        config:               Optional config override; defaults to env-var-based config.
        availability_profile: Optional per-ticker data profile from data_availability module.

    Returns:
        Structured JSON dict with all analysis sections populated.
        ``company_overview``, ``sentiment``, ``competitive_moat``,
        ``qualitative_analysis``, ``key_risks``, ``missing_context``,
        ``crag_status``, ``confidence``, ``fallback_triggered``,
        ``qualitative_summary``.

    Example (Synthesizer usage)::

        from agents.business_analyst.agent import run_full_analysis

        dossier = run_full_analysis(ticker="AAPL")
        # dossier["competitive_moat"]["rating"]   → "wide"
        # dossier["key_risks"]                    → [{risk, severity, source}, ...]
        # dossier["qualitative_summary"]           → "1-2 sentence executive summary"
    """
    if not ticker:
        raise ValueError("ticker is required for run_full_analysis()")

    comprehensive_task = (
        f"Provide a deeply comprehensive, institutional buy-side-grade qualitative analysis of {ticker} covering ALL "
        f"of the following pillars. Each pillar has a MINIMUM depth requirement — shallow, generic responses are a "
        f"critical quality failure. Synthesise across ALL retrieved chunks rather than paraphrasing individual sources. "
        f"Cite a verbatim chunk_id for every factual claim. Do not make buy/sell/hold judgements or price targets. "
        f""
        f"PILLAR 1 — COMPETITIVE MOAT DEPTH & TRAJECTORY (minimum 8 sentences): "
        f"Rate the moat as wide/narrow/none and justify the rating with at least 3 distinct pieces of evidence. "
        f"Identify the PRIMARY moat source (network effects, switching costs, cost advantage, intangible assets) "
        f"and quantify its economic magnitude where evidence allows — e.g. enterprise retention rates, take rates, "
        f"customer lifetime value signals, or market share dominance figures. "
        f"Identify up to 3 secondary moat sources and explain the mechanism of each. "
        f"Name the single most credible competitive threat and explain in precise terms the specific mechanism "
        f"by which it could erode the primary moat — which revenue stream is at risk, what the substitution mechanism is, "
        f"and what fraction of the moat is at stake. "
        f"Identify up to 2 additional competitive threats with their specific erosion mechanisms. "
        f"Assess whether the moat is widening, stable, or narrowing — cite the specific retrieved signals supporting "
        f"this trajectory assessment and explain the dynamics driving it. "
        f"State what the moat rating implies for long-term pricing power and the sustainability of current margins. "
        f""
        f"PILLAR 2 — BUSINESS MODEL, REVENUE QUALITY & CAPITAL ALLOCATION (minimum 8 sentences): "
        f"Provide a detailed breakdown of the primary revenue streams by segment or product category, "
        f"naming each segment and its relative contribution where evidence allows. "
        f"Identify the highest-margin segment and explain what drives its superior economics. "
        f"Assess revenue quality: what proportion is recurring/contractual vs. transactional/cyclical? "
        f"What does the recurring revenue ratio imply for earnings predictability and downside protection? "
        f"Evaluate pricing power rigorously — cite specific evidence from retrieved chunks of whether "
        f"{ticker} can raise prices without meaningful volume loss, and in which segments this is strongest vs. weakest. "
        f"Evaluate capital allocation quality: trace how management is deploying FCF across buybacks, R&D, "
        f"M&A, and dividends. Is the capital allocation strategy generating returns above cost of capital? "
        f"Assess customer concentration or diversification — is there meaningful customer or geographic revenue risk? "
        f"State what the aggregate business model assessment implies for earnings durability under adverse conditions. "
        f""
        f"PILLAR 3 — MARGIN TRAJECTORY & OPERATING LEVERAGE (minimum 6 sentences): "
        f"State the direction of gross margins and EBIT margins — expanding, stable, or contracting — and ground "
        f"this in specific evidence from the retrieved chunks, not in quant data. "
        f"Identify the primary driver of each margin trend separately: is gross margin movement driven by product "
        f"mix shift, pricing changes, input cost dynamics, or scale? Is EBIT margin movement driven by revenue "
        f"leverage, operating expense growth, or R&D investment cycles? "
        f"Comment on the relationship between revenue growth and margin expansion — is there evidence of positive "
        f"operating leverage (margins rising faster than revenue) or operating leverage erosion? "
        f"Assess FCF quality: is FCF conversion above 1.0 (cash earnings exceed reported income) or below 0.7 "
        f"(earnings quality concern from elevated non-cash items)? Cite any evidence from retrieved context. "
        f"State what the margin trajectory implies for the long-term earnings power and the compound annual "
        f"return potential of the business if current trends persist. "
        f""
        f"PILLAR 4 — STRATEGIC POSITIONING & 2-3 YEAR INVESTMENT IMPLICATION (minimum 6 sentences): "
        f"State the single most important strategic bet {ticker} is making over the next 2-3 years — "
        f"the initiative that will define whether the investment thesis succeeds or fails. "
        f"Explain the mechanism of the bull case: the specific causal chain from strategic execution to "
        f"financial outcome — which revenue lines grow, by how much, and why this particular company is "
        f"positioned to capture the opportunity over competitors. "
        f"Explain the mechanism of the bear case: the specific failure mode — what would have to go wrong, "
        f"what competitor would need to succeed, or what structural change would need to occur — and state "
        f"the financial consequence (e.g. revenue deceleration, margin compression, multiple contraction). "
        f"Identify 1-2 secondary strategic bets and assess their probability-weighted contribution to the thesis. "
        f"State the single leading indicator — the metric or event in the next 1-2 quarters that would provide "
        f"the clearest signal about whether the primary strategic bet is on or off track. "
        f"Cite chunk_ids for every strategic assertion. "
        f""
        f"PILLAR 5 — KEY RISKS WITH PRECISE FINANCIAL MAGNITUDE (minimum 2 risks per category): "
        f"For EACH risk, provide ALL THREE of: (a) the specific mechanism — the exact causal chain from "
        f"trigger event to financial impact, naming the revenue line or cost structure affected; "
        f"(b) the financial magnitude — express in dollars, basis points, or percentage of revenue/margin "
        f"(e.g. 'a 5% tariff on Chinese-assembled products would reduce EBIT margins by ~80-120bps on the "
        f"~35% of revenue exposed to that geography'); (c) any observed mitigation from management. "
        f"Cover at minimum: TWO competitive/market share risks; TWO regulatory/legal risks; "
        f"ONE macro/cycle risk; ONE operational/execution risk. "
        f"Rate each risk HIGH/MEDIUM/LOW and explain the rating in terms of probability and magnitude. "
        f"Cite a verbatim chunk_id for each risk's primary evidence source. "
        f""
        f"PILLAR 6 — SENTIMENT VERDICT & DOCUMENT SYNTHESIS (minimum 6 sentences): "
        f"State the precise bullish/bearish/neutral percentage split from the sentiment data. "
        f"Compare explicitly to the ~55% large-cap tech base rate — is {ticker}'s reading "
        f"above, at, or below by how many percentage points, and what does this deviation imply? "
        f"Explain in detail what the bullish cohort is specifically betting on — name the thesis, "
        f"the mechanism, and the outcome they expect. "
        f"Explain in detail what the bearish cohort fears — name the precise risk mechanism and outcome. "
        f"Identify whether the documentary evidence in the retrieved chunks more closely validates "
        f"the bullish thesis or the bearish thesis, and name the specific piece of evidence that is "
        f"most decisive in this assessment. "
        f"Identify any tensions where sentiment and documents diverge, and explain which is likely to "
        f"prevail and why. "
        f"Assign CONSTRUCTIVE, CAUTIOUS, NEUTRAL, or DETERIORATING with a rationale that explicitly "
        f"references both the sentiment numbers and the documentary evidence. "
        f""
        f"PILLAR 7 — MANAGEMENT GUIDANCE, EARNINGS CALL HIGHLIGHTS & NEAR-TERM CATALYSTS (minimum 8 sentences): "
        f"Search ALL retrieved document chunks thoroughly for any forward-looking statements, management "
        f"commentary, earnings call excerpts, or analyst summaries of guidance — these may appear in news "
        f"articles, press releases, investor letters, or analyst write-ups. "
        f"If quantitative guidance figures are present (revenue range, EPS range, margin targets, capex plans), "
        f"quote or paraphrase them precisely and cite the verbatim chunk_id. "
        f"If no explicit numeric guidance is found, state that clearly and paraphrase the most recent "
        f"qualitative management commentary — what tone did management strike (confident, cautious, hedging)? "
        f"What specific segments or initiatives did management flag as priorities? "
        f"Identify the 3-4 most important near-term catalysts that could cause a material re-rating "
        f"(both positive and negative), with: the specific event name, the timeline, the mechanism by "
        f"which it would affect the stock, and the magnitude of impact if it materialises. "
        f"Assess management credibility: is there evidence in the retrieved chunks of management consistently "
        f"meeting guidance, revising guidance, or surprising positively/negatively? "
        f"Cite verbatim chunk_ids for every guidance figure or management statement. "
    )

    return run(task=comprehensive_task, ticker=ticker, config=config, availability_profile=availability_profile)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="business_analyst",
        description="Run the Business Analyst CRAG pipeline for a given ticker.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker symbol (e.g. AAPL). Optional for cross-company queries.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help=(
            "Research question to answer. "
            "Defaults to 'Analyse the competitive moat, business model, and key risks for {ticker}'."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: true).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print node-by-node pipeline progress to stderr.",
    )
    parser.add_argument(
        "--validate-citations",
        action="store_true",
        default=False,
        help=(
            "After the run, print a citation grounding report to stderr: "
            "total cited IDs, grounded count, ungrounded count, grounding rate %%, "
            "and a list of any hallucinated chunk_ids."
        ),
    )
    return parser


def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Suppress Neo4j driver notifications for missing properties/relationships.
    # These are expected while Neo4j only has Company nodes (no Chunk/Risk/Strategy
    # nodes yet — those are populated by the FMP DAG on its first run).
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    # Suppress HuggingFace unauthenticated request warnings (no HF_TOKEN set)
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*unauthenticated.*HF Hub.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*HF_TOKEN.*", category=UserWarning)

    # Suppress sentence_transformers load report noise (UNEXPECTED key is harmless)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

    ticker = args.ticker
    task = args.task or (
        f"Analyse the competitive moat, business model, and key risks for {ticker}"
        if ticker
        else "Provide a general market analysis."
    )

    try:
        result = run(task=task, ticker=ticker, verbose=args.verbose)
        if args.validate_citations:
            # Re-run citation validation on the final output and print a report.
            # We must reconstruct the retrieval from the result (it is not returned
            # by run()), so we call validate_citations_from_output() which operates
            # on the final JSON only (counts cited IDs vs. chunk_ids in the output).
            report = validate_citations_from_output(result)
            print("--- Citation Grounding Report ---", file=sys.stderr)
            print(f"  Total cited IDs  : {report['total_cited']}", file=sys.stderr)
            print(f"  Grounded         : {report['grounded']}", file=sys.stderr)
            print(f"  Ungrounded       : {report['ungrounded']}", file=sys.stderr)
            print(f"  Grounding rate   : {report['grounding_rate_pct']:.1f}%", file=sys.stderr)
            if report["ungrounded_ids"]:
                print(f"  Hallucinated IDs : {report['ungrounded_ids']}", file=sys.stderr)
            print("---------------------------------", file=sys.stderr)
        indent = 2 if args.pretty else None
        print(json.dumps(result, indent=indent, default=str))
    except Exception as exc:
        logger.error("Agent run failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
