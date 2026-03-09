"""Business Analyst Agent — Optimized Adaptive Agentic Graph RAG pipeline.

Architecture:

    Query + Ticker
        │
        ▼
    metadata_precheck         ←  Neo4j chunk counts, pgvector status, sentiment availability
        │
        ▼
    classify_query            ←  LLM (lightweight): SIMPLE / NUMERICAL / COMPLEX
        │
        ├─ SIMPLE     → fast_path_retrieval  (vector + BM25, small top_k) → generate_analysis
        ├─ NUMERICAL  → numerical_path       (Cypher metric extraction + sentiment fetch) → generate_analysis
        └─ COMPLEX    → complex_retrieval    (multi-stage bi+cross-encoder + RRF + graph) → crag_evaluate
                                                 │
                                                 ▼
    crag_evaluate             ←  CORRECT (>0.6) / AMBIGUOUS (0.4-0.6) / INCORRECT (<0.4)
        │
        ├─ CORRECT    → generate_analysis
        ├─ AMBIGUOUS  → rewrite_query → retry complex_retrieval (max 2 loops)
        └─ INCORRECT  → web_search_fallback → generate_analysis
        │
        ▼ (All paths converge)
    semantic_cache_check      ←  Hit? Return cached → Miss? Proceed + cache result
        │
        ▼
    format_json_output        ←  Structured JSON for Supervisor
        │
       END → return to Supervisor

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
from typing import Any, Dict, List, Optional, TypedDict, cast

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


def _print_progress(step: str, detail: str = "", *, symbol: str = ">>") -> None:
    """Print a pipeline step to stderr when verbose mode is active.

    Output goes to stderr so it does not pollute the JSON stdout output.
    Format:  [step N] >> STEP_NAME  detail
    """
    if not _VERBOSE:
        return
    global _STEP_COUNTER
    _STEP_COUNTER += 1
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{_STEP_COUNTER:02d}] {symbol} {step}"
    if detail:
        line += f"  |  {detail}"
    print(line, file=sys.stderr, flush=True)


def _reset_step_counter() -> None:
    global _STEP_COUNTER
    _STEP_COUNTER = 0



class AgentState(TypedDict, total=False):
    """Shared mutable state threaded through every node."""

    # Inputs
    task: str
    ticker: Optional[str]

    # Availability profile (optional, injected by orchestration planner)
    availability_profile: Optional[Dict[str, bool]]

    # Query classification — SIMPLE / NUMERICAL / COMPLEX
    query_class: Optional[str]

    # Metadata pre-check profile (Neo4j counts, pgvector status, sentiment flag)
    metadata_profile: Optional[MetadataProfile]

    # Enrichment
    sentiment: Optional[SentimentSnapshot]
    company_node: Optional[Dict[str, Any]]   # raw Neo4j Company node properties
    community_summary: Optional[str]         # graph-community summary (2A: Graph RAG)
    retrieval: Optional[RetrievalResult]
    rewrite_count: int          # guard: max config.max_rewrite_loops rewrites

    # CRAG evaluation
    crag_status: Optional[CRAGStatus]
    confidence: float

    # Generation
    llm_output: Optional[Dict[str, Any]]
    fallback_triggered: bool
    web_search_result: Optional[Dict[str, Any]]

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
    return {**state, "metadata_profile": profile}


def _node_classify_query(
    state: AgentState,
    toolkit: BusinessAnalystToolkit,
    llm: LLMClient,
) -> AgentState:
    """Classify the query as SIMPLE, NUMERICAL, or COMPLEX using a lightweight LLM.

    - SIMPLE   → fast_path_retrieval (vector + BM25, small top_k, <3 s target)
    - NUMERICAL → numerical_path (Cypher time-series + sentiment metrics)
    - COMPLEX  → complex_retrieval (multi-stage bi+cross-encoder + RRF + graph)

    Falls back to COMPLEX on any error so the full pipeline is always available.
    """
    query = state.get("task", "")
    _print_progress("CLASSIFY QUERY", f"query={query[:80]!r}")
    t0 = time.monotonic()
    query_class = llm.classify_query(query)
    elapsed = time.monotonic() - t0
    logger.info("[ClassifyQuery] query=%r → class=%s", query[:80], query_class)
    _print_progress(
        "CLASSIFY QUERY done",
        f"class={query_class}  ({elapsed:.2f}s)",
        symbol="  OK",
    )
    return {**state, "query_class": query_class}


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


def _node_fast_path_retrieval(state: AgentState, toolkit: BusinessAnalystToolkit) -> AgentState:
    """Fast-path retrieval for SIMPLE queries.

    Uses a smaller top_k budget (``config.fast_path_top_k``) and skips
    graph traversal and cross-encoder reranking to minimise latency.
    Target: <3 s wall-clock on warm cache.

    Pre-sets crag_status=CORRECT and confidence=1.0 because SIMPLE queries
    bypass CRAG evaluation entirely — the retrieved context is used directly.
    """
    query = state.get("task", "")
    ticker = state.get("ticker")
    if ticker and not query.upper().startswith(ticker.upper()):
        query = f"{ticker.upper()}: {query}"
    _print_progress("FAST PATH RETRIEVAL", f"ticker={ticker}")
    t0 = time.monotonic()
    retrieval = toolkit.retrieve_fast(query, ticker)
    elapsed = time.monotonic() - t0
    n_chunks = len(retrieval.chunks) if retrieval else 0
    top_score = retrieval.chunks[0].score if (retrieval and retrieval.chunks) else 0.0
    _print_progress(
        "FAST PATH RETRIEVAL done",
        f"chunks={n_chunks}  top_score={top_score:.3f}  ({elapsed:.2f}s)",
        symbol="  OK",
    )
    return {
        **state,
        "retrieval": retrieval,
        "crag_status": CRAGStatus.CORRECT,
        "confidence": 1.0,
    }


def _node_numerical_path(state: AgentState, toolkit: BusinessAnalystToolkit) -> AgentState:
    """Numerical-path retrieval for NUMERICAL queries.

    Uses the fast-path retriever (sufficient for metric lookups) and
    additionally ensures sentiment data is fetched when not already present.

    Pre-sets crag_status=CORRECT and confidence=1.0 because NUMERICAL queries
    bypass CRAG evaluation entirely.
    """
    query = state.get("task", "")
    ticker = state.get("ticker")
    if ticker and not query.upper().startswith(ticker.upper()):
        query = f"{ticker.upper()}: {query}"
    _print_progress("NUMERICAL PATH RETRIEVAL", f"ticker={ticker}")
    t0 = time.monotonic()
    retrieval = toolkit.retrieve_fast(query, ticker)
    elapsed = time.monotonic() - t0
    n_chunks = len(retrieval.chunks) if retrieval else 0
    _print_progress(
        "NUMERICAL PATH RETRIEVAL done",
        f"chunks={n_chunks}  ({elapsed:.2f}s)",
        symbol="  OK",
    )
    # Ensure sentiment is available for numerical context
    if state.get("sentiment") is None and ticker:
        sentiment = toolkit.fetch_sentiment(ticker)
        return {
            **state,
            "retrieval": retrieval,
            "sentiment": sentiment,
            "crag_status": CRAGStatus.CORRECT,
            "confidence": 1.0,
        }
    return {
        **state,
        "retrieval": retrieval,
        "crag_status": CRAGStatus.CORRECT,
        "confidence": 1.0,
    }


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


def _node_crag_evaluate(state: AgentState, toolkit: BusinessAnalystToolkit) -> AgentState:
    """Score top retrieved chunks and classify as CORRECT / AMBIGUOUS / INCORRECT."""
    retrieval: Optional[RetrievalResult] = state.get("retrieval")
    chunks = retrieval.chunks if retrieval else []
    ticker = state.get("ticker")
    _print_progress("CRAG EVALUATE", f"ticker={ticker}  chunks_to_score={len(chunks)}")
    t0 = time.monotonic()
    evaluation = toolkit.evaluate(chunks, ticker=ticker)
    elapsed = time.monotonic() - t0
    _print_progress(
        "CRAG EVALUATE done",
        f"status={evaluation.status.value}  confidence={evaluation.confidence:.3f}  ({elapsed:.2f}s)",
        symbol="  OK",
    )
    return {
        **state,
        "crag_status": evaluation.status,
        "confidence": evaluation.confidence,
    }


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
    # If there are no retrieved chunks at all, short-circuit immediately.
    # The LLM must not be allowed to fabricate analysis from zero source material.
    if not chunks:
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

    # Build a rich context block for the LLM — ordered by relevance score
    context_parts: List[str] = []

    # Task question framed prominently so the LLM keeps it front-of-mind
    context_parts.append(f"=== ANALYST QUESTION ===\n{task}\n")

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
            context_parts.append(
                f"\n[chunk_id: {chunk.chunk_id}] (relevance={chunk.score:.3f}, source={chunk.source}{band_label})\n"
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


def _node_rewrite_query(state: AgentState, llm: LLMClient) -> AgentState:
    """Rewrite the query to improve retrieval (AMBIGUOUS path, max 1 iteration)."""
    original_query = state.get("task", "")
    _print_progress("REWRITE QUERY", f"original={original_query[:80]!r}")
    rewritten = llm.rewrite_query(original_query)
    logger.info("Query rewritten: %r → %r", original_query, rewritten)
    _print_progress("REWRITE QUERY done", f"rewritten={rewritten[:80]!r}", symbol="  OK")
    rewrite_count = state.get("rewrite_count", 0) + 1
    return {**state, "task": rewritten, "rewrite_count": rewrite_count}


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
    # Strip trailing punctuation (], ]., ],) that the LLM appends when citing inside arrays/sentences
    # Matches:  neo4j::TICKER::...  pgvec::...  qdrant::...  or bare  TICKER::...
    _RAW_CITED = re.findall(r'(?:neo4j|pgvec|qdrant)::[^\s",\]]+', output_text)
    _RAW_CITED += re.findall(r'\b[A-Z]{1,6}::[^\s",\]]+', output_text)
    cited_ids = {cid.rstrip("].,;") for cid in _RAW_CITED}
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
    # Matches:  neo4j::TICKER::...  pgvec::...  qdrant::...  or bare  TICKER::...
    raw_cited = re.findall(r'(?:neo4j|pgvec|qdrant)::[^\s",\]]+', output_text)
    raw_cited += re.findall(r'\b[A-Z]{1,6}::[^\s",\]]+', output_text)
    cited_ids = {cid.rstrip("].,;") for cid in raw_cited}
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
            cid = m.group(1).rstrip("].,;")
            # Preserve PostgreSQL / sentiment citations — they aren't Neo4j chunks
            if "sentiment_trends" in cid or "postgresql" in cid:
                return m.group(0)
            if _is_grounded(cid):
                return m.group(0)
            logger.debug("Stripping ungrounded inline citation from prose: %s", cid)
            return ""

        # Matches [neo4j::...], [pgvec::...], [qdrant::...], or bare [TICKER::...]
        return re.sub(r'\[((?:neo4j|pgvec|qdrant)::[^\]]+|[A-Z]{1,6}::[^\]]+)\]', _replace_match, s)
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
    
    # Fallback: if LLM put moat analysis in qualitative_summary instead of competitive_moat
    if not competitive_moat and qualitative_summary and not qualitative_summary.startswith("INSUFFICIENT_DATA"):
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

    # Verbose: citation grounding summary
    citation_report = validate_citations(output, retrieval)
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

def _route_after_crag(state: AgentState, toolkit: BusinessAnalystToolkit) -> str:
    """Branch after CRAG evaluation.

    Routing logic:
    - CORRECT (score >= config.crag_correct_threshold)    → generate_analysis immediately
    - AMBIGUOUS (between thresholds)                       → rewrite_query up to max_rewrite_loops,
                                                             then generate_analysis
    - INCORRECT (score < config.crag_ambiguous_threshold) → web_search_fallback only if
                                                             enable_web_fallback=True AND
                                                             there is genuinely no local data;
                                                             otherwise → generate_analysis

    The web_search_fallback is a last resort, not a default for low cosine scores.
    """
    crag_status = state.get("crag_status")
    rewrite_count = state.get("rewrite_count", 0)
    max_loops = toolkit.config.max_rewrite_loops

    if crag_status == CRAGStatus.CORRECT:
        return "generate_analysis"

    if crag_status == CRAGStatus.AMBIGUOUS and rewrite_count < max_loops:
        return "rewrite_query"

    # INCORRECT or AMBIGUOUS-exhausted: check whether any local data exists at all
    retrieval: Optional[RetrievalResult] = state.get("retrieval")
    has_chunks = bool(retrieval and retrieval.chunks)
    has_graph = bool(retrieval and retrieval.graph_facts)
    has_sentiment = state.get("sentiment") is not None

    if has_chunks or has_graph or has_sentiment:
        # Local data exists — let the LLM generate from what we have.
        # The system prompt instructs it to surface gaps in missing_context.
        return "generate_analysis"

    # Truly no data at all — use web fallback if enabled, else generate anyway
    # (LLM will produce INSUFFICIENT_DATA response per prompt instructions)
    if toolkit.config.enable_web_fallback:
        return "web_search_fallback"
    return "generate_analysis"


def _route_after_rewrite(state: AgentState) -> str:
    """Always go back to complex retrieval after a query rewrite."""
    return "complex_retrieval"


def _route_after_classification(state: AgentState) -> str:
    """Route to the appropriate retrieval path based on query_class."""
    query_class = (state.get("query_class") or "COMPLEX").upper()
    if query_class == "SIMPLE":
        return "fast_path_retrieval"
    if query_class == "NUMERICAL":
        return "numerical_path"
    return "complex_retrieval"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    toolkit: BusinessAnalystToolkit,
    llm: LLMClient,
) -> Any:  # CompiledStateGraph — typed as Any to avoid stub incompatibilities
    """Assemble the Optimized Adaptive Agentic Graph RAG pipeline and return a compiled graph."""

    graph = StateGraph(AgentState)

    # --- Phase 1: Metadata pre-check + sentiment enrichment ---
    graph.add_node(
        "metadata_precheck",
        lambda state: _node_metadata_precheck(cast(AgentState, state), toolkit),
    )
    graph.add_node(
        "fetch_sentiment_data",
        lambda state: _node_fetch_sentiment(cast(AgentState, state), toolkit),
    )

    # --- Phase 2: Query classification ---
    graph.add_node(
        "classify_query",
        lambda state: _node_classify_query(cast(AgentState, state), toolkit, llm),
    )

    # --- Phase 3: Adaptive retrieval paths ---
    graph.add_node(
        "fast_path_retrieval",
        lambda state: _node_fast_path_retrieval(cast(AgentState, state), toolkit),
    )
    graph.add_node(
        "numerical_path",
        lambda state: _node_numerical_path(cast(AgentState, state), toolkit),
    )
    graph.add_node(
        "complex_retrieval",
        lambda state: _node_complex_retrieval(cast(AgentState, state), toolkit),
    )

    # --- Phase 4: CRAG evaluate (complex path only) ---
    graph.add_node(
        "crag_evaluate",
        lambda state: _node_crag_evaluate(cast(AgentState, state), toolkit),
    )

    # --- Phase 5: Generation ---
    graph.add_node(
        "generate_analysis",
        lambda state: _node_generate_analysis(cast(AgentState, state), llm),
    )
    graph.add_node(
        "rewrite_query",
        lambda state: _node_rewrite_query(cast(AgentState, state), llm),
    )
    graph.add_node(
        "web_search_fallback",
        lambda state: _node_web_search_fallback(cast(AgentState, state), toolkit),
    )

    # --- Phase 6: Semantic cache check + output formatting ---
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

    # Phase 1 → Phase 2
    graph.add_edge("metadata_precheck", "fetch_sentiment_data")
    graph.add_edge("fetch_sentiment_data", "classify_query")

    # Phase 2 → Phase 3: three-way adaptive routing
    graph.add_conditional_edges(
        "classify_query",
        lambda state: _route_after_classification(cast(AgentState, state)),
        {
            "fast_path_retrieval": "fast_path_retrieval",
            "numerical_path": "numerical_path",
            "complex_retrieval": "complex_retrieval",
        },
    )

    # SIMPLE and NUMERICAL paths skip CRAG and go straight to generation
    graph.add_edge("fast_path_retrieval", "generate_analysis")
    graph.add_edge("numerical_path", "generate_analysis")

    # COMPLEX path goes through CRAG evaluate
    graph.add_edge("complex_retrieval", "crag_evaluate")

    # Conditional branching after CRAG evaluation (complex path only)
    graph.add_conditional_edges(
        "crag_evaluate",
        lambda state: _route_after_crag(cast(AgentState, state), toolkit),
        {
            "generate_analysis": "generate_analysis",
            "rewrite_query": "rewrite_query",
            "web_search_fallback": "web_search_fallback",
        },
    )

    # After rewrite → back to complex retrieval (creates the AMBIGUOUS loop)
    graph.add_conditional_edges(
        "rewrite_query",
        _route_after_rewrite,
        {"complex_retrieval": "complex_retrieval"},
    )

    # All generation paths converge → semantic cache check → output formatting
    graph.add_edge("generate_analysis", "semantic_cache_check")
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
        "rewrite_count": 0,
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
