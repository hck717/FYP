"""Business Analyst Agent — LangGraph CRAG pipeline.

Architecture (mirrors README.md diagram):

    Query + Ticker
        │
        ▼
    fetch_sentiment_data      ←  PostgreSQL: bullish/bearish/neutral %
        │
        ▼
    hybrid_retrieval          ←  Qdrant vector search (768-dim, nomic-embed-text)
                              ←  Neo4j Cypher graph traversal (Company nodes only)
                              ←  BM25 sparse keyword scoring
        │
        ▼
    hybrid_rerank             ←  30% BM25 + 70% Cross-Encoder
        │
        ▼
    crag_evaluate             ←  CORRECT (>0.55) / AMBIGUOUS (0.35-0.55) / INCORRECT (<0.35)
        │
        ├─ CORRECT    → generate_analysis  (LLM from graph context)
        ├─ AMBIGUOUS  → rewrite_query → retry hybrid_retrieval (once)
        └─ INCORRECT  → web_search_fallback
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
import unicodedata
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict, cast

from langgraph.graph import END, StateGraph

from .config import BusinessAnalystConfig, load_config
from .llm import LLMClient
from .schema import CRAGStatus, RetrievalResult, SentimentSnapshot, serialise_chunk
from .tools import BusinessAnalystToolkit
from .web_search_interface import web_search_fallback as _call_web_search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """Shared mutable state threaded through every node."""

    # Inputs
    task: str
    ticker: Optional[str]

    # Availability profile (optional, injected by orchestration planner)
    availability_profile: Optional[Dict[str, bool]]

    # Enrichment
    sentiment: Optional[SentimentSnapshot]
    company_node: Optional[Dict[str, Any]]   # raw Neo4j Company node properties
    community_summary: Optional[str]         # graph-community summary (2A: Graph RAG)
    retrieval: Optional[RetrievalResult]
    rewrite_count: int          # guard: max 1 rewrite loop

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

def _node_fetch_sentiment(state: AgentState, toolkit: BusinessAnalystToolkit) -> AgentState:
    """Fetch bullish/bearish/neutral % from PostgreSQL sentiment_trends.
    Also fetches the Company node properties from Neo4j for company_overview,
    and builds a graph-community summary (2A: Graph RAG).
    """
    ticker = state.get("ticker")
    sentiment = toolkit.fetch_sentiment(ticker)
    if sentiment is None:
        logger.info("No sentiment data found for ticker=%s", ticker)
    company_node = toolkit.fetch_company_overview(ticker)
    community_summary = toolkit.fetch_community_summary(ticker)
    if community_summary:
        logger.info("[BA] Graph community summary for %s: %s", ticker, community_summary[:120])
    return {**state, "sentiment": sentiment, "company_node": company_node, "community_summary": community_summary}


def _node_hybrid_retrieval(state: AgentState, toolkit: BusinessAnalystToolkit) -> AgentState:
    """Dense (Neo4j vector) + sparse (BM25) + graph (Cypher) retrieval.

    Short-circuits immediately with an empty retrieval when the availability
    profile indicates that neither Neo4j chunks nor Qdrant vectors exist for
    this ticker — avoids noisy failed vector searches against empty stores.
    """
    query = state.get("task", "")
    ticker = state.get("ticker")
    profile: Optional[Dict[str, bool]] = state.get("availability_profile")

    # NOTE: We intentionally do NOT short-circuit retrieval based on the availability
    # profile's has_any_qualitative flag.  The profile is derived from a fast
    # data_availability check that queries Neo4j and Qdrant with strict ticker-match
    # filters — if those checks fail or return stale results (e.g. the Qdrant scroll
    # finds no AAPL vector but Neo4j has 20+ chunks), we would incorrectly skip a
    # retrieval that would otherwise succeed.  The retriever itself gracefully handles
    # empty results and the CRAG evaluator will route to web_search_fallback if truly
    # nothing is found.  We only honour the profile to log a warning.
    if profile is not None and not profile.get("has_any_qualitative", True):
        logger.warning(
            "[BA] Availability profile reports no qualitative data for ticker=%s "
            "— attempting retrieval anyway (profile may be stale or incomplete).",
            ticker,
        )

    # Prepend the ticker symbol to the query so the embedding carries a
    # company-identity signal.  This pushes the dense vector closer to
    # on-ticker documents and away from semantically similar but off-ticker
    # content (e.g. "AAPL: Provide a comprehensive analysis of Apple Inc.")
    if ticker and not query.upper().startswith(ticker.upper()):
        query = f"{ticker.upper()}: {query}"

    retrieval = toolkit.retrieve(query, ticker)
    return {**state, "retrieval": retrieval}


def _node_hybrid_rerank(state: AgentState, toolkit: BusinessAnalystToolkit) -> AgentState:
    """Re-rank is already applied inside HybridRetriever.retrieve().

    This node exists as an explicit graph step so the architecture matches
    the README diagram. It is a no-op at runtime — reranking happened in
    the retrieval node — but it makes the pipeline visible and testable.
    """
    return state


def _node_crag_evaluate(state: AgentState, toolkit: BusinessAnalystToolkit) -> AgentState:
    """Score top retrieved chunks and classify as CORRECT / AMBIGUOUS / INCORRECT."""
    retrieval: Optional[RetrievalResult] = state.get("retrieval")
    chunks = retrieval.chunks if retrieval else []
    ticker = state.get("ticker")
    evaluation = toolkit.evaluate(chunks, ticker=ticker)
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

    # Final ticker guard — even after HybridRetriever's post-merge filter a
    # chunk may have slipped through with unknown/missing metadata (the helper
    # lets those pass).  Re-apply here so the LLM context is guaranteed clean.
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
        raw = llm.generate(
            query=task,
            ticker=ticker,
            context=context,
            sentiment=sentiment,
        )
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        raw = {
            "qualitative_summary": f"GENERATION_ERROR: {exc}",
            "crag_status": state.get("crag_status", CRAGStatus.INCORRECT).value  # type: ignore[union-attr]
        }
    return {**state, "llm_output": raw, "fallback_triggered": False}


def _node_rewrite_query(state: AgentState, llm: LLMClient) -> AgentState:
    """Rewrite the query to improve retrieval (AMBIGUOUS path, max 1 iteration)."""
    original_query = state.get("task", "")
    rewritten = llm.rewrite_query(original_query)
    logger.info("Query rewritten: %r → %r", original_query, rewritten)
    rewrite_count = state.get("rewrite_count", 0) + 1
    return {**state, "task": rewritten, "rewrite_count": rewrite_count}


def _node_web_search_fallback(state: AgentState, toolkit: BusinessAnalystToolkit) -> AgentState:
    """Trigger Web Search Agent when CRAG confidence is INCORRECT (AMBIGUOUS exhausted or < 0.35)."""
    ticker = state.get("ticker")
    query = state.get("task", "")
    logger.info("CRAG INCORRECT — triggering web search fallback for ticker=%s", ticker)

    result = _call_web_search(query=query, ticker=ticker, config=toolkit.config)
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
    _RAW_CITED = re.findall(r'qdrant::[A-Z]+::[^\s",\]]+', output_text)
    _RAW_CITED += re.findall(r'neo4j::[^\s",\]]+', output_text)
    cited_ids = {cid.rstrip("].,;") for cid in _RAW_CITED}
    # Use prefix matching with Unicode normalization: a cited ID is "grounded" if it
    # matches any real ID after normalising Unicode (LLM uses curly quotes, Qdrant uses ASCII).
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
    # Exclude PostgreSQL-sourced citations like "qdrant::TSLA::sentiment_trends" —
    # these are the LLM citing the sentiment source, not a Qdrant chunk ID.
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


def _strip_ungrounded_inline_citations(
    obj: Any,
    real_ids: set[str],
) -> Any:
    """Recursively scan obj and replace ungrounded inline citations in string values.

    Any ``[qdrant::TICKER::slug]`` or ``[neo4j::...]`` token embedded in a string
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
        # (e.g. \u2019 RIGHT SINGLE QUOTATION MARK) while the actual Qdrant
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
            # Preserve PostgreSQL / sentiment citations — they aren't Qdrant chunks
            if "sentiment_trends" in cid or "postgresql" in cid:
                return m.group(0)
            if _is_grounded(cid):
                return m.group(0)
            logger.debug("Stripping ungrounded inline citation from prose: %s", cid)
            return ""

        return re.sub(r'\[(qdrant::[A-Z]+::[^\]]+|neo4j::[^\]]+)\]', _replace_match, s)

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

    # If fallback was triggered, merge web search facts into key_risks / missing_context
    key_risks = llm_output.get("key_risks") or []
    missing_context = llm_output.get("missing_context") or []
    qualitative_summary = llm_output.get("qualitative_summary", "")
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
    # only handles the [qdrant::...] inline-token format, not bare string arrays.
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
    - CORRECT (score >= 0.55)   → generate_analysis immediately
    - AMBIGUOUS (0.35-0.55)     → rewrite_query once, then generate_analysis
    - INCORRECT (score < 0.35)  → web_search_fallback only if enable_web_fallback=True
                                   AND there is genuinely no local data at all;
                                   otherwise → generate_analysis (LLM handles thin context)

    The web_search_fallback is a last resort, not a default for low cosine scores.
    News article cosine similarity against qualitative queries naturally lands in
    the 0.3-0.5 range; routing to fallback in that case discards useful context.
    """
    crag_status = state.get("crag_status")
    rewrite_count = state.get("rewrite_count", 0)

    if crag_status == CRAGStatus.CORRECT:
        return "generate_analysis"

    if crag_status == CRAGStatus.AMBIGUOUS and rewrite_count < 1:
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
    """Always go back to retrieval after a query rewrite."""
    return "hybrid_retrieval"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    toolkit: BusinessAnalystToolkit,
    llm: LLMClient,
) -> Any:  # CompiledStateGraph — typed as Any to avoid stub incompatibilities
    """Assemble the LangGraph CRAG pipeline and return a compiled graph."""

    graph = StateGraph(AgentState)

    # Register nodes (bind toolkit/llm via closures)
    graph.add_node(
        "fetch_sentiment_data",
        lambda state: _node_fetch_sentiment(cast(AgentState, state), toolkit),
    )
    graph.add_node(
        "hybrid_retrieval",
        lambda state: _node_hybrid_retrieval(cast(AgentState, state), toolkit),
    )
    graph.add_node(
        "hybrid_rerank",
        lambda state: _node_hybrid_rerank(cast(AgentState, state), toolkit),
    )
    graph.add_node(
        "crag_evaluate",
        lambda state: _node_crag_evaluate(cast(AgentState, state), toolkit),
    )
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
    graph.add_node(
        "format_json_output",
        _node_format_json_output,
    )

    # Entry point
    graph.set_entry_point("fetch_sentiment_data")

    # Linear edges
    graph.add_edge("fetch_sentiment_data", "hybrid_retrieval")
    graph.add_edge("hybrid_retrieval", "hybrid_rerank")
    graph.add_edge("hybrid_rerank", "crag_evaluate")

    # Conditional branching after CRAG evaluation
    graph.add_conditional_edges(
        "crag_evaluate",
        lambda state: _route_after_crag(cast(AgentState, state), toolkit),
        {
            "generate_analysis": "generate_analysis",
            "rewrite_query": "rewrite_query",
            "web_search_fallback": "web_search_fallback",
        },
    )

    # After rewrite → back to retrieval (creates the AMBIGUOUS loop)
    graph.add_conditional_edges(
        "rewrite_query",
        _route_after_rewrite,
        {"hybrid_retrieval": "hybrid_retrieval"},
    )

    # Convergence: both generation paths lead to output formatting
    graph.add_edge("generate_analysis", "format_json_output")
    graph.add_edge("web_search_fallback", "format_json_output")

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
) -> Dict[str, Any]:
    """Run the Business Analyst CRAG pipeline and return a structured JSON dict.

    Args:
        task:                 Natural language query (e.g. "What is Apple's competitive moat?").
        ticker:               Optional ticker symbol (e.g. "AAPL").
        config:               Optional config override; defaults to env-var-based config.
        availability_profile: Optional per-ticker data profile from data_availability module.
                              When provided, used to skip dead code paths (e.g. empty vector stores).

    Returns:
        Structured JSON dict conforming to the output schema in README.md.
    """
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
        result = run(task=task, ticker=ticker)
        indent = 2 if args.pretty else None
        print(json.dumps(result, indent=indent, default=str))
    except Exception as exc:
        logger.error("Agent run failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
