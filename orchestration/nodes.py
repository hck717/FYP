"""Node implementations for the orchestration LangGraph pipeline.

Each function is a pure node — takes state, returns partial state update.

Pipeline (native LangGraph parallel):
  planner → [node_business_analyst, node_quant_fundamental, node_financial_modelling,
              node_web_search (optional), node_stock_research (optional)]
          → node_summarizer → node_post_processing → END

  LangGraph native fan-out: the planner routes directly to all enabled agent nodes
  via conditional edges, and LangGraph executes these branches simultaneously.
  Each agent has its own per-agent ReAct retry counter (agent_react_iterations dict)
  and per-agent conditional edge that routes back if below the retry cap.

  node_post_processing runs RLAIF scoring AND persists episodic failure patterns
  to PostgreSQL in a single node after the summarizer.

Multi-ticker support
--------------------
When the planner resolves multiple ticker symbols (e.g. "Compare MSFT vs AAPL"),
every agent node iterates over all tickers in ``state["tickers"]`` and accumulates
per-ticker result dicts in the ``*_outputs`` list keys.  The legacy single-value
``*_output`` aliases are set to the first ticker's result for backward-compat.
"""

from __future__ import annotations

import logging
import os
import queue
import time
import uuid
import json
from typing import Any, Dict, List, Optional, cast

from .llm import plan_query, summarise_results, summarise_results_structured
from .state import OrchestrationState
from . import feedback

logger = logging.getLogger(__name__)

_MAX_REACT_ITERATIONS = 3  # safety cap (legacy sequential path only)


# ── Per-session agent progress queues ────────────────────────────────────────
# The Streamlit UI subscribes to a session's queue before the graph runs.
# node_parallel_agents pushes AgentProgressEvent dicts as each agent
# finishes so the UI can show live per-agent status without waiting for the
# entire parallel_agents node to complete.

_progress_queues: Dict[str, "queue.Queue[Dict[str, Any]]"] = {}


def _extract_agent_excerpt(name: str, result_list: List[Dict[str, Any]]) -> Optional[str]:
    """Return a short one-liner excerpt of an agent's output for the live UI panel.

    Picks the most informative narrative field from the first result dict.
    Never raises — returns None on any failure.
    """
    try:
        if not result_list:
            return None
        r = result_list[0]
        raw: Optional[str] = None
        if name == "business_analyst":
            raw = (
                r.get("qualitative_summary")
                or (r.get("qualitative_analysis") or {}).get("narrative")
                or (r.get("sentiment_verdict") or {}).get("label")
            )
        elif name == "quant_fundamental":
            raw = (
                r.get("quantitative_summary")
                or (r.get("cot_validation_notes") or [None])[0]
            )
        elif name == "web_search":
            raw = (
                r.get("sentiment_rationale")
                or (r.get("breaking_news") or [None])[0]
            )
        elif name == "financial_modelling":
            raw = (
                r.get("quantitative_summary")
                or (r.get("moe_consensus") or {}).get("consensus_narrative")
            )
        elif name == "stock_research":
            raw = (
                r.get("broker_consensus")
                or r.get("transcript_comparison")
            )
        if raw and isinstance(raw, str):
            # Trim to ~120 chars without cutting mid-word
            raw = raw.strip().replace("\n", " ")
            if len(raw) > 120:
                raw = raw[:117].rsplit(" ", 1)[0] + "…"
            return raw
    except Exception:
        pass
    return None


def subscribe_agent_progress(session_id: str) -> "queue.Queue[Dict[str, Any]]":
    """Create (or replace) a progress queue for the given session_id.

    Call this before starting the graph stream.  Drain it from the UI thread
    while the parallel_agents node is running.  Call unsubscribe_agent_progress
    when done to free memory.

    Queue items are dicts::

        {"agent": str, "status": "started" | "done" | "error",
         "ticker": str | None, "elapsed_ms": int, "error": str | None}

    A sentinel ``{"agent": "__done__", "status": "done"}`` is pushed when
    node_parallel_agents finishes so the consumer knows to stop polling.
    """
    q: queue.Queue[Dict[str, Any]] = queue.Queue()
    _progress_queues[session_id] = q
    return q


def unsubscribe_agent_progress(session_id: str) -> None:
    """Remove the progress queue for the given session_id."""
    _progress_queues.pop(session_id, None)


def _push_progress(session_id: str, event: Dict[str, Any]) -> None:
    """Push a progress event to the session queue if one is subscribed."""
    _pq = _progress_queues.get(session_id)
    if _pq is not None:
        try:
            _pq.put_nowait(event)
        except Exception:
            pass


def _incr_agent_iter(state: OrchestrationState, agent_name: str) -> Dict[str, int]:
    """Return a partial agent_react_iterations dict with only this agent's incremented count.

    Since agent_react_iterations uses a _merge_dicts reducer, each agent node only
    needs to return its own {agent_name: count} entry — LangGraph merges all agents'
    entries automatically.
    """
    iters: Dict[str, int] = dict(state.get("agent_react_iterations") or {})
    new_count = iters.get(agent_name, 0) + 1
    return {agent_name: new_count}


# ── A2: Implicit telemetry helper ────────────────────────────────────────────

def _log_telemetry(
    run_id: str,
    agent_name: str,
    event_type: str,
    latency_ms: Optional[int] = None,
    complexity_declared: Optional[int] = None,
    react_loops_used: Optional[int] = None,
    notes: Optional[str] = None,
) -> None:
    """A2: Write one telemetry event to agent_run_telemetry (non-fatal).

    event_type values:
        'latency'             — per-agent wall-clock time
        'complexity_mismatch' — declared complexity vs. actual loops
        'crag_fallback'       — BA agent fell back from vector to web search
        'timeout'             — an agent error contained the word 'timeout'
    """
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "airflow"),
            user=os.getenv("POSTGRES_USER", "airflow"),
            password=os.getenv("POSTGRES_PASSWORD", "airflow"),
        )
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_run_telemetry
                    (run_id, agent_name, event_type, latency_ms,
                     complexity_declared, react_loops_used, notes, recorded_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                """,
                (run_id, agent_name, event_type, latency_ms,
                 complexity_declared, react_loops_used, notes),
            )
        conn.close()
    except Exception as exc:
        logger.debug("[telemetry] Non-fatal log failure: %s", exc)


# ── Node 1: Planner ───────────────────────────────────────────────────────────

def node_planner(state: OrchestrationState) -> OrchestrationState:
    """Use local DeepSeek to parse the user query and decide which agents to run.

    Populates:
      - plan, ticker, tickers, run_business_analyst, run_quant_fundamental, run_web_search
      - episodic_hints: pre-emptive hints from agent_episodic_memory for known failure patterns
      - output_language: detected language requirement from user query
    """
    user_query = state.get("user_query", "")
    logger.info("[planner] Analysing query: %r", user_query)

    # Detect language requirement from user query
    output_language: Optional[str] = None
    query_lower = user_query.lower()
    language_keywords = {
        "cantonese": ["cantonese", "in cantonese", "write in cantonese", "cantonese language"],
        "spanish": ["spanish", "in spanish", "espanol", "write in spanish", "spanish language"],
        "mandarin": ["mandarin", "in mandarin", "putonghua", "write in mandarin", "mandarin language"],
        "french": ["french", "in french", "francais", "write in french", "french language"],
        "german": ["german", "in german", "deutsch", "write in german", "german language"],
        "japanese": ["japanese", "in japanese", "nihongo", "write in japanese", "japanese language"],
        "korean": ["korean", "in korean", "hangul", "write in korean", "korean language"],
        "portuguese": ["portuguese", "in portuguese", "portugues", "write in portuguese"],
        "italian": ["italian", "in italian", "italiano", "write in italian"],
        "dutch": ["dutch", "in dutch", "nederlands", "write in dutch"],
        "russian": ["russian", "in russian", "russkiy", "write in russian"],
        "arabic": ["arabic", "in arabic", "arabi", "write in arabic"],
        "hindi": ["hindi", "in hindi", "write in hindi"],
        "thai": ["thai", "in thai", "write in thai"],
        "vietnamese": ["vietnamese", "in vietnamese", "tieng viet", "write in vietnamese"],
        "indonesian": ["indonesian", "in indonesian", "bahasa", "write in indonesian"],
    }
    for lang, keywords in language_keywords.items():
        if any(kw in query_lower for kw in keywords):
            output_language = lang
            logger.info("[planner] Detected language requirement: %s", output_language)
            break

    # --- in-context learning: inject worst past cases into planner prompt ---
    worst_case_context: str = ""
    try:
        worst_cases = feedback.get_worst_cases(limit=5)
        if worst_cases:
            lines = ["PAST FAILURES TO AVOID (worst scored runs - learn from these):"]
            for i, w in enumerate(worst_cases, 1):
                human_signal = ""
                if w.get("helpful") is False:
                    comment = w.get("comment") or ""
                    tags = w.get("issue_tags") or []
                    if isinstance(tags, str):
                        try:
                            tags = json.loads(tags)
                        except Exception:
                            tags = []
                    human_signal = f" | USER DOWNVOTE: {', '.join(tags)}" + (f" - '{comment}'" if comment else "")

                weaknesses = w.get("weaknesses") or []
                if isinstance(weaknesses, str):
                    try:
                        weaknesses = json.loads(weaknesses)
                    except Exception:
                        weaknesses = [weaknesses]

                lines.append(
                    f"  [{i}] Query: \"{str(w.get('user_query', ''))[:80]}\" | "
                    f"Ticker: {w.get('ticker', 'N/A')} | "
                    f"Score: {float(w.get('overall_score') or 0):.1f}/10 | "
                    f"Blamed: {w.get('agent_blamed', 'unknown')} | "
                    f"Weaknesses: {'; '.join(str(x) for x in weaknesses[:2])}"
                    f"{human_signal}"
                )
            lines.append(
                "ACTION: Adjust your routing to avoid repeating these failure patterns. "
                "If a similar query + ticker combination appears, increase react_max_iterations "
                "or force run_web_search=true as a fallback."
            )
            worst_case_context = "\n".join(lines)
            logger.info("[planner] Injecting %d worst-case examples into planner context.", len(worst_cases))
    except Exception as exc:
        logger.warning("[planner] Worst-case context injection failed (non-fatal): %s", exc)

    plan = plan_query(user_query, worst_case_context=worst_case_context)

    # Extract and remove the thinking trace from the plan dict (it's metadata, not routing data)
    planner_trace: str = plan.pop("planner_trace", "") or ""
    if planner_trace:
        logger.info("[planner] Thinking trace captured (%d chars).", len(planner_trace))

    # --- resolve tickers ---------------------------------------------------
    # The planner LLM may return a single ticker or a list in the "tickers"
    # field for comparison queries.  We also run the QF agent's own multi-
    # ticker extractor as a belt-and-suspenders fallback.
    from agents.quant_fundamental.agent import extract_tickers_from_prompt  # type: ignore[import]

    plan_tickers: List[str] = []

    # Prefer an explicit "tickers" list from the planner (new schema)
    if plan.get("tickers") and isinstance(plan["tickers"], list):
        plan_tickers = [str(t).strip().upper() for t in plan["tickers"] if t]

    # Fall back to legacy single "ticker" field
    if not plan_tickers and plan.get("ticker"):
        plan_tickers = [str(plan["ticker"]).strip().upper()]

    # Last resort: regex extraction from the raw query
    if not plan_tickers:
        plan_tickers = extract_tickers_from_prompt(user_query)

    # Deduplicate while preserving order
    seen: set = set()
    tickers: List[str] = []
    for t in plan_tickers:
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)

    ticker: Optional[str] = tickers[0] if tickers else None

    run_ba = bool(plan.get("run_business_analyst", True))
    run_qf = bool(plan.get("run_quant_fundamental", True))
    run_ws = bool(plan.get("run_web_search", False))
    run_fm = bool(plan.get("run_financial_modelling", False))

    # ReAct looping disabled by design: run each agent node once.
    raw_complexity = plan.get("complexity", 1)
    react_max = 1

    # --- data availability check -------------------------------------------
    # Run once per request in the planner so every downstream node can branch
    # on what is actually present without hitting the DBs again.
    data_availability: Optional[Dict[str, Any]] = None
    try:
        from .data_availability import check_all  # type: ignore[import]
        data_availability = check_all(tickers=tickers or None)
        logger.info("[planner] Data availability: %s", data_availability.get("summary", ""))
        # Adjust agent flags based on availability
        avail_degraded = data_availability.get("degraded_tiers") or []
        # If postgres is fully down, QF will produce no numbers — still run it
        # so it returns a graceful "NO_DATA" output rather than crashing.
        # We never *skip* agents based on availability — we just let them degrade.
    except Exception as exc:
        logger.warning("[planner] Data availability check failed: %s", exc)

    # --- episodic memory lookup --------------------------------------------
    # Query stored failure patterns; if a similar past query hit known agent
    # failures, adjust the plan pre-emptively (e.g. force web search).
    episodic_hints: Optional[Dict[str, Any]] = None
    try:
        from .episodic_memory import lookup_similar_failures, build_preemptive_plan_hints  # type: ignore[import]
        failures = lookup_similar_failures(user_query, tickers=tickers or None)
        episodic_hints = build_preemptive_plan_hints(failures)
        if episodic_hints:
            logger.info("[planner] Episodic hints from %d similar past failure(s): %s", len(failures), episodic_hints)
            if episodic_hints.get("force_web_search"):
                logger.info("[planner] Episodic memory → forcing run_web_search=True")
                run_ws = True
            degraded = episodic_hints.get("degraded_agents") or []
            if degraded:
                logger.info("[planner] Episodic memory → known degraded agents: %s", degraded)
    except Exception as exc:
        logger.warning("[planner] Episodic memory lookup failed (non-fatal): %s", exc)

    logger.info(
        "[planner] tickers=%s  ba=%s  quant=%s  web=%s  fm=%s  complexity=%s  react_max=%d",
        tickers, run_ba, run_qf, run_ws, run_fm, raw_complexity, react_max,
    )

    return {
        **state,
        "plan": plan,
        "ticker": ticker,
        "tickers": tickers,
        "run_business_analyst": run_ba,
        "run_quant_fundamental": run_qf,
        "run_web_search": run_ws,
        "run_financial_modelling": run_fm,
        "run_stock_research": bool(plan.get("run_stock_research", False)),
        "run_macro": bool(plan.get("run_macro", False)),
        "run_insider_news": bool(plan.get("run_insider_news", False)),
        "data_availability": data_availability,
        "episodic_hints": episodic_hints,
        "planner_trace": planner_trace,
        # Prefer language detected from query text; fall back to UI-supplied value in state
        "output_language": output_language or state.get("output_language"),
        "react_steps": [],
        "react_iteration": 0,
        "react_max_iterations": react_max,
        "agent_errors": {},
        # Reset all output lists/aliases
        "business_analyst_outputs": [],
        "quant_fundamental_outputs": [],
        "web_search_outputs": [],
        "financial_modelling_outputs": [],
        "stock_research_outputs": [],
        "macro_outputs": [],
        "insider_news_outputs": [],
        "business_analyst_output": None,
        "quant_fundamental_output": None,
        "web_search_output": None,
        "financial_modelling_output": None,
        "stock_research_output": None,
        "macro_output": None,
        "insider_news_output": None,
    }


# ── Node 2: Business Analyst ──────────────────────────────────────────────────

def node_business_analyst(state: OrchestrationState) -> OrchestrationState:
    """Call the Business Analyst CRAG pipeline for every resolved ticker."""
    tickers  = state.get("tickers") or []
    query    = state.get("user_query", "")
    outputs: List[Dict[str, Any]] = []

    # Fall back to legacy single ticker if list is empty
    if not tickers and state.get("ticker"):
        tickers = [state["ticker"]]  # type: ignore[list-item]

    session_id: str = state.get("session_id", "") or ""  # type: ignore[assignment]
    run_id: str = session_id or str(uuid.uuid4())[:8]
    complexity: int = state.get("react_max_iterations") or 1
    start_t = time.time()
    # own_step: only this agent's step — will be merged via operator.add reducer
    own_step: Dict[str, Any] = {"tool": "business_analyst", "input": {"tickers": tickers, "query": query}}
    _push_progress(session_id, {"agent": "business_analyst", "status": "started", "tickers": tickers, "elapsed_ms": 0, "error": None})

    logger.info("[business_analyst] Running for tickers=%s", tickers)

    try:
        from agents.business_analyst.agent import run_full_analysis, run  # type: ignore[import]
        from .data_availability import ticker_data_profile  # type: ignore[import]

        data_availability = state.get("data_availability")

        for t in tickers:
            logger.info("[business_analyst] Processing ticker=%s", t)
            # Build per-ticker profile so BA can skip empty vector stores
            profile = None
            if data_availability and t:
                try:
                    profile = ticker_data_profile(data_availability, t)
                except Exception:
                    pass
            if t:
                result = run_full_analysis(ticker=t, availability_profile=profile)
            else:
                result = run(task=query, ticker=None)
            outputs.append(result)

        obs_parts = [
            f"{r.get('ticker', '?')}:crag={r.get('crag_status')},conf={r.get('confidence', 0):.2f}"
            for r in outputs
        ]
        own_step["observation"] = "Success — " + " | ".join(obs_parts)
        logger.info("[business_analyst] Done for %d ticker(s).", len(outputs))

        elapsed_ms = int((time.time() - start_t) * 1000)
        _excerpt = _extract_agent_excerpt("business_analyst", outputs)
        _thinking_trace = outputs[0].get("thinking_trace") or [] if outputs else []
        _push_progress(session_id, {"agent": "business_analyst", "status": "done", "tickers": tickers, "elapsed_ms": elapsed_ms, "error": None, "excerpt": _excerpt, "thinking_trace": _thinking_trace})
        # A2: CRAG fallback telemetry
        if outputs and outputs[0].get("crag_status") == "fallback":
            _log_telemetry(run_id, "business_analyst", "crag_fallback",
                           notes=f"crag_status=fallback ticker={outputs[0].get('ticker', '?')}")
        _log_telemetry(run_id, "business_analyst", "latency", latency_ms=elapsed_ms,
                       complexity_declared=complexity, react_loops_used=_incr_agent_iter(state, "business_analyst").get("business_analyst", 1))

        first = outputs[0] if outputs else None
        return {
            "business_analyst_outputs": outputs,
            "business_analyst_output": first,
            "react_steps": [own_step],
            "agent_errors": {},
            "agent_react_iterations": _incr_agent_iter(state, "business_analyst"),
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[business_analyst] Failed: %s", msg)
        own_step["observation"] = f"Error: {msg}"
        elapsed_ms = int((time.time() - start_t) * 1000)
        _push_progress(session_id, {"agent": "business_analyst", "status": "error", "tickers": tickers, "elapsed_ms": elapsed_ms, "error": msg[:200]})
        if "timeout" in msg.lower():
            _log_telemetry(run_id, "business_analyst", "timeout", latency_ms=elapsed_ms,
                           complexity_declared=complexity, notes=msg[:500])
        else:
            _log_telemetry(run_id, "business_analyst", "latency", latency_ms=elapsed_ms,
                           complexity_declared=complexity, notes=f"error: {msg[:200]}")
        return {
            "react_steps": [own_step],
            "agent_errors": {"business_analyst": msg},
            "agent_react_iterations": _incr_agent_iter(state, "business_analyst"),
        }


# ── Node 3: Quant Fundamental ─────────────────────────────────────────────────

def node_quant_fundamental(state: OrchestrationState) -> OrchestrationState:
    """Call the Quantitative Fundamental pipeline for every resolved ticker."""
    tickers  = state.get("tickers") or []
    query    = state.get("user_query", "")
    outputs: List[Dict[str, Any]] = []

    # Fall back to legacy single ticker if list is empty
    if not tickers and state.get("ticker"):
        tickers = [state["ticker"]]  # type: ignore[list-item]

    session_id: str = state.get("session_id", "") or ""  # type: ignore[assignment]
    run_id: str = session_id or str(uuid.uuid4())[:8]
    complexity: int = state.get("react_max_iterations") or 1
    start_t = time.time()
    own_step: Dict[str, Any] = {"tool": "quant_fundamental", "input": {"tickers": tickers, "query": query}}
    _push_progress(session_id, {"agent": "quant_fundamental", "status": "started", "tickers": tickers, "elapsed_ms": 0, "error": None})

    logger.info("[quant_fundamental] Running for tickers=%s", tickers)

    try:
        from agents.quant_fundamental.agent import run_full_analysis  # type: ignore[import]
        from .data_availability import ticker_data_profile  # type: ignore[import]

        data_availability = state.get("data_availability")

        if tickers:
            # Call run_full_analysis once per ticker for clean, isolated results
            for t in tickers:
                logger.info("[quant_fundamental] Processing ticker=%s", t)
                # Build per-ticker profile for availability-aware fetch skip
                profile = None
                if data_availability and t:
                    try:
                        profile = ticker_data_profile(data_availability, t)
                    except Exception:
                        pass
                result = run_full_analysis(ticker=str(t), availability_profile=profile)
                # run_full_analysis returns a dict for single ticker
                if isinstance(result, list):
                    outputs.extend(result)
                else:
                    outputs.append(result)
        else:
            # Prompt-only fallback — may return list or dict
            result = run_full_analysis(prompt=query)
            if isinstance(result, list):
                outputs.extend(result)
            else:
                outputs.append(result)

        obs_parts = [
            f"{r.get('ticker', '?')}:dq={r.get('data_quality', {}).get('status')}"
            for r in outputs
        ]
        own_step["observation"] = "Success — " + " | ".join(obs_parts)
        logger.info("[quant_fundamental] Done for %d ticker(s).", len(outputs))

        elapsed_ms = int((time.time() - start_t) * 1000)
        _excerpt = _extract_agent_excerpt("quant_fundamental", outputs)
        _push_progress(session_id, {"agent": "quant_fundamental", "status": "done", "tickers": tickers, "elapsed_ms": elapsed_ms, "error": None, "excerpt": _excerpt})
        _log_telemetry(run_id, "quant_fundamental", "latency", latency_ms=elapsed_ms,
                       complexity_declared=complexity, react_loops_used=_incr_agent_iter(state, "quant_fundamental").get("quant_fundamental", 1))

        first = outputs[0] if outputs else None
        return {
            "quant_fundamental_outputs": outputs,
            "quant_fundamental_output": first,
            "react_steps": [own_step],
            "agent_errors": {},
            "agent_react_iterations": _incr_agent_iter(state, "quant_fundamental"),
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[quant_fundamental] Failed: %s", msg)
        own_step["observation"] = f"Error: {msg}"
        elapsed_ms = int((time.time() - start_t) * 1000)
        _push_progress(session_id, {"agent": "quant_fundamental", "status": "error", "tickers": tickers, "elapsed_ms": elapsed_ms, "error": msg[:200]})
        if "timeout" in msg.lower():
            _log_telemetry(run_id, "quant_fundamental", "timeout", latency_ms=elapsed_ms,
                           complexity_declared=complexity, notes=msg[:500])
        else:
            _log_telemetry(run_id, "quant_fundamental", "latency", latency_ms=elapsed_ms,
                           complexity_declared=complexity, notes=f"error: {msg[:200]}")
        return {
            "react_steps": [own_step],
            "agent_errors": {"quant_fundamental": msg},
            "agent_react_iterations": _incr_agent_iter(state, "quant_fundamental"),
        }


# ── Node 4: Web Search ────────────────────────────────────────────────────────

def node_web_search(state: OrchestrationState) -> OrchestrationState:
    """Call the Web Search agent (Perplexity Sonar) for every resolved ticker.

    This node is only reached when the planner explicitly sets run_web_search=True.
    For multi-ticker queries, one search is performed per ticker.
    """
    tickers  = state.get("tickers") or []
    query    = state.get("user_query", "")
    outputs: List[Dict[str, Any]] = []

    # Fall back to legacy single ticker if list is empty
    if not tickers and state.get("ticker"):
        tickers = [state["ticker"]]  # type: ignore[list-item]
    # If still no tickers, do a single query-only search
    if not tickers:
        tickers = [None]  # type: ignore[list-item]

    session_id: str = state.get("session_id", "") or ""  # type: ignore[assignment]
    run_id: str = session_id or str(uuid.uuid4())[:8]
    complexity: int = state.get("react_max_iterations") or 1
    start_t = time.time()
    own_step: Dict[str, Any] = {"tool": "web_search", "input": {"tickers": tickers, "query": query}}
    _push_progress(session_id, {"agent": "web_search", "status": "started", "tickers": tickers, "elapsed_ms": 0, "error": None})

    logger.info("[web_search] Running for tickers=%s", tickers)

    try:
        from agents.web_search.agent import run_web_search_agent, WebSearchInput  # type: ignore[import]

        for t in tickers:
            logger.info("[web_search] Processing ticker=%s", t)
            agent_input = WebSearchInput(
                query=query,
                ticker=t,
                recency_filter="week",
                model=None,
            )
            result = run_web_search_agent(agent_input)
            outputs.append(dict(result))

        obs_parts = [
            f"{r.get('ticker', '?')}:sentiment={r.get('sentiment_signal')}"
            for r in outputs
        ]
        own_step["observation"] = "Success — " + " | ".join(obs_parts)
        logger.info("[web_search] Done for %d ticker(s).", len(outputs))

        elapsed_ms = int((time.time() - start_t) * 1000)
        _excerpt = _extract_agent_excerpt("web_search", outputs)
        _push_progress(session_id, {"agent": "web_search", "status": "done", "tickers": tickers, "elapsed_ms": elapsed_ms, "error": None, "excerpt": _excerpt})
        _log_telemetry(run_id, "web_search", "latency", latency_ms=elapsed_ms,
                       complexity_declared=complexity, react_loops_used=_incr_agent_iter(state, "web_search").get("web_search", 1))

        first = outputs[0] if outputs else None
        return {
            "web_search_outputs": outputs,
            "web_search_output": first,
            "react_steps": [own_step],
            "agent_errors": {},
            "agent_react_iterations": _incr_agent_iter(state, "web_search"),
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[web_search] Failed: %s", msg)
        own_step["observation"] = f"Error: {msg}"
        elapsed_ms = int((time.time() - start_t) * 1000)
        _push_progress(session_id, {"agent": "web_search", "status": "error", "tickers": tickers, "elapsed_ms": elapsed_ms, "error": msg[:200]})
        if "timeout" in msg.lower():
            _log_telemetry(run_id, "web_search", "timeout", latency_ms=elapsed_ms,
                           complexity_declared=complexity, notes=msg[:500])
        else:
            _log_telemetry(run_id, "web_search", "latency", latency_ms=elapsed_ms,
                           complexity_declared=complexity, notes=f"error: {msg[:200]}")
        return {
            "react_steps": [own_step],
            "agent_errors": {"web_search": msg},
            "agent_react_iterations": _incr_agent_iter(state, "web_search"),
        }


# ── Node 5: Financial Modelling ───────────────────────────────────────────────

def node_financial_modelling(state: OrchestrationState) -> OrchestrationState:
    """Call the Financial Modelling agent (DCF/WACC/Comps/Technicals) for every ticker."""
    tickers = state.get("tickers") or []
    query   = state.get("user_query", "")
    outputs: List[Dict[str, Any]] = []

    # Fall back to legacy single ticker if list is empty
    if not tickers and state.get("ticker"):
        tickers = [state["ticker"]]  # type: ignore[list-item]

    session_id: str = state.get("session_id", "") or ""  # type: ignore[assignment]
    run_id: str = session_id or str(uuid.uuid4())[:8]
    complexity: int = state.get("react_max_iterations") or 1
    start_t = time.time()
    own_step: Dict[str, Any] = {"tool": "financial_modelling", "input": {"tickers": tickers, "query": query}}
    _push_progress(session_id, {"agent": "financial_modelling", "status": "started", "tickers": tickers, "elapsed_ms": 0, "error": None})

    logger.info("[financial_modelling] Running for tickers=%s", tickers)

    try:
        from agents.financial_modelling.agent import run_full_analysis  # type: ignore[import]

        for t in tickers:
            logger.info("[financial_modelling] Processing ticker=%s", t)
            result = run_full_analysis(ticker=str(t))
            outputs.append(result)

        obs_parts = [
            f"{r.get('ticker', '?')}:dcf_base={r.get('valuation', {}).get('dcf', {}).get('intrinsic_value_base')}"
            for r in outputs
        ]
        own_step["observation"] = "Success — " + " | ".join(obs_parts)
        logger.info("[financial_modelling] Done for %d ticker(s).", len(outputs))

        elapsed_ms = int((time.time() - start_t) * 1000)
        _excerpt = _extract_agent_excerpt("financial_modelling", outputs)
        _push_progress(session_id, {"agent": "financial_modelling", "status": "done", "tickers": tickers, "elapsed_ms": elapsed_ms, "error": None, "excerpt": _excerpt})
        _log_telemetry(run_id, "financial_modelling", "latency", latency_ms=elapsed_ms,
                       complexity_declared=complexity, react_loops_used=_incr_agent_iter(state, "financial_modelling").get("financial_modelling", 1))

        first = outputs[0] if outputs else None
        return {
            "financial_modelling_outputs": outputs,
            "financial_modelling_output": first,
            "react_steps": [own_step],
            "agent_errors": {},
            "agent_react_iterations": _incr_agent_iter(state, "financial_modelling"),
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[financial_modelling] Failed: %s", msg)
        own_step["observation"] = f"Error: {msg}"
        elapsed_ms = int((time.time() - start_t) * 1000)
        _push_progress(session_id, {"agent": "financial_modelling", "status": "error", "tickers": tickers, "elapsed_ms": elapsed_ms, "error": msg[:200]})
        if "timeout" in msg.lower():
            _log_telemetry(run_id, "financial_modelling", "timeout", latency_ms=elapsed_ms,
                           complexity_declared=complexity, notes=msg[:500])
        else:
            _log_telemetry(run_id, "financial_modelling", "latency", latency_ms=elapsed_ms,
                           complexity_declared=complexity, notes=f"error: {msg[:200]}")
        return {
            "react_steps": [own_step],
            "agent_errors": {"financial_modelling": msg},
            "agent_react_iterations": _incr_agent_iter(state, "financial_modelling"),
        }


# ── Node 5b: Stock Research ──────────────────────────────────────────────────

def node_stock_research(state: OrchestrationState) -> OrchestrationState:
    """Call the Stock Research agent (PDF broker reports + earnings transcripts) for every ticker.

    This node is only reached when the planner explicitly sets run_stock_research=True,
    typically for queries that ask about broker reports, analyst ratings, earnings call
    tone, or transcript-level signals.
    """
    tickers = state.get("tickers") or []
    outputs: List[Dict[str, Any]] = []

    if not tickers and state.get("ticker"):
        tickers = [state["ticker"]]  # type: ignore[list-item]

    session_id: str = state.get("session_id", "") or ""  # type: ignore[assignment]
    run_id: str = session_id or str(uuid.uuid4())[:8]
    complexity: int = state.get("react_max_iterations") or 1
    start_t = time.time()
    own_step: Dict[str, Any] = {"tool": "stock_research", "input": {"tickers": tickers}}
    _push_progress(session_id, {"agent": "stock_research", "status": "started", "tickers": tickers, "elapsed_ms": 0, "error": None})

    logger.info("[stock_research] Running for tickers=%s", tickers)

    try:
        from agents.stock_research_agent.agent import run_full_analysis  # type: ignore[import]

        for t in tickers:
            logger.info("[stock_research] Processing ticker=%s", t)
            result = run_full_analysis(ticker=str(t))
            outputs.append(result)

        obs_parts = [
            f"{r.get('ticker', '?')}:brokers={len(r.get('broker_parsed', []))}"
            for r in outputs
        ]
        own_step["observation"] = "Success — " + " | ".join(obs_parts)
        logger.info("[stock_research] Done for %d ticker(s).", len(outputs))

        elapsed_ms = int((time.time() - start_t) * 1000)
        _excerpt = _extract_agent_excerpt("stock_research", outputs)
        _push_progress(session_id, {"agent": "stock_research", "status": "done", "tickers": tickers, "elapsed_ms": elapsed_ms, "error": None, "excerpt": _excerpt})
        _log_telemetry(run_id, "stock_research", "latency", latency_ms=elapsed_ms,
                       complexity_declared=complexity, react_loops_used=_incr_agent_iter(state, "stock_research").get("stock_research", 1))

        first = outputs[0] if outputs else None
        return {
            "stock_research_outputs": outputs,
            "stock_research_output": first,
            "react_steps": [own_step],
            "agent_errors": {},
            "agent_react_iterations": _incr_agent_iter(state, "stock_research"),
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[stock_research] Failed: %s", msg)
        own_step["observation"] = f"Error: {msg}"
        elapsed_ms = int((time.time() - start_t) * 1000)
        _push_progress(session_id, {"agent": "stock_research", "status": "error", "tickers": tickers, "elapsed_ms": elapsed_ms, "error": msg[:200]})
        if "timeout" in msg.lower():
            _log_telemetry(run_id, "stock_research", "timeout", latency_ms=elapsed_ms,
                           complexity_declared=complexity, notes=msg[:500])
        else:
            _log_telemetry(run_id, "stock_research", "latency", latency_ms=elapsed_ms,
                           complexity_declared=complexity, notes=f"error: {msg[:200]}")
        return {
            "react_steps": [own_step],
            "agent_errors": {"stock_research": msg},
            "agent_react_iterations": _incr_agent_iter(state, "stock_research"),
        }


def node_macro(state: OrchestrationState) -> OrchestrationState:
    """Call the Macro agent for every ticker.

    This node is only reached when the planner explicitly sets run_macro=True,
    typically for queries that involve macro-economic analysis, market regime,
    or macroeconomic themes.
    """
    tickers = state.get("tickers") or []

    if not tickers and state.get("ticker"):
        tickers = [state.get("ticker")]

    session_id: str = state.get("session_id", "") or ""
    run_id: str = session_id or str(uuid.uuid4())[:8]
    complexity: int = state.get("react_max_iterations") or 1
    start_t = time.time()
    own_step: Dict[str, Any] = {"tool": "macro", "input": {"tickers": tickers}}
    _push_progress(session_id, {"agent": "macro", "status": "started", "tickers": tickers, "elapsed_ms": 0, "error": None})

    logger.info("[macro] Running for tickers=%s", tickers)

    try:
        from agents.macro_agent.agent import run_full_analysis  # type: ignore[import]

        outputs: List[Dict[str, Any]] = []
        for t in tickers:
            logger.info("[macro] Processing ticker=%s", t)
            result = run_full_analysis(ticker=str(t))
            outputs.append(result)

        obs_parts = [
            f"{r.get('ticker', '?')}:themes={len(r.get('macro_themes', []))}"
            for r in outputs
        ]
        own_step["observation"] = "Success — " + " | ".join(obs_parts)
        logger.info("[macro] Done for %d ticker(s).", len(outputs))

        elapsed_ms = int((time.time() - start_t) * 1000)
        _excerpt = _extract_agent_excerpt("macro", outputs)
        _push_progress(session_id, {"agent": "macro", "status": "done", "tickers": tickers, "elapsed_ms": elapsed_ms, "error": None, "excerpt": _excerpt})
        _log_telemetry(run_id, "macro", "latency", latency_ms=elapsed_ms,
                       complexity_declared=complexity, react_loops_used=_incr_agent_iter(state, "macro").get("macro", 1))

        first = outputs[0] if outputs else None
        return {
            "macro_outputs": outputs,
            "macro_output": first,
            "react_steps": [own_step],
            "agent_errors": {},
            "agent_react_iterations": _incr_agent_iter(state, "macro"),
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[macro] Failed: %s", msg)
        own_step["observation"] = f"Error: {msg}"
        elapsed_ms = int((time.time() - start_t) * 1000)
        _push_progress(session_id, {"agent": "macro", "status": "error", "tickers": tickers, "elapsed_ms": elapsed_ms, "error": msg[:200]})
        if "timeout" in msg.lower():
            _log_telemetry(run_id, "macro", "timeout", latency_ms=elapsed_ms,
                           complexity_declared=complexity, notes=msg[:500])
        else:
            _log_telemetry(run_id, "macro", "latency", latency_ms=elapsed_ms,
                           complexity_declared=complexity, notes=f"error: {msg[:200]}")
        return {
            "react_steps": [own_step],
            "agent_errors": {"macro": msg},
            "agent_react_iterations": _incr_agent_iter(state, "macro"),
        }


def node_insider_news(state: OrchestrationState) -> OrchestrationState:
    """Call the Insider News agent for every ticker.

    This node is only reached when the planner explicitly sets run_insider_news=True,
    typically for queries that involve insider trading activity or news sentiment analysis.
    """
    tickers = state.get("tickers") or []

    if not tickers and state.get("ticker"):
        tickers = [state.get("ticker")]

    session_id: str = state.get("session_id", "") or ""
    run_id: str = session_id or str(uuid.uuid4())[:8]
    complexity: int = state.get("react_max_iterations") or 1
    start_t = time.time()
    own_step: Dict[str, Any] = {"tool": "insider_news", "input": {"tickers": tickers}}
    _push_progress(session_id, {"agent": "insider_news", "status": "started", "tickers": tickers, "elapsed_ms": 0, "error": None})

    logger.info("[insider_news] Running for tickers=%s", tickers)

    try:
        from agents.insider_news_agent.agent import run_full_analysis  # type: ignore[import]

        outputs: List[Dict[str, Any]] = []
        for t in tickers:
            logger.info("[insider_news] Processing ticker=%s", t)
            result = run_full_analysis(ticker=str(t))
            outputs.append(result)

        obs_parts = [
            f"{r.get('ticker', '?')}:insiders={r.get('data_coverage', {}).get('insider_transactions_count', 0)}"
            for r in outputs
        ]
        own_step["observation"] = "Success — " + " | ".join(obs_parts)
        logger.info("[insider_news] Done for %d ticker(s).", len(outputs))

        elapsed_ms = int((time.time() - start_t) * 1000)
        _excerpt = _extract_agent_excerpt("insider_news", outputs)
        _push_progress(session_id, {"agent": "insider_news", "status": "done", "tickers": tickers, "elapsed_ms": elapsed_ms, "error": None, "excerpt": _excerpt})
        _log_telemetry(run_id, "insider_news", "latency", latency_ms=elapsed_ms,
                       complexity_declared=complexity, react_loops_used=_incr_agent_iter(state, "insider_news").get("insider_news", 1))

        first = outputs[0] if outputs else None
        return {
            "insider_news_outputs": outputs,
            "insider_news_output": first,
            "react_steps": [own_step],
            "agent_errors": {},
            "agent_react_iterations": _incr_agent_iter(state, "insider_news"),
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[insider_news] Failed: %s", msg)
        own_step["observation"] = f"Error: {msg}"
        elapsed_ms = int((time.time() - start_t) * 1000)
        _push_progress(session_id, {"agent": "insider_news", "status": "error", "tickers": tickers, "elapsed_ms": elapsed_ms, "error": msg[:200]})
        if "timeout" in msg.lower():
            _log_telemetry(run_id, "insider_news", "timeout", latency_ms=elapsed_ms,
                           complexity_declared=complexity, notes=msg[:500])
        else:
            _log_telemetry(run_id, "insider_news", "latency", latency_ms=elapsed_ms,
                           complexity_declared=complexity, notes=f"error: {msg[:200]}")
        return {
            "react_steps": [own_step],
            "agent_errors": {"insider_news": msg},
            "agent_react_iterations": _incr_agent_iter(state, "insider_news"),
        }


# (node_parallel_agents and node_react_check removed — replaced by LangGraph native
#  fan-out edges in graph.py and per-agent conditional retry edges.)

# ── Node 7: Summarizer ────────────────────────────────────────────────────────

def node_summarizer(state: OrchestrationState) -> OrchestrationState:
    """Use local DeepSeek to synthesise all agent outputs into a final report."""
    user_query    = state.get("user_query", "")
    tickers       = state.get("tickers") or []
    ticker        = state.get("ticker")
    output_language = state.get("output_language")

    # Prefer the multi-output lists; fall back to legacy single-output aliases.
    # Cast to List[Dict[str, Any]] by filtering out any None values.
    _ba_single   = state.get("business_analyst_output")
    _qf_single   = state.get("quant_fundamental_output")
    _ws_single   = state.get("web_search_output")
    _fm_single   = state.get("financial_modelling_output")

    ba_outputs: List[Dict[str, Any]]    = list(state.get("business_analyst_outputs") or (
        [_ba_single] if _ba_single else []
    ))
    quant_outputs: List[Dict[str, Any]] = list(state.get("quant_fundamental_outputs") or (
        [_qf_single] if _qf_single else []
    ))
    web_outputs: List[Dict[str, Any]]   = list(state.get("web_search_outputs") or (
        [_ws_single] if _ws_single else []
    ))
    fm_outputs: List[Dict[str, Any]]    = list(state.get("financial_modelling_outputs") or (
        [_fm_single] if _fm_single else []
    ))
    sr_outputs: List[Dict[str, Any]]    = list(state.get("stock_research_outputs") or [])
    macro_outputs: List[Dict[str, Any]] = list(state.get("macro_outputs") or [])
    insider_news_outputs: List[Dict[str, Any]] = list(state.get("insider_news_outputs") or [])
    errors        = state.get("agent_errors") or {}

    data_availability = state.get("data_availability")

    # Log a clear ReAct execution trace before summarising
    agent_results_summary = {
        "business_analyst":   f"{len(ba_outputs)} result(s)" if ba_outputs else "not run / no output",
        "quant_fundamental":  f"{len(quant_outputs)} result(s)" if quant_outputs else "not run / no output",
        "web_search":         f"{len(web_outputs)} result(s)" if web_outputs else "not run / no output",
        "financial_modelling":f"{len(fm_outputs)} result(s)" if fm_outputs else "not run / no output",
        "stock_research":    f"{len(sr_outputs)} result(s)" if sr_outputs else "not run / no output",
        "macro":             f"{len(macro_outputs)} result(s)" if macro_outputs else "not run / no output",
        "insider_news":      f"{len(insider_news_outputs)} result(s)" if insider_news_outputs else "not run / no output",
    }
    logger.info(
        "[summarizer] ReAct execution complete. tickers=%s  agent_outputs=%s  errors=%s",
        tickers, agent_results_summary, list(errors.keys()) or "none",
    )

    # Warn loudly if an agent was supposed to run but produced no output
    if state.get("run_business_analyst") and not ba_outputs:
        logger.warning("[summarizer] business_analyst was enabled but produced no output — "
                       "check agent_errors for crash details.")
    if state.get("run_quant_fundamental") and not quant_outputs:
        logger.warning("[summarizer] quant_fundamental was enabled but produced no output — "
                       "check agent_errors for crash details.")
    if state.get("run_financial_modelling") and not fm_outputs:
        logger.warning("[summarizer] financial_modelling was enabled but produced no output — "
                       "check agent_errors for crash details.")
    if state.get("run_macro") and not macro_outputs:
        logger.warning("[summarizer] macro was enabled but produced no output — "
                       "check agent_errors for crash details.")
    if state.get("run_insider_news") and not insider_news_outputs:
        logger.warning("[summarizer] insider_news was enabled but produced no output — "
                       "check agent_errors for crash details.")

    summarizer_trace_out: List[str] = []
    final_summary = summarise_results_structured(
        user_query=user_query,
        tickers=tickers,
        ba_outputs=ba_outputs,
        quant_outputs=quant_outputs,
        web_outputs=web_outputs,
        fm_outputs=fm_outputs,
        sr_outputs=sr_outputs,
        macro_outputs=macro_outputs,
        insider_news_outputs=insider_news_outputs,
        data_availability=data_availability,
        output_language=output_language,
        _trace_out=summarizer_trace_out,
    )

    # Final safeguard: ensure UI-selected output_language is respected even if
    # the structured summarizer path returned untranslated content.
    if output_language:
        try:
            from .llm import translate_text  # type: ignore[import]
            final_summary = translate_text(final_summary, str(output_language))
        except Exception as exc:
            logger.warning("[summarizer] Final translation safeguard failed: %s", exc)
    summarizer_trace: str = summarizer_trace_out[0] if summarizer_trace_out else ""
    if summarizer_trace:
        logger.info("[summarizer] Thinking trace captured (%d chars).", len(summarizer_trace))

    # Push __done__ sentinel so the UI progress consumer knows all agents finished
    session_id_sum: str = state.get("session_id", "") or ""  # type: ignore[assignment]
    _push_progress(session_id_sum, {"agent": "__done__", "status": "done", "tickers": tickers, "elapsed_ms": 0, "error": None})

    output: Dict[str, Any] = {
        "user_query": user_query,
        "ticker": ticker,
        "tickers": tickers,
        "plan": state.get("plan"),
        "react_steps": state.get("react_steps", []),
        # Multi-ticker lists
        "business_analyst_outputs": ba_outputs,
        "quant_fundamental_outputs": quant_outputs,
        "web_search_outputs": web_outputs,
        "financial_modelling_outputs": fm_outputs,
        "stock_research_outputs": sr_outputs,
        "macro_outputs": macro_outputs,
        "insider_news_outputs": insider_news_outputs,
        # Legacy single aliases (first ticker)
        "business_analyst_output": ba_outputs[0] if ba_outputs else None,
        "quant_fundamental_output": quant_outputs[0] if quant_outputs else None,
        "web_search_output": web_outputs[0] if web_outputs else None,
        "financial_modelling_output": fm_outputs[0] if fm_outputs else None,
        "stock_research_output": sr_outputs[0] if sr_outputs else None,
        "macro_output": macro_outputs[0] if macro_outputs else None,
        "insider_news_output": insider_news_outputs[0] if insider_news_outputs else None,
        "agent_errors": errors,
        "final_summary": final_summary,
        "planner_trace": state.get("planner_trace", ""),
        "summarizer_trace": summarizer_trace,
        "output_language": output_language,
    }

    return {**state, "final_summary": final_summary, "summarizer_trace": summarizer_trace, "output": output}


# ── Node 7b: Post-Processing (RLAIF scoring + episodic memory) ────────────────

def node_post_processing(state: OrchestrationState) -> OrchestrationState:
    """Combined post-processing: RLAIF scoring + episodic memory persistence.

    Replaces the old node_rlaif_scorer + node_memory_update + node_translator trio.
    Translation is now handled inside summarise_results_structured (Stage 4).

    1. Score the final report via DeepSeek RLAIF judge.
    2. Persist any agent failure patterns to agent_episodic_memory PostgreSQL table.
    """
    import uuid as _uuid

    final_summary: str = cast(str, state.get("final_summary", "") or "")
    user_query    = state.get("user_query", "")
    ticker        = state.get("ticker")
    tickers       = state.get("tickers") or []
    errors        = state.get("agent_errors") or {}
    run_id        = state.get("rl_feedback_run_id") or str(_uuid.uuid4())[:12]

    # ── Part 1: RLAIF scoring ─────────────────────────────────────────────────
    agent_outputs = {
        "business_analyst_output":   state.get("business_analyst_output"),
        "quant_fundamental_output":  state.get("quant_fundamental_output"),
        "financial_modelling_output":state.get("financial_modelling_output"),
        "web_search_output":         state.get("web_search_output"),
    }

    logger.info("[post_processing] Scoring report for run_id=%s", run_id)
    scores = None
    try:
        scores = feedback.score_report_with_rlaif(
            run_id=run_id,
            user_query=user_query,
            final_summary=final_summary,
            agent_outputs=agent_outputs,
            ticker=ticker,
        )
        logger.info(
            "[post_processing] RLAIF scores: overall=%.2f, factual=%.2f, citation=%.2f, "
            "analysis=%.2f, structure=%.2f, language=%.2f",
            scores.get("overall_score", 0),
            scores.get("factual_accuracy", 0),
            scores.get("citation_completeness", 0),
            scores.get("analysis_depth", 0),
            scores.get("structure_compliance", 0),
            scores.get("language_quality", 0),
        )
    except Exception as exc:
        logger.warning("[post_processing] RLAIF scoring failed (non-fatal): %s", exc)

    # ── Part 2: Episodic memory persistence ───────────────────────────────────
    # Use per-agent iteration counts from the new agent_react_iterations dict;
    # fall back to the legacy react_iteration counter for backward compatibility.
    agent_iters = state.get("agent_react_iterations") or {}
    react_iters = state.get("react_iteration") or 1

    gap_agents: List[str] = []
    if state.get("run_business_analyst") and not state.get("business_analyst_outputs"):
        gap_agents.append("business_analyst")
    if state.get("run_quant_fundamental") and not state.get("quant_fundamental_outputs"):
        gap_agents.append("quant_fundamental")
    if state.get("run_web_search") and not state.get("web_search_outputs"):
        gap_agents.append("web_search")
    if state.get("run_financial_modelling") and not state.get("financial_modelling_outputs"):
        gap_agents.append("financial_modelling")
    if state.get("run_stock_research") and not state.get("stock_research_outputs"):
        gap_agents.append("stock_research")
    if state.get("run_macro") and not state.get("macro_outputs"):
        gap_agents.append("macro")
    if state.get("run_insider_news") and not state.get("insider_news_outputs"):
        gap_agents.append("insider_news")

    error_agents      = list(errors.keys())
    agents_to_record  = list(dict.fromkeys(gap_agents + error_agents))

    if agents_to_record:
        try:
            from .episodic_memory import record_failure  # type: ignore[import]
            for agent_name in agents_to_record:
                reason = "INSUFFICIENT_DATA" if agent_name in gap_agents else "ERROR"
                if agent_name in errors:
                    err_msg = errors[agent_name]
                    if "timeout" in err_msg.lower():
                        reason = "TIMEOUT"
                    elif "insufficient" in err_msg.lower() or "no data" in err_msg.lower():
                        reason = "INSUFFICIENT_DATA"
                    else:
                        reason = "ERROR"
                # Use per-agent iteration count if available, else global
                iters_used = agent_iters.get(agent_name, react_iters)
                for t in (tickers or ["UNKNOWN"]):
                    record_failure(
                        user_query=user_query,
                        ticker=t,
                        failure_agent=agent_name,
                        failure_reason=reason,
                        react_iterations_used=iters_used,
                    )
            logger.info(
                "[post_processing] Recorded %d failure event(s) for agents=%s tickers=%s",
                len(agents_to_record) * max(len(tickers), 1),
                agents_to_record, tickers,
            )
        except Exception as exc:
            logger.warning("[post_processing] Episodic memory persistence failed (non-fatal): %s", exc)
    else:
        logger.debug("[post_processing] No failures to record.")

    return {
        **state,
        "rl_feedback_scores": scores,
        "rl_feedback_run_id": run_id,
    }


__all__ = [
    "node_planner",
    "node_business_analyst",
    "node_quant_fundamental",
    "node_web_search",
    "node_financial_modelling",
    "node_stock_research",
    "node_summarizer",
    "node_post_processing",
    "subscribe_agent_progress",
    "unsubscribe_agent_progress",
]
