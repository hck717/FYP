"""Node implementations for the orchestration LangGraph pipeline.

Each function is a pure node — takes state, returns partial state update.

Pipeline:
  planner → parallel_agents → react_check → (loop | summarizer) → memory_update → END

  parallel_agents fans out BA, QF, FM (and optionally WS) concurrently using
  a ThreadPoolExecutor so all enabled agents run at the same time.  Wall-clock
  time is bounded by the slowest single agent rather than their sum.

  node_memory_update runs after summarizer to persist any failure patterns to
  the agent_episodic_memory PostgreSQL table for use in future runs.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, cast

from .llm import plan_query, summarise_results
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

    plan = plan_query(user_query)

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

    # complexity drives max ReAct passes — no upper cap
    raw_complexity = plan.get("complexity", 2)
    try:
        react_max = max(1, int(raw_complexity))
    except (TypeError, ValueError):
        react_max = 2

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
        "business_analyst_output": None,
        "quant_fundamental_output": None,
        "web_search_output": None,
        "financial_modelling_output": None,
    }


# ── Node 2: Business Analyst ──────────────────────────────────────────────────

def node_business_analyst(state: OrchestrationState) -> OrchestrationState:
    """Call the Business Analyst CRAG pipeline for every resolved ticker."""
    tickers  = state.get("tickers") or []
    query    = state.get("user_query", "")
    steps    = list(state.get("react_steps") or [])
    errors   = dict(state.get("agent_errors") or {})
    outputs: List[Dict[str, Any]] = []

    # Fall back to legacy single ticker if list is empty
    if not tickers and state.get("ticker"):
        tickers = [state["ticker"]]  # type: ignore[list-item]

    logger.info("[business_analyst] Running for tickers=%s", tickers)
    steps.append({"tool": "business_analyst", "input": {"tickers": tickers, "query": query}})

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
        steps[-1]["observation"] = "Success — " + " | ".join(obs_parts)
        logger.info("[business_analyst] Done for %d ticker(s).", len(outputs))

        first = outputs[0] if outputs else None
        return {
            **state,
            "business_analyst_outputs": outputs,
            "business_analyst_output": first,
            "react_steps": steps,
            "agent_errors": errors,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[business_analyst] Failed: %s", msg)
        steps[-1]["observation"] = f"Error: {msg}"
        errors["business_analyst"] = msg
        return {**state, "react_steps": steps, "agent_errors": errors}


# ── Node 3: Quant Fundamental ─────────────────────────────────────────────────

def node_quant_fundamental(state: OrchestrationState) -> OrchestrationState:
    """Call the Quantitative Fundamental pipeline for every resolved ticker."""
    tickers  = state.get("tickers") or []
    query    = state.get("user_query", "")
    steps    = list(state.get("react_steps") or [])
    errors   = dict(state.get("agent_errors") or {})
    outputs: List[Dict[str, Any]] = []

    # Fall back to legacy single ticker if list is empty
    if not tickers and state.get("ticker"):
        tickers = [state["ticker"]]  # type: ignore[list-item]

    logger.info("[quant_fundamental] Running for tickers=%s", tickers)
    steps.append({"tool": "quant_fundamental", "input": {"tickers": tickers, "query": query}})

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
        steps[-1]["observation"] = "Success — " + " | ".join(obs_parts)
        logger.info("[quant_fundamental] Done for %d ticker(s).", len(outputs))

        first = outputs[0] if outputs else None
        return {
            **state,
            "quant_fundamental_outputs": outputs,
            "quant_fundamental_output": first,
            "react_steps": steps,
            "agent_errors": errors,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[quant_fundamental] Failed: %s", msg)
        steps[-1]["observation"] = f"Error: {msg}"
        errors["quant_fundamental"] = msg
        return {**state, "react_steps": steps, "agent_errors": errors}


# ── Node 4: Web Search ────────────────────────────────────────────────────────

def node_web_search(state: OrchestrationState) -> OrchestrationState:
    """Call the Web Search agent (Perplexity Sonar) for every resolved ticker.

    This node is only reached when the planner explicitly sets run_web_search=True.
    For multi-ticker queries, one search is performed per ticker.
    """
    tickers  = state.get("tickers") or []
    query    = state.get("user_query", "")
    steps    = list(state.get("react_steps") or [])
    errors   = dict(state.get("agent_errors") or {})
    outputs: List[Dict[str, Any]] = []

    # Fall back to legacy single ticker if list is empty
    if not tickers and state.get("ticker"):
        tickers = [state["ticker"]]  # type: ignore[list-item]
    # If still no tickers, do a single query-only search
    if not tickers:
        tickers = [None]  # type: ignore[list-item]

    logger.info("[web_search] Running for tickers=%s", tickers)
    steps.append({"tool": "web_search", "input": {"tickers": tickers, "query": query}})

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
        steps[-1]["observation"] = "Success — " + " | ".join(obs_parts)
        logger.info("[web_search] Done for %d ticker(s).", len(outputs))

        first = outputs[0] if outputs else None
        return {
            **state,
            "web_search_outputs": outputs,
            "web_search_output": first,
            "react_steps": steps,
            "agent_errors": errors,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[web_search] Failed: %s", msg)
        steps[-1]["observation"] = f"Error: {msg}"
        errors["web_search"] = msg
        return {**state, "react_steps": steps, "agent_errors": errors}


# ── Node 5: Financial Modelling ───────────────────────────────────────────────

def node_financial_modelling(state: OrchestrationState) -> OrchestrationState:
    """Call the Financial Modelling agent (DCF/WACC/Comps/Technicals) for every ticker."""
    tickers = state.get("tickers") or []
    query   = state.get("user_query", "")
    steps   = list(state.get("react_steps") or [])
    errors  = dict(state.get("agent_errors") or {})
    outputs: List[Dict[str, Any]] = []

    # Fall back to legacy single ticker if list is empty
    if not tickers and state.get("ticker"):
        tickers = [state["ticker"]]  # type: ignore[list-item]

    logger.info("[financial_modelling] Running for tickers=%s", tickers)
    steps.append({"tool": "financial_modelling", "input": {"tickers": tickers, "query": query}})

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
        steps[-1]["observation"] = "Success — " + " | ".join(obs_parts)
        logger.info("[financial_modelling] Done for %d ticker(s).", len(outputs))

        first = outputs[0] if outputs else None
        return {
            **state,
            "financial_modelling_outputs": outputs,
            "financial_modelling_output": first,
            "react_steps": steps,
            "agent_errors": errors,
        }

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.error("[financial_modelling] Failed: %s", msg)
        steps[-1]["observation"] = f"Error: {msg}"
        errors["financial_modelling"] = msg
        return {**state, "react_steps": steps, "agent_errors": errors}


# ── Node 6: Parallel agent dispatcher ────────────────────────────────────────

def node_parallel_agents(state: OrchestrationState) -> OrchestrationState:
    """Run all enabled agents (BA, QF, FM, WS) concurrently via ThreadPoolExecutor.

    Wall-clock time is bounded by the slowest individual agent rather than
    their sum.  Each agent still iterates over all tickers internally.

    Error isolation: a failure in one agent never cancels the others.  Errors
    are captured in ``agent_errors`` and the remaining outputs are passed to
    the summarizer.
    """
    tickers = state.get("tickers") or []
    query   = state.get("user_query", "")
    errors: Dict[str, str] = dict(state.get("agent_errors") or {})
    steps:  List[Dict[str, Any]] = list(state.get("react_steps") or [])

    run_ba = state.get("run_business_analyst", False)
    run_qf = state.get("run_quant_fundamental", False)
    run_ws = state.get("run_web_search", False)
    run_fm = state.get("run_financial_modelling", False)

    # On ReAct retry passes: skip agents that already produced output — only
    # re-run agents that have no results yet (gaps) or had their errors cleared
    # by node_react_check.  This prevents successful agents from running twice.
    already_ba = bool(state.get("business_analyst_outputs"))
    already_qf = bool(state.get("quant_fundamental_outputs"))
    already_ws = bool(state.get("web_search_outputs"))
    already_fm = bool(state.get("financial_modelling_outputs"))

    dispatch_ba = run_ba and not already_ba
    dispatch_qf = run_qf and not already_qf
    dispatch_ws = run_ws and not already_ws
    dispatch_fm = run_fm and not already_fm

    iteration = state.get("react_iteration", 0)
    logger.info(
        "[parallel_agents] pass=%d  dispatching: ba=%s qf=%s ws=%s fm=%s  tickers=%s",
        iteration, dispatch_ba, dispatch_qf, dispatch_ws, dispatch_fm, tickers,
    )

    # ── Build task list ───────────────────────────────────────────────────────
    # Each task is a (name, callable) pair.  The callable receives no args —
    # all context is captured via closure over `state`.

    data_availability = state.get("data_availability")

    def _run_ba() -> List[Dict[str, Any]]:
        from agents.business_analyst.agent import run_full_analysis  # type: ignore[import]
        from .data_availability import ticker_data_profile            # type: ignore[import]
        outputs: List[Dict[str, Any]] = []
        for t in (tickers or [None]):  # type: ignore[list-item]
            profile = None
            if data_availability and t:
                try:
                    profile = ticker_data_profile(data_availability, t)
                except Exception:
                    pass
            if t:
                outputs.append(run_full_analysis(ticker=t, availability_profile=profile))
            else:
                from agents.business_analyst.agent import run as _ba_run  # type: ignore[import]
                outputs.append(_ba_run(task=query, ticker=None))
        return outputs

    def _run_qf() -> List[Dict[str, Any]]:
        from agents.quant_fundamental.agent import run_full_analysis  # type: ignore[import]
        from .data_availability import ticker_data_profile             # type: ignore[import]
        outputs: List[Dict[str, Any]] = []
        if tickers:
            for t in tickers:
                profile = None
                if data_availability and t:
                    try:
                        profile = ticker_data_profile(data_availability, t)
                    except Exception:
                        pass
                result = run_full_analysis(ticker=str(t), availability_profile=profile)
                if isinstance(result, list):
                    outputs.extend(result)
                else:
                    outputs.append(result)
        else:
            result = run_full_analysis(prompt=query)
            if isinstance(result, list):
                outputs.extend(result)
            else:
                outputs.append(result)
        return outputs

    def _run_ws() -> List[Dict[str, Any]]:
        from agents.web_search.agent import run_web_search_agent, WebSearchInput  # type: ignore[import]
        outputs: List[Dict[str, Any]] = []
        targets = tickers or [None]  # type: ignore[list-item]
        for t in targets:
            agent_input = WebSearchInput(
                query=query,
                ticker=t,
                recency_filter="week",
                model=None,
            )
            outputs.append(dict(run_web_search_agent(agent_input)))
        return outputs

    def _run_fm() -> List[Dict[str, Any]]:
        from agents.financial_modelling.agent import run_full_analysis  # type: ignore[import]
        outputs: List[Dict[str, Any]] = []
        for t in (tickers or []):
            outputs.append(run_full_analysis(ticker=str(t)))
        return outputs

    tasks: Dict[str, Any] = {}
    if dispatch_ba:
        tasks["business_analyst"] = _run_ba
    if dispatch_qf:
        tasks["quant_fundamental"] = _run_qf
    if dispatch_ws:
        tasks["web_search"] = _run_ws
    if dispatch_fm:
        tasks["financial_modelling"] = _run_fm

    # ── Execute concurrently ──────────────────────────────────────────────────
    # Seed with whatever outputs already exist from previous passes
    ba_outputs: List[Dict[str, Any]]  = list(state.get("business_analyst_outputs") or [])
    qf_outputs: List[Dict[str, Any]]  = list(state.get("quant_fundamental_outputs") or [])
    ws_outputs: List[Dict[str, Any]]  = list(state.get("web_search_outputs") or [])
    fm_outputs: List[Dict[str, Any]]  = list(state.get("financial_modelling_outputs") or [])

    # A2: derive a stable run_id for telemetry (session_id set by UI, else generate one)
    run_id: str = state.get("session_id", "") or str(uuid.uuid4())[:8]  # type: ignore[arg-type]
    complexity: int = state.get("react_max_iterations") or 1

    # Live-progress queue (subscribed by the Streamlit UI before the graph runs)
    _pq = _progress_queues.get(run_id)

    def _push(event: Dict[str, Any]) -> None:
        if _pq is not None:
            try:
                _pq.put_nowait(event)
            except Exception:
                pass

    if not tasks:
        logger.warning("[parallel_agents] No agents enabled — proceeding to summarizer.")
    else:
        n_workers = len(tasks)  # one thread per agent
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="agent") as pool:
            # A2: record start time per future so we can measure latency
            future_to_name: Dict[Any, str] = {}
            future_to_start: Dict[Any, float] = {}
            for name, fn in tasks.items():
                fut = pool.submit(fn)
                future_to_name[fut] = name
                future_to_start[fut] = time.time()
                _push({"agent": name, "status": "started", "tickers": tickers, "elapsed_ms": 0, "error": None})

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                elapsed_ms = int((time.time() - future_to_start[future]) * 1000)
                try:
                    result_list: List[Dict[str, Any]] = future.result()
                    obs_parts = [
                        f"{r.get('ticker', '?')}" for r in result_list
                    ]
                    steps.append({
                        "tool": name,
                        "input": {"tickers": tickers, "query": query},
                        "observation": f"Success — {', '.join(obs_parts)}",
                    })
                    logger.info("[parallel_agents] %s done (%d result(s)).", name, len(result_list))
                    _excerpt = _extract_agent_excerpt(name, result_list)
                    # Extract thinking trace from agent results for real-time display
                    _thinking_trace = None
                    if result_list and isinstance(result_list[0], dict):
                        _thinking_trace = result_list[0].get("thinking_trace") or []
                    _push({"agent": name, "status": "done", "tickers": tickers, "elapsed_ms": elapsed_ms, "error": None, "excerpt": _excerpt, "thinking_trace": _thinking_trace})
                    if name == "business_analyst":
                        ba_outputs = result_list
                        # A2: detect CRAG fallback
                        if result_list and result_list[0].get("crag_status") == "fallback":
                            _log_telemetry(
                                run_id, "business_analyst", "crag_fallback",
                                notes=f"crag_status=fallback ticker={result_list[0].get('ticker', '?')}",
                            )
                    elif name == "quant_fundamental":
                        qf_outputs = result_list
                    elif name == "web_search":
                        ws_outputs = result_list
                    elif name == "financial_modelling":
                        fm_outputs = result_list
                    # A2: log latency for every successful agent
                    _log_telemetry(
                        run_id, name, "latency",
                        latency_ms=elapsed_ms,
                        complexity_declared=complexity,
                        react_loops_used=iteration,
                    )
                except Exception as exc:
                    msg = f"{type(exc).__name__}: {exc}"
                    logger.error("[parallel_agents] %s failed: %s", name, msg)
                    errors[name] = msg
                    _push({"agent": name, "status": "error", "tickers": tickers, "elapsed_ms": elapsed_ms, "error": msg[:200]})
                    steps.append({
                        "tool": name,
                        "input": {"tickers": tickers, "query": query},
                        "observation": f"Error: {msg}",
                    })
                    # A2: detect timeout events
                    if "timeout" in msg.lower():
                        _log_telemetry(
                            run_id, name, "timeout",
                            latency_ms=elapsed_ms,
                            complexity_declared=complexity,
                            react_loops_used=iteration,
                            notes=msg[:500],
                        )
                    else:
                        _log_telemetry(
                            run_id, name, "latency",
                            latency_ms=elapsed_ms,
                            complexity_declared=complexity,
                            react_loops_used=iteration,
                            notes=f"error: {msg[:200]}",
                        )

    # Sentinel: tell the UI that this pass of parallel_agents is fully complete
    _push({"agent": "__done__", "status": "done", "tickers": tickers, "elapsed_ms": 0, "error": None})

    return {
        **state,
        "business_analyst_outputs":   ba_outputs,
        "quant_fundamental_outputs":  qf_outputs,
        "web_search_outputs":         ws_outputs,
        "financial_modelling_outputs": fm_outputs,
        "business_analyst_output":    ba_outputs[0]  if ba_outputs  else None,
        "quant_fundamental_output":   qf_outputs[0]  if qf_outputs  else None,
        "web_search_output":          ws_outputs[0]  if ws_outputs  else None,
        "financial_modelling_output": fm_outputs[0]  if fm_outputs  else None,
        "react_steps": steps,
        "agent_errors": errors,
    }


# ── Node 6b: ReAct check ──────────────────────────────────────────────────────

def node_react_check(state: OrchestrationState) -> OrchestrationState:
    """Evaluate whether another agent pass adds value, or if we should proceed to summariser.

    This node runs after every ``node_parallel_agents`` pass.  It increments
    ``react_iteration`` and decides—via ``_should_loop`` in graph.py—whether to
    loop back for a re-run of failed/gap agents or to advance to the summariser.

    Decision criteria (stored in state so the conditional edge can read them):
      - Any agent that was enabled but produced NO output (gap) → prefer re-run
      - Any agent that raised an error (in ``agent_errors``) → prefer re-run
      - If ``react_iteration + 1 >= react_max_iterations`` → proceed to summariser
        regardless of gaps (safety cap from complexity scoring).

    The node itself does NOT re-enable or re-disable agents.  When the graph loops
    back to ``node_parallel_agents``, only agents that have NO outputs yet are
    re-dispatched (the dispatcher skips agents that already have results).  This
    means a successful agent is never run twice — only failed/empty ones retry.
    """
    iteration   = (state.get("react_iteration") or 0) + 1
    react_max   = state.get("react_max_iterations") or 1
    errors      = state.get("agent_errors") or {}

    # Identify agents with gaps: enabled but produced no output
    gaps: List[str] = []
    if state.get("run_business_analyst") and not state.get("business_analyst_outputs"):
        gaps.append("business_analyst")
    if state.get("run_quant_fundamental") and not state.get("quant_fundamental_outputs"):
        gaps.append("quant_fundamental")
    if state.get("run_web_search") and not state.get("web_search_outputs"):
        gaps.append("web_search")
    if state.get("run_financial_modelling") and not state.get("financial_modelling_outputs"):
        gaps.append("financial_modelling")

    # Agents that errored on this pass are also considered gaps for retry purposes
    error_agents = [a for a in errors if a in (
        "business_analyst", "quant_fundamental", "web_search", "financial_modelling"
    )]
    retry_agents = list(dict.fromkeys(gaps + error_agents))  # deduplicated, order-preserved

    should_loop = bool(retry_agents) and iteration < react_max

    logger.info(
        "[react_check] iteration=%d/%d  gaps=%s  errors=%s  → %s",
        iteration, react_max,
        gaps or "none",
        list(errors.keys()) or "none",
        "loop" if should_loop else "summarizer",
    )

    # A2: detect complexity mismatch — actual loops exceeded declared complexity
    complexity_declared = state.get("react_max_iterations") or 1
    run_id: str = state.get("session_id", "") or str(uuid.uuid4())[:8]  # type: ignore[arg-type]
    if iteration > complexity_declared:
        _log_telemetry(
            run_id, "orchestration", "complexity_mismatch",
            complexity_declared=complexity_declared,
            react_loops_used=iteration,
            notes=f"gaps={gaps} errors={list(errors.keys())}",
        )

    # C3: ReAct Loop Pruning — auto-disable agents with >3 timeouts in last 24h
    # Check agent_run_telemetry to see which agents have been repeatedly timing out.
    # If found, disable that agent for this run and log an auto_disabled event.
    _KNOWN_AGENTS = ("business_analyst", "quant_fundamental", "web_search", "financial_modelling")
    _TIMEOUT_PRUNE_THRESHOLD = int(os.getenv("REACT_PRUNE_TIMEOUT_THRESHOLD", "999"))

    def _count_recent_timeouts(agent_name: str) -> int:
        """Query agent_run_telemetry for timeout events in last 24h."""
        try:
            import psycopg2  # type: ignore[import]
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
                    SELECT COUNT(*) FROM agent_run_telemetry
                    WHERE agent_name = %s
                      AND event_type = 'timeout'
                      AND recorded_at >= NOW() - INTERVAL '24 hours'
                    """,
                    (agent_name,),
                )
                row = cur.fetchone()
                count = int(row[0]) if row else 0
            conn.close()
            return count
        except Exception as exc:
            logger.debug("[react_check/c3] Could not count timeouts for %s: %s", agent_name, exc)
            return 0

    # Only prune agents that are currently enabled (run_* flag = True) and have
    # no output yet (i.e. they would be dispatched again on the next loop pass).
    _run_flags = {
        "business_analyst":   "run_business_analyst",
        "quant_fundamental":  "run_quant_fundamental",
        "web_search":         "run_web_search",
        "financial_modelling":"run_financial_modelling",
    }
    _output_keys = {
        "business_analyst":   "business_analyst_outputs",
        "quant_fundamental":  "quant_fundamental_outputs",
        "web_search":         "web_search_outputs",
        "financial_modelling":"financial_modelling_outputs",
    }

    # Build a dict of state overrides for auto-disabled agents
    prune_overrides: Dict[str, Any] = {}
    for agent in _KNOWN_AGENTS:
        # Skip if agent is already disabled or already has output
        if not state.get(_run_flags[agent]):
            continue
        if state.get(_output_keys[agent]):
            continue
        timeout_count = _count_recent_timeouts(agent)
        if timeout_count > _TIMEOUT_PRUNE_THRESHOLD:
            logger.warning(
                "[react_check/c3] AUTO-DISABLING %s — %d timeout events in last 24h "
                "(threshold=%d).",
                agent, timeout_count, _TIMEOUT_PRUNE_THRESHOLD,
            )
            prune_overrides[_run_flags[agent]] = False
            _log_telemetry(
                run_id, agent, "auto_disabled",
                react_loops_used=iteration,
                notes=f"timeout_count_24h={timeout_count} threshold={_TIMEOUT_PRUNE_THRESHOLD}",
            )
            # Remove from retry_agents so we don't try to loop for it
            if agent in retry_agents:
                retry_agents.remove(agent)

    # Re-evaluate should_loop after pruning
    should_loop = bool(retry_agents) and iteration < react_max

    # Clear errors for agents that will be retried so they get a clean slate
    new_errors = {k: v for k, v in errors.items() if k not in retry_agents} if should_loop else errors

    return {
        **state,
        **prune_overrides,
        "react_iteration": iteration,
        "agent_errors": new_errors,
    }


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
    errors        = state.get("agent_errors") or {}

    data_availability = state.get("data_availability")

    # Log a clear ReAct execution trace before summarising
    agent_results_summary = {
        "business_analyst":   f"{len(ba_outputs)} result(s)" if ba_outputs else "not run / no output",
        "quant_fundamental":  f"{len(quant_outputs)} result(s)" if quant_outputs else "not run / no output",
        "web_search":         f"{len(web_outputs)} result(s)" if web_outputs else "not run / no output",
        "financial_modelling":f"{len(fm_outputs)} result(s)" if fm_outputs else "not run / no output",
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

    summarizer_trace_out: List[str] = []
    final_summary = summarise_results(
        user_query=user_query,
        tickers=tickers,
        ba_outputs=ba_outputs,
        quant_outputs=quant_outputs,
        web_outputs=web_outputs,
        fm_outputs=fm_outputs,
        data_availability=data_availability,
        _trace_out=summarizer_trace_out,
    )
    summarizer_trace: str = summarizer_trace_out[0] if summarizer_trace_out else ""
    if summarizer_trace:
        logger.info("[summarizer] Thinking trace captured (%d chars).", len(summarizer_trace))

    # NOTE: Translation is now handled by the translator agent (node_translator)
    # This keeps all agents working in English, with translation done at the end

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
        # Legacy single aliases (first ticker)
        "business_analyst_output": ba_outputs[0] if ba_outputs else None,
        "quant_fundamental_output": quant_outputs[0] if quant_outputs else None,
        "web_search_output": web_outputs[0] if web_outputs else None,
        "financial_modelling_output": fm_outputs[0] if fm_outputs else None,
        "agent_errors": errors,
        "final_summary": final_summary,
        "planner_trace": state.get("planner_trace", ""),
        "summarizer_trace": summarizer_trace,
        "output_language": output_language,
    }

    return {**state, "final_summary": final_summary, "summarizer_trace": summarizer_trace, "output": output}


# ── Node 7b: RLAIF Scorer ───────────────────────────────────────────────────

def node_rlaif_scorer(state: OrchestrationState) -> OrchestrationState:
    """Score the generated report using RLAIF (Reinforcement Learning from AI Feedback).
    
    After the summarizer generates a report, this node calls DeepSeek as a judge
    to score the report on multiple dimensions and stores the feedback for
    short-term learning.
    
    Scores are stored in the rl_feedback table for analysis.
    """
    import uuid
    
    final_summary = state.get("final_summary", "")
    user_query = state.get("user_query", "")
    ticker = state.get("ticker")
    run_id = state.get("rl_feedback_run_id") or str(uuid.uuid4())[:12]
    
    # Collect agent outputs for reference
    agent_outputs = {
        "business_analyst_output": state.get("business_analyst_output"),
        "quant_fundamental_output": state.get("quant_fundamental_output"),
        "financial_modelling_output": state.get("financial_modelling_output"),
        "web_search_output": state.get("web_search_output"),
    }
    
    logger.info("[rlaif_scorer] Scoring report for run_id=%s", run_id)
    
    try:
        scores = feedback.score_report_with_rlaif(
            run_id=run_id,
            user_query=user_query,
            final_summary=final_summary,
            agent_outputs=agent_outputs,
            ticker=ticker,
        )
        
        logger.info(
            "[rlaif_scorer] Scores: overall=%.2f, factual=%.2f, citation=%.2f, "
            "analysis=%.2f, structure=%.2f, language=%.2f",
            scores.get("overall_score", 0),
            scores.get("factual_accuracy", 0),
            scores.get("citation_completeness", 0),
            scores.get("analysis_depth", 0),
            scores.get("structure_compliance", 0),
            scores.get("language_quality", 0),
        )
        
        return {
            **state,
            "rl_feedback_scores": scores,
            "rl_feedback_run_id": run_id,
        }
        
    except Exception as exc:
        logger.warning("[rlaif_scorer] Failed to score report: %s", exc)
        return {
            **state,
            "rl_feedback_scores": None,
            "rl_feedback_run_id": run_id,
        }


# ── Node 8: Episodic Memory Update ───────────────────────────────────────────

def node_memory_update(state: OrchestrationState) -> OrchestrationState:
    """Persist failure patterns to agent_episodic_memory after the pipeline completes.

    Runs after the summarizer.  For each agent that was enabled but produced no
    output (gap) AND multiple ReAct iterations were consumed, we write a failure
    record keyed by the user query embedding + ticker so the planner can
    pre-empt the same failure on future identical queries.

    This node is a no-op if:
      - No agents failed (no errors, no gaps)
      - Only 1 ReAct iteration was consumed (single-pass failures may be transient)
      - The episodic_memory module is unavailable (graceful degradation)
    """
    errors        = state.get("agent_errors") or {}
    tickers       = state.get("tickers") or []
    user_query    = state.get("user_query", "")
    react_iters   = state.get("react_iteration") or 1

    # Identify agents that were enabled but yielded no output (persistent gaps)
    gap_agents: List[str] = []
    if state.get("run_business_analyst") and not state.get("business_analyst_outputs"):
        gap_agents.append("business_analyst")
    if state.get("run_quant_fundamental") and not state.get("quant_fundamental_outputs"):
        gap_agents.append("quant_fundamental")
    if state.get("run_web_search") and not state.get("web_search_outputs"):
        gap_agents.append("web_search")
    if state.get("run_financial_modelling") and not state.get("financial_modelling_outputs"):
        gap_agents.append("financial_modelling")

    # Also include agents that ended with an error
    error_agents = list(errors.keys())
    agents_to_record = list(dict.fromkeys(gap_agents + error_agents))

    if not agents_to_record:
        logger.debug("[memory_update] No failures to record.")
        return state

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

            for ticker in (tickers or ["UNKNOWN"]):
                record_failure(
                    user_query=user_query,
                    ticker=ticker,
                    failure_agent=agent_name,
                    failure_reason=reason,
                    react_iterations_used=react_iters,
                )

        logger.info(
            "[memory_update] Recorded %d failure event(s) for agents=%s tickers=%s",
            len(agents_to_record) * max(len(tickers), 1),
            agents_to_record,
            tickers,
        )
    except Exception as exc:
        logger.warning("[memory_update] Failed to persist episodic memory (non-fatal): %s", exc)

    return state


# ── Node 9: Translator Agent ─────────────────────────────────────────────────

def node_translator(state: OrchestrationState) -> OrchestrationState:
    """Translate the final summary to the target language using DeepSeek chat.
    
    This is a simple translation agent that uses deepseek-chat for translation.
    It runs after the summarizer and before memory_update.
    """
    final_summary = state.get("final_summary", "")
    output_language = state.get("output_language")
    
    if not output_language or not final_summary:
        logger.info("[translator] No translation requested or no summary to translate")
        return state
    
    logger.info("[translator] Translating final summary to %s", output_language)
    
    try:
        from .llm import translate_text
        translated_summary = translate_text(final_summary, output_language)
        
        logger.info("[translator] Translation to %s complete", output_language)
        
        return {
            **state,
            "final_summary": translated_summary,
        }
    except Exception as exc:
        logger.warning("[translator] Translation failed: %s", exc)
        # Return original summary if translation fails
        return state


__all__ = [
    "node_planner",
    "node_business_analyst",
    "node_quant_fundamental",
    "node_web_search",
    "node_financial_modelling",
    "node_parallel_agents",
    "node_react_check",
    "node_summarizer",
    "node_rlaif_scorer",
    "node_translator",
    "node_memory_update",
]
