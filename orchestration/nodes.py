"""Node implementations for the orchestration LangGraph pipeline.

Each function is a pure node — takes state, returns partial state update.

Pipeline:
  planner → parallel_agents → summarizer

  parallel_agents fans out BA, QF, FM (and optionally WS) concurrently using
  a ThreadPoolExecutor so all enabled agents run at the same time.  Wall-clock
  time is bounded by the slowest single agent rather than their sum.

  The legacy sequential ReAct nodes (react_dispatch / react_check) are kept for
  backward-compatibility but are no longer wired into the default graph.

Multi-ticker support
--------------------
When the planner resolves multiple ticker symbols (e.g. "Compare MSFT vs AAPL"),
every agent node iterates over all tickers in ``state["tickers"]`` and accumulates
per-ticker result dicts in the ``*_outputs`` list keys.  The legacy single-value
``*_output`` aliases are set to the first ticker's result for backward-compat.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, cast

from .llm import plan_query, summarise_results
from .state import OrchestrationState

logger = logging.getLogger(__name__)

_MAX_REACT_ITERATIONS = 3  # safety cap (legacy sequential path only)


# ── Node 1: Planner ───────────────────────────────────────────────────────────

def node_planner(state: OrchestrationState) -> OrchestrationState:
    """Use local DeepSeek to parse the user query and decide which agents to run.

    Populates:
      - plan, ticker, tickers, run_business_analyst, run_quant_fundamental, run_web_search
    """
    user_query = state.get("user_query", "")
    logger.info("[planner] Analysing query: %r", user_query)

    plan = plan_query(user_query)

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

    # complexity 1 → 1 pass, complexity 2 → 2 passes, complexity 3 → 3 passes
    raw_complexity = plan.get("complexity", 2)
    try:
        react_max = max(1, min(3, int(raw_complexity)))
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

    if not tasks:
        logger.warning("[parallel_agents] No agents enabled — proceeding to summarizer.")
    else:
        n_workers = len(tasks)  # one thread per agent
        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="agent") as pool:
            future_to_name = {pool.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(future_to_name):
                name = future_to_name[future]
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
                    if name == "business_analyst":
                        ba_outputs = result_list
                    elif name == "quant_fundamental":
                        qf_outputs = result_list
                    elif name == "web_search":
                        ws_outputs = result_list
                    elif name == "financial_modelling":
                        fm_outputs = result_list
                except Exception as exc:
                    msg = f"{type(exc).__name__}: {exc}"
                    logger.error("[parallel_agents] %s failed: %s", name, msg)
                    errors[name] = msg
                    steps.append({
                        "tool": name,
                        "input": {"tickers": tickers, "query": query},
                        "observation": f"Error: {msg}",
                    })

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

    # Clear errors for agents that will be retried so they get a clean slate
    new_errors = {k: v for k, v in errors.items() if k not in retry_agents} if should_loop else errors

    return {
        **state,
        "react_iteration": iteration,
        "agent_errors": new_errors,
    }


# ── Node 7: Summarizer ────────────────────────────────────────────────────────

def node_summarizer(state: OrchestrationState) -> OrchestrationState:
    """Use local DeepSeek to synthesise all agent outputs into a final report."""
    user_query    = state.get("user_query", "")
    tickers       = state.get("tickers") or []
    ticker        = state.get("ticker")

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

    final_summary = summarise_results(
        user_query=user_query,
        tickers=tickers,
        ba_outputs=ba_outputs,
        quant_outputs=quant_outputs,
        web_outputs=web_outputs,
        fm_outputs=fm_outputs,
        data_availability=data_availability,
    )

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
    }

    return {**state, "final_summary": final_summary, "output": output}


__all__ = [
    "node_planner",
    "node_business_analyst",
    "node_quant_fundamental",
    "node_web_search",
    "node_financial_modelling",
    "node_parallel_agents",
    "node_react_check",
    "node_summarizer",
]
