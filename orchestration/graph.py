"""Main LangGraph orchestration graph — parallel multi-agent pipeline with ReAct loop.

Architecture (default — parallel dispatch + ReAct loop):

                          ┌──────────────────────────────────────────────────────┐
  user_query              │              ORCHESTRATION GRAPH                      │
      │                   │                                                       │
      ▼                   │  planner (llama3.2:latest)                            │
  [planner]  ─────────────┼──► decides: ticker, which agents, complexity (1-3)   │
      │                   │                                                       │
      ▼                   │  parallel_agents  (ThreadPoolExecutor)               │
  [parallel_agents] ──────┼──► BA + QF + FM + WS run concurrently               │◄─┐
      │                   │    wall-clock = max(T_BA, T_QF, T_FM, T_WS)          │  │
      ▼                   │                                                       │  │ loop if
  [react_check] ──────────┼──► evaluate gaps/errors; decide loop or proceed      │  │ gaps & iters left
      │                   │                                                       │  │
      │  (loop)           │                                                       │──┘
      │  (done)           │                                                       │
      ▼                   │  summarizer (deepseek-r1:8b)                          │
  [summarizer] (deepseek) │                                                       │
      │                   └──────────────────────────────────────────────────────┘
      ▼
   output dict / final_summary

ReAct loop behaviour:
  - complexity 1 → max 1 pass  (simple metric look-up: no retry)
  - complexity 2 → max 2 passes (moderate analysis: one retry on gaps/errors)
  - complexity 3 → max 3 passes (full report: up to two retries on gaps/errors)
  - On each loop iteration only agents with NO output yet (gaps) or errors are re-run.
    A successful agent is never re-executed.

The legacy sequential ReAct graph (build_sequential_graph) is preserved for
debugging and is accessible via the ORCHESTRATION_SEQUENTIAL=1 env var.

Usage:
    from orchestration.graph import build_graph, run

    result = run("What is Apple's competitive moat and valuation?")
    print(result["final_summary"])
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure the repo root is on sys.path so agent imports work regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from langgraph.graph import END, StateGraph

from .nodes import (
    node_business_analyst,
    node_financial_modelling,
    node_parallel_agents,
    node_planner,
    node_quant_fundamental,
    node_react_check,
    node_summarizer,
    node_web_search,
)
from .state import OrchestrationState

logger = logging.getLogger(__name__)

_MAX_REACT_ITERATIONS = 6  # safety cap — legacy sequential path only


# ── Parallel graph (default) ──────────────────────────────────────────────────

def _should_loop(state: OrchestrationState) -> str:
    """Conditional edge router after react_check.

    Returns "parallel_agents" to loop (re-run gap/error agents) or
    "summarizer" to proceed when all agents are done or the iteration cap is reached.

    The actual decision logic is mirrored from ``node_react_check``:
      - gaps: an enabled agent produced no output
      - errors: an enabled agent raised an exception (cleared in node_react_check for retry)
      - iteration already incremented in node_react_check before this edge fires
    """
    iteration = state.get("react_iteration") or 0
    react_max = state.get("react_max_iterations") or 1

    # If we've already consumed all allowed passes, go to summarizer
    if iteration >= react_max:
        return "summarizer"

    # Check for gaps (enabled agents with no output) — these merit a retry
    if state.get("run_business_analyst") and not state.get("business_analyst_outputs"):
        return "parallel_agents"
    if state.get("run_quant_fundamental") and not state.get("quant_fundamental_outputs"):
        return "parallel_agents"
    if state.get("run_web_search") and not state.get("web_search_outputs"):
        return "parallel_agents"
    if state.get("run_financial_modelling") and not state.get("financial_modelling_outputs"):
        return "parallel_agents"

    # No gaps — proceed to summarizer regardless of remaining iterations
    return "summarizer"

def build_graph() -> Any:
    """Assemble and compile the parallel orchestration LangGraph with ReAct loop.

    Topology: planner → parallel_agents → react_check → (parallel_agents | summarizer) → END

    All enabled agents (BA, QF, FM, WS) run concurrently inside
    ``node_parallel_agents`` via a ThreadPoolExecutor.  After each pass,
    ``node_react_check`` evaluates whether any enabled agents failed or produced
    no output.  If gaps remain AND the current iteration is below
    ``react_max_iterations`` (derived from the planner's complexity score 1-3),
    the graph loops back to ``node_parallel_agents`` to retry only the gap agents.
    Otherwise it advances to the summarizer.

    Returns a compiled StateGraph ready to call with .invoke() or .stream().
    """
    graph = StateGraph(OrchestrationState)

    graph.add_node("planner",          node_planner)
    graph.add_node("parallel_agents",  node_parallel_agents)
    graph.add_node("react_check",      node_react_check)
    graph.add_node("summarizer",       node_summarizer)

    graph.set_entry_point("planner")
    graph.add_edge("planner",          "parallel_agents")
    graph.add_edge("parallel_agents",  "react_check")

    graph.add_conditional_edges(
        "react_check",
        _should_loop,
        {
            "parallel_agents": "parallel_agents",
            "summarizer":      "summarizer",
        },
    )

    graph.add_edge("summarizer", END)

    return graph.compile()


# ── Legacy sequential ReAct graph (debug / fallback) ─────────────────────────

def _react_dispatch(state: OrchestrationState) -> OrchestrationState:
    """No-op pass-through node; routing logic is in conditional edges."""
    return state


def _route_from_dispatch(state: OrchestrationState) -> str:
    """Fan-out: pick the first enabled agent not yet done (priority order)."""
    if state.get("run_business_analyst") and not state.get("business_analyst_outputs"):
        return "business_analyst"
    if state.get("run_quant_fundamental") and not state.get("quant_fundamental_outputs"):
        return "quant_fundamental"
    if state.get("run_web_search") and not state.get("web_search_outputs"):
        return "web_search"
    if state.get("run_financial_modelling") and not state.get("financial_modelling_outputs"):
        return "financial_modelling"
    return "summarizer"


def _route_after_agent(state: OrchestrationState) -> str:
    """After an agent completes, loop back or proceed to summarizer."""
    ba_needed = state.get("run_business_analyst") and not state.get("business_analyst_outputs")
    qf_needed = state.get("run_quant_fundamental") and not state.get("quant_fundamental_outputs")
    ws_needed = state.get("run_web_search") and not state.get("web_search_outputs")
    fm_needed = state.get("run_financial_modelling") and not state.get("financial_modelling_outputs")

    if ba_needed or qf_needed or ws_needed or fm_needed:
        iteration = (state.get("react_iteration") or 0) + 1
        if iteration >= _MAX_REACT_ITERATIONS:
            logger.warning(
                "[react_check] Max iterations reached (%d) — proceeding to summarizer.", iteration
            )
            return "summarizer"
        return "react_dispatch"
    return "summarizer"


def _increment_iteration(state: OrchestrationState) -> OrchestrationState:
    return {**state, "react_iteration": (state.get("react_iteration") or 0) + 1}


def build_sequential_graph() -> Any:
    """Build the legacy sequential ReAct graph (one agent at a time).

    Useful for debugging individual agent failures without parallelism.
    Enable with the env var: ORCHESTRATION_SEQUENTIAL=1
    """
    graph = StateGraph(OrchestrationState)

    graph.add_node("planner",              node_planner)
    graph.add_node("react_dispatch",       _react_dispatch)
    graph.add_node("business_analyst",     node_business_analyst)
    graph.add_node("quant_fundamental",    node_quant_fundamental)
    graph.add_node("web_search",           node_web_search)
    graph.add_node("financial_modelling",  node_financial_modelling)
    graph.add_node("react_check",          _increment_iteration)
    graph.add_node("summarizer",           node_summarizer)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "react_dispatch")

    graph.add_conditional_edges(
        "react_dispatch",
        _route_from_dispatch,
        {
            "business_analyst":    "business_analyst",
            "quant_fundamental":   "quant_fundamental",
            "web_search":          "web_search",
            "financial_modelling": "financial_modelling",
            "summarizer":          "summarizer",
        },
    )

    graph.add_edge("business_analyst",    "react_check")
    graph.add_edge("quant_fundamental",   "react_check")
    graph.add_edge("web_search",          "react_check")
    graph.add_edge("financial_modelling", "react_check")

    graph.add_conditional_edges(
        "react_check",
        _route_after_agent,
        {
            "react_dispatch": "react_dispatch",
            "summarizer":     "summarizer",
        },
    )

    graph.add_edge("summarizer", END)
    return graph.compile()


# ── Public run helpers ────────────────────────────────────────────────────────

# Module-level compiled graph (built once, reused across Streamlit reruns)
_compiled_graph: Any = None


def _get_graph() -> Any:
    global _compiled_graph
    if _compiled_graph is None:
        use_sequential = os.getenv("ORCHESTRATION_SEQUENTIAL", "").strip() in ("1", "true", "yes")
        if use_sequential:
            logger.info("[orchestration] Using legacy sequential ReAct graph (ORCHESTRATION_SEQUENTIAL=1).")
            _compiled_graph = build_sequential_graph()
        else:
            _compiled_graph = build_graph()
    return _compiled_graph


def run(
    user_query: str,
    session_id: str = "",
) -> Dict[str, Any]:
    """Run the full orchestration pipeline for a user query.

    Args:
        user_query: Natural language question from the user.
        session_id: Optional identifier for the session (for logging).

    Returns:
        The final state dict, always containing:
          - "final_summary": str — the DeepSeek narrative
          - "ticker": str | None
          - "plan": dict
          - "react_steps": list
          - "business_analyst_output": dict | None
          - "quant_fundamental_output": dict | None
          - "web_search_output": dict | None
          - "agent_errors": dict
    """
    graph = _get_graph()
    initial_state: OrchestrationState = {
        "user_query": user_query,
        "session_id": session_id,
        "react_steps": [],
        "react_iteration": 0,
        "agent_errors": {},
    }
    logger.info("[orchestration.run] user_query=%r session=%s", user_query, session_id)
    final_state = graph.invoke(initial_state)
    return dict(final_state)


def stream(
    user_query: str,
    session_id: str = "",
):
    """Stream state updates from the orchestration graph.

    Yields (node_name, partial_state_dict) tuples as each node completes.
    Useful for streaming live progress to the Streamlit UI.

    In the parallel graph the nodes are: planner → parallel_agents → react_check → summarizer.
    For complexity-1 queries the UI sees exactly four events.  For complexity-2/3 queries,
    parallel_agents + react_check may repeat before the summarizer fires.
    """
    graph = _get_graph()
    initial_state: OrchestrationState = {
        "user_query": user_query,
        "session_id": session_id,
        "react_steps": [],
        "react_iteration": 0,
        "agent_errors": {},
    }
    for event in graph.stream(initial_state, stream_mode="updates"):
        if not isinstance(event, dict):
            continue
        for node_name, node_output in event.items():
            yield node_name, node_output


__all__ = ["build_graph", "build_sequential_graph", "run", "stream"]
