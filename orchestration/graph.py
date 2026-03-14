"""Main LangGraph orchestration graph — parallel multi-agent pipeline with ReAct loop.

Architecture (default — parallel dispatch + ReAct loop):

                          ┌──────────────────────────────────────────────────────┐
  user_query              │              ORCHESTRATION GRAPH                      │
      │                   │                                                       │
      ▼                   │  planner (llama3.2:latest)                            │
  [planner]  ─────────────┼──► decides: ticker, which agents, complexity (1-3)   │
      │                   │    also queries episodic_memory for known failures     │
      ▼                   │                                                       │
  [parallel_agents] ──────┼──► BA + QF + FM + WS run concurrently               │◄─┐
      │                   │    wall-clock = max(T_BA, T_QF, T_FM, T_WS)          │  │
      ▼                   │                                                       │  │ loop if
  [react_check] ──────────┼──► evaluate gaps/errors; decide loop or proceed      │  │ gaps & iters left
      │                   │                                                       │  │
      │  (loop)           │                                                       │──┘
      │  (done)           │                                                       │
      ▼                   │  summarizer (deepseek-r1:8b)                          │
  [summarizer] (deepseek) │                                                       │
      │                   │  memory_update                                        │
      ▼                   │  persists failure patterns to agent_episodic_memory   │
  [memory_update]         └──────────────────────────────────────────────────────┘
      │
      ▼
   output dict / final_summary

ReAct loop behaviour:
  - complexity 1 → max 1 pass  (simple metric look-up: no retry)
  - complexity 2 → max 2 passes (moderate analysis: one retry on gaps/errors)
  - complexity 3 → max 3 passes (full report: up to two retries on gaps/errors)
  - On each loop iteration only agents with NO output yet (gaps) or errors are re-run.
    A successful agent is never re-executed.

Usage:
    from orchestration.graph import build_graph, run

    result = run("What is Apple's competitive moat and valuation?")
    print(result["final_summary"])
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure the repo root is on sys.path so agent imports work regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from langgraph.graph import END, StateGraph

from .nodes import (
    node_memory_update,
    node_parallel_agents,
    node_planner,
    node_react_check,
    node_rlaif_scorer,
    node_summarizer,
    subscribe_agent_progress,
    unsubscribe_agent_progress,
)
from .state import OrchestrationState

logger = logging.getLogger(__name__)


# ── Parallel graph ────────────────────────────────────────────────────────────

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

    Topology:
      planner → parallel_agents → react_check → (parallel_agents | summarizer)
              → translator → memory_update → END

    All enabled agents (BA, QF, FM, WS) run concurrently inside
    ``node_parallel_agents`` via a ThreadPoolExecutor.  After each pass,
    ``node_react_check`` evaluates whether any enabled agents failed or produced
    no output.  If gaps remain AND the current iteration is below
    ``react_max_iterations`` (derived from the planner's complexity score 1-3),
    the graph loops back to ``node_parallel_agents`` to retry only the gap agents.
    Otherwise it advances to the summarizer.

    After the summarizer, ``node_translator`` translates the final summary to the
    requested language (if any). Then ``node_memory_update`` persists any failure patterns
    to the ``agent_episodic_memory`` PostgreSQL table for future pre-emption.

    Returns a compiled StateGraph ready to call with .invoke() or .stream().
    """
    from .nodes import node_translator, node_rlaif_scorer

    graph = StateGraph(OrchestrationState)

    graph.add_node("planner",          node_planner)
    graph.add_node("parallel_agents",  node_parallel_agents)
    graph.add_node("react_check",      node_react_check)
    graph.add_node("summarizer",       node_summarizer)
    graph.add_node("rlaif_scorer",     node_rlaif_scorer)
    graph.add_node("translator",       node_translator)
    graph.add_node("memory_update",    node_memory_update)

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

    graph.add_edge("summarizer",       "rlaif_scorer")
    graph.add_edge("rlaif_scorer",    "translator")
    graph.add_edge("translator",       "memory_update")
    graph.add_edge("memory_update",    END)

    return graph.compile()


def get_graph() -> Any:
    """Return the compiled orchestration graph for visualization.

    The returned LangGraph compiled graph object supports:
        - ``.get_graph()``                 → DrawableGraph
        - ``.get_graph().draw_mermaid()``  → Mermaid diagram string
        - ``.get_graph().draw_ascii()``    → ASCII art string

    Example (Streamlit):
        from orchestration.graph import get_graph
        mermaid_str = get_graph().get_graph().draw_mermaid()
    """
    return _get_graph()


# ── Public run helpers ────────────────────────────────────────────────────────

# Module-level compiled graph (built once, reused across Streamlit reruns)
_compiled_graph: Any = None


def _get_graph() -> Any:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run(
    user_query: str,
    session_id: str = "",
    output_language: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full orchestration pipeline for a user query.

    Args:
        user_query: Natural language question from the user.
        session_id: Optional identifier for the session (for logging).
        output_language: Optional language code for translation (e.g., "cantonese", "mandarin").

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
        "output_language": output_language,
    }
    logger.info("[orchestration.run] user_query=%r session=%s output_language=%s", user_query, session_id, output_language)
    final_state = graph.invoke(initial_state)
    return dict(final_state)


def stream(
    user_query: str,
    session_id: str = "",
    output_language: Optional[str] = None,
):
    """Stream state updates from the orchestration graph.

    Yields (node_name, partial_state_dict) tuples as each node completes.
    Useful for streaming live progress to the Streamlit UI.

    The first yielded tuple is always::

        ("__session__", {"session_id": <str>})

    This lets the caller subscribe to the agent progress queue for the session
    *before* the graph starts running (call subscribe_agent_progress(session_id)
    after receiving this first event).

    In the parallel graph the nodes are:
      planner → parallel_agents → react_check → summarizer → translator → memory_update.
    For complexity-1 queries the UI sees exactly six node events.  For
    complexity-2/3 queries, parallel_agents + react_check may repeat.
    """
    if not session_id:
        session_id = str(uuid.uuid4())[:12]
    graph = _get_graph()
    initial_state: OrchestrationState = {
        "user_query": user_query,
        "session_id": session_id,
        "react_steps": [],
        "react_iteration": 0,
        "agent_errors": {},
        "output_language": output_language,
    }
    # Yield session_id first so caller can subscribe to the progress queue
    yield "__session__", {"session_id": session_id}
    for event in graph.stream(initial_state, stream_mode="updates"):
        if not isinstance(event, dict):
            continue
        for node_name, node_output in event.items():
            yield node_name, node_output


__all__ = ["build_graph", "run", "stream", "subscribe_agent_progress", "unsubscribe_agent_progress"]
