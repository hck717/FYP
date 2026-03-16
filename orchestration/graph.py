"""Main LangGraph orchestration graph — native parallel multi-agent pipeline.

Architecture (native LangGraph fan-out + per-agent ReAct retry):

                          ┌─────────────────────────────────────────────────────┐
  user_query              │              ORCHESTRATION GRAPH                     │
      │                   │                                                      │
      ▼                   │  planner (deepseek-chat)                             │
  [planner]  ─────────────┼──► decides: ticker, which agents, complexity (1-3)  │
      │                   │    also queries episodic_memory for known failures    │
      │                   │                                                      │
      ├──────────────────►│  node_business_analyst    ┐                         │
      ├──────────────────►│  node_quant_fundamental   │  LangGraph native        │
      ├──────────────────►│  node_web_search (opt)    │  fan-out — executed      │
      ├──────────────────►│  node_financial_modelling │  simultaneously          │
      └──────────────────►│  node_stock_research (opt)┘                         │
                          │         │                                            │
                          │  Per-agent conditional edges:                        │
                          │    • if no output AND iters < max → retry agent      │
                          │    • otherwise → node_summarizer (fan-in)            │
                          │                                                      │
                          │  [node_summarizer] — deepseek-chat (Stage1+2+3+4)   │
                          │         │                                            │
                          │  [node_post_processing]                              │
                          │    RLAIF scoring + episodic memory persistence       │
                          │         │                                            │
                          │        END                                           │
                          └─────────────────────────────────────────────────────┘

ReAct retry behaviour (per-agent):
  - Each agent node increments its own counter in agent_react_iterations.
  - The conditional edge after each agent checks: no output AND counter < react_max
    → re-route back to that agent for a retry pass.
  - react_max is set by the planner from complexity (1=no retry, 2=1 retry, 3=2 retries).
  - A successful agent is never re-executed.

Translation:
  - Translation (if output_language is set) is performed inside node_summarizer
    as Stage 4 of summarise_results_structured, saving one extra API call.

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
    node_business_analyst,
    node_financial_modelling,
    node_planner,
    node_post_processing,
    node_quant_fundamental,
    node_stock_research,
    node_summarizer,
    node_web_search,
    subscribe_agent_progress,
    unsubscribe_agent_progress,
)
from .state import OrchestrationState

logger = logging.getLogger(__name__)


# ── Per-agent conditional edge functions ──────────────────────────────────────
# Each returns the agent node name (for retry) or "summarizer" (to proceed).

def _route_after_ba(state: OrchestrationState) -> str:
    """Route after business_analyst: retry if no output and under iteration cap."""
    if not state.get("run_business_analyst"):
        return "summarizer"
    iters = (state.get("agent_react_iterations") or {}).get("business_analyst", 0)
    react_max = state.get("react_max_iterations") or 1
    if not state.get("business_analyst_outputs") and iters < react_max:
        logger.info("[route] business_analyst retry (iter=%d/%d)", iters, react_max)
        return "node_business_analyst"
    return "summarizer"


def _route_after_qf(state: OrchestrationState) -> str:
    """Route after quant_fundamental: retry if no output and under iteration cap."""
    if not state.get("run_quant_fundamental"):
        return "summarizer"
    iters = (state.get("agent_react_iterations") or {}).get("quant_fundamental", 0)
    react_max = state.get("react_max_iterations") or 1
    if not state.get("quant_fundamental_outputs") and iters < react_max:
        logger.info("[route] quant_fundamental retry (iter=%d/%d)", iters, react_max)
        return "node_quant_fundamental"
    return "summarizer"


def _route_after_ws(state: OrchestrationState) -> str:
    """Route after web_search: retry if no output and under iteration cap."""
    if not state.get("run_web_search"):
        return "summarizer"
    iters = (state.get("agent_react_iterations") or {}).get("web_search", 0)
    react_max = state.get("react_max_iterations") or 1
    if not state.get("web_search_outputs") and iters < react_max:
        logger.info("[route] web_search retry (iter=%d/%d)", iters, react_max)
        return "node_web_search"
    return "summarizer"


def _route_after_fm(state: OrchestrationState) -> str:
    """Route after financial_modelling: retry if no output and under iteration cap."""
    if not state.get("run_financial_modelling"):
        return "summarizer"
    iters = (state.get("agent_react_iterations") or {}).get("financial_modelling", 0)
    react_max = state.get("react_max_iterations") or 1
    if not state.get("financial_modelling_outputs") and iters < react_max:
        logger.info("[route] financial_modelling retry (iter=%d/%d)", iters, react_max)
        return "node_financial_modelling"
    return "summarizer"


def _route_after_sr(state: OrchestrationState) -> str:
    """Route after stock_research: retry if no output and under iteration cap."""
    if not state.get("run_stock_research"):
        return "summarizer"
    iters = (state.get("agent_react_iterations") or {}).get("stock_research", 0)
    react_max = state.get("react_max_iterations") or 1
    if not state.get("stock_research_outputs") and iters < react_max:
        logger.info("[route] stock_research retry (iter=%d/%d)", iters, react_max)
        return "node_stock_research"
    return "summarizer"


# ── Planner fan-out routing ────────────────────────────────────────────────────

def _route_after_planner(state: OrchestrationState) -> list[str]:
    """Fan out from planner to all enabled agent nodes simultaneously.

    LangGraph executes all returned node names concurrently (native parallelism).
    Disabled agents are not included — the graph never routes to them.
    Always includes at least one agent to prevent routing to an empty list.
    """
    targets: list[str] = []
    if state.get("run_business_analyst", True):
        targets.append("node_business_analyst")
    if state.get("run_quant_fundamental", True):
        targets.append("node_quant_fundamental")
    if state.get("run_web_search"):
        targets.append("node_web_search")
    if state.get("run_financial_modelling"):
        targets.append("node_financial_modelling")
    if state.get("run_stock_research"):
        targets.append("node_stock_research")
    # Fallback: if planner disabled everything, route to BA anyway
    if not targets:
        targets.append("node_business_analyst")
    return targets


def build_graph() -> Any:
    """Assemble and compile the native parallel LangGraph orchestration graph.

    Topology:
      planner → [ba, qf, ws, fm, sr] (native fan-out)
              → per-agent conditional retry edges
              → summarizer (fan-in — waits for all branches)
              → node_post_processing → END

    Translation (if output_language set) is done inside node_summarizer Stage 4.
    RLAIF scoring + episodic memory are merged in node_post_processing.

    Returns a compiled StateGraph ready to call with .invoke() or .stream().
    """
    graph = StateGraph(OrchestrationState)

    # Register nodes
    graph.add_node("planner",                node_planner)
    graph.add_node("node_business_analyst",  node_business_analyst)
    graph.add_node("node_quant_fundamental", node_quant_fundamental)
    graph.add_node("node_web_search",        node_web_search)
    graph.add_node("node_financial_modelling", node_financial_modelling)
    graph.add_node("node_stock_research",    node_stock_research)
    graph.add_node("summarizer",             node_summarizer)
    graph.add_node("post_processing",        node_post_processing)

    # Entry point
    graph.set_entry_point("planner")

    # Fan-out: planner → all enabled agents simultaneously (native LangGraph parallel)
    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "node_business_analyst":   "node_business_analyst",
            "node_quant_fundamental":  "node_quant_fundamental",
            "node_web_search":         "node_web_search",
            "node_financial_modelling":"node_financial_modelling",
            "node_stock_research":     "node_stock_research",
        },
    )

    # Per-agent retry edges: each agent → retry self OR proceed to summarizer
    graph.add_conditional_edges(
        "node_business_analyst",
        _route_after_ba,
        {"node_business_analyst": "node_business_analyst", "summarizer": "summarizer"},
    )
    graph.add_conditional_edges(
        "node_quant_fundamental",
        _route_after_qf,
        {"node_quant_fundamental": "node_quant_fundamental", "summarizer": "summarizer"},
    )
    graph.add_conditional_edges(
        "node_web_search",
        _route_after_ws,
        {"node_web_search": "node_web_search", "summarizer": "summarizer"},
    )
    graph.add_conditional_edges(
        "node_financial_modelling",
        _route_after_fm,
        {"node_financial_modelling": "node_financial_modelling", "summarizer": "summarizer"},
    )
    graph.add_conditional_edges(
        "node_stock_research",
        _route_after_sr,
        {"node_stock_research": "node_stock_research", "summarizer": "summarizer"},
    )

    # Fan-in: LangGraph waits for all branches before proceeding to summarizer
    graph.add_edge("summarizer",      "post_processing")
    graph.add_edge("post_processing", END)

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
        "agent_react_iterations": {},
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

    In the native parallel graph, agent nodes run simultaneously.
    The UI progress queue receives per-agent "started"/"done"/"error" events
    pushed directly from each agent node.
    """
    if not session_id:
        session_id = str(uuid.uuid4())[:12]
    graph = _get_graph()
    initial_state: OrchestrationState = {
        "user_query": user_query,
        "session_id": session_id,
        "react_steps": [],
        "react_iteration": 0,
        "agent_react_iterations": {},
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
