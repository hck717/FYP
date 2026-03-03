"""Orchestration package for the Agentic Investment Analyst.

Provides:
  - build_graph() — assemble and compile the LangGraph pipeline
  - run(user_query)  — synchronous execution, returns final state dict
  - stream(user_query) — yields (node_name, partial_state) as each node finishes

Quick start:
    from orchestration import run
    result = run("What is Apple's competitive moat and valuation?")
    print(result["final_summary"])
"""

from .graph import build_graph, run, stream

__all__ = ["build_graph", "run", "stream"]
