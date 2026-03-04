"""Shared state schema for the orchestration LangGraph pipeline.

This TypedDict is the single source of truth for data flowing between:
  planner → [business_analyst, quant_fundamental, web_search, financial_modelling] → summarizer

Multi-ticker support
--------------------
When the user asks a comparison query (e.g. "Compare MSFT vs AAPL"), the
planner populates ``tickers`` with all resolved symbols.  Each agent node
iterates over every ticker in ``tickers`` and stores a *list* of per-ticker
result dicts in the corresponding ``*_outputs`` key.  The summarizer then
receives all lists and writes a comparative research note.

For backward-compatibility, the legacy single-value keys
(``ticker``, ``business_analyst_output``, ``quant_fundamental_output``,
``web_search_output``) are still present and point to the first ticker's data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class OrchestrationState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    user_query: str                         # original chat message from the user
    session_id: str                         # optional session identifier

    # ── Planner output ───────────────────────────────────────────────────────
    plan: Optional[Dict[str, Any]]          # structured plan from DeepSeek planner

    # Single-ticker (legacy compat) — always the *first* resolved ticker
    ticker: Optional[str]
    # Multi-ticker list — length ≥ 1 for any resolved query
    tickers: List[str]

    run_business_analyst: bool              # whether to call BA agent
    run_quant_fundamental: bool             # whether to call quant agent
    run_web_search: bool                    # whether to call web search agent
    run_financial_modelling: bool           # whether to call financial modelling agent

    # ── ReAct iteration tracking ─────────────────────────────────────────────
    react_steps: List[Dict[str, Any]]       # [{tool, input, observation}, ...]
    react_iteration: int                    # current pass index (0-based)
    react_max_iterations: int               # max passes allowed — set by planner from complexity (1-3)

    # ── Agent raw outputs (multi-ticker lists) ───────────────────────────────
    # Each list contains one result dict per ticker in ``tickers``.
    business_analyst_outputs: List[Dict[str, Any]]
    quant_fundamental_outputs: List[Dict[str, Any]]
    web_search_outputs: List[Dict[str, Any]]
    financial_modelling_outputs: List[Dict[str, Any]]

    # Legacy single-output aliases — set to outputs[0] for backward-compat
    business_analyst_output: Optional[Dict[str, Any]]
    quant_fundamental_output: Optional[Dict[str, Any]]
    web_search_output: Optional[Dict[str, Any]]
    financial_modelling_output: Optional[Dict[str, Any]]

    # ── Data availability snapshot ───────────────────────────────────────────
    # Populated by node_planner once per request; consumed by all agent nodes
    # and the summarizer to avoid dead code-paths and surface data gaps.
    data_availability: Optional[Dict[str, Any]]

    # ── Episodic memory hints ─────────────────────────────────────────────────
    # Populated by node_planner from agent_episodic_memory table lookups.
    # Keys: force_web_search (bool), degraded_agents (List[str]).
    # Used to pre-empt known failure patterns before agents are dispatched.
    episodic_hints: Optional[Dict[str, Any]]

    # ── Errors ───────────────────────────────────────────────────────────────
    agent_errors: Dict[str, str]            # {agent_name: error_message}

    # ── Final output ─────────────────────────────────────────────────────────
    final_summary: Optional[str]            # narrative from DeepSeek summarizer
    output: Optional[Dict[str, Any]]        # full structured response to UI


__all__ = ["OrchestrationState"]
