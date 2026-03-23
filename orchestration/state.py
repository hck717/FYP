"""Shared state schema for the orchestration LangGraph pipeline.

This TypedDict is the single source of truth for data flowing between:
  planner → [business_analyst, quant_fundamental, web_search, financial_modelling,
             stock_research] → summarizer

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

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


def _merge_dicts(a: Optional[Dict], b: Optional[Dict]) -> Dict:
    """Merge two dicts, with b taking precedence on key conflicts."""
    result = dict(a or {})
    result.update(b or {})
    return result


class OrchestrationState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    user_query: str                         # original chat message from the user
    session_id: str                         # optional session identifier
    output_language: Optional[str]         # detected language requirement (e.g. "cantonese", "spanish")

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
    run_stock_research: bool                # whether to call stock research (PDF broker/transcript) agent
    run_macro: bool                         # whether to call macro agent
    run_insider_news: bool                 # whether to call insider news agent

    # ── ReAct iteration tracking ─────────────────────────────────────────────
    # react_steps uses operator.add so concurrent agent writes are concatenated
    react_steps: Annotated[List[Dict[str, Any]], operator.add]  # [{tool, input, observation}, ...]
    react_iteration: int                    # current pass index (0-based)
    react_max_iterations: int               # max passes allowed — set by planner from complexity (1-3)
    # Per-agent independent retry counters — _merge_dicts so concurrent writes are merged
    agent_react_iterations: Annotated[Optional[Dict[str, int]], _merge_dicts]  # {agent_name: retry_count}

    # ── Agent raw outputs (multi-ticker lists) ───────────────────────────────
    # Each list contains one result dict per ticker in ``tickers``.
    business_analyst_outputs: List[Dict[str, Any]]
    quant_fundamental_outputs: List[Dict[str, Any]]
    web_search_outputs: List[Dict[str, Any]]
    financial_modelling_outputs: List[Dict[str, Any]]
    stock_research_outputs: List[Dict[str, Any]]
    macro_outputs: List[Dict[str, Any]]
    insider_news_outputs: List[Dict[str, Any]]

    # Legacy single-output aliases — set to outputs[0] for backward-compat
    business_analyst_output: Optional[Dict[str, Any]]
    quant_fundamental_output: Optional[Dict[str, Any]]
    web_search_output: Optional[Dict[str, Any]]
    financial_modelling_output: Optional[Dict[str, Any]]
    stock_research_output: Optional[Dict[str, Any]]
    macro_output: Optional[Dict[str, Any]]
    insider_news_output: Optional[Dict[str, Any]]

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
    # _merge_dicts so concurrent parallel agent writes are combined
    agent_errors: Annotated[Dict[str, str], _merge_dicts]  # {agent_name: error_message}

    # ── Thinking traces ──────────────────────────────────────────────────────
    planner_trace: Optional[str]            # DeepSeek reasoning from node_planner
    summarizer_trace: Optional[str]         # DeepSeek reasoning from node_summarizer

    # ── Final output ─────────────────────────────────────────────────────────
    final_summary: Optional[str]            # narrative from DeepSeek summarizer
    output: Optional[Dict[str, Any]]        # full structured response to UI

    # ── RLAIF Feedback ────────────────────────────────────────────────────────
    rl_feedback_scores: Optional[Dict[str, Any]]  # RLAIF scores from AI judge
    rl_feedback_run_id: Optional[str]      # Unique run ID for this analysis


__all__ = ["OrchestrationState"]
