#!/usr/bin/env python3
"""Integration-style tests for planner worst-case context injection."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestration.state import OrchestrationState


def test_planner_injects_worst_cases(monkeypatch):
    """Planner passes worst-case context into plan_query when available."""
    from orchestration import nodes

    captured: dict[str, str] = {"ctx": ""}

    # Avoid DB access from feedback helper
    monkeypatch.setattr(
        nodes.feedback,
        "get_worst_cases",
        lambda limit=5: [
            {
                "user_query": "Analyze TSLA earnings call sentiment and broker disagreement",
                "ticker": "TSLA",
                "overall_score": 4.2,
                "agent_blamed": "business_analyst",
                "weaknesses": ["missing citations", "shallow analysis"],
                "specific_feedback": "too shallow",
                "helpful": False,
                "issue_tags": ["Analysis too shallow"],
                "comment": "completely useless",
            }
        ],
    )

    # Avoid external checks/import side effects
    monkeypatch.setattr(
        "orchestration.data_availability.check_all",
        lambda tickers=None: {"summary": "ok", "degraded_tiers": []},
        raising=True,
    )
    monkeypatch.setattr(
        "orchestration.episodic_memory.lookup_similar_failures",
        lambda user_query, tickers=None: [],
        raising=True,
    )
    monkeypatch.setattr(
        "orchestration.episodic_memory.build_preemptive_plan_hints",
        lambda failures: {},
        raising=True,
    )

    dummy_qf_agent = types.ModuleType("agents.quant_fundamental.agent")
    setattr(dummy_qf_agent, "extract_tickers_from_prompt", lambda q: ["AAPL"])
    monkeypatch.setitem(sys.modules, "agents.quant_fundamental.agent", dummy_qf_agent)

    def _mock_plan_query(user_query, worst_case_context=""):
        captured["ctx"] = worst_case_context
        return {
            "tickers": ["AAPL"],
            "ticker": "AAPL",
            "run_business_analyst": True,
            "run_quant_fundamental": True,
            "run_web_search": False,
            "run_financial_modelling": False,
            "run_stock_research": False,
            "complexity": 2,
            "planner_trace": "",
        }

    monkeypatch.setattr(nodes, "plan_query", _mock_plan_query)

    state = cast(OrchestrationState, {"user_query": "What is AAPL's competitive moat?"})
    out = nodes.node_planner(state)

    assert captured["ctx"] != ""
    assert "PAST FAILURES" in captured["ctx"]
    assert "Blamed:" in captured["ctx"]
    assert out.get("ticker") == "AAPL"


def test_get_worst_cases_cold_start(monkeypatch):
    """get_worst_cases returns [] when rl_feedback has fewer than min_runs."""
    from orchestration import feedback

    class _Cursor:
        def __init__(self):
            self._sql = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            self._sql = sql

        def fetchone(self):
            if "COUNT(*) AS cnt" in self._sql:
                return {"cnt": 1}
            return None

        def fetchall(self):
            return []

    class _Conn:
        def cursor(self, cursor_factory=None):
            return _Cursor()

        def close(self):
            return None

    monkeypatch.setattr(feedback, "ensure_feedback_tables_exist", lambda: None)
    monkeypatch.setattr(feedback, "_get_pg_conn", lambda: _Conn())

    result = feedback.get_worst_cases(limit=5, min_runs=3)
    assert result == []


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
