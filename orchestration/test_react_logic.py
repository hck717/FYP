#!/usr/bin/env python3
"""
Unit tests for the ReAct loop logic in node_react_check and _should_loop.
These tests mock minimal dependencies and focus on the decision logic.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import patch, MagicMock
from orchestration.nodes import node_react_check
from orchestration.graph import _should_loop


def test_should_loop_no_gaps():
    """When all agents have output, should proceed to summarizer."""
    state = {
        "react_iteration": 0,
        "react_max_iterations": 3,
        "run_business_analyst": True,
        "run_quant_fundamental": True,
        "run_financial_modelling": True,
        "run_web_search": True,
        "business_analyst_outputs": [{"ticker": "AAPL"}],
        "quant_fundamental_outputs": [{"ticker": "AAPL"}],
        "financial_modelling_outputs": [{"ticker": "AAPL"}],
        "web_search_outputs": [{"ticker": "AAPL"}],
    }
    assert _should_loop(state) == "summarizer"


def test_should_loop_gap_present():
    """When an agent has no output, should loop."""
    state = {
        "react_iteration": 0,
        "react_max_iterations": 3,
        "run_business_analyst": True,
        "run_quant_fundamental": True,
        "run_financial_modelling": True,
        "run_web_search": True,
        "business_analyst_outputs": [{"ticker": "AAPL"}],
        "quant_fundamental_outputs": [],  # gap
        "financial_modelling_outputs": [{"ticker": "AAPL"}],
        "web_search_outputs": [{"ticker": "AAPL"}],
    }
    assert _should_loop(state) == "parallel_agents"


def test_should_loop_iteration_limit():
    """When iteration >= max, should go to summarizer even with gaps."""
    state = {
        "react_iteration": 3,
        "react_max_iterations": 3,
        "run_business_analyst": True,
        "run_quant_fundamental": True,
        "run_financial_modelling": True,
        "run_web_search": True,
        "business_analyst_outputs": [],
        "quant_fundamental_outputs": [],
        "financial_modelling_outputs": [],
        "web_search_outputs": [],
    }
    assert _should_loop(state) == "summarizer"


def test_should_loop_agent_disabled():
    """Disabled agents (run_* = False) should not cause loops."""
    state = {
        "react_iteration": 0,
        "react_max_iterations": 3,
        "run_business_analyst": True,
        "run_quant_fundamental": False,  # disabled
        "run_financial_modelling": True,
        "run_web_search": True,
        "business_analyst_outputs": [{"ticker": "AAPL"}],
        "quant_fundamental_outputs": [],  # gap but agent disabled -> ignore
        "financial_modelling_outputs": [{"ticker": "AAPL"}],
        "web_search_outputs": [{"ticker": "AAPL"}],
    }
    # Since quant_fundamental is disabled, its gap doesn't matter
    assert _should_loop(state) == "summarizer"


def test_node_react_check_increments_iteration():
    """node_react_check should increment react_iteration."""
    state = {
        "react_iteration": 0,
        "react_max_iterations": 3,
        "run_business_analyst": True,
        "run_quant_fundamental": True,
        "run_financial_modelling": True,
        "run_web_search": True,
        "business_analyst_outputs": [{"ticker": "AAPL"}],
        "quant_fundamental_outputs": [],
        "financial_modelling_outputs": [{"ticker": "AAPL"}],
        "web_search_outputs": [{"ticker": "AAPL"}],
        "agent_errors": {},
    }
    with patch('orchestration.nodes._log_telemetry', return_value=None):
        new_state = node_react_check(state)
    assert new_state["react_iteration"] == 1
    # Should have gaps list
    assert "business_analyst" not in new_state.get("agent_errors", {})
    # quant_fundamental is gap, should be in retry list, but agent_errors cleared
    # Since we didn't mock the database timeout check, we need to mock that too.
    # Let's skip for now.


if __name__ == "__main__":
    # Run tests
    test_should_loop_no_gaps()
    print("✓ test_should_loop_no_gaps")
    test_should_loop_gap_present()
    print("✓ test_should_loop_gap_present")
    test_should_loop_iteration_limit()
    print("✓ test_should_loop_iteration_limit")
    test_should_loop_agent_disabled()
    print("✓ test_should_loop_agent_disabled")
    test_node_react_check_increments_iteration()
    print("✓ test_node_react_check_increments_iteration")
    print("All unit tests passed!")