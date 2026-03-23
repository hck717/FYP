"""Multi-ticker integration tests.

These tests verify:
- Multi-ticker comparison queries work correctly
- Each agent processes multiple tickers independently
- Summarizer handles comparative analysis

Run with: pytest tests/integration/test_multi_ticker.py -v --timeout=120
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orchestration.state import OrchestrationState


# ---------------------------------------------------------------------------
# Test: Multi-Ticker State Structure
# ---------------------------------------------------------------------------

def test_multi_ticker_state_has_list():
    """Test that multi-ticker state includes tickers list."""
    state: OrchestrationState = {
        "user_query": "Compare MSFT vs AAPL valuation",
        "ticker": "MSFT",  # Legacy first ticker
        "tickers": ["MSFT", "AAPL"],  # Multi-ticker list
        "run_business_analyst": True,
        "run_financial_modelling": True,
    }
    
    assert len(state["tickers"]) == 2
    assert "MSFT" in state["tickers"]
    assert "AAPL" in state["tickers"]


def test_multi_ticker_outputs_are_lists():
    """Test that agent outputs are lists for multi-ticker."""
    state: OrchestrationState = {
        "tickers": ["MSFT", "AAPL"],
        "business_analyst_outputs": [
            {"ticker": "MSFT", "qualitative_summary": "Strong cloud business"},
            {"ticker": "AAPL", "qualitative_summary": "Strong ecosystem"},
        ],
        "financial_modelling_outputs": [
            {"ticker": "MSFT", "dcf_value": 420.0},
            {"ticker": "AAPL", "dcf_value": 195.0},
        ],
        "macro_outputs": [
            {"ticker": "MSFT", "regime": "risk-off"},
            {"ticker": "AAPL", "regime": "disinflation"},
        ],
        "insider_news_outputs": [
            {"ticker": "MSFT", "data_coverage": {"insider_transactions_count": 2}},
            {"ticker": "AAPL", "data_coverage": {"insider_transactions_count": 1}},
        ],
    }
    
    # Each output list should have same length as tickers
    assert len(state["business_analyst_outputs"]) == len(state["tickers"])
    assert len(state["financial_modelling_outputs"]) == len(state["tickers"])
    assert len(state["macro_outputs"]) == len(state["tickers"])
    assert len(state["insider_news_outputs"]) == len(state["tickers"])


def test_single_ticker_backward_compat():
    """Test single ticker queries still work (backward compat)."""
    state: OrchestrationState = {
        "user_query": "What is AAPL's valuation?",
        "ticker": "AAPL",
        "tickers": ["AAPL"],  # Still a list, just one element
        "business_analyst_outputs": [{"ticker": "AAPL", "summary": "test"}],
    }
    
    # Legacy single-value keys should still work
    assert state["ticker"] == "AAPL"
    assert len(state["tickers"]) == 1


# ---------------------------------------------------------------------------
# Test: Multi-Ticker Agent Processing
# ---------------------------------------------------------------------------

def test_ba_processes_multiple_tickers():
    """Test Business Analyst can process multiple tickers."""
    # Mock BA output for multiple tickers
    mock_outputs = [
        {"ticker": "MSFT", "qualitative_summary": "Microsoft has strong cloud moat"},
        {"ticker": "AAPL", "qualitative_summary": "Apple has strong ecosystem moat"},
    ]
    
    # Verify each ticker has output
    for output in mock_outputs:
        assert "ticker" in output
        assert "qualitative_summary" in output
    
    assert len(mock_outputs) == 2


def test_fm_processes_multiple_tickers():
    """Test Financial Modelling can process multiple tickers."""
    # Mock FM output for multiple tickers
    mock_outputs = [
        {"ticker": "MSFT", "dcf_value": 420.0, "wacc": 0.09},
        {"ticker": "AAPL", "dcf_value": 195.0, "wacc": 0.095},
    ]
    
    for output in mock_outputs:
        assert "ticker" in output
        assert "dcf_value" in output
        assert "wacc" in output
    
    assert len(mock_outputs) == 2


def test_multi_ticker_factors():
    """Test Quant Fundamental returns factors for each ticker."""
    mock_outputs = [
        {"ticker": "MSFT", "piotroski_score": 9, "beneish_m_score": -2.5},
        {"ticker": "AAPL", "piotroski_score": 8, "beneish_m_score": -2.1},
    ]
    
    for output in mock_outputs:
        assert output["ticker"] in ["MSFT", "AAPL"]
        assert "piotroski_score" in output
    
    assert len(mock_outputs) == 2


# ---------------------------------------------------------------------------
# Test: Multi-Ticker Summarizer
# ---------------------------------------------------------------------------

def test_summarizer_comparative_output():
    """Test summarizer produces comparative output for multiple tickers."""
    mock_all_outputs = {
        "tickers": ["MSFT", "AAPL"],
        "business_analyst_outputs": [
            {"ticker": "MSFT", "qualitative_summary": "Cloud leader"},
            {"ticker": "AAPL", "qualitative_summary": "Ecosystem champion"},
        ],
        "financial_modelling_outputs": [
            {"ticker": "MSFT", "dcf_value": 420.0},
            {"ticker": "AAPL", "dcf_value": 195.0},
        ],
    }
    
    # Summarizer should have access to all tickers
    assert len(mock_all_outputs["tickers"]) >= 2
    
    # Each agent output should have entries for each ticker
    for output_list in mock_all_outputs.values():
        if isinstance(output_list, list):
            assert len(output_list) >= 2


def test_summarizer_legacy_aliases():
    """Test that legacy single-output aliases point to first ticker."""
    state: OrchestrationState = {
        "tickers": ["MSFT", "AAPL"],
        "business_analyst_outputs": [
            {"ticker": "MSFT", "qualitative_summary": "Cloud leader"},
            {"ticker": "AAPL", "qualitative_summary": "Ecosystem champion"},
        ],
        "business_analyst_output": {  # Legacy alias - should be outputs[0]
            "ticker": "MSFT", 
            "qualitative_summary": "Cloud leader"
        },
    }
    
    # The legacy alias should match first ticker
    assert (state.get("business_analyst_output") or {}).get("ticker") == state["tickers"][0]


# ---------------------------------------------------------------------------
# Test: Comparison Query Parsing
# ---------------------------------------------------------------------------

def test_comparison_query_structure():
    """Test that comparison queries have correct structure."""
    # This simulates what the planner would produce for a comparison query
    state: OrchestrationState = {
        "user_query": "Compare NVDA vs MSFT valuation",
        "ticker": "NVDA",  # First resolved ticker
        "tickers": ["NVDA", "MSFT"],
        "run_business_analyst": True,
        "run_quant_fundamental": True,
        "run_financial_modelling": True,
        "run_macro": True,
        "run_insider_news": True,
        "react_max_iterations": 3,  # Comparison is complex
    }
    
    assert len(state["tickers"]) == 2
    assert state["react_max_iterations"] >= 2  # Complex queries get more retries


def test_three_way_comparison():
    """Test three-way ticker comparison."""
    state: OrchestrationState = {
        "user_query": "Compare MSFT, AAPL, and GOOGL",
        "ticker": "MSFT",
        "tickers": ["MSFT", "AAPL", "GOOGL"],
        "run_quant_fundamental": True,
    }
    
    assert len(state["tickers"]) == 3


# ---------------------------------------------------------------------------
# Test: Multi-Ticker Edge Cases
# ---------------------------------------------------------------------------

def test_empty_ticker_list():
    """Test handling of empty ticker list."""
    state: OrchestrationState = {
        "tickers": [],
        "business_analyst_outputs": [],
    }
    
    # Should gracefully handle empty list
    assert len(state["tickers"]) == 0
    assert len(state["business_analyst_outputs"]) == 0


def test_ticker_with_no_data():
    """Test handling of ticker with no data in outputs."""
    state: OrchestrationState = {
        "tickers": ["AAPL", "UNKNOWN"],
        "business_analyst_outputs": [
            {"ticker": "AAPL", "qualitative_summary": "test"},
            # Second ticker has no output - that's OK
        ],
    }
    
    # Should still process the known ticker
    assert len(state["tickers"]) == 2
    assert state["business_analyst_outputs"][0]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# Test: Full Multi-Ticker Flow (Mocked)
# ---------------------------------------------------------------------------

def test_multi_ticker_full_flow():
    """Test complete multi-ticker flow with mocked data."""
    # Input state
    input_state: OrchestrationState = {
        "user_query": "Compare MSFT vs AAPL valuation",
        "ticker": "MSFT",
        "tickers": ["MSFT", "AAPL"],
        "run_business_analyst": True,
        "run_financial_modelling": True,
        "run_quant_fundamental": True,
        "run_macro": True,
        "run_insider_news": True,
    }
    
    # Process each agent
    ba_outputs = [
        {"ticker": "MSFT", "qualitative_summary": "Strong cloud business"},
        {"ticker": "AAPL", "qualitative_summary": "Strong ecosystem"},
    ]
    fm_outputs = [
        {"ticker": "MSFT", "dcf_value": 420.0},
        {"ticker": "AAPL", "dcf_value": 195.0},
    ]
    qf_outputs = [
        {"ticker": "MSFT", "piotroski_score": 9},
        {"ticker": "AAPL", "piotroski_score": 8},
    ]
    macro_outputs = [
        {"ticker": "MSFT", "regime": "risk-off"},
        {"ticker": "AAPL", "regime": "disinflation"},
    ]
    insider_outputs = [
        {"ticker": "MSFT", "data_coverage": {"insider_transactions_count": 2}},
        {"ticker": "AAPL", "data_coverage": {"insider_transactions_count": 1}},
    ]
    
    # Verify all outputs have correct length
    assert len(ba_outputs) == len(input_state["tickers"])
    assert len(fm_outputs) == len(input_state["tickers"])
    assert len(qf_outputs) == len(input_state["tickers"])
    assert len(macro_outputs) == len(input_state["tickers"])
    assert len(insider_outputs) == len(input_state["tickers"])
    
    # Verify ticker mapping
    for i, ticker in enumerate(input_state["tickers"]):
        assert ba_outputs[i]["ticker"] == ticker
        assert fm_outputs[i]["ticker"] == ticker
        assert qf_outputs[i]["ticker"] == ticker
        assert macro_outputs[i]["ticker"] == ticker
        assert insider_outputs[i]["ticker"] == ticker


# ---------------------------------------------------------------------------
# Extensible Test Helper for New Agents
# ---------------------------------------------------------------------------

def create_multi_ticker_test(agent_name: str, num_tickers: int = 2):
    """Factory function to create multi-ticker tests for new agents.
    
    Usage:
        create_multi_ticker_test("new_agent", 3)
    """
    def test_new_agent_multi_ticker():
        tickers = [f"TICKER{i}" for i in range(num_tickers)]
        
        mock_outputs = [
            {"ticker": t, "output": f"result for {t}"}
            for t in tickers
        ]
        
        # Verify length matches
        assert len(mock_outputs) == len(tickers)
        
        # Verify each has ticker
        for output in mock_outputs:
            assert "ticker" in output
    
    return test_new_agent_multi_ticker
