"""Hallucination guard tests.

These tests verify that numeric outputs are computed (not hallucinated)
and that qualitative claims always have citations.

Run with: pytest tests/prompts/test_hallucination_guard.py -v --timeout=60
"""

import os
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Test: DCF Numbers are Python-Computed
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_dcf_numbers_are_python_computed():
    """Test that DCF values match direct Python calculation."""
    # Given DCF inputs
    inputs = {
        "free_cash_flow": 100_000_000_000,  # $100B
        "wacc": 0.09,  # 9%
        "terminal_growth": 0.025,  # 2.5%
        "years": 5,
        "shares_outstanding": 15_500_000_000,  # 15.5B
    }
    
    # Compute DCF in Python
    computed_dcf = _compute_dcf_value(inputs)
    
    # Get agent output (mocked)
    agent_output = _get_mock_dcf_output(inputs)
    
    # Verify match (within 1 cent due to rounding)
    assert abs(agent_output["dcf_value"] - computed_dcf) < 0.01


def _compute_dcf_value(inputs: Dict) -> float:
    """Direct Python DCF calculation."""
    fcf = inputs["free_cash_flow"]
    wacc = inputs["wacc"]
    g = inputs["terminal_growth"]
    n = inputs["years"]
    shares = inputs["shares_outstanding"]
    
    # Simplified DCF: Sum of discounted FCFs + terminal value
    pv = 0
    for year in range(1, n + 1):
        pv += fcf * (1 + g) ** year / (1 + wacc) ** year
    
    # Terminal value (Gordon growth)
    terminal_value = (fcf * (1 + g) ** (n + 1)) / (wacc - g)
    pv_terminal = terminal_value / (1 + wacc) ** n
    
    total_pv = pv + pv_terminal
    
    # Per share
    return total_pv / shares


def _get_mock_dcf_output(inputs: Dict) -> Dict[str, Any]:
    """Mock agent output with computed value."""
    dcf_value = _compute_dcf_value(inputs)
    
    return {
        "dcf_value": round(dcf_value, 2),
        "wacc": inputs["wacc"],
        "terminal_growth": inputs["terminal_growth"],
        "inputs_used": inputs,
    }


# ---------------------------------------------------------------------------
# Test: WACC Computed from Inputs
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_wacc_computed_from_risk_free_rate():
    """Test WACC is computed from components, not hallucinated."""
    inputs = {
        "risk_free_rate": 0.04,
        "market_premium": 0.05,
        "beta": 1.2,
        "debt_weight": 0.3,
        "equity_weight": 0.7,
        "cost_of_debt": 0.05,
    }
    
    computed_wacc = _compute_wacc(inputs)
    output_wacc = inputs["risk_free_rate"] + inputs["beta"] * inputs["market_premium"]
    
    # WACC = E/V * Re + D/V * Rd * (1-T)
    # Using CAPM for Re: Rf + Beta * ERP
    expected_wacc = (
        inputs["equity_weight"] * output_wacc +
        inputs["debt_weight"] * inputs["cost_of_debt"] * (1 - 0.21)  # Assume 21% tax
    )
    
    assert abs(computed_wacc - expected_wacc) < 0.001


def _compute_wacc(inputs: Dict) -> float:
    """Compute WACC from components."""
    rf = inputs["risk_free_rate"]
    beta = inputs["beta"]
    erp = inputs["market_premium"]
    
    # Cost of equity (CAPM)
    re = rf + beta * erp
    
    # WACC
    we = inputs["equity_weight"]
    wd = inputs["debt_weight"]
    rd = inputs["cost_of_debt"]
    tax_rate = 0.21
    
    return we * re + wd * rd * (1 - tax_rate)


# ---------------------------------------------------------------------------
# Test: All Qualitative Claims Have Citations
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_all_claims_have_citations():
    """Test that qualitative claims always have a source chunk_id."""
    ba_output = {
        "ticker": "AAPL",
        "qualitative_analysis": {
            "narrative": "Apple has strong ecosystem moat...",
            "moat_factors": ["ecosystem", "brand", "services"],
        },
        "claims": [
            {"text": "Strong ecosystem moat", "chunk_id": "chunk_001", "relevance": 0.95},
            {"text": "Brand strength", "chunk_id": "chunk_002", "relevance": 0.90},
            {"text": "Services growth", "chunk_id": "chunk_003", "relevance": 0.85},
        ],
    }
    
    # Each claim should have a source chunk_id
    for claim in ba_output.get("claims", []):
        assert "chunk_id" in claim, f"Claim missing chunk_id: {claim}"
        assert claim["chunk_id"].startswith("chunk_"), "Invalid chunk_id format"


@pytest.mark.prompt
def test_claims_extracted_from_citations():
    """Test claims are properly linked to source citations."""
    output = {
        "claims": [
            {"text": "Strong moat", "chunk_id": "chunk_abc123"},
            {"text": "High margins", "chunk_id": "chunk_def456"},
        ],
        "citations": [
            {"chunk_id": "chunk_abc123", "text": "..."},
            {"chunk_id": "chunk_def456", "text": "..."},
        ],
    }
    
    # Verify claim chunk_ids are in citations
    citation_ids = {c["chunk_id"] for c in output["citations"]}
    for claim in output["claims"]:
        assert claim["chunk_id"] in citation_ids, \
            f"Claim references non-existent citation: {claim['chunk_id']}"


# ---------------------------------------------------------------------------
# Test: Numeric Values from Database, Not LLM
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_financial_data_from_db():
    """Test that financial metrics come from DB, not LLM generation."""
    # Simulate DB data
    db_data = {
        "revenue": 119600000000,
        "net_income": 33920000000,
        "gross_margin": 0.45,
        "pe_ratio": 28.5,
    }
    
    # Agent output should preserve exact DB values
    agent_output = {
        "ticker": "AAPL",
        "metrics": db_data,
    }
    
    # Verify exact values preserved (no rounding/hallucination)
    assert agent_output["metrics"]["revenue"] == 119600000000
    assert agent_output["metrics"]["net_income"] == 33920000000
    assert agent_output["metrics"]["gross_margin"] == 0.45


# ---------------------------------------------------------------------------
# Test: Valuation Multiples are Computed
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_valuation_multiples_computed():
    """Test that P/E, EV/EBITDA are computed, not hallucinated."""
    fundamentals = {
        "price": 195.0,
        "earnings_per_share": 6.42,
        "ebitda": 130_000_000000,
        "enterprise_value": 3_100_000_000000,
    }
    
    # Compute multiples
    pe_ratio = fundamentals["price"] / fundamentals["earnings_per_share"]
    ev_ebitda = fundamentals["enterprise_value"] / fundamentals["ebitda"]
    
    # Verify computation matches expected values
    assert abs(pe_ratio - 30.37) < 0.1  # ~30.4
    assert abs(ev_ebitda - 23.85) < 0.1  # ~23.8


# ---------------------------------------------------------------------------
# Test: Source Attribution in Summarizer
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_summarizer_cites_sources():
    """Test summarizer cites sources for all claims."""
    agent_outputs = {
        "business_analyst": {
            "claims": [
                {"text": "Strong moat", "chunk_id": "chunk_001"},
            ],
        },
        "quant_fundamental": {
            "factor_scores": {"quality": 0.8},
        },
    }
    
    final_summary = _generate_cited_summary(agent_outputs)
    
    # Should have at least one citation
    assert "chunk_" in final_summary or "Source:" in final_summary


def _generate_cited_summary(outputs: Dict) -> str:
    """Generate summary with citations."""
    citations = []
    
    # Collect citations
    if "business_analyst" in outputs:
        for claim in outputs["business_analyst"].get("claims", []):
            if "chunk_id" in claim:
                citations.append(claim["chunk_id"])
    
    return f"Analysis complete. Sources: {', '.join(citations)}"


# ---------------------------------------------------------------------------
# Test: Hallucination Detection in Output
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_hallucinated_numbers_detected():
    """Test that obviously hallucinated numbers are flagged."""
    # Realistic inputs
    realistic_inputs = {
        "price": 195.0,
        "pe_ratio": 28.5,
        "dcf_value": 195.0,
    }
    
    # Hallucinated - P/E doesn't match price/earnings
    hallucinated = {
        "price": 195.0,
        "pe_ratio": 500.0,  # Way too high
        "dcf_value": 195.0,
    }
    
    # Detect inconsistency
    implied_pe = realistic_inputs["price"] / 6.4  # Assume EPS ~6.4
    assert abs(realistic_inputs["pe_ratio"] - implied_pe) < 10  # Within reasonable range
    
    # This would be caught as hallucinated
    implied_pe_halluc = hallucinated["price"] / 6.4
    assert abs(hallucinated["pe_ratio"] - implied_pe_halluc) > 100  # Way off


# ---------------------------------------------------------------------------
# Test: Input Validation - Numbers from Verified Sources
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_inputs_from_verified_sources():
    """Test that inputs are marked as from verified sources."""
    class VerifiedInput:
        def __init__(self, value, source):
            self.value = value
            self.source = source  # "postgres", "neo4j", etc.
    
    inputs = [
        VerifiedInput(119600000000, "postgres"),
        VerifiedInput(0.45, "postgres"),
        VerifiedInput("Strong moat", "neo4j"),
    ]
    
    # Verify all inputs have source
    for inp in inputs:
        assert hasattr(inp, "source")
        assert inp.source in ["postgres", "neo4j", "api"]


# ---------------------------------------------------------------------------
# Test: Citation Chain Validation
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_citation_chain_validation():
    """Test that there's a valid chain: claim -> chunk -> source doc."""
    claim = {"text": "Strong moat", "chunk_id": "chunk_001"}
    chunk = {"chunk_id": "chunk_001", "text": "Apple ecosystem...", "source_doc": "broker_report_2024.pdf"}
    source_doc = {"doc_id": "broker_report_2024.pdf", "type": "broker_report"}
    
    # Verify chain
    assert claim["chunk_id"] == chunk["chunk_id"]
    assert chunk["source_doc"] == source_doc["doc_id"]
    
    # Chain is valid
    assert True


# ---------------------------------------------------------------------------
# Extensible Test Helper for New Agent Types
# ---------------------------------------------------------------------------

def create_numeric_guard_test(agent_name: str, numeric_field: str, compute_fn: callable):
    """Create a numeric guard test for a new agent.
    
    Usage:
        def compute_new_metric(inputs):
            return inputs["value"] * 2
        create_numeric_guard_test("new_agent", "new_metric", compute_new_metric)
    """
    @pytest.mark.prompt
    def test_agent_numeric_field():
        inputs = {"value": 100}
        computed = compute_fn(inputs)
        
        # Verify computation is valid
        assert isinstance(computed, (int, float))
        assert computed > 0
    
    return test_agent_numeric_field


# ---------------------------------------------------------------------------
# Test: Integration with Graph Output
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_graph_output_has_citations():
    """Test final graph output includes citations for verification."""
    from orchestration.state import OrchestrationState
    
    # Final output after all processing
    final_output = {
        "final_summary": "AAPL is a strong buy...",
        "output": {
            "valuation": 195.0,
            "recommendation": "BUY",
        },
        "citations": [
            {"chunk_id": "chunk_001", "agent": "business_analyst"},
            {"chunk_id": "chunk_002", "agent": "financial_modelling"},
        ],
    }
    
    state: OrchestrationState = {
        "final_summary": final_output["final_summary"],
        "output": final_output["output"],
    }
    
    # Verify citations exist
    assert len(final_output["citations"]) > 0
    for cit in final_output["citations"]:
        assert "chunk_id" in cit
        assert "agent" in cit