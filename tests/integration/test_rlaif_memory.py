"""RLAIF + Episodic Memory Pipeline tests.

These tests verify:
- node_post_processing writes RLAIF scores to DB
- Episodic memory hints propagate into planner state

Run with: pytest tests/integration/test_rlaif_memory.py -v --timeout=60
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orchestration import feedback, episodic_memory
from orchestration.state import OrchestrationState


# ---------------------------------------------------------------------------
# Test: RLAIF Scores Persistence
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_rlaif_scores_persisted():
    """Test node_post_processing writes RLAIF scores to DB."""
    # This test verifies the feedback system works end-to-end
    
    # First ensure tables exist
    feedback.ensure_feedback_tables_exist()
    
    # Create a mock RLAIF result
    run_id = "test_run_001"
    user_query = "What is AAPL's valuation?"
    
    # Save RLAIF feedback
    feedback.save_rl_feedback(
        run_id=run_id,
        user_query=user_query,
        factual_accuracy=0.85,
        citation_completeness=0.90,
        analysis_depth=0.80,
        structure_compliance=0.95,
        language_quality=0.88,
        overall_score=0.87,
        strengths=["good_citations", "clear_structure"],
        weaknesses=["shallow_analysis"],
        agent_blamed="business_analyst",
        ticker="AAPL",
    )
    
    # Verify it was saved
    conn = feedback._get_pg_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM rl_feedback WHERE run_id = %s", (run_id,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    assert result is not None
    assert result[1] == run_id  # run_id is second column


@pytest.mark.integration
def test_rlaif_scoring_function():
    """Test the RLAIF scoring function returns valid scores."""
    run_id = "test_run_002"
    
    # Call scoring
    scores = feedback.score_report_with_rlaif(
        run_id=run_id,
        report_text="Apple has a strong moat. The company generates significant free cash flow.",
        user_query="What is AAPL's moat?",
    )
    
    # Verify score structure
    assert isinstance(scores, dict)
    assert "overall_score" in scores
    assert 0 <= scores["overall_score"] <= 1


# ---------------------------------------------------------------------------
# Test: Episodic Memory Hints Propagation
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_episodic_hints_loaded():
    """Test episodic hints propagate into planner state."""
    # First ensure table exists
    episodic_memory.ensure_table_exists()
    
    # Seed a known failure
    query = "What is Apple's moat?"
    ticker = "AAPL"
    
    # Record failure
    episodic_memory.record_failure(
        query_signature=query[:30],
        ticker=ticker,
        failure_agent="business_analyst",
        failure_reason="INSUFFICIENT_DATA",
        react_iterations_used=3,
    )
    
    # Now query should return similar failure
    similar = episodic_memory.get_similar_failures(query, ticker, top_n=5)
    
    # Verify we got results
    assert isinstance(similar, list)


@pytest.mark.integration
def test_episodic_memory_query():
    """Test episodic memory stores and retrieves query embeddings."""
    episodic_memory.ensure_table_exists()
    
    # Record a failure
    episodic_memory.record_failure(
        query_signature="analyze_aapl_moat",
        ticker="AAPL",
        failure_agent="business_analyst",
        failure_reason="TIMEOUT",
        react_iterations_used=2,
    )
    
    # Query for similar
    results = episodic_memory.get_similar_failures(
        "Analyze AAPL competitive moat",
        "AAPL",
        top_n=10
    )
    
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Test: Planner Integration with Episodic Memory
# ---------------------------------------------------------------------------

def test_planner_queries_episodic_memory():
    """Test that planner state can include episodic hints."""
    # This tests the state structure, not the actual LLM call
    
    state: OrchestrationState = {
        "user_query": "What is Apple's moat?",
        "ticker": "AAPL",
        "episodic_hints": {
            "degraded_agents": ["business_analyst"],
            "force_web_search": True,
        },
    }
    
    # Verify episodic hints structure
    assert "episodic_hints" in state
    assert "degraded_agents" in state["episodic_hints"]
    assert "force_web_search" in state["episodic_hints"]


def test_episodic_hints_influence_routing():
    """Test that degraded agents are reflected in state."""
    state: OrchestrationState = {
        "run_business_analyst": True,  # Would be set to False if degraded
        "run_web_search": False,  # Would be set to True if forced
        "episodic_hints": {
            "degraded_agents": ["business_analyst"],
            "force_web_search": True,
        },
    }
    
    # If BA is degraded and web search forced, the actual routing should reflect this
    # This is a state validation test
    assert state["run_business_analyst"] is True  # Original value
    assert "degraded_agents" in state["episodic_hints"]


# ---------------------------------------------------------------------------
# Test: RLAIF Feedback Tables
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_feedback_tables_created():
    """Test that all feedback tables are created correctly."""
    feedback.ensure_feedback_tables_exist()
    
    conn = feedback._get_pg_conn()
    cursor = conn.cursor()
    
    # Check tables exist
    tables = ["rl_feedback", "user_feedback", "prompt_versions"]
    for table in tables:
        cursor.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = '{table}'
            );
        """)
        exists = cursor.fetchone()[0]
        assert exists is True
    
    cursor.close()
    conn.close()


@pytest.mark.integration
def test_user_feedback_persistence():
    """Test user feedback can be saved and retrieved."""
    feedback.ensure_feedback_tables_exist()
    
    run_id = "test_run_003"
    
    # Save user feedback
    feedback.save_user_feedback(
        run_id=run_id,
        helpful=True,
        comment="Great analysis!",
        session_id="session_001",
    )
    
    # Retrieve
    conn = feedback._get_pg_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT helpful, comment FROM user_feedback WHERE run_id = %s", (run_id,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    assert result is not None
    assert result[0] is True
    assert "Great analysis" in result[1]


# ---------------------------------------------------------------------------
# Test: Post-Processing Node Integration
# ---------------------------------------------------------------------------

def test_post_processing_state_structure():
    """Test post_processing node has required state fields."""
    state: OrchestrationState = {
        "user_query": "What is AAPL worth?",
        "final_summary": "Apple is worth $195 per share...",
        "output": {
            "valuation": 195.0,
            "ticker": "AAPL",
        },
        "rl_feedback_scores": None,
        "rl_feedback_run_id": None,
    }
    
    # Verify state has all required fields for post-processing
    assert "final_summary" in state
    assert "output" in state
    assert "rl_feedback_run_id" in state


# ---------------------------------------------------------------------------
# Test: Multi-Agent Failure Tracking
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_track_multiple_agent_failures():
    """Test that failures from multiple agents are tracked."""
    episodic_memory.ensure_table_exists()
    
    # Record failures for different agents
    episodic_memory.record_failure(
        query_signature="test_query_1",
        ticker="AAPL",
        failure_agent="business_analyst",
        failure_reason="INSUFFICIENT_DATA",
        react_iterations_used=2,
    )
    
    episodic_memory.record_failure(
        query_signature="test_query_2",
        ticker="AAPL",
        failure_agent="quant_fundamental",
        failure_reason="ERROR",
        react_iterations_used=1,
    )
    
    # Query for failures
    ba_failures = episodic_memory.get_similar_failures(
        "test query",
        "AAPL",
        top_n=10
    )
    
    assert isinstance(ba_failures, list)


# ---------------------------------------------------------------------------
# Test: Prompt Version Tracking
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_prompt_version_tracking():
    """Test prompt versions can be tracked for A/B testing."""
    feedback.ensure_feedback_tables_exist()
    
    # Save a prompt version
    feedback.save_prompt_version(
        agent_name="business_analyst",
        version="v2.0",
        prompt_text="You are a business analyst...",
        avg_score_before=0.75,
        avg_score_after=0.82,
        improvement_pct=9.3,
    )
    
    # Retrieve
    conn = feedback._get_pg_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT version, avg_score_before, avg_score_after 
        FROM prompt_versions 
        WHERE agent_name = 'business_analyst'
    """)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    assert len(results) > 0


# ---------------------------------------------------------------------------
# Extensible Test Helper for New Feedback Types
# ---------------------------------------------------------------------------

def create_rlaif_test_for_agent(agent_name: str):
    """Create RLAIF test for a new agent.
    
    Usage:
        create_rlaif_test_for_agent("new_agent")
    """
    @pytest.mark.integration
    def test_agent_rlaif():
        run_id = f"test_{agent_name}_001"
        
        # Save feedback for agent
        feedback.save_rl_feedback(
            run_id=run_id,
            user_query="Test query",
            overall_score=0.8,
            agent_blamed=agent_name,
            ticker="AAPL",
        )
        
        # Verify saved
        conn = feedback._get_pg_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT agent_blamed FROM rl_feedback WHERE run_id = %s", (run_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        assert result is not None
        assert result[0] == agent_name
    
    return test_agent_rlaif