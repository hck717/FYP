"""Test fixtures and configuration for the FYP testing suite.

This conftest provides:
- Mock fixtures for database connections (PostgreSQL, Neo4j)
- Mock LLM responses for prompt testing
- Shared test data (tickers, sample chunks, agent outputs)
- Agent registry for extensible testing of new agents

The testing framework is designed to be easily extensible:
- Add new agents to AGENT_REGISTRY to test them automatically
- Add new nodes to NODE_REGISTRY for graph node testing
- Modify MOCK_* fixtures to change mock behavior globally
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Test Configuration
# ---------------------------------------------------------------------------

TEST_TICKER = "AAPL"
TEST_TICKER_2 = "MSFT"
TEST_TICKER_3 = "TSLA"

# All available agents in the system (extensible registry)
AGENT_REGISTRY = {
    "business_analyst": {
        "module": "agents.business_analyst.agent",
        "node_name": "node_business_analyst",
        "output_key": "business_analyst_outputs",
        "requires_db": ["neo4j", "postgres"],
    },
    "quant_fundamental": {
        "module": "agents.quant_fundamental.agent",
        "node_name": "node_quant_fundamental",
        "output_key": "quant_fundamental_outputs",
        "requires_db": ["postgres"],
    },
    "web_search": {
        "module": "agents.web_search.agent",
        "node_name": "node_web_search",
        "output_key": "web_search_outputs",
        "requires_db": [],
    },
    "financial_modelling": {
        "module": "agents.financial_modelling.agent",
        "node_name": "node_financial_modelling",
        "output_key": "financial_modelling_outputs",
        "requires_db": ["postgres", "neo4j"],
    },
    "stock_research": {
        "module": "agents.stock_research_agent.agent",
        "node_name": "node_stock_research",
        "output_key": "stock_research_outputs",
        "requires_db": ["neo4j", "postgres"],
    },
}

# All nodes in the orchestration graph (extensible registry)
NODE_REGISTRY = {
    "planner": "orchestration.nodes.node_planner",
    "summarizer": "orchestration.nodes.node_summarizer",
    "post_processing": "orchestration.nodes.node_post_processing",
    **{v["node_name"]: f"orchestration.nodes.{v['node_name']}" for v in AGENT_REGISTRY.values()},
}

# ---------------------------------------------------------------------------
# Database Connection Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_pg_conn():
    """Mock PostgreSQL connection."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    return conn


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver."""
    driver = MagicMock()
    session = MagicMock()
    result = MagicMock()
    session.run.return_value = result
    result.single.return_value = {"name": "chunk_embedding"}
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver


# ---------------------------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def test_env():
    """Set test environment variables."""
    original_env = dict(os.environ)
    os.environ.update({
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "test_airflow",
        "POSTGRES_USER": "test",
        "POSTGRES_PASSWORD": "test",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "test",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "EMBEDDING_MODEL": "nomic-embed-text",
    })
    yield
    os.environ.clear()
    os.environ.update(original_env)


# ---------------------------------------------------------------------------
# Mock Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_neo4j_chunk():
    """Sample Neo4j chunk for testing."""
    return {
        "chunk_id": "chunk_abc123",
        "text": "Apple has a strong competitive moat through its brand, ecosystem integration, and loyal customer base.",
        "source": "broker_report",
        "ticker": "AAPL",
        "published_date": "2024-01-15",
    }


@pytest.fixture
def sample_neo4j_chunks():
    """Multiple Neo4j chunks for testing."""
    return [
        {
            "chunk_id": "chunk_abc123",
            "text": "Apple has a strong competitive moat through its brand, ecosystem integration, and loyal customer base.",
            "source": "broker_report",
            "ticker": "AAPL",
            "published_date": "2024-01-15",
        },
        {
            "chunk_id": "chunk_def456",
            "text": "The iPhone remains the flagship product, generating over 50% of total revenue.",
            "source": "earnings_call",
            "ticker": "AAPL",
            "published_date": "2024-02-01",
        },
    ]


@pytest.fixture
def sample_financial_data():
    """Sample financial statement data from PostgreSQL."""
    return {
        "ticker": "AAPL",
        "period": "2024-Q1",
        "revenue": 119600000000,
        "net_income": 33920000000,
        "total_assets": 352755000000,
        "total_equity": 62058000000,
        "operating_cash_flow": 37860000000,
    }


@pytest.fixture
def sample_valuation_data():
    """Sample valuation metrics from PostgreSQL."""
    return {
        "ticker": "AAPL",
        "pe_ratio": 28.5,
        "ev_ebitda": 22.3,
        "price_to_book": 45.2,
        "dividend_yield": 0.52,
    }


# ---------------------------------------------------------------------------
# Mock Agent Output Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ba_output():
    """Mock Business Analyst agent output."""
    return {
        "ticker": "AAPL",
        "qualitative_summary": "Apple demonstrates strong moat characteristics through brand strength and ecosystem lock-in.",
        "qualitative_analysis": {
            "moat_factors": ["brand_loyalty", "ecosystem", "pricing_power"],
            "risks": ["regulatory", "China_dependency"],
            "narrative": "Apple's competitive moat is sustained by high switching costs and ecosystem integration.",
        },
        "sentiment_verdict": {"label": "positive", "confidence": 0.85},
        "citations": [{"chunk_id": "chunk_abc123", "relevance": 0.92}],
    }


@pytest.fixture
def mock_qf_output():
    """Mock Quant Fundamental agent output."""
    return {
        "ticker": "AAPL",
        "piotroski_score": 8,
        "beneish_m_score": -2.1,
        "altman_z": 4.5,
        "quantitative_summary": "Strong fundamentals with high profitability scores.",
        "factor_scores": {"quality": 0.8, "value": 0.6, "momentum": 0.7},
    }


@pytest.fixture
def mock_ws_output():
    """Mock Web Search agent output."""
    return {
        "ticker": "AAPL",
        "breaking_news": ["Apple announces new AI features"],
        "sentiment_rationale": "Positive sentiment from recent product announcements.",
    }


@pytest.fixture
def mock_fm_output():
    """Mock Financial Modelling agent output."""
    return {
        "ticker": "AAPL",
        "dcf_value": 195.50,
        "wacc": 0.095,
        "terminal_growth": 0.025,
        "quantitative_summary": "DCF suggests fair value of $195.50 based on 5-year projections.",
    }


@pytest.fixture
def mock_sr_output():
    """Mock Stock Research agent output."""
    return {
        "ticker": "AAPL",
        "broker_consensus": "Outperform",
        "transcript_comparison": "Management optimistic about services growth.",
    }


@pytest.fixture
def mock_all_agent_outputs():
    """Complete mock of all agent outputs for summarizer testing."""
    return {
        "ticker": "AAPL",
        "tickers": ["AAPL"],
        "business_analyst_outputs": [{}],
        "quant_fundamental_outputs": [{}],
        "web_search_outputs": [{}],
        "financial_modelling_outputs": [{}],
        "stock_research_outputs": [{}],
    }


@pytest.fixture
def mock_multi_ticker_outputs():
    """Mock outputs for multi-ticker comparison."""
    return {
        "tickers": ["MSFT", "AAPL"],
        "business_analyst_outputs": [
            {"ticker": "MSFT", "qualitative_summary": "Microsoft has strong cloud moat."},
            {"ticker": "AAPL", "qualitative_summary": "Apple has strong ecosystem moat."},
        ],
        "financial_modelling_outputs": [
            {"ticker": "MSFT", "dcf_value": 420.0},
            {"ticker": "AAPL", "dcf_value": 195.0},
        ],
    }


# ---------------------------------------------------------------------------
# Mock LLM Response Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_planner_response():
    """Mock planner LLM response."""
    return {
        "ticker": "AAPL",
        "run_business_analyst": True,
        "run_quant_fundamental": True,
        "run_web_search": False,
        "run_financial_modelling": True,
        "run_stock_research": True,
        "react_max_iterations": 2,
        "output_language": None,
    }


@pytest.fixture
def mock_summarizer_response():
    """Mock summarizer LLM response."""
    return {
        "investment_thesis": "Strong buy based on moat and valuation.",
        "valuation": "DCF suggests $195 fair value.",
        "risks": ["regulatory", "China dependency"],
        "final_summary": "Apple represents a strong investment opportunity...",
    }


# ---------------------------------------------------------------------------
# Helper Fixtures for Extensibility
# ---------------------------------------------------------------------------

def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """Get configuration for a specific agent."""
    return AGENT_REGISTRY.get(agent_name, {})


def get_node_config(node_name: str) -> Optional[str]:
    """Get module path for a specific node."""
    return NODE_REGISTRY.get(node_name)


@pytest.fixture
def agent_registry():
    """Return the agent registry for introspection."""
    return AGENT_REGISTRY


@pytest.fixture
def node_registry():
    """Return the node registry for introspection."""
    return NODE_REGISTRY


# ---------------------------------------------------------------------------
# Pytest Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require live services)"
    )
    config.addinivalue_line(
        "markers", "prompt: marks tests as prompt/LLM tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (no external dependencies)"
    )


# ---------------------------------------------------------------------------
# Citation Test Data
# ---------------------------------------------------------------------------

# Sample citation data for broker research, earnings calls, macro reports
SAMPLE_BROKER_REPORT_CITATION = {
    "chunk_id": "broker_chunk_001",
    "text": "Goldman Sachs maintains Buy rating on AAPL with price target $220.",
    "source_type": "broker_report",
    "ticker": "AAPL",
    "broker": "Goldman Sachs",
    "rating": "Buy",
    "price_target": 220.0,
}

SAMPLE_EARNINGS_TRANSCRIPT_CITATION = {
    "chunk_id": "transcript_chunk_001",
    "text": "CEO Tim Cook: 'Services revenue grew 16% year-over-year to a record $23.1B'.",
    "source_type": "earnings_call",
    "ticker": "AAPL",
    "quarter": "2024-Q1",
    "speaker": "Tim Cook",
    "date": "2024-01-25",
}

SAMPLE_MACRO_REPORT_CITATION = {
    "chunk_id": "macro_chunk_001",
    "text": "Fed signals potential rate cuts in Q2 2024.",
    "source_type": "macro_report",
    "ticker": "MARKET",
    "report_type": "fed_meeting_notes",
    "date": "2024-01-31",
}


@pytest.fixture
def sample_citations():
    """All sample citation types for testing."""
    return [
        SAMPLE_BROKER_REPORT_CITATION,
        SAMPLE_EARNINGS_TRANSCRIPT_CITATION,
        SAMPLE_MACRO_REPORT_CITATION,
    ]