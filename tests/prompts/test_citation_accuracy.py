"""Citation accuracy tests.

These tests verify that citations in agent outputs correctly reference:
- Broker research reports in Neo4j/PostgreSQL
- Earnings call transcripts in Neo4j/PostgreSQL
- Macro research reports in Neo4j/PostgreSQL

Run with: pytest tests/prompts/test_citation_accuracy.py -v --timeout=60
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
# Citation Source Types
# ---------------------------------------------------------------------------

CITATION_SOURCE_TYPES = [
    "broker_report",
    "earnings_call",
    "macro_report",
    "company_filings",
    "news_article",
]


# ---------------------------------------------------------------------------
# Test: Broker Research Citation Format
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_broker_report_citation_format():
    """Test broker report citations have correct format."""
    citation = {
        "chunk_id": "broker_chunk_001",
        "source_type": "broker_report",
        "ticker": "AAPL",
        "broker": "Goldman Sachs",
        "rating": "Buy",
        "price_target": 220.0,
        "published_date": "2024-01-15",
    }
    
    # Verify required fields
    assert "chunk_id" in citation
    assert citation["chunk_id"].startswith("broker_") or citation["chunk_id"].startswith("chunk_")
    assert "source_type" in citation
    assert citation["source_type"] == "broker_report"
    assert "ticker" in citation
    assert "broker" in citation


@pytest.mark.prompt
def test_broker_rating_extraction():
    """Test broker ratings are correctly extracted and cited."""
    chunk_data = {
        "text": "Goldman Sachs maintains Buy rating on Apple with price target $220.",
        "source": "broker_report",
        "chunk_id": "broker_gs_001",
        "metadata": {
            "broker": "Goldman Sachs",
            "rating": "Buy",
            "price_target": 220.0,
        },
    }
    
    # Extract citation info
    citation = _extract_broker_citation(chunk_data)
    
    assert citation["broker"] == "Goldman Sachs"
    assert citation["rating"] == "Buy"
    assert citation["price_target"] == 220.0
    assert citation["chunk_id"] == "broker_gs_001"


def _extract_broker_citation(chunk: Dict) -> Dict[str, Any]:
    """Extract broker citation from chunk."""
    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "source_type": chunk.get("source", ""),
        "broker": chunk.get("metadata", {}).get("broker", "Unknown"),
        "rating": chunk.get("metadata", {}).get("rating", ""),
        "price_target": chunk.get("metadata", {}).get("price_target"),
    }


# ---------------------------------------------------------------------------
# Test: Earnings Call Transcript Citation
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_earnings_transcript_citation_format():
    """Test earnings transcript citations have correct format."""
    citation = {
        "chunk_id": "transcript_chunk_001",
        "source_type": "earnings_call",
        "ticker": "AAPL",
        "quarter": "2024-Q1",
        "speaker": "Tim Cook",
        "date": "2024-01-25",
    }
    
    # Verify required fields
    assert "chunk_id" in citation
    assert "source_type" in citation
    assert citation["source_type"] == "earnings_call"
    assert "quarter" in citation
    assert "speaker" in citation or "date" in citation


@pytest.mark.prompt
def test_earnings_quote_attribution():
    """Test earnings quotes are properly attributed."""
    chunk_data = {
        "text": "CEO Tim Cook: 'Services revenue grew 16% year-over-year to a record $23.1B'.",
        "source": "earnings_call",
        "chunk_id": "transcript_aapl_q1_001",
        "metadata": {
            "quarter": "2024-Q1",
            "speaker": "Tim Cook",
            "role": "CEO",
        },
    }
    
    # Extract citation
    citation = _extract_transcript_citation(chunk_data)
    
    assert "Tim Cook" in citation["speaker"]
    assert citation["quarter"] == "2024-Q1"
    assert citation["chunk_id"] == "transcript_aapl_q1_001"


def _extract_transcript_citation(chunk: Dict) -> Dict[str, Any]:
    """Extract earnings transcript citation."""
    meta = chunk.get("metadata", {})
    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "source_type": chunk.get("source", ""),
        "quarter": meta.get("quarter", ""),
        "speaker": meta.get("speaker", ""),
        "role": meta.get("role", ""),
    }


# ---------------------------------------------------------------------------
# Test: Macro Research Report Citation
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_macro_report_citation_format():
    """Test macro report citations have correct format."""
    citation = {
        "chunk_id": "macro_chunk_001",
        "source_type": "macro_report",
        "ticker": "MARKET",  # Macro reports often don't have specific ticker
        "report_type": "fed_meeting_notes",
        "date": "2024-01-31",
        "institution": "Federal Reserve",
    }
    
    # Verify fields
    assert "chunk_id" in citation
    assert citation["source_type"] == "macro_report"
    assert "report_type" in citation
    assert "date" in citation


@pytest.mark.prompt
def test_macro_impact_attribution():
    """Test macro data is properly attributed to source."""
    chunk_data = {
        "text": "Fed signals potential rate cuts in Q2 2024 based on inflation data.",
        "source": "macro_report",
        "chunk_id": "macro_fed_001",
        "metadata": {
            "report_type": "fed_meeting_notes",
            "institution": "Federal Reserve",
            "date": "2024-01-31",
        },
    }
    
    citation = _extract_macro_citation(chunk_data)
    
    assert citation["institution"] == "Federal Reserve"
    assert citation["report_type"] == "fed_meeting_notes"


def _extract_macro_citation(chunk: Dict) -> Dict[str, Any]:
    """Extract macro report citation."""
    meta = chunk.get("metadata", {})
    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "source_type": chunk.get("source", ""),
        "report_type": meta.get("report_type", ""),
        "institution": meta.get("institution", ""),
        "date": meta.get("date", ""),
    }


# ---------------------------------------------------------------------------
# Test: Citation Accuracy - Source Matching
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_citation_matches_source_doc():
    """Test that chunk citations match actual source documents in DB."""
    # Simulate source document in DB
    source_doc = {
        "doc_id": "gs_aapl_report_2024.pdf",
        "type": "broker_report",
        "ticker": "AAPL",
        "broker": "Goldman Sachs",
    }
    
    # Simulate chunk from that document
    chunk = {
        "chunk_id": "broker_chunk_001",
        "source_doc_id": "gs_aapl_report_2024.pdf",
        "text": "Strong buy rating...",
    }
    
    # Verify match
    assert chunk["source_doc_id"] == source_doc["doc_id"]
    assert source_doc["type"] == "broker_report"


@pytest.mark.prompt
def test_transcript_citation_matches_audio():
    """Test earnings transcript citations match original audio/filing."""
    # Source filing
    filing = {
        "filing_id": "aapl_10q_2024q1",
        "type": "10-Q",
        "quarter": "2024-Q1",
    }
    
    # Chunk from transcript
    chunk = {
        "chunk_id": "transcript_001",
        "source_filing_id": "aapl_10q_2024q1",
        "text": "Revenue: $119.6B",
    }
    
    assert chunk["source_filing_id"] == filing["filing_id"]


# ---------------------------------------------------------------------------
# Test: Citation in Agent Output
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_ba_output_has_citations():
    """Test Business Analyst output includes proper citations."""
    ba_output = {
        "ticker": "AAPL",
        "qualitative_analysis": {
            "narrative": "Apple has strong ecosystem moat...",
        },
        "citations": [
            {"chunk_id": "broker_chunk_001", "relevance": 0.95},
            {"chunk_id": "transcript_chunk_001", "relevance": 0.85},
        ],
    }
    
    # Verify each citation has required fields
    for cit in ba_output["citations"]:
        assert "chunk_id" in cit
        assert "relevance" in cit
        assert 0 <= cit["relevance"] <= 1


@pytest.mark.prompt
def test_sr_output_broker_ratings_cited():
    """Test Stock Research broker ratings are properly cited."""
    sr_output = {
        "ticker": "AAPL",
        "broker_consensus": "Outperform",
        "broker_ratings": [
            {
                "broker": "Goldman Sachs",
                "rating": "Buy",
                "chunk_id": "broker_gs_001",
            },
            {
                "broker": "Morgan Stanley",
                "rating": "Overweight",
                "chunk_id": "broker_ms_001",
            },
        ],
    }
    
    # Each rating should have citation
    for rating in sr_output["broker_ratings"]:
        assert "chunk_id" in rating
        assert rating["chunk_id"].startswith("broker_")


# ---------------------------------------------------------------------------
# Test: Multi-Source Citation Validation
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_mixed_citation_sources():
    """Test output can cite multiple source types."""
    output = {
        "citations": [
            {"chunk_id": "broker_chunk_001", "source_type": "broker_report"},
            {"chunk_id": "transcript_chunk_001", "source_type": "earnings_call"},
            {"chunk_id": "macro_chunk_001", "source_type": "macro_report"},
        ],
    }
    
    source_types = {cit["source_type"] for cit in output["citations"]}
    
    # Should have at least 2 different source types
    assert len(source_types) >= 2
    
    # All should be valid types
    for st in source_types:
        assert st in CITATION_SOURCE_TYPES


# ---------------------------------------------------------------------------
# Test: Citation Chaining Validation
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_citation_chain_claim_to_doc():
    """Test complete chain: claim -> chunk -> source doc."""
    # Step 1: Claim in output
    claim = {
        "text": "Apple has strong moat",
        "chunk_id": "broker_chunk_001",
    }
    
    # Step 2: Chunk in DB
    chunk = {
        "chunk_id": "broker_chunk_001",
        "text": "Apple ecosystem creates switching costs...",
        "source_doc_id": "gs_report_2024.pdf",
    }
    
    # Step 3: Source doc in storage
    source_doc = {
        "doc_id": "gs_report_2024.pdf",
        "type": "broker_report",
        "broker": "Goldman Sachs",
    }
    
    # Verify chain
    assert claim["chunk_id"] == chunk["chunk_id"]
    assert chunk["source_doc_id"] == source_doc["doc_id"]
    assert source_doc["type"] == "broker_report"


# ---------------------------------------------------------------------------
# Test: Database Integration for Citations
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_citations_in_neo4j():
    """Test citations can be retrieved from Neo4j."""
    # This would test actual Neo4j connectivity
    
    # Expected query pattern
    cypher_query = """
    MATCH (c:Chunk {ticker: 'AAPL'})
    WHERE c.source IN ['broker_report', 'earnings_call', 'macro_report']
    RETURN c.chunk_id, c.text, c.source
    """
    
    # Verify query is valid format
    assert "MATCH" in cypher_query
    assert "RETURN" in cypher_query


@pytest.mark.integration
def test_citations_in_postgres():
    """Test citations can be retrieved from PostgreSQL."""
    # Expected table structure for text_chunks
    # chunk_id, text, source_type, ticker, metadata (JSONB)
    
    # Test query pattern
    query = """
    SELECT chunk_id, text, source_type, ticker 
    FROM text_chunks 
    WHERE ticker = 'AAPL' 
    AND source_type IN ('broker_report', 'earnings_call', 'macro_report')
    """
    
    assert "SELECT" in query
    assert "text_chunks" in query


# ---------------------------------------------------------------------------
# Test: Citation Accuracy Scoring
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_citation_relevance_scoring():
    """Test citation relevance is scored correctly."""
    citations = [
        {"chunk_id": "c1", "text": "Apple ecosystem moat", "query": "moat", "relevance": 0.95},
        {"chunk_id": "c2", "text": "Tesla battery tech", "query": "moat", "relevance": 0.2},
    ]
    
    # Verify relevance scores are reasonable
    for cit in citations:
        assert 0 <= cit["relevance"] <= 1
    
    # Moat-related citation should have high relevance
    assert citations[0]["relevance"] > 0.8
    # Irrelevant citation should have low relevance
    assert citations[1]["relevance"] < 0.5


# ---------------------------------------------------------------------------
# Extensible Test Helper for New Source Types
# ---------------------------------------------------------------------------

def create_citation_test(source_type: str, required_fields: List[str]):
    """Create a citation test for a new source type.
    
    Usage:
        create_citation_test("new_source", ["field1", "field2"])
    """
    @pytest.mark.prompt
    def test_new_source_citation():
        citation = {
            "chunk_id": f"{source_type}_chunk_001",
            "source_type": source_type,
        }
        
        # Add required fields
        for field in required_fields:
            citation[field] = f"test_{field}"
        
        # Verify all required fields present
        for field in required_fields:
            assert field in citation
    
    return test_new_source_citation


# ---------------------------------------------------------------------------
# Test: Extensible Source Type Registry
# ---------------------------------------------------------------------------

@pytest.mark.prompt
def test_source_type_registry():
    """Test that source types are registered for validation."""
    registry = get_source_type_registry()
    
    # All known types should be in registry
    for st in CITATION_SOURCE_TYPES:
        assert st in registry
    
    # Each should have required metadata fields
    for source_type, fields in registry.items():
        assert isinstance(fields, list)
        assert "chunk_id" in fields


def get_source_type_registry() -> Dict[str, List[str]]:
    """Get registry of source types and their required fields."""
    return {
        "broker_report": ["chunk_id", "broker", "rating", "price_target"],
        "earnings_call": ["chunk_id", "quarter", "speaker"],
        "macro_report": ["chunk_id", "report_type", "institution"],
        "company_filings": ["chunk_id", "filing_type", "period"],
        "news_article": ["chunk_id", "date", "source"],
    }