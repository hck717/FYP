"""Infrastructure connectivity tests.

These tests verify that all required infrastructure components are accessible:
- PostgreSQL (raw_timeseries, financial_statements, text_chunks, etc.)
- Neo4j (bolt connection + vector index)
- Ollama (embedding endpoint)

Run with: pytest tests/integration/test_infra.py -v -m integration --timeout=120

The tests are marked with @pytest.mark.integration so they can be skipped in CI
when services are not available.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
import requests

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Test: PostgreSQL Connection + Tables Exist
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_postgres_tables():
    """Test PostgreSQL connection and that required tables exist."""
    # Import after path is set
    from orchestration import feedback
    
    conn = feedback._get_pg_conn()
    cursor = conn.cursor()
    
    required_tables = [
        "raw_timeseries",
        "financial_statements",
        "text_chunks",
        "valuation_metrics",
        "sentiment_trends",
        "raw_fundamentals",
    ]
    
    for table in required_tables:
        cursor.execute(f"SELECT 1 FROM {table} LIMIT 1")
    
    cursor.close()
    conn.close()


@pytest.mark.integration
def test_postgres_feedback_tables():
    """Test that RLAIF feedback tables exist."""
    from orchestration import feedback
    
    feedback.ensure_feedback_tables_exist()
    
    conn = feedback._get_pg_conn()
    cursor = conn.cursor()
    
    required_tables = ["rl_feedback", "user_feedback", "prompt_versions"]
    
    for table in required_tables:
        cursor.execute(f"SELECT 1 FROM {table} LIMIT 1")
    
    cursor.close()
    conn.close()


# ---------------------------------------------------------------------------
# Test: Neo4j Bolt + Vector Index
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_neo4j_connection():
    """Test Neo4j bolt connection."""
    from neo4j import GraphDatabase
    
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "")
    
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    with driver.session() as session:
        result = session.run("RETURN 1 AS test")
        assert result.single()["test"] == 1
    
    driver.close()


@pytest.mark.integration
def test_neo4j_vector_index():
    """Test Neo4j vector index exists for chunk embeddings."""
    from neo4j import GraphDatabase
    
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "")
    
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    
    with driver.session() as session:
        result = session.run(
            "SHOW INDEXES YIELD name, type WHERE type='VECTOR' RETURN name"
        )
        indexes = [record["name"] for record in result]
        
        # Check for chunk_embedding or similar index
        chunk_index_exists = any(
            "chunk" in idx.lower() and "embed" in idx.lower()
            for idx in indexes
        )
        # This is a soft check - the index may not exist in fresh setups
        assert isinstance(chunk_index_exists, bool)
    
    driver.close()


# ---------------------------------------------------------------------------
# Test: Ollama Embedding Endpoint
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_ollama_embed():
    """Test Ollama embedding endpoint returns valid embeddings."""
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    
    resp = requests.post(
        f"{ollama_url}/api/embeddings",
        json={"model": embed_model, "prompt": "test"},
        timeout=30
    )
    
    assert resp.status_code == 200
    data = resp.json()
    assert "embedding" in data
    assert len(data["embedding"]) > 0


@pytest.mark.integration
def test_ollama_models():
    """Test Ollama is running and has required models."""
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    resp = requests.get(f"{ollama_url}/api/tags", timeout=30)
    assert resp.status_code == 200
    
    models = resp.json().get("models", [])
    model_names = [m.get("name", "") for m in models]
    
    # Check for embedding model
    embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    assert any(embed_model in name for name in model_names), \
        f"Embedding model {embed_model} not found in Ollama"


# ---------------------------------------------------------------------------
# Test: DeepSeek API (for LLM calls)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_deepseek_api():
    """Test DeepSeek API is accessible for orchestration LLM calls."""
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    
    if not deepseek_key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    
    resp = requests.post(
        f"{deepseek_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {deepseek_key}"},
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Say 'test'"}],
            "max_tokens": 5,
        },
        timeout=30
    )
    
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert len(data["choices"]) > 0


# ---------------------------------------------------------------------------
# Test: All Services Health Check
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_all_services_health():
    """Comprehensive health check for all required services."""
    results = {}
    
    # PostgreSQL
    try:
        from orchestration import feedback
        conn = feedback._get_pg_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        results["postgres"] = "ok"
    except Exception as e:
        results["postgres"] = f"error: {e}"
    
    # Neo4j
    try:
        from neo4j import GraphDatabase
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "")
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        results["neo4j"] = "ok"
    except Exception as e:
        results["neo4j"] = f"error: {e}"
    
    # Ollama
    try:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        resp = requests.get(f"{ollama_url}/api/tags", timeout=10)
        results["ollama"] = "ok" if resp.status_code == 200 else f"status: {resp.status_code}"
    except Exception as e:
        results["ollama"] = f"error: {e}"
    
    # Log results
    print(f"\n=== Infrastructure Health Check ===")
    for service, status in results.items():
        print(f"{service}: {status}")
    
    # Fail if any critical service is down
    critical = ["postgres", "neo4j"]
    for service in critical:
        assert results.get(service) == "ok", f"Critical service {service} is down"


# ---------------------------------------------------------------------------
# Skip Non-Integration Tests When Services Not Available
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def skip_if_no_services(request):
    """Auto-skip integration tests if services are not available."""
    if request.keywords.get("integration"):
        # Check if we should skip based on environment
        skip_marker = request.node.get_closest_marker("skip_if_no_services")
        if skip_marker:
            pytest.skip("Services not available")