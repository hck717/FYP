"""Centralised configuration for the Business Analyst agent.

This module normalises environment variables so that every other component
can rely on a single source of truth. The defaults match the values already
documented inside docker-compose.yml, .env, and agents/business_analyst/README.md.

Changes vs original:
  - LLM provider switched from Ollama to DeepSeek API (deepseek-reasoner model).
  - ollama_base_url retained for embeddings only (nomic-embed-text:v1.5 via Ollama).
  - deepseek_api_key field added (reads DEEPSEEK_API_KEY env var).
  - llm_num_ctx removed (Ollama-only; DeepSeek API uses max_tokens instead).
  - CRAG thresholds (crag_correct_threshold, crag_ambiguous_threshold) are
    now configurable via env vars CRAG_CORRECT_THRESHOLD / CRAG_AMBIGUOUS_THRESHOLD.
  - max_rewrite_loops default raised to 3 (env: BA_MAX_REWRITE_LOOPS).
  - embedding_model default updated to nomic-embed-text:v1.5 (locked version).
  - embedding_model_version field added to track version in Neo4j Chunk nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import os
from typing import Dict, Any, Optional


def _env(key: str, default: str | None = None) -> str:
    value = os.getenv(key)
    if value is None or value == "":
        if default is None:
            raise EnvironmentError(f"Missing required env var: {key}")
        return default
    return value


# Determine if running inside Docker
_IN_DOCKER = Path("/.dockerenv").exists()
# Ollama is kept for embeddings only (nomic-embed-text:v1.5, 768-dim).
# Inside Docker, Ollama is a named service reachable via http://ollama:11434.
# Outside Docker (local dev), use localhost.
_DEFAULT_OLLAMA = "http://ollama:11434" if _IN_DOCKER else "http://localhost:11434"


@dataclass(slots=True)
class BusinessAnalystConfig:
    """Runtime configuration container."""

    # Datastores
    neo4j_uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: _env("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "SecureNeo4jPass2025!"))
    neo4j_chunk_index: str = field(default_factory=lambda: os.getenv("NEO4J_CHUNK_INDEX", "chunk_embedding"))

    postgres_host: str = field(default_factory=lambda: _env("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(_env("POSTGRES_PORT", "5432")))
    postgres_db: str = field(default_factory=lambda: _env("POSTGRES_DB", "airflow"))
    postgres_user: str = field(default_factory=lambda: _env("POSTGRES_USER", "airflow"))
    postgres_password: str = field(default_factory=lambda: _env("POSTGRES_PASSWORD", "airflow"))

    # LLM — DeepSeek API (deepseek-reasoner) for all generation.
    # Ollama is NOT used for LLM generation; see ollama_base_url below for embeddings.
    llm_provider: str = field(default_factory=lambda: os.getenv("BUSINESS_ANALYST_LLM_PROVIDER", "deepseek"))
    llm_model: str = field(default_factory=lambda: os.getenv("BUSINESS_ANALYST_MODEL", os.getenv("LLM_MODEL_BUSINESS_ANALYST", "deepseek-reasoner")))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("BUSINESS_ANALYST_TEMPERATURE", "0.2")))
    # max_tokens controls DeepSeek API output budget (replaces Ollama's num_predict/num_ctx).
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("BUSINESS_ANALYST_MAX_TOKENS", "8000")))

    # DeepSeek API key — required for LLM generation.
    deepseek_api_key: str = field(default_factory=lambda: _env("DEEPSEEK_API_KEY", ""))

    # Ollama base URL — kept for embeddings only (nomic-embed-text:v1.5).
    # For local development outside Docker, set OLLAMA_BASE_URL=http://localhost:11434
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", _DEFAULT_OLLAMA))

    # Embedding model — locked to tagged version to prevent silent model drift.
    # Use EMBEDDING_MODEL env var to override (e.g. nomic-embed-text:latest for dev).
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "nomic-embed-text:v1.5"))
    embedding_dimension: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "768")))
    # Track which embedding model version produced the vectors stored in Neo4j.
    # Bump this when you pull a new model to invalidate old vectors.
    embedding_model_version: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL_VERSION", "1.0"))

    reranker_model: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    # Retrieval knobs
    top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "15")))
    rag_score_threshold: float = field(default_factory=lambda: float(os.getenv("RAG_SCORE_THRESHOLD", "0.6")))
    business_analyst_max_chunks: int = field(default_factory=lambda: int(os.getenv("BUSINESS_ANALYST_MAX_CHUNKS", "500")))
    business_analyst_chunk_size: int = field(default_factory=lambda: int(os.getenv("BUSINESS_ANALYST_CHUNK_SIZE", "512")))
    business_analyst_overlap: int = field(default_factory=lambda: int(os.getenv("BUSINESS_ANALYST_OVERLAP", "50")))

    # CRAG thresholds — now fully configurable via environment variables.
    # Tune these without code changes: set CRAG_CORRECT_THRESHOLD / CRAG_AMBIGUOUS_THRESHOLD in .env
    # Recommended starting point: CORRECT > 0.6, AMBIGUOUS 0.4–0.6, INCORRECT < 0.4
    crag_correct_threshold: float = field(default_factory=lambda: float(os.getenv("CRAG_CORRECT_THRESHOLD", "0.6")))
    crag_ambiguous_threshold: float = field(default_factory=lambda: float(os.getenv("CRAG_AMBIGUOUS_THRESHOLD", "0.4")))

    # Paths
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    agent_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "ingestion" / "etl" / "agent_data" / "business_analyst")

    # Networking / timeouts — 120s default suits DeepSeek API round-trips.
    request_timeout: Optional[int] = field(
        default_factory=lambda: (
            int(os.getenv("BUSINESS_ANALYST_REQUEST_TIMEOUT"))  # type: ignore[arg-type]
            if os.getenv("BUSINESS_ANALYST_REQUEST_TIMEOUT", "").strip()
            else 120
        )
    )
    neo4j_verify: bool = field(default_factory=lambda: os.getenv("NEO4J_ENCRYPT", "false").lower() == "true")

    # Feature flags
    enable_web_fallback: bool = field(default_factory=lambda: os.getenv("BUSINESS_ANALYST_ENABLE_WEB_FALLBACK", "true").lower() == "true")
    metadata_cache_ttl: float = field(default_factory=lambda: float(os.getenv("BUSINESS_ANALYST_METADATA_CACHE_TTL", "60")))
    semantic_cache_ttl: float = field(default_factory=lambda: float(os.getenv("BUSINESS_ANALYST_SEMANTIC_CACHE_TTL", "300")))
    semantic_cache_max_entries: int = field(default_factory=lambda: int(os.getenv("BUSINESS_ANALYST_SEMANTIC_CACHE_MAX_ENTRIES", "128")))

    # Adaptive routing knobs
    fast_path_top_k: int = field(default_factory=lambda: int(os.getenv("BA_FAST_PATH_TOP_K", "15")))
    multi_stage_recall_k: int = field(default_factory=lambda: int(os.getenv("BA_MULTI_STAGE_RECALL_K", "100")))
    # Minimum chunks per section in the final fused result for section-diversity guarantee.
    # Ensures earnings_call and broker_report sections are always represented even when
    # their RRF scores are lower than other sections for a given query.
    # Set to 0 to disable. Override via BA_MIN_CHUNKS_PER_SECTION env var.
    min_chunks_per_section: int = field(default_factory=lambda: int(os.getenv("BA_MIN_CHUNKS_PER_SECTION", "3")))
    # max_rewrite_loops: hard cap on CRAG rewrite cycles to prevent infinite loops.
    # Configurable via BA_MAX_REWRITE_LOOPS env var. Default: 3.
    max_rewrite_loops: int = field(default_factory=lambda: int(os.getenv("BA_MAX_REWRITE_LOOPS", "3")))
    # query_classifier_model: retained for reference but the classifier is rule-based only.
    # No LLM call is made for classification. This field is a no-op.
    query_classifier_model: str = field(default_factory=lambda: os.getenv("BA_QUERY_CLASSIFIER_MODEL", "rule-based"))

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["agent_data_dir"] = str(self.agent_data_dir)
        data["repo_root"] = str(self.repo_root)
        return data


def load_config() -> BusinessAnalystConfig:
    """Helper used by modules that want a one-liner config."""
    return BusinessAnalystConfig()


__all__ = ["BusinessAnalystConfig", "load_config"]
