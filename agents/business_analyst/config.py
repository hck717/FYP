"""Centralised configuration for the Business Analyst agent.

This module normalises environment variables so that every other component
can rely on a single source of truth. The defaults match the values already
documented inside docker-compose.yml, .env, and agents/business_analyst/README.md.
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


# Determine if running inside Docker for Ollama URL
_IN_DOCKER = Path("/.dockerenv").exists()
_DEFAULT_OLLAMA = "http://host.docker.internal:11434" if _IN_DOCKER else "http://localhost:11434"


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

    # Models
    llm_provider: str = field(default_factory=lambda: os.getenv("BUSINESS_ANALYST_LLM_PROVIDER", "ollama"))
    llm_model: str = field(default_factory=lambda: os.getenv("BUSINESS_ANALYST_MODEL", os.getenv("LLM_MODEL_BUSINESS_ANALYST", "deepseek-r1:8b")))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("BUSINESS_ANALYST_TEMPERATURE", "0.2")))
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("BUSINESS_ANALYST_MAX_TOKENS", "16000")))
    # num_ctx: Ollama context window size.  The default Ollama value is 4096 which is
    # smaller than our prompt alone (~4500–8000 tokens).  We must set this explicitly
    # or the prompt is silently truncated and the model generates garbage.
    llm_num_ctx: int = field(default_factory=lambda: int(os.getenv("BUSINESS_ANALYST_NUM_CTX", "32768")))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", _DEFAULT_OLLAMA))

    # Embedding model — all vectors use Ollama nomic-embed-text (768-dim).
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "nomic-embed-text"))
    embedding_dimension: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "768")))

    reranker_model: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    # Retrieval knobs
    top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "15")))
    rag_score_threshold: float = field(default_factory=lambda: float(os.getenv("RAG_SCORE_THRESHOLD", "0.6")))
    business_analyst_max_chunks: int = field(default_factory=lambda: int(os.getenv("BUSINESS_ANALYST_MAX_CHUNKS", "500")))
    business_analyst_chunk_size: int = field(default_factory=lambda: int(os.getenv("BUSINESS_ANALYST_CHUNK_SIZE", "512")))
    business_analyst_overlap: int = field(default_factory=lambda: int(os.getenv("BUSINESS_ANALYST_OVERLAP", "50")))

    # CRAG thresholds — updated for Optimized Adaptive pipeline
    # CORRECT > 0.6, AMBIGUOUS 0.4-0.6, INCORRECT < 0.4
    crag_correct_threshold: float = field(default_factory=lambda: float(os.getenv("CRAG_CORRECT_THRESHOLD", "0.6")))
    crag_ambiguous_threshold: float = field(default_factory=lambda: float(os.getenv("CRAG_AMBIGUOUS_THRESHOLD", "0.4")))

    # Paths
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    agent_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "ingestion" / "etl" / "agent_data" / "business_analyst")

    # Networking / timeouts — no hard cap by default; quality over speed.
    # deepseek-r1:8b generates ~10-15 tok/s; at 5000 max tokens that is ~5-8 min.
    # Set BUSINESS_ANALYST_REQUEST_TIMEOUT=<seconds> env var to add a cap if needed.
    # Default: None (no timeout) - important for deep analysis
    request_timeout: Optional[int] = field(
        default_factory=lambda: (
            int(os.getenv("BUSINESS_ANALYST_REQUEST_TIMEOUT"))  # type: ignore[arg-type]
            if os.getenv("BUSINESS_ANALYST_REQUEST_TIMEOUT", "").strip()
            else None  # No timeout by default - removed previous default
        )
    )
    neo4j_verify: bool = field(default_factory=lambda: os.getenv("NEO4J_ENCRYPT", "false").lower() == "true")

    # Feature flags
    enable_web_fallback: bool = field(default_factory=lambda: os.getenv("BUSINESS_ANALYST_ENABLE_WEB_FALLBACK", "true").lower() == "true")
    metadata_cache_ttl: float = field(default_factory=lambda: float(os.getenv("BUSINESS_ANALYST_METADATA_CACHE_TTL", "60")))
    semantic_cache_ttl: float = field(default_factory=lambda: float(os.getenv("BUSINESS_ANALYST_SEMANTIC_CACHE_TTL", "300")))
    semantic_cache_max_entries: int = field(default_factory=lambda: int(os.getenv("BUSINESS_ANALYST_SEMANTIC_CACHE_MAX_ENTRIES", "128")))

    # Adaptive routing knobs
    # fast_path_top_k: smaller recall budget for simple/fast queries (<= 15 is safe)
    fast_path_top_k: int = field(default_factory=lambda: int(os.getenv("BA_FAST_PATH_TOP_K", "15")))
    # multi_stage_recall_k: Stage-1 bi-encoder recall count before cross-encoder rerank
    multi_stage_recall_k: int = field(default_factory=lambda: int(os.getenv("BA_MULTI_STAGE_RECALL_K", "100")))
    # max_rewrite_loops: up to 2 query rewrites on AMBIGUOUS path
    max_rewrite_loops: int = field(default_factory=lambda: int(os.getenv("BA_MAX_REWRITE_LOOPS", "2")))
    # query_classifier_model: lightweight model used for fast query classification
    query_classifier_model: str = field(default_factory=lambda: os.getenv("BA_QUERY_CLASSIFIER_MODEL", os.getenv("ORCHESTRATION_PLANNER_MODEL", "llama3.2:latest")))

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["agent_data_dir"] = str(self.agent_data_dir)
        data["repo_root"] = str(self.repo_root)
        return data


def load_config() -> BusinessAnalystConfig:
    """Helper used by modules that want a one-liner config."""

    return BusinessAnalystConfig()


__all__ = ["BusinessAnalystConfig", "load_config"]
