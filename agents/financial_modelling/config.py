"""Centralised configuration for the Financial Modelling agent.

All env-var defaults match docker-compose.yml and .env documentation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def _env(key: str, default: Optional[str] = None) -> str:
    value = os.getenv(key)
    if value is None or value == "":
        if default is None:
            raise EnvironmentError(f"Missing required env var: {key}")
        return default
    return value


@dataclass(slots=True)
class FinancialModellingConfig:
    """Runtime configuration container for the Financial Modelling agent."""

    # PostgreSQL — structured financial data
    postgres_host: str = field(default_factory=lambda: _env("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(_env("POSTGRES_PORT", "5432")))
    postgres_db: str = field(default_factory=lambda: _env("POSTGRES_DB", "airflow"))
    postgres_user: str = field(default_factory=lambda: _env("POSTGRES_USER", "airflow"))
    postgres_password: str = field(default_factory=lambda: _env("POSTGRES_PASSWORD", "airflow"))

    # Neo4j — peer group selection
    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687")
    )
    neo4j_user: str = field(
        default_factory=lambda: os.getenv("NEO4J_USER", "neo4j")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "changeme_neo4j_password")
    )

    # LLM — used ONLY for the quantitative_summary narrative.
    # Use deepseek-r1:8b (same as BA and Summarizer) so Ollama never has to swap
    # models mid-pipeline.  A single model stays resident in GPU/ANE memory across
    # BA → QF → FM → Summarizer, eliminating the ~30-60s reload penalty per swap.
    llm_model: str = field(
        default_factory=lambda: os.getenv(
            "LLM_MODEL_FINANCIAL_MODELING",
            os.getenv("LLM_MODEL_FINANCIAL_MODELLING", "deepseek-r1:8b"),
        )
    )
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("FM_LLM_TEMPERATURE", "0.1"))
    )
    llm_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("FM_LLM_MAX_TOKENS", "1500"))
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    # No hard cap by default — quality over speed.
    # Set FM_REQUEST_TIMEOUT=<seconds> env var to add a cap if needed.
    request_timeout: Optional[int] = field(
        default_factory=lambda: (
            int(os.getenv("FM_REQUEST_TIMEOUT"))  # type: ignore[arg-type]
            if os.getenv("FM_REQUEST_TIMEOUT", "").strip()
            else None
        )
    )

    # DCF configuration
    dcf_discount_rate: float = field(
        default_factory=lambda: float(
            os.getenv("FIN_MODEL_DCF_DISCOUNT_RATE", os.getenv("DCF_WACC", "0.09"))
        )
    )
    dcf_terminal_growth_rate: float = field(
        default_factory=lambda: float(
            os.getenv("FIN_MODEL_TERMINAL_GROWTH_RATE", os.getenv("DCF_TERMINAL_GROWTH_RATE", "0.025"))
        )
    )
    dcf_forecast_years: int = field(
        default_factory=lambda: int(os.getenv("DCF_FORECAST_YEARS", "5"))
    )

    # Scenario probability weights
    scenario_prob_bear: float = field(
        default_factory=lambda: float(os.getenv("DCF_PROB_BEAR", "0.25"))
    )
    scenario_prob_base: float = field(
        default_factory=lambda: float(os.getenv("DCF_PROB_BASE", "0.55"))
    )
    scenario_prob_bull: float = field(
        default_factory=lambda: float(os.getenv("DCF_PROB_BULL", "0.20"))
    )

    # Analysis knobs
    price_history_days: int = field(
        default_factory=lambda: int(os.getenv("PRICE_HISTORY_DAYS", "365"))
    )
    beta_lookback_days: int = field(
        default_factory=lambda: int(os.getenv("BETA_LOOKBACK_DAYS", "60"))
    )
    comps_sector_peers: int = field(
        default_factory=lambda: int(os.getenv("COMPS_SECTOR_PEERS", "5"))
    )

    # SQL query timeout in seconds
    sql_timeout: int = field(
        default_factory=lambda: int(os.getenv("FM_SQL_TIMEOUT", "30"))
    )

    # Paths
    repo_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    def as_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        data = asdict(self)
        data["repo_root"] = str(self.repo_root)
        return data


def load_config() -> FinancialModellingConfig:
    """Helper used by modules that want a one-liner config."""
    return FinancialModellingConfig()


__all__ = ["FinancialModellingConfig", "load_config"]
