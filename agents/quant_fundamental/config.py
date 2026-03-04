"""Centralised configuration for the Quantitative Fundamental agent.

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
class QuantFundamentalConfig:
    """Runtime configuration container for the Quantitative Fundamental agent."""

    # PostgreSQL — structured financial data
    postgres_host: str = field(default_factory=lambda: _env("POSTGRES_HOST", "localhost"))
    postgres_port: int = field(default_factory=lambda: int(_env("POSTGRES_PORT", "5432")))
    postgres_db: str = field(default_factory=lambda: _env("POSTGRES_DB", "airflow"))
    postgres_user: str = field(default_factory=lambda: _env("POSTGRES_USER", "airflow"))
    postgres_password: str = field(default_factory=lambda: _env("POSTGRES_PASSWORD", "airflow"))

    # LLM — used ONLY for quantitative_summary narrative.
    # Use deepseek-r1:8b (same as BA and Summarizer) so Ollama never has to swap
    # models mid-pipeline.  A single model stays resident in GPU/ANE memory across
    # BA → QF → FM → Summarizer, eliminating the ~30-60s reload penalty per swap.
    llm_model: str = field(
        default_factory=lambda: os.getenv(
            "LLM_MODEL_QUANTITATIVE",
            os.getenv("BUSINESS_ANALYST_MODEL", "deepseek-r1:8b"),
        )
    )
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("QUANT_LLM_TEMPERATURE", "0.1"))
    )
    llm_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("QUANT_LLM_MAX_TOKENS", "4096"))
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    # No hard cap by default — quality over speed.
    # Set QUANT_REQUEST_TIMEOUT=<seconds> env var to add a cap if needed.
    request_timeout: Optional[int] = field(
        default_factory=lambda: (
            int(os.getenv("QUANT_REQUEST_TIMEOUT"))  # type: ignore[arg-type]
            if os.getenv("QUANT_REQUEST_TIMEOUT", "").strip()
            else None
        )
    )

    # Analysis knobs
    analysis_time_range: str = field(
        default_factory=lambda: os.getenv("ANALYSIS_TIME_RANGE", "TTM")
    )
    beta_lookback_days: int = field(
        default_factory=lambda: int(os.getenv("BETA_LOOKBACK_DAYS", "60"))
    )
    sharpe_lookback_days: int = field(
        default_factory=lambda: int(os.getenv("SHARPE_LOOKBACK_DAYS", "365"))
    )
    anomaly_zscore_threshold: float = field(
        default_factory=lambda: float(os.getenv("ANOMALY_ZSCORE_THRESHOLD", "2.0"))
    )
    rolling_mean_years: int = field(
        default_factory=lambda: int(os.getenv("ROLLING_MEAN_YEARS", "3"))
    )

    # SQL query timeout in seconds (0 = no timeout)
    sql_timeout: int = field(
        default_factory=lambda: int(os.getenv("QUANT_AGENT_SQL_TIMEOUT", "0"))
    )

    # Beneish M-Score threshold for manipulation risk
    beneish_threshold: float = field(
        default_factory=lambda: float(os.getenv("BENEISH_THRESHOLD", "-2.22"))
    )

    # Piotroski F-Score thresholds
    piotroski_strong_threshold: int = field(
        default_factory=lambda: int(os.getenv("PIOTROSKI_STRONG_THRESHOLD", "7"))
    )
    piotroski_weak_threshold: int = field(
        default_factory=lambda: int(os.getenv("PIOTROSKI_WEAK_THRESHOLD", "2"))
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


def load_config() -> QuantFundamentalConfig:
    """Helper used by modules that want a one-liner config."""
    return QuantFundamentalConfig()


__all__ = ["QuantFundamentalConfig", "load_config"]
