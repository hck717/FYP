"""Shared dataclasses / enums for the Business Analyst agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CRAGStatus(str, Enum):
    CORRECT = "CORRECT"
    AMBIGUOUS = "AMBIGUOUS"
    INCORRECT = "INCORRECT"


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    text: str
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    chunks: List[Chunk]
    graph_facts: List[Dict[str, Any]]
    bm25_debug: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SentimentSnapshot:
    bullish_pct: float
    bearish_pct: float
    neutral_pct: float
    trend: str
    source: str = "postgresql:sentiment_trends"


def serialise_chunk(chunk: Chunk) -> Dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "score": chunk.score,
        "source": chunk.source,
        "metadata": chunk.metadata,
    }


__all__ = [
    "CRAGStatus",
    "Chunk",
    "RetrievalResult",
    "SentimentSnapshot",
    "serialise_chunk",
]
