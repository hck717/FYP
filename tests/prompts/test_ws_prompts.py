"""Web Search prompt tests with metrics logging."""

from __future__ import annotations

import time
from typing import Any, Dict

import pytest


def _mock_ws_prompt_output(ticker: str) -> Dict[str, Any]:
    return {
        "agent": "web_search",
        "ticker": ticker,
        "sentiment_signal": "NEUTRAL",
        "breaking_news": [{"title": "Test", "url": "https://example.com"}],
        "raw_citations": ["https://example.com"],
    }


@pytest.mark.prompt
def test_ws_output_schema_and_metrics(record_observation):
    start = time.perf_counter()
    output = _mock_ws_prompt_output("AAPL")
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert output["agent"] == "web_search"
    assert output["ticker"] == "AAPL"
    assert isinstance(output["breaking_news"], list)

    record_observation(
        category="prompt_ws",
        name="ws_schema",
        outputs={"web_search_outputs": [output]},
        metrics={
            "duration_ms": round(elapsed_ms, 3),
            "data_amount": len(output["breaking_news"]),
            "data_type": "list[dict]",
            "latency_ms": round(elapsed_ms, 3),
        },
    )
