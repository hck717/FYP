"""Stock Research prompt tests with metrics logging."""

from __future__ import annotations

import time
from typing import Any, Dict

import pytest


def _mock_sr_prompt_output(ticker: str) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "broker_consensus": "Outperform",
        "broker_ratings": [
            {"broker": "Goldman Sachs", "rating": "Buy"},
            {"broker": "JP Morgan", "rating": "Overweight"},
        ],
        "transcript_comparison": "Management tone stable vs prior quarter.",
    }


@pytest.mark.prompt
def test_sr_output_schema_and_metrics(record_observation):
    start = time.perf_counter()
    output = _mock_sr_prompt_output("AAPL")
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert output["ticker"] == "AAPL"
    assert "broker_consensus" in output
    assert isinstance(output["broker_ratings"], list)

    record_observation(
        category="prompt_sr",
        name="sr_schema",
        outputs={"stock_research_outputs": [output]},
        metrics={
            "duration_ms": round(elapsed_ms, 3),
            "data_amount": len(output["broker_ratings"]),
            "data_type": "list[dict]",
            "latency_ms": round(elapsed_ms, 3),
        },
    )
