"""Insider News agent prompt tests with metrics logging."""

from __future__ import annotations

import time
from typing import Any, Dict

import pytest


def _mock_insider_news_output(ticker: str) -> Dict[str, Any]:
    return {
        "agent": "insider_news",
        "ticker": ticker,
        "insider_analysis": {"net_signal": "bullish"},
        "news_analysis": {"headline_sentiment": "neutral"},
        "data_coverage": {"insider_transactions_count": 2, "news_items_count": 4},
    }


@pytest.mark.prompt
def test_insider_news_output_schema_and_metrics(record_observation):
    start = time.perf_counter()
    output = _mock_insider_news_output("AAPL")
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert output["agent"] == "insider_news"
    assert output["ticker"] == "AAPL"
    assert "data_coverage" in output

    record_observation(
        category="prompt_insider_news",
        name="insider_news_schema",
        outputs={"insider_news_outputs": [output]},
        metrics={
            "duration_ms": round(elapsed_ms, 3),
            "data_amount": int(output["data_coverage"]["news_items_count"]),
            "data_type": "dict",
            "latency_ms": round(elapsed_ms, 3),
        },
    )
