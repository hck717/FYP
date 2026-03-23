"""Macro agent prompt tests with metrics logging."""

from __future__ import annotations

import time
from typing import Any, Dict

import pytest


def _mock_macro_prompt_output(ticker: str) -> Dict[str, Any]:
    return {
        "agent": "macro",
        "ticker": ticker,
        "regime": "disinflation",
        "top_macro_drivers": ["rates", "usd", "oil"],
        "per_report_summaries": [{"report": "fed_notes", "signal": "dovish"}],
    }


@pytest.mark.prompt
def test_macro_output_schema_and_metrics(record_observation):
    start = time.perf_counter()
    output = _mock_macro_prompt_output("AAPL")
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert output["agent"] == "macro"
    assert output["ticker"] == "AAPL"
    assert isinstance(output["top_macro_drivers"], list)

    record_observation(
        category="prompt_macro",
        name="macro_schema",
        outputs={"macro_outputs": [output]},
        metrics={
            "duration_ms": round(elapsed_ms, 3),
            "data_amount": len(output["top_macro_drivers"]),
            "data_type": "list[str]",
            "latency_ms": round(elapsed_ms, 3),
        },
    )
