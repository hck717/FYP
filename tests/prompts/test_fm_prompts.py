"""Financial Modelling prompt tests with metrics logging."""

from __future__ import annotations

import time
from typing import Any, Dict

import pytest


def _mock_fm_prompt_output(ticker: str) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "dcf_value": 195.5,
        "wacc": 0.095,
        "terminal_growth": 0.025,
        "valuation": {"dcf": {"intrinsic_value_base": 195.5}},
    }


@pytest.mark.prompt
def test_fm_output_schema_and_metrics(record_observation):
    start = time.perf_counter()
    output = _mock_fm_prompt_output("AAPL")
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert output["ticker"] == "AAPL"
    assert "dcf_value" in output
    assert "valuation" in output

    record_observation(
        category="prompt_fm",
        name="fm_schema",
        outputs={"financial_modelling_outputs": [output]},
        metrics={
            "duration_ms": round(elapsed_ms, 3),
            "data_amount": 1,
            "data_type": "dict",
            "latency_ms": round(elapsed_ms, 3),
        },
    )
