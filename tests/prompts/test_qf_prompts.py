"""Quant Fundamental prompt tests with metrics logging."""

from __future__ import annotations

import time
from typing import Any, Dict

import pytest


def _mock_qf_prompt_output(ticker: str) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "piotroski_score": 8,
        "beneish_m_score": -2.1,
        "altman_z": 4.2,
        "factor_scores": {"quality": 0.8, "value": 0.6, "momentum": 0.7},
    }


@pytest.mark.prompt
def test_qf_output_schema_and_metrics(record_observation):
    start = time.perf_counter()
    output = _mock_qf_prompt_output("AAPL")
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert output["ticker"] == "AAPL"
    assert "piotroski_score" in output
    assert "factor_scores" in output

    record_observation(
        category="prompt_qf",
        name="qf_schema",
        outputs={"quant_fundamental_outputs": [output]},
        metrics={
            "duration_ms": round(elapsed_ms, 3),
            "data_amount": 1,
            "data_type": "dict",
            "latency_ms": round(elapsed_ms, 3),
        },
    )
