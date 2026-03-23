"""Prompt-level orchestration telemetry test with quantified metrics logging."""

from __future__ import annotations

import os
import time

import pytest

from tests.metrics import sizeof


def _mock_orchestration_state() -> dict:
    return {
        "business_analyst_outputs": [{"ticker": "AAPL", "qualitative_summary": "Strong moat"}],
        "quant_fundamental_outputs": [{"ticker": "AAPL", "piotroski_score": 8}],
        "web_search_outputs": [{"ticker": "AAPL", "sentiment_signal": "NEUTRAL"}],
        "financial_modelling_outputs": [{"ticker": "AAPL", "dcf_value": 195.5}],
        "stock_research_outputs": [{"ticker": "AAPL", "broker_consensus": "Outperform"}],
        "macro_outputs": [{"ticker": "AAPL", "regime": "disinflation"}],
        "insider_news_outputs": [{"ticker": "AAPL", "insider_analysis": {"net_signal": "bullish"}}],
        "final_summary": "AAPL screens positively across qualitative, quantitative, valuation, macro, and insider inputs.",
    }


@pytest.mark.prompt
def test_orchestration_prompt_telemetry(record_observation):
    start = time.perf_counter()
    # Optional live E2E run is disabled by default to keep prompt tests deterministic
    # and independent from external LLM API credentials.
    if os.getenv("PROMPT_TELEMETRY_LIVE_RUN", "0") == "1":
        from orchestration.graph import run

        state = run(
            "Analyze AAPL across business, fundamentals, valuation, macro and insider flows"
        )
    else:
        state = _mock_orchestration_state()

    total_ms = (time.perf_counter() - start) * 1000.0

    outputs = {
        "business_analyst_outputs": state.get("business_analyst_outputs") or [],
        "quant_fundamental_outputs": state.get("quant_fundamental_outputs") or [],
        "web_search_outputs": state.get("web_search_outputs") or [],
        "financial_modelling_outputs": state.get("financial_modelling_outputs") or [],
        "stock_research_outputs": state.get("stock_research_outputs") or [],
        "macro_outputs": state.get("macro_outputs") or [],
        "insider_news_outputs": state.get("insider_news_outputs") or [],
        "final_summary": state.get("final_summary") or "",
    }

    output_bytes = sizeof(outputs)

    assert "final_summary" in state

    record_observation(
        category="prompt_orchestration",
        name="orchestration_full_prompt_flow",
        outputs=outputs,
        metrics={
            "duration_ms": round(total_ms, 3),
            "latency_ms": round(total_ms, 3),
            "data_type": "orchestration_state",
            "data_amount": sum(len(v) for k, v in outputs.items() if k.endswith("_outputs") and isinstance(v, list)),
            "output_size_bytes": output_bytes,
        },
    )
