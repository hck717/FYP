# agents/business_analyst/tests/test_agent.py
"""
Unit + integration tests for the Business Analyst Agent.

Run:
    source .venv/bin/activate
    python -m pytest agents/business_analyst/tests/ -v

Integration tests (require live services) are marked with @pytest.mark.integration
and skipped by default. Run with:
    pytest -v -m integration
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from agents.business_analyst.agent import (
    run_business_analyst_agent,
    BusinessAnalystInput,
    _rewrite_query,
    _call_ollama,
)
from agents.business_analyst.tools import (
    crag_evaluate,
    extract_json_from_response,
    _safe_float,
)


# ── Unit Tests ─────────────────────────────────────────────────────────────────

class TestCragEvaluate:
    def test_correct_threshold(self):
        assert crag_evaluate(0.85) == "CORRECT"
        assert crag_evaluate(0.71) == "CORRECT"

    def test_ambiguous_threshold(self):
        assert crag_evaluate(0.65) == "AMBIGUOUS"
        assert crag_evaluate(0.50) == "AMBIGUOUS"

    def test_incorrect_threshold(self):
        assert crag_evaluate(0.49) == "INCORRECT"
        assert crag_evaluate(0.0)  == "INCORRECT"


class TestExtractJson:
    def test_raw_json(self):
        raw = '{"agent": "business_analyst", "ticker": "AAPL"}'
        result = extract_json_from_response(raw)
        assert result["agent"] == "business_analyst"

    def test_markdown_fenced_json(self):
        raw = '```json\n{"agent": "business_analyst"}\n```'
        result = extract_json_from_response(raw)
        assert result["agent"] == "business_analyst"

    def test_invalid_returns_none(self):
        result = extract_json_from_response("not json at all")
        assert result is None

    def test_json_with_preamble(self):
        raw = 'Here is the analysis: {"agent": "business_analyst", "ticker": "NVDA"}'
        result = extract_json_from_response(raw)
        assert result is not None
        assert result["ticker"] == "NVDA"


class TestSafeFloat:
    def test_converts_string(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_invalid_returns_none(self):
        assert _safe_float("N/A") is None


# ── Agent Tests (mocked) ───────────────────────────────────────────────────────

MOCK_JSON_OUTPUT = {
    "agent": "business_analyst",
    "ticker": "AAPL",
    "query_date": "2026-02-24",
    "company_overview": {
        "name": "Apple Inc",
        "sector": "Technology",
        "market_cap": 3250000000000,
        "pe_ratio": 31.2,
        "profit_margin": 0.263,
    },
    "sentiment": {
        "bullish_pct": 65.0,
        "bearish_pct": 20.0,
        "neutral_pct": 15.0,
        "trend": "improving",
        "source": "postgresql:sentiment_trends",
    },
    "competitive_moat": {
        "rating": "wide",
        "key_strengths": ["Ecosystem lock-in", "Services recurring revenue"],
        "sources": ["chunk_id_001", "chunk_id_042"],
    },
    "key_risks": [
        {"risk": "China revenue dependency", "severity": "HIGH", "source": "chunk_id_018"}
    ],
    "missing_context": [],
    "crag_status": "CORRECT",
    "confidence": 0.85,
    "fallback_triggered": False,
    "qualitative_summary": "Apple maintains a wide moat through ecosystem integration and growing services.",
    "error": None,
}


class TestAgentWithMocks:
    @patch("agents.business_analyst.agent.fetch_sentiment")
    @patch("agents.business_analyst.agent.fetch_company_profile")
    @patch("agents.business_analyst.agent.hybrid_retrieve")
    @patch("agents.business_analyst.agent.qdrant_news_search")
    @patch("agents.business_analyst.agent._call_ollama")
    def test_correct_path(
        self, mock_ollama, mock_news, mock_hybrid,
        mock_profile, mock_sentiment
    ):
        """CORRECT path: score > 0.7, clean JSON output."""
        mock_sentiment.return_value = {
            "bullish_pct": 65, "bearish_pct": 20, "neutral_pct": 15,
            "trend": "improving", "source": "postgresql:sentiment_trends"
        }
        mock_profile.return_value = {
            "name": "Apple Inc", "sector": "Technology",
            "market_cap": 3.25e12, "pe_ratio": 31.2, "profit_margin": 0.263
        }
        mock_hybrid.return_value = (["[chunk_id_001] Apple has strong ecosystem lock-in"], 0.85)
        mock_news.return_value   = ["[news|2026-02-20] Apple services revenue grows 17% YoY"]
        mock_ollama.return_value = json.dumps(MOCK_JSON_OUTPUT)

        result = run_business_analyst_agent(
            BusinessAnalystInput(task="What is Apple's competitive moat?", ticker="AAPL")
        )

        assert result["agent"] == "business_analyst"
        assert result["crag_status"] == "CORRECT"
        assert result["confidence"] == pytest.approx(0.85)
        assert result["fallback_triggered"] is False
        assert result["error"] is None
        assert result["competitive_moat"]["rating"] == "wide"

    @patch("agents.business_analyst.agent.fetch_sentiment", return_value={})
    @patch("agents.business_analyst.agent.fetch_company_profile", return_value={})
    @patch("agents.business_analyst.agent.hybrid_retrieve")
    def test_incorrect_path(self, mock_hybrid, *_):
        """INCORRECT path: score < 0.5 → fallback_triggered=True, no Ollama call."""
        mock_hybrid.return_value = ([], 0.3)

        result = run_business_analyst_agent(
            BusinessAnalystInput(task="What is Palantir's AI strategy?", ticker="PLTR")
        )

        assert result["crag_status"] == "INCORRECT"
        assert result["fallback_triggered"] is True
        assert result["confidence"] == pytest.approx(0.3)
        assert len(result["missing_context"]) > 0

    @patch("agents.business_analyst.agent.fetch_sentiment", return_value={})
    @patch("agents.business_analyst.agent.fetch_company_profile", return_value={})
    @patch("agents.business_analyst.agent.hybrid_retrieve")
    @patch("agents.business_analyst.agent._rewrite_query")
    @patch("agents.business_analyst.agent.qdrant_news_search", return_value=[])
    @patch("agents.business_analyst.agent._call_ollama")
    def test_ambiguous_recovers(
        self, mock_ollama, mock_news, mock_rewrite, mock_hybrid, *_
    ):
        """AMBIGUOUS path: rewrite → retry → CORRECT."""
        # First call AMBIGUOUS, retry CORRECT
        mock_hybrid.side_effect = [
            (["[chunk_id_050] Tesla faces margin pressure from BYD"], 0.62),
            (["[chunk_id_050] Tesla faces gross margin decline Q4 2025"], 0.81),
        ]
        mock_rewrite.return_value = "Tesla gross margin decline BYD competition 2025"
        mock_ollama.return_value  = json.dumps({
            **MOCK_JSON_OUTPUT,
            "ticker": "TSLA",
            "crag_status": "CORRECT",
            "confidence": 0.81,
        })

        result = run_business_analyst_agent(
            BusinessAnalystInput(task="Tesla risks", ticker="TSLA")
        )

        assert result["crag_status"] == "CORRECT"
        assert result["fallback_triggered"] is False
        mock_rewrite.assert_called_once()

    @patch("agents.business_analyst.agent.fetch_sentiment", return_value={})
    @patch("agents.business_analyst.agent.fetch_company_profile", return_value={})
    @patch("agents.business_analyst.agent.hybrid_retrieve")
    @patch("agents.business_analyst.agent.qdrant_news_search", return_value=[])
    @patch("agents.business_analyst.agent._call_ollama")
    def test_json_parse_failure_returns_error(
        self, mock_ollama, mock_news, mock_hybrid, *_
    ):
        """If Ollama returns garbage, agent returns graceful error output."""
        mock_hybrid.return_value = (["some context"], 0.8)
        mock_ollama.return_value = "This is not JSON at all."

        result = run_business_analyst_agent(
            BusinessAnalystInput(task="Analyse MSFT", ticker="MSFT")
        )

        assert result["fallback_triggered"] is True
        assert result["error"] is not None
        assert "JSON" in result["error"]


# ── Integration Tests (require live services) ──────────────────────────────────

@pytest.mark.integration
class TestIntegration:
    """These tests require Docker services + Ollama to be running."""

    def test_live_aapl(self):
        """Full pipeline test against live services with AAPL."""
        result = run_business_analyst_agent(
            BusinessAnalystInput(
                task="What is Apple's competitive moat and key business risks?",
                ticker="AAPL"
            )
        )
        print(json.dumps(result, indent=2))
        assert result["agent"] == "business_analyst"
        assert result["ticker"] == "AAPL"
        assert result["crag_status"] in ("CORRECT", "AMBIGUOUS", "INCORRECT")
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["error"] is None or result["fallback_triggered"] is True

    def test_live_unknown_ticker(self):
        """Un-ingested ticker should return fallback_triggered=True."""
        result = run_business_analyst_agent(
            BusinessAnalystInput(
                task="Analyse this company",
                ticker="ZZZZ"
            )
        )
        assert result["fallback_triggered"] is True
        assert result["crag_status"] == "INCORRECT"
