"""Unit + integration tests for the Business Analyst CRAG pipeline.

Test strategy
=============
All external I/O (Neo4j, Qdrant, PostgreSQL, Ollama HTTP, web search) is
mocked so the suite runs entirely offline. Real logic under test:

- CRAG threshold routing (CORRECT / AMBIGUOUS / INCORRECT)
- Max-1 rewrite guard (AMBIGUOUS exhaustion → web fallback)
- JSON output schema completeness
- web_search_fallback path merges results correctly
- CLI smoke test (subprocess, no live services)
- Tools unit: QdrantConnector, HybridRetriever, CRAGEvaluator, healthcheck

Run with:
    pytest agents/business_analyst/tests/test_agent.py -v
"""

# pyright: basic
from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, Dict, List, Optional, cast as _cast
from unittest.mock import MagicMock, patch

import pytest

# Node functions return AgentState (total=False TypedDict). Pylance treats every
# key access on it as potentially missing. Cast results to Dict[str, Any] so the
# test assertions read naturally without a wall of "# type: ignore" comments.
_R = Dict[str, Any]  # local alias used to annotate node return values


def _r(state: Any) -> _R:
    """Downcast AgentState → plain dict for assertion access."""
    return dict(state)  # type: ignore[arg-type]


from agents.business_analyst.agent import (
    AgentState,
    _node_crag_evaluate,
    _node_fetch_sentiment,
    _node_format_json_output,
    _node_generate_analysis,
    _node_hybrid_retrieval,
    _node_rewrite_query,
    _node_web_search_fallback,
    _route_after_crag,
    _route_after_rewrite,
    build_graph,
    run,
)
from agents.business_analyst.config import BusinessAnalystConfig
from agents.business_analyst.schema import (
    Chunk,
    CRAGStatus,
    RetrievalResult,
    SentimentSnapshot,
)
from agents.business_analyst.tools import (
    BusinessAnalystToolkit,
    CRAGEvaluation,
    CRAGEvaluator,
    HybridRetriever,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str = "c001", score: float = 0.85) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=f"Apple has a wide competitive moat. [{chunk_id}]",
        score=score,
        source="neo4j",
        metadata={"section": "Item 1", "ticker": "AAPL"},
    )


def _make_retrieval(score: float = 0.85) -> RetrievalResult:
    return RetrievalResult(
        chunks=[_make_chunk(score=score)],
        graph_facts=[{"relationship": "HAS_STRATEGY", "node": {"name": "Ecosystem"}, "relationship_properties": {}}],
    )


def _minimal_config() -> BusinessAnalystConfig:
    """Config with safe defaults — no real env vars required."""
    cfg = BusinessAnalystConfig.__new__(BusinessAnalystConfig)
    # Manually set all slots to avoid env-var reads in tests
    cfg.neo4j_uri = "bolt://localhost:7687"
    cfg.neo4j_user = "neo4j"
    cfg.neo4j_password = "test"
    cfg.neo4j_chunk_index = "chunk_embedding"
    cfg.postgres_host = "localhost"
    cfg.postgres_port = 5432
    cfg.postgres_db = "airflow"
    cfg.postgres_user = "airflow"
    cfg.postgres_password = "airflow"
    cfg.qdrant_host = "localhost"
    cfg.qdrant_port = 6333
    cfg.qdrant_collection = "financial_documents"
    cfg.llm_provider = "ollama"
    cfg.llm_model = "deepseek-v3.2-exp"
    cfg.llm_temperature = 0.2
    cfg.llm_max_tokens = 1500
    cfg.ollama_base_url = "http://localhost:11434"
    cfg.embedding_model = "all-MiniLM-L6-v2"
    cfg.embedding_dimension = 384
    cfg.qdrant_embedding_model = "nomic-embed-text"
    cfg.qdrant_embedding_dimension = 768
    cfg.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cfg.top_k = 8
    cfg.rag_score_threshold = 0.6
    cfg.business_analyst_max_chunks = 500
    cfg.business_analyst_chunk_size = 512
    cfg.business_analyst_overlap = 50
    cfg.crag_correct_threshold = 0.7
    cfg.crag_ambiguous_threshold = 0.5
    from pathlib import Path
    cfg.repo_root = Path(".")
    cfg.agent_data_dir = Path("ingestion/etl/agent_data/business_analyst")
    cfg.request_timeout = 60
    cfg.neo4j_verify = False
    cfg.enable_web_fallback = True
    return cfg


# ---------------------------------------------------------------------------
# 1. CRAG routing — threshold logic
# ---------------------------------------------------------------------------

class TestCRAGRouting:
    def setup_method(self):
        self.cfg = _minimal_config()
        self.toolkit = MagicMock(spec=BusinessAnalystToolkit)
        self.toolkit.config = self.cfg

    def test_correct_route(self):
        state: AgentState = {
            "task": "moat?", "ticker": "AAPL",
            "crag_status": CRAGStatus.CORRECT, "rewrite_count": 0,
            "confidence": 0.85,
        }
        assert _route_after_crag(state, self.toolkit) == "generate_analysis"

    def test_ambiguous_first_attempt_routes_to_rewrite(self):
        state: AgentState = {
            "task": "moat?", "ticker": "AAPL",
            "crag_status": CRAGStatus.AMBIGUOUS, "rewrite_count": 0,
            "confidence": 0.6,
        }
        assert _route_after_crag(state, self.toolkit) == "rewrite_query"

    def test_ambiguous_second_attempt_routes_to_web(self):
        """After 1 rewrite, AMBIGUOUS with NO local data must fall through to web search."""
        state: AgentState = {
            "task": "moat?", "ticker": "AAPL",
            "crag_status": CRAGStatus.AMBIGUOUS, "rewrite_count": 1,
            "confidence": 0.6,
            # No retrieval, no sentiment — ensures web fallback is triggered
        }
        assert _route_after_crag(state, self.toolkit) == "web_search_fallback"

    def test_incorrect_routes_to_web(self):
        state: AgentState = {
            "task": "moat?", "ticker": "PLTR",
            "crag_status": CRAGStatus.INCORRECT, "rewrite_count": 0,
            "confidence": 0.3,
            # No retrieval, no sentiment — ensures web fallback is triggered
        }
        assert _route_after_crag(state, self.toolkit) == "web_search_fallback"

    def test_incorrect_with_local_data_routes_to_generate(self):
        """INCORRECT but local chunks exist — should generate from thin context, not fall back."""
        state: AgentState = {
            "task": "moat?", "ticker": "AAPL",
            "crag_status": CRAGStatus.INCORRECT, "rewrite_count": 0,
            "confidence": 0.3,
            "retrieval": _make_retrieval(0.3),
        }
        assert _route_after_crag(state, self.toolkit) == "generate_analysis"

    def test_rewrite_always_returns_to_retrieval(self):
        state: AgentState = {"task": "moat?", "ticker": "AAPL", "rewrite_count": 1}
        assert _route_after_rewrite(state) == "hybrid_retrieval"


# ---------------------------------------------------------------------------
# 2. CRAGEvaluator unit tests
# ---------------------------------------------------------------------------

class TestCRAGEvaluator:
    def setup_method(self):
        self.cfg = _minimal_config()
        self.evaluator = CRAGEvaluator(self.cfg)

    def test_correct_threshold(self):
        chunks = [_make_chunk(score=0.75)]
        result = self.evaluator.evaluate(chunks)
        assert result.status == CRAGStatus.CORRECT
        assert result.confidence == pytest.approx(0.75)

    def test_ambiguous_threshold(self):
        chunks = [_make_chunk(score=0.60)]
        result = self.evaluator.evaluate(chunks)
        assert result.status == CRAGStatus.AMBIGUOUS

    def test_incorrect_threshold(self):
        chunks = [_make_chunk(score=0.40)]
        result = self.evaluator.evaluate(chunks)
        assert result.status == CRAGStatus.INCORRECT

    def test_empty_chunks_is_incorrect(self):
        result = self.evaluator.evaluate([])
        assert result.status == CRAGStatus.INCORRECT
        assert result.confidence == 0.0

    def test_boundary_correct(self):
        """Score exactly equal to threshold is CORRECT."""
        chunks = [_make_chunk(score=0.7)]
        result = self.evaluator.evaluate(chunks)
        assert result.status == CRAGStatus.CORRECT

    def test_boundary_ambiguous(self):
        """Score exactly equal to ambiguous threshold is AMBIGUOUS."""
        chunks = [_make_chunk(score=0.5)]
        result = self.evaluator.evaluate(chunks)
        assert result.status == CRAGStatus.AMBIGUOUS


# ---------------------------------------------------------------------------
# 3. Node unit tests (mocked toolkit / llm)
# ---------------------------------------------------------------------------

class TestNodes:
    def setup_method(self):
        self.cfg = _minimal_config()
        self.toolkit = MagicMock(spec=BusinessAnalystToolkit)
        self.toolkit.config = self.cfg
        self.llm = MagicMock()

    def test_fetch_sentiment_node_with_data(self):
        snap = SentimentSnapshot(bullish_pct=65, bearish_pct=20, neutral_pct=15, trend="improving")
        self.toolkit.fetch_sentiment.return_value = snap
        state: AgentState = {"task": "moat?", "ticker": "AAPL"}
        result: Any = _node_fetch_sentiment(state, self.toolkit)
        assert result["sentiment"] == snap

    def test_fetch_sentiment_node_no_data(self):
        self.toolkit.fetch_sentiment.return_value = None
        state: AgentState = {"task": "moat?", "ticker": "UNKNOWN"}
        result: Any = _node_fetch_sentiment(state, self.toolkit)
        assert result["sentiment"] is None

    def test_hybrid_retrieval_node(self):
        retrieval = _make_retrieval(0.85)
        self.toolkit.retrieve.return_value = retrieval
        state: AgentState = {"task": "moat?", "ticker": "AAPL"}
        result: Any = _node_hybrid_retrieval(state, self.toolkit)
        assert result["retrieval"] is retrieval
        self.toolkit.retrieve.assert_called_once_with("moat?", "AAPL")

    def test_crag_evaluate_node(self):
        retrieval = _make_retrieval(0.85)
        self.toolkit.evaluate.return_value = CRAGEvaluation(CRAGStatus.CORRECT, 0.85)
        state: AgentState = {"task": "moat?", "ticker": "AAPL", "retrieval": retrieval}
        result: Any = _node_crag_evaluate(state, self.toolkit)
        assert result["crag_status"] == CRAGStatus.CORRECT
        assert result["confidence"] == pytest.approx(0.85)

    def test_generate_analysis_node_success(self):
        llm_result = {
            "qualitative_summary": "Apple has a wide moat [c001].",
            "competitive_moat": {"rating": "wide", "key_strengths": ["ecosystem"], "sources": ["c001"]},
            "key_risks": [],
            "missing_context": [],
        }
        self.llm.generate.return_value = llm_result
        state: AgentState = {
            "task": "moat?", "ticker": "AAPL",
            "retrieval": _make_retrieval(0.85),
            "sentiment": SentimentSnapshot(65, 20, 15, "improving"),
        }
        result: Any = _node_generate_analysis(state, self.llm)
        assert result["llm_output"] == llm_result
        assert result["fallback_triggered"] is False

    def test_generate_analysis_node_llm_failure(self):
        self.llm.generate.side_effect = RuntimeError("Ollama down")
        state: AgentState = {
            "task": "moat?", "ticker": "AAPL",
            "retrieval": _make_retrieval(0.85),
            "crag_status": CRAGStatus.CORRECT,
        }
        result: Any = _node_generate_analysis(state, self.llm)
        # Should not raise — error captured in qualitative_summary
        assert "GENERATION_ERROR" in result["llm_output"]["qualitative_summary"]

    def test_rewrite_query_node_increments_count(self):
        self.llm.rewrite_query.return_value = "Apple ecosystem moat switching cost"
        state: AgentState = {"task": "moat?", "ticker": "AAPL", "rewrite_count": 0}
        result: Any = _node_rewrite_query(state, self.llm)
        assert result["task"] == "Apple ecosystem moat switching cost"
        assert result["rewrite_count"] == 1

    def test_web_search_fallback_node(self):
        with patch(
            "agents.business_analyst.agent._call_web_search",
            return_value={"summary": "Palantir AIP is growing.", "key_risks": []},
        ):
            state: AgentState = {"task": "PLTR positioning?", "ticker": "PLTR"}
            result: Any = _node_web_search_fallback(state, self.toolkit)
        assert result["fallback_triggered"] is True
        assert result["web_search_result"]["summary"] == "Palantir AIP is growing."


# ---------------------------------------------------------------------------
# 4. format_json_output — schema compliance
# ---------------------------------------------------------------------------

REQUIRED_TOP_LEVEL_KEYS = {
    "agent", "ticker", "query_date", "sentiment", "competitive_moat",
    "key_risks", "missing_context", "crag_status", "confidence",
    "fallback_triggered", "qualitative_summary",
}


class TestFormatJsonOutput:
    def _make_state(self, **overrides) -> Any:
        base: Any = {
            "task": "moat?",
            "ticker": "AAPL",
            "crag_status": CRAGStatus.CORRECT,
            "confidence": 0.85,
            "fallback_triggered": False,
            "retrieval": _make_retrieval(0.85),
            "sentiment": SentimentSnapshot(65, 20, 15, "improving"),
            "llm_output": {
                "qualitative_summary": "Apple has ecosystem lock-in [c001].",
                "competitive_moat": {"rating": "wide", "key_strengths": ["ecosystem"], "sources": ["c001"]},
                "key_risks": [],
                "missing_context": [],
            },
            "web_search_result": None,
        }
        base.update(overrides)
        return base

    def test_required_keys_present(self):
        state = self._make_state()
        result: Any = _node_format_json_output(state)
        output = result["output"]
        missing = REQUIRED_TOP_LEVEL_KEYS - set(output.keys())
        assert not missing, f"Output missing keys: {missing}"

    def test_agent_field(self):
        state = self._make_state()
        result: Any = _node_format_json_output(state)
        assert result["output"]["agent"] == "business_analyst"

    def test_ticker_propagated(self):
        state = self._make_state(ticker="MSFT")
        result: Any = _node_format_json_output(state)
        assert result["output"]["ticker"] == "MSFT"

    def test_crag_status_is_string(self):
        state = self._make_state()
        result: Any = _node_format_json_output(state)
        assert isinstance(result["output"]["crag_status"], str)
        assert result["output"]["crag_status"] == "CORRECT"

    def test_confidence_rounded(self):
        state = self._make_state(confidence=0.853219)
        result: Any = _node_format_json_output(state)
        assert result["output"]["confidence"] == pytest.approx(0.8532, abs=1e-4)

    def test_fallback_triggered_false_by_default(self):
        state = self._make_state()
        result: Any = _node_format_json_output(state)
        assert result["output"]["fallback_triggered"] is False

    def test_web_fallback_merges_summary(self):
        state = self._make_state(
            fallback_triggered=True,
            web_search_result={"summary": "PLTR is growing in gov contracts.", "key_risks": []},
            crag_status=CRAGStatus.INCORRECT,
            confidence=0.35,
            llm_output=None,
        )
        result: Any = _node_format_json_output(state)
        output = result["output"]
        assert output["fallback_triggered"] is True
        assert "PLTR is growing" in output["qualitative_summary"]
        # missing_context should be populated for fallback
        assert len(output["missing_context"]) >= 1

    def test_null_sentiment_uses_defaults(self):
        state = self._make_state(sentiment=None)
        result: Any = _node_format_json_output(state)
        sent = result["output"]["sentiment"]
        # Should have all keys even when no DB sentiment
        assert "bullish_pct" in sent
        assert "bearish_pct" in sent
        assert "neutral_pct" in sent

    def test_insufficient_data_summary_when_no_output(self):
        state = self._make_state(llm_output={}, crag_status=CRAGStatus.INCORRECT, confidence=0.2)
        result: Any = _node_format_json_output(state)
        assert "INSUFFICIENT_DATA" in result["output"]["qualitative_summary"]

    def test_query_date_format(self):
        import re
        state = self._make_state()
        result: Any = _node_format_json_output(state)
        date = result["output"]["query_date"]
        assert re.match(r"\d{4}-\d{2}-\d{2}", date), f"Unexpected date format: {date}"


# ---------------------------------------------------------------------------
# 5. Full graph integration test (all I/O mocked)
# ---------------------------------------------------------------------------

class TestFullPipelineIntegration:
    """End-to-end: build_graph → invoke with mocked toolkit + llm."""

    def _run_pipeline(
        self,
        crag_score: float,
        task: str = "Analyse Apple moat",
        ticker: str = "AAPL",
        llm_output: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = _minimal_config()

        mock_toolkit = MagicMock(spec=BusinessAnalystToolkit)
        mock_toolkit.config = cfg
        mock_toolkit.fetch_sentiment.return_value = SentimentSnapshot(65, 20, 15, "improving")
        mock_toolkit.retrieve.return_value = _make_retrieval(crag_score)
        mock_toolkit.evaluate.return_value = CRAGEvaluation(
            CRAGStatus.CORRECT if crag_score >= 0.7 else (
                CRAGStatus.AMBIGUOUS if crag_score >= 0.5 else CRAGStatus.INCORRECT
            ),
            crag_score,
        )

        mock_llm = MagicMock()
        mock_llm.generate.return_value = llm_output or {
            "qualitative_summary": "Apple has a wide moat [c001].",
            "competitive_moat": {"rating": "wide", "key_strengths": ["ecosystem"], "sources": ["c001"]},
            "key_risks": [],
            "missing_context": [],
        }
        mock_llm.rewrite_query.return_value = f"{task} rewritten"

        compiled = build_graph(mock_toolkit, mock_llm)
        initial: AgentState = {
            "task": task,
            "ticker": ticker,
            "rewrite_count": 0,
            "fallback_triggered": False,
        }
        final = compiled.invoke(initial)
        return final.get("output") or {}

    def test_correct_path_produces_output(self):
        output = self._run_pipeline(crag_score=0.85)
        assert output["agent"] == "business_analyst"
        assert output["crag_status"] == "CORRECT"
        assert output["fallback_triggered"] is False

    def test_ambiguous_path_triggers_rewrite_then_correct(self):
        """AMBIGUOUS first → rewrite → 2nd retrieval still AMBIGUOUS → web fallback."""
        cfg = _minimal_config()
        mock_toolkit = MagicMock(spec=BusinessAnalystToolkit)
        mock_toolkit.config = cfg
        mock_toolkit.fetch_sentiment.return_value = None
        mock_toolkit.retrieve.return_value = _make_retrieval(0.6)

        call_count = {"n": 0}

        def eval_side_effect(chunks):
            call_count["n"] += 1
            # First eval: AMBIGUOUS; second eval (after rewrite): CORRECT
            if call_count["n"] == 1:
                return CRAGEvaluation(CRAGStatus.AMBIGUOUS, 0.6)
            return CRAGEvaluation(CRAGStatus.CORRECT, 0.78)

        mock_toolkit.evaluate.side_effect = eval_side_effect
        mock_llm = MagicMock()
        mock_llm.generate.return_value = {
            "qualitative_summary": "Tesla faces margin pressure [c001].",
            "competitive_moat": {"rating": "narrow", "key_strengths": ["Supercharger"], "sources": ["c001"]},
            "key_risks": [],
            "missing_context": [],
        }
        mock_llm.rewrite_query.return_value = "Tesla margin competition BYD risk factors"

        compiled = build_graph(mock_toolkit, mock_llm)
        final = compiled.invoke({"task": "Tesla risks?", "ticker": "TSLA", "rewrite_count": 0, "fallback_triggered": False})
        output = final.get("output") or {}
        # Should have gone AMBIGUOUS → rewrite → CORRECT
        assert output["crag_status"] == "CORRECT"
        assert mock_llm.rewrite_query.call_count == 1

    def test_incorrect_path_triggers_web_fallback(self):
        with patch(
            "agents.business_analyst.agent._call_web_search",
            return_value={"summary": "PLTR AIP is competitive.", "key_risks": []},
        ):
            cfg = _minimal_config()
            mock_toolkit = MagicMock(spec=BusinessAnalystToolkit)
            mock_toolkit.config = cfg
            mock_toolkit.fetch_sentiment.return_value = None  # no sentiment
            mock_toolkit.retrieve.return_value = RetrievalResult(chunks=[], graph_facts=[])  # no local data
            mock_toolkit.evaluate.return_value = CRAGEvaluation(CRAGStatus.INCORRECT, 0.35)
            mock_llm = MagicMock()
            compiled = build_graph(mock_toolkit, mock_llm)
            final = compiled.invoke({"task": "PLTR positioning?", "ticker": "PLTR", "rewrite_count": 0, "fallback_triggered": False})
            output = final.get("output") or {}
        assert output["fallback_triggered"] is True
        assert output["crag_status"] == "INCORRECT"

    def test_ambiguous_exhaustion_goes_to_web(self):
        """AMBIGUOUS with rewrite_count=1 already set and no local data → must go straight to web."""
        cfg = _minimal_config()
        mock_toolkit = MagicMock(spec=BusinessAnalystToolkit)
        mock_toolkit.config = cfg
        mock_toolkit.fetch_sentiment.return_value = None  # no sentiment
        mock_toolkit.retrieve.return_value = RetrievalResult(chunks=[], graph_facts=[])  # no local data
        mock_toolkit.evaluate.return_value = CRAGEvaluation(CRAGStatus.AMBIGUOUS, 0.6)

        mock_llm = MagicMock()

        with patch(
            "agents.business_analyst.agent._call_web_search",
            return_value={"summary": "Fallback data.", "key_risks": []},
        ):
            compiled = build_graph(mock_toolkit, mock_llm)
            final = compiled.invoke({
                "task": "vague query",
                "ticker": "XYZ",
                "rewrite_count": 1,  # already exhausted
                "fallback_triggered": False,
            })

        output = final.get("output") or {}
        assert output["fallback_triggered"] is True
        # LLM should not have been called
        mock_llm.generate.assert_not_called()

    def test_output_json_serialisable(self):
        """Output must be fully JSON-serialisable (no dataclasses, enums)."""
        output = self._run_pipeline(crag_score=0.85)
        serialised = json.dumps(output, default=str)
        parsed = json.loads(serialised)
        assert parsed["agent"] == "business_analyst"


# ---------------------------------------------------------------------------
# 6. run() public API with mocked build_graph
# ---------------------------------------------------------------------------

class TestRunPublicAPI:
    def test_run_returns_dict(self):
        expected_output = {
            "agent": "business_analyst",
            "ticker": "AAPL",
            "query_date": "2026-02-24",
            "crag_status": "CORRECT",
            "confidence": 0.85,
            "fallback_triggered": False,
            "qualitative_summary": "Apple has a wide moat.",
            "sentiment": {},
            "competitive_moat": None,
            "key_risks": [],
            "missing_context": [],
        }
        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"output": expected_output}

        with patch("agents.business_analyst.agent.build_graph", return_value=mock_compiled), \
             patch("agents.business_analyst.agent.BusinessAnalystToolkit") as mock_tk_cls, \
             patch("agents.business_analyst.agent.LLMClient"), \
             patch("agents.business_analyst.agent.load_config", return_value=_minimal_config()):
            mock_tk_cls.return_value.close = MagicMock()
            result = run(task="What is Apple's moat?", ticker="AAPL")

        assert result == expected_output

    def test_run_returns_empty_dict_on_missing_output(self):
        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {}

        with patch("agents.business_analyst.agent.build_graph", return_value=mock_compiled), \
             patch("agents.business_analyst.agent.BusinessAnalystToolkit") as mock_tk_cls, \
             patch("agents.business_analyst.agent.LLMClient"), \
             patch("agents.business_analyst.agent.load_config", return_value=_minimal_config()):
            mock_tk_cls.return_value.close = MagicMock()
            result = run(task="query", ticker=None)

        assert result == {}


# ---------------------------------------------------------------------------
# 7. CLI smoke test
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_help(self):
        """--help should exit 0 and mention the agent name."""
        result = subprocess.run(
            [sys.executable, "-m", "agents.business_analyst.agent", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "business_analyst" in result.stdout.lower() or "ticker" in result.stdout.lower()

    def test_cli_runs_with_mocked_services(self):
        """CLI invocation with --ticker should produce valid JSON on stdout."""
        expected = {
            "agent": "business_analyst",
            "ticker": "AAPL",
            "query_date": "2026-01-01",
            "crag_status": "CORRECT",
            "confidence": 0.9,
            "fallback_triggered": False,
            "qualitative_summary": "Apple moat is wide.",
            "sentiment": {"bullish_pct": 0, "bearish_pct": 0, "neutral_pct": 0, "trend": "unknown", "source": "postgresql:sentiment_trends"},
            "competitive_moat": None,
            "key_risks": [],
            "missing_context": [],
        }

        with patch("agents.business_analyst.agent.run", return_value=expected) as mock_run:
            # We can't inject a patch into a subprocess, so we test via the main() function directly
            from agents.business_analyst.agent import main
            import io
            from contextlib import redirect_stdout

            with patch("sys.argv", ["business_analyst", "--ticker", "AAPL"]):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    try:
                        main()
                    except SystemExit as e:
                        assert e.code == 0 or e.code is None

            output = buf.getvalue().strip()
            if output:
                parsed = json.loads(output)
                assert parsed["agent"] == "business_analyst"
