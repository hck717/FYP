#!/usr/bin/env python3
"""Integration-style tests for planner worst-case context injection."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestration.state import OrchestrationState


def test_planner_injects_worst_cases(monkeypatch):
    """Planner passes worst-case context into plan_query when available."""
    from orchestration import nodes

    captured: dict[str, str] = {"ctx": ""}

    # Avoid DB access from feedback helper
    monkeypatch.setattr(
        nodes.feedback,
        "get_lowest_rlaif_cases",
        lambda limit=5: [
            {
                "run_id": "r1",
                "user_query": "Analyze TSLA earnings call sentiment and broker disagreement",
                "ticker": "TSLA",
                "overall_score": 4.2,
                "agent_blamed": "business_analyst",
                "weaknesses": ["missing citations", "shallow analysis"],
                "specific_feedback": "too shallow",
            }
        ],
    )
    monkeypatch.setattr(
        nodes.feedback,
        "get_latest_user_feedback",
        lambda limit=5: [
            {
                "run_id": "r1",
                "helpful": False,
                "issue_tags": ["Analysis too shallow"],
                "comment": "completely useless",
            }
        ],
    )

    # Avoid external checks/import side effects
    monkeypatch.setattr(
        "orchestration.data_availability.check_all",
        lambda tickers=None: {"summary": "ok", "degraded_tiers": []},
        raising=True,
    )
    monkeypatch.setattr(
        "orchestration.episodic_memory.lookup_similar_failures",
        lambda user_query, tickers=None: [],
        raising=True,
    )
    monkeypatch.setattr(
        "orchestration.episodic_memory.build_preemptive_plan_hints",
        lambda failures: {},
        raising=True,
    )

    dummy_qf_agent = types.ModuleType("agents.quant_fundamental.agent")
    setattr(dummy_qf_agent, "extract_tickers_from_prompt", lambda q: ["AAPL"])
    monkeypatch.setitem(sys.modules, "agents.quant_fundamental.agent", dummy_qf_agent)

    def _mock_plan_query(user_query, worst_case_context=""):
        captured["ctx"] = worst_case_context
        return {
            "tickers": ["AAPL"],
            "ticker": "AAPL",
            "run_business_analyst": True,
            "run_quant_fundamental": True,
            "run_web_search": False,
            "run_financial_modelling": False,
            "run_stock_research": False,
            "complexity": 2,
            "planner_trace": "",
        }

    monkeypatch.setattr(nodes, "plan_query", _mock_plan_query)

    state = cast(OrchestrationState, {"user_query": "What is AAPL's competitive moat?"})
    out = nodes.node_planner(state)

    assert captured["ctx"] != ""
    assert "PAST LOW-RLAIF CASES" in captured["ctx"]
    assert "Blamed:" in captured["ctx"]
    assert "LATEST HUMAN FEEDBACK SIGNALS" in captured["ctx"]
    assert out.get("ticker") == "AAPL"


def test_planner_comprehensive_chinese_enables_all_agents(monkeypatch):
    """Chinese comprehensive report queries should fan out to all major agents."""
    from orchestration import nodes

    monkeypatch.setattr(nodes.feedback, "get_lowest_rlaif_cases", lambda limit=5: [])
    monkeypatch.setattr(nodes.feedback, "get_latest_user_feedback", lambda limit=5: [])
    monkeypatch.setattr(
        "orchestration.data_availability.check_all",
        lambda tickers=None: {"summary": "ok", "degraded_tiers": []},
        raising=True,
    )
    monkeypatch.setattr(
        "orchestration.episodic_memory.lookup_similar_failures",
        lambda user_query, tickers=None: [],
        raising=True,
    )
    monkeypatch.setattr(
        "orchestration.episodic_memory.build_preemptive_plan_hints",
        lambda failures: {},
        raising=True,
    )

    dummy_qf_agent = types.ModuleType("agents.quant_fundamental.agent")
    setattr(dummy_qf_agent, "extract_tickers_from_prompt", lambda q: ["AAPL"])
    monkeypatch.setitem(sys.modules, "agents.quant_fundamental.agent", dummy_qf_agent)

    def _mock_plan_query(user_query, worst_case_context=""):
        return {
            "tickers": ["AAPL"],
            "ticker": "AAPL",
            "run_business_analyst": True,
            "run_quant_fundamental": True,
            "run_web_search": False,
            "run_financial_modelling": False,
            "run_stock_research": False,
            "run_macro": False,
            "run_insider_news": False,
            "complexity": 1,
            "planner_trace": "",
        }

    monkeypatch.setattr(nodes, "plan_query", _mock_plan_query)

    state = cast(OrchestrationState, {"user_query": "请给我AAPL股票研究报告和执行摘要"})
    out = nodes.node_planner(state)

    assert out.get("run_business_analyst") is True
    assert out.get("run_quant_fundamental") is True
    assert out.get("run_financial_modelling") is True
    assert out.get("run_stock_research") is True
    assert out.get("run_web_search") is True
    assert out.get("run_macro") is True
    assert out.get("run_insider_news") is True
    assert int(out.get("react_max_iterations") or 0) >= 2


def test_planner_normalizes_ui_output_language(monkeypatch):
    """UI-provided title-case language should be normalized for summarizer."""
    from orchestration import nodes

    monkeypatch.setattr(nodes.feedback, "get_lowest_rlaif_cases", lambda limit=5: [])
    monkeypatch.setattr(nodes.feedback, "get_latest_user_feedback", lambda limit=5: [])
    monkeypatch.setattr(
        "orchestration.data_availability.check_all",
        lambda tickers=None: {"summary": "ok", "degraded_tiers": []},
        raising=True,
    )
    monkeypatch.setattr(
        "orchestration.episodic_memory.lookup_similar_failures",
        lambda user_query, tickers=None: [],
        raising=True,
    )
    monkeypatch.setattr(
        "orchestration.episodic_memory.build_preemptive_plan_hints",
        lambda failures: {},
        raising=True,
    )

    dummy_qf_agent = types.ModuleType("agents.quant_fundamental.agent")
    setattr(dummy_qf_agent, "extract_tickers_from_prompt", lambda q: ["MSFT"])
    monkeypatch.setitem(sys.modules, "agents.quant_fundamental.agent", dummy_qf_agent)

    monkeypatch.setattr(
        nodes,
        "plan_query",
        lambda user_query, worst_case_context="": {
            "tickers": ["MSFT"],
            "ticker": "MSFT",
            "run_business_analyst": True,
            "run_quant_fundamental": True,
            "run_web_search": False,
            "run_financial_modelling": False,
            "run_stock_research": False,
            "run_macro": False,
            "run_insider_news": False,
            "complexity": 1,
            "planner_trace": "",
        },
    )

    state = cast(OrchestrationState, {
        "user_query": "give me msft report",
        "output_language": "Japanese",
    })
    out = nodes.node_planner(state)

    assert out.get("output_language") == "japanese"


def test_planner_detects_language_typos_and_aliases(monkeypatch):
    """Planner should detect common misspellings/aliases for language intents."""
    from orchestration import nodes

    monkeypatch.setattr(nodes.feedback, "get_lowest_rlaif_cases", lambda limit=5: [])
    monkeypatch.setattr(nodes.feedback, "get_latest_user_feedback", lambda limit=5: [])
    monkeypatch.setattr(
        "orchestration.data_availability.check_all",
        lambda tickers=None: {"summary": "ok", "degraded_tiers": []},
        raising=True,
    )
    monkeypatch.setattr(
        "orchestration.episodic_memory.lookup_similar_failures",
        lambda user_query, tickers=None: [],
        raising=True,
    )
    monkeypatch.setattr(
        "orchestration.episodic_memory.build_preemptive_plan_hints",
        lambda failures: {},
        raising=True,
    )

    dummy_qf_agent = types.ModuleType("agents.quant_fundamental.agent")
    setattr(dummy_qf_agent, "extract_tickers_from_prompt", lambda q: ["MSFT"])
    monkeypatch.setitem(sys.modules, "agents.quant_fundamental.agent", dummy_qf_agent)

    monkeypatch.setattr(
        nodes,
        "plan_query",
        lambda user_query, worst_case_context="": {
            "tickers": ["MSFT"],
            "ticker": "MSFT",
            "run_business_analyst": True,
            "run_quant_fundamental": True,
            "run_web_search": False,
            "run_financial_modelling": False,
            "run_stock_research": False,
            "run_macro": False,
            "run_insider_news": False,
            "complexity": 1,
            "planner_trace": "",
        },
    )

    state = cast(OrchestrationState, {"user_query": "please answer in mandrain"})
    out = nodes.node_planner(state)
    assert out.get("output_language") == "mandarin"


def test_get_worst_cases_cold_start(monkeypatch):
    """get_worst_cases returns [] when rl_feedback has fewer than min_runs."""
    from orchestration import feedback

    class _Cursor:
        def __init__(self):
            self._sql = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            self._sql = sql

        def fetchone(self):
            if "COUNT(*) AS cnt" in self._sql:
                return {"cnt": 1}
            return None

        def fetchall(self):
            return []

    class _Conn:
        def cursor(self, cursor_factory=None):
            return _Cursor()

        def close(self):
            return None

    monkeypatch.setattr(feedback, "ensure_feedback_tables_exist", lambda: None)
    monkeypatch.setattr(feedback, "_get_pg_conn", lambda: _Conn())

    result = feedback.get_worst_cases(limit=5, min_runs=3)
    assert result == []


def test_get_latest_user_feedback_returns_latest_rows(monkeypatch):
    """latest user feedback helper should return most recent rows safely."""
    from orchestration import feedback

    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            return None

        def fetchall(self):
            return [
                {
                    "run_id": "r2",
                    "session_id": "s1",
                    "helpful": False,
                    "comment": "missing macro context",
                    "issue_tags": ["Missing macro"],
                    "report_version": "v1",
                    "timestamp": None,
                }
            ]

    class _Conn:
        def cursor(self, cursor_factory=None):
            return _Cursor()

        def close(self):
            return None

    monkeypatch.setattr(feedback, "ensure_feedback_tables_exist", lambda: None)
    monkeypatch.setattr(feedback, "_get_pg_conn", lambda: _Conn())

    rows = feedback.get_latest_user_feedback(limit=5)
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "r2"


def test_get_lowest_rlaif_cases_orders_by_score(monkeypatch):
    """lowest rlaif helper should return rows without crashing on normalization."""
    from orchestration import feedback

    class _Cursor:
        def __init__(self):
            self._count_phase = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            if "COUNT(*) AS cnt" in sql:
                self._count_phase = True
            else:
                self._count_phase = False

        def fetchone(self):
            return {"cnt": 10} if self._count_phase else None

        def fetchall(self):
            return [
                {
                    "run_id": "r-low",
                    "user_query": "analyze aapl",
                    "ticker": "AAPL",
                    "overall_score": 3.2,
                    "agent_blamed": "summarizer",
                    "weaknesses": ["shallow analysis"],
                    "specific_feedback": "too generic",
                    "timestamp": None,
                }
            ]

    class _Conn:
        def cursor(self, cursor_factory=None):
            return _Cursor()

        def close(self):
            return None

    monkeypatch.setattr(feedback, "ensure_feedback_tables_exist", lambda: None)
    monkeypatch.setattr(feedback, "_get_pg_conn", lambda: _Conn())

    rows = feedback.get_lowest_rlaif_cases(limit=5, min_runs=3)
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "r-low"


def test_score_report_with_rlaif_uses_output_language(monkeypatch):
    """score_report_with_rlaif should pass detected language to judge."""
    from orchestration import feedback

    captured: Dict[str, Any] = {"lang": None}

    monkeypatch.setattr(feedback, "ensure_feedback_tables_exist", lambda: None)

    def _mock_judge(report, user_query, agent_outputs_summary, report_language="english"):
        captured["lang"] = report_language
        return {
            "citation_completeness": 8.0,
            "analysis_depth": 8.0,
            "structure_compliance": 8.0,
            "language_quality": 8.0,
            "overall_score": 8.0,
            "strengths": ["ok"],
            "weaknesses": ["ok"],
            "specific_feedback": "ok",
            "agent_blamed": "none",
        }

    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            return None

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self, cursor_factory=None):
            return _Cursor()

        def close(self):
            return None

    monkeypatch.setattr(feedback, "_call_deepseek_judge", _mock_judge)
    monkeypatch.setattr(feedback, "_get_pg_conn", lambda: _Conn())

    scores = feedback.score_report_with_rlaif(
        run_id="run_lang_es",
        user_query="Analiza AAPL en espanol",
        final_summary="Este es un informe de prueba.",
        agent_outputs={},
        ticker="AAPL",
        output_language="spanish",
    )

    assert isinstance(scores, dict)
    assert captured["lang"] == "spanish"


def test_infer_report_language_detects_non_english_text():
    """Language inference should identify non-Latin scripts when no hint is provided."""
    from orchestration import feedback

    lang = feedback._infer_report_language(
        output_language=None,
        user_query="",
        final_summary="這是一份測試報告。",
    )
    assert lang in ("chinese", "mandarin", "cantonese")


def test_translate_text_strict_non_english_retry(monkeypatch):
    """Translator should retry if first pass returns English for Japanese target."""
    from orchestration import llm

    calls = {"n": 0}

    def _mock_generate(model, prompt, max_tokens=4096, temperature=0.0, system_prompt=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "## Executive Summary\nThis is still English [1]."
        return "## Executive Summary\nこれは日本語の要約です [1]。"

    monkeypatch.setattr(llm, "_deepseek_generate", _mock_generate)

    src = "## Executive Summary\nThis is a source report [1]."
    out = llm.translate_text(src, "Japanese")

    assert calls["n"] == 2
    assert "これは日本語" in out


def test_translate_text_alias_misspelling_mandrain(monkeypatch):
    """Translator should canonicalize common misspelling 'mandrain'."""
    from orchestration import llm

    def _mock_generate(model, prompt, max_tokens=4096, temperature=0.0, system_prompt=None, **kwargs):
        return "這是一份中文摘要 [1]。"

    monkeypatch.setattr(llm, "_deepseek_generate", _mock_generate)

    src = "## Executive Summary\nThis is a source report [1]."
    out = llm.translate_text(src, "mandrain")
    assert "中文" in out or "摘要" in out


def test_structured_multi_ticker_falls_back_to_multi_ticker_summariser(monkeypatch):
    """Structured path should bypass single-ticker schema for comparisons."""
    from orchestration import llm

    def _mock_summary(**kwargs):
        return "## Executive Summary\nComparison summary for both tickers."

    monkeypatch.setattr(llm, "summarise_results", _mock_summary)

    out = llm.summarise_results_structured(
        user_query="Compare AAPL vs NVDA",
        tickers=["AAPL", "NVDA"],
        ba_outputs=[{"ticker": "AAPL"}, {"ticker": "NVDA"}],
        quant_outputs=[{"ticker": "AAPL"}, {"ticker": "NVDA"}],
        web_outputs=[],
        fm_outputs=[],
        sr_outputs=[],
        macro_outputs=[],
        insider_news_outputs=[],
    )

    assert "Comparison summary" in out


def test_translate_text_english_passthrough(monkeypatch):
    """English target should bypass translation call entirely."""
    from orchestration import llm

    def _boom(*args, **kwargs):
        raise AssertionError("_deepseek_generate should not be called for English")

    monkeypatch.setattr(llm, "_deepseek_generate", _boom)

    src = "## Executive Summary\nKeep English." 
    out = llm.translate_text(src, "English")
    assert out == src


def test_deepseek_generate_handles_empty_choices(monkeypatch):
    """DeepSeek client should raise RuntimeError when API returns empty choices."""
    from orchestration import llm

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": []}

    monkeypatch.setattr(llm.requests, "post", lambda *args, **kwargs: _Resp())

    try:
        llm._deepseek_generate("deepseek-chat", "hi")
        assert False, "Expected RuntimeError for empty choices"
    except RuntimeError as exc:
        assert "no choices" in str(exc).lower()


def test_deepseek_generate_reads_api_key_at_call_time(monkeypatch):
    """DeepSeek client should use latest env var value, not stale import-time key."""
    from orchestration import llm

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-dynamic-test-key")

    captured = {"auth": ""}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def _mock_post(url, json=None, headers=None, timeout=None):
        captured["auth"] = (headers or {}).get("Authorization", "")
        return _Resp()

    monkeypatch.setattr(llm.requests, "post", _mock_post)

    out = llm._deepseek_generate("deepseek-chat", "hello")
    assert out == "ok"
    assert captured["auth"] == "Bearer sk-dynamic-test-key"


def test_citation_dedup_with_index_offset_no_index_error():
    """Citation builder should dedup safely when index_offset > 0."""
    from orchestration.citations import build_citation_block

    ba_output = {
        "qualitative_summary": (
            "Evidence from [neo4j::AAPL::risk_factors::0] is reiterated "
            "again [neo4j::AAPL::risk_factors::0]."
        ),
        "sentiment": {"bullish_pct": 55},
    }

    ref_block, chunk_map = build_citation_block(
        ba_output=ba_output,
        quant_output=None,
        web_output=None,
        fm_output=None,
        sr_output=None,
        macro_output=None,
        insider_news_output=None,
        ticker="AAPL",
        index_offset=10,
    )

    assert isinstance(ref_block, str)
    assert "[11]" in ref_block or "[12]" in ref_block
    assert isinstance(chunk_map, dict)
    assert "neo4j::AAPL::risk_factors::0" in chunk_map


def test_local_fallback_next_step_not_key_for_non_auth_error():
    """Fallback guidance should not always point to API key for non-auth errors."""
    from orchestration.nodes import _build_local_fallback_summary

    msg = _build_local_fallback_summary(
        user_query="compare a vs b",
        tickers=["A", "B"],
        ba_outputs=[{}, {}],
        quant_outputs=[{}, {}],
        web_outputs=[],
        fm_outputs=[{}, {}],
        sr_outputs=[{}, {}],
        macro_outputs=[{}, {}],
        insider_news_outputs=[{}, {}],
        errors={"summarizer": "IndexError: list index out of range"},
    )

    assert "IndexError" in msg
    assert "Set a valid `DEEPSEEK_API_KEY`" not in msg


def test_local_fallback_next_step_key_for_auth_error():
    """Fallback guidance should mention API key when auth failure is present."""
    from orchestration.nodes import _build_local_fallback_summary

    msg = _build_local_fallback_summary(
        user_query="compare a vs b",
        tickers=["A", "B"],
        ba_outputs=[{}, {}],
        quant_outputs=[{}, {}],
        web_outputs=[],
        fm_outputs=[{}, {}],
        sr_outputs=[{}, {}],
        macro_outputs=[{}, {}],
        insider_news_outputs=[{}, {}],
        errors={"summarizer": "HTTPError: 401 Unauthorized"},
    )

    assert "Set a valid `DEEPSEEK_API_KEY`" in msg


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
