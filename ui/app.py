"""Streamlit UI — Agentic Investment Analyst
A1: User feedback (thumb-up/down + text) logged to PostgreSQL citation_tracking / query_logs.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psycopg2
import streamlit as st
from psycopg2.extras import RealDictCursor

# ── Import orchestration pipeline ────────────────────────────────────────────
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orchestration.graph import stream as orchestration_stream

logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Investment Analyst",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# ── PostgreSQL feedback helpers ───────────────────────────────────────────────

def _pg_connect():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "financial_data"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )


def _log_query_feedback(
    run_id: str,
    ticker: Optional[str],
    query_text: str,
    overall_rating: int,          # -1, 0, 1
    feedback_text: str,
    complexity: Optional[int],
    agents_used: Optional[List[str]],
    final_note: Optional[str],
) -> None:
    """Upsert a query-level feedback record into query_logs."""
    try:
        with _pg_connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_logs
                    (run_id, ticker, query_text, overall_rating, feedback_text,
                     complexity, agents_used, final_note, recorded_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (run_id) DO UPDATE SET
                    overall_rating = EXCLUDED.overall_rating,
                    feedback_text  = EXCLUDED.feedback_text
                """,
                (
                    run_id,
                    ticker,
                    query_text,
                    overall_rating,
                    feedback_text,
                    complexity,
                    json.dumps(agents_used or []),
                    (final_note or "")[:4096],
                ),
            )
    except Exception as exc:
        logger.warning("Failed to log query feedback: %s", exc)


def _log_citation_feedback(
    run_id: str,
    ticker: Optional[str],
    query_text: str,
    agent_name: str,
    feedback_score: int,   # -1, 0, 1
    feedback_text: str,
) -> None:
    """Insert an agent-card level feedback record into citation_tracking."""
    try:
        with _pg_connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO citation_tracking
                    (run_id, ticker, query_text, agent_name, was_cited,
                     feedback_score, feedback_text, recorded_at)
                VALUES (%s, %s, %s, %s, TRUE, %s, %s, NOW())
                """,
                (run_id, ticker, query_text, agent_name, feedback_score, feedback_text),
            )
    except Exception as exc:
        logger.warning("Failed to log citation feedback: %s", exc)


# ── UI helpers ────────────────────────────────────────────────────────────────

def _agent_display_name(key: str) -> str:
    return {
        "business_analyst_output":    "Business Analyst",
        "quant_fundamental_output":   "Quant Fundamental",
        "web_search_output":          "Web Search",
        "financial_modelling_output": "Financial Modelling",
    }.get(key, key.replace("_", " ").title())


def _render_agent_card(
    agent_key: str,
    agent_output: Dict[str, Any],
    run_id: str,
    ticker: Optional[str],
    query_text: str,
) -> None:
    """Render a collapsible agent output card with per-agent thumbs feedback."""
    label = _agent_display_name(agent_key)
    with st.expander(f"**{label}**", expanded=False):
        st.json(agent_output, expanded=2)

        st.markdown("---")
        st.markdown(f"*Was this {label} analysis useful?*")

        col1, col2, col3 = st.columns([1, 1, 4])
        fb_key = f"fb_{agent_key}_{run_id}"

        with col1:
            if st.button("👍", key=f"up_{fb_key}", help="Helpful"):
                _log_citation_feedback(run_id, ticker, query_text, label, 1, "")
                st.success("Thanks for the feedback!")
        with col2:
            if st.button("👎", key=f"dn_{fb_key}", help="Not helpful"):
                _log_citation_feedback(run_id, ticker, query_text, label, -1, "")
                st.warning("Feedback recorded.")
        with col3:
            fb_text = st.text_input(
                "Optional comment",
                key=f"txt_{fb_key}",
                placeholder="What was wrong or missing?",
                label_visibility="collapsed",
            )
            if fb_text and st.button("Submit comment", key=f"sub_{fb_key}"):
                _log_citation_feedback(run_id, ticker, query_text, label, 0, fb_text)
                st.info("Comment logged.")


def _render_feedback_panel(
    run_id: str,
    ticker: Optional[str],
    query_text: str,
    complexity: Optional[int],
    agents_used: List[str],
    final_note: Optional[str],
) -> None:
    """Overall note feedback panel (thumbs + text)."""
    st.markdown("---")
    st.subheader("Rate this research note")

    col1, col2 = st.columns([3, 7])
    with col1:
        rating = st.radio(
            "Overall rating",
            options=["👍 Helpful", "😐 Neutral", "👎 Not helpful"],
            horizontal=True,
            key=f"overall_rating_{run_id}",
        )
    with col2:
        feedback_text = st.text_area(
            "Additional feedback (optional)",
            placeholder="What was missing, incorrect, or unclear?",
            key=f"overall_text_{run_id}",
            height=80,
        )

    if st.button("Submit feedback", key=f"submit_overall_{run_id}"):
        score_map = {"👍 Helpful": 1, "😐 Neutral": 0, "👎 Not helpful": -1}
        score = score_map.get(rating, 0)
        _log_query_feedback(
            run_id=run_id,
            ticker=ticker,
            query_text=query_text,
            overall_rating=score,
            feedback_text=feedback_text,
            complexity=complexity,
            agents_used=agents_used,
            final_note=final_note,
        )
        st.success("Feedback saved — thank you!")


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("Agentic Investment Analyst")
    st.caption(
        "Powered by LangGraph · DeepSeek-R1 · Llama 3.2 · Perplexity Sonar "
        "· PostgreSQL · Neo4j · Qdrant"
    )

    # ── Session state initialisation ─────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history: List[Dict[str, Any]] = []
    if "current_result" not in st.session_state:
        st.session_state.current_result: Optional[Dict[str, Any]] = None
    if "run_id" not in st.session_state:
        st.session_state.run_id: str = ""

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")
        recency = st.selectbox(
            "News recency", ["week", "day", "month"], index=0
        )
        st.markdown("---")
        st.markdown("**Recent queries**")
        for h in st.session_state.history[-5:][::-1]:
            st.markdown(f"- {h['query'][:60]}")

    # ── Query input ───────────────────────────────────────────────────────────
    with st.form("query_form", clear_on_submit=False):
        query = st.text_input(
            "Enter your investment research question",
            placeholder="e.g. Analyze Apple's valuation and competitive position",
        )
        submitted = st.form_submit_button("Analyse", use_container_width=True)

    if not submitted or not query.strip():
        if st.session_state.current_result:
            _render_results(st.session_state.current_result, st.session_state.run_id)
        return

    # ── Run pipeline ──────────────────────────────────────────────────────────
    run_id = str(uuid.uuid4())
    st.session_state.run_id = run_id
    st.session_state.history.append({"query": query, "run_id": run_id})

    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    final_state: Dict[str, Any] = {}
    node_events: List[str] = []

    with st.spinner("Running multi-agent analysis..."):
        try:
            for node_name, node_output in orchestration_stream(
                user_query=query,
                session_id=run_id,
            ):
                node_events.append(node_name)
                final_state.update(node_output or {})
                progress_placeholder.info(
                    f"**Pipeline progress:** {' → '.join(node_events)}"
                )
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            logger.error("Pipeline failed: %s", exc, exc_info=True)
            return

    progress_placeholder.empty()
    st.session_state.current_result = final_state
    _render_results(final_state, run_id)


def _render_results(state: Dict[str, Any], run_id: str) -> None:
    """Render the full result: final note, agent cards, feedback panel."""
    query_text = state.get("user_query", "")
    ticker: Optional[str] = state.get("ticker")
    final_note: Optional[str] = state.get("final_summary")
    plan: Optional[Dict[str, Any]] = state.get("plan") or {}
    complexity: Optional[int] = plan.get("complexity") if plan else None
    agent_errors: Dict[str, str] = state.get("agent_errors") or {}

    # ── Final research note ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"Research Note{f' — {ticker}' if ticker else ''}")
    if final_note:
        st.markdown(final_note)
    else:
        st.warning("No summary was generated.")

    # ── Agent output cards ────────────────────────────────────────────────────
    agent_keys = [
        "business_analyst_output",
        "quant_fundamental_output",
        "financial_modelling_output",
        "web_search_output",
    ]
    agents_present = [k for k in agent_keys if state.get(k)]

    if agents_present:
        st.markdown("---")
        st.subheader("Agent Analysis Cards")
        for key in agents_present:
            _render_agent_card(key, state[key], run_id, ticker, query_text)

    # ── Errors ────────────────────────────────────────────────────────────────
    if agent_errors:
        with st.expander("Agent errors", expanded=False):
            for agent, msg in agent_errors.items():
                st.error(f"**{agent}**: {msg}")

    # ── Overall feedback panel ────────────────────────────────────────────────
    agents_used = [_agent_display_name(k) for k in agents_present]
    _render_feedback_panel(
        run_id=run_id,
        ticker=ticker,
        query_text=query_text,
        complexity=complexity,
        agents_used=agents_used,
        final_note=final_note,
    )


if __name__ == "__main__":
    main()
