"""RLAIF (Reinforcement Learning from AI Feedback) and Explicit Feedback system.

This module provides:
1. RLAIF scoring - Uses DeepSeek Chat API to automatically score reports
2. Database storage for feedback (rl_feedback, user_feedback, prompt_versions)
3. Short-term learning mechanisms based on feedback patterns

PostgreSQL tables:
  - rl_feedback: AI-generated scores for each report dimension
  - user_feedback: Explicit user ratings and comments from UI
  - prompt_versions: Tracking prompt changes for A/B testing
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

_DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
_DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
_DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def _normalise_weaknesses(value: Any) -> List[str]:
    """Return weaknesses as a normalized list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        v = value.strip()
        return [v] if v else []
    return [str(value).strip()]


def _is_uninformative_worst_case(row: Dict[str, Any]) -> bool:
    """True when a row is unusable for planner learning (judge unavailable/etc)."""
    weaknesses = [w.lower() for w in _normalise_weaknesses(row.get("weaknesses"))]
    specific_feedback = str(row.get("specific_feedback") or "").lower()

    if any("judge unavailable" in w for w in weaknesses):
        return True
    if "error calling judge" in specific_feedback:
        return True
    if "unauthorized" in specific_feedback and "deepseek" in specific_feedback:
        return True
    return False


def _infer_report_language(
    output_language: Optional[str],
    user_query: str,
    final_summary: str,
) -> str:
    """Infer report language so the judge can evaluate multilingual outputs reliably."""
    if isinstance(output_language, str) and output_language.strip():
        return output_language.strip().lower()

    q = (user_query or "").lower()
    query_hints = {
        "spanish": ["in spanish", "espanol"],
        "french": ["in french", "francais"],
        "german": ["in german", "deutsch"],
        "japanese": ["in japanese", "nihongo"],
        "korean": ["in korean", "hangul"],
        "mandarin": ["in mandarin", "putonghua", "chinese"],
        "cantonese": ["in cantonese", "yue"],
        "vietnamese": ["in vietnamese", "tieng viet"],
        "arabic": ["in arabic"],
    }
    for lang, hints in query_hints.items():
        if any(h in q for h in hints):
            return lang

    s = final_summary or ""
    for ch in s:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7AF:
            return "korean"
        if 0x3040 <= code <= 0x30FF:
            return "japanese"
        if 0x4E00 <= code <= 0x9FFF:
            return "chinese"
        if 0x0600 <= code <= 0x06FF:
            return "arabic"
        if 0x0400 <= code <= 0x04FF:
            return "cyrillic"
        if 0x0900 <= code <= 0x097F:
            return "devanagari"
        if 0x0E00 <= code <= 0x0E7F:
            return "thai"

    return "english"


def _get_pg_conn():
    """Open a new psycopg2 connection using env vars."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "airflow"),
        user=os.getenv("POSTGRES_USER", "airflow"),
        password=os.getenv("POSTGRES_PASSWORD", "airflow"),
    )


def ensure_feedback_tables_exist() -> None:
    """Create feedback tables if they do not exist yet."""
    ddl = """
    -- RLAIF feedback table (AI-generated scores)
    CREATE TABLE IF NOT EXISTS rl_feedback (
        id                  SERIAL PRIMARY KEY,
        run_id             VARCHAR(50) NOT NULL,
        user_query         TEXT NOT NULL,
        timestamp          TIMESTAMP DEFAULT NOW(),
        factual_accuracy   FLOAT,
        citation_completeness FLOAT,
        analysis_depth     FLOAT,
        structure_compliance FLOAT,
        language_quality   FLOAT,
        overall_score      FLOAT,
        strengths          JSONB DEFAULT '[]',
        weaknesses         JSONB DEFAULT '[]',
        specific_feedback  TEXT,
        agent_blamed       VARCHAR(50),
        report_excerpt     TEXT,
        ticker             VARCHAR(20)
    );
    CREATE INDEX IF NOT EXISTS idx_rl_feedback_run_id ON rl_feedback (run_id);
    CREATE INDEX IF NOT EXISTS idx_rl_feedback_timestamp ON rl_feedback (timestamp);
    CREATE INDEX IF NOT EXISTS idx_rl_feedback_agent_blamed ON rl_feedback (agent_blamed);

    -- User feedback table (explicit ratings)
    CREATE TABLE IF NOT EXISTS user_feedback (
        id                  SERIAL PRIMARY KEY,
        run_id             VARCHAR(50) NOT NULL,
        session_id         VARCHAR(50),
        timestamp          TIMESTAMP DEFAULT NOW(),
        helpful            BOOLEAN NOT NULL,
        comment            TEXT,
        issue_tags         JSONB DEFAULT '[]',
        report_version     VARCHAR(20)
    );
    CREATE INDEX IF NOT EXISTS idx_user_feedback_run_id ON user_feedback (run_id);
    CREATE INDEX IF NOT EXISTS idx_user_feedback_timestamp ON user_feedback (timestamp);

    -- Prompt versions table (tracking prompt changes)
    CREATE TABLE IF NOT EXISTS prompt_versions (
        id                  SERIAL PRIMARY KEY,
        agent_name         VARCHAR(50) NOT NULL,
        version            VARCHAR(20) NOT NULL,
        prompt_text        TEXT NOT NULL,
        deployed_at        TIMESTAMP DEFAULT NOW(),
        deployed_to        FLOAT DEFAULT 1.0,
        avg_score_before   FLOAT,
        avg_score_after    FLOAT,
        improvement_pct    FLOAT,
        weaknesses_addressed JSONB DEFAULT '[]'
    );
    CREATE INDEX IF NOT EXISTS idx_prompt_versions_agent ON prompt_versions (agent_name);
    CREATE INDEX IF NOT EXISTS idx_prompt_versions_version ON prompt_versions (version);
    """
    try:
        conn = _get_pg_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
        conn.close()
        logger.info("[feedback] Feedback tables created/verified successfully")
    except Exception as exc:
        logger.warning("[feedback] Could not create feedback tables: %s", exc)


def _call_deepseek_judge(
    report: str,
    user_query: str,
    agent_outputs_summary: str,
    report_language: str = "english",
) -> Dict[str, Any]:
    """Call DeepSeek as a judge to score the report on multiple dimensions.
    
    Returns a dict with scores for each dimension and overall assessment.
    """
    import requests

    judge_system_prompt = """You are an expert equity research analyst and quality judge.
Your job is to evaluate investment research reports generated by an AI system.

You must score the report on these 4 dimensions (0-10 scale):

1. CITATION COMPLETENESS: Does every claim with numbers have [N] citation? Are citations properly indexed?

2. ANALYSIS DEPTH: Does it explain WHY numbers matter? Or just dump data without analysis?

3. STRUCTURE COMPLIANCE: The report MUST contain ALL 11 of these sections in order:
   1. Executive Summary
   2. Company Overview
   3. Financial Performance
   4. Key Financial Ratios & Valuation
   5. Sentiment & Market Positioning
   6. Growth Prospects
   7. Risk Factors
   8. Competitive Landscape
   9. Management & Governance
   10. Macroeconomic Factors
   11. Analyst Verdict
   Score 10 = all 11 present; subtract 1 point per missing section. Read the ENTIRE report before scoring.

4. LANGUAGE QUALITY: Professional tone in the REPORT LANGUAGE, no generic hype language,
   and terminology appropriate for institutional investment analysis.

IMPORTANT: Do NOT score factual_accuracy. The upstream agent pipelines are deterministic and
grounded in database outputs; this evaluator focuses on citation quality, analytical quality,
structure compliance, and language quality only.

MULTILINGUAL REQUIREMENT:
- The report may be in any language.
- Evaluate quality in the report's own language; do NOT penalize non-English output.
- Judge clarity, professionalism, and analytical precision relative to that language.

IMPORTANT: You must provide a JSON response with these exact keys:
- citation_completeness (float 0-10)
- analysis_depth (float 0-10)
- structure_compliance (float 0-10)
- language_quality (float 0-10)
- overall_score (float 0-10, weighted average)
- strengths (list of 3 strings)
- weaknesses (list of 3 strings)
- specific_feedback (string with detailed feedback)
- agent_blamed (string: "business_analyst" | "quant_fundamental" | "financial_modelling" | "web_search" | "summarizer" | "none")

The overall_score should weight: citation_completeness 30%, analysis_depth 35%,
structure_compliance 20%, language_quality 15%.

agent_blamed should identify which agent most likely caused any issues:
- If missing citations → summarizer
- If analysis shallow → business_analyst
- If missing recent news → web_search
- If none of the above → "none"
"""

    judge_user_prompt = f"""Evaluate this research report:

USER QUERY: {user_query}

REPORT LANGUAGE: {report_language}

AGENT OUTPUTS SUMMARY (ground truth — use these to check factual accuracy):
{agent_outputs_summary}

REPORT TO EVALUATE (full report, {len(report)} characters):
{report}

Provide your evaluation as a JSON object."""

    try:
        headers = {
            "Authorization": f"Bearer {_DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        messages = [
            {"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": judge_user_prompt},
        ]
        payload = {
            "model": _DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 3000,
        }
        resp = requests.post(
            f"{_DEEPSEEK_BASE_URL}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Extract JSON from response
        return _extract_json_from_response(content)
    except Exception as exc:
        logger.warning("[feedback] DeepSeek judge call failed: %s", exc)
        return {
            "citation_completeness": 5.0,
            "analysis_depth": 5.0,
            "structure_compliance": 5.0,
            "language_quality": 5.0,
            "overall_score": 5.0,
            "strengths": ["Unable to evaluate"],
            "weaknesses": ["Judge unavailable"],
            "specific_feedback": f"Error calling judge: {exc}",
            "agent_blamed": "none",
        }


def _extract_json_from_response(text: str) -> Dict[str, Any]:
    """Extract JSON from the judge's response.
    
    Uses brace-balanced extraction to correctly handle nested JSON objects
    (e.g. lists in 'strengths'/'weaknesses' fields).
    """
    import re

    # Strip markdown code fences if present
    text = re.sub(r'```(?:json)?\s*', '', text).strip()

    # Try to parse entire response as JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Brace-balanced scan: find the outermost { ... } block
    start = text.find('{')
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # malformed — fall through to fallback
    
    # Fallback to default scores
    logger.warning("[feedback] Could not parse judge response as JSON")
    return {
        "citation_completeness": 5.0,
        "analysis_depth": 5.0,
        "structure_compliance": 5.0,
        "language_quality": 5.0,
        "overall_score": 5.0,
        "strengths": ["Parse error"],
        "weaknesses": ["Could not evaluate"],
        "specific_feedback": "Could not parse judge response",
        "agent_blamed": "none",
    }


def score_report_with_rlaif(
    run_id: str,
    user_query: str,
    final_summary: str,
    agent_outputs: Dict[str, Any],
    ticker: Optional[str] = None,
    output_language: Optional[str] = None,
) -> Dict[str, Any]:
    """Score a report using RLAIF and store the feedback.
    
    Args:
        run_id: Unique identifier for this run
        user_query: The original user query
        final_summary: The generated report
        agent_outputs: Dict containing all agent outputs for reference
        ticker: Optional ticker symbol
        output_language: Optional target output language detected by orchestration
        
    Returns:
        Dict with RLAIF scores and feedback
    """
    ensure_feedback_tables_exist()
    
    # Build agent outputs summary for the judge
    outputs_parts = []
    if agent_outputs.get("business_analyst_output"):
        ba = agent_outputs["business_analyst_output"]
        # sentiment may be a plain string (e.g. "positive") or a dict with 'verdict'
        _ba_sent = ba.get("sentiment")
        if isinstance(_ba_sent, dict):
            sent_verdict = _ba_sent.get("verdict", "N/A")
        elif isinstance(_ba_sent, str):
            sent_verdict = _ba_sent
        else:
            sent_verdict = "N/A"
        outputs_parts.append(
            f"Business Analyst: confidence={ba.get('confidence')}, sentiment={sent_verdict}, "
            f"analysis_present={'yes' if ba.get('analysis') else 'no'}"
        )
    if agent_outputs.get("quant_fundamental_output"):
        qf = agent_outputs["quant_fundamental_output"]
        vf = qf.get("value_factors") or {}
        qfact = qf.get("quality_factors") or {}
        gf = qf.get("growth_factors") or {}
        mf = qf.get("momentum_factors") or {}
        # Also look at top-level keys for convenience
        pe = vf.get("pe_trailing") or qf.get("pe_trailing")
        ev_ebitda = vf.get("ev_ebitda") or qf.get("ev_ebitda")
        roe_raw = qfact.get("roe") or qf.get("roe")
        roic_raw = qfact.get("roic") or qf.get("roic")
        piotroski = qfact.get("piotroski_f_score") or qf.get("piotroski_f_score")
        beneish = qfact.get("beneish_m_score") or qf.get("beneish_m_score")
        altman_z = qfact.get("altman_z_score") or qf.get("altman_z_score")
        # Format percentages clearly
        roe_fmt = f"{roe_raw * 100:.2f}%" if roe_raw is not None else "N/A"
        roic_fmt = f"{roic_raw * 100:.2f}%" if roic_raw is not None else "N/A"
        outputs_parts.append(
            f"Quant Fundamental: P/E={pe}, EV/EBITDA={ev_ebitda}, "
            f"ROE={roe_fmt}, ROIC={roic_fmt}, "
            f"Piotroski={piotroski}/9, Beneish={beneish}, Altman-Z={altman_z}"
        )
        # Add quarterly financials so judge can verify revenue/income figures
        qt = qf.get("quarterly_trends") or []
        if qt:
            q_latest = qt[0]
            rev_b = q_latest.get("revenue", 0) / 1e9 if q_latest.get("revenue") else None
            ni_b = q_latest.get("net_income", 0) / 1e9 if q_latest.get("net_income") else None
            oi_b = q_latest.get("operating_income", 0) / 1e9 if q_latest.get("operating_income") else None
            period = q_latest.get("period", "latest")
            gm = q_latest.get("gross_margin")
            gm_fmt = f"{gm*100:.1f}%" if gm else "N/A"
            rev_fmt = f"${rev_b:.1f}B" if rev_b else "N/A"
            ni_fmt  = f"${ni_b:.1f}B" if ni_b else "N/A"
            oi_fmt  = f"${oi_b:.1f}B" if oi_b else "N/A"
            outputs_parts.append(
                f"Latest Quarter ({period}): Revenue={rev_fmt}, Net_Income={ni_fmt}, "
                f"Operating_Income={oi_fmt}, Gross_Margin={gm_fmt}"
            )
            # TTM totals (sum of 4 most recent quarters)
            if len(qt) >= 4:
                ttm_rev = sum((q.get("revenue") or 0) for q in qt[:4]) / 1e9
                ttm_ni  = sum((q.get("net_income") or 0) for q in qt[:4]) / 1e9
                ttm_oi  = sum((q.get("operating_income") or 0) for q in qt[:4]) / 1e9
                outputs_parts.append(
                    f"TTM (last 4 quarters): Revenue=${ttm_rev:.1f}B, Net_Income=${ttm_ni:.1f}B, "
                    f"Operating_Income=${ttm_oi:.1f}B"
                )
        # Other key metrics
        km = qf.get("key_metrics") or {}
        gm_ann = km.get("gross_margin")
        ebit_m = km.get("ebit_margin")
        curr_r = km.get("current_ratio")
        d2e    = km.get("debt_to_equity")
        beta   = (qf.get("momentum_risk") or qf.get("momentum_factors") or {}).get("beta_60d")
        sharpe = (qf.get("momentum_risk") or qf.get("momentum_factors") or {}).get("sharpe_ratio_12m")
        si     = qf.get("short_interest") or {}
        sh_out = si.get("shares_outstanding")
        outputs_parts.append(
            f"Fundamentals: gross_margin={f'{gm_ann*100:.1f}%' if gm_ann else 'N/A'}, "
            f"ebit_margin={f'{ebit_m*100:.1f}%' if ebit_m else 'N/A'}, "
            f"current_ratio={curr_r or 'N/A'}, D/E={d2e or 'N/A'}, "
            f"beta_60d={f'{beta:.4f}' if beta is not None else 'N/A'}, "
            f"sharpe_ratio_12m={f'{sharpe:.4f}' if sharpe is not None else 'N/A'}, "
            f"shares_out={f'{sh_out/1e9:.2f}B' if sh_out else 'N/A'}"
        )
        # YoY growth
        yoy = qf.get("yoy_deltas") or {}
        rev_yoy = yoy.get("revenue_yoy_pct")
        ni_yoy  = yoy.get("net_income_yoy_pct") or yoy.get("net_income_yoy_pct")
        if rev_yoy is not None:
            outputs_parts.append(
                f"YoY Growth: Revenue={rev_yoy:.1f}%, Net_Income={ni_yoy:.1f}%"
            )
        # Run fact-checker to surface any residual violations in the report
        try:
            from .validation import validate_quant_output, FactChecker  # type: ignore[import]
            _fm_o = agent_outputs.get("financial_modelling_output")
            _metrics = validate_quant_output(qf, fm_output=_fm_o)
            if _metrics:
                _fc = FactChecker()
                _violations = _fc.find_violations(final_summary, _metrics)
                if _violations:
                    viol_lines = "; ".join(
                        f"{v['metric']}: reported {v['reported']} vs expected {v['expected']} ({v['deviation_pct']}% off)"
                        for v in _violations
                    )
                    outputs_parts.append(
                        f"[FactChecker] REMAINING VIOLATIONS after post-processing: {viol_lines}"
                    )
                else:
                    outputs_parts.append("[FactChecker] All key metrics match DB ground truth.")
        except Exception as _fce:
            pass
    if agent_outputs.get("financial_modelling_output"):
        fm = agent_outputs["financial_modelling_output"]
        dcf = (fm.get("valuation") or {}).get("dcf") or {}
        cur_price = fm.get("current_price")
        outputs_parts.append(
            f"Financial Modelling: current_price=${cur_price}, "
            f"DCF_base={dcf.get('intrinsic_value_base')}, "
            f"DCF_bull={dcf.get('intrinsic_value_bull')}, "
            f"DCF_bear={dcf.get('intrinsic_value_bear')}, "
            f"WACC={dcf.get('wacc_used')}, "
            f"DCF_weighted={dcf.get('intrinsic_value_weighted')}"
        )
        # Three-statement model ground truth — the most recent annual period
        tsm = fm.get("three_statement_model") or {}
        inc_stmts = tsm.get("income_statements") or []
        bs_stmts  = tsm.get("balance_sheets") or []
        cf_stmts  = tsm.get("cash_flows") or []
        if inc_stmts:
            inc = inc_stmts[0]
            rev_b  = (inc.get("revenue") or 0) / 1e9
            ni_b   = (inc.get("net_income") or 0) / 1e9
            oi_b   = (inc.get("operating_income") or 0) / 1e9
            ebitda = (inc.get("ebitda") or 0) / 1e9
            outputs_parts.append(
                f"Income Statement ({inc.get('period')}): Revenue=${rev_b:.1f}B, "
                f"Net_Income=${ni_b:.1f}B, Operating_Income=${oi_b:.1f}B, EBITDA=${ebitda:.1f}B"
            )
        if bs_stmts:
            bs = bs_stmts[0]
            assets  = (bs.get("total_assets") or 0) / 1e9
            liab    = (bs.get("total_liabilities") or 0) / 1e9
            equity  = assets - liab if assets and liab else None
            cash    = (bs.get("cash_and_equivalents") or 0) / 1e9
            ltd     = (bs.get("long_term_debt") or 0) / 1e9
            outputs_parts.append(
                f"Balance Sheet ({bs.get('period')}): Total_Assets=${assets:.1f}B, "
                f"Total_Liabilities=${liab:.1f}B, Equity={f'${equity:.1f}B' if equity else 'N/A'}, "
                f"Cash=${cash:.1f}B, LT_Debt=${ltd:.1f}B"
            )
        if cf_stmts:
            cf = cf_stmts[0]
            ocf  = (cf.get("operating_cash_flow") or 0) / 1e9
            fcf  = (cf.get("free_cash_flow") or 0) / 1e9
            capex = (cf.get("capital_expenditures") or 0) / 1e9
            outputs_parts.append(
                f"Cash Flow ({cf.get('period')}): Operating_CF=${ocf:.1f}B, FCF=${fcf:.1f}B, Capex=${capex:.1f}B"
            )
        # Dividends
        div = fm.get("dividends") or {}
        div_yield = div.get("dividend_yield")
        div_ann   = div.get("annual_dividend")
        if div_yield is not None:
            outputs_parts.append(
                f"Dividends: yield={f'{div_yield*100:.2f}%'}, annual=${div_ann}"
            )
        # Earnings
        earn = fm.get("earnings") or {}
        eps_act = earn.get("last_eps_actual")
        beat    = earn.get("beat_streak")
        if eps_act is not None:
            outputs_parts.append(
                f"Earnings: last_EPS=${eps_act}, beat_streak={beat}"
            )
    if agent_outputs.get("web_search_output"):
        ws = agent_outputs["web_search_output"]
        outputs_parts.append(
            f"Web Search: sentiment={ws.get('sentiment_signal')}, "
            f"news_count={len(ws.get('breaking_news', []))}"
        )
    
    agent_outputs_summary = "\n".join(outputs_parts) if outputs_parts else "No agent outputs available"
    
    # Call DeepSeek judge (language-aware for multilingual summaries)
    report_language = _infer_report_language(output_language, user_query, final_summary)
    scores = _call_deepseek_judge(
        final_summary,
        user_query,
        agent_outputs_summary,
        report_language=report_language,
    )
    
    # Store in database
    try:
        conn = _get_pg_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rl_feedback
                        (run_id, user_query, factual_accuracy, citation_completeness, 
                         analysis_depth, structure_compliance, language_quality, 
                         overall_score, strengths, weaknesses, specific_feedback, 
                         agent_blamed, report_excerpt, ticker)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        user_query,
                        scores.get("factual_accuracy"),
                        scores.get("citation_completeness"),
                        scores.get("analysis_depth"),
                        scores.get("structure_compliance"),
                        scores.get("language_quality"),
                        scores.get("overall_score"),
                        json.dumps(scores.get("strengths", [])),
                        json.dumps(scores.get("weaknesses", [])),
                        scores.get("specific_feedback"),
                        scores.get("agent_blamed"),
                        final_summary[:1000] if final_summary else None,
                        ticker,
                    ),
                )
        conn.close()
        logger.info("[feedback] RLAIF scores stored for run_id=%s, overall=%.2f", 
                    run_id, scores.get("overall_score", 0))
    except Exception as exc:
        logger.warning("[feedback] Failed to store RLAIF feedback: %s", exc)
    
    # Check if score is below threshold and log warning
    overall_score = scores.get("overall_score", 0)
    if overall_score < 7.0:
        logger.warning(
            "[feedback] Low RLAIF score (%.2f) for run_id=%s - agent_blamed=%s",
            overall_score, run_id, scores.get("agent_blamed")
        )
    
    return scores


def store_user_feedback(
    run_id: str,
    session_id: Optional[str],
    helpful: bool,
    comment: Optional[str] = None,
    issue_tags: Optional[List[str]] = None,
    report_version: Optional[str] = None,
) -> bool:
    """Store explicit user feedback from the UI.
    
    Args:
        run_id: Unique identifier for the run
        session_id: Optional session identifier
        helpful: True if user clicked thumbs up, False if thumbs down
        comment: Optional user comment
        issue_tags: List of issue types selected by user
        report_version: Which prompt version was used
        
    Returns:
        True if stored successfully
    """
    ensure_feedback_tables_exist()
    
    try:
        conn = _get_pg_conn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_feedback
                        (run_id, session_id, helpful, comment, issue_tags, report_version)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        session_id,
                        helpful,
                        comment,
                        json.dumps(issue_tags or []),
                        report_version,
                    ),
                )
        conn.close()
        logger.info("[feedback] User feedback stored for run_id=%s, helpful=%s", run_id, helpful)
        return True
    except Exception as exc:
        logger.warning("[feedback] Failed to store user feedback: %s", exc)
        return False


def get_recent_rl_feedback(days: int = 7, min_score: float = 7.0) -> List[Dict[str, Any]]:
    """Get recent RLAIF feedback with low scores for analysis.
    
    Args:
        days: Number of days to look back
        min_score: Maximum score threshold (returns scores below this)
        
    Returns:
        List of low-scoring feedback records
    """
    ensure_feedback_tables_exist()
    
    try:
        conn = _get_pg_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM rl_feedback
                WHERE timestamp >= NOW() - INTERVAL '%s days'
                  AND overall_score < %s
                ORDER BY overall_score ASC, timestamp DESC
                """,
                (days, min_score),
            )
            rows = cur.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as exc:
        logger.warning("[feedback] Failed to get recent RLAIF feedback: %s", exc)
        return []


def get_user_feedback_summary(days: int = 7) -> Dict[str, Any]:
    """Get summary statistics for user feedback.
    
    Args:
        days: Number of days to look back
        
    Returns:
        Dict with summary statistics
    """
    ensure_feedback_tables_exist()
    
    try:
        conn = _get_pg_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get total counts
            cur.execute(
                """
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN helpful = true THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN helpful = false THEN 1 ELSE 0 END) as negative
                FROM user_feedback
                WHERE timestamp >= NOW() - INTERVAL '%s days'
                """,
                (days,),
            )
            counts = cur.fetchone() or {}
            
            # Get issue tag distribution
            cur.execute(
                """
                SELECT issue_tags FROM user_feedback
                WHERE timestamp >= NOW() - INTERVAL '%s days'
                  AND issue_tags IS NOT NULL
                  AND jsonb_array_length(issue_tags) > 0
                """,
                (days,),
            )
            tag_counts: Dict[str, int] = {}
            for row in cur.fetchall():
                tags = row.get("issue_tags", [])
                if isinstance(tags, str):
                    tags = json.loads(tags)
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
        conn.close()

        total = counts.get("total", 0) or 0
        positive = counts.get("positive", 0) or 0
        negative = counts.get("negative", 0) or 0
        
        return {
            "total_feedback": total,
            "positive_count": positive,
            "negative_count": negative,
            "positive_pct": (positive / total * 100) if total > 0 else 0,
            "negative_pct": (negative / total * 100) if total > 0 else 0,
            "issue_tag_counts": dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)),
        }
    except Exception as exc:
        logger.warning("[feedback] Failed to get user feedback summary: %s", exc)
        return {
            "total_feedback": 0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_pct": 0,
            "negative_pct": 0,
            "issue_tag_counts": {},
        }


def get_agent_performance_summary(days: int = 7) -> Dict[str, Any]:
    """Get RLAIF performance summary by agent.
    
    Args:
        days: Number of days to look back
        
    Returns:
        Dict with per-agent performance metrics
    """
    ensure_feedback_tables_exist()
    
    try:
        conn = _get_pg_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT 
                    agent_blamed,
                    COUNT(*) as count,
                    AVG(overall_score) as avg_score,
                    AVG(factual_accuracy) as avg_factual,
                    AVG(citation_completeness) as avg_citation,
                    AVG(analysis_depth) as avg_analysis
                FROM rl_feedback
                WHERE timestamp >= NOW() - INTERVAL '%s days'
                  AND agent_blamed IS NOT NULL
                GROUP BY agent_blamed
                ORDER BY avg_score ASC
                """,
                (days,),
            )
            rows = cur.fetchall()
        conn.close()
        return {row["agent_blamed"]: dict(row) for row in rows}
    except Exception as exc:
        logger.warning("[feedback] Failed to get agent performance summary: %s", exc)
        return {}


def check_low_score_alert(days: int = 1, threshold: float = 7.0) -> List[Dict[str, Any]]:
    """Check for runs with low RLAIF scores that may need immediate attention.
    
    Args:
        days: Number of days to look back
        threshold: Score threshold to alert on
        
    Returns:
        List of low-scoring runs
    """
    ensure_feedback_tables_exist()
    
    try:
        conn = _get_pg_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT run_id, user_query, overall_score, agent_blamed, weaknesses, timestamp
                FROM rl_feedback
                WHERE timestamp >= NOW() - INTERVAL '%s days'
                  AND overall_score < %s
                ORDER BY timestamp DESC
                """,
                (days, threshold),
            )
            rows = cur.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as exc:
        logger.warning("[feedback] Failed to check low score alerts: %s", exc)
        return []


def get_worst_cases(
    limit: int = 5,
    min_runs: int = 3,
) -> List[Dict[str, Any]]:
    """Fetch the lowest-scoring runs combining RLAIF + human feedback signals.

    Scoring logic:
      - Base score = rl_feedback.overall_score
      - Penalty of 1.5 if the user also explicitly said unhelpful (thumbs down)
      - Penalised score is used for ordering (ascending = worst first)

    Args:
        limit: How many worst cases to return.
        min_runs: Only activate if at least this many runs exist in the DB.

    Returns:
        List of dicts with keys:
          user_query, ticker, overall_score, agent_blamed,
          weaknesses, specific_feedback, helpful, issue_tags, comment
    """
    ensure_feedback_tables_exist()
    try:
        conn = _get_pg_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM rl_feedback")
            total = (cur.fetchone() or {}).get("cnt", 0)
            if total < min_runs:
                logger.debug(
                    "[feedback] get_worst_cases: only %d run(s) in DB, need %d before activating in-context learning.",
                    total,
                    min_runs,
                )
                return []

            cur.execute(
                """
                SELECT
                    r.user_query,
                    r.ticker,
                    r.overall_score,
                    r.agent_blamed,
                    r.weaknesses,
                    r.specific_feedback,
                    u.helpful,
                    u.issue_tags,
                    u.comment,
                    (r.overall_score - CASE WHEN u.helpful = false THEN 1.5 ELSE 0.0 END) AS penalised_score
                FROM rl_feedback r
                LEFT JOIN user_feedback u ON r.run_id = u.run_id
                ORDER BY penalised_score ASC
                LIMIT %s
                """,
                (max(limit * 10, 20),),
            )
            rows = cur.fetchall()
        conn.close()

        # Filter out rows that carry no useful learning signal (e.g. judge unavailable).
        filtered_rows: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for row in rows:
            item = dict(row)
            if _is_uninformative_worst_case(item):
                continue

            weaknesses = _normalise_weaknesses(item.get("weaknesses"))
            item["weaknesses"] = weaknesses

            # Deduplicate repetitive failures so planner context is diverse.
            key = (
                str(item.get("user_query") or "").strip().lower(),
                str(item.get("ticker") or "").strip().upper(),
                str(item.get("agent_blamed") or "none").strip().lower(),
                "|".join(sorted(w.lower() for w in weaknesses[:3])),
            )
            if key in seen:
                continue
            seen.add(key)
            filtered_rows.append(item)
            if len(filtered_rows) >= limit:
                break

        return filtered_rows
    except Exception as exc:
        logger.warning("[feedback] get_worst_cases failed (non-fatal): %s", exc)
        return []


def get_lowest_rlaif_cases(limit: int = 5, min_runs: int = 3) -> List[Dict[str, Any]]:
    """Fetch lowest-scored RLAIF rows only (no user-feedback join sorting)."""
    ensure_feedback_tables_exist()
    try:
        conn = _get_pg_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM rl_feedback")
            total = (cur.fetchone() or {}).get("cnt", 0)
            if total < min_runs:
                logger.debug(
                    "[feedback] get_lowest_rlaif_cases: only %d run(s), need %d.",
                    total,
                    min_runs,
                )
                return []

            cur.execute(
                """
                SELECT
                    run_id,
                    user_query,
                    ticker,
                    overall_score,
                    agent_blamed,
                    weaknesses,
                    specific_feedback,
                    timestamp
                FROM rl_feedback
                ORDER BY overall_score ASC NULLS LAST, timestamp DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = [dict(r) for r in cur.fetchall()]
        conn.close()

        filtered: List[Dict[str, Any]] = []
        for row in rows:
            if _is_uninformative_worst_case(row):
                continue
            row["weaknesses"] = _normalise_weaknesses(row.get("weaknesses"))
            filtered.append(row)
        return filtered
    except Exception as exc:
        logger.warning("[feedback] get_lowest_rlaif_cases failed (non-fatal): %s", exc)
        return []


def get_latest_user_feedback(limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch latest explicit human feedback rows for planner in-context learning."""
    ensure_feedback_tables_exist()
    try:
        conn = _get_pg_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    run_id,
                    session_id,
                    helpful,
                    comment,
                    issue_tags,
                    report_version,
                    timestamp
                FROM user_feedback
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = [dict(r) for r in cur.fetchall()]
        conn.close()

        for row in rows:
            tags = row.get("issue_tags")
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except Exception:
                    tags = [tags]
            row["issue_tags"] = [str(t).strip() for t in (tags or []) if str(t).strip()]
            row["comment"] = str(row.get("comment") or "").strip()
        return rows
    except Exception as exc:
        logger.warning("[feedback] get_latest_user_feedback failed (non-fatal): %s", exc)
        return []


__all__ = [
    "ensure_feedback_tables_exist",
    "score_report_with_rlaif",
    "store_user_feedback",
    "get_recent_rl_feedback",
    "get_user_feedback_summary",
    "get_agent_performance_summary",
    "check_low_score_alert",
    "get_worst_cases",
    "get_lowest_rlaif_cases",
    "get_latest_user_feedback",
]
