"""
Step 7: DeepSeek-reasoner synthesis with 3 concurrent LLM tasks.

For macro analysis, we run:
  1. **Task A — Macro Regime Detection**: Identify regime (risk-on/off/stagflation) 
     + 3-5 macro themes with direction, confidence, transmission channel, etc.
  2. **Task B — Per-Report Key Idea + Stock Linkage**: For EACH macro report,
     summarize thesis and link to target ticker.
  3. **Task C — Impact Synthesis**: Top 2-3 macro drivers + top risk scenario.

Uses **deepseek-reasoner** model (not deepseek-chat) with temperature=0
for extended thinking / chain-of-thought capability.

Output per task: {"task": ..., "analysis": ..., "citations_found": [...]}

Run:
    python agent_step7_synthesis.py
"""

from __future__ import annotations

import os
import warnings
import re
import time
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from pydantic import SecretStr

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
TICKER   = "AAPL"

# Lazy LLM singleton — instantiated on first use so import never crashes when
# DEEPSEEK_API_KEY is absent (e.g. during unit tests or import-time checks).
_llm: "ChatOpenAI | None" = None

def _get_llm() -> "ChatOpenAI":
    global _llm
    if _llm is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise EnvironmentError("DEEPSEEK_API_KEY not found in environment / .env")
        warnings.filterwarnings(
            "ignore",
            message=r"Parameters \{'max_tokens'\} should be specified explicitly.*",
            category=UserWarning,
        )
        _llm = ChatOpenAI(
            model="deepseek-reasoner",  # Use reasoner model (not deepseek-chat)
            base_url="https://api.deepseek.com",
            api_key=SecretStr(api_key),
            temperature=0,
            model_kwargs={"max_tokens": 8000},  # Reasoner needs more tokens for thinking
        )
    return _llm

# ── Citation extractor ────────────────────────────────────────────────────────

_CITATION_RE       = re.compile(r"\[([^\[\]]+?)\s+p\.(\d+)\]")
_PLACEHOLDER_RE    = re.compile(r"\[doc_name\s+p\.\d+\]|\[Document\s+p\.\d+\]|\[source\s+p\.\d+\]", re.IGNORECASE)


def extract_citations(text: str) -> list[dict]:
    """Pull all [doc_name p.N] citations from an LLM response."""
    return [
        {"doc_name": m.group(1).strip(), "page": int(m.group(2))}
        for m in _CITATION_RE.finditer(text)
    ]


def has_placeholder_citations(text: str) -> bool:
    """Return True if the response contains generic placeholder citations instead of real ones."""
    return bool(_PLACEHOLDER_RE.search(text))


# ── Prompt builders ───────────────────────────────────────────────────────────

CITATION_RULE = (
    "Every claim MUST cite the source as [doc_name p.N] using the EXACT filename and page number from the evidence. "
    "No placeholder names like 'doc_name'. No ungrounded statements."
)

# Cap evidence to avoid oversized prompts
MAX_EVIDENCE_CHARS = 4000

def _cap_evidence(ev: str) -> str:
    return ev[:MAX_EVIDENCE_CHARS] + "\n...[truncated]" if len(ev) > MAX_EVIDENCE_CHARS else ev


def _prompt_macro_regime(
    ticker: str,
    macro_evidence: str,
) -> str:
    """Task A: Macro regime detection."""
    return f"""Macro regime analysis for analyzing investment in {ticker}. {CITATION_RULE}

Your task:
1. Identify the current MACRO REGIME from the macro reports. Choose one of:
   - risk-on
   - risk-off
   - growth-at-risk
   - stagflationary
   - other (specify)

2. For the identified regime, list 3-5 MACRO THEMES that define it, with:
   - Theme name
   - Direction for {ticker}: bullish | bearish | neutral
   - Confidence (0.0-1.0)
   - Transmission channel (how this affects {ticker})
   - Impact magnitude: high | medium | low
   - Time horizon: immediate | medium-term | long-term

Return as JSON:
{{
    "regime": "risk-off",
    "themes": [
        {{
            "theme": "Fed Rate Hikes",
            "direction": "bearish",
            "confidence": 0.85,
            "transmission_channel": "higher discount rates reduce equity valuations",
            "impact_magnitude": "high",
            "time_horizon": "immediate"
        }},
        ...
    ]
}}

EVIDENCE:
{_cap_evidence(macro_evidence)}

Cite every claim with [doc_name p.N]. Return ONLY valid JSON."""


def _prompt_per_report_linkage(
    ticker: str,
    macro_evidence: str,
) -> str:
    """Task B: Per-report key idea + stock linkage."""
    return f"""Per-report macro analysis for {ticker}. {CITATION_RULE}

Your task:
For each distinct macro report mentioned in the evidence, provide:
- Report name (and date if available)
- 2-3 sentence summary of the thesis
- Explicit linkage to {ticker} via transmission channels

Return as JSON:
{{
    "reports": [
        {{
            "report_name": "Federal Reserve - December Policy Meeting",
            "report_date": "2025-12-15",
            "summary": "The Fed held rates steady... [cite as [report_name p.N]]",
            "stock_relevance": "{ticker} would be affected via ... [cite as [report_name p.N]]"
        }},
        ...
    ]
}}

EVIDENCE:
{_cap_evidence(macro_evidence)}

Cite every claim with [doc_name p.N]. Return ONLY valid JSON."""


def _prompt_impact_synthesis(
    ticker: str,
    macro_evidence: str,
) -> str:
    """Task C: Impact synthesis."""
    return f"""Macro impact synthesis for {ticker}. {CITATION_RULE}

Your task:
1. Identify the TOP 2-3 MACRO DRIVERS for {ticker} and briefly explain each
2. Identify the TOP RISK SCENARIO that breaks the bull case

Return as JSON:
{{
    "top_drivers": [
        {{
            "driver": "Fed Rate Path",
            "explanation": "{ticker} valuations are sensitive to... [cite as [report_name p.N]]"
        }},
        {{
            "driver": "USD Strength",
            "explanation": "{ticker} has exposure to... [cite as [report_name p.N]]"
        }}
    ],
    "top_risk": "Unexpected inflation spike",
    "risk_scenario": "If inflation surprises to the upside, the Fed could... which would impact {ticker} via... [cite as [report_name p.N]]"
}}

EVIDENCE:
{_cap_evidence(macro_evidence)}

Cite every claim with [doc_name p.N]. Return ONLY valid JSON."""


# ── Task runner ───────────────────────────────────────────────────────────────

def run_analysis_task(
    task_name:    str,
    prompt:       str,
    max_retries:  int = 2,
    max_conn_retries: int = 5,
) -> dict:
    """
    Send prompt to DeepSeek-reasoner, return structured result with citation list.
    - Retries up to max_retries times if placeholder citations are detected.
    - Retries up to max_conn_retries times on APIConnectionError (transient network drops).
    """
    from openai import APIConnectionError as _APIConnectionError

    text = ""
    for attempt in range(1, max_retries + 2):
        print(f"\n  Calling DeepSeek-reasoner for task: {task_name} (attempt {attempt}) ...")
        # Inner retry loop for transient connection errors
        for conn_try in range(1, max_conn_retries + 1):
            try:
                response = _get_llm().invoke(prompt)
                break
            except _APIConnectionError as e:
                wait = 15 * conn_try  # 15s, 30s, 45s, 60s, 75s
                print(f"  [WARN] Connection error (try {conn_try}/{max_conn_retries}): {e} — retrying in {wait}s ...")
                time.sleep(wait)
        else:
            raise RuntimeError(f"DeepSeek unreachable after {max_conn_retries} connection attempts for task '{task_name}'")

        raw  = response.content
        text = raw if isinstance(raw, str) else str(raw)

        if has_placeholder_citations(text):
            print(f"  WARNING: placeholder citations detected in {task_name} — retrying ...")
            continue

        citations = extract_citations(text)
        print(f"  Done. {len(text)} chars, {len(citations)} citations found.")
        return {
            "task":            task_name,
            "analysis":        text,
            "citations_found": citations,
        }

    # All retries exhausted — keep whatever we got, flag the issue
    citations = extract_citations(text)
    print(f"  WARNING: {task_name} still has placeholder citations after {max_retries+1} attempts.")
    return {
        "task":            task_name,
        "analysis":        text,
        "citations_found": citations,
    }


def _format_evidence_simple(docs: list[Document]) -> str:
    """Format documents as simple concatenated text with doc headers."""
    if not docs:
        return "[No evidence documents provided]"
    
    parts = []
    for i, doc in enumerate(docs, 1):
        doc_name = doc.metadata.get("doc_name", f"document_{i}")
        page = doc.metadata.get("page_number", i)
        institution = doc.metadata.get("institution", "")
        filing_date = doc.metadata.get("filing_date", "")
        
        header = f"[{doc_name} p.{page}]"
        if institution:
            header += f" ({institution})"
        if filing_date:
            header += f" [{filing_date}]"
        
        parts.append(header)
        parts.append(doc.page_content[:500])  # Truncate long passages
        parts.append("")
    
    return "\n".join(parts)


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_full_analysis(
    ticker: str,
    macro_pages: list[Document],
    earnings_pages: list[Document],
    macro_doc_names: list[str],
) -> dict:
    """
    Run all 3 macro analysis tasks for a ticker.
    
    Parameters
    ----------
    ticker:
        Uppercase ticker symbol.
    macro_pages:
        Document list of all macro report chunks.
    earnings_pages:
        Document list of latest earnings call for ticker.
    macro_doc_names:
        List of unique source_name values from macro chunks.
    
    Returns
    -------
    dict with keys: ticker, regime, macro_themes, per_report_summaries,
                    top_macro_drivers, top_risk, risk_scenario, citations
    """
    print(f"\n{'='*60}")
    print(f"  Macro analysis: {ticker}")
    print(f"{'='*60}")
    t0 = time.time()

    # Format evidence
    macro_evidence = _format_evidence_simple(macro_pages)
    
    # Build prompts for all 3 tasks
    prompt_a = _prompt_macro_regime(ticker, macro_evidence)
    prompt_b = _prompt_per_report_linkage(ticker, macro_evidence)
    prompt_c = _prompt_impact_synthesis(ticker, macro_evidence)

    # Run all 3 tasks concurrently
    print("\n  Running Tasks A, B, C concurrently ...")
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=3) as pool:
        future_a = pool.submit(run_analysis_task, "macro_regime",        prompt_a)
        future_b = pool.submit(run_analysis_task, "per_report_linkage",  prompt_b)
        future_c = pool.submit(run_analysis_task, "impact_synthesis",    prompt_c)
        task_a = future_a.result()
        task_b = future_b.result()
        task_c = future_c.result()
    print(f"  [timing] All 3 tasks concurrent: {time.time()-t1:.1f}s")

    # Helper to extract JSON from LLM response (which may contain thinking/preamble)
    def _extract_json_from_response(text: str) -> dict | None:
        """Extract JSON object from text that may contain non-JSON preamble."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON block in text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        return None

    # Parse JSON responses
    regime = ""
    macro_themes = []
    per_report_summaries = []
    top_macro_drivers = []
    top_risk = ""
    risk_scenario = ""

    # Task A: Macro regime
    try:
        task_a_json = _extract_json_from_response(task_a["analysis"])
        if task_a_json:
            regime = task_a_json.get("regime", "")
            for theme_dict in task_a_json.get("themes", []):
                macro_themes.append({
                    "theme": theme_dict.get("theme", ""),
                    "direction": theme_dict.get("direction", "neutral"),
                    "confidence": float(theme_dict.get("confidence", 0.5)),
                    "transmission_channel": theme_dict.get("transmission_channel", ""),
                    "impact_magnitude": theme_dict.get("impact_magnitude", "medium"),
                    "time_horizon": theme_dict.get("time_horizon", "medium-term"),
                })
    except Exception as e:
        print(f"  WARNING: Failed to parse Task A JSON: {e}")

    # Task B: Per-report linkage
    try:
        task_b_json = _extract_json_from_response(task_b["analysis"])
        if task_b_json:
            for report_dict in task_b_json.get("reports", []):
                per_report_summaries.append({
                    "report_name": report_dict.get("report_name", ""),
                    "summary": report_dict.get("summary", ""),
                    "stock_relevance": report_dict.get("stock_relevance", ""),
                    "report_date": report_dict.get("report_date", ""),
                })
    except Exception as e:
        print(f"  WARNING: Failed to parse Task B JSON: {e}")

    # Task C: Impact synthesis
    try:
        task_c_json = _extract_json_from_response(task_c["analysis"])
        if task_c_json:
            for driver_dict in task_c_json.get("top_drivers", []):
                top_macro_drivers.append(driver_dict.get("driver", ""))
            top_risk = task_c_json.get("top_risk", "")
            risk_scenario = task_c_json.get("risk_scenario", "")
    except Exception as e:
        print(f"  WARNING: Failed to parse Task C JSON: {e}")

    # Collect all citations
    all_citations = []
    for task in [task_a, task_b, task_c]:
        all_citations.extend(task.get("citations_found") or [])

    print(f"  [timing] TOTAL for {ticker}: {time.time()-t0:.1f}s")

    return {
        "ticker":               ticker,
        "regime":               regime,
        "macro_themes":         macro_themes,
        "per_report_summaries": per_report_summaries,
        "top_macro_drivers":    top_macro_drivers,
        "top_risk":             top_risk,
        "risk_scenario":        risk_scenario,
        "citations":            all_citations,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test with mock data
    from langchain_core.documents import Document
    
    mock_macro = [
        Document(
            page_content="The Federal Reserve raised rates by 25bps in December meeting...",
            metadata={"doc_name": "Federal Reserve Report Dec 2025.pdf", "page_number": 1, "institution": "Federal Reserve", "filing_date": "2025-12-15"},
        ),
    ]
    mock_earnings = [
        Document(
            page_content="Apple reported strong iPhone sales in Q4 2025...",
            metadata={"doc_name": "AAPL Earnings Call Q4 2025.pdf", "page_number": 1, "filing_date": "2025-01-29"},
        ),
    ]
    mock_names = ["Federal Reserve Report Dec 2025.pdf"]

    result = run_full_analysis(TICKER, mock_macro, mock_earnings, mock_names)

    print(f"\n\n{'='*60}")
    print(f"  MACRO ANALYSIS COMPLETE: {result['ticker']}")
    print(f"{'='*60}")
    print(f"Regime: {result['regime']}")
    print(f"Themes: {len(result['macro_themes'])}")
    print(f"Reports: {len(result['per_report_summaries'])}")
    print(f"Drivers: {result['top_macro_drivers']}")
    print(f"Citations: {len(result['citations'])}")
