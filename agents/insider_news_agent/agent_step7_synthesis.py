"""
Step 7: DeepSeek-v4-pro synthesis with 3 concurrent LLM tasks.

For insider + news analysis, we run:
  1. **Task A — Insider Trading Pattern Analysis**: Analyze insider activity,
     buy/sell ratios, conviction levels, insider ranks, red flags.
  2. **Task B — News Sentiment & Catalysts**: Extract news themes, sentiment,
     major catalysts, credibility assessment.
  3. **Task C — Combined Investment Thesis**: Synthesize insider + news signals
     into coherent investment thesis with bull/bear cases and recommendation.

Uses **deepseek-v4-pro** model (not deepseek-chat) with temperature=0
for extended thinking capability.

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

# Lazy LLM singleton
_llm: "ChatOpenAI | None" = None


def _get_llm() -> "ChatOpenAI":
    """Initialize deepseek-v4-pro LLM (lazy singleton)."""
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
            model="deepseek-v4-pro",
            base_url="https://api.deepseek.com",
            api_key=SecretStr(api_key),
            temperature=0,
            model_kwargs={"max_tokens": 8000, "reasoning_effort": "high"},  # Enable thinking mode
        )
    return _llm


# ── Citation extraction ───────────────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[([^\[\]]+?)\s+p\.(\d+)\]")
_PLACEHOLDER_RE = re.compile(
    r"\[doc_name\s+p\.\d+\]|\[Document\s+p\.\d+\]|\[source\s+p\.\d+\]",
    re.IGNORECASE,
)


def extract_citations(text: str) -> list[dict]:
    """Pull all [doc_name p.N] citations from an LLM response."""
    return [
        {"doc_name": m.group(1).strip(), "page": int(m.group(2))}
        for m in _CITATION_RE.finditer(text)
    ]


def has_placeholder_citations(text: str) -> bool:
    """Return True if the response contains placeholder citations."""
    return bool(_PLACEHOLDER_RE.search(text))


# ── Prompt builders ───────────────────────────────────────────────────────────

CITATION_RULE = (
    "Every claim MUST cite the source as [source_name p.N] using the EXACT source name and page number. "
    "No placeholder names. No ungrounded statements."
)

MAX_EVIDENCE_CHARS = 4000


def _cap_evidence(ev: str) -> str:
    """Truncate evidence to prevent oversized prompts."""
    return ev[:MAX_EVIDENCE_CHARS] + "\n...[truncated]" if len(ev) > MAX_EVIDENCE_CHARS else ev


def _prompt_insider_analysis(
    ticker: str,
    insider_evidence: str,
) -> str:
    """Prompt for Task A: Insider trading pattern analysis."""
    return f"""
You are an expert financial analyst specializing in insider trading signals.

Analyze the insider trading transactions below for {ticker} and provide:

1. Summary of insider activity (buys vs sells, dollar value, frequency)
2. Buy/sell ratio (if possible, calculate the ratio of buy shares to sell shares)
3. Net position assessment ("net buyers", "net sellers", or "neutral")
4. Insider conviction level ("high", "medium", or "low")
5. Notable insider names and titles (C-suite vs board vs employees)
6. Insider sentiment ("bullish", "bearish", or "neutral")
7. Any red flags or concerning patterns

EVIDENCE:
{_cap_evidence(insider_evidence)}

CITATION RULE:
{CITATION_RULE}

Format your response as JSON with these exact keys:
{{
  "activity_summary": "str - 2-3 sentence summary",
  "buy_sell_ratio": "float or null",
  "net_position": "net buyers | net sellers | neutral",
  "conviction": "high | medium | low",
  "notable_insiders": [
    {{"name": "str", "title": "str", "recent_activity": "str"}}
  ],
  "insider_sentiment": "bullish | bearish | neutral",
  "red_flags": ["str", ...]
}}

Respond ONLY with the JSON object, no other text.
"""


def _prompt_news_analysis(
    ticker: str,
    news_evidence: str,
) -> str:
    """Prompt for Task B: News sentiment and catalysts analysis."""
    return f"""
You are an expert financial analyst specializing in news sentiment and market catalysts.

Analyze the news articles below for {ticker} and provide:

1. Overall sentiment assessment (positive, negative, or neutral)
2. Average sentiment score (-1.0 to 1.0 if possible)
3. Sentiment trend (improving, deteriorating, or stable)
4. Major positive catalysts (product launches, partnerships, earnings beats, etc.)
5. Major negative catalysts (lawsuits, regulatory issues, earnings misses, etc.)
6. Key news themes/topics (AI, Regulation, M&A, Earnings, Competition, etc.)
7. Credibility assessment based on news sources

EVIDENCE:
{_cap_evidence(news_evidence)}

CITATION RULE:
{CITATION_RULE}

Format your response as JSON with these exact keys:
{{
  "sentiment_summary": "str - 2-3 sentence summary",
  "avg_sentiment_score": "float between -1.0 and 1.0",
  "sentiment_trend": "improving | deteriorating | stable",
  "positive_catalysts": ["str", ...],
  "negative_catalysts": ["str", ...],
  "key_themes": ["str", ...],
  "credibility": "str - 1-2 sentence credibility assessment"
}}

Respond ONLY with the JSON object, no other text.
"""


def _prompt_combined_thesis(
    ticker: str,
    insider_analysis: dict,
    news_analysis: dict,
    insider_evidence: str,
    news_evidence: str,
) -> str:
    """Prompt for Task C: Combined investment thesis."""
    return f"""
You are an expert investment strategist synthesizing multiple data sources.

Given the insider trading analysis and news analysis for {ticker}, create a comprehensive
investment thesis that:

1. Assesses signal alignment (insider sentiment vs news sentiment)
2. Identifies bull case evidence (why to be bullish)
3. Identifies bear case evidence (why to be bearish)
4. Lists key risks (top 2-3 downside risks)
5. Lists key opportunities (top 2-3 upside opportunities)
6. Provides overall investment recommendation (buy, hold, or sell)
7. Assesses conviction level (high, medium, or low)

Insider Analysis Summary:
{json.dumps(insider_analysis, indent=2)}

News Analysis Summary:
{json.dumps(news_analysis, indent=2)}

CITATION RULE:
{CITATION_RULE}

Format your response as JSON with these exact keys:
{{
  "combined_thesis": "str - 3-5 sentence investment thesis",
  "signal_alignment": "aligned | mixed | conflicting",
  "bull_case": "str - evidence for bullish case",
  "bear_case": "str - evidence for bearish case",
  "key_risks": ["str", ...],
  "key_opportunities": ["str", ...],
  "recommendation": "buy | hold | sell",
  "conviction": "high | medium | low"
}}

Respond ONLY with the JSON object, no other text.
"""


# ── LLM task execution ─────────────────────────────────────────────────────────

def _run_task_a(ticker: str, insider_evidence: str, max_retries: int = 2) -> dict:
    """Run Task A: Insider trading pattern analysis."""
    llm = _get_llm()
    prompt = _prompt_insider_analysis(ticker, insider_evidence)

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Calling DeepSeek-reasoner for task: insider_analysis (attempt {attempt}) ...")
            response = llm.invoke(prompt)
            text = response.content

            # Check for placeholder citations
            if has_placeholder_citations(text):
                if attempt < max_retries:
                    print(f"    Placeholder citations detected, retrying...")
                    continue
                print(f"    WARNING: Placeholder citations in final response")

            # Extract JSON
            analysis = json.loads(text)
            citations = extract_citations(str(response.content))

            return {
                "task": "insider_analysis",
                "analysis": analysis,
                "citations": citations,
                "chars": len(text),
            }

        except json.JSONDecodeError as e:
            if attempt < max_retries:
                print(f"    JSON parse error, retrying...")
                continue
            return {
                "task": "insider_analysis",
                "error": f"JSON decode error: {e}",
            }
        except Exception as e:
            if attempt < max_retries:
                print(f"    Error, retrying: {e}")
                continue
            return {
                "task": "insider_analysis",
                "error": str(e),
            }

    return {"task": "insider_analysis", "error": "Max retries exceeded"}


def _run_task_b(ticker: str, news_evidence: str, max_retries: int = 2) -> dict:
    """Run Task B: News sentiment and catalysts analysis."""
    llm = _get_llm()
    prompt = _prompt_news_analysis(ticker, news_evidence)

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Calling DeepSeek-reasoner for task: news_analysis (attempt {attempt}) ...")
            response = llm.invoke(prompt)
            text = response.content

            if has_placeholder_citations(text):
                if attempt < max_retries:
                    print(f"    Placeholder citations detected, retrying...")
                    continue
                print(f"    WARNING: Placeholder citations in final response")

            analysis = json.loads(text)
            citations = extract_citations(str(response.content))

            return {
                "task": "news_analysis",
                "analysis": analysis,
                "citations": citations,
                "chars": len(text),
            }

        except json.JSONDecodeError as e:
            if attempt < max_retries:
                print(f"    JSON parse error, retrying...")
                continue
            return {
                "task": "news_analysis",
                "error": f"JSON decode error: {e}",
            }
        except Exception as e:
            if attempt < max_retries:
                print(f"    Error, retrying: {e}")
                continue
            return {
                "task": "news_analysis",
                "error": str(e),
            }

    return {"task": "news_analysis", "error": "Max retries exceeded"}


def _run_task_c(
    ticker: str,
    insider_analysis: dict,
    news_analysis: dict,
    insider_evidence: str,
    news_evidence: str,
    max_retries: int = 2,
) -> dict:
    """Run Task C: Combined investment thesis."""
    llm = _get_llm()
    prompt = _prompt_combined_thesis(ticker, insider_analysis, news_analysis, insider_evidence, news_evidence)

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Calling DeepSeek-reasoner for task: combined_thesis (attempt {attempt}) ...")
            response = llm.invoke(prompt)
            text = response.content

            if has_placeholder_citations(text):
                if attempt < max_retries:
                    print(f"    Placeholder citations detected, retrying...")
                    continue
                print(f"    WARNING: Placeholder citations in final response")

            analysis = json.loads(text)
            citations = extract_citations(str(response.content))

            return {
                "task": "combined_thesis",
                "analysis": analysis,
                "citations": citations,
                "chars": len(text),
            }

        except json.JSONDecodeError as e:
            if attempt < max_retries:
                print(f"    JSON parse error, retrying...")
                continue
            return {
                "task": "combined_thesis",
                "error": f"JSON decode error: {e}",
            }
        except Exception as e:
            if attempt < max_retries:
                print(f"    Error, retrying: {e}")
                continue
            return {
                "task": "combined_thesis",
                "error": str(e),
            }

    return {"task": "combined_thesis", "error": "Max retries exceeded"}


# ── Main synthesis function ───────────────────────────────────────────────────

def run_synthesis(
    insider_docs: list[Document],
    news_docs: list[Document],
    ticker: str,
) -> dict:
    """
    Run 3 concurrent LLM tasks to synthesize insider + news analysis.

    Returns
    -------
    Dict with insider_analysis, news_analysis, combined_thesis, all_citations
    """
    # Build evidence strings
    insider_evidence = "\n\n".join([doc.page_content for doc in insider_docs]) if insider_docs else "No insider data available."
    news_evidence = "\n\n".join([doc.page_content for doc in news_docs]) if news_docs else "No news data available."

    print(f"\n  Running Tasks A, B, C concurrently ...\n")

    # Run tasks concurrently
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=3) as executor:
        # Task A: Insider analysis (only needs insider_evidence)
        future_a = executor.submit(_run_task_a, ticker, insider_evidence)

        # Task B: News analysis (only needs news_evidence)
        future_b = executor.submit(_run_task_b, ticker, news_evidence)

        # Task A needs to complete first for Task C
        result_a = future_a.result()
        result_b = future_b.result()

        # Extract insider_analysis for Task C
        insider_analysis = result_a.get("analysis", {})
        news_analysis = result_b.get("analysis", {})

        # Task C: Combined thesis (needs A + B outputs)
        future_c = executor.submit(
            _run_task_c,
            ticker,
            insider_analysis,
            news_analysis,
            insider_evidence,
            news_evidence,
        )
        result_c = future_c.result()

    elapsed = time.time() - start_time

    # Aggregate results
    all_citations = []
    for result in [result_a, result_b, result_c]:
        if "citations" in result:
            all_citations.extend(result["citations"])

    # Deduplicate citations
    seen = set()
    unique_citations = []
    for cit in all_citations:
        key = (cit.get("doc_name"), cit.get("page"))
        if key not in seen:
            seen.add(key)
            unique_citations.append(cit)

    return {
        "insider_analysis": result_a.get("analysis", {}),
        "insider_error": result_a.get("error"),
        "news_analysis": result_b.get("analysis", {}),
        "news_error": result_b.get("error"),
        "combined_thesis": result_c.get("analysis", {}),
        "thesis_error": result_c.get("error"),
        "all_citations": unique_citations,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    # Test run
    from agent_step1_pg import fetch_insider_and_news_data

    ticker = "AAPL"
    insider_docs, news_docs, meta = fetch_insider_and_news_data(ticker)

    print(f"\n{'='*60}")
    print(f"  Synthesis: {ticker}")
    print(f"{'='*60}")

    result = run_synthesis(insider_docs, news_docs, ticker)

    print(f"\n[timing] All 3 tasks concurrent: {result['elapsed_seconds']:.1f}s")
    print(f"Insider Analysis: {result.get('insider_error') or 'OK'}")
    print(f"News Analysis: {result.get('news_error') or 'OK'}")
    print(f"Combined Thesis: {result.get('thesis_error') or 'OK'}")
    print(f"Citations found: {len(result['all_citations'])}")
