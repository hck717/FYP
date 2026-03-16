"""
Step 7: DeepSeek synthesis with mandatory citation prompts.

For each analysis task, we:
  1. Retrieve the most relevant chunks from the FAISS evidence index.
  2. Build a focused prompt that includes deterministic features + raw evidence.
  3. Call DeepSeek (via ChatOpenAI wrapper) at temperature=0.
  4. Require EVERY claim to be cited as [doc_name p.N].

Analysis tasks (run in order):
  A. transcript_comparison  — KPI changes, tone shifts, guidance changes latest vs previous
  B. qa_behavior            — Q&A evasiveness, hedge ratio delta, dropped KPIs, confidence
  C. broker_consensus       — structured table + short cited commentary (no long narrative)

Output per task: {"task": ..., "analysis": ..., "citations_found": [...]}

Run:
    python agent_step7_synthesis.py
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from agent_step1_load import list_stock_files, load_pdf_pages
from agent_step3_parse_quality import (
    flag_quality_issues, tag_transcript_sections,
    tag_broker_sections, filter_usable,
)
from agent_step4_broker_labels import extract_all_broker_labels
from agent_step4b_broker_parse import (
    parse_all_broker_docs, format_broker_summary_for_llm,
)
from agent_step5_transcript_features import (
    extract_transcript_features, compare_kpi_coverage,
)
from agent_step6_embeddings import (
    chunk_documents, build_index, build_index_neo4j, retrieve, retrieve_broker_evidence,
    format_evidence_pack, TOP_K,
)

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
TICKER   = "AAPL"
BASE_DIR = Path("data_reports")

# Lazy LLM singleton — instantiated on first use so import never crashes when
# DEEPSEEK_API_KEY is absent (e.g. during unit tests or import-time checks).
_llm: "ChatOpenAI | None" = None

def _get_llm() -> "ChatOpenAI":
    global _llm
    if _llm is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise EnvironmentError("DEEPSEEK_API_KEY not found in environment / .env")
        _llm = ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key=api_key,
            temperature=0,
            max_tokens=500,
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

# Cap evidence packs to avoid oversized prompts
MAX_EVIDENCE_CHARS = 3000

def _cap_evidence(ev: str) -> str:
    return ev[:MAX_EVIDENCE_CHARS] + "\n...[truncated]" if len(ev) > MAX_EVIDENCE_CHARS else ev


def _prompt_transcript_comparison(
    ticker: str,
    latest_feat: dict,
    prev_feat:   dict,
    kpi_diff:    dict,
    evidence:    str,
) -> str:
    return f"""Analyst report for {ticker} earnings call comparison. {CITATION_RULE}

LATEST: {latest_feat['doc_name']}
- KPI density: {latest_feat['kpi_per_1k_words']}/1k words | Hedge ratio: {latest_feat['hedge_ratio']} | Evasive Q&A: {latest_feat['evasive_count']}
- Q&A vs Prep hedge delta: {latest_feat['qa_vs_prep_hedge_delta']} | Pivots/1k: {latest_feat['pivot_per_1k_words']}
- Guidance ranges: {latest_feat['guidance_ranges'][:5]}

PREVIOUS: {prev_feat['doc_name']}
- KPI density: {prev_feat['kpi_per_1k_words']}/1k words | Hedge ratio: {prev_feat['hedge_ratio']}
- Guidance ranges: {prev_feat['guidance_ranges'][:5]}

Dropped KPIs: {kpi_diff['dropped_kpis']} | Added KPIs: {kpi_diff['added_kpis']}

EVIDENCE:
{_cap_evidence(evidence)}

Write 2 short paragraphs: (1) KPI/guidance changes with cited numbers, (2) tone/confidence shift. Cite every claim. Keep under 150 words.
"""


def _prompt_qa_behavior(
    ticker:      str,
    latest_feat: dict,
    evidence:    str,
) -> str:
    return f"""Analyst report for {ticker} Q&A behavior. {CITATION_RULE}

TRANSCRIPT: {latest_feat['doc_name']}
- Q&A hedge ratio: {latest_feat['qa_hedge_ratio']} vs Prep: {latest_feat['prep_hedge_ratio']} (delta: {latest_feat['qa_vs_prep_hedge_delta']})
- Evasive phrases: {latest_feat['evasive_count']} | Pivots/1k: {latest_feat['pivot_per_1k_words']}

EVIDENCE:
{_cap_evidence(evidence)}

Write 2 short paragraphs: (1) confidence/evasiveness delta with cited quotes, (2) what this signals. Cite every claim. Keep under 120 words.
"""


def _prompt_broker_consensus(
    ticker:         str,
    broker_parsed:  list[dict],
    evidence:       str,
    broker_summary: str,
) -> str:
    bullish = sum(1 for b in broker_parsed if b["rating"] == "bullish")
    neutral = sum(1 for b in broker_parsed if b["rating"] == "neutral")
    bearish = sum(1 for b in broker_parsed if b["rating"] == "bearish")
    rating_dist = f"{bullish} bullish / {neutral} neutral / {bearish} bearish"

    return f"""Broker consensus report for {ticker}. {CITATION_RULE}

Ratings: {rating_dist}
{broker_summary}

EVIDENCE:
{_cap_evidence(evidence)}

Write 4 short sections with ### headers:
### Consensus Core — shared thesis (1 sentence, cited number)
### Where Consensus Breaks — key disagreement (1 sentence)
### Consensus Fragility — top risk that breaks the bull case (1 sentence, cited)
### What to Watch — 2 specific metrics next quarter (cited)

Keep under 150 words total. Cite every claim with [doc_name p.N]. Do NOT use [Deterministic Data] as a citation.
"""


# ── Task runner ───────────────────────────────────────────────────────────────

def run_analysis_task(
    task_name:    str,
    prompt:       str,
    max_retries:  int = 2,
    max_conn_retries: int = 5,
) -> dict:
    """
    Send prompt to DeepSeek, return structured result with citation list.
    - Retries up to max_retries times if placeholder citations are detected.
    - Retries up to max_conn_retries times on APIConnectionError (transient network drops).
    """
    from openai import APIConnectionError as _APIConnectionError

    text = ""
    for attempt in range(1, max_retries + 2):
        print(f"\n  Calling DeepSeek for task: {task_name} (attempt {attempt}) ...")
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


# ── Main orchestrator ─────────────────────────────────────────────────────────

def run_full_analysis(
    ticker: str,
    base_dir: Path,
    use_neo4j: bool = False,
    transcript_pages: list | None = None,
    broker_pages: list | None = None,
    latest_name: str = "",
    previous_name: str = "",
) -> dict:
    """
    Run all 3 analysis tasks for a ticker (transcript comparison, Q&A, broker consensus).
    Returns dict with keys: ticker, tasks, features, broker_labels, broker_parsed.

    Parameters
    ----------
    ticker:
        Uppercase ticker symbol.
    base_dir:
        Root path to local PDF data (used in PDF mode only).
    use_neo4j:
        If True, uses the pre-loaded *transcript_pages* and *broker_pages* from
        Neo4j instead of loading from local PDFs.  The caller (agent.py) must
        supply transcript_pages, broker_pages, latest_name, previous_name.
    transcript_pages, broker_pages:
        Document lists from agent_step1_neo4j.load_neo4j_pages.
        Only used when use_neo4j=True.
    latest_name, previous_name:
        Human-readable source names for the two transcripts.
        Only used when use_neo4j=True.
    """
    print(f"\n{'='*60}")
    print(f"  Full analysis: {ticker}  [mode={'neo4j' if use_neo4j else 'pdf'}]")
    print(f"{'='*60}")
    t0 = time.time()

    if use_neo4j and transcript_pages is not None and broker_pages is not None:
        # ── Neo4j mode ────────────────────────────────────────────
        broker_pages_raw  = broker_pages
        _latest_name      = latest_name or "latest_transcript"
        _previous_name    = previous_name or "previous_transcript"

        # Quality + tagging (works the same on Document objects from Neo4j)
        flag_quality_issues(transcript_pages)
        flag_quality_issues(broker_pages_raw)
        tag_transcript_sections(transcript_pages)
        tag_broker_sections(broker_pages_raw)

        usable_t = filter_usable(transcript_pages)
        usable_b = filter_usable(broker_pages_raw)
        print(f"  [timing] neo4j_load+quality_tag: {time.time()-t0:.1f}s")

    else:
        # ── PDF mode (original behaviour) ─────────────────────────
        broker_pdfs, transcript_pdfs = list_stock_files(base_dir, ticker)
        latest_t   = transcript_pdfs[0]
        previous_t = transcript_pdfs[1]
        _latest_name   = latest_t.name
        _previous_name = previous_t.name

        _transcript_pages  = []
        _transcript_pages += load_pdf_pages([latest_t],   ticker, "transcript", "latest")
        _transcript_pages += load_pdf_pages([previous_t], ticker, "transcript", "previous")
        broker_pages_raw    = load_pdf_pages(broker_pdfs,  ticker, "broker")
        print(f"  [timing] load_pdfs: {time.time()-t0:.1f}s")

        t1 = time.time()
        flag_quality_issues(_transcript_pages)
        flag_quality_issues(broker_pages_raw)
        tag_transcript_sections(_transcript_pages)
        tag_broker_sections(broker_pages_raw)

        usable_t = filter_usable(_transcript_pages)
        usable_b = filter_usable(broker_pages_raw)
        transcript_pages = _transcript_pages
        print(f"  [timing] quality_tag: {time.time()-t1:.1f}s")

    all_pages = usable_t + usable_b

    # ── Features ──────────────────────────────────────────────────
    t2 = time.time()
    latest_pages   = [p for p in usable_t if p.metadata.get("period") == "latest"]
    previous_pages = [p for p in usable_t if p.metadata.get("period") == "previous"]

    # In Neo4j mode period may be empty string — fall back to first/second half
    if not latest_pages and not previous_pages and usable_t:
        mid = len(usable_t) // 2
        latest_pages   = usable_t[:mid] or usable_t
        previous_pages = usable_t[mid:] or usable_t[:1]

    latest_feat  = extract_transcript_features(latest_pages,   _latest_name)
    prev_feat    = extract_transcript_features(previous_pages, _previous_name)
    kpi_diff     = compare_kpi_coverage(latest_feat, prev_feat)

    # Broker labels use all non-blank/garbage pages for max keyword coverage
    all_broker_for_labels = [
        p for p in broker_pages_raw
        if not p.metadata.get("is_blank") and not p.metadata.get("is_garbage")
    ]
    broker_labels  = extract_all_broker_labels(all_broker_for_labels)
    broker_parsed  = parse_all_broker_docs(usable_b, broker_labels)
    print(f"  [timing] features_parse: {time.time()-t2:.1f}s")

    # ── Embeddings ────────────────────────────────────────────────
    t3 = time.time()
    print("\n  Building embedding index ...")
    if use_neo4j:
        evidence = build_index_neo4j(ticker, transcript_pages, broker_pages_raw)
    else:
        chunks   = chunk_documents(all_pages)
        evidence = build_index(chunks)
    print(f"  [timing] chunk+index(embed): {time.time()-t3:.1f}s")

    # ── Task A: Transcript comparison ─────────────────────────────
    q_transcript = "KPI changes revenue earnings guidance tone shift management confidence"
    ev_transcript = format_evidence_pack(
        retrieve(evidence, q_transcript, top_k=TOP_K, filter_doc_type="transcript")
    )
    prompt_a = _prompt_transcript_comparison(ticker, latest_feat, prev_feat, kpi_diff, ev_transcript)

    # ── Task B: Q&A behavior ──────────────────────────────────────
    q_qa = "Q&A analyst questions evasive response hedge uncertainty management deflection"
    ev_qa = format_evidence_pack(
        retrieve(evidence, q_qa, top_k=TOP_K, filter_section="qa")
    )
    prompt_b = _prompt_qa_behavior(ticker, latest_feat, ev_qa)

    # Run Task A and Task B concurrently (they are independent)
    print("\n  Running Task A (transcript) and Task B (Q&A) concurrently ...")
    t4 = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(run_analysis_task, "transcript_comparison", prompt_a)
        future_b = pool.submit(run_analysis_task, "qa_behavior",           prompt_b)
        task_a = future_a.result()
        task_b = future_b.result()
    print(f"  [timing] Task A+B concurrent: {time.time()-t4:.1f}s")

    # ── Task C: Broker consensus (per-doc retrieval) ───────────────
    t5 = time.time()
    broker_ev_chunks = retrieve_broker_evidence(evidence)
    ev_broker        = format_evidence_pack(broker_ev_chunks)
    broker_summary   = format_broker_summary_for_llm(broker_parsed)
    prompt_c = _prompt_broker_consensus(ticker, broker_parsed, ev_broker, broker_summary)
    task_c   = run_analysis_task("broker_consensus", prompt_c)
    print(f"  [timing] Task C (broker): {time.time()-t5:.1f}s")

    print(f"  [timing] TOTAL for {ticker}: {time.time()-t0:.1f}s")

    return {
        "ticker":              ticker,
        "latest_transcript":   _latest_name,
        "previous_transcript": _previous_name,
        "features": {
            "latest":   latest_feat,
            "previous": prev_feat,
            "kpi_diff": kpi_diff,
        },
        "broker_labels":  broker_labels,
        "broker_parsed":  broker_parsed,
        "tasks": [task_a, task_b, task_c],
    }


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_full_analysis(TICKER, BASE_DIR)

    print(f"\n\n{'='*60}")
    print(f"  ANALYSIS COMPLETE: {result['ticker']}")
    print(f"{'='*60}")
    for task in result["tasks"]:
        print(f"\n### {task['task'].upper()} ###")
        print(task["analysis"])
        print(f"\n  [{len(task['citations_found'])} citations]")
