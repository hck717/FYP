"""
Step 8: Save analysis outputs.

Takes the result dict from Step 7 (run_full_analysis) and writes:
  - output/<ticker>/report.md    — human-readable Markdown report
  - output/<ticker>/analysis.json — structured JSON with all features + citations

Report structure:
  1. Executive View          — one-screen key signals (<20 sec read)
  2. Deterministic Feature Summary table
  3. KPI Coverage Changes
  4. Broker Snapshot Table   — deterministic, one row per broker
  5. Broker Consensus Commentary — short, cited, LLM (Task C)
  6. Transcript Comparison   — LLM, cited (Task A)
  7. Q&A Behavior Analysis   — LLM, cited (Task B)
  8. Citations Index

Run (single ticker):
    python agent_step8_save_outputs.py

Run (all 5 tickers):
    Set TICKERS list below and run.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from agent_step4b_broker_parse import format_broker_table_md
from agent_step7_synthesis import run_full_analysis

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS  = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"]
BASE_DIR = Path("data_reports")
OUT_DIR  = Path("output")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _feat_table_row(feat: dict, label: str) -> str:
    return (
        f"| {label} | {feat['word_count']:,} | {feat['kpi_per_1k_words']} | "
        f"{feat['hedge_ratio']} | {feat['prep_hedge_ratio']} | "
        f"{feat['qa_hedge_ratio']} | {feat['qa_vs_prep_hedge_delta']:+.4f} | "
        f"{feat['pivot_per_1k_words']} | {feat['evasive_count']} | "
        f"{feat['guidance_range_count']} |"
    )


def _signal_arrow(val: float, better_is_lower: bool = False) -> str:
    """Simple directional indicator for exec view."""
    if better_is_lower:
        return "↓" if val < 0 else "↑" if val > 0 else "→"
    return "↑" if val > 0 else "↓" if val < 0 else "→"


def _build_exec_view(result: dict) -> str:
    """
    One-screen executive summary. Key signals only. Readable in <20 seconds.
    """
    ticker   = result["ticker"]
    lf       = result["features"]["latest"]
    pf       = result["features"]["previous"]
    kd       = result["features"]["kpi_diff"]
    bp       = result["broker_parsed"]
    labels   = result["broker_labels"]

    dist     = Counter(l["rating"] for l in labels)
    bull     = dist.get("bullish", 0)
    neut     = dist.get("neutral", 0)
    bear     = dist.get("bearish", 0)

    hedge_delta   = lf["hedge_ratio"] - pf["hedge_ratio"]
    kpi_delta     = lf["kpi_per_1k_words"] - pf["kpi_per_1k_words"]
    evasive_note  = f"{lf['evasive_count']} evasive phrases in Q&A" if lf["evasive_count"] else "No evasive Q&A phrases"
    qa_hedge_flag = " ⚠ Q&A more hedged than prepared remarks" if lf["qa_vs_prep_hedge_delta"] > 0.02 else ""

    # Price targets
    pts = [b["price_target"] for b in bp if b["price_target"]]
    pt_range = f"${min(pts):.0f}–${max(pts):.0f}" if len(pts) >= 2 else (f"${pts[0]:.0f}" if pts else "N/A")
    upsides  = [b["upside_pct"] for b in bp if b["upside_pct"] is not None]
    avg_upside = f"{sum(upsides)/len(upsides):+.1%}" if upsides else "N/A"

    dropped_str = ", ".join(f"`{k}`" for k in kd["dropped_kpis"]) if kd["dropped_kpis"] else "none"
    added_str   = ", ".join(f"`{k}`" for k in kd["added_kpis"])   if kd["added_kpis"]   else "none"

    lines = [
        f"## Executive View — {ticker}\n",
        f"> Generated {datetime.now().strftime('%Y-%m-%d')}  |  "
        f"Latest: `{result['latest_transcript']}`\n\n",
        "| Signal | Latest | vs Previous | Flag |\n",
        "|--------|--------|-------------|------|\n",
        f"| Hedge ratio (full doc) | {lf['hedge_ratio']} | {hedge_delta:+.4f} "
        f"| {'⚠ more hedged' if hedge_delta > 0.02 else 'stable'} |\n",
        f"| KPI density (/1k words) | {lf['kpi_per_1k_words']} | {kpi_delta:+.2f} "
        f"| {'↑ more specific' if kpi_delta > 0 else '↓ less specific'} |\n",
        f"| Q&A hedge vs Prep delta | {lf['qa_vs_prep_hedge_delta']:+.4f} | — "
        f"| {'⚠ evasive in Q&A' if lf['qa_vs_prep_hedge_delta'] > 0.02 else 'consistent'} |\n",
        f"| Evasive Q&A phrases | {lf['evasive_count']} | — "
        f"| {'⚠ flag' if lf['evasive_count'] >= 3 else 'ok'} |\n",
        f"| Guidance ranges mentioned | {lf['guidance_range_count']} | — | — |\n",
        f"| Dropped KPIs | {dropped_str} | — | {'⚠' if kd['dropped_kpis'] else '✓'} |\n",
        f"| Added KPIs | {added_str} | — | — |\n",
        f"| Broker consensus | {bull}B / {neut}N / {bear}S | — | "
        f"{'⚠ mixed' if bear > 0 else 'bullish lean' if bull > neut else 'neutral'} |\n",
        f"| PT range (broker) | {pt_range} | — | avg upside {avg_upside} |\n",
        "\n",
    ]
    return "".join(lines)


# ── Markdown builder ──────────────────────────────────────────────────────────

def build_markdown_report(result: dict) -> str:
    ticker   = result["ticker"]
    date     = datetime.now().strftime("%Y-%m-%d")
    feats    = result["features"]
    lf       = feats["latest"]
    pf       = feats["previous"]
    kpi_diff = feats["kpi_diff"]
    labels   = result["broker_labels"]
    bp       = result["broker_parsed"]

    sections = []

    # ── Header ─────────────────────────────────────────────────────
    sections.append(f"# {ticker} Investment Analysis Report\n\n")
    sections.append(f"**Generated**: {date}  \n")
    sections.append(f"**Latest transcript**: {result['latest_transcript']}  \n")
    sections.append(f"**Previous transcript**: {result['previous_transcript']}  \n")
    sections.append(f"**Broker reports**: {len(labels)} documents  \n\n")
    sections.append("---\n\n")

    # ── 1. Executive View ──────────────────────────────────────────
    sections.append(_build_exec_view(result))
    sections.append("---\n\n")

    # ── 2. Deterministic Feature Summary ──────────────────────────
    sections.append("## Deterministic Feature Summary\n\n")
    sections.append(
        "| Transcript | Words | KPI/1k | Hedge(all) | Hedge(prep) | Hedge(QA) | QA-Prep Δ | Pivots/1k | Evasive | Guidance# |\n"
        "|------------|------:|-------:|-----------:|------------:|----------:|----------:|----------:|--------:|----------:|\n"
    )
    sections.append(_feat_table_row(lf, "Latest")   + "\n")
    sections.append(_feat_table_row(pf, "Previous") + "\n")
    sections.append("\n")

    if lf["guidance_ranges"]:
        sections.append("### Latest Transcript Guidance Ranges\n\n")
        for g in lf["guidance_ranges"]:
            sections.append(f"- {g}\n")
        sections.append("\n")

    # ── 3. KPI Coverage Changes ────────────────────────────────────
    sections.append("## KPI Coverage Changes\n\n")
    if kpi_diff["dropped_kpis"]:
        sections.append("**Dropped KPIs** (in previous, not in latest):  \n")
        sections.append(", ".join(f"`{k}`" for k in kpi_diff["dropped_kpis"]) + "\n\n")
    if kpi_diff["added_kpis"]:
        sections.append("**Added KPIs** (in latest, not in previous):  \n")
        sections.append(", ".join(f"`{k}`" for k in kpi_diff["added_kpis"]) + "\n\n")
    if not kpi_diff["dropped_kpis"] and not kpi_diff["added_kpis"]:
        sections.append("No KPI coverage changes detected.\n\n")

    sections.append("---\n\n")

    # ── 4. Broker Snapshot Table ───────────────────────────────────
    sections.append("## Broker Snapshot\n\n")
    sections.append(format_broker_table_md(bp))
    sections.append("\n\n")
    dist = Counter(l["rating"] for l in labels)
    sections.append(f"**Rating distribution**: {dict(dist)}\n\n")
    sections.append("---\n\n")

    # ── 5–7. LLM analysis sections ─────────────────────────────────
    task_titles = {
        "broker_consensus":      "Broker Consensus Commentary",
        "transcript_comparison": "Transcript Comparison (Latest vs Previous)",
        "qa_behavior":           "Q&A Behavior Analysis",
    }
    # Render in desired order
    task_order = ["broker_consensus", "transcript_comparison", "qa_behavior"]
    task_map   = {t["task"]: t for t in result["tasks"]}

    for task_key in task_order:
        task = task_map.get(task_key)
        if not task:
            continue
        title = task_titles.get(task_key, task_key)
        sections.append(f"## {title}\n\n")
        sections.append(task["analysis"].strip())
        sections.append("\n\n---\n\n")

    # ── 8. Citations Index ─────────────────────────────────────────
    sections.append("## Citations Index\n\n")
    all_citations: list[dict] = []
    for task in result["tasks"]:
        for c in task["citations_found"]:
            all_citations.append({"task": task["task"], **c})

    if all_citations:
        by_doc: dict[str, list[int]] = {}
        for c in all_citations:
            by_doc.setdefault(c["doc_name"], []).append(c["page"])
        for doc, pages in sorted(by_doc.items()):
            unique_pages = sorted(set(pages))
            sections.append(f"- **{doc}**: pages {unique_pages}\n")
    else:
        sections.append("_No citations extracted._\n")

    return "".join(sections)


# ── JSON builder ──────────────────────────────────────────────────────────────

def build_json_output(result: dict) -> dict:
    return {
        "ticker":               result["ticker"],
        "generated_at":         datetime.now().isoformat(),
        "latest_transcript":    result["latest_transcript"],
        "previous_transcript":  result["previous_transcript"],
        "features": {
            "latest":   result["features"]["latest"],
            "previous": result["features"]["previous"],
            "kpi_diff": result["features"]["kpi_diff"],
        },
        "broker_labels":  result["broker_labels"],
        "broker_parsed":  result["broker_parsed"],
        "analysis_tasks": [
            {
                "task":            t["task"],
                "analysis":        t["analysis"],
                "citations_found": t["citations_found"],
            }
            for t in result["tasks"]
        ],
    }


# ── Save ──────────────────────────────────────────────────────────────────────

def save_outputs(result: dict, out_dir: Path):
    ticker_dir = out_dir / result["ticker"]
    ticker_dir.mkdir(parents=True, exist_ok=True)

    md_path = ticker_dir / "report.md"
    md_path.write_text(build_markdown_report(result), encoding="utf-8")
    print(f"  Saved: {md_path}")

    json_path = ticker_dir / "analysis.json"
    json_path.write_text(
        json.dumps(build_json_output(result), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"  Saved: {json_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

import time as _time
import sys as _sys

if __name__ == "__main__":
    # Allow overriding tickers from command line: python agent_step8_save_outputs.py AAPL MSFT
    run_tickers = _sys.argv[1:] if len(_sys.argv) > 1 else TICKERS

    # Try Neo4j mode first; fall back to PDF if not available.
    # Import Neo4j loader lazily so the script still works without neo4j package.
    def _run_ticker(ticker: str) -> dict:
        try:
            from agent_step1_neo4j import load_neo4j_pages  # type: ignore[import]
            t_pages, b_pages, latest_n, prev_n = load_neo4j_pages(ticker)
            print(f"  [mode=neo4j] {len(t_pages)} transcript docs, {len(b_pages)} broker docs")
            return run_full_analysis(
                ticker, BASE_DIR,
                use_neo4j=True,
                transcript_pages=t_pages,
                broker_pages=b_pages,
                latest_name=latest_n,
                previous_name=prev_n,
            )
        except Exception as neo4j_exc:
            print(f"  [neo4j unavailable: {neo4j_exc}] — falling back to PDF mode")
            return run_full_analysis(ticker, BASE_DIR, use_neo4j=False)

    wall_start = _time.time()
    for ticker in run_tickers:
        print(f"\n{'='*60}")
        print(f"  Running full pipeline for: {ticker}")
        print(f"{'='*60}")
        try:
            result = _run_ticker(ticker)
            save_outputs(result, OUT_DIR)
            print(f"\n  [DONE] {ticker} — outputs saved to {OUT_DIR / ticker}/")
        except Exception as e:
            print(f"\n  [ERROR] {ticker} failed: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  All tickers complete in {_time.time()-wall_start:.1f}s. Check output/ directory.")
    print(f"{'='*60}")
