"""Shared test metrics helpers for prompt and integration observability.

This module records per-test telemetry to JSONL and can generate a Markdown
summary report with quantified metrics.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_LOG_PATH = OUTPUT_DIR / "test_metrics.jsonl"
SUMMARY_MD_PATH = OUTPUT_DIR / "test_metrics_summary.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sizeof(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=True, default=str).encode("utf-8"))
    except Exception:
        return 0


def sizeof(value: Any) -> int:
    """Public size helper used by tests."""
    return _sizeof(value)


def _shape(value: Any) -> Dict[str, Any]:
    t = type(value).__name__
    if isinstance(value, dict):
        return {"type": t, "keys": len(value), "size_bytes": _sizeof(value)}
    if isinstance(value, list):
        sample_type = type(value[0]).__name__ if value else "empty"
        return {
            "type": t,
            "items": len(value),
            "item_type": sample_type,
            "size_bytes": _sizeof(value),
        }
    if isinstance(value, (str, bytes, bytearray)):
        return {"type": t, "length": len(value), "size_bytes": _sizeof(value)}
    return {"type": t, "size_bytes": _sizeof(value)}


def utc_epoch_ms() -> int:
    return int(time.time() * 1000)


def measure_latency_ms(fn, *args, **kwargs) -> tuple[float, Any]:
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, out


def record_metric(entry: Dict[str, Any]) -> None:
    payload = {
        "timestamp_utc": _now_iso(),
        **entry,
    }
    with METRICS_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")


def clear_metrics_log() -> None:
    if METRICS_LOG_PATH.exists():
        METRICS_LOG_PATH.unlink()


def output_snapshot(outputs: Dict[str, Any]) -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    for key, value in outputs.items():
        snap[key] = _shape(value)
    return snap


def db_connection_metrics(label: str, connect_fn) -> Dict[str, Any]:
    """Measure connection time/latency for a DB connector factory."""
    start = time.perf_counter()
    conn = connect_fn()
    connect_ms = (time.perf_counter() - start) * 1000.0
    return {
        "label": label,
        "connect_ms": round(connect_ms, 3),
        "connected": conn is not None,
    }


def summarize_metrics() -> Dict[str, Any]:
    if not METRICS_LOG_PATH.exists():
        return {
            "records": 0,
            "by_category": {},
            "total_duration_ms": 0.0,
            "avg_duration_ms": 0.0,
            "total_output_bytes": 0,
        }

    rows: List[Dict[str, Any]] = []
    with METRICS_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    by_category: Dict[str, int] = {}
    total_duration_ms = 0.0
    total_output_bytes = 0
    latency_values: List[float] = []
    connection_values: List[float] = []
    data_amount_total = 0
    data_type_counts: Dict[str, int] = {}

    for r in rows:
        cat = str(r.get("category", "uncategorized"))
        by_category[cat] = by_category.get(cat, 0) + 1

        dur = r.get("duration_ms", 0)
        if isinstance(dur, (int, float)):
            total_duration_ms += float(dur)

        out_bytes = r.get("output_size_bytes", 0)
        if isinstance(out_bytes, (int, float)):
            total_output_bytes += int(out_bytes)

        metrics = r.get("metrics")
        if isinstance(metrics, dict):
            lat = metrics.get("latency_ms")
            if isinstance(lat, (int, float)):
                latency_values.append(float(lat))

            conn_ms = metrics.get("connection_time_ms")
            if isinstance(conn_ms, (int, float)):
                connection_values.append(float(conn_ms))

            amount = metrics.get("data_amount")
            if isinstance(amount, (int, float)):
                data_amount_total += int(amount)

            dtype = metrics.get("data_type")
            if isinstance(dtype, str) and dtype:
                data_type_counts[dtype] = data_type_counts.get(dtype, 0) + 1

    avg_duration_ms = (total_duration_ms / len(rows)) if rows else 0.0

    return {
        "records": len(rows),
        "by_category": by_category,
        "total_duration_ms": round(total_duration_ms, 3),
        "avg_duration_ms": round(avg_duration_ms, 3),
        "total_output_bytes": total_output_bytes,
        "latency_avg_ms": round(sum(latency_values) / len(latency_values), 3) if latency_values else 0.0,
        "latency_max_ms": round(max(latency_values), 3) if latency_values else 0.0,
        "connection_avg_ms": round(sum(connection_values) / len(connection_values), 3) if connection_values else 0.0,
        "connection_max_ms": round(max(connection_values), 3) if connection_values else 0.0,
        "data_amount_total": data_amount_total,
        "data_type_counts": data_type_counts,
    }


def write_metrics_summary_md(extra_sections: Optional[Iterable[str]] = None) -> Path:
    s = summarize_metrics()
    lines = [
        "# Test Metrics Summary",
        "",
        "## Overview",
        f"- Records: {s['records']}",
        f"- Total duration: {s['total_duration_ms']} ms",
        f"- Average duration: {s['avg_duration_ms']} ms",
        f"- Total output size: {s['total_output_bytes']} bytes",
        f"- Avg latency: {s['latency_avg_ms']} ms",
        f"- Max latency: {s['latency_max_ms']} ms",
        f"- Avg connection time: {s['connection_avg_ms']} ms",
        f"- Max connection time: {s['connection_max_ms']} ms",
        f"- Total data amount (reported units): {s['data_amount_total']}",
        "",
        "## Records By Category",
    ]

    by_cat = s.get("by_category", {})
    if by_cat:
        for cat, count in sorted(by_cat.items(), key=lambda x: x[0]):
            lines.append(f"- {cat}: {count}")
    else:
        lines.append("- No records")

    lines.append("")
    lines.append("## Data Types Reported")
    dtype_counts = s.get("data_type_counts", {})
    if dtype_counts:
        for dtype, count in sorted(dtype_counts.items(), key=lambda x: x[0]):
            lines.append(f"- {dtype}: {count}")
    else:
        lines.append("- No data_type metrics reported")

    if extra_sections:
        lines.append("")
        lines.append("## Additional Details")
        for section in extra_sections:
            lines.append(f"- {section}")

    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return SUMMARY_MD_PATH
