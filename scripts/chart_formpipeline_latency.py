#!/usr/bin/env python3
from __future__ import annotations

"""
Parse FormPipeline latency logs and print a compact ASCII chart.

Supports two formats emitted by src/programs/form_pipeline/orchestrator.py:
1) Debug line (AI_FORM_DEBUG=true):
   [FormPipeline] requestId=... latencyMs=... steps=... plannerLatencyMs=... rendererLatencyMs=...
2) JSON line (AI_FORM_LOG_LATENCY=true):
   {"event":"step3_latency","totalMs":...,"plannerMs":...,"rendererMs":...,"emittedSteps":...,...}

Examples:
  # From a file
  python scripts/chart_formpipeline_latency.py logs.txt

  # From stdin
  tail -n 500 service.log | python scripts/chart_formpipeline_latency.py -
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Iterator, Optional


@dataclass(frozen=True)
class LatencyRow:
    request_id: str
    steps: int
    total_ms: int
    planner_ms: int
    renderer_ms: int
    post_ms: int
    planner_cache_hit: Optional[bool] = None
    render_cache_hit: Optional[bool] = None


_KV_RE = re.compile(r"(?P<k>[A-Za-z][A-Za-z0-9_]*)=(?P<v>\S+)")


def _as_int(x: object, default: int = 0) -> int:
    try:
        return int(float(str(x)))
    except Exception:
        return default


def _as_bool(x: object) -> Optional[bool]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def _parse_json_latency(line: str) -> Optional[LatencyRow]:
    s = line.strip()
    if not s.startswith("{"):
        return None
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("event") != "step3_latency":
        return None

    steps = _as_int(obj.get("emittedSteps"), default=-1)
    if steps < 0:
        steps = _as_int(obj.get("plannedItems"), default=0)

    return LatencyRow(
        request_id=str(obj.get("requestId") or ""),
        steps=steps,
        total_ms=_as_int(obj.get("totalMs")),
        planner_ms=_as_int(obj.get("plannerMs")),
        renderer_ms=_as_int(obj.get("rendererMs")),
        post_ms=_as_int(obj.get("postProcessingMs")),
        planner_cache_hit=_as_bool(obj.get("plannerCacheHit")),
        render_cache_hit=_as_bool(obj.get("renderCacheHit")),
    )


def _parse_debug_latency(line: str) -> Optional[LatencyRow]:
    if "[FormPipeline]" not in line or "latencyMs=" not in line:
        return None
    kv = {m.group("k"): m.group("v") for m in _KV_RE.finditer(line)}
    if not kv:
        return None
    if "latencyMs" not in kv:
        return None

    return LatencyRow(
        request_id=str(kv.get("requestId") or ""),
        steps=_as_int(kv.get("steps")),
        total_ms=_as_int(kv.get("latencyMs")),
        planner_ms=_as_int(kv.get("plannerLatencyMs")),
        renderer_ms=_as_int(kv.get("rendererLatencyMs")),
        post_ms=_as_int(kv.get("postLatencyMs")),
        planner_cache_hit=_as_bool(kv.get("plannerCacheHit")),
        render_cache_hit=_as_bool(kv.get("renderCacheHit")),
    )


def parse_rows(lines: Iterable[str]) -> Iterator[LatencyRow]:
    for line in lines:
        row = _parse_json_latency(line) or _parse_debug_latency(line)
        if not row:
            continue
        if row.steps <= 0 or row.total_ms <= 0:
            continue
        yield row


def _bar(*, planner_ms: float, renderer_ms: float, max_total_ms: float, width: int) -> str:
    total = max(planner_ms + renderer_ms, 0.0)
    if max_total_ms <= 0:
        return ""
    scale = float(width) / float(max_total_ms)
    planner_len = int(round(planner_ms * scale))
    renderer_len = int(round(renderer_ms * scale))
    # Avoid bars that exceed width due to rounding.
    while planner_len + renderer_len > width:
        if renderer_len >= planner_len and renderer_len > 0:
            renderer_len -= 1
        elif planner_len > 0:
            planner_len -= 1
        else:
            break
    pad = max(0, width - (planner_len + renderer_len))
    return f"|{'#' * planner_len}{'=' * renderer_len}{' ' * pad}|"


def _pct(part: float, whole: float) -> str:
    if whole <= 0:
        return "0%"
    return f"{int(round(100.0 * part / whole))}%"


def _read_lines_from_path(path: str) -> Iterable[str]:
    if path == "-":
        return sys.stdin
    return open(path, "r", encoding="utf-8", errors="replace")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="ASCII latency chart for FormPipeline planner vs renderer.")
    ap.add_argument("paths", nargs="*", default=["-"], help="Log file path(s) or '-' for stdin.")
    ap.add_argument("--width", type=int, default=50, help="Bar width in characters.")
    ap.add_argument(
        "--include-cache-buckets",
        action="store_true",
        help="Also split groups by cache-hit status (cold/warm).",
    )
    args = ap.parse_args(argv)

    rows: list[LatencyRow] = []
    for p in args.paths:
        for row in parse_rows(_read_lines_from_path(p)):
            rows.append(row)

    if not rows:
        print("No latency rows found. Provide lines with [FormPipeline] latencyMs=... or JSON event=step3_latency.")
        return 2

    def group_key(r: LatencyRow) -> tuple:
        if not args.include_cache_buckets:
            return (r.steps,)
        planner_bucket = "warm" if r.planner_cache_hit else "cold"
        render_bucket = "warm" if r.render_cache_hit else "cold"
        return (r.steps, planner_bucket, render_bucket)

    groups: dict[tuple, list[LatencyRow]] = {}
    for r in rows:
        groups.setdefault(group_key(r), []).append(r)

    # Compute max total (avg) to normalize bars.
    avg_totals = []
    for k, rs in groups.items():
        avg_totals.append(mean([x.total_ms for x in rs]))
    max_total = max(avg_totals) if avg_totals else 0.0

    print("Legend: '#'=planner, '='=renderer (scaled to max avg total in output)")
    print()

    # Sort groups: primarily by step count, then buckets.
    for k in sorted(groups.keys()):
        rs = groups[k]
        avg_total = mean([x.total_ms for x in rs])
        avg_planner = mean([x.planner_ms for x in rs])
        avg_renderer = mean([x.renderer_ms for x in rs])
        avg_post = mean([x.post_ms for x in rs])
        n = len(rs)

        key_label = f"steps={k[0]}"
        if len(k) > 1:
            key_label += f" planner={k[1]} render={k[2]}"

        bar = _bar(planner_ms=avg_planner, renderer_ms=avg_renderer, max_total_ms=max_total, width=args.width)
        print(
            f"{key_label:<30} n={n:<3} total={int(round(avg_total))}ms "
            f"P={int(round(avg_planner))}ms({_pct(avg_planner, avg_total)}) "
            f"R={int(round(avg_renderer))}ms({_pct(avg_renderer, avg_total)}) "
            f"post={int(round(avg_post))}ms {bar}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
