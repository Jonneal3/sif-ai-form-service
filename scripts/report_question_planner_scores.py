#!/usr/bin/env python3
from __future__ import annotations

"""
Generate a compact quality report for question-planner outputs.

Inputs supported:
- Demo pack JSONL (each line: {"inputs":{...},"outputs":{...}})
- Any JSONL where each line includes planner_context_json + question_plan_json (or context/plan aliases)

Examples:
  # Report on an optimized demo pack from a specific run
  PYTHONPATH=.:src python3 scripts/report_question_planner_scores.py \
    src/programs/question_planner/data/optimized_outputs/runs/<RUN_ID>/question_planner_demo_pack.jsonl

  # Report on a JSONL log stream
  cat planner_logs.jsonl | PYTHONPATH=.:src python3 scripts/report_question_planner_scores.py --stdin
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from metrics.question_planner.plan_quality import score_question_plan, score_question_plan_optimizer


def _try_json_loads(s: Any) -> Any:
    try:
        return json.loads(str(s))
    except Exception:
        return None


def _compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _as_json_str(x: Any) -> str:
    if isinstance(x, (dict, list)):
        return _compact(x)
    return str(x or "").strip()


def _extract_fields(obj: Dict[str, Any]) -> Tuple[str, str]:
    """
    Return (planner_context_json, question_plan_json) as JSON strings.
    Accepts:
      - demo pack shape: {"inputs":{...},"outputs":{...}}
      - flat shape: {"planner_context_json": "...", "question_plan_json": "..."}
      - aliases: context/plannerContextJson and plan/questionPlanJson
    """
    inputs = obj.get("inputs") if isinstance(obj.get("inputs"), dict) else None
    outputs = obj.get("outputs") if isinstance(obj.get("outputs"), dict) else None

    if inputs is not None and outputs is not None:
        ctx = (
            inputs.get("planner_context_json")
            or inputs.get("context")
            or inputs.get("plannerContextJson")
            or inputs.get("planner_context")
            or ""
        )
        plan = (
            outputs.get("question_plan_json")
            or outputs.get("plan")
            or outputs.get("questionPlanJson")
            or outputs.get("question_plan")
            or ""
        )
        return (_as_json_str(ctx), _as_json_str(plan))

    ctx = obj.get("planner_context_json") or obj.get("context") or obj.get("plannerContextJson") or ""
    plan = obj.get("question_plan_json") or obj.get("plan") or obj.get("questionPlanJson") or ""
    return (_as_json_str(ctx), _as_json_str(plan))


def _service_summary(planner_context_json: str) -> str:
    parsed = _try_json_loads(planner_context_json)
    if isinstance(parsed, dict):
        s = str(parsed.get("services_summary") or parsed.get("grounding_summary") or "").strip()
        if s:
            return s
    return ""


def _read_jsonl_lines(lines: Iterable[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in lines:
        t = str(line or "").strip()
        if not t:
            continue
        obj = _try_json_loads(t)
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _render_table(rows: List[Tuple[str, float]]) -> str:
    if not rows:
        return ""
    w = max(len(k) for k, _ in rows)
    lines = [f"{'metric'.ljust(w)}  avg"]
    lines.append(f"{'-'*w}  ---")
    for k, v in rows:
        lines.append(f"{k.ljust(w)}  {v:6.2f}")
    return "\n".join(lines) + "\n"


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="", help="JSONL file to report on.")
    ap.add_argument("--stdin", action="store_true", help="Read JSONL from stdin.")
    ap.add_argument("--out", default="", help="Write the report to a file (otherwise stdout).")
    ap.add_argument("--worst", type=int, default=5, help="Show the worst N examples.")
    args = ap.parse_args(argv)

    if not args.stdin and not str(args.path or "").strip():
        raise SystemExit("Provide a JSONL path or pass --stdin.")

    if args.stdin:
        records = _read_jsonl_lines(sys.stdin)
    else:
        p = Path(str(args.path))
        if not p.exists():
            raise SystemExit(f"Not found: {p}")
        records = _read_jsonl_lines(p.read_text(encoding="utf-8").splitlines())

    scored: List[Dict[str, Any]] = []
    for rec in records:
        ctx_json, plan_json = _extract_fields(rec)
        if not ctx_json.strip() or not plan_json.strip():
            continue
        res = score_question_plan(planner_context_json=ctx_json, question_plan_json=plan_json)
        opt = score_question_plan_optimizer(planner_context_json=ctx_json, question_plan_json=plan_json)
        scored.append(
            {
                "score": float(res.score),
                "optimizer_score": float(opt),
                "groups": {k: float(v) for k, v in (res.groups or {}).items()},
                "breakdown": {k: float(v) for k, v in (res.breakdown or {}).items()},
                "notes": list(res.notes or []),
                "services_summary": _service_summary(ctx_json),
            }
        )

    if not scored:
        raise SystemExit("No usable records found (missing planner_context_json/question_plan_json).")

    scores = [x["score"] for x in scored]
    scores_sorted = sorted(scores)
    mean = statistics.fmean(scores)
    median = statistics.median(scores_sorted)
    p10 = scores_sorted[max(0, int(0.10 * (len(scores_sorted) - 1)))]
    p90 = scores_sorted[max(0, int(0.90 * (len(scores_sorted) - 1)))]

    optimizer_scores = [float(x.get("optimizer_score", 0.0)) for x in scored]
    optimizer_scores_sorted = sorted(optimizer_scores)
    opt_mean = statistics.fmean(optimizer_scores)
    opt_median = statistics.median(optimizer_scores_sorted)
    opt_p10 = optimizer_scores_sorted[max(0, int(0.10 * (len(optimizer_scores_sorted) - 1)))]
    opt_p90 = optimizer_scores_sorted[max(0, int(0.90 * (len(optimizer_scores_sorted) - 1)))]

    # Aggregate group/breakdown averages.
    group_keys = sorted({k for x in scored for k in (x.get("groups") or {}).keys()})
    breakdown_keys = sorted({k for x in scored for k in (x.get("breakdown") or {}).keys()})

    def avg_map(items: List[Dict[str, Any]], keys: List[str], field: str) -> List[Tuple[str, float]]:
        rows: List[Tuple[str, float]] = []
        for k in keys:
            vals = [float(x.get(field, {}).get(k, 0.0)) for x in items]
            rows.append((k, statistics.fmean(vals)))
        return rows

    groups_avg = avg_map(scored, group_keys, "groups")
    breakdown_avg = avg_map(scored, breakdown_keys, "breakdown")

    worst_n = max(0, int(args.worst or 0))
    worst = sorted(scored, key=lambda x: float(x.get("optimizer_score", 0.0)))[:worst_n] if worst_n else []

    lines: List[str] = []
    lines.append("Question Planner Quality Report")
    lines.append("")
    lines.append(f"examples: {len(scored)}")
    lines.append(f"score: mean={mean:.2f} median={median:.2f} min={min(scores):.2f} max={max(scores):.2f} p10={p10:.2f} p90={p90:.2f}")
    lines.append(
        f"optimizer_score: mean={opt_mean:.2f} median={opt_median:.2f} min={min(optimizer_scores):.2f} max={max(optimizer_scores):.2f} p10={opt_p10:.2f} p90={opt_p90:.2f}"
    )
    lines.append("")
    lines.append("Groups (avg)")
    lines.append(_render_table(groups_avg).rstrip())
    lines.append("")
    lines.append("Breakdown (avg)")
    lines.append(_render_table(breakdown_avg).rstrip())

    if worst:
        lines.append("")
        lines.append(f"Worst {len(worst)} examples (by optimizer_score)")
        for i, ex in enumerate(worst):
            s = float(ex.get("score", 0.0))
            opt_s = float(ex.get("optimizer_score", 0.0))
            summary = str(ex.get("services_summary") or "").strip()
            notes = "; ".join([str(x) for x in (ex.get("notes") or []) if str(x).strip()])[:240]
            lines.append(f"- #{i+1} optimizer_score={opt_s:.2f} score={s:.2f} services_summary={summary[:140]}")
            if notes:
                lines.append(f"  notes={notes}")

    report = "\n".join(lines).rstrip() + "\n"
    if str(args.out or "").strip():
        out_path = Path(str(args.out)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(list(sys.argv[1:])))
