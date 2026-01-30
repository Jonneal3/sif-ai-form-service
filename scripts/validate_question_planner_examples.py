#!/usr/bin/env python3
from __future__ import annotations

"""
Validate the question planner demo examples file.

This keeps our demo/trainset sane as we iterate on constraints.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _try_json_loads(v: Any) -> Any:
    try:
        return json.loads(str(v))
    except Exception:
        return None


def _as_obj(v: Any) -> Any:
    if isinstance(v, str):
        parsed = _try_json_loads(v)
        return parsed if parsed is not None else v
    return v


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "path",
        nargs="?",
        default="src/programs/question_planner/data/examples/demo_examples.json",
        help="Path to demo examples JSON file.",
    )
    ap.add_argument("--min-steps", type=int, default=8, help="Minimum plan length per example.")
    ap.add_argument("--min-steps-total", type=int, default=7, help="Expected batch_constraints.minStepsTotal.")
    ap.add_argument("--max-steps-total", type=int, default=12, help="Expected batch_constraints.maxStepsTotal.")
    ap.add_argument("--max-steps", type=int, default=12, help="Expected max_steps value.")
    args = ap.parse_args(argv)

    path = Path(str(args.path))
    if not path.exists():
        raise SystemExit(f"Not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("Expected a JSON array of examples.")

    problems: List[str] = []
    for i, rec in enumerate(raw):
        if not isinstance(rec, dict):
            problems.append(f"[{i}] not an object")
            continue

        ctx = _as_obj(rec.get("planner_context_json"))
        plan = _as_obj(rec.get("question_plan_json"))

        if not isinstance(ctx, dict):
            problems.append(f"[{i}] planner_context_json not an object")
            continue
        if not isinstance(plan, dict) or not isinstance(plan.get("plan"), list):
            problems.append(f"[{i}] question_plan_json missing plan[]")
            continue

        steps = len(plan.get("plan") or [])
        if steps < int(args.min_steps):
            problems.append(f"[{i}] plan too short: {steps} < {int(args.min_steps)}")

        if int(rec.get("max_steps") or 0) != int(args.max_steps):
            problems.append(f"[{i}] max_steps expected {int(args.max_steps)} got {rec.get('max_steps')}")

        asked = ctx.get("asked_step_ids")
        if asked != ["step-service-primary"]:
            problems.append(f"[{i}] asked_step_ids expected ['step-service-primary'] got {asked}")

        answered = ctx.get("answered_qa")
        if not isinstance(answered, list) or not answered:
            problems.append(f"[{i}] answered_qa missing/empty")
        else:
            has_service = any(isinstance(x, dict) and x.get("stepId") == "step-service-primary" for x in answered)
            if not has_service:
                problems.append(f"[{i}] answered_qa missing step-service-primary")

        bc = ctx.get("batch_constraints") if isinstance(ctx.get("batch_constraints"), dict) else {}
        if bc.get("minStepsTotal") != int(args.min_steps_total) or bc.get("maxStepsTotal") != int(args.max_steps_total):
            problems.append(
                f"[{i}] batch_constraints expected {{minStepsTotal:{int(args.min_steps_total)},maxStepsTotal:{int(args.max_steps_total)}}} got {bc}"
            )

    if problems:
        for p in problems:
            print(p)
        raise SystemExit(f"Invalid demo examples: {len(problems)} problem(s)")

    print(f"ok: {path} ({len(raw)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(list(__import__("sys").argv[1:])))
