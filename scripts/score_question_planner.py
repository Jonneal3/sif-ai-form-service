#!/usr/bin/env python3
from __future__ import annotations

"""
Score Question Planner output with the plan-quality metric (LLM-judged).

Examples:
  python3 scripts/score_question_planner.py --context-json '{"services_summary":"Industry: Bathroom Remodeling. Service: Guest bath refresh."}' --plan-json '{"plan":[{"key":"style_direction","question":"Which overall style best matches what you want?"}]}'

  cat planner_logs.txt | python3 scripts/score_question_planner.py --stdin-json
"""

import argparse
import json
import sys
from typing import Any, Dict

from metrics.question_planner.plan_quality import score_question_plan


def _try_json(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


def _as_json_str(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-json", default="", help="Planner context JSON string (or plain JSON).")
    ap.add_argument("--plan-json", default="", help="Planner output question_plan_json (or plain JSON).")
    ap.add_argument("--stdin-json", action="store_true", help="Read newline-delimited JSON objects from stdin.")
    args = ap.parse_args(argv)

    if args.stdin_json:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            obj = _try_json(line)
            if not isinstance(obj, dict):
                continue
            ctx = obj.get("planner_context_json") or obj.get("context") or obj.get("plannerContextJson") or ""
            plan = obj.get("question_plan_json") or obj.get("plan") or obj.get("questionPlanJson") or ""
            res = score_question_plan(planner_context_json=str(ctx), question_plan_json=str(plan))
            out: Dict[str, Any] = {
                "score": round(res.score, 4),
                "breakdown": {k: round(float(v), 4) for k, v in res.breakdown.items()},
                "groups": {k: round(float(v), 4) for k, v in (res.groups or {}).items()},
                "notes": res.notes,
            }
            print(_as_json_str(out))
        return 0

    ctx_raw = args.context_json.strip()
    plan_raw = args.plan_json.strip()
    if not ctx_raw or not plan_raw:
        raise SystemExit("Provide --context-json and --plan-json, or use --stdin-json.")

    # Allow passing plain JSON (not JSON string) by normalizing it.
    ctx_obj = _try_json(ctx_raw)
    if isinstance(ctx_obj, (dict, list)):
        ctx_raw = _as_json_str(ctx_obj)
    plan_obj = _try_json(plan_raw)
    if isinstance(plan_obj, (dict, list)):
        plan_raw = _as_json_str(plan_obj)

    res = score_question_plan(planner_context_json=ctx_raw, question_plan_json=plan_raw)
    print(f"score={res.score:.4f}")  # 0..100
    if res.groups:
        for k, v in sorted(res.groups.items()):
            print(f"group.{k}={float(v):.4f}")  # 0..100
    for k, v in sorted(res.breakdown.items()):
        print(f"{k}={float(v):.4f}")  # 0..100
    if res.notes:
        print("notes=" + "; ".join(res.notes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
