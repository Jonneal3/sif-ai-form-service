from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from eval.datasets import load_eval_cases
from eval.metrics import compute_metrics


def _print_json(obj: Any) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run offline evaluation for sif-ai-form-service DSPy planner.")
    ap.add_argument("--cases", default="eval_cases.jsonl", help="JSONL filename under eval/ (default: eval_cases.jsonl)")
    ap.add_argument("--fail-fast", action="store_true", help="Stop on first failing case.")
    ap.add_argument("--print-responses", action="store_true", help="Print raw responses (can be large).")
    args = ap.parse_args()

    cases = load_eval_cases(args.cases)
    if not cases:
        sys.stderr.write(f"[eval] No cases found for eval/{args.cases}\n")
        return 2

    # Import the planner lazily so running `python -m eval.run_eval` is the single entrypoint.
    from flow_planner import next_steps_jsonl

    passed = 0
    failed = 0

    for c in cases:
        sys.stderr.write(f"[eval] case={c.name}\n")
        try:
            result = next_steps_jsonl(dict(c.payload), stream=False)
        except Exception as e:
            failed += 1
            sys.stderr.write(f"[eval] ❌ exception: {type(e).__name__}: {e}\n")
            if args.fail_fast:
                return 1
            continue

        if args.print_responses:
            _print_json({"case": c.name, "result": result})

        metrics = compute_metrics(c.payload, result if isinstance(result, dict) else {"ok": False})
        ok = (
            metrics.within_max_steps
            and metrics.ids_all_normalized
            and metrics.no_steps_in_already_asked
            and metrics.types_all_allowed
            and metrics.has_min_step_when_needed
        )

        if ok:
            passed += 1
            sys.stderr.write(
                f"[eval] ✅ passed steps={metrics.num_steps} normalized={metrics.ids_all_normalized} errors={list(metrics.errors)}\n"
            )
        else:
            failed += 1
            sys.stderr.write(
                f"[eval] ❌ failed steps={metrics.num_steps} normalized={metrics.ids_all_normalized} errors={list(metrics.errors)}\n"
            )
            if args.fail_fast:
                return 1

    sys.stderr.write(f"[eval] done passed={passed} failed={failed}\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


