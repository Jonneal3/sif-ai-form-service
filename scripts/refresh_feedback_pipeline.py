#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _run_step(description: str, command: list[str]) -> None:
    print(f"\n==> {description}")
    print("Running:", " ".join(shlex.quote(arg) for arg in command))
    res = subprocess.run(command)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, raw_val = line.split("=", 1)
            key = key.strip()
            val = raw_val.strip()
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            os.environ.setdefault(key, val)


def main() -> None:
    _load_dotenv(Path(".env"))
    parser = argparse.ArgumentParser(
        description="Run the telemetry→feedback export, eval, and optional optimizer in sequence."
    )
    parser.add_argument("--since", help="Pass-through ISO timestamp filter for export.")
    parser.add_argument("--limit", type=int, help="Pass-through limit for export.")
    parser.add_argument(
        "--include-negative",
        action="store_true",
        help="Include downvotes/low ratings without send_to_dataset (pass-through to export).",
    )
    parser.add_argument(
        "--cases-out",
        default="eval/feedback_cases.jsonl",
        help="Path where export script should write eval cases.",
    )
    parser.add_argument(
        "--failures-out",
        default="eval/eval_cases_failures.jsonl",
        help="Path where export script should write failures.",
    )
    parser.add_argument(
        "--optimize-out",
        default="examples/next_steps_examples.optimized.jsonl",
        help="Output path for the optimizer pack.",
    )
    parser.add_argument(
        "--skip-optimize",
        action="store_true",
        help="Skip running the optimizer step (just export + eval).",
    )
    parser.add_argument(
        "--dataset",
        help="Explicit dataset path for `eval.run_eval` (defaults to `--cases-out`).",
    )
    parser.add_argument(
        "--collect-insights",
        action="store_true",
        help="Run the telemetry insights summarizer after exporting.",
    )
    parser.add_argument(
        "--insights-limit",
        type=int,
        default=1000,
        help="Row limit passed to the insights summarizer.",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases_out)
    cases_path.parent.mkdir(parents=True, exist_ok=True)

    export_cmd = [
        sys.executable,
        "scripts/export_eval_cases_from_feedback.py",
        "--out",
        str(cases_path),
        "--failures-out",
        args.failures_out,
    ]
    if args.since:
        export_cmd.extend(["--since", args.since])
    if args.limit:
        export_cmd.extend(["--limit", str(args.limit)])
    if args.include_negative:
        export_cmd.append("--include-negative")

    _run_step("Exporting feedback into eval cases", export_cmd)

    if not cases_path.exists() or cases_path.stat().st_size == 0:
        print("\nNo eval cases were exported from telemetry; skipping eval + optimizer.")
        print("Add feedback rows (event_type=step_feedback) to `telemetry_events` and rerun.")
        return

    dataset_path = Path(str(args.dataset)) if args.dataset else cases_path
    eval_cmd = [sys.executable, "-m", "eval.run_eval", "--dataset", str(dataset_path)]
    _run_step("Running invariant eval", eval_cmd)

    if not args.skip_optimize:
        optimize_cmd = [
            sys.executable,
            "-m",
            "eval.optimize",
            "--cases",
            str(cases_path),
            "--out",
            args.optimize_out,
        ]
        _run_step("Generating optimized demo pack", optimize_cmd)

    if args.collect_insights:
        insights_cmd = [
            sys.executable,
            "scripts/telemetry_insights.py",
            "--checkpoint",
            ".telemetry_checkpoint.json",
            "--summary",
            "data/telemetry_summary.json",
            "--limit",
            str(args.insights_limit),
        ]
        _run_step("Summarizing telemetry insights", insights_cmd)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
