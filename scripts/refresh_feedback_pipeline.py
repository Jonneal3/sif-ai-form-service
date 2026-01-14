#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def _count_jsonl_entries(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _append_run_history(
    *,
    cases_path: Path,
    failures_path: Path,
    report_path: Path,
    summary_path: Path,
    optimize_pack: Path,
) -> None:
    history_path = Path("data/optimizer_runs.jsonl")
    history_path.parent.mkdir(parents=True, exist_ok=True)

    report_data = _load_json(report_path) or {}
    summary_data = _load_json(summary_path) or {}
    summary_runs = summary_data.get("runs") or []
    summary_run = summary_runs[-1] if summary_runs else None
    dropoffs = summary_data.get("dropoffs") or []
    issues: list[str] = []
    if dropoffs:
        for drop in dropoffs[:3]:
            issues.append(
                f"dropoff {drop.get('batch_id')}/{drop.get('step_id')} ({drop.get('event_type')})"
            )
    if report_data:
        examples = report_data.get("examples") or []
        worst = sorted(examples, key=lambda r: r.get("score", 0))[:3]
        for item in worst:
            issues.append(f"eval drop {item.get('name')} ({item.get('score', 0):.3f})")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cases_exported": _count_jsonl_entries(cases_path),
        "failures_exported": _count_jsonl_entries(failures_path),
        "eval_report": {
            "avg_score": report_data.get("avg_score"),
            "count": report_data.get("count"),
        }
        if report_data
        else None,
        "telemetry": {
            "processed": summary_run.get("processed") if summary_run else None,
            "dropoff_events": summary_run.get("dropoff_events") if summary_run else None,
            "feedback_events": summary_run.get("feedback_events") if summary_run else None,
        }
        if summary_run
        else None,
        "optimized_pack": str(optimize_pack),
        "optimized_demo_count": _count_jsonl_entries(optimize_pack),
        "issues": issues,
    }
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")

    if summary_run:
        print(
            "\nTelemetry insights (last run): "
            f"{summary_run.get('processed', '?')} rows, "
            f"{summary_run.get('dropoff_events', '?')} dropoffs, "
            f"{summary_run.get('feedback_events', '?')} feedback events."
        )
    if report_data:
        avg_score = report_data.get("avg_score")
        avg_display = f"{avg_score:.3f}" if isinstance(avg_score, (int, float)) else "n/a"
        count_display = report_data.get("count") or 0
        print(f"Eval report: avg_score={avg_display} over {count_display} examples.")
    print(f"Recorded optimizer summary to {history_path}")
def main() -> None:
    _load_dotenv(Path(".env"))
    _load_dotenv(Path(".env.local"))  # Also load .env.local (takes precedence)
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
    parser.add_argument(
        "--optimizer-max-tokens",
        type=int,
        help="Override `eval.optimize --max-tokens` (defaults to $DSPY_NEXT_STEPS_MAX_TOKENS or 1200).",
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
        optimizer_tokens = args.optimizer_max_tokens
        if optimizer_tokens is None:
            optimizer_tokens = int(os.getenv("DSPY_NEXT_STEPS_MAX_TOKENS") or "1200")
        optimize_cmd.extend(["--max-tokens", str(optimizer_tokens)])
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

    summary_path = Path("data/telemetry_summary.json")
    report_path = Path("eval/reports/next_steps_eval.json")
    _append_run_history(
        cases_path=cases_path,
        failures_path=Path(args.failures_out),
        report_path=report_path,
        summary_path=summary_path,
        optimize_pack=Path(args.optimize_out),
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
