#!/usr/bin/env python3
from __future__ import annotations

"""
Offline optimizer run for the DSPy Question Planner.

This compiles `QuestionPlannerProgram` using a DSPy optimizer and saves:
- the compiled program artifact (DSPy `.save()` JSON)
- optionally, a JSONL demo-pack that can be loaded at runtime via `DSPY_PLANNER_DEMO_PACK`

Run (recommended):
  PYTHONPATH=.:src python3 -m optimizers.optimize_question_planner --help
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_outputs_dir() -> Path:
    return _repo_root() / "src" / "programs" / "question_planner" / "data" / "optimized_outputs"


def _load_dotenv_files() -> None:
    """
    Match service behavior: best-effort load `.env` + `.env.local` when present.
    This keeps local runs convenient without requiring `source .env.local`.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    root = _repo_root()
    load_dotenv(root / ".env", override=False)
    load_dotenv(root / ".env.local", override=False)


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _try_json_loads(text: Any) -> Any:
    try:
        return json.loads(str(text))
    except Exception:
        return None


def _ensure_json_str(v: Any) -> str:
    if isinstance(v, (dict, list)):
        return _compact_json(v)
    return str(v or "").strip()


def _load_records(path: Path) -> List[Dict[str, Any]]:
    """
    Loads either:
      - JSON array of {"inputs": {...}, "outputs": {...}} records
      - JSONL with one record per line (same shape)
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() == ".jsonl":
        out: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = _try_json_loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        return out

    raw = _try_json_loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    raise ValueError(f"Expected a JSON array in {path}")


def _normalize_trainset_records(
    *,
    records: Iterable[Dict[str, Any]],
    default_max_steps: int,
    default_allowed_mini_types: List[str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records:
        inputs = rec.get("inputs") if isinstance(rec.get("inputs"), dict) else None
        outputs = rec.get("outputs") if isinstance(rec.get("outputs"), dict) else None

        # Support "flat" demo records:
        # {"planner_context_json": {...}, "max_steps": 12, "allowed_mini_types": [...], "question_plan_json": {...}}
        if inputs is None:
            if not isinstance(rec, dict):
                continue
            inputs = {
                "planner_context_json": rec.get("planner_context_json")
                or rec.get("context")
                or rec.get("plannerContextJson")
                or rec.get("planner_context")
                or "",
                "max_steps": rec.get("max_steps"),
                "allowed_mini_types": rec.get("allowed_mini_types"),
            }
            outputs = {
                "question_plan_json": rec.get("question_plan_json")
                or rec.get("plan")
                or rec.get("questionPlanJson")
                or rec.get("question_plan")
                or ""
            }

        if not isinstance(inputs, dict) or not inputs:
            continue
        if not isinstance(outputs, dict):
            outputs = {}

        norm_inputs = dict(inputs)
        if "max_steps" not in norm_inputs:
            norm_inputs["max_steps"] = int(default_max_steps)
        if not norm_inputs.get("max_steps"):
            norm_inputs["max_steps"] = int(default_max_steps)
        if "allowed_mini_types" not in norm_inputs:
            norm_inputs["allowed_mini_types"] = list(default_allowed_mini_types)
        if not norm_inputs.get("allowed_mini_types"):
            norm_inputs["allowed_mini_types"] = list(default_allowed_mini_types)

        norm_inputs["planner_context_json"] = _ensure_json_str(norm_inputs.get("planner_context_json"))

        norm_outputs = dict(outputs)
        if norm_outputs.get("question_plan_json"):
            norm_outputs["question_plan_json"] = _ensure_json_str(norm_outputs.get("question_plan_json"))
        else:
            norm_outputs.pop("question_plan_json", None)

        out.append({"inputs": norm_inputs, "outputs": norm_outputs})
    return out


def _as_dspy_examples(records: Iterable[Dict[str, Any]]) -> List[Any]:
    """
    Like `programs.dspy_demos.as_dspy_examples`, but allows unlabeled records.
    """
    import dspy  # type: ignore

    examples: List[Any] = []
    for rec in records:
        inputs = rec.get("inputs") if isinstance(rec.get("inputs"), dict) else {}
        outputs = rec.get("outputs") if isinstance(rec.get("outputs"), dict) else {}
        if not inputs:
            continue

        fields: Dict[str, Any] = dict(inputs)
        if outputs:
            fields.update(outputs)

        try:
            ex = dspy.Example(**fields).with_inputs("planner_context_json", "max_steps", "allowed_mini_types")
        except Exception:
            continue
        examples.append(ex)
    return examples


def _configure_dspy_for_optimizer() -> None:
    import dspy  # type: ignore

    from programs.common.dspy_runtime import make_dspy_lm_for_module
    from programs.common.dspy_runtime import configure_dspy
    from programs.common.env import env_float, env_int

    cfg = make_dspy_lm_for_module(module_env_prefix="DSPY_PLANNER", allow_small_models=False)
    if not cfg:
        raise SystemExit(
            "DSPy LM is not configured. Set `DSPY_PROVIDER` + provider API key (e.g. `OPENAI_API_KEY` or `GROQ_API_KEY`)."
        )

    default_timeout = env_float("DSPY_LLM_TIMEOUT_SEC", 20.0)
    default_temperature = env_float("DSPY_TEMPERATURE", 0.7)
    default_max_tokens = env_int("DSPY_NEXT_STEPS_MAX_TOKENS", 2000)
    default_num_retries = env_int("DSPY_LLM_NUM_RETRIES", 2)

    planner_timeout = env_float("DSPY_PLANNER_TIMEOUT_SEC", default_timeout)
    planner_temperature = env_float("DSPY_PLANNER_TEMPERATURE", default_temperature)
    planner_max_tokens = env_int("DSPY_PLANNER_MAX_TOKENS", default_max_tokens)
    planner_num_retries = env_int("DSPY_PLANNER_NUM_RETRIES", default_num_retries)

    lm = dspy.LM(
        model=cfg["model"],
        temperature=planner_temperature,
        max_tokens=planner_max_tokens,
        timeout=planner_timeout,
        num_retries=max(0, int(planner_num_retries)),
    )
    configure_dspy(lm)


def _write_demo_pack_jsonl(*, demos: Iterable[Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _as_str(v: Any) -> str:
        return str(v or "").strip()

    input_keys = ["planner_context_json", "max_steps", "allowed_mini_types"]
    with out_path.open("w", encoding="utf-8") as f:
        for ex in demos:
            inputs = {k: getattr(ex, k, None) for k in input_keys}
            outputs = {"question_plan_json": _as_str(getattr(ex, "question_plan_json", None))}
            # Normalize planner_context_json to a compact JSON string (stable diffs).
            inputs["planner_context_json"] = _ensure_json_str(inputs.get("planner_context_json"))
            rec = {"inputs": inputs, "outputs": outputs}
            f.write(_compact_json(rec) + "\n")


def main(argv: List[str]) -> int:
    _load_dotenv_files()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trainset",
        default=str(_repo_root() / "src" / "programs" / "question_planner" / "data" / "examples" / "demo_examples.json"),
        help="Training set path (.json array or .jsonl records). Defaults to the planner demo examples.",
    )
    ap.add_argument(
        "--trainset-limit",
        type=int,
        default=0,
        help="Optional cap on number of training examples used by the optimizer (0 = all). Useful for fast/cheap runs.",
    )
    ap.add_argument(
        "--trainset-shuffle",
        action="store_true",
        help="Shuffle before applying --trainset-limit (deterministic given --seed).",
    )
    ap.add_argument(
        "--out",
        default=str(_default_outputs_dir() / "question_planner_optimized.json"),
        help="Path to write the compiled DSPy program artifact.",
    )
    ap.add_argument(
        "--export-demo-pack",
        default=str(_default_outputs_dir() / "question_planner_demo_pack.jsonl"),
        help="Path to write a JSONL demo-pack for runtime (set `DSPY_PLANNER_DEMO_PACK` to use it).",
    )
    ap.add_argument("--skip-export-demo-pack", action="store_true", help="Do not export a JSONL demo-pack.")
    ap.add_argument(
        "--usage-report",
        default="",
        help="Optional path to write a token-usage JSON report for metric judge calls (best-effort).",
    )
    ap.add_argument(
        "--allow-custom-output-paths",
        action="store_true",
        help="Allow writing outputs outside the default optimized_outputs folder.",
    )
    ap.add_argument("--max-labeled-demos", type=int, default=3)
    ap.add_argument("--max-bootstrapped-demos", type=int, default=4)
    ap.add_argument(
        "--num-candidate-programs",
        type=int,
        default=6,
        help="How many candidate programs to evaluate in random search. Lower = cheaper/faster.",
    )
    ap.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Parallelism for optimizer eval. Keep low to avoid provider rate limits (esp. Groq TPM).",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    random.seed(int(args.seed))

    default_dir = _default_outputs_dir().resolve()
    out_path = Path(str(args.out)).resolve()
    export_demo_pack = (not bool(args.skip_export_demo_pack)) and bool(str(args.export_demo_pack or "").strip())
    demo_pack_path = Path(str(args.export_demo_pack)).resolve() if export_demo_pack else None

    if not args.allow_custom_output_paths:
        if default_dir not in (out_path, *out_path.parents):
            raise SystemExit(f"Refusing to write outside {default_dir}. Pass --allow-custom-output-paths to override.")
        if demo_pack_path is not None and default_dir not in (demo_pack_path, *demo_pack_path.parents):
            raise SystemExit(f"Refusing to write outside {default_dir}. Pass --allow-custom-output-paths to override.")

    _configure_dspy_for_optimizer()

    from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch  # type: ignore
    from metrics.question_planner.plan_quality import question_planner_quality_metric
    from programs.question_planner.program import QuestionPlannerProgram

    trainset_path = Path(str(args.trainset))
    raw_records = _load_records(trainset_path)
    norm_records = _normalize_trainset_records(
        records=raw_records,
        default_max_steps=12,
        default_allowed_mini_types=["multiple_choice"],
    )
    if int(args.trainset_limit or 0) > 0:
        limit = int(args.trainset_limit)
        if bool(args.trainset_shuffle):
            norm_records = list(norm_records)
            random.shuffle(norm_records)
        norm_records = norm_records[:limit]
    trainset = _as_dspy_examples(norm_records)
    if not trainset:
        raise SystemExit(f"No usable training examples found in: {trainset_path}")

    program = QuestionPlannerProgram()
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=question_planner_quality_metric,
        max_labeled_demos=int(args.max_labeled_demos),
        max_bootstrapped_demos=int(args.max_bootstrapped_demos),
        num_candidate_programs=int(args.num_candidate_programs),
        num_threads=int(args.num_threads),
    )

    optimized_program = optimizer.compile(program, trainset=trainset)

    # In small-data regimes, random search can pick the "uncompiled" candidate which may not
    # attach any demos. If the caller asked for a demo-pack, fall back to BootstrapFewShot
    # to force an explicit demo set.
    if export_demo_pack:
        demos = getattr(getattr(optimized_program, "prog", None), "demos", None)
        if not demos:
            print("warn: compiled program has no demos; falling back to BootstrapFewShot to build a demo-pack")
            fallback = BootstrapFewShot(
                metric=question_planner_quality_metric,
                max_labeled_demos=int(args.max_labeled_demos),
                max_bootstrapped_demos=int(args.max_bootstrapped_demos),
                max_rounds=1,
            )
            optimized_program = fallback.compile(program, trainset=trainset)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    optimized_program.save(str(out_path))

    if export_demo_pack:
        demos = getattr(getattr(optimized_program, "prog", None), "demos", None)
        if not demos:
            raise SystemExit("Compiled program has no `.prog.demos` to export (unexpected).")
        assert demo_pack_path is not None
        _write_demo_pack_jsonl(demos=demos, out_path=demo_pack_path)

    # Token usage report (best-effort; currently tracks metric judge calls when provider supports usage).
    if str(args.usage_report or "").strip():
        try:
            from metrics.question_planner.plan_quality import get_metric_usage_summary

            report_path = Path(str(args.usage_report)).resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(_compact_json(get_metric_usage_summary()) + "\n", encoding="utf-8")
            print(f"ok: wrote usage report to {report_path}")
        except Exception as e:
            print(f"warn: failed to write usage report ({type(e).__name__})")

    print(f"ok: saved compiled program to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
