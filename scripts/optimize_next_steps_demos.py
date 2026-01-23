#!/usr/bin/env python3
from __future__ import annotations

"""
Offline "optimizer" for demo packs.

Today, this script implements a safe default:
- `copy` mode writes `batch_{i}.optimized.jsonl` next to each base pack.
  (This enables the runtime "prefer optimized pack" code path immediately.)

Later, you can add real DSPy teleprompt optimization in `bootstrap` mode.

Run:
  # safe default (no LLM calls)
  PYTHONPATH=.:src python scripts/optimize_next_steps_demos.py --mode copy

  # real optimization (requires DSPy LM configured)
  # PYTHONPATH=.:src python scripts/optimize_next_steps_demos.py --mode bootstrap
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + ("\n" if rows else ""), encoding="utf-8")


def _copy_optimize_pack(base_pack: Path) -> Path:
    rows = _load_jsonl(base_pack)
    out_rows: list[dict] = []
    for r in rows:
        meta = r.get("meta") if isinstance(r.get("meta"), dict) else {}
        meta = dict(meta)
        meta.setdefault("optimized", True)
        meta.setdefault("optimized_source", "copy")
        meta.setdefault("base_pack", str(base_pack))
        out = dict(r)
        out["meta"] = meta
        out_rows.append(out)

    out_path = base_pack.with_suffix("")  # strip .jsonl
    out_path = Path(str(out_path) + ".optimized.jsonl")
    _write_jsonl(out_path, out_rows)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["copy", "bootstrap"], default="copy")
    ap.add_argument(
        "--examples-root",
        default=str(_repo_root() / "src" / "programs" / "batch_generator" / "examples"),
    )
    ap.add_argument("--use-case", default="", help="If set, only optimize this use_case folder")
    ap.add_argument("--total-batches", type=int, default=0, help="If set, only optimize b{N} folders")
    args = ap.parse_args()

    root = Path(args.examples_root)
    if not root.exists():
        raise SystemExit(f"examples root not found: {root}")

    use_cases = [args.use_case] if args.use_case else ["scene", "scene_placement", "tryon"]
    written: list[Path] = []

    for uc in use_cases:
        uc_dir = root / uc
        if not uc_dir.exists():
            continue
        b_dirs = [p for p in uc_dir.glob("b*") if p.is_dir()]
        if args.total_batches:
            b_dirs = [p for p in b_dirs if p.name == f"b{int(args.total_batches)}"]
        for bdir in sorted(b_dirs):
            for pack in sorted(bdir.glob("batch_*.jsonl")):
                if pack.name.endswith(".optimized.jsonl"):
                    continue
                if args.mode == "copy":
                    written.append(_copy_optimize_pack(pack))
                else:
                    # Bootstrap mode: generate demos by running the program and scoring candidates.
                    # This is intentionally offline-only and requires a DSPy LM configuration.
                    try:
                        import dspy  # type: ignore

                        from optimizers.bootstrap_fewshot import bootstrap_few_shot  # type: ignore
                        from programs.batch_generator.engine.dspy_program import BatchStepsProgram
                    except Exception as e:  # pragma: no cover
                        raise SystemExit(f"bootstrap mode unavailable: {e}")

                    # Build a tiny trainset from the current pack's own inputs (context_json/max_steps/allowed_mini_types).
                    # We intentionally do *not* provide gold outputs; the metric scores the prediction itself.
                    records = _load_jsonl(pack)
                    trainset: list[Any] = []
                    for r in records:
                        ins = r.get("inputs") if isinstance(r.get("inputs"), dict) else {}
                        if not ins:
                            continue
                        try:
                            ex = dspy.Example(**ins).with_inputs("context_json", "max_steps", "allowed_mini_types")
                        except Exception:
                            continue
                        trainset.append(ex)

                    def metric(example: Any, pred: Any, trace: Any = None) -> float:  # type: ignore[override]
                        try:
                            raw = getattr(pred, "mini_steps_jsonl", "") or ""
                            steps: list[dict] = []
                            for l in str(raw).splitlines():
                                try:
                                    obj = json.loads(l)
                                except Exception:
                                    continue
                                if isinstance(obj, dict):
                                    steps.append(obj)
                            # Basic validators: schema + no filler patterns.
                            # (More validators and a proxy judge can be added here.)
                            ok, _errs = _validate_steps_schema_for_opt(steps)
                            ok2, _errs2 = _validate_no_filler_patterns_for_opt(steps)
                            return 1.0 if (ok and ok2) else 0.0
                        except Exception:
                            return 0.0

                    program = BatchStepsProgram(demo_pack=str(pack))
                    compiled = bootstrap_few_shot(program=program, trainset=trainset, metric=metric, max_demos=8)

                    demos = getattr(getattr(compiled, "prog", None), "demos", None)
                    if not demos:
                        written.append(_copy_optimize_pack(pack))
                        continue
                    out_rows: List[Dict[str, Any]] = []
                    for d in demos:
                        data = getattr(d, "_store", None) or getattr(d, "__dict__", None) or {}
                        inputs = {k: data.get(k) for k in ("context_json", "max_steps", "allowed_mini_types") if k in data}
                        outputs = {k: data.get(k) for k in ("mini_steps_jsonl",) if k in data}
                        if not inputs or not outputs:
                            continue
                        out_rows.append(
                            {
                                "meta": {
                                    "optimized": True,
                                    "optimized_source": "bootstrap_few_shot",
                                    "base_pack": str(pack),
                                },
                                "inputs": inputs,
                                "outputs": outputs,
                            }
                        )
                    if not out_rows:
                        written.append(_copy_optimize_pack(pack))
                        continue
                    out_path = Path(str(pack.with_suffix("")) + ".optimized.jsonl")
                    _write_jsonl(out_path, out_rows)
                    written.append(out_path)

    print(f"Wrote {len(written)} optimized packs")
    return 0


def _validate_steps_schema_for_opt(steps: list[dict]) -> tuple[bool, list[str]]:
    # Keep this local so bootstrap mode has a stable minimal dependency surface.
    from schemas.ui_steps import FileUploadUI, GalleryUI, MultipleChoiceUI, TextInputUI

    type_to_model = {
        "text": TextInputUI,
        "text_input": TextInputUI,
        "multiple_choice": MultipleChoiceUI,
        "choice": MultipleChoiceUI,
        "yes_no": MultipleChoiceUI,
        "upload": FileUploadUI,
        "file_upload": FileUploadUI,
        "gallery": GalleryUI,
    }
    errs: list[str] = []
    for i, s in enumerate(steps or []):
        t = str((s or {}).get("type") or "").strip().lower()
        m = type_to_model.get(t)
        if not m:
            errs.append(f"unsupported type '{t}'")
            continue
        try:
            m.model_validate(s)
        except Exception as e:
            errs.append(str(e))
    return (len(errs) == 0, errs)


def _validate_no_filler_patterns_for_opt(steps: list[dict]) -> tuple[bool, list[str]]:
    pat = re.compile(r"\b(option\s*[a-d]|category\s*\d+)\b", re.IGNORECASE)
    errs: list[str] = []
    for s in steps or []:
        blob = json.dumps(s, ensure_ascii=True).lower()
        if pat.search(blob):
            errs.append("filler pattern detected")
            break
    return (len(errs) == 0, errs)


if __name__ == "__main__":
    raise SystemExit(main())

