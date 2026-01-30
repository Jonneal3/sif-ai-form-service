#!/usr/bin/env python3
from __future__ import annotations

"""
Pretty-print question planner demos in a human-readable form.

Supports:
- Compiled DSPy artifact JSON (contains `prog.demos[]`)
- Demo pack JSONL (each line is {"inputs":...,"outputs":...})

Examples:
  PYTHONPATH=.:src python3 scripts/inspect_question_planner_demos.py \
    src/programs/question_planner/data/optimized_outputs/question_planner_optimized.json

  PYTHONPATH=.:src python3 scripts/inspect_question_planner_demos.py \
    src/programs/question_planner/data/optimized_outputs/question_planner_demo_pack.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _try_json_loads(s: Any) -> Any:
    try:
        return json.loads(str(s))
    except Exception:
        return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = _try_json_loads(line)
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _extract_from_compiled(obj: Any) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return []
    prog = obj.get("prog")
    if not isinstance(prog, dict):
        return []
    demos = prog.get("demos")
    if not isinstance(demos, list):
        return []
    return [d for d in demos if isinstance(d, dict)]


def _extract_from_demo_pack(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records:
        inputs = rec.get("inputs") if isinstance(rec.get("inputs"), dict) else {}
        outputs = rec.get("outputs") if isinstance(rec.get("outputs"), dict) else {}
        if not inputs or not outputs:
            continue
        demo = dict(inputs)
        demo.update(outputs)
        out.append(demo)
    return out


def _decode_demo_fields(demo: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(demo)
    for k in ("planner_context_json", "question_plan_json"):
        v = out.get(k)
        parsed = _try_json_loads(v) if isinstance(v, str) else None
        if parsed is not None:
            out[k] = parsed
    return out


def _demo_title(demo: Dict[str, Any]) -> str:
    ctx = demo.get("planner_context_json")
    if isinstance(ctx, dict):
        s = str(ctx.get("services_summary") or ctx.get("grounding_summary") or "").strip()
        if s:
            return s[:120]
    return ""


def _render_plain_demo(demo: Dict[str, Any]) -> str:
    ctx = demo.get("planner_context_json") if isinstance(demo.get("planner_context_json"), dict) else {}
    plan_obj = demo.get("question_plan_json") if isinstance(demo.get("question_plan_json"), dict) else {}
    plan = plan_obj.get("plan") if isinstance(plan_obj, dict) else None
    plan = plan if isinstance(plan, list) else []

    lines: List[str] = []
    summary = str(ctx.get("services_summary") or ctx.get("grounding_summary") or "").strip()
    if summary:
        lines.append(f"Service: {summary}")

    answered = ctx.get("answered_qa")
    if isinstance(answered, list) and answered:
        lines.append("Already answered:")
        for qa in answered:
            if not isinstance(qa, dict):
                continue
            q = str(qa.get("question") or "").strip()
            a = qa.get("answer")
            lines.append(f"- {q} -> {a}")

    lines.append("Plan:")
    for idx, step in enumerate(plan, start=1):
        if not isinstance(step, dict):
            continue
        key = str(step.get("key") or "").strip()
        question = str(step.get("question") or "").strip()
        if key:
            lines.append(f"{idx}. [{key}] {question}")
        else:
            lines.append(f"{idx}. {question}")

        options = step.get("option_hints")
        if isinstance(options, list) and options:
            opt_lines: List[str] = []
            for opt in options:
                if isinstance(opt, str):
                    label = opt.strip()
                elif isinstance(opt, dict):
                    label = str(opt.get("label") or opt.get("value") or "").strip()
                else:
                    label = ""
                if label:
                    opt_lines.append(f"   - {label}")
            lines.extend(opt_lines)

    return "\n".join(lines).rstrip()


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to compiled artifact JSON or demo pack JSONL.")
    ap.add_argument("--raw", action="store_true", help="Do not parse JSON-in-string fields.")
    ap.add_argument(
        "--format",
        default="json",
        choices=["json", "plain"],
        help="Output format: json (pretty JSON) or plain (plain-English).",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max demos to print (0 = no limit).")
    ap.add_argument("--out", default="", help="Write output to a file instead of stdout.")
    args = ap.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Not found: {path}")

    if path.suffix.lower() == ".jsonl":
        demos = _extract_from_demo_pack(_read_jsonl(path))
    else:
        demos = _extract_from_compiled(_read_json(path))

    if args.limit and args.limit > 0:
        demos = demos[: int(args.limit)]

    chunks: List[str] = []
    for i, d in enumerate(demos):
        demo = dict(d) if isinstance(d, dict) else {}
        if not args.raw:
            demo = _decode_demo_fields(demo)
        title = _demo_title(demo)
        hdr = f"== demo {i} ==" if not title else f"== demo {i}: {title} =="
        chunks.append(hdr)
        if args.format == "plain":
            chunks.append(_render_plain_demo(demo))
        else:
            chunks.append(json.dumps(demo, ensure_ascii=False, indent=2, sort_keys=True))
        chunks.append("")

    rendered = "\n".join(chunks).rstrip() + "\n"
    if str(args.out or "").strip():
        out_path = Path(str(args.out)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(list(__import__('sys').argv[1:])))
