#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _best_effort_parse_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    t = value.strip()
    if not t:
        return None
    try:
        return json.loads(t)
    except Exception:
        return None


def _derive_tags_from_comment(comment: Any, tags: list[str]) -> list[str]:
    derived: set[str] = {str(t or "").strip().lower() for t in tags if str(t or "").strip()}
    text = str(comment or "").lower()
    if "slider" in text and ("unit" in text or "units" in text or "prefix" in text or "suffix" in text):
        derived.add("slider_requires_units")
    if ("not sure" in text or "no not sure" in text) and ("option" in text or "options" in text):
        derived.add("choice_requires_not_sure")
    return sorted(x for x in derived if x)


def _backfill_obj(obj: dict) -> tuple[dict, bool]:
    inputs = obj.get("inputs")
    meta = obj.get("meta")
    if not isinstance(inputs, dict) or not isinstance(meta, dict):
        return obj, False
    expected = meta.get("expected")
    if not isinstance(expected, dict):
        return obj, False

    ctx = _best_effort_parse_json(inputs.get("context_json")) or {}
    if not isinstance(ctx, dict):
        ctx = {}

    changed = False
    if "feedback" not in ctx:
        ctx["feedback"] = expected
        changed = True
    if "feedback_metric_tags" not in ctx:
        fb_tags = expected.get("feedback_tags")
        tags = [str(t) for t in fb_tags] if isinstance(fb_tags, list) else []
        ctx["feedback_metric_tags"] = _derive_tags_from_comment(expected.get("comment"), tags)
        changed = True

    if changed:
        inputs["context_json"] = json.dumps(ctx, ensure_ascii=False)
        obj["inputs"] = inputs
    return obj, changed


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill feedback context into eval/feedback_cases.jsonl records.")
    ap.add_argument("--path", default="eval/feedback_cases.jsonl", help="JSONL file to backfill (in-place).")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Missing {path}")
        return 1

    lines = path.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    changed_count = 0
    for line in lines:
        t = (line or "").strip()
        if not t:
            continue
        try:
            obj = json.loads(t)
        except Exception:
            out_lines.append(t)
            continue
        if isinstance(obj, dict):
            obj2, changed = _backfill_obj(obj)
            if changed:
                changed_count += 1
            out_lines.append(json.dumps(obj2, ensure_ascii=False))
        else:
            out_lines.append(t)

    path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    print(f"Backfilled {changed_count} records in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

