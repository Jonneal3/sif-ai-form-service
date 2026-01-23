from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import dspy


def load_jsonl_records(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    records: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def as_dspy_examples(records: Iterable[dict], *, input_keys: list[str]) -> list[dspy.Example]:
    demos: list[dspy.Example] = []
    for rec in records:
        inputs = rec.get("inputs") if isinstance(rec.get("inputs"), dict) else {}
        outputs = rec.get("outputs") if isinstance(rec.get("outputs"), dict) else {}
        if not inputs or not outputs:
            continue
        try:
            ex = dspy.Example(**inputs, **outputs).with_inputs(*input_keys)
        except Exception:
            continue
        demos.append(ex)
    return demos


def iter_jsonl_objects(text: str) -> Iterable[Any]:
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue

