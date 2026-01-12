"""
DSPy Examples Registry (Stage 1)

Goal:
- Provide a single, obvious place to store and load `dspy.Example` demos.
- Keep examples INDUSTRY-AGNOSTIC.
- Keep examples industry-agnostic; avoid hardcoded vertical facts.

Examples are stored as JSONL so adding more is as easy as appending lines.
Each line is a JSON object with:
  - "inputs": dict
  - "outputs": dict
  - optional "meta": dict
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ExampleRecord:
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    meta: Dict[str, Any] | None = None


def _examples_dir() -> str:
    return os.path.join(os.path.dirname(__file__))


def load_jsonl_records(filename: str) -> List[ExampleRecord]:
    # Allow callers to reference a pack by:
    # - plain filename (looked up under this repo's `examples/` dir)
    # - repo-relative or absolute path (useful for shared packs via git submodule/subtree)
    path = str(filename or "").strip()
    if not path:
        return []
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(_examples_dir(), path)
    records: List[ExampleRecord] = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = (line or "").strip()
            if not t:
                continue
            obj = json.loads(t)
            inputs = obj.get("inputs") if isinstance(obj, dict) else None
            outputs = obj.get("outputs") if isinstance(obj, dict) else None
            meta = obj.get("meta") if isinstance(obj, dict) else None
            if isinstance(inputs, dict) and isinstance(outputs, dict):
                records.append(ExampleRecord(inputs=inputs, outputs=outputs, meta=meta if isinstance(meta, dict) else None))
    return records


def as_dspy_examples(
    records: Iterable[ExampleRecord],
    *,
    input_keys: Optional[List[str]] = None,
) -> List[Any]:
    """
    Convert ExampleRecord -> `dspy.Example` list.

    Notes:
    - We keep this import local so the registry can be imported even when `dspy` isn't installed.
    - `input_keys` lets us explicitly mark which fields are treated as inputs for the demo.
    """
    try:
        import dspy  # type: ignore
    except Exception:
        return []

    demos: List[Any] = []
    for r in records:
        ex = dspy.Example(**(r.inputs or {}), **(r.outputs or {}))
        if input_keys:
            try:
                ex = ex.with_inputs(*input_keys)
            except Exception:
                # Some DSPy versions differ; fail-open so registry never crashes runtime.
                pass
        demos.append(ex)
    return demos


def load_examples_pack(pack: str) -> List[ExampleRecord]:
    """
    Known packs:
    - form_planner_examples.jsonl
    - copy_examples.jsonl
    - schema_examples.jsonl
    """
    return load_jsonl_records(pack)


