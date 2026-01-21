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


def load_jsonl_records(path: str) -> List[ExampleRecord]:
    p = str(path or "").strip()
    if not p:
        return []
    if not os.path.exists(p):
        return []
    out: List[ExampleRecord] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            t = (line or "").strip()
            if not t:
                continue
            obj = json.loads(t)
            inputs = obj.get("inputs") if isinstance(obj, dict) else None
            outputs = obj.get("outputs") if isinstance(obj, dict) else None
            meta = obj.get("meta") if isinstance(obj, dict) else None
            if isinstance(inputs, dict) and isinstance(outputs, dict):
                out.append(
                    ExampleRecord(
                        inputs=inputs,
                        outputs=outputs,
                        meta=meta if isinstance(meta, dict) else None,
                    )
                )
    return out


def as_dspy_examples(
    records: Iterable[ExampleRecord],
    *,
    input_keys: Optional[List[str]] = None,
) -> List[Any]:
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
                pass
        demos.append(ex)
    return demos
