from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import dspy

from programs.batch_generator.signatures.signature import BatchNextStepsJSONL


def _load_jsonl_records(path: str) -> list[dict]:
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


def _as_dspy_examples(records: Iterable[dict], *, input_keys: list[str]) -> list[dspy.Example]:
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


class BatchStepsProgram(dspy.Module):
    """
    Thin DSPy wrapper used by the orchestrator.

    - Holds a `dspy.Predict(Signature)` program
    - Optionally attaches demos (few-shot) if a demo pack path is provided
    """

    def __init__(self, *, demo_pack: str = "") -> None:
        super().__init__()
        self.prog = dspy.Predict(BatchNextStepsJSONL)
        if demo_pack:
            demos = _as_dspy_examples(
                _load_jsonl_records(demo_pack),
                input_keys=[
                    "context_json",
                    "max_steps",
                    "allowed_mini_types",
                ],
            )
            if demos:
                setattr(self.prog, "demos", demos)

    def forward(self, *, context_json: str, max_steps: int, allowed_mini_types: list[str]):  # type: ignore[override]
        return self.prog(
            context_json=context_json,
            max_steps=max_steps,
            allowed_mini_types=allowed_mini_types,
        )


__all__ = ["BatchStepsProgram"]

