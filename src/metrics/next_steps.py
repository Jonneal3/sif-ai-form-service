from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence


def parse_jsonl_objects(text: str) -> List[Any]:
    """
    Best-effort parse of a JSONL string (one JSON value per line).

    Returns a list of parsed JSON objects. Skips blank lines and lines that fail JSON parsing.
    """
    if not text:
        return []
    objects: List[Any] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            objects.append(json.loads(line))
        except Exception:
            continue
    return objects


@dataclass(frozen=True)
class NextStepsMetrics:
    ok: bool
    valid_jsonl_rate: float
    parsed_steps: int
    raw_lines: int
    max_steps: Optional[int]
    within_max_steps: Optional[bool]
    id_coverage: Optional[float]


def _extract_step_ids(steps: Sequence[Any]) -> List[str]:
    ids: List[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = step.get("id") or step.get("stepId") or step.get("step_id")
        if step_id is None:
            continue
        s = str(step_id).strip()
        if s:
            ids.append(s)
    return ids


def compute_next_steps_metrics(
    *,
    predicted_jsonl: str,
    expected_step_ids: Optional[Iterable[str]] = None,
    max_steps: Optional[int] = None,
) -> NextStepsMetrics:
    """
    Lightweight metrics for `NextStepsJSONL` outputs.

    - `valid_jsonl_rate`: fraction of non-empty lines that are valid JSON.
    - `within_max_steps`: whether parsed step count is <= `max_steps` (if provided).
    - `id_coverage`: fraction of `expected_step_ids` that appear in parsed steps (if provided).
    """
    lines = [ln.strip() for ln in str(predicted_jsonl or "").splitlines() if ln.strip()]
    raw_lines = len(lines)
    parsed = parse_jsonl_objects(predicted_jsonl)
    parsed_steps = len([x for x in parsed if isinstance(x, dict)])

    valid_jsonl_rate = 1.0 if raw_lines == 0 else min(1.0, max(0.0, len(parsed) / raw_lines))

    within_max_steps: Optional[bool] = None
    if isinstance(max_steps, int) and max_steps > 0:
        within_max_steps = parsed_steps <= max_steps

    id_coverage: Optional[float] = None
    if expected_step_ids is not None:
        expected = {str(s).strip() for s in expected_step_ids if str(s).strip()}
        if expected:
            predicted_ids = set(_extract_step_ids([x for x in parsed if isinstance(x, dict)]))
            id_coverage = len(expected & predicted_ids) / len(expected)
        else:
            id_coverage = None

    ok = valid_jsonl_rate >= 0.9 and (within_max_steps is not False)

    return NextStepsMetrics(
        ok=ok,
        valid_jsonl_rate=valid_jsonl_rate,
        parsed_steps=parsed_steps,
        raw_lines=raw_lines,
        max_steps=max_steps,
        within_max_steps=within_max_steps,
        id_coverage=id_coverage,
    )

