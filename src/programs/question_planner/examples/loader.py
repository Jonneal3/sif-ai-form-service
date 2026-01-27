from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dspy

from programs.dspy_demos import as_dspy_examples


DEFAULT_MAX_STEPS = 8
DEFAULT_ALLOWED_MINI_TYPES: list[str] = [
    "multiple_choice",
    "yes_no",
    "segmented_choice",
    "chips_multi",
    "slider",
    "range_slider",
]


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _load_examples_json() -> list[dict]:
    path = Path(__file__).with_name("demo_examples.json")
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def default_design_demos() -> list[dspy.Example]:
    """
    Load pretty-printed JSON examples and convert to DSPy demos.

    JSON file is a list of:
      { "services_summary": str, "plan": [ { "key": str, "question": str }, ... ] }
    """

    records: list[dict] = []
    for item in _load_examples_json():
        if not isinstance(item, dict):
            continue
        services_summary = str(item.get("services_summary") or "").strip()
        plan = item.get("plan")
        if not services_summary or not isinstance(plan, list) or not plan:
            continue

        context = {"services_summary": services_summary, "answered_qa": []}
        records.append(
            {
                "inputs": {
                    "planner_context_json": _compact_json(context),
                    "max_steps": int(DEFAULT_MAX_STEPS),
                    "allowed_mini_types": list(DEFAULT_ALLOWED_MINI_TYPES),
                },
                "outputs": {"question_plan_json": _compact_json({"plan": plan})},
            }
        )

    return as_dspy_examples(records, input_keys=["planner_context_json", "max_steps", "allowed_mini_types"])


__all__ = ["DEFAULT_ALLOWED_MINI_TYPES", "DEFAULT_MAX_STEPS", "default_design_demos"]

