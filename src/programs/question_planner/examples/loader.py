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

def _ensure_json_str(v: Any) -> str:
    """
    Allow demos to store JSON inputs/outputs as either compact JSON strings
    or native JSON objects (dict/list). DSPy sees only strings at runtime.
    """
    if isinstance(v, (dict, list)):
        return _compact_json(v)
    return str(v or "").strip()


def default_design_demos() -> list[dspy.Example]:
    """
    Load pretty-printed JSON examples and convert to DSPy demos.

    Supported formats (per list item):

    1) Explicit DSPy record (preferred):
      { "inputs": {...}, "outputs": {...} }

      Inputs should match the signature:
        - planner_context_json: str | dict
        - max_steps: int
        - allowed_mini_types: [str, ...]
      Outputs:
        - question_plan_json: str | dict (object with top-level `plan` array)

    2) Legacy human-friendly format (back-compat):
      { "services_summary": str, "plan": [ { "key": str, "question": str }, ... ] }
    """

    records: list[dict] = []
    for item in _load_examples_json():
        if not isinstance(item, dict):
            continue

        # Preferred explicit record: {"inputs": {...}, "outputs": {...}}
        if isinstance(item.get("inputs"), dict) and isinstance(item.get("outputs"), dict):
            inputs = dict(item["inputs"])
            outputs = dict(item["outputs"])

            # Fill defaults if omitted.
            if "max_steps" not in inputs:
                inputs["max_steps"] = int(DEFAULT_MAX_STEPS)
            if "allowed_mini_types" not in inputs:
                inputs["allowed_mini_types"] = list(DEFAULT_ALLOWED_MINI_TYPES)

            # Normalize JSON payloads to strings.
            inputs["planner_context_json"] = _ensure_json_str(inputs.get("planner_context_json"))
            outputs["question_plan_json"] = _ensure_json_str(outputs.get("question_plan_json"))

            # Ensure required fields exist.
            if not str(inputs.get("planner_context_json") or "").strip():
                continue
            if not outputs.get("question_plan_json"):
                continue

            records.append({"inputs": inputs, "outputs": outputs})
            continue

        # Legacy shape: {"services_summary": ..., "plan": [...]}
        services_summary = str(item.get("services_summary") or "").strip()
        plan = item.get("plan")
        if not services_summary or not isinstance(plan, list) or not plan:
            continue

        context = {"services_summary": services_summary, "answered_qa": [], "asked_step_ids": []}
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

