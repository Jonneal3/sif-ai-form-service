from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dspy

from programs.dspy_demos import as_dspy_examples


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _jsonl(lines: list[dict]) -> str:
    # Deterministic JSONL: one object per line, compact and sorted keys.
    return "\n".join([_compact_json(x) for x in lines if isinstance(x, dict)]).strip() + "\n"

def _ensure_json_str(v: Any) -> str:
    """
    Allow demos to store JSON inputs/outputs as either compact JSON strings
    or native JSON objects (dict/list). DSPy sees only strings at runtime.
    """
    if isinstance(v, (dict, list)):
        return _compact_json(v)
    return str(v or "").strip()

def _ensure_jsonl(v: Any) -> str:
    """
    Allow demos to store renderer output as JSONL string OR as a list of dicts.
    """
    if isinstance(v, str):
        return v.strip() + ("\n" if v.strip() and not v.strip().endswith("\n") else "")
    if isinstance(v, list):
        return _jsonl([x for x in v if isinstance(x, dict)])
    return ""


def _load_examples_json() -> list[dict]:
    path = Path(__file__).with_name("demo_examples.json")
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def default_renderer_demos() -> list[dspy.Example]:
    """
    Load pretty-printed JSON examples and convert to DSPy demos.

    Supported formats (per list item):

    1) Explicit DSPy record (preferred):
      { "inputs": {...}, "outputs": {...} }

      Inputs should match the signature:
        - question_plan_json: str | dict (object with top-level `plan` array)
        - render_context_json: str | dict
        - max_steps: int
        - allowed_mini_types: [str, ...]
      Outputs:
        - mini_steps_jsonl: str (JSONL) | list[dict] (one UI step per item)

    2) Legacy human-friendly format (back-compat):
      {
        "services_summary": str,
        "plan": [ { "key": str, "question": str, "type_hint"?: str }, ... ],
        "allowed_mini_types": [str, ...],
        "rendered_steps": [ { ...ui step... }, ... ]
      }
    """
    records: list[dict] = []
    for item in _load_examples_json():
        if not isinstance(item, dict):
            continue

        # Preferred explicit record: {"inputs": {...}, "outputs": {...}}
        if isinstance(item.get("inputs"), dict) and isinstance(item.get("outputs"), dict):
            inputs = dict(item["inputs"])
            outputs = dict(item["outputs"])

            # Normalize JSON payloads to strings.
            inputs["question_plan_json"] = _ensure_json_str(inputs.get("question_plan_json"))
            inputs["render_context_json"] = _ensure_json_str(inputs.get("render_context_json"))
            outputs["mini_steps_jsonl"] = _ensure_jsonl(outputs.get("mini_steps_jsonl"))

            # Fill max_steps if omitted (best-effort: prefer plan length).
            if "max_steps" not in inputs:
                try:
                    parsed = json.loads(str(inputs.get("question_plan_json") or "{}"))
                except Exception:
                    parsed = {}
                plan = parsed.get("plan") if isinstance(parsed, dict) else None
                n = len(plan) if isinstance(plan, list) else 0
                inputs["max_steps"] = int(n or 1)

            # Validate required fields.
            if not str(inputs.get("question_plan_json") or "").strip():
                continue
            if not str(inputs.get("render_context_json") or "").strip():
                continue
            if not outputs.get("mini_steps_jsonl"):
                continue

            records.append({"inputs": inputs, "outputs": outputs})
            continue

        services_summary = str(item.get("services_summary") or "").strip()
        plan = item.get("plan")
        allowed = item.get("allowed_mini_types")
        rendered_steps = item.get("rendered_steps")
        if not services_summary:
            continue
        if not isinstance(plan, list) or not plan:
            continue
        if not isinstance(rendered_steps, list) or not rendered_steps:
            continue

        allowed_types: list[str] = []
        if isinstance(allowed, list) and allowed:
            allowed_types = [str(x).strip() for x in allowed if str(x).strip()]
        if not allowed_types:
            allowed_types = ["multiple_choice", "segmented_choice", "chips_multi", "slider", "range_slider"]

        # Keep render context minimal and realistic; renderer prompt is strict about inputs.
        render_context = {"services_summary": services_summary}
        question_plan_json = _compact_json({"plan": plan})

        records.append(
            {
                "inputs": {
                    "question_plan_json": question_plan_json,
                    "render_context_json": _compact_json(render_context),
                    "max_steps": int(len(plan)),
                    "allowed_mini_types": list(allowed_types),
                },
                "outputs": {"mini_steps_jsonl": _jsonl(rendered_steps)},
            }
        )

    return as_dspy_examples(
        records,
        input_keys=["question_plan_json", "render_context_json", "max_steps", "allowed_mini_types"],
    )


__all__ = ["default_renderer_demos"]

