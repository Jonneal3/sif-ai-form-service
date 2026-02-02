from __future__ import annotations

import json

import dspy

from programs.dspy_demos import as_dspy_examples, load_jsonl_records
from programs.question_planner.data.examples.loader import default_design_demos
from programs.question_planner.signature import QuestionPlannerSignature

def _extract_first_json_object(text: str) -> dict | None:
    s = str(text or "")
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = s[start : i + 1]
                try:
                    parsed = json.loads(blob)
                except Exception:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _extract_first_plan_json_object(text: str) -> dict | None:
    """
    Best-effort: scan for the first *complete* JSON object that contains a top-level `plan` list.

    Some models occasionally echo input/context JSON before emitting the plan; extracting the first
    JSON object blindly can select the wrong payload. This helper prefers the first object that
    actually matches the contract shape we need.
    """
    s = str(text or "")
    if not s:
        return None

    i = 0
    while True:
        start = s.find("{", i)
        if start < 0:
            return None

        depth = 0
        in_str = False
        esc = False
        for j in range(start, len(s)):
            ch = s[j]
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blob = s[start : j + 1]
                    try:
                        parsed = json.loads(blob)
                    except Exception:
                        break
                    if isinstance(parsed, dict) and isinstance(parsed.get("plan"), list):
                        return parsed
                    break

        i = start + 1


def _compact_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sanitize_question_plan_json(raw: object) -> str:
    """
    Best-effort normalization for `question_plan_json`.

    DSPy / the LM sometimes returns extra markers/prose around the JSON. This keeps runtime
    and demo-packs stable by extracting + compacting the first JSON object when possible.
    """
    text = str(raw or "").strip()
    if not text:
        return ""

    parsed: dict | None = None
    try:
        loaded = json.loads(text)
        if isinstance(loaded, list):
            parsed = {"plan": loaded}
        elif isinstance(loaded, dict) and isinstance(loaded.get("plan"), list):
            parsed = loaded
    except Exception:
        parsed = _extract_first_plan_json_object(text) or _extract_first_json_object(text)

    if not isinstance(parsed, dict) or not isinstance(parsed.get("plan"), list):
        return text

    plan = parsed.get("plan")
    if isinstance(plan, list):
        for item in plan:
            if not isinstance(item, dict):
                continue

            # Normalize common field aliases.
            if "type_hint" not in item and "typeHint" in item:
                item["type_hint"] = item.pop("typeHint")
            if "allow_multiple" not in item and "allowMultiple" in item:
                item["allow_multiple"] = item.pop("allowMultiple")
            if "allow_other" not in item and "allowOther" in item:
                item["allow_other"] = item.pop("allowOther")

            if "option_hints" not in item:
                if "optionHints" in item:
                    item["option_hints"] = item.pop("optionHints")
                elif "answer_hints" in item:
                    item["option_hints"] = item.pop("answer_hints")
                elif "options" in item:
                    item["option_hints"] = item.pop("options")

    return _compact_json(parsed)

class QuestionPlannerProgram(dspy.Module):
    """
    Thin DSPy wrapper for the planner call.
    """

    def __init__(self, *, demo_pack: str = "") -> None:
        super().__init__()
        self.prog = dspy.Predict(QuestionPlannerSignature)
        demos = []
        if demo_pack:
            demos = as_dspy_examples(
                load_jsonl_records(demo_pack),
                input_keys=[
                    "planner_context_json",
                    "max_steps",
                    "allowed_mini_types",
                ],
            )
        if not demos:
            demos = default_design_demos()
        if demos:
            setattr(self.prog, "demos", demos)

    def forward(  # type: ignore[override]
        self,
        *,
        planner_context_json: str,
        max_steps: int,
        allowed_mini_types: list[str],
    ):
        pred = self.prog(
            planner_context_json=planner_context_json,
            max_steps=max_steps,
            allowed_mini_types=allowed_mini_types,
        )
        try:
            cleaned = _sanitize_question_plan_json(getattr(pred, "question_plan_json", None))
            if cleaned:
                setattr(pred, "question_plan_json", cleaned)
        except Exception:
            pass
        return pred


__all__ = ["QuestionPlannerProgram"]
