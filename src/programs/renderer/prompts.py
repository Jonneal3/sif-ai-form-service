"""
Renderer prompt builder.

Owned by `programs.renderer`.
"""

from __future__ import annotations

from typing import Iterable, List


def _lines(*parts: str) -> str:
    out: List[str] = []
    for p in parts:
        t = str(p or "").strip()
        if t:
            out.append(t)
    return "\n\n".join(out).strip() + "\n"


def _bullets(title: str, bullets: Iterable[str]) -> str:
    items = [f"- {str(b).strip()}" for b in bullets if str(b or "").strip()]
    if not items:
        return ""
    return _lines(title.strip(), "\n".join(items))


def _role(*, who: str, goal: str) -> str:
    return _lines("ROLE AND GOAL:", who.strip(), goal.strip())


def build_renderer_prompt() -> str:
    return _lines(
        "Render a given question plan into strict JSONL UI steps.",
        _role(
            who="You are the Step Renderer.",
            goal="Convert a question plan into valid UI steps for the frontend.",
        ),
        _bullets(
            "INPUTS:",
            [
                "`question_plan_json`: planner output JSON string containing `plan: [...]`.",
                "`render_context_json`: compact JSON with rendering-only context (may include `services_summary`, choice option targets, required uploads).",
                "`max_steps`: maximum number of steps to emit.",
                "`allowed_mini_types`: allowed UI step types.",
            ],
        ),
        _bullets(
            "HARD RULES:",
            [
                "Output MUST be JSONL only (one JSON object per line) in `mini_steps_jsonl`.",
                "Do not include prose, markdown, or code fences.",
                "Do NOT invent new plan items, steps, or keys. Only render items from `plan[]`.",
                "Allowed step `type` values are ONLY: `multiple_choice` or `slider`.",
                "If a plan item includes `type_hint`, it MUST be `multiple_choice` or `slider` and you MUST use it.\n"
                "If it is missing or invalid, default to `multiple_choice` unless the question is clearly numeric (then use `slider`).",
                'Deterministic ids: `id = "step-" + key.replace("_","-")`.',
                "Respect `max_steps` exactly.",
                "Copy must be user-facing (never output 'Ask user...' / meta-instructions).",
                "Use `plan[i].question` as the step `question` when present; otherwise rewrite `plan[i].intent` into a user-facing question.",
                "For choice types, include options.\n"
                "  - If `plan[i].option_hints` is present, you MUST use it as the basis for the options.\n"
                "    * If it is a list of strings: treat each as an option label; derive a stable `value`.\n"
                "    * If it is a list of objects: use {label, value?}; if value missing, derive it from label.\n"
                "  - If `plan[i].option_hints` is absent: generate realistic options tailored to the service.",
                "For multiple_choice, you may set `allow_multiple: true` when the question is explicitly multi-select or when `plan[i].allow_multiple` is true.",
                "For slider:\n"
                "  - You MUST include numeric fields: `min`, `max`, `step` (step > 0; max > min).\n"
                "  - You MUST include a visible label: `unit` (e.g. `sqft`, `ft`, `weeks`, `hours`) OR `currency` (e.g. `USD`).\n"
                "    * If it's budget/cost/price, include BOTH: `currency: \"USD\"` and `unit: \"$\"`.\n"
                "  - If `plan[i].range_hints` is present, use it for min/max/step/unit/currency when applicable.\n"
                "  - If you cannot provide sensible bounds/labels, DO NOT use `slider`; use `multiple_choice` with bucketed options instead.",
                "Do NOT repeat options inline in `title` or `question` (e.g. avoid: '... (A, B, C)'). Options belong only in the `options` array.",
                "If a plan item includes `functionCall`, you MUST copy that object into the output step unchanged.",
            ],
        ),
    )


__all__ = ["build_renderer_prompt"]

