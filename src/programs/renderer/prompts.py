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
                "If a plan item includes `type_hint`, you MUST set the output step `type` to that exact value.",
                'Deterministic ids: `id = "step-" + key.replace("_","-")`.',
                "Respect `max_steps` exactly.",
                "Copy must be user-facing (never output 'Ask user...' / meta-instructions).",
                "Use `plan[i].question` as the step `question` when present; otherwise rewrite `plan[i].intent` into a user-facing question.",
                "For choice types, include options (use `option_hints` when present; otherwise generate realistic options).",
            ],
        ),
    )


__all__ = ["build_renderer_prompt"]

