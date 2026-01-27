"""
Prompt library used by DSPy signatures.

This module is intentionally "fixed" and reusable across programs.
"""

from __future__ import annotations

from typing import Iterable, List


CONTEXT_JSON_FIELDS = """`planner_context_json` typically includes:
- **Service context**: `services_summary` (primary), plus optional `industry` and `service`
- **State/memory**: `answered_qa` (list of {stepId, question, answer}), `asked_step_ids` (dedupe)
- **Hints/constraints** (hint-only; do not overfit):
  - `allowed_mini_types_hint`
  - `choice_option_min` / `choice_option_max` / `choice_option_target`
  - `batch_constraints` (e.g. min/max steps per batch, token budget)
  - `required_uploads`
""".strip()


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


def _section(*, title: str, body: str) -> str:
    return _lines(title.strip(), str(body or "").strip())


def _goal_and_instructions(*, who: str, instructions: str) -> str:
    return _lines("GOAL AND INSTRUCTIONS:", who.strip(), instructions.strip())


def _planner_goal_and_instructions() -> str:
    return _goal_and_instructions(
        who="You are the Form Planner.",
        instructions=(
            "Your job is to decide which questions to ask next, like a real designer/estimator.\n"
            "You are vertical-agnostic: your approach should work for any industry/service.\n"
            "The examples you see may include other industries—do NOT copy an industry's specifics unless the current `services_summary` calls for it.\n"
            "Generalize across industries: keep the intake structure consistent, but tailor the question content to the current service.\n"
            "Ask the minimum set of questions that meaningfully reduces uncertainty.\n"
            "Prefer high-signal questions that drive scope, cost, feasibility, and timeline.\n"
            "Use memory (`answered_qa`, `asked_step_ids`) to avoid repeats and stay consistent.\n"
            "Use constraints/hints (allowed types, option targets, batch constraints, required uploads) as guidance, not rigid requirements.\n"
            "You do NOT output UI steps. You output a plan (keys + intent) for what to ask next."
        ),
    )


def build_planner_prompt() -> str:
    return _lines(
        "Create a question plan (NOT UI steps).",
        _planner_goal_and_instructions(),
        _section(title="CONTEXT FIELDS:", body=CONTEXT_JSON_FIELDS),
        _bullets(
            "INPUTS:",
            [
                "`planner_context_json`: compact JSON with service + memory + constraints (see above).",
                "`max_steps`: maximum number of plan items to emit.",
                "`allowed_mini_types`: allowed UI step types (hint only; do not overfit).",
            ],
        ),
        _bullets(
            "HARD RULES:",
            [
                "Output MUST be JSON only (no prose, no markdown, no code fences) in `question_plan_json`.",
                "Return at most `max_steps` plan items.",
                "Do NOT repeat already asked steps (use `answered_qa[].stepId` and/or `asked_step_ids` when provided).",
                "Do NOT invent step ids. Only output `key`. The renderer will assign `id = step-<key>`.",
                "Each plan item MUST include a user-facing `question` string (what the user will see).",
                "`question` must be direct + concrete (no 'Ask user...' / meta-instructions).",
                "Use `services_summary` to keep questions/wording relevant; avoid invented facts.",
                "Avoid overly-generic buckets unless unavoidable (e.g. 'Basic/Mid/High/Luxury').",
                "For multi-select lists, keep options tightly relevant (don’t mix unrelated categories).",
            ],
        ),
    )


__all__ = ["CONTEXT_JSON_FIELDS", "build_planner_prompt"]

