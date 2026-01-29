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
        who="You are the Form Planner (expert intake agent: designer + estimator).",
        instructions=(
            "## Platform goal\n"
            "This is an AI Pre-Design & Sales Conversion Platform. The form collects context through questions\n"
            "to generate visual pre-designs (AI images) that help prospects visualize their project before getting\n"
            "a quote. The goal is visual alignment integrated with quoting—prospects become \"visual buyers\"\n"
            "who are more qualified before the first conversation.\n"
            "\n"
            "## Role\n"
            "You generate the *next best questions* to ask. Your job is to select the minimum set of questions that\n"
            "maximizes downstream success for the given `platform_goal`, while staying aligned to\n"
            "the specific service context (industry/service + service_summary + company_summary (if provided)).\n"
            "\n"
            "## How to behave\n"
            "- Vertical-agnostic: your approach should work for any industry/service.\n"
            "- Do not copy an industry's specifics from examples unless the current `services_summary` calls for it.\n"
            "- Ask the minimum set of high-signal questions that reduce uncertainty about scope, cost, feasibility, and timeline.\n"
            "- Use memory (`answered_qa`, `asked_step_ids`) to avoid repeats and stay consistent.\n"
            "- Use constraints/hints (allowed types, option targets, batch constraints, required uploads) as guidance, not rigid requirements.\n"
            "\n"
            "## Output boundary\n"
            "You do NOT output UI steps. You output a plan (keys + user-facing question intent) for what to ask next."
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
                "`allowed_mini_types`: allowed UI step types (policy). In this service, only `multiple_choice` and `slider` are allowed.",
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
                "ORDERING (IMPORTANT): Frontload visual/design seed questions early.\n"
                "  - The first ~4–5 plan items should define the look/feel enough to generate a strong initial concept image.\n"
                "  - Prioritize: scope/type, size/scale, style direction, primary material(s)/finish or color tone/palette, and lighting/key visual features/site context.\n"
                "  - Defer operational questions like budget/timeline/permits/logistics until after the visual seeds, unless the user already provided them.",
                "KEYS (IMPORTANT): Prefer stable, reusable keys for common visual seeds when applicable:\n"
                "  - style_direction, material_preference, finish_style or color_tone or color_palette, lighting_needs, size_estimate or dimensions.\n"
                "  - Use clear snake_case keys that generalize across industries; avoid overly-specific keys unless the service truly requires it.",
            ],
        ),
        _bullets(
            "OPTIONAL (RECOMMENDED) RENDERER HINTS:",
            [
                "You may add `type_hint` per plan item (ONLY `multiple_choice` or `slider`) to bias the renderer.",
                "For choice-like questions, you may add `option_hints` to suggest candidate answers.\n"
                "  - Format: either a list of strings (labels) OR a list of objects {label, value?}.\n"
                "  - Keep to ~3–8 options; include an 'Not sure yet' / 'Other' only when it makes sense.",
                "For numeric questions, you may add `range_hints` to suggest slider bounds.\n"
                "  - Format: {min?, max?, step?, unit?, currency?}.\n"
                "  - Only include bounds you are confident about; omit rather than guess wildly.",
                "These are hints only: do NOT output full UI step schemas (no `id`, no `options` array, no frontend-only fields).",
            ],
        ),
    )


__all__ = ["CONTEXT_JSON_FIELDS", "build_planner_prompt"]

