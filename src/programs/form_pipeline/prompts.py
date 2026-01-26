"""
Prompt building helpers used by DSPy signatures.

This is a flat replacement for the old `prompts/blocks/*` and `prompts/builders/*` files.
"""

from __future__ import annotations


CONTEXT_JSON_FIELDS = """`context_json` typically includes:
- **Platform/goal**: `platform_goal`, `business_context`, `goal_intent`
- **Vertical context**: `industry`, `service`, optional `grounding_summary` / `vertical_context`
- **Use case**: `use_case` (e.g. scene/tryon) and any stage hints under `flow_guide`
- **State/memory**: `known_answers`, `asked_step_ids` / `already_asked_keys`, optional `answered_qa`
- **Constraints**: `batch_constraints` (e.g. min/max steps per batch, token budget)
- **Uploads**: `required_uploads` (if any) and upload-related policies
- **Copy/style**: `copy_style` / `copy_context` (tone + question rules)
"""

NEXT_STEPS_HARD_RULES = """HARD RULES:
- Output MUST be JSONL only (one JSON object per line) in `mini_steps_jsonl`.
- Each line MUST be a valid UI step object for the API (id/type/question/required/etc).
- Ids must be deterministic and stable: use `step-<key>` style ids.
- Respect `max_steps` and `allowed_mini_types` exactly. Never emit a type not in `allowed_mini_types`.
- Use `already_asked_keys` / `asked_step_ids` to avoid repeats.
- Use `items` / `form_plan` and `attribute_families` when available to stay on-target (don’t invent random keys).
- Use `grounding_summary` to keep options relevant; avoid invented facts.
- Avoid generic filler options (e.g., Option A/B/C, Category 1/2/3, red/blue/green).
- For choice questions, prefer 4–10 options (vary the count; don’t always output 3).
- Keep question copy chill and clear: short, concrete, one thing at a time; follow `copy_style` when present.
"""

PLANNER_HARD_RULES = """HARD RULES:
- Output MUST be JSON only (no prose, no markdown, no code fences) in `question_plan_json`.
- Return at most `max_steps` plan items.
- Do NOT repeat already asked steps.
  A key is considered already asked if `step-` + key with `_` replaced by `-` is in `asked_step_ids`.
- Do NOT invent step ids. Only output `key`. The renderer will assign `id = step-<key>`.
"""

RENDERER_HARD_RULES = """HARD RULES:
- Output MUST be JSONL only (one JSON object per line) in `mini_steps_jsonl`.
- Do not include prose, markdown, or code fences.
- Do NOT invent new plan items, steps, or keys. Only render items from `plan[]`.
- If a plan item includes `type_hint`, you MUST set the output step `type` to that exact value.
- Deterministic ids: `id = "step-" + key.replace("_","-")`.
- Do NOT output a step if its id is already in `asked_step_ids`.
- Respect `max_steps` exactly.
- For choice types, include options:
  - Prefer `option_hints` from the plan when present.
  - Otherwise generate grounded options within `choice_option_*` constraints.
"""

GROUNDING_HARD_RULES = """HARD RULES:
- Output MUST be plain text only (no JSON, no markdown).
- Keep it short (a few sentences).
- Use only the provided input context. Do not invent detailed service facts.
"""


NEXT_STEPS_ROLE_AND_GOAL = """ROLE AND GOAL:
You are an expert intake agent (think: a great designer + estimator) generating the next best
set of questions to ask. Your job is to select the minimum set of questions that maximizes
downstream success for the given `platform_goal` / `goal_intent`, while staying aligned to
the specific service context (industry/service + grounding_summary).
"""

PLANNER_ROLE_AND_GOAL = """ROLE AND GOAL:
You are the Form Planner.
Your job is to decide which questions to ask next, like a real designer/estimator.
You do NOT output UI steps. You output a plan (keys + intent).
"""

RENDERER_ROLE_AND_GOAL = """ROLE AND GOAL:
You are the Step Renderer.
Your job is to convert a question plan into valid UI steps for the frontend.
"""

GROUNDING_ROLE_AND_GOAL = """ROLE AND GOAL:
You write a short grounding summary for the current service.
This summary helps the next model call stay on-topic.
"""


def build_planner_prompt() -> str:
    return "\n".join(
        [
            "Create a question plan (NOT UI steps).",
            "",
            PLANNER_ROLE_AND_GOAL.strip(),
            "",
            PLANNER_HARD_RULES.strip(),
            "",
        ]
    ).strip() + "\n"


def build_renderer_prompt() -> str:
    return "\n".join(
        [
            "Render a given question plan into strict JSONL UI steps.",
            "",
            RENDERER_ROLE_AND_GOAL.strip(),
            "",
            RENDERER_HARD_RULES.strip(),
            "",
        ]
    ).strip() + "\n"


def build_grounding_prompt() -> str:
    return "\n".join(
        [
            "Create a short grounding summary for the current service.",
            "",
            GROUNDING_ROLE_AND_GOAL.strip(),
            "",
            GROUNDING_HARD_RULES.strip(),
            "",
        ]
    ).strip() + "\n"


__all__ = [
    "CONTEXT_JSON_FIELDS",
    "NEXT_STEPS_HARD_RULES",
    "PLANNER_HARD_RULES",
    "RENDERER_HARD_RULES",
    "GROUNDING_HARD_RULES",
    "NEXT_STEPS_ROLE_AND_GOAL",
    "PLANNER_ROLE_AND_GOAL",
    "RENDERER_ROLE_AND_GOAL",
    "GROUNDING_ROLE_AND_GOAL",
    "build_planner_prompt",
    "build_renderer_prompt",
    "build_grounding_prompt",
]

