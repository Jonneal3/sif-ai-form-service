from __future__ import annotations

import dspy


class BatchNextStepsJSONL(dspy.Signature):
    """
    Generate the next UI steps as JSONL (one JSON object per line).

    This signature is the batch generator: it receives a single JSON blob (`context_json`) that contains
    everything the model needs to generate the next batch of UI steps (questions/options).

    ROLE AND GOAL:
    You are an expert intake agent (think: a great designer + estimator) generating the next best
    set of questions to ask. Your job is to select the minimum set of questions that maximizes
    downstream success for the given `platform_goal` / `goal_intent`, while staying aligned to
    the specific service context (industry/service + grounding_summary).

    HARD RULES:
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

    `context_json` typically includes:
    - **Platform/goal**: `platform_goal`, `business_context`, `goal_intent`
    - **Vertical context**: `industry`, `service`, optional `grounding_summary` / `vertical_context`
    - **Use case**: `use_case` (e.g. scene/tryon) and any stage hints under `flow_guide`
    - **State/memory**: `known_answers`, `asked_step_ids` / `already_asked_keys`, optional `answered_qa`
    - **Constraints**: `batch_constraints` (e.g. min/max steps per batch, token budget), plus per-call `max_steps`
    - **Uploads**: `required_uploads` (if any) and upload-related policies
    - **Copy/style**: `copy_style` / `copy_context` (tone + question rules), used to shape wording

    Output rules:
    - Output MUST be JSONL only (one JSON object per line) in `mini_steps_jsonl`.
    - Do not include prose, markdown, or code fences.

    Copy / quality controls (when present in `context_json`):
    - `copy_style`: JSON snippet with tone + phrasing rules.
    - `grounding_summary`: use this as your anchor; avoid invented vertical facts.
    - Avoid overly-specific location option lists unless the service is clearly outdoor-specific; prefer
      cross-vertical choices like inside/outside/multiple/not sure.
    """

    context_json: str = dspy.InputField(
        desc=(
            "Compact JSON context for this request (serialized as a string). "
            "Includes platform_goal/business_context/goal_intent, industry/service + grounding, current state "
            "(known_answers + already_asked_keys/asked_step_ids + answered_qa), constraints/flow guide, and copy_style."
        )
    )
    max_steps: int = dspy.InputField(desc="Maximum number of steps to emit")
    allowed_mini_types: list[str] = dspy.InputField(desc="Allowed UI step types")

    mini_steps_jsonl: str = dspy.OutputField(
        desc=(
            "JSONL string, one UI step per line. Output ONLY JSONL (no prose, no markdown, no code fences). "
            "Each line must be a valid JSON object. Prefer deterministic ids (`step-...`), ask one thing at a time, "
            "and choose concrete options grounded in `grounding_summary` when provided."
        )
    )


__all__ = ["BatchNextStepsJSONL"]

