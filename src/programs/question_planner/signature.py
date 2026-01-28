from __future__ import annotations

import dspy

from programs.question_planner.prompt_library import build_planner_prompt


class QuestionPlannerSignature(dspy.Signature):
    """
    Planner signature.

    Prompt text lives in `programs.question_planner.prompt_library`.
    """

    planner_context_json: str = dspy.InputField(
        desc="JSON string ONLY. Service context + memory (services_summary, industry/service, answered_qa, asked_step_ids, batch_constraints, etc.)."
    )
    max_steps: int = dspy.InputField(desc="Maximum number of plan items to emit.")
    allowed_mini_types: list[str] = dspy.InputField(desc="Allowed UI step types (hint only; renderer will enforce).")
    question_plan_json: str = dspy.OutputField(
        desc=(
            "JSON string ONLY. Must be an object with top-level `plan` array. Each plan item must include:\n"
            "- key: string (snake_case)\n"
            "- question: string (user-facing)\n"
            "Optional fields to help the renderer:\n"
            "- type_hint: string (e.g. multiple_choice, chips_multi, slider)\n"
            "- option_hints: array of strings OR array of {label, value?} (for choice steps)\n"
            "- range_hints: {min?, max?, step?, unit?, currency?} (for slider/range_slider steps)\n"
        )
    )

__all__ = ["QuestionPlannerSignature"]

# Keep the signature file short: pull the prompt from the prompts folder.
QuestionPlannerSignature.__doc__ = build_planner_prompt()

