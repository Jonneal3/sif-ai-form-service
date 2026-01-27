from __future__ import annotations

import dspy

from programs.form_pipeline.prompt_library import build_planner_prompt


class QuestionPlannerSignature(dspy.Signature):
    """
    Planner signature.

    Prompt text lives in `programs.form_pipeline.prompt_library`.
    """

    planner_context_json: str = dspy.InputField(
        desc="Compact JSON string with service context + memory (services_summary, industry/service, answered_qa, asked_step_ids)."
    )
    max_steps: int = dspy.InputField(desc="Maximum number of plan items to emit")
    allowed_mini_types: list[str] = dspy.InputField(desc="Allowed UI step types (hint only for planning)")
    question_plan_json: str = dspy.OutputField(desc="JSON string only. Must be an object with a top-level `plan` array.")

__all__ = ["QuestionPlannerSignature"]

# Keep the signature file short: pull the prompt from the prompts folder.
QuestionPlannerSignature.__doc__ = build_planner_prompt()

