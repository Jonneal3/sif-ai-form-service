from __future__ import annotations

import dspy

from programs.form_pipeline.prompts import build_renderer_prompt


class RenderStepsJSONL(dspy.Signature):
    """
    Renderer signature.

    Prompt text lives in `programs.form_pipeline.prompts`.
    """

    question_plan_json: str = dspy.InputField(desc="Planner output JSON string containing `plan: [...]`.")
    render_context_json: str = dspy.InputField(desc="Compact JSON string with only rendering-relevant fields.")
    max_steps: int = dspy.InputField(desc="Maximum number of steps to emit")
    allowed_mini_types: list[str] = dspy.InputField(desc="Allowed UI step types")

    mini_steps_jsonl: str = dspy.OutputField(
        desc="JSONL string, one UI step per line. Output ONLY JSONL (no prose, no markdown, no code fences)."
    )


__all__ = ["RenderStepsJSONL"]

# Keep the signature file short: pull the prompt from the prompts folder.
RenderStepsJSONL.__doc__ = build_renderer_prompt()

