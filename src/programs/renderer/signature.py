from __future__ import annotations

import dspy

from programs.renderer.prompts import build_renderer_prompt


class RenderStepsJSONL(dspy.Signature):
    """
    Renderer signature.

    Prompt text lives in `programs.renderer.prompts`.
    """

    question_plan_json: str = dspy.InputField(
        desc="JSON string ONLY. Must be an object with top-level `plan: [...]`. Each plan item must include at least {key, question} and may include type_hint/required."
    )
    render_context_json: str = dspy.InputField(
        desc="JSON string ONLY. Rendering context (e.g. services_summary, choice_option_* constraints, required_uploads)."
    )
    max_steps: int = dspy.InputField(desc="Maximum number of UI steps to emit (should match plan length for this call).")
    allowed_mini_types: list[str] = dspy.InputField(desc="Allowed UI step types for emitted steps.")
    mini_steps_jsonl: str = dspy.OutputField(
        desc="JSONL string ONLY: one validated UI step object per line. No prose, no markdown, no code fences."
    )


__all__ = ["RenderStepsJSONL"]

# Keep the signature file short: pull the prompt from the prompts folder.
RenderStepsJSONL.__doc__ = build_renderer_prompt()

