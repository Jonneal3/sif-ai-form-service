from __future__ import annotations

import dspy

from programs.dspy_demos import as_dspy_examples, load_jsonl_records
from programs.form_pipeline.prompts import build_grounding_prompt


class GroundingSummarySignature(dspy.Signature):
    """
    Grounding summary signature.

    Prompt text lives in `programs.form_pipeline.prompts`.
    """

    grounding_context_json: str = dspy.InputField(
        desc=("Compact JSON string that includes industry/service, goal info, and any short memory.")
    )

    grounding_summary: str = dspy.OutputField(
        desc=(
            "Short grounding summary text. Plain text only. "
            "Keep it factual based on the input fields; do not invent detailed service facts."
        )
    )


# Keep the signature file short: pull the prompt from the prompts module.
GroundingSummarySignature.__doc__ = build_grounding_prompt()


class GroundingSummaryProgram(dspy.Module):
    """
    Thin DSPy wrapper for the grounding summary call.
    """

    def __init__(self, *, demo_pack: str = "") -> None:
        super().__init__()
        self.prog = dspy.Predict(GroundingSummarySignature)
        if demo_pack:
            demos = as_dspy_examples(
                load_jsonl_records(demo_pack),
                input_keys=["grounding_context_json"],
            )
            if demos:
                setattr(self.prog, "demos", demos)

    def forward(self, *, grounding_context_json: str):  # type: ignore[override]
        return self.prog(grounding_context_json=grounding_context_json)


__all__ = ["GroundingSummaryProgram", "GroundingSummarySignature"]

