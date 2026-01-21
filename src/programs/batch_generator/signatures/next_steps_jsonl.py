from __future__ import annotations

import dspy


class BatchNextStepsJSONL(dspy.Signature):
    """
    Generate the next UI steps as JSONL (one JSON object per line).

    This signature is the "batch generator" layer: it assumes `context_json` already contains any
    form planning outputs (e.g., `form_plan`, `batch_policy`) plus the current form state and constraints.

    In particular, the orchestrator should include:
    - `batch_phase_id`: which phase we're generating now
    - `batch_phase_policy`: the definition for that phase (purpose/focus/limits), derived from the plan
    """

    context_json: str = dspy.InputField(
        desc=(
            "Compact JSON context for this request (includes `form_plan` and `batch_policy` when available). "
            "Must also include the current phase id, typically as `batch_phase_id`."
        )
    )
    max_steps: int = dspy.InputField(desc="Maximum number of steps to emit")
    allowed_mini_types: list[str] = dspy.InputField(desc="Allowed UI step types")

    mini_steps_jsonl: str = dspy.OutputField(desc="JSONL string, one UI step per line")


__all__ = ["BatchNextStepsJSONL"]
