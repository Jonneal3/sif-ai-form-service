from __future__ import annotations

import dspy


class NextStepsJSONL(dspy.Signature):
    """Generate the next UI steps as JSONL (one JSON object per line)."""

    context_json: str = dspy.InputField(desc="Compact JSON context for this request")
    batch_id: str = dspy.InputField(desc="Stable batch/phase identifier")
    max_steps: int = dspy.InputField(desc="Maximum number of steps to emit")
    allowed_mini_types: list[str] = dspy.InputField(desc="Allowed UI step types")

    mini_steps_jsonl: str = dspy.OutputField(desc="JSONL string, one UI step per line")
