from __future__ import annotations

import dspy

from programs.form_planner.signatures.json_signatures import NextStepsJSONL


class FlowPlannerModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.Predict(NextStepsJSONL)

    def forward(self, *, context_json: str, batch_id: str, max_steps: int, allowed_mini_types: list[str]):  # type: ignore[override]
        return self.prog(
            context_json=context_json,
            batch_id=batch_id,
            max_steps=max_steps,
            allowed_mini_types=allowed_mini_types,
        )


__all__ = ["FlowPlannerModule"]
