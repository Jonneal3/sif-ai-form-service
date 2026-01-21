from __future__ import annotations

import dspy

from programs.form_planner.signatures.form_plan_signatures import FormPlanJSON


class FormPlanModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.Predict(FormPlanJSON)

    def forward(self, *, context_json: str, current_phase_id: str):  # type: ignore[override]
        return self.prog(context_json=context_json, current_phase_id=current_phase_id)


__all__ = ["FormPlanModule"]

