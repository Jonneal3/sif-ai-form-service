"""
DSPy Module wrapper for one-time form planning (FormPlannerJSON).
"""

from __future__ import annotations

from typing import Any

import dspy  # type: ignore

from app.signatures.json_signatures import FormPlannerJSON


class FormPlannerModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.Predict(FormPlannerJSON)

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        return self.prog(**kwargs)
