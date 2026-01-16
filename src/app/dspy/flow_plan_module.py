"""
DSPy Module wrapper for first-call flow plan generation (FlowPlanJSON).
"""

from __future__ import annotations

from typing import Any

import dspy  # type: ignore

from app.signatures.json_signatures import FlowPlanJSON


class FlowPlanModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.Predict(FlowPlanJSON)

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        return self.prog(**kwargs)

