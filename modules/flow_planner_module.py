"""
DSPy Module wrapper for the flow planner.

This keeps the runtime path aligned with DSPy’s Module pattern while delegating
schema validation and streaming to flow_planner.py.
"""

from __future__ import annotations

from typing import Any

import dspy  # type: ignore

from modules.signatures.json_signatures import MustHaveCopyJSON, NextStepsJSONL


class FlowPlannerModule(dspy.Module):
    """
    DSPy Module: wraps a single NextStepsJSONL predictor.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.Predict(NextStepsJSONL)
        self.copy_prog = dspy.Predict(MustHaveCopyJSON)

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        return self.prog(**kwargs)

    def generate_copy(self, **kwargs: Any) -> dspy.Prediction:
        return self.copy_prog(**kwargs)
