"""
DSPy Module wrapper for batch step generation (NextStepsJSONL).
"""

from __future__ import annotations

from typing import Any

import dspy  # type: ignore

from app.signatures.json_signatures import NextStepsJSONL


class BatchGeneratorModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.Predict(NextStepsJSONL)

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        return self.prog(**kwargs)
