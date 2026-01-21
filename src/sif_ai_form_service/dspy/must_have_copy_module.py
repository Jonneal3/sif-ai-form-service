"""
DSPy Module wrapper for required copy generation (MustHaveCopyJSON).
"""

from __future__ import annotations

from typing import Any

import dspy  # type: ignore

from sif_ai_form_service.signatures.json_signatures import MustHaveCopyJSON


class MustHaveCopyModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.Predict(MustHaveCopyJSON)

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        return self.prog(**kwargs)
