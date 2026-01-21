"""
DSPy Module wrapper for image prompt generation.

This keeps prompt construction (LLM) separate from image rendering (provider/tool).
"""

from __future__ import annotations

from typing import Any

import dspy  # type: ignore

from sif_ai_form_service.signatures.json_signatures import ImagePromptJSON


class ImagePromptModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.Predict(ImagePromptJSON)

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        return self.prog(**kwargs)
