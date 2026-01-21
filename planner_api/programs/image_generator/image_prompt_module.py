from __future__ import annotations

import dspy


class _ImagePromptSignature(dspy.Signature):
    context_json: str = dspy.InputField(desc="Compact JSON context")
    batch_id: str = dspy.InputField(desc="Stable batch/phase identifier")

    prompt_json: str = dspy.OutputField(desc="A JSON object string matching ImagePromptSpec")


class ImagePromptModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.Predict(_ImagePromptSignature)

    def forward(self, *, context_json: str, batch_id: str):  # type: ignore[override]
        return self.prog(context_json=context_json, batch_id=batch_id)


__all__ = ["ImagePromptModule"]
