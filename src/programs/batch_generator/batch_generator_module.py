from __future__ import annotations

import dspy

from programs.batch_generator.signatures.json_signatures import BatchGeneratorJSON


class BatchGeneratorModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.Predict(BatchGeneratorJSON)

    def forward(self, *, context_json: str, batch_id: str):  # type: ignore[override]
        return self.prog(context_json=context_json, batch_id=batch_id)


__all__ = ["BatchGeneratorModule"]
