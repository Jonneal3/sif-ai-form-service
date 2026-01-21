from __future__ import annotations

import dspy

from programs.batch_generator.signatures.next_steps_jsonl import BatchNextStepsJSONL


class BatchStepsModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.Predict(BatchNextStepsJSONL)

    def forward(self, *, context_json: str, max_steps: int, allowed_mini_types: list[str]):  # type: ignore[override]
        return self.prog(
            context_json=context_json,
            max_steps=max_steps,
            allowed_mini_types=allowed_mini_types,
        )


__all__ = ["BatchStepsModule"]
