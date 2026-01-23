from __future__ import annotations

import json
from typing import Any, Mapping

import dspy

from programs.batch_generator.signatures.json_signatures import BatchGeneratorJSON


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True, sort_keys=True)


class BatchGeneratorModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.Predict(BatchGeneratorJSON)

    def forward(  # type: ignore[override]
        self,
        *,
        context_json: str | None = None,
        context: Mapping[str, Any] | None = None,
    ):
        """
        Accepts either a pre-serialized `context_json` or a structured `context` mapping.

        Keeping the DSPy signature stable (single `context_json` blob) avoids schema churn while still
        letting callers work with dicts at runtime.
        """
        if context_json is None:
            if context is None:
                raise TypeError("Provide either `context_json` or `context`.")
            context_json = _compact_json(context)
        return self.prog(context_json=context_json)


__all__ = ["BatchGeneratorModule"]
