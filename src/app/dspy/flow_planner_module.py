"""
High-level DSPy wrapper composing the planning/generation/copy programs.

This module is intended to reflect the runtime architecture:
1) Flow plan (first-call plan JSON)
2) Batch generator (next steps)
3) Must-have copy generator
"""

from __future__ import annotations

from typing import Any

import dspy  # type: ignore

from app.dspy.batch_generator_module import BatchGeneratorModule
from app.dspy.must_have_copy_module import MustHaveCopyModule
from app.signatures.json_signatures import FlowPlanJSON


class FlowPlanModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.Predict(FlowPlanJSON)

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        return self.prog(**kwargs)


class FlowPlannerModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_generator = BatchGeneratorModule()
        self.must_have_copy = MustHaveCopyModule()

        # Preserve attribute names used elsewhere in the repo.
        self.prog = self.batch_generator.prog
        self.copy_prog = self.must_have_copy.prog

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        return self.batch_generator(**kwargs)

    def generate_copy(self, **kwargs: Any) -> dspy.Prediction:
        return self.must_have_copy(**kwargs)
