"""
High-level DSPy wrapper composing the planning/generation/copy programs.

This module is intended to reflect the runtime architecture:
1) Form planner (one-time)
2) Batch generator (next steps)
3) Must-have copy generator
"""

from __future__ import annotations

from typing import Any

import dspy  # type: ignore

from app.dspy.batch_generator_module import BatchGeneratorModule
from app.dspy.form_planner_module import FormPlannerModule
from app.dspy.must_have_copy_module import MustHaveCopyModule


class FlowPlannerModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_generator = BatchGeneratorModule()
        self.form_planner = FormPlannerModule()
        self.must_have_copy = MustHaveCopyModule()

        # Preserve attribute names used elsewhere in the repo.
        self.prog = self.batch_generator.prog
        self.copy_prog = self.must_have_copy.prog
        self.plan_prog = self.form_planner.prog

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        return self.batch_generator(**kwargs)

    def generate_copy(self, **kwargs: Any) -> dspy.Prediction:
        return self.must_have_copy(**kwargs)

    def plan_form(self, **kwargs: Any) -> dspy.Prediction:
        return self.form_planner(**kwargs)
