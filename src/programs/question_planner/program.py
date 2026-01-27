from __future__ import annotations

import dspy

from programs.dspy_demos import as_dspy_examples, load_jsonl_records
from programs.question_planner.examples.loader import default_design_demos
from programs.question_planner.signature import QuestionPlannerSignature

class QuestionPlannerProgram(dspy.Module):
    """
    Thin DSPy wrapper for the planner call.
    """

    def __init__(self, *, demo_pack: str = "") -> None:
        super().__init__()
        self.prog = dspy.Predict(QuestionPlannerSignature)
        demos = []
        if demo_pack:
            demos = as_dspy_examples(
                load_jsonl_records(demo_pack),
                input_keys=[
                    "planner_context_json",
                    "max_steps",
                    "allowed_mini_types",
                ],
            )
        if not demos:
            demos = default_design_demos()
        if demos:
            setattr(self.prog, "demos", demos)

    def forward(  # type: ignore[override]
        self,
        *,
        planner_context_json: str,
        max_steps: int,
        allowed_mini_types: list[str],
    ):
        return self.prog(
            planner_context_json=planner_context_json,
            max_steps=max_steps,
            allowed_mini_types=allowed_mini_types,
        )


__all__ = ["QuestionPlannerProgram"]

