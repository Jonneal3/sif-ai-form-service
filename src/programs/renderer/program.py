from __future__ import annotations

import dspy

from programs.dspy_demos import as_dspy_examples, load_jsonl_records
from programs.renderer.signature import RenderStepsJSONL

class RendererProgram(dspy.Module):
    """
    Thin DSPy wrapper for the renderer call.
    """

    def __init__(self, *, demo_pack: str = "") -> None:
        super().__init__()
        self.prog = dspy.Predict(RenderStepsJSONL)
        if demo_pack:
            demos = as_dspy_examples(
                load_jsonl_records(demo_pack),
                input_keys=[
                    "question_plan_json",
                    "render_context_json",
                    "max_steps",
                    "allowed_mini_types",
                ],
            )
            if demos:
                setattr(self.prog, "demos", demos)

    def forward(  # type: ignore[override]
        self,
        *,
        question_plan_json: str,
        render_context_json: str,
        max_steps: int,
        allowed_mini_types: list[str],
    ):
        return self.prog(
            question_plan_json=question_plan_json,
            render_context_json=render_context_json,
            max_steps=max_steps,
            allowed_mini_types=allowed_mini_types,
        )


__all__ = ["RendererProgram"]

