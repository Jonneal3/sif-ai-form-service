from __future__ import annotations

import dspy


class _MustHaveCopySignature(dspy.Signature):
    context_json: str = dspy.InputField(desc="Compact JSON context")
    mini_steps_jsonl: str = dspy.InputField(desc="The UI steps JSONL for this batch")

    must_have_copy_json: str = dspy.OutputField(desc="JSON string of required copy fields")


class MustHaveCopyModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prog = dspy.Predict(_MustHaveCopySignature)

    def forward(self, *, context_json: str, mini_steps_jsonl: str):  # type: ignore[override]
        return self.prog(context_json=context_json, mini_steps_jsonl=mini_steps_jsonl)


__all__ = ["MustHaveCopyModule"]
