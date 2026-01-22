from __future__ import annotations

from typing import Any, Dict


QUESTION_HINTS: dict[str, dict[str, str]] = {
    "early": {"length": "short", "tone": "simple, broad"},
    "middle": {"length": "medium", "tone": "more specific, quantifying"},
    "late": {"length": "long", "tone": "detailed, pointed"},
}


def get_question_hints(stage: str) -> Dict[str, Any]:
    return dict(QUESTION_HINTS.get(str(stage or "").strip().lower(), QUESTION_HINTS["early"]))

