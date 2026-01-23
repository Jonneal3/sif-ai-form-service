from __future__ import annotations

from typing import List


FLOW_COMPONENTS: dict[str, list[str]] = {
    # Early = easiest, mostly structured.
    "early": ["multiple_choice"],
    # Middle = add quantifiers/controls.
    "middle": ["multiple_choice", "yes_no", "slider", "range_slider"],
    # Late = allow detail and uploads.
    "late": ["multiple_choice", "yes_no", "slider", "range_slider", "file_upload"],
}


def allowed_components(stage: str) -> List[str]:
    return list(FLOW_COMPONENTS.get(str(stage or "").strip().lower(), FLOW_COMPONENTS["early"]))


__all__ = ["allowed_components", "FLOW_COMPONENTS"]

