from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional

from .form_pipeline import next_steps_jsonl, stream_next_steps_jsonl
from .image_pipeline import build_image_prompt

__all__ = [
    "run_form",
    "run_form_stream",
    "run_image",
    "next_steps_jsonl",
    "stream_next_steps_jsonl",
    "build_image_prompt",
]


def run_form(payload: Dict[str, Any]) -> Dict[str, Any]:
    return next_steps_jsonl(payload)


async def run_form_stream(payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
    async for event in stream_next_steps_jsonl(payload):
        yield event


def run_image(payload: Dict[str, Any], *, prompt_template: Optional[str] = None) -> Dict[str, Any]:
    return build_image_prompt(payload, prompt_template=prompt_template)
