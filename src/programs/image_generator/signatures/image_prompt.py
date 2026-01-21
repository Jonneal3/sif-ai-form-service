from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class ImagePromptSpec(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    prompt: str = ""
    negative_prompt: str = Field(default="", alias="negativePrompt")
    style_tags: List[str] = Field(default_factory=list, alias="styleTags")
    metadata: Dict[str, Any] = Field(default_factory=dict)
