from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ImagePromptSpec(BaseModel):
    """Validated output for image prompt generation."""

    model_config = ConfigDict(populate_by_name=True)

    prompt: str = Field(..., description="Primary image generation prompt.")
    negative_prompt: str = Field(
        default="",
        alias="negativePrompt",
        description="Optional negative prompt to reduce undesired artifacts.",
    )
    style_tags: List[str] = Field(
        default_factory=list,
        alias="styleTags",
        description="Optional list of style tags (provider/model-agnostic).",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured metadata (aspect ratio, camera, constraints, etc).",
    )

