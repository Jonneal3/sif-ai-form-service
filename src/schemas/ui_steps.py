from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class _UIStepBase(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str = Field(default="")
    type: str = Field(default="")
    title: Optional[str] = None
    question: Optional[str] = None
    subtext: Optional[str] = None
    required: Optional[bool] = None
    metric_gain: Optional[float] = Field(default=None, alias="metricGain")


class MiniOption(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    label: str = ""
    value: str = ""


class TextInputUI(_UIStepBase):
    type: Literal["text", "text_input"]


class IntroUI(_UIStepBase):
    type: Literal["intro"]


class RatingUI(_UIStepBase):
    type: Literal["rating"]


class DatePickerUI(_UIStepBase):
    type: Literal["date_picker"]


class ColorPickerUI(_UIStepBase):
    type: Literal["color_picker"]


class LeadCaptureUI(_UIStepBase):
    type: Literal["lead_capture"]


class PricingUI(_UIStepBase):
    type: Literal["pricing"]


class ConfirmationUI(_UIStepBase):
    type: Literal["confirmation"]


class DesignerUI(_UIStepBase):
    type: Literal["designer"]


class FileUploadUI(_UIStepBase):
    type: Literal["file_upload", "upload", "file_picker"]


class BudgetCardsUI(_UIStepBase):
    type: Literal["budget_cards"]


class MultipleChoiceUI(_UIStepBase):
    type: Literal[
        "multiple_choice",
        "choice",
        "segmented_choice",
        "chips_multi",
        "yes_no",
        "image_choice_grid",
    ]
    options: List[Union[MiniOption, str]] = Field(default_factory=list)


class SearchableSelectUI(_UIStepBase):
    type: Literal["searchable_select"]
    options: List[Union[MiniOption, str]] = Field(default_factory=list)


class CompositeUI(_UIStepBase):
    type: Literal["composite"]
    blocks: List[Dict[str, Any]] = Field(default_factory=list)


class GalleryUI(_UIStepBase):
    type: Literal["gallery"]
