from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    # Back-compat: the widget contract historically treated these as the same
    # "numeric control" family and the model often emits `slider` / `range_slider`.
    type: Literal["rating"]
    scale_min: float = Field(alias="scaleMin")
    scale_max: float = Field(alias="scaleMax")
    step: Optional[float] = None
    anchors: Optional[Dict[str, str]] = None

    @model_validator(mode="after")
    def _validate_rating_contract(self) -> "RatingUI":
        try:
            if float(self.scale_max) <= float(self.scale_min):
                raise ValueError("rating requires scale_max > scale_min")
        except Exception as e:
            raise ValueError("rating requires numeric scale_min/scale_max") from e
        if self.step is not None:
            try:
                if float(self.step) <= 0:
                    raise ValueError("rating requires step > 0")
            except Exception as e:
                raise ValueError("rating requires numeric step") from e
        return self


class SliderUI(_UIStepBase):
    """
    Canonical numeric control for a single value.

    NOTE: We allow extra fields (widget/back-compat), but enforce that numeric sliders
    have usable bounds and a unit/currency label so the UI can render correctly.
    """

    type: Literal["slider", "range_slider"]

    # Core numeric contract (required)
    min: float
    max: float
    step: float = 1

    # Display label contract (at least one recommended)
    unit: Optional[str] = None
    currency: Optional[str] = None
    unit_type: Optional[str] = Field(default=None, alias="unitType")

    @model_validator(mode="after")
    def _validate_slider_contract(self) -> "SliderUI":
        # Bounds must be sensible
        try:
            if float(self.max) <= float(self.min):
                raise ValueError("slider requires max > min")
        except Exception as e:
            raise ValueError("slider requires numeric min/max") from e
        try:
            if float(self.step) <= 0:
                raise ValueError("slider requires step > 0")
        except Exception as e:
            raise ValueError("slider requires numeric step") from e

        # Unit/currency guidance: require at least one non-empty label so the widget doesn't
        # render ambiguous numbers (e.g. missing $/sqft/etc.). If neither is provided, fail.
        unit = str(self.unit or "").strip()
        currency = str(self.currency or "").strip()
        unit_type = str(self.unit_type or "").strip()
        if not unit and not currency and not unit_type:
            raise ValueError("slider requires at least one of unit/currency/unitType")
        return self


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
    ranges: List[Dict[str, Any]]
    allow_custom: Optional[bool] = None
    custom_min: Optional[float] = None
    custom_max: Optional[float] = None
    currency_code: Optional[str] = None

    @model_validator(mode="after")
    def _validate_budget_cards_contract(self) -> "BudgetCardsUI":
        if not self.ranges:
            raise ValueError("budget_cards requires ranges")
        return self


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
    multi_select: Optional[bool] = None
    min_selections: Optional[int] = None
    max_selections: Optional[int] = None
    min_options: Optional[int] = None
    max_options: Optional[int] = None
    allow_other: Optional[bool] = None
    other_label: Optional[str] = None
    other_placeholder: Optional[str] = None
    other_requires_text: Optional[bool] = None
    variant: Optional[Literal["list", "grid", "compact", "cards"]] = None
    columns: Optional[int] = None


class SearchableSelectUI(_UIStepBase):
    type: Literal["searchable_select"]
    options: List[Union[MiniOption, str]] = Field(default_factory=list)


class CompositeUI(_UIStepBase):
    type: Literal["composite"]
    blocks: List[Dict[str, Any]] = Field(default_factory=list)


class GalleryUI(_UIStepBase):
    type: Literal["gallery"]


def _collect_ui_step_type_literals() -> list[str]:
    """
    Canonical list of all UI step `type` strings supported by the current schema models.

    This is used as the "shared" source of truth for validating allowed types and
    filtering caller-provided type lists.
    """
    out: set[str] = set()
    for cls in (
        TextInputUI,
        IntroUI,
        RatingUI,
        SliderUI,
        DatePickerUI,
        ColorPickerUI,
        LeadCaptureUI,
        PricingUI,
        ConfirmationUI,
        DesignerUI,
        FileUploadUI,
        BudgetCardsUI,
        MultipleChoiceUI,
        SearchableSelectUI,
        CompositeUI,
        GalleryUI,
    ):
        try:
            ann = cls.model_fields["type"].annotation  # pydantic v2
        except Exception:
            ann = None
        if ann is None:
            continue
        for v in get_args(ann):
            if isinstance(v, str) and v.strip():
                out.add(v.strip())
    return sorted(out)


# Shared, schema-derived list of valid `type` strings.
UI_STEP_TYPE_VALUES: list[str] = _collect_ui_step_type_literals()


__all__ = [
    "BudgetCardsUI",
    "ColorPickerUI",
    "CompositeUI",
    "ConfirmationUI",
    "DatePickerUI",
    "DesignerUI",
    "FileUploadUI",
    "GalleryUI",
    "IntroUI",
    "LeadCaptureUI",
    "MiniOption",
    "MultipleChoiceUI",
    "PricingUI",
    "RatingUI",
    "SliderUI",
    "SearchableSelectUI",
    "TextInputUI",
    "UI_STEP_TYPE_VALUES",
]
