"""
UI schema models for validated steps and form plan metadata.

These classes mirror the front-end contract and are used to validate
and normalize LLM outputs into deterministic UI step objects.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class MiniOption(BaseModel):
    label: str = Field(..., description="Option label (user-facing)")
    value: str = Field(..., description="Option value (stable identifier)")
    description: Optional[str] = None
    icon: Optional[str] = None
    image_url: Optional[str] = Field(None, alias="imageUrl")

    model_config = ConfigDict(populate_by_name=True)


class UIStepComponent(BaseModel):
    type: str = Field(..., description="Component kind (e.g. 'headline', 'helper', 'options')")
    key: Optional[str] = None
    text: Optional[str] = None
    required: bool = False
    props: Dict[str, Any] = Field(default_factory=dict)


class UIStepMetadata(BaseModel):
    components: Optional[List[UIStepComponent]] = None
    validation: Dict[str, Any] = Field(default_factory=dict)
    presentation: Dict[str, Any] = Field(default_factory=dict)
    ai_hint: Optional[str] = None
    family: Optional[str] = None


class UIStepBase(BaseModel):
    """
    Base class for all UI step schemas.
    Matches StepDefinition in sif-widget/types/ai-form.ts
    """

    id: str = Field(..., description="Deterministic step id (unique within session)")
    type: str = Field(..., description="Component type (e.g. slider, choice, budget_cards)")
    question: str = Field(..., description="The primary prompt/headline for the step")
    subtext: Optional[str] = None
    humanism: Optional[str] = None
    visual_hint: Optional[str] = None
    required: bool = True
    metric_gain: float = 0.1
    metadata: Optional[UIStepMetadata] = Field(
        default=None,
        validation_alias=AliasChoices("metadata", "blueprint"),
        serialization_alias="metadata",
    )

    model_config = ConfigDict(populate_by_name=True, extra="allow")


# Backward-compatible type name (older code/models used `blueprint`)
UIStepBlueprint = UIStepMetadata


class TextInputUI(UIStepBase):
    type: Literal["text", "text_input"] = "text"
    placeholder: Optional[str] = None
    multiline: bool = False
    max_length: Optional[int] = None


class MultipleChoiceUI(UIStepBase):
    type: Literal["choice", "multiple_choice", "segmented_choice", "chips_multi", "yes_no", "image_choice_grid"] = "choice"
    options: List[MiniOption]
    multi_select: bool = False
    variant: Literal["list", "grid", "compact", "cards"] = "list"
    columns: int = 1


class RatingUI(UIStepBase):
    type: Literal["slider", "rating", "range_slider"] = "slider"
    min: float = 0
    max: float = 100
    step: float = 1
    unit: Optional[str] = None
    format: Optional[Literal["currency"]] = None
    prefix: str = ""
    suffix: str = ""


class BudgetCardsUI(UIStepBase):
    type: Literal["budget_cards"] = "budget_cards"
    ranges: List[Dict[str, Any]]  # [{label, min, max}]
    allow_custom: bool = True
    custom_min: float = 0
    custom_max: float = 10000
    currency_code: str = "USD"


class FileUploadUI(UIStepBase):
    type: Literal["upload", "file_upload", "file_picker"] = "upload"
    max_files: int = 1
    upload_role: Optional[Literal["sceneImage", "userImage", "productImage"]] = "sceneImage"
    allow_skip: bool = True


class IntroUI(UIStepBase):
    type: Literal["intro"] = "intro"
    brand: Optional[str] = None
    bullets: Optional[List[str]] = None


class DatePickerUI(UIStepBase):
    type: Literal["date_picker"] = "date_picker"
    min_date: Optional[str] = None
    max_date: Optional[str] = None


class ColorPickerUI(UIStepBase):
    type: Literal["color_picker"] = "color_picker"
    colors: Optional[List[str]] = None


class SearchableSelectUI(UIStepBase):
    type: Literal["searchable_select"] = "searchable_select"
    options: List[MiniOption]
    multi_select: bool = False
    max_selections: Optional[int] = None
    search_placeholder: Optional[str] = None


class LeadCaptureUI(UIStepBase):
    type: Literal["lead_capture"] = "lead_capture"
    required_inputs: Optional[List[Literal["email", "phone", "name"]]] = None
    require_terms: Optional[bool] = None
    compact: Optional[bool] = None


class PricingUI(UIStepBase):
    type: Literal["pricing"] = "pricing"
    pricing_breakdown: Optional[List[Dict[str, Any]]] = None
    total_amount: Optional[float] = None
    currency_code: Optional[str] = None
    call_to_action: Optional[str] = None


class ConfirmationUI(UIStepBase):
    type: Literal["confirmation"] = "confirmation"
    summary_text: Optional[str] = None
    confirmation_message: Optional[str] = None


class DesignerUI(UIStepBase):
    type: Literal["designer"] = "designer"
    allow_refinements: Optional[bool] = None


class CompositeUI(UIStepBase):
    type: Literal["composite"] = "composite"
    blocks: List[Dict[str, Any]]


class FormPlanItem(BaseModel):
    """
    Python mirror of `FormPlanItem` in `types/ai-form.ts`.
    """

    key: str = Field(..., description="Stable key for this item (used to derive step ids)")
    goal: str = Field(..., description="What this question accomplishes")
    why: str = Field(..., description="Why this is needed (prompt usefulness)")
    component_hint: str = Field(..., description="High-level component hint (e.g. choice, budget_cards, slider, text)")
    priority: Literal["critical", "high", "medium", "low"] = Field(...)
    importance_weight: float = Field(..., ge=0.0, le=1.0)
    expected_metric_gain: float = Field(..., ge=0.0, le=1.0)


class UIPlacement(BaseModel):
    """
    Placement instructions for deterministic/structural steps that the frontend renders
    without consuming LLM budget (e.g., uploads, lead_capture, pricing, designer).
    """

    id: str = Field(..., description="Deterministic step id (e.g. step-upload-scene)")
    type: str = Field(..., description="UIStep type (e.g. upload, lead_capture, pricing)")
    role: Optional[str] = Field(default=None, description="Optional upload_role or placement role")
    position: Literal["start", "end", "after", "before"] = Field(
        ...,
        description="Where to place relative to an anchor step, or at start/end.",
    )
    anchor_step_id: Optional[str] = Field(
        default=None,
        description="When position is after/before, place relative to this step id.",
    )
    deterministic: bool = Field(default=True, description="True if frontend should not expect LLM generation")


class UIPlan(BaseModel):
    v: int = Field(default=1, description="Version for forwards-compatible parsing")
    placements: List[UIPlacement] = Field(default_factory=list)


class StepCopy(BaseModel):
    headline: str = Field(..., description="Customer-facing question/title in plain language")
    subtext: Optional[str] = Field(None, description="Short helper sentence under the headline")
    helper: Optional[str] = Field(None, description="Optional tooltip/modal helper in layman's terms")
    examples: Optional[List[str]] = Field(None, description="2-4 example answers")
    typical: Optional[Dict[str, Any]] = Field(None, description="Optional typical guidance")
