"""
DSPy Signatures + Typed Schemas

This file defines the *contracts* for DSPy modules:
- NextStepsJSONL: Main signature for unified /flow/new-batch endpoint (DSPy decides which questions to ask)
- BatchGeneratorJSON: Legacy signature for batch mini-step generation

Important principles:
- Examples must be INDUSTRY-AGNOSTIC. Vertical facts must come from `grounding_preview` (DB/RAG).
- Outputs are JSON/JSONL strings, parsed + validated with Pydantic in the Module layer.

---

### DSPy beginner notes

DSPy uses **Signatures** to describe the inputs/outputs of a “program”.

Think of a Signature as a typed contract:
- `dspy.InputField(...)` describes what you pass in
- `dspy.OutputField(...)` describes what you want back

In this repo we intentionally make outputs **strings** (e.g. `mini_steps_jsonl: str`) even though they
represent structured data, because LLMs always output text. We then:
1) parse the string into Python objects (JSON)
2) validate with Pydantic models (schema enforcement)

This “string output + validate” pattern is the simplest way to make LLM output production-safe.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import dspy  # type: ignore
from pydantic import BaseModel, ConfigDict, Field, field_validator, AliasChoices


def _slugify_value(s: str) -> str:
    import re

    t = str(s or "").strip().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"^_+|_+$", "", t)
    return (t or "option")[:48]


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


class UIStepBlueprint(BaseModel):
    components: Optional[List[UIStepComponent]] = None
    validation: Dict[str, Any] = Field(default_factory=dict)
    presentation: Dict[str, Any] = Field(default_factory=dict)
    ai_hint: Optional[str] = None


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
    blueprint: Optional[UIStepBlueprint] = None

    model_config = ConfigDict(populate_by_name=True, extra="allow")


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


class GenericUI(UIStepBase):
    """Fallback for any component type not yet explicitly mirrored."""
    data: Dict[str, Any] = Field(default_factory=dict)


class BatchGeneratorJSON(dspy.Signature):
    """
    Generate ALL batch mini-schemas in ONE response.

    HARD RULES (must-follow):
    - Output MUST be JSON ONLY in `mini_steps_json` (no prose, no markdown).
    - `mini_steps_json` MUST be a JSON ARRAY.
    - You MUST generate EXACTLY one mini-step per input item in `items_json`.
    - You MUST reuse the provided input item's `id` as the mini-step `id`.
      Example: if items_json contains {"id":"step-style_direction", ...}
      then your output mini step MUST include {"id":"step-style_direction", ...}
    - Do NOT invent new ids like "mini-step-1".
    """

    batch_id: str = dspy.InputField(desc="ContextCore | PersonalGuide")
    industry: str = dspy.InputField(desc="Industry/vertical")
    service: str = dspy.InputField(desc="Service/subcategory label (may be empty)")
    items_json: str = dspy.InputField(
        desc=(
            "JSON array of deterministic step skeleton items. "
            "You MUST generate exactly one mini step per item AND MUST reuse item.id as the output mini step id."
        )
    )
    allowed_mini_types: str = dspy.InputField(desc="Comma-separated allowed mini types")
    max_steps: str = dspy.InputField(desc="Max steps to generate for this batch")
    max_tokens_hint: str = dspy.InputField(desc="Token budget hint (e.g. 1500-3000)")
    already_asked_keys_json: str = dspy.InputField(desc="JSON array of ids/keys already asked")
    personalization_summary: str = dspy.InputField(desc="Short summary of user so far (Batch2 guidance); may be empty")
    grounding_preview: str = dspy.InputField(desc="Short vertical grounding preview (RAG snippet)")

    mini_steps_json: str = dspy.OutputField(desc="JSON ARRAY ONLY. Must be a list of mini steps.")


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


class StepCopy(BaseModel):
    headline: str = Field(..., description="Customer-facing question/title in plain language")
    subtext: Optional[str] = Field(None, description="Short helper sentence under the headline")
    helper: Optional[str] = Field(None, description="Optional tooltip/modal helper in layman's terms")
    examples: Optional[List[str]] = Field(None, description="2-4 example answers")
    typical: Optional[Dict[str, Any]] = Field(None, description="Optional typical guidance")


class NextStepsJSONL(dspy.Signature):
    """
    Single-call, streaming-friendly contract for the unified /flow/new-batch endpoint.

    ULTIMATE GOAL: GENERATE A KILLER PROMPT FOR AI IMAGE GENERATION
    Every question you generate should be formulated with this end goal in mind: once the form is answered,
    the collected answers will be synthesized into a detailed prompt that generates AI images showing what
    the user wants to see. Think like an industry professional designer: "What would an expert designer
    in this industry ask to generate a great preliminary design?"

    STRATEGIC QUESTION FORMULATION:
    - Each question should collect information that directly contributes to creating a compelling visual prompt.
    - Think about what visual elements matter most: style, colors, materials, layout, mood, lighting, scale, context.
    - Prioritize questions that reveal the user's visual intent and aesthetic preferences.
    - Consider what details would help an AI image generator produce accurate, desirable results.
    - Questions should feel natural and conversational, but every answer should serve the visual generation goal.

    PLATFORM GOAL:
    This is an AI Pre-Design & Sales Conversion Platform. The form collects context through questions
    to generate visual pre-designs (AI images) that help prospects visualize their project before getting
    a quote. The goal is visual alignment integrated with quoting—prospects become "visual buyers"
    who are more qualified before the first conversation. Questions should guide users toward visual
    clarity that enables accurate pricing and compelling imagery.

    BATCH GOALS:
    - ContextCore (Batch1): Fast, broad context questions to establish baseline visual understanding (style, goal, key preferences)
    - PersonalGuide (Batch2): Deeper, personalized follow-ups based on Batch1 answers (refinements, specific details, edge cases)

    VERTICAL NEUTRALITY (CRITICAL FOR PROMPT OPTIMIZATION):
    - Questions SHOULD be industry-specific and relevant to the user's vertical (e.g., "What style of kitchen cabinets?" for kitchen remodeling).
    - HOWEVER: ALL industry-specific facts MUST come from `grounding_preview` (DB/RAG) at runtime, NOT from hardcoded knowledge.
    - `industry` and `service` are labels for REFERENCE ONLY—use them to understand context, but source all factual details from `grounding_preview`.
    - This ensures the optimized prompt remains industry-agnostic (works across all verticals) while questions are still tailored via RAG.
    - Use `grounding_preview` to: tailor question wording, generate relevant options, provide industry-appropriate examples.

    HARD RULES:
    - Output MUST be JSONL only in `mini_steps_jsonl` (one JSON object per line).
    - Each line MUST be a valid MiniStep object (TextInputMini / MultipleChoiceMini / RatingMini / FileUploadMini).
    - Ids must be deterministic and stable: use `step-<key>` style ids.
    - Do NOT invent vertical facts; only use `grounding_preview`.
    """

    platform_goal: str = dspy.InputField(
        desc="Overall platform purpose: visual pre-selling before quote, pricing funnel conversion, lead qualification through visual alignment. ULTIMATE GOAL: Generate a killer prompt for AI image generation. Every question should contribute to creating compelling visual prompts that show what the user wants to see."
    )
    batch_id: str = dspy.InputField(desc="ContextCore (fast broad questions) | PersonalGuide (deeper follow-ups)")
    business_context: str = dspy.InputField(desc="Generic business context (industry-agnostic, for tone/style guidance)")
    industry: str = dspy.InputField(
        desc="Industry label for context reference. Use to understand the vertical, but source all factual details from grounding_preview."
    )
    service: str = dspy.InputField(
        desc="Service label for context reference. Use to understand the vertical, but source all factual details from grounding_preview."
    )
    grounding_preview: str = dspy.InputField(
        desc="DB/RAG grounding snippet with vertical-specific facts. THIS is your PRIMARY source for industry/service details. Use this to generate industry-relevant questions, options, and examples. Do NOT invent industry facts—only use what's provided here."
    )
    required_uploads_json: str = dspy.InputField(
        desc="JSON array of required uploads [{stepId, role}] for must-have copy generation (e.g. sceneImage, userImage, productImage)"
    )
    personalization_summary: str = dspy.InputField(
        desc="Summary of user so far (Batch2 only; empty string for Batch1). Use to personalize follow-up questions."
    )
    known_answers_json: str = dspy.InputField(desc="JSON object: current stepDataSoFar (answers so far)")
    already_asked_keys_json: str = dspy.InputField(desc="JSON array of step ids/keys already asked")
    form_plan_json: str = dspy.InputField(
        desc="JSON array of FormPlanItem[] if already known; otherwise empty string and you must produce it. IMPORTANT: If form_plan_json is provided (non-empty), you MUST generate mini steps for items in this plan that are NOT in already_asked_keys_json. Each FormPlanItem has a 'key' field - convert it to step id as 'step-{key}' (replacing underscores with hyphens)."
    )
    batch_state_json: str = dspy.InputField(
        desc="JSON object with callsUsed, maxCalls, callsRemaining, satietySoFar, satietyRemaining, missingHighImpactKeys, mustHaveCopyNeeded"
    )
    max_steps: str = dspy.InputField(desc="Max number of mini steps to output this call (string int). You MUST generate at least one step if form_plan_json contains items not in already_asked_keys_json.")
    allowed_mini_types: str = dspy.InputField(desc="Comma-separated allowed mini types (e.g. 'multiple_choice,text_input'). Only generate steps of these types.")

    produced_form_plan_json: str = dspy.OutputField(
        desc="If input form_plan_json was empty, output JSON array FormPlanItem[] here; else output empty string."
    )
    must_have_copy_json: str = dspy.OutputField(
        desc="JSON object mapping copy keys (e.g. 'budget', upload stepIds) to StepCopy objects. May be empty."
    )
    ready_for_image_gen: str = dspy.OutputField(desc="true/false string. True if we should proceed to structural steps.")
    mini_steps_jsonl: str = dspy.OutputField(
        desc="CRITICAL OUTPUT FIELD: You MUST output JSONL text here (one JSON object per line, no prose, no markdown, no code fences). Generate steps for form_plan_json items that are NOT in already_asked_keys_json. Each line must be a valid JSON object with: id (step-{key} format), type (one of allowed_mini_types), question (user-facing question text), and required fields for that type. For multiple_choice steps, you MUST include a valid 'options' array with real option objects (NOT placeholders like '<<max_depth>>'). Each option must have 'label' (user-facing text) and 'value' (stable identifier). Generate 3-5 relevant options based on the question context and grounding_preview. Output format: plain text with one JSON object per line. Example: {\"id\":\"step-project-goal\",\"type\":\"multiple_choice\",\"question\":\"What is your project goal?\",\"options\":[{\"label\":\"Renovation\",\"value\":\"renovation\"},{\"label\":\"New Build\",\"value\":\"new_build\"}]}\n{\"id\":\"step-space-type\",\"type\":\"multiple_choice\",\"question\":\"What type of space?\",\"options\":[{\"label\":\"Kitchen\",\"value\":\"kitchen\"},{\"label\":\"Bathroom\",\"value\":\"bathroom\"}]} If form_plan_json has items and you generate 0 steps, you are FAILING the task. Minimum 1 step required if form_plan_json is non-empty and has unasked items. DO NOT wrap in markdown code blocks. DO NOT add explanatory text. DO NOT use placeholder values like '<<max_depth>>' in options. Output ONLY the JSONL lines with real, valid option data."
    )


