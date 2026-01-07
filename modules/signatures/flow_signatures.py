"""
DSPy Signatures + Typed Schemas

This file defines the *contracts* for DSPy modules:
- NextStepsJSONL: Main signature for unified /flow/new-batch endpoint (DSPy decides which questions to ask)
- BatchGeneratorJSON: Legacy signature for batch mini-step generation

Important principles:
- Examples must be INDUSTRY-AGNOSTIC. Vertical facts must come from `grounding_preview` (DB/RAG).
- Outputs are JSON/JSONL strings, parsed + validated with Pydantic in the Module layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import dspy  # type: ignore
from pydantic import BaseModel, Field, field_validator, AliasChoices


def _slugify_value(s: str) -> str:
    import re

    t = str(s or "").strip().lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"^_+|_+$", "", t)
    return (t or "option")[:48]


class MiniOption(BaseModel):
    label: str = Field(..., description="Option label (user-facing)")
    value: str = Field(..., description="Option value (stable identifier)")


class MiniStepBase(BaseModel):
    """
    Base class for all mini-step schemas.
    
    COMMON CAPABILITIES ACROSS ALL COMPONENTS:
    - Skip button: Automatically added if required=false (user can skip the question)
    - Visual hints: Use visual_hint to guide UI rendering (optional metadata)
    - Humanism: Short friendly phrase to make questions feel more conversational
    - Metric gain: How much this question contributes to overall confidence (0.0-1.0)
    """
    id: str = Field(..., description="Deterministic step id (unique within session)")
    type: Literal["text_input", "multiple_choice", "rating", "file_upload"] = Field(...)
    question: str = Field(
        ...,
        validation_alias=AliasChoices("question", "prompt"),
        description="Customer-facing question text. Make it clear, conversational, and focused on visual/image generation needs.",
    )
    humanism: Optional[str] = Field(
        None, 
        description="Optional short friendly phrase (e.g., 'No pressure!' or 'This helps us visualize your space'). Makes questions feel more human."
    )
    visual_hint: Optional[str] = Field(
        None, 
        description="Optional visual hint for UI/metadata. Can guide rendering or provide context for image generation."
    )
    required: Optional[bool] = Field(
        None, 
        description="Whether the step is required. If false, a skip button is automatically added. Use false for optional questions."
    )
    metric_gain: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Expected confidence contribution (0.0-1.0). Higher = more important for image generation. Critical questions should be 0.15-0.25."
    )


class TextInputMini(MiniStepBase):
    """
    Text input component capabilities:
    - User types freeform text
    - Can set max_length (1-1000 characters) to limit response size
    - Placeholder text guides user on what to enter
    - Can be optional (required=false) which adds a skip button
    - Use for: open-ended questions, descriptions, names, custom answers
    """
    type: Literal["text_input"] = "text_input"
    max_length: Optional[int] = Field(
        None, 
        ge=1, 
        le=1000, 
        description="Max characters allowed (1-1000). Use shorter limits (50-200) for names/titles, longer (500-1000) for descriptions."
    )
    placeholder: Optional[str] = Field(
        None, 
        description="Placeholder text shown in empty field. Use to guide format (e.g., 'e.g., Modern, Traditional, Industrial')"
    )


class MultipleChoiceMini(MiniStepBase):
    """
    Multiple choice component capabilities:
    - Can have 1-10 options (typically 3-6 is optimal for clarity)
    - Single-select (default): user picks one option - best for mutually exclusive choices
    - Multi-select (multi_select=true): user can pick multiple, optionally capped (max_selections)
    - Can be optional (required=false) which adds a skip button
    - "Other" option pattern: Add {"label": "Other", "value": "other"} as last option to allow custom text input
    - Use for: style preferences, categories, priorities, material choices, color palettes, layout preferences
    - Avoid: Too many options (over 6-7 becomes overwhelming), use text_input for truly open-ended questions
    """
    type: Literal["multiple_choice"] = "multiple_choice"
    options: List[MiniOption] = Field(
        ..., 
        description="Answer options (1-10 options recommended, 3-6 is optimal). Can include 'Other' option for custom input. Each option needs label (user-facing) and value (stable identifier)."
    )
    multi_select: Optional[bool] = Field(
        False, 
        description="If true, user can select multiple options. Use for: 'select all that apply' questions, preferences that aren't mutually exclusive."
    )
    max_selections: Optional[int] = Field(
        None, 
        ge=1, 
        le=10, 
        description="If multi_select=true, cap how many options user can pick (1-10). Leave None for unlimited (up to all options)."
    )

    @field_validator("options", mode="before")
    @classmethod
    def _normalize_options(cls, v):
        # Accept either ["A","B"] or [{"label","value"}] and normalize.
        if not isinstance(v, list):
            return v
        out: List[Dict[str, str]] = []
        for o in v:
            if isinstance(o, str):
                label = o.strip()
                if not label:
                    continue
                out.append({"label": label, "value": _slugify_value(label)})
                continue
            if isinstance(o, dict):
                raw_label = o.get("label") or o.get("value") or ""
                label = str(raw_label).strip()
                raw_value = o.get("value") or _slugify_value(label)
                value = str(raw_value).strip() or _slugify_value(label)
                if not label:
                    label = value
                out.append({"label": label, "value": value})
        return out


class RatingMini(MiniStepBase):
    """
    Rating/slider component capabilities:
    - Numeric scale from scale_min to scale_max (e.g., 1-5, 0-10)
    - User slides or clicks to select a number
    - Can set step size (default 1) for granularity
    - Can add anchor labels (min_label, max_label) to clarify scale ends
    - Can be optional (required=false) which adds a skip button
    - Use for: intensity, satisfaction, priority levels, numeric preferences
    """
    type: Literal["rating"] = "rating"
    scale_min: int = Field(..., description="Minimum value (typically 1 or 0)")
    scale_max: int = Field(..., description="Maximum value (typically 5 or 10, keep range reasonable)")
    step: Optional[int] = Field(1, description="Step size (1 = whole numbers, 0.5 = half steps). Keep at 1 for simplicity.")
    anchors: Optional[Dict[str, str]] = Field(
        None, 
        description="Optional labels for scale ends. Use {'min_label': 'Not important', 'max_label': 'Very important'} to clarify meaning."
    )


class FileUploadMini(MiniStepBase):
    type: Literal["file_upload"] = "file_upload"
    allowed_file_types: Optional[List[str]] = Field(None)
    max_size_mb: Optional[float] = Field(None, ge=0.1, le=100.0)
    upload_role: Optional[str] = Field(None)


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
        desc="CRITICAL OUTPUT FIELD: You MUST output JSONL text here (one JSON object per line, no prose, no markdown, no code fences). Generate steps for form_plan_json items that are NOT in already_asked_keys_json. Each line must be a valid JSON object with: id (step-{key} format), type (one of allowed_mini_types), question (user-facing question text), and required fields for that type. Output format: plain text with one JSON object per line. Example: {\"id\":\"step-project-goal\",\"type\":\"multiple_choice\",\"question\":\"What is your project goal?\"}\n{\"id\":\"step-space-type\",\"type\":\"multiple_choice\",\"question\":\"What type of space?\"} If form_plan_json has items and you generate 0 steps, you are FAILING the task. Minimum 1 step required if form_plan_json is non-empty and has unasked items. DO NOT wrap in markdown code blocks. DO NOT add explanatory text. Output ONLY the JSONL lines."
    )


