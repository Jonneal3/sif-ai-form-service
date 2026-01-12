"""
DSPy Signatures (LLM contracts only).

This file defines the *contracts* for DSPy programs:
- NextStepsJSONL: Main signature for unified /flow/new-batch endpoint

Signatures define the input/output interface and task behavior; they are not model training.
DSPy uses the docstring + field descriptions to build prompt formats and optimize exemplars.

Important principles:
- Examples must be INDUSTRY-AGNOSTIC. Avoid hardcoded vertical facts.
- Outputs are JSON/JSONL strings, parsed + validated with Pydantic in the runtime layer.
- Hard constraints live outside signatures (module/runtime): token cutoffs, step caps, and JSONL parsing/cleanup are enforced there.
"""

from __future__ import annotations

import dspy  # type: ignore


class NextStepsJSONL(dspy.Signature):
    """
    Single-call, streaming-friendly contract for the unified /flow/new-batch endpoint.

    ROLE AND GOAL:
    You are an expert visual-intake agent. Your job is to select the minimum set of questions that capture
    image-critical visual attributes (e.g., color, material, texture, finish, shape, scale, lighting,
    composition, and environment). The answers are combined into one prompt for AI image generation.
    Ask only what improves visual fidelity and scene accuracy.

    HARD RULES:
    - Output MUST be JSONL only in `mini_steps_jsonl` (one JSON object per line).
    - Each line MUST be a valid MiniStep object (TextInputMini / MultipleChoiceMini / RatingMini / FileUploadMini).
    - Ids must be deterministic and stable: use `step-<key>` style ids.
    """

    # Core Context
    context_json: str = dspy.InputField(
        desc="Compact JSON string with platform_goal, business_context, industry, service, required_uploads, personalization_summary, known_answers, already_asked_keys, form_plan, batch_state, and optional items/subcategories."
    )
    batch_id: str = dspy.InputField(desc="Batch identifier or label for the current call.")

    # Constraints
    max_steps: int = dspy.InputField(desc="Max total number of steps for the overall form.")
    allowed_mini_types: list[str] = dspy.InputField(
        desc="Allowed mini types as a list (e.g. ['multiple_choice','text_input']). Only generate steps of these types."
    )

    mini_steps_jsonl: str = dspy.OutputField(
        desc="CRITICAL OUTPUT FIELD: You MUST output JSONL text here (one JSON object per line, no prose, no markdown, no code fences). Each line must be a valid JSON object with: id (step-{key} format), type (one of allowed_mini_types), question (user-facing question text), and required fields for that type. Focus questions on visual attributes needed for image generation. For multiple_choice steps, you MUST include a valid 'options' array with real option objects (NOT placeholders like '<<max_depth>>'). Each option must have 'label' (user-facing text) and 'value' (stable identifier). Generate 3-5 relevant options based on the question context. Output format: plain text with one JSON object per line. Example: {\"id\":\"step-color-family\",\"type\":\"multiple_choice\",\"question\":\"Which color family fits best?\",\"options\":[{\"label\":\"Warm neutrals\",\"value\":\"warm_neutrals\"},{\"label\":\"Cool grays\",\"value\":\"cool_grays\"}]}\n{\"id\":\"step-texture\",\"type\":\"multiple_choice\",\"question\":\"What texture do you prefer?\",\"options\":[{\"label\":\"Smooth\",\"value\":\"smooth\"},{\"label\":\"Textured\",\"value\":\"textured\"}]} DO NOT wrap in markdown code blocks. DO NOT add explanatory text. DO NOT use placeholder values like '<<max_depth>>' in options. Output ONLY the JSONL lines with real, valid option data."
    )


class MustHaveCopyJSON(dspy.Signature):
    """
    Lightweight copy-generation contract for required copy only.
    """

    context_json: str = dspy.InputField(
        desc="Compact JSON string with context plus must_have_copy_needed details."
    )
    mini_steps_jsonl: str = dspy.InputField(
        desc="JSONL lines for steps already generated in this batch (may be empty)."
    )
    must_have_copy_json: str = dspy.OutputField(
        desc="JSON object mapping copy keys (e.g. 'budget', upload stepIds) to StepCopy objects. Output JSON only."
    )
