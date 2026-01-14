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
    You are an expert intake agent for pricing-first visual pre-designs. Your job is to select the minimum
    set of questions that maximize quote accuracy and visual fidelity. Prioritize measurable scope, size,
    materials, budget, timeline, and constraints when they affect pricing or output quality. Ask only what
    changes the estimate or the final image.

    HARD RULES:
    - Output MUST be JSONL only in `mini_steps_jsonl` (one JSON object per line).
    - Each line MUST be a valid MiniStep object (TextInputMini / MultipleChoiceMini / RatingMini / FileUploadMini).
    - Ids must be deterministic and stable: use `step-<key>` style ids.
    - Choose 4–6 attribute_families most relevant to the current service, use_case, and goal_intent.
    - Every question MUST map to exactly one attribute_family (set `blueprint.family`).
    - Use service_anchor_terms to keep options service-relevant.
    - Avoid generic filler options (red/blue/green, circle/square/triangle, abstract/experimental styles).
    - If platform_goal suggests pricing/quote/estimate, prioritize scope/size/material/budget/timeline first.
    - Use items/form_plan and instance_subcategories when available to keep questions concrete.
    - Avoid vague questions (e.g., "any constraints?" or "any additional info?") unless specific and useful.
    - Avoid generic style options (e.g., modern/traditional) unless clearly grounded in service context.
    - For sliders/ratings, always include a clear unit/prefix/suffix (e.g., prefix='$', suffix='sq ft', suffix='ft', or unit='years').
    """

    # Core Context
    context_json: str = dspy.InputField(
        desc="Compact JSON string with platform_goal, business_context, industry, service, use_case, goal_intent, attribute_families, service_anchor_terms, required_uploads, personalization_summary, known_answers, already_asked_keys, form_plan, batch_state, items, and instance_subcategories."
    )
    batch_id: str = dspy.InputField(desc="Batch identifier or label for the current call.")

    # Constraints
    max_steps: int = dspy.InputField(desc="Max total number of steps for the overall form.")
    allowed_mini_types: list[str] = dspy.InputField(
        desc="Allowed mini types as a list (e.g. ['multiple_choice','text_input']). Only generate steps of these types."
    )

    mini_steps_jsonl: str = dspy.OutputField(
        desc="CRITICAL OUTPUT FIELD: You MUST output JSONL text here (one JSON object per line, no prose, no markdown, no code fences). Each line must be a valid JSON object with: id (step-{key} format), type (one of allowed_mini_types), question (user-facing question text), and required fields for that type. Include blueprint.family to map each step to a single attribute_family. If a question is about size, budget, or timeline and sliders/ratings are allowed, use them; otherwise use ranges in options. For sliders, include explicit units (prefix/suffix/unit). Use format=currency or prefix='$' for budgets. For multiple_choice steps, you MUST include a valid 'options' array with real option objects (NOT placeholders like '<<max_depth>>'). Each option must have 'label' (user-facing text) and 'value' (stable identifier). Generate 3-5 relevant options based on the question context, service_anchor_terms, and instance_subcategories. Output format: plain text with one JSON object per line. Example: {\"id\":\"step-area-size\",\"type\":\"multiple_choice\",\"question\":\"What size range fits best?\",\"blueprint\":{\"family\":\"area_size\"},\"options\":[{\"label\":\"Under 200 sq ft\",\"value\":\"under_200\"},{\"label\":\"200-500 sq ft\",\"value\":\"200_500\"}]}\n{\"id\":\"step-budget\",\"type\":\"slider\",\"question\":\"What budget range fits best?\",\"min\":1000,\"max\":20000,\"step\":500,\"prefix\":\"$\",\"blueprint\":{\"family\":\"budget_range\"}} DO NOT wrap in markdown code blocks. DO NOT add explanatory text. DO NOT use placeholder values like '<<max_depth>>' in options. Output ONLY the JSONL lines with real, valid option data."
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


class ImagePromptJSON(dspy.Signature):
    """
    Build a production-ready image generation prompt from the session context.

    ROLE AND GOAL:
    You take structured intake context (industry, service, goal_intent, known_answers, form_plan, uploads)
    and produce a single, model-ready prompt plus optional negative prompt and style tags.

    HARD RULES:
    - Output MUST be valid JSON only in `prompt_json` (no prose, no markdown, no code fences).
    - JSON MUST contain at least: {"prompt": "..."}.
    - Prefer concrete, visual, non-abstract descriptors grounded in known_answers/service anchors.
    - Do NOT invent measurements or materials unless clearly implied by known_answers.
    """

    context_json: str = dspy.InputField(
        desc="Compact JSON string with platform_goal, business_context, industry, service, use_case, goal_intent, attribute_families, service_anchor_terms, required_uploads, personalization_summary, known_answers, already_asked_keys, form_plan, batch_state, items, and instance_subcategories."
    )
    batch_id: str = dspy.InputField(desc="Batch identifier or label that triggered image generation.")
    prompt_json: str = dspy.OutputField(
        desc="JSON ONLY (no markdown). Must include: prompt (string). Optional: negativePrompt (string), styleTags (string[]), metadata (object)."
    )
