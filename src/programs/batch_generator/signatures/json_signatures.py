from __future__ import annotations

import dspy


class BatchGeneratorJSON(dspy.Signature):
    """
    Generate a batch configuration as JSON.

    This repo passes most planning inputs as a single `context_json` blob (stringified JSON) to keep
    the DSPy signature stable while the underlying request schema evolves.

    PLATFORM GOAL:
    This is an AI Pre-Design & Sales Conversion Platform. The form collects context through questions
    to generate visual pre-designs (AI images) that help prospects visualize their project before getting
    a quote. The goal is visual alignment integrated with quoting—prospects become "visual buyers"
    who are more qualified before the first conversation.

    BATCH GOALS:
    - ContextCore (Batch1): Fast, broad context capture to establish baseline understanding.
    - Details/PersonalGuide (Batch2+): Deeper, personalized follow-ups based on prior answers/plan.

    CONTEXT BREAKDOWN (what `context_json` typically contains):
    - FORM CONTEXT: overall goal + business context + industry/service (+ optional grounding/anchors)
    - BATCH CONTEXT: which batch/phase we are generating + constraints/tokens/allowed types
    - FORM STATE: answers so far + asked step ids + any existing plan/policy

    VERTICAL NEUTRALITY:
    - If `context_json` includes a grounding snippet/preview, treat it as the source of vertical facts.
    - Avoid inventing industry-specific facts that are not present in the provided context.

    HARD RULES:
    - Output MUST be JSON ONLY in `batches_json` (no prose, no markdown, no code fences).
    - `batches_json` MUST be a valid JSON object/array when parsed.
    """

    context_json: str = dspy.InputField(
        desc=(
            "Compact JSON context for this request (serialized as a string). "
            "Typically includes: platform goal + business context + industry/service (+ optional grounding), "
            "batch info/constraints (max_steps, max_tokens, allowed types), and "
            "form state (answers, asked_step_ids, form_plan and/or batch_policy)."
        )
    )
    batch_id: str = dspy.InputField(desc="Stable batch/phase identifier")
    batches_json: str = dspy.OutputField(desc="JSON string representing batch configuration")
