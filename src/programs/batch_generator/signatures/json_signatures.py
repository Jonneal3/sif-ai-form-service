from __future__ import annotations

import dspy


class BatchGeneratorJSON(dspy.Signature):
    """
    Generate a batch configuration as JSON.

    This repo passes most planning inputs as a single `context_json` blob (stringified JSON) to keep
    the DSPy signature stable while the underlying request schema evolves.

    FLOW GUIDE (deterministic skeleton):
    The backend may include a hardcoded flow guide under `context_json.flow_guide`, built from:
    - `src/programs/batch_generator/form_planning/batch_ordering.py` (stage: early/middle/late)
    - `src/programs/batch_generator/form_planning/components_allowed.py` (allowed component types)
    - `src/programs/batch_generator/form_planning/question_tonality.py` (question style hints)

    CONSTRAINTS (hard limits):
    The backend may include backend-owned limits under `context_json.batch_constraints` (and/or
    defaults from `src/programs/batch_generator/form_planning/static_constraints.py`).

    Key idea (abstraction boundary):
    - The model needs the *information* (industry, goal, constraints, grounding/RAG, prior answers, policies).
    - DSPy needs a *stable interface* (minimal, long-lived signature fields).

    Packing planning state into `context_json` gives the model the same information it would have received
    as many separate fields, without creating schema churn across modules/demos/optimizers.

    PLATFORM GOAL:
    This is an AI Pre-Design & Sales Conversion Platform. The form collects context through questions
    to generate visual pre-designs (AI images) that help prospects visualize their project before getting
    a quote. The goal is visual alignment integrated with quoting—prospects become "visual buyers"
    who are more qualified before the first conversation.

    BATCH GOALS (phase IDs are semantic, not numeric):
    - ContextCore: Fast, broad context capture to establish baseline understanding.
    - Details/WrapUp: Deeper, personalized follow-ups based on prior answers/plan.

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
    - If `context_json.flow_guide` is present, batches MUST obey it (especially allowed component types).
    - If `context_json.batch_constraints` is present, batches MUST respect it (batch counts/limits).
    - Phases are represented by list order (`phaseIndex`).
    - Avoid intent language in compiler-layer outputs (no purpose/goals/why); emit constraints + allowed types + neutral guidance only.

    Example-authoring rules (DSPy demos):
    - `src/programs/batch_generator/examples/current/batch_generator_examples_rules.md`
    """

    context_json: str = dspy.InputField(
        desc=(
            "Compact JSON context for this request (serialized as a string). "
            "Typically includes: platform goal + business context + industry/service (+ optional grounding), "
            "batch info/constraints (max_steps, max_tokens, allowed types), and "
            "form state (answers, asked_step_ids, form_plan and/or batch_policy). "
            "May also include `flow_guide` and `batch_constraints` from `programs.batch_generator.form_planning`."
        )
    )
    batches_json: str = dspy.OutputField(desc="JSON string representing batch configuration")


class BatchNextStepsJSONL(dspy.Signature):
    """
    Generate the next UI steps as JSONL (one JSON object per line).

    This signature is the "batch generator" layer: it assumes `context_json` already contains any
    form planning outputs (e.g., `form_plan`, `batch_policy`) plus the current form state and constraints.

    In particular, the orchestrator should include:
    - `batch_phase_id`: which phase we're generating now
    - `batch_phase_policy`: the definition for that phase (purpose/focus/limits), derived from the plan
    """

    context_json: str = dspy.InputField(
        desc=(
            "Compact JSON context for this request (includes `form_plan` and `batch_policy` when available). "
            "Must also include the current phase id, typically as `batch_phase_id`."
        )
    )
    max_steps: int = dspy.InputField(desc="Maximum number of steps to emit")
    allowed_mini_types: list[str] = dspy.InputField(desc="Allowed UI step types")

    mini_steps_jsonl: str = dspy.OutputField(desc="JSONL string, one UI step per line")
