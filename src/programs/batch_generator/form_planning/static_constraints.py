from __future__ import annotations

from typing import Any, Dict

# Hardcoded, backend-owned defaults for form constraints.
# Env vars and request payload fields may still override these at runtime.

DEFAULT_MAX_BATCH_CALLS = 2
DEFAULT_MAX_STEPS_PER_BATCH = 5
DEFAULT_TOKEN_BUDGET_TOTAL = 3000


def resolve_max_calls(*, use_case: str, goal_intent: str, default_max_calls: int) -> int:
    """
    Determine backend-owned max batch calls.

    This is intentionally deterministic and conservative; it can be extended as the product
    learns better defaults by use case/goal.
    """
    uc = str(use_case or "").strip().lower()
    gi = str(goal_intent or "").strip().lower()

    # Slightly higher cap for pricing flows where more quantifiers are common.
    if uc == "pricing" or gi == "pricing":
        return max(default_max_calls, 3)

    # Keep image prompting flows short by default.
    if uc in ("image", "image_generation", "tryon", "try_on", "scene") or gi in ("image", "tryon", "try_on"):
        return min(default_max_calls, 2)

    return int(default_max_calls)


def default_batch_constraints(*, max_batches: int) -> Dict[str, Any]:
    """
    Canonical hardcoded constraints snapshot (used for responses/bootstrap).
    """
    mb = int(max_batches) if isinstance(max_batches, int) and max_batches > 0 else DEFAULT_MAX_BATCH_CALLS
    max_steps_total = DEFAULT_MAX_STEPS_PER_BATCH * mb
    return {
        "maxBatches": mb,
        "maxStepsTotal": max_steps_total,
        "maxStepsPerBatch": DEFAULT_MAX_STEPS_PER_BATCH,
        "tokenBudgetTotal": DEFAULT_TOKEN_BUDGET_TOTAL,
    }

