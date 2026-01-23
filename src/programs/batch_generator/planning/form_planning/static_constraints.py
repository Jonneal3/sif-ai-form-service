from __future__ import annotations

# Hardcoded, backend-owned defaults for form constraints.
# Keep this file intentionally logic-free.

DEFAULT_CONSTRAINTS = {
    "maxBatches": 3,
    # Keep batches short to reduce variance and improve completion.
    # Use a range so different stages can clamp within it.
    "minStepsPerBatch": 2,
    "maxStepsPerBatch": 4,
    "tokenBudgetTotal": 3000,
}


__all__ = ["DEFAULT_CONSTRAINTS"]

