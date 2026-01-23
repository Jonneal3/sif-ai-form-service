from __future__ import annotations


def resolve_stage(*, batch_index: int, total_batches: int) -> str:
    """
    Returns a stage label: early / middle / late

    - `batch_index` is 0-based.
    - `total_batches` is the planned max batches/calls.
    """
    try:
        idx = int(batch_index)
    except Exception:
        idx = 0
    try:
        total = int(total_batches)
    except Exception:
        total = 1

    if total <= 1 or idx <= 0:
        return "early"
    if idx < total - 1:
        return "middle"
    return "late"


__all__ = ["resolve_stage"]

