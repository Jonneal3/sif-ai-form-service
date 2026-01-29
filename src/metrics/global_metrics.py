from __future__ import annotations

"""
Form-wide analytics metrics (product analytics, not DSPy optimization metrics).

This module intentionally starts simple and can be expanded as we standardize
the session log schema and add more signals.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

from .session_log import FormSessionLog


@dataclass(frozen=True)
class GlobalMetrics:
    total_sessions: int
    completed_sessions: int
    completion_rate: Optional[float]


def compute_global_metrics(sessions: Iterable[FormSessionLog]) -> GlobalMetrics:
    total = 0
    completed = 0
    for s in sessions:
        total += 1
        # Best-effort: accept multiple possible completion keys.
        is_completed = s.get("completed") or s.get("completedSession") or s.get("isCompleted")
        if is_completed is True:
            completed += 1
    rate = None
    if total > 0:
        rate = completed / float(total)
    return GlobalMetrics(total_sessions=total, completed_sessions=completed, completion_rate=rate)


__all__ = ["GlobalMetrics", "compute_global_metrics"]

