from __future__ import annotations

"""
Optional exploratory metrics.

Keep these metrics cheap and best-effort: they should never break production flows.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

from .session_log import FormSessionLog


@dataclass(frozen=True)
class ExploratoryMetrics:
    total_sessions: int
    avg_batches_per_session: Optional[float]


def compute_exploratory_metrics(sessions: Iterable[FormSessionLog]) -> ExploratoryMetrics:
    total = 0
    batches_counts: list[int] = []
    for s in sessions:
        total += 1
        batches = s.get("batches")
        if isinstance(batches, list):
            batches_counts.append(len([b for b in batches if isinstance(b, dict)]))
    avg = None
    if batches_counts:
        avg = sum(batches_counts) / float(len(batches_counts))
    return ExploratoryMetrics(total_sessions=total, avg_batches_per_session=avg)


__all__ = ["ExploratoryMetrics", "compute_exploratory_metrics"]

