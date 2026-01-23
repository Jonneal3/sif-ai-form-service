"""Batch-level metrics focused on question/answer structure and quality via dropoff patterns.

Key metrics (measured by dropoffs/abandonment):
1. Batch Dropoff Rate - how many users started but didn't complete (inverse quality signal)
2. Step Quality (per batch) - measured by dropoff patterns and engagement
3. Batch Cohesion - logical grouping, measured by completion vs abandonment patterns
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

from .session_log import BatchSessionLog, FormSessionLog


def _safe_mean(values: List[float]) -> Optional[float]:
    """Calculate mean, returning None for empty lists."""
    if not values:
        return None
    return sum(values) / len(values)


def _iter_batches(session: FormSessionLog) -> Iterable[BatchSessionLog]:
    """Extract batches from a session log."""
    batches = session.get("batches")
    if isinstance(batches, list):
        for b in batches:
            if isinstance(b, dict):
                yield b  # type: ignore[misc]


def _batch_id(batch: BatchSessionLog) -> Optional[str]:
    """Extract batch ID, returning None if missing or invalid."""
    bid = batch.get("batch_id")
    if isinstance(bid, str) and bid.strip():
        return bid.strip()
    return None


def batch_dropoff_rate(sessions: Iterable[FormSessionLog]) -> Dict[str, Optional[float]]:
    """
    Batch Dropoff Rate: How many users started but didn't complete the batch?
    
    Returns a rate (0.0-1.0) per batch_id, where:
    - 0.0 = all users completed (best)
    - 1.0 = all users dropped off (worst)
    
    This is a direct signal of question/answer structure quality.
    High dropoff = questions are confusing, poorly structured, or causing abandonment.
    """
    totals: Dict[str, int] = {}
    dropoffs: Dict[str, int] = {}
    
    for s in sessions:
        for b in _iter_batches(s):
            bid = _batch_id(b)
            if not bid:
                continue
            
            totals[bid] = totals.get(bid, 0) + 1
            
            # User dropped off if batch was started but not completed
            completed = b.get("completed")
            if completed is False:
                dropoffs[bid] = dropoffs.get(bid, 0) + 1
            # Also check if steps_answered is significantly less than steps_total
            elif completed is None:
                steps_answered = b.get("steps_answered")
                steps_total = b.get("steps_total") or b.get("max_steps")
                if isinstance(steps_answered, int) and isinstance(steps_total, int) and steps_total > 0:
                    answered_rate = steps_answered / steps_total
                    # If less than 50% answered and not explicitly completed, count as dropoff
                    if answered_rate < 0.5:
                        dropoffs[bid] = dropoffs.get(bid, 0) + 1
    
    out: Dict[str, Optional[float]] = {}
    for bid, total in totals.items():
        if total <= 0:
            out[bid] = None
        else:
            out[bid] = dropoffs.get(bid, 0) / total
    return out


def step_quality_per_batch(sessions: Iterable[FormSessionLog]) -> Dict[str, Optional[float]]:
    """
    Step Quality (per batch): Did the steps/questions make sense?
    
    Returns a score (1-5) per batch_id, where 5 = best quality, 1 = worst.
    
    Measured primarily by dropoff patterns:
    - Low dropoff rate = high quality (users engaged and completed)
    - High dropoff rate = low quality (users abandoned)
    - Step answered rate as secondary signal
    - Optional: incorporates question_difficulty_feedback if available
    """
    values: Dict[str, List[float]] = {}
    
    # First calculate dropoff rates
    dropoff_rates = batch_dropoff_rate(sessions)
    
    for s in sessions:
        for b in _iter_batches(s):
            bid = _batch_id(b)
            if not bid:
                continue
            
            score = None
            
            # Primary signal: dropoff rate (inverse - low dropoff = high quality)
            dropoff = dropoff_rates.get(bid)
            if dropoff is not None:
                # Convert dropoff rate (0.0-1.0) to quality score (1-5)
                # 0.0 dropoff -> 5.0 quality, 1.0 dropoff -> 1.0 quality
                score = 5.0 - (dropoff * 4.0)
            
            # Secondary signal: step answered rate
            steps_answered = b.get("steps_answered")
            steps_total = b.get("steps_total") or b.get("max_steps")
            if isinstance(steps_answered, int) and isinstance(steps_total, int) and steps_total > 0:
                answered_rate = steps_answered / steps_total
                answered_score = 1.0 + (answered_rate * 4.0)  # 0.0 -> 1.0, 1.0 -> 5.0
                
                if score is not None:
                    # Weighted: 70% dropoff-based, 30% answered rate
                    score = (score * 0.7) + (answered_score * 0.3)
                else:
                    score = answered_score
            
            # Optional: incorporate difficulty feedback if available
            difficulty = b.get("question_difficulty_feedback")
            if isinstance(difficulty, (int, float)) and 1 <= difficulty <= 5:
                if score is not None:
                    # Average with difficulty feedback (smaller weight)
                    score = (score * 0.8) + (float(difficulty) * 0.2)
                else:
                    score = float(difficulty)
            
            if score is not None:
                values.setdefault(bid, []).append(score)
    
    return {bid: _safe_mean(v) for bid, v in values.items()}


def step_answered_rate(sessions: Iterable[FormSessionLog]) -> Dict[str, Optional[float]]:
    """
    Step answered rate: how many questions were actually answered.
    
    Returns a rate (0.0-1.0) per batch_id.
    """
    rates: Dict[str, List[float]] = {}
    for s in sessions:
        for b in _iter_batches(s):
            bid = _batch_id(b)
            if not bid:
                continue
            
            steps_answered = b.get("steps_answered")
            steps_total = b.get("steps_total") or b.get("max_steps")
            
            if isinstance(steps_answered, int) and isinstance(steps_total, int) and steps_total > 0:
                rate = steps_answered / steps_total
                rates.setdefault(bid, []).append(rate)
    
    return {bid: _safe_mean(v) for bid, v in rates.items()}


def batch_cohesion(sessions: Iterable[FormSessionLog]) -> Dict[str, Optional[float]]:
    """
    Batch Cohesion: Are the steps logically grouped? Did the batch obey the flow guide?
    
    Returns a score (1-5) per batch_id, measured by completion vs abandonment patterns.
    
    High cohesion = users complete the batch (low dropoff)
    Low cohesion = users abandon mid-batch (high dropoff, indicating poor grouping/flow)
    
    Score factors:
    - Dropoff rate (primary - inverse relationship)
    - Batch completion status
    - Flow guide adherence (if available)
    - Batch number consistency (early batches should have better cohesion)
    """
    values: Dict[str, List[float]] = {}
    dropoff_rates = batch_dropoff_rate(sessions)
    
    for s in sessions:
        for b in _iter_batches(s):
            bid = _batch_id(b)
            if not bid:
                continue
            
            score = None
            
            # Factor 1: Dropoff rate (primary signal - inverse)
            dropoff = dropoff_rates.get(bid)
            if dropoff is not None:
                # Low dropoff = high cohesion
                score = 5.0 - (dropoff * 4.0)
            
            # Factor 2: Completion status (reinforces dropoff signal)
            completed = b.get("completed")
            if completed is True:
                if score is not None:
                    score = min(5.0, score + 0.5)  # Boost for explicit completion
                else:
                    score = 5.0
            elif completed is False:
                if score is not None:
                    score = max(1.0, score - 1.0)  # Penalize explicit non-completion
                else:
                    score = 2.0
            
            # Factor 3: Flow guide adherence (if available)
            flow_adherence = b.get("flow_guide_adherence")
            if isinstance(flow_adherence, (int, float)) and 1 <= flow_adherence <= 5:
                if score is not None:
                    # Weighted average: 80% dropoff-based, 20% explicit flow adherence
                    score = (score * 0.8) + (float(flow_adherence) * 0.2)
                else:
                    score = float(flow_adherence)
            
            # Factor 4: Batch number consistency (early batches should be easier/more cohesive)
            batch_number = b.get("batch_number")
            if isinstance(batch_number, int) and score is not None:
                # Lower batch numbers (earlier) should have higher cohesion if completed
                if completed is True and batch_number <= 2:
                    score = min(5.0, score + 0.3)
                # Later batches with high dropoff are especially problematic
                elif batch_number > 3 and dropoff is not None and dropoff > 0.5:
                    score = max(1.0, score - 0.5)
            
            if score is not None:
                values.setdefault(bid, []).append(score)
    
    return {bid: _safe_mean(v) for bid, v in values.items()}


def step_abandonment_rate(sessions: Iterable[FormSessionLog]) -> Dict[str, Optional[float]]:
    """
    Step Abandonment Rate: Which steps/questions cause users to leave?
    
    Returns a rate (0.0-1.0) per batch_id indicating how often steps were shown but not answered.
    
    High abandonment = questions are problematic (confusing, too hard, poorly structured).
    This is a direct measure of question/answer structure quality.
    """
    abandonment_rates: Dict[str, List[float]] = {}
    
    for s in sessions:
        for b in _iter_batches(s):
            bid = _batch_id(b)
            if not bid:
                continue
            
            steps_answered = b.get("steps_answered")
            steps_shown = b.get("steps_shown") or b.get("steps_total") or b.get("max_steps")
            
            # If we have both answered and shown counts
            if isinstance(steps_answered, int) and isinstance(steps_shown, int) and steps_shown > 0:
                # Abandonment = steps shown but not answered
                abandoned = steps_shown - steps_answered
                abandonment_rate = abandoned / steps_shown
                abandonment_rates.setdefault(bid, []).append(abandonment_rate)
    
    return {bid: _safe_mean(v) for bid, v in abandonment_rates.items()}


@dataclass(frozen=True)
class BatchMetrics:
    """
    Batch-level metrics focused on question/answer structure and quality via dropoff patterns.
    
    Primary metrics (dropoff-focused):
    1. Batch Dropoff Rate - how many users started but didn't complete (0.0-1.0, lower is better)
    2. Step Quality - measured by dropoff patterns and engagement (1-5, higher is better)
    3. Batch Cohesion - logical grouping, measured by completion vs abandonment (1-5, higher is better)
    
    Detailed breakdowns:
    - Step Answered Rate - engagement metric (0.0-1.0)
    - Step Abandonment Rate - which steps cause users to leave (0.0-1.0, lower is better)
    """
    batch_dropoff_rate: Dict[str, Optional[float]]
    step_quality: Dict[str, Optional[float]]
    batch_cohesion: Dict[str, Optional[float]]
    step_answered_rate: Dict[str, Optional[float]]
    step_abandonment_rate: Dict[str, Optional[float]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


def compute_batch_metrics(sessions: Iterable[FormSessionLog]) -> BatchMetrics:
    """
    Compute batch-level metrics from session logs, focused on dropoff patterns.
    
    These metrics measure question/answer structure and quality by tracking:
    - How many users drop off (abandon the batch)
    - Which steps cause abandonment
    - Engagement patterns (answered rates)
    
    High dropoff/abandonment = poor question structure or quality.
    """
    session_list = list(sessions)
    return BatchMetrics(
        batch_dropoff_rate=batch_dropoff_rate(session_list),
        step_quality=step_quality_per_batch(session_list),
        batch_cohesion=batch_cohesion(session_list),
        step_answered_rate=step_answered_rate(session_list),
        step_abandonment_rate=step_abandonment_rate(session_list),
    )

