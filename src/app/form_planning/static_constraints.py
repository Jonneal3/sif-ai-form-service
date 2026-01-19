from __future__ import annotations

from typing import Optional


def resolve_max_calls(*, use_case: str, goal_intent: str, env_default: int) -> int:
    """
    Backend-owned cap on how many batch calls the overall flow may use.

    This is intentionally small and static; expand it as you add more
    use-cases / plans. `env_default` is the final fallback.
    """
    use = (use_case or "").strip().lower()
    goal = (goal_intent or "").strip().lower()

    static: Optional[int] = None

    # Keep pricing-style flows short by default.
    if goal == "pricing":
        static = 2

    # "Scene" and "tryon" flows typically have a 2-call arc as well (questions, then uploads/gen).
    if static is None and use in {"scene", "tryon"}:
        static = 2

    n = static if isinstance(static, int) and static > 0 else env_default
    return max(1, min(10, int(n)))

