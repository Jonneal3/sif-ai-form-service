from __future__ import annotations

from typing import Any, Callable, Iterable, Optional


def bootstrap_few_shot(
    *,
    program: Any,
    trainset: Iterable[Any],
    metric: Callable[..., Any],
    max_demos: int = 8,
    seed: Optional[int] = 0,
) -> Any:
    """
    Compile a DSPy `program` with a BootstrapFewShot-style optimizer when available.

    This is a starter helper meant to be adapted for your specific program + dataset.
    It intentionally avoids hard-coding provider/model configuration.
    """
    try:
        from dspy.teleprompt import BootstrapFewShot  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "DSPy BootstrapFewShot optimizer is unavailable in this environment/version. "
            "If you're on DSPy v3, check the teleprompt API and update this helper accordingly."
        ) from e

    kwargs: dict[str, Any] = {"metric": metric, "max_bootstrapped_demos": max_demos, "max_labeled_demos": max_demos}
    if seed is not None:
        kwargs["seed"] = int(seed)

    optimizer = BootstrapFewShot(**kwargs)
    return optimizer.compile(program, trainset=trainset)

