from __future__ import annotations

import os
from typing import Any, Dict, Optional

from programs.common.env import prefixed_model


def make_dspy_lm_for_module(*, module_env_prefix: str, allow_small_models: bool) -> Optional[Dict[str, str]]:
    """
    Resolve the DSPy LM config for a module using env overrides.

    Env resolution order (example for module_env_prefix="DSPY_PLANNER"):
      - DSPY_PLANNER_PROVIDER / DSPY_PROVIDER
      - DSPY_PLANNER_MODEL_LOCK / DSPY_MODEL_LOCK / default
      - DSPY_PLANNER_MODEL / DSPY_MODEL / DSPY_PLANNER_MODEL_LOCK
    """
    prefix = str(module_env_prefix or "").strip().upper()
    provider = (os.getenv(f"{prefix}_PROVIDER") or os.getenv("DSPY_PROVIDER") or "groq").lower()
    locked_model = os.getenv(f"{prefix}_MODEL_LOCK") or os.getenv("DSPY_MODEL_LOCK") or "openai/gpt-oss-20b"
    requested_model = os.getenv(f"{prefix}_MODEL") or os.getenv("DSPY_MODEL") or locked_model
    model_name = str(requested_model or locked_model).strip()

    # Safety guard: keep planner on a strong model unless explicitly allowed.
    if not allow_small_models:
        is_gpt_oss = "gpt-oss" in model_name.lower()
        if not is_gpt_oss and ("8b" in model_name.lower() or "8-b" in model_name.lower() or "instant" in model_name.lower()):
            model_name = str(locked_model or model_name).strip()

    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            return None
        return {"provider": "groq", "model": prefixed_model("groq", model_name), "modelName": model_name}

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return None
        return {"provider": "openai", "model": prefixed_model("openai", model_name), "modelName": model_name}

    return None


def configure_dspy(lm: Any) -> bool:
    """
    Configure DSPy with a LiteLLM-backed LM instance.
    Returns whether usage tracking is enabled.
    """
    try:
        import dspy  # type: ignore
    except Exception:
        return False

    telemetry_on = os.getenv("AI_FORM_TOKEN_TELEMETRY") == "true" or os.getenv("AI_FORM_DEBUG") == "true"
    track_usage = os.getenv("DSPY_TRACK_USAGE") == "true" or telemetry_on
    try:
        dspy.settings.configure(lm=lm, track_usage=track_usage)
        return track_usage
    except Exception:
        return False


def extract_dspy_usage(prediction: Any) -> Optional[Dict[str, Any]]:
    try:
        get_usage = getattr(prediction, "get_lm_usage", None)
        if callable(get_usage):
            usage = get_usage()
            if isinstance(usage, dict) and usage:
                return usage
    except Exception:
        return None
    return None


__all__ = ["make_dspy_lm_for_module", "configure_dspy", "extract_dspy_usage"]

