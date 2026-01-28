from __future__ import annotations

import os


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def prefixed_model(provider: str, model_name: str) -> str:
    """
    LiteLLM model strings are commonly provider-prefixed: "openai/gpt-4.1", "groq/llama-...".
    """
    p = str(provider or "").strip().lower()
    m = str(model_name or "").strip()
    if not p:
        return m
    if m.startswith(f"{p}/"):
        return m
    return f"{p}/{m}"


__all__ = ["env_bool", "env_int", "env_float", "prefixed_model"]

