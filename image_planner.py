"""
Image prompt + image generation orchestration.

This file is intentionally parallel to flow_planner.py:
- Uses the same compact session context builder for consistency.
- Uses DSPy for prompt construction (optional).
- Calls a provider/tool layer for actual image generation.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple


def _best_effort_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _deterministic_prompt(context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Fallback prompt builder when DSPy/LLM is not configured.

    Returns (prompt, metadata).
    """
    industry = (context.get("industry") or "").strip()
    service = (context.get("service") or "").strip()
    goal_intent = (context.get("goal_intent") or "").strip()
    personalization = (context.get("personalization_summary") or "").strip()
    answers = context.get("known_answers") if isinstance(context.get("known_answers"), dict) else {}

    parts: list[str] = []
    if service or industry:
        parts.append(f"Create an initial visual concept for: {service or industry}.")
    if goal_intent:
        parts.append(f"Goal intent: {goal_intent}.")
    if personalization:
        parts.append(f"User context: {personalization}.")
    if answers:
        preview_items = []
        for k, v in list(answers.items())[:12]:
            kk = str(k or "").strip()
            if not kk:
                continue
            vv = v
            if isinstance(vv, (dict, list)):
                continue
            preview_items.append(f"{kk}={vv}")
        if preview_items:
            parts.append("Known preferences: " + ", ".join(preview_items) + ".")

    if not parts:
        parts.append("Create a clean, realistic, high-quality visual concept based on the user's intake.")

    metadata = {"builder": "deterministic", "industry": industry, "service": service}
    return " ".join(parts).strip(), metadata


def build_image_prompt(payload: Dict[str, Any], *, prompt_template: Optional[str] = None) -> Dict[str, Any]:
    """
    Build an image prompt spec using DSPy when available, otherwise deterministic fallback.
    """
    request_id = f"image_prompt_{int(time.time() * 1000)}"

    # Allow explicit prompt override from the caller.
    if prompt_template and str(prompt_template).strip():
        return {
            "ok": True,
            "requestId": request_id,
            "prompt": {
                "prompt": str(prompt_template).strip(),
                "negativePrompt": "",
                "styleTags": [],
                "metadata": {"builder": "override"},
            },
        }

    # Reuse flow_planner's context builder so prompt inputs stay aligned.
    try:
        from flow_planner import _build_context as _build_context  # type: ignore
        from flow_planner import _compact_json as _compact_json  # type: ignore
        from flow_planner import _configure_dspy as _configure_dspy  # type: ignore
        from flow_planner import _make_dspy_lm as _make_dspy_lm  # type: ignore
    except Exception:
        _build_context = None
        _compact_json = None
        _configure_dspy = None
        _make_dspy_lm = None

    context: Dict[str, Any] = {}
    if callable(_build_context):
        try:
            context = _build_context(payload) or {}
        except Exception:
            context = {}

    lm_cfg = _make_dspy_lm() if callable(_make_dspy_lm) else None
    if not lm_cfg:
        prompt, metadata = _deterministic_prompt(context)
        return {
            "ok": True,
            "requestId": request_id,
            "prompt": {
                "prompt": prompt,
                "negativePrompt": "",
                "styleTags": [],
                "metadata": metadata,
            },
        }

    try:
        import dspy  # type: ignore
        from modules.image_prompt_module import ImagePromptModule
        from modules.schemas.image_prompt import ImagePromptSpec
    except Exception as e:
        prompt, metadata = _deterministic_prompt(context)
        metadata["dspyError"] = str(e)
        return {
            "ok": True,
            "requestId": request_id,
            "prompt": {
                "prompt": prompt,
                "negativePrompt": "",
                "styleTags": [],
                "metadata": metadata,
            },
        }

    llm_timeout = float(os.getenv("DSPY_LLM_TIMEOUT_SEC") or "20")
    temperature = float(os.getenv("DSPY_TEMPERATURE") or "0.5")
    max_tokens = int(os.getenv("DSPY_IMAGE_PROMPT_MAX_TOKENS") or "900")

    lm = dspy.LM(
        model=lm_cfg["model"],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=llm_timeout,
        num_retries=0,
    )
    if callable(_configure_dspy):
        _configure_dspy(lm)

    batch_id = str(payload.get("batchId") or payload.get("batch_id") or "")[:40] or "unknown"
    context_json = _compact_json(context) if callable(_compact_json) else json.dumps(context)

    module = ImagePromptModule()
    pred = module(context_json=context_json, batch_id=batch_id)
    raw = getattr(pred, "prompt_json", None) or ""
    obj = _best_effort_parse_json(str(raw))
    if not isinstance(obj, dict):
        prompt, metadata = _deterministic_prompt(context)
        metadata["dspyRaw"] = str(raw)[:500]
        return {
            "ok": True,
            "requestId": request_id,
            "prompt": {
                "prompt": prompt,
                "negativePrompt": "",
                "styleTags": [],
                "metadata": metadata,
            },
        }

    try:
        spec = ImagePromptSpec.model_validate(obj).model_dump(by_alias=True)
    except Exception:
        # If schema is slightly off, salvage the primary prompt field.
        prompt_text = str(obj.get("prompt") or "").strip()
        prompt, metadata = _deterministic_prompt(context)
        if prompt_text:
            prompt = prompt_text
            metadata = {"builder": "dspy_salvage", "rawKeys": list(obj.keys())[:20]}
        spec = {
            "prompt": prompt,
            "negativePrompt": str(obj.get("negativePrompt") or obj.get("negative_prompt") or ""),
            "styleTags": obj.get("styleTags") if isinstance(obj.get("styleTags"), list) else [],
            "metadata": metadata,
        }

    return {"ok": True, "requestId": request_id, "prompt": spec}

