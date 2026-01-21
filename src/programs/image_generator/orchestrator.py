"""
Image prompt + image generation orchestration.

This file is intentionally parallel to `src/app/pipeline/form_pipeline.py`:
- Uses the same compact session context builder for consistency.
- Uses DSPy for prompt construction (optional).
- Calls a provider/tool layer for actual image generation.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


def _best_effort_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def build_image_prompt(payload: Dict[str, Any], *, prompt_template: Optional[str] = None) -> Dict[str, Any]:
    """
    Build an image prompt spec using DSPy.
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

    # Reuse form planner's context builder so prompt inputs stay aligned.
    try:
        from programs.form_planner.orchestrator import _build_context as _build_context  # type: ignore
        from programs.form_planner.orchestrator import _compact_json as _compact_json  # type: ignore
        from programs.form_planner.orchestrator import _configure_dspy as _configure_dspy  # type: ignore
        from programs.form_planner.orchestrator import _make_dspy_lm as _make_dspy_lm  # type: ignore
    except Exception:
        return {
            "ok": False,
            "error": "Image prompt builder unavailable (context imports failed)",
            "requestId": request_id,
        }

    context: Dict[str, Any] = {}
    try:
        context = _build_context(payload) or {}
    except Exception as e:
        return {
            "ok": False,
            "error": f"Failed to build prompt context: {type(e).__name__}: {e}",
            "requestId": request_id,
        }

    lm_cfg = _make_dspy_lm()
    if not lm_cfg:
        return {
            "ok": False,
            "error": "DSPy LM not configured",
            "requestId": request_id,
        }

    try:
        import dspy

        from programs.image_generator.image_prompt_module import ImagePromptModule
        from programs.image_generator.signatures.image_prompt import ImagePromptSpec
    except Exception as e:
        return {
            "ok": False,
            "error": f"DSPy image prompt setup failed: {type(e).__name__}: {e}",
            "requestId": request_id,
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
        return {
            "ok": False,
            "error": "DSPy returned invalid prompt JSON",
            "requestId": request_id,
            "dspyRaw": str(raw)[:500],
        }

    try:
        spec = ImagePromptSpec.model_validate(obj).model_dump(by_alias=True)
    except Exception:
        return {
            "ok": False,
            "error": "DSPy returned prompt JSON that does not match schema",
            "requestId": request_id,
            "dspyRaw": str(raw)[:500],
        }

    return {"ok": True, "requestId": request_id, "prompt": spec}
