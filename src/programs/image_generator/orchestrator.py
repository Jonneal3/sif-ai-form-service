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
        from programs.form_pipeline.orchestrator import _build_context as _build_context  # type: ignore
        from programs.form_pipeline.orchestrator import _compact_json as _compact_json  # type: ignore
        from programs.form_pipeline.orchestrator import _configure_dspy as _configure_dspy  # type: ignore
        from programs.form_pipeline.orchestrator import _make_dspy_lm as _make_dspy_lm  # type: ignore
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


def generate_image(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    End-to-end image generation:
    - Build prompt via DSPy (unless caller provides an explicit prompt)
    - Call the configured image provider (mock or Replicate)
    - Return `{ images: string[], predictionId }` for widget compatibility
    """
    # Caller override: allow directly supplied prompt strings (widget sometimes sends this).
    prompt_override = payload.get("prompt") if isinstance(payload.get("prompt"), str) else None
    prompt_override = prompt_override.strip() if isinstance(prompt_override, str) else None

    # Lightweight request log (safe: no tokens; prompt is truncated).
    try:
        session_id = payload.get("sessionId") or (payload.get("session") or {}).get("sessionId")
        instance_id = payload.get("instanceId") or (payload.get("session") or {}).get("instanceId")
        use_case = payload.get("useCase")
        model_id_log = payload.get("modelId") or payload.get("model_id")
        num_outputs_log = payload.get("numOutputs") or payload.get("num_outputs")
        ref_count = len(payload.get("referenceImages") or []) if isinstance(payload.get("referenceImages"), list) else 0
        prompt_preview = (prompt_override or "").replace("\n", " ").strip()[:160]
        print(
            "[image_generator] generate_image request",
            {
                "instanceId": str(instance_id or "")[:80] or None,
                "sessionId": str(session_id or "")[:80] or None,
                "useCase": str(use_case or "")[:40] or None,
                "modelId": str(model_id_log or "")[:120] or None,
                "numOutputs": num_outputs_log,
                "hasPrompt": bool(prompt_override),
                "promptPreview": prompt_preview or None,
                "referenceImagesCount": ref_count,
            },
            flush=True,
        )
    except Exception:
        pass

    # If the caller provided a prompt, don't require DSPy to be configured.
    if prompt_override:
        request_id = f"image_{int(time.time() * 1000)}"
        prompt_result: Dict[str, Any] = {
            "ok": True,
            "requestId": request_id,
            "prompt": {
                "prompt": prompt_override,
                "negativePrompt": "",
                "styleTags": [],
                "metadata": {"builder": "caller"},
            },
        }
    else:
        prompt_template = payload.get("promptTemplate")
        prompt_result = build_image_prompt(payload, prompt_template=prompt_template)
        if not isinstance(prompt_result, dict) or not prompt_result.get("ok"):
            return prompt_result

    prompt_obj = prompt_result.get("prompt") if isinstance(prompt_result.get("prompt"), dict) else {}
    prompt_text = ((prompt_obj.get("prompt") if isinstance(prompt_obj, dict) else "") or "").strip()

    # Prefer caller-provided negativePrompt, else use DSPy-generated negativePrompt
    negative_prompt = payload.get("negativePrompt") if isinstance(payload.get("negativePrompt"), str) else None
    if not negative_prompt and isinstance(prompt_obj, dict):
        negative_prompt = prompt_obj.get("negativePrompt") if isinstance(prompt_obj.get("negativePrompt"), str) else None

    # Wire through common widget fields
    def _as_int(v: Any) -> Optional[int]:
        try:
            if v is None:
                return None
            n = int(v)
            return n
        except Exception:
            return None

    def _as_float(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            return float(v)
        except Exception:
            return None

    num_outputs = payload.get("numOutputs") or payload.get("num_outputs") or 1
    try:
        n = int(num_outputs)
    except Exception:
        n = 1

    model_id = payload.get("modelId") or payload.get("model_id") or None
    if not isinstance(model_id, str):
        model_id = None

    reference_images = payload.get("referenceImages")
    if not isinstance(reference_images, list):
        reference_images = None
    else:
        reference_images = [str(x) for x in reference_images if isinstance(x, str) and x.strip()][:8]

    width = _as_int(payload.get("width"))
    height = _as_int(payload.get("height"))
    num_inference_steps = _as_int(payload.get("numInferenceSteps") or payload.get("num_inference_steps"))
    guidance_scale = _as_float(payload.get("guidanceScale") or payload.get("guidance_scale"))

    # Provider call
    from providers.image_generation import generate_images  # local import (keeps module light)

    provider_resp = generate_images(
        prompt=prompt_text,
        num_outputs=n,
        output_format=str(payload.get("outputFormat") or payload.get("output_format") or "url"),
        model_id=model_id,
        use_case=str(payload.get("useCase") or "").strip() or None,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        reference_images=reference_images,
    )

    try:
        # We now pass-through the provider's response (Replicate prediction JSON).
        pred_id = provider_resp.get("id") if isinstance(provider_resp, dict) else None
        pred_status = provider_resp.get("status") if isinstance(provider_resp, dict) else None
        pred_out = provider_resp.get("output") if isinstance(provider_resp, dict) else None
        images_count = len(pred_out) if isinstance(pred_out, list) else (1 if isinstance(pred_out, str) else 0)
        print(
            "[image_generator] generate_image provider_response",
            {
                "provider": str(os.getenv("IMAGE_PROVIDER") or "mock").lower(),
                "status": pred_status,
                "id": pred_id,
                "imagesCount": images_count,
                "hasError": bool(provider_resp.get("error")) if isinstance(provider_resp, dict) else None,
            },
            flush=True,
        )
    except Exception:
        pass

    # Pass-through: return exactly what the provider returned (Replicate prediction JSON).
    if not isinstance(provider_resp, dict):
        return {"status": "failed", "error": "Image provider returned invalid response"}
    return provider_resp
