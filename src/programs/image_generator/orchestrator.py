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

from programs.image_generator.prompt_builder import build_image_prompt_text, extract_negative_prompt, extract_reference_images


_VERBOSE_LOG_ENV = "IMAGE_LOG_DETAILED_PAYLOADS"
_VERBOSE_LOG_CACHE: Optional[bool] = None
_PRETTY_JSON_LIMIT = 6000


def _verbose_logging_enabled() -> bool:
    global _VERBOSE_LOG_CACHE
    if _VERBOSE_LOG_CACHE is None:
        val = str(os.getenv(_VERBOSE_LOG_ENV) or "").strip().lower()
        _VERBOSE_LOG_CACHE = val in {"1", "true", "yes"}
    return _VERBOSE_LOG_CACHE


def _pretty_json(obj: Any, *, max_chars: int = _PRETTY_JSON_LIMIT) -> str:
    try:
        text = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(obj)
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text


def _log_verbose(label: str, data: Any) -> None:
    if not _verbose_logging_enabled():
        return
    try:
        text = _pretty_json(data)
        print(f"[image_generator] {label}:\n{text}", flush=True)
    except Exception:
        print(f"[image_generator] {label}: (unable to render payload)", flush=True)


def _best_effort_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _tail_lines(text: str, *, max_lines: int = 30, max_chars: int = 1200) -> str:
    t = str(text or "").strip()
    if not t:
        return ""
    lines = t.splitlines()
    tail = "\n".join(lines[-max_lines:])
    tail = tail.strip()
    if len(tail) > max_chars:
        tail = tail[-max_chars:].lstrip()
    return tail


def _extract_provider_error(provider_resp: Any) -> str:
    """
    Best-effort extraction of a human-readable error message from a provider response.
    Works for Replicate prediction objects and our timeout shape.
    """
    if not isinstance(provider_resp, dict):
        return ""

    # Common fields across providers/shapes.
    for k in ("message", "error", "detail", "title"):
        v = provider_resp.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Replicate: sometimes `error` is null but logs contain the failure reason.
    # Only treat logs as an error surface when status indicates failure/cancel/timeout.
    status = str(provider_resp.get("status") or "").strip().lower()
    if status in {"failed", "timeout", "canceled"}:
        logs = provider_resp.get("logs")
        if isinstance(logs, str) and logs.strip():
            return _tail_lines(logs, max_lines=24, max_chars=1200)

    return ""


def build_image_prompt(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build an image prompt spec.

    Default behavior is deterministic prompt construction from `stepDataSoFar` + reference images.
    Set `IMAGE_PROMPT_MODE=dspy` to use the legacy DSPy-generated prompt JSON path.
    """
    request_id = f"image_prompt_{int(time.time() * 1000)}"

    mode = str(os.getenv("IMAGE_PROMPT_MODE") or "deterministic").strip().lower()
    if mode != "dspy":
        try:
            from programs.image_generator.signatures.image_prompt import ImagePromptSpec
        except Exception as e:
            return {
                "ok": False,
                "error": f"Image prompt schema unavailable: {type(e).__name__}: {e}",
                "requestId": request_id,
            }

        # Deterministic prompt construction.
        try:
            obj = build_image_prompt_text(payload)
            spec = ImagePromptSpec.model_validate(obj).model_dump(by_alias=True)
        except Exception as e:
            return {
                "ok": False,
                "error": f"Failed to build image prompt: {type(e).__name__}: {e}",
                "requestId": request_id,
            }
        return {"ok": True, "requestId": request_id, "prompt": spec}

    # Legacy: DSPy-generated prompt JSON.
    # Reuse form planner's context builder so prompt inputs stay aligned.
    try:
        from programs.form_pipeline.orchestrator import _build_context as _build_context  # type: ignore
        from programs.form_pipeline.orchestrator import _compact_json as _compact_json  # type: ignore
        from programs.form_pipeline.orchestrator import _configure_dspy as _configure_dspy  # type: ignore
        from programs.form_pipeline.orchestrator import _make_dspy_lm as _make_dspy_lm  # type: ignore
    except Exception as e:
        return {
            "ok": False,
            "error": f"Image prompt builder unavailable (context imports failed): {type(e).__name__}: {e}",
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
    - Build prompt via DSPy from the provided context
    - Call the image provider (Replicate)
    - Return `{ images: string[], predictionId }` for widget compatibility
    """
    request_id = f"image_{int(time.time() * 1000)}"

    _log_verbose("payload_received", payload)

    # Lightweight request log (safe: no tokens; prompt is truncated).
    try:
        session_id = payload.get("sessionId") or (payload.get("session") or {}).get("sessionId")
        instance_id = payload.get("instanceId") or (payload.get("session") or {}).get("instanceId")
        use_case = payload.get("useCase")
        model_id_log = payload.get("modelId") or payload.get("model_id")
        num_outputs_log = payload.get("numOutputs") or payload.get("num_outputs")
        ref_count = len(payload.get("referenceImages") or []) if isinstance(payload.get("referenceImages"), list) else 0
        print(
            "[image_generator] generate_image request",
            {
                "instanceId": str(instance_id or "")[:80] or None,
                "sessionId": str(session_id or "")[:80] or None,
                "useCase": str(use_case or "")[:40] or None,
                "modelId": str(model_id_log or "")[:120] or None,
                "numOutputs": num_outputs_log,
                "referenceImagesCount": ref_count,
            },
            flush=True,
        )
    except Exception:
        pass

    prompt_result = build_image_prompt(payload)
    if not isinstance(prompt_result, dict) or not prompt_result.get("ok"):
        return prompt_result

    prompt_obj = prompt_result.get("prompt") if isinstance(prompt_result.get("prompt"), dict) else {}
    _log_verbose("generated_prompt_spec", prompt_obj)
    prompt_text = ((prompt_obj.get("prompt") if isinstance(prompt_obj, dict) else "") or "").strip()

    # Prefer prompt-spec negativePrompt, fall back to payload negativePrompt.
    negative_prompt = None
    if isinstance(prompt_obj, dict) and isinstance(prompt_obj.get("negativePrompt"), str):
        negative_prompt = str(prompt_obj.get("negativePrompt") or "").strip() or None
    if not negative_prompt:
        negative_prompt = extract_negative_prompt(payload) or None

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

    num_outputs = (
        payload.get("numOutputs")
        or payload.get("num_outputs")
        or payload.get("gallery_max_images")
        or payload.get("galleryMaxImages")
        or 1
    )
    try:
        n = int(num_outputs)
    except Exception:
        n = 1

    model_id = payload.get("modelId") or payload.get("model_id") or None
    if not isinstance(model_id, str):
        model_id = None

    reference_images, scene_image, product_image = extract_reference_images(payload)
    reference_images_list: Optional[list[str]] = reference_images if reference_images else None

    width = _as_int(payload.get("width"))
    height = _as_int(payload.get("height"))
    num_inference_steps = _as_int(payload.get("numInferenceSteps") or payload.get("num_inference_steps"))
    guidance_scale = _as_float(payload.get("guidanceScale") or payload.get("guidance_scale"))

    # Provider call
    from providers.image_generation import generate_images  # local import (keeps module light)

    provider_name = "replicate"
    try:
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
            reference_images=reference_images_list,
            scene_image=scene_image,
            product_image=product_image,
        )
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(
            "[image_generator] generate_image provider_exception",
            {"provider": provider_name, "requestId": request_id, "error": msg[:500]},
            flush=True,
        )
        return {
            "ok": False,
            "error": "image_provider_exception",
            "message": msg,
            "provider": provider_name,
            "requestId": request_id,
            "status": "failed",
        }

    try:
        # We now pass-through the provider's response (Replicate prediction JSON).
        pred_id = provider_resp.get("id") if isinstance(provider_resp, dict) else None
        pred_status = str(provider_resp.get("status") or "") if isinstance(provider_resp, dict) else ""
        pred_out = provider_resp.get("output") if isinstance(provider_resp, dict) else None
        images_count = len(pred_out) if isinstance(pred_out, list) else (1 if isinstance(pred_out, str) else 0)
        err_msg = _extract_provider_error(provider_resp)
        print(
            "[image_generator] generate_image provider_response",
            {
                "provider": provider_name,
                "status": pred_status or None,
                "id": pred_id,
                "imagesCount": images_count,
                "hasError": bool(err_msg) or (str(pred_status).lower() in {"failed", "timeout", "canceled"}),
                "error": (err_msg[:220] + "…") if (isinstance(err_msg, str) and len(err_msg) > 220) else (err_msg or None),
            },
            flush=True,
        )
    except Exception:
        pass

    # Pass-through, but add a consistent `ok` + error surface for clients.
    if not isinstance(provider_resp, dict):
        return {
            "ok": False,
            "error": "invalid_provider_response",
            "message": "Image provider returned invalid response",
            "provider": provider_name,
            "requestId": request_id,
            "status": "failed",
        }

    status = str(provider_resp.get("status") or "").lower()
    if status in {"failed", "timeout", "canceled"}:
        msg = _extract_provider_error(provider_resp) or f"Image generation {status}."
        return {
            **provider_resp,
            "ok": False,
            "error": "image_generation_failed",
            "message": msg,
            "provider": provider_name,
            "requestId": request_id,
        }

    return {**provider_resp, "ok": True, "provider": provider_name, "requestId": request_id}
