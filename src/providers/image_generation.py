from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests


_LOG_VERBOSE_KEY = "IMAGE_LOG_DETAILED_PAYLOADS"
_LOG_VERBOSE_CACHE: Optional[bool] = None
_LOG_JSON_LIMIT = 6000


def _verbose_provider_logging_enabled() -> bool:
    global _LOG_VERBOSE_CACHE
    if _LOG_VERBOSE_CACHE is None:
        val = str(os.getenv(_LOG_VERBOSE_KEY) or "").strip().lower()
        _LOG_VERBOSE_CACHE = val in {"1", "true", "yes"}
    return _LOG_VERBOSE_CACHE


def _pretty_json(obj: Any, *, max_chars: int = _LOG_JSON_LIMIT) -> str:
    try:
        text = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(obj)
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text


def _log_provider(label: str, data: Any) -> None:
    if not _verbose_provider_logging_enabled():
        return
    try:
        text = _pretty_json(data)
        print(f"[image_generator] {label}:\n{text}", flush=True)
    except Exception:
        print(f"[image_generator] {label}: (unable to serialize)", flush=True)


def _replicate_api_token() -> str:
    token = str(os.getenv("REPLICATE_API_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("REPLICATE_API_TOKEN is not set (required for image generation)")
    return token


def _normalize_use_case(raw: Any) -> str:
    v = str(raw or "").strip().lower().replace("_", "-")
    if v in {"tryon", "try-on"}:
        return "tryon"
    if v == "scene-placement":
        return "scene-placement"
    if v == "scene":
        return "scene"
    return ""


def _replicate_default_model_id(*, use_case: Optional[str] = None) -> str:
    """
    Pick a Replicate model based on the instance `useCase` when available.

    Expected env vars (see `.env.local`):
    - TRYON_REPLICATE_MODEL_ID
    - SCENE_REPLICATE_MODEL_ID
    - SCENE_PLACEMENT_REPLICATE_MODEL_ID
    Fallback:
    - REPLICATE_MODEL_ID / REPLICATE_MODEL / REPLICATE_MODEL_VERSION
    """
    uc = _normalize_use_case(use_case)
    if uc == "tryon":
        m = str(os.getenv("TRYON_REPLICATE_MODEL_ID") or "").strip()
        if m:
            return m
    if uc == "scene":
        m = str(os.getenv("SCENE_REPLICATE_MODEL_ID") or "").strip()
        if m:
            return m
    if uc == "scene-placement":
        m = str(os.getenv("SCENE_PLACEMENT_REPLICATE_MODEL_ID") or "").strip()
        if m:
            return m

    # Accept a few common env names to reduce friction across repos.
    model_id = (
        str(os.getenv("REPLICATE_MODEL_ID") or "").strip()
        or str(os.getenv("REPLICATE_MODEL") or "").strip()
        or str(os.getenv("REPLICATE_MODEL_VERSION") or "").strip()
    )
    if not model_id:
        if uc == "tryon":
            raise RuntimeError(
                "No Replicate model configured for useCase='tryon'. "
                "Set TRYON_REPLICATE_MODEL_ID (recommended) or REPLICATE_MODEL_ID."
            )
        if uc == "scene":
            raise RuntimeError(
                "No Replicate model configured for useCase='scene'. "
                "Set SCENE_REPLICATE_MODEL_ID (recommended) or REPLICATE_MODEL_ID."
            )
        if uc == "scene-placement":
            raise RuntimeError(
                "No Replicate model configured for useCase='scene-placement'. "
                "Set SCENE_PLACEMENT_REPLICATE_MODEL_ID (recommended) or REPLICATE_MODEL_ID."
            )
        raise RuntimeError(
            "REPLICATE_MODEL_ID is not set (required for image generation). "
            "Example: black-forest-labs/flux-1.1-pro"
        )
    return model_id


def _replicate_create_prediction(*, model_id: str, input: Dict[str, Any]) -> Dict[str, Any]:
    token = _replicate_api_token()
    url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Replicate supports creating predictions by either:
    # - `version`: a version ID (hash) or sometimes a fully-qualified "owner/name:version"
    # - `model`: "owner/name"
    #
    # The widget often uses "owner/name" (e.g. "black-forest-labs/flux-1.1-pro"), so we
    # attempt `model` first for that shape, then fall back to `version`.
    model_str = str(model_id or "").strip()
    model_only = model_str.split(":", 1)[0] if ":" in model_str else model_str

    tried: list[tuple[str, Any]] = []
    payloads: list[Dict[str, Any]] = []
    if "/" in model_only and not all(c in "0123456789abcdef" for c in model_only.lower()):
        payloads.append({"model": model_only, "input": input})
    payloads.append({"version": model_str, "input": input})

    last_status = None
    last_data: Any = None
    for p in payloads:
        resp = requests.post(url, headers=headers, json=p, timeout=30)
        last_status = resp.status_code
        try:
            data = resp.json()
        except Exception:
            data = {"error": resp.text[:800]}
        tried.append((("model" if "model" in p else "version"), p.get("model") or p.get("version")))
        last_data = data
        if resp.ok:
            if not isinstance(data, dict) or not data.get("id"):
                raise RuntimeError(f"Replicate create returned unexpected payload: {data}")
            return data

    raise RuntimeError(f"Replicate create failed ({last_status}) tried={tried}: {last_data}")


def _replicate_get_prediction(prediction_id: str) -> Dict[str, Any]:
    token = _replicate_api_token()
    url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
    resp = requests.get(
        url,
        headers={"Authorization": f"Token {token}", "Accept": "application/json"},
        timeout=30,
    )
    try:
        data = resp.json()
    except Exception:
        data = {"error": resp.text[:800]}
    if not resp.ok:
        raise RuntimeError(f"Replicate get failed ({resp.status_code}): {data}")
    if not isinstance(data, dict) or not data.get("id"):
        raise RuntimeError(f"Replicate get returned unexpected payload: {data}")
    return data


def _replicate_wait_for_completion(prediction_id: str, *, timeout_sec: float) -> Dict[str, Any]:
    deadline = time.time() + max(5.0, float(timeout_sec or 0))
    last: Dict[str, Any] = {}
    while time.time() < deadline:
        last = _replicate_get_prediction(prediction_id)
        status = str(last.get("status") or "").lower()
        if status in {"succeeded", "failed", "canceled"}:
            return last
        time.sleep(1.0)
    return {**last, "status": "timeout", "error": "Prediction timed out"}


def _normalize_replicate_output_to_urls(output: Any) -> List[str]:
    # Replicate commonly returns either:
    # - a string URL
    # - an array of string URLs
    # - (rarely) objects; keep only string-ish values
    if output is None:
        return []
    if isinstance(output, str):
        s = output.strip()
        return [s] if s else []
    if isinstance(output, list):
        out: List[str] = []
        for item in output:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                # Best-effort: some models may return { url: "..." }
                u = item.get("url") if isinstance(item.get("url"), str) else None
                if u and u.strip():
                    out.append(u.strip())
        return out
    return []


def generate_images(
    *,
    prompt: str,
    num_outputs: int = 1,
    output_format: str = "url",
    model_id: Optional[str] = None,
    use_case: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    reference_images: Optional[List[str]] = None,
    scene_image: Optional[str] = None,
    product_image: Optional[str] = None,
) -> Dict[str, Any]:
    n = max(1, min(8, int(num_outputs or 1)))
    prompt = str(prompt or "").strip()

    model = str(model_id or "").strip() or _replicate_default_model_id(use_case=use_case)
    timeout_sec = float(os.getenv("REPLICATE_TIMEOUT_SEC") or "60")

    # Minimal cross-model input. Replicate models differ; unknown keys are typically ignored.
    inp: Dict[str, Any] = {"prompt": prompt}
    # Common knobs (best-effort)
    inp["num_outputs"] = n
    if negative_prompt and str(negative_prompt).strip():
        inp["negative_prompt"] = str(negative_prompt).strip()
    if isinstance(width, int) and width > 0:
        inp["width"] = width
    if isinstance(height, int) and height > 0:
        inp["height"] = height
    if isinstance(num_inference_steps, int) and num_inference_steps > 0:
        inp["num_inference_steps"] = num_inference_steps
    if isinstance(guidance_scale, (int, float)) and float(guidance_scale) > 0:
        inp["guidance_scale"] = float(guidance_scale)
    if reference_images and isinstance(reference_images, list) and reference_images:
        # Some models expect `image` or `input_image`. Send both (ignored if unsupported).
        first = next((x for x in reference_images if isinstance(x, str) and x.strip()), None)
        if first:
            inp["image"] = first
            inp["input_image"] = first

    # Scene-placement / multi-image best-effort. Different Replicate models use different key names;
    # Replicate typically ignores unknown keys, so we can safely include a few common variants.
    if isinstance(scene_image, str) and scene_image.strip():
        scene_u = scene_image.strip()
        inp.setdefault("image", scene_u)
        inp.setdefault("input_image", scene_u)
        inp["scene_image"] = scene_u
        inp["background_image"] = scene_u
    if isinstance(product_image, str) and product_image.strip():
        prod_u = product_image.strip()
        inp["product_image"] = prod_u
        inp["subject_image"] = prod_u
        inp["overlay_image"] = prod_u

    _log_provider("replicate_request_payload", inp)
    created = _replicate_create_prediction(model_id=model, input=inp)
    prediction_id = str(created.get("id") or "")
    status = str(created.get("status") or "")
    output = created.get("output")
    urls = _normalize_replicate_output_to_urls(output)
    final = created

    # If output isn't ready yet, poll.
    if not urls and str(status).lower() not in {"succeeded", "failed", "canceled"}:
        final = _replicate_wait_for_completion(prediction_id, timeout_sec=timeout_sec)
        status = str(final.get("status") or status)
        # keep polling result in `final`

    # Pass through the raw Replicate prediction response (exact shape from Replicate API),
    # so callers can read `id`, `status`, `output`, `input`, etc.
    response_payload = final if isinstance(final, dict) else {"status": "failed", "error": "Invalid Replicate response"}
    _log_provider("replicate_response_payload", response_payload)
    return response_payload
