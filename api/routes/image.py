from __future__ import annotations

import os
import time
from typing import Any, Dict

import anyio
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from api.models import ImageRequest, MinimalImageRequest
from api.supabase_client import build_planner_payload_from_supabase, insert_telemetry_event
from image_planner import build_image_prompt
from modules.image_generation import generate_images

router = APIRouter(prefix="/api/image", tags=["image"])


@router.post("")
async def image(request_body: Dict[str, Any] = Body(...)) -> Response:
    """
    Image prompt + image generation.

    The frontend should call this after a batch is completed (i.e., after the user has answered
    the questions) to generate one or more image variants from the aggregated state.
    """
    req_v1: MinimalImageRequest | None = None
    req_v2: ImageRequest | None = None
    try:
        # Backward compatible envelope (session/currentBatch/state).
        if isinstance(request_body.get("session"), dict) and isinstance(request_body.get("currentBatch"), dict):
            req_v1 = MinimalImageRequest.model_validate(request_body)
        else:
            req_v2 = ImageRequest.model_validate(request_body)
    except ValidationError as e:
        return JSONResponse({"ok": False, "error": f"Invalid request: {e}"}, status_code=422)

    if req_v1:
        try:
            payload = await build_planner_payload_from_supabase(
                session_id=req_v1.session_id,
                instance_id=req_v1.instance_id,
                batch_id=req_v1.batch_id,
                batch_state={
                    "callsUsed": 0,
                    "maxCalls": 2,
                    "callsRemaining": 2,
                    "tokensTotalBudget": 0,
                    "tokensUsedSoFar": 0,
                    "satietySoFar": req_v1.state.satiety_current,
                    "satietyRemaining": req_v1.current_batch.satiety_remaining,
                    "batch1PredictedSatietyIfCompleted": req_v1.current_batch.satiety_target,
                    "plannedSatietyGainThisCall": 0,
                    "plannedSatietyAfterThisCall": 0,
                    "plannedStepIdsThisCall": [],
                    "missingHighImpactKeys": [],
                    "mustHaveCopyNeeded": {"budget": False, "uploads": []},
                },
                answers=req_v1.answers,
                asked_step_ids=req_v1.asked_step_ids,
                form_plan=req_v1.form_plan,
                batch_policy=getattr(req_v1.state, "batch_policy", None),
                psychology_plan=getattr(req_v1.state, "psychology_plan", None),
                batch_number=req_v1.current_batch.batch_number,
                personalization_summary=req_v1.personalization_summary,
                goal=req_v1.platform_goal,
                business_context=req_v1.business_context,
                industry=req_v1.industry,
                service=req_v1.service,
                max_steps_override=req_v1.current_batch.max_steps,
                allowed_step_types_override=req_v1.current_batch.allowed_component_types,
                required_uploads=req_v1.current_batch.required_uploads,
                items=None,
            )
            payload["currentBatch"] = req_v1.current_batch.model_dump(by_alias=True)
            payload["session"] = {"sessionId": req_v1.session_id, "instanceId": req_v1.instance_id}
            if req_v1.request:
                payload["request"] = req_v1.request.model_dump(by_alias=True)
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Failed to build payload: {str(e)}"}, status_code=500)
    else:
        # New minimal shape: instanceId + useCase + either prompt OR (stepDataSoFar + config)
        assert req_v2 is not None
        cfg_raw = request_body.get("config") if isinstance(request_body.get("config"), dict) else {}
        cfg = req_v2.config.model_dump(by_alias=True) if req_v2.config else {}
        payload = {
            "batchId": cfg.get("batchId") or cfg_raw.get("batchId") or cfg_raw.get("batch_id") or "ImageGen",
            "useCase": req_v2.use_case,
            "platformGoal": cfg.get("platformGoal") or cfg_raw.get("platformGoal") or cfg_raw.get("platform_goal") or "",
            "businessContext": cfg.get("businessContext") or cfg_raw.get("businessContext") or cfg_raw.get("business_context") or "",
            "industry": cfg.get("industry") or cfg_raw.get("industry") or "General",
            "service": cfg.get("service") or cfg_raw.get("service") or "",
            "personalizationSummary": cfg.get("personalizationSummary") or cfg_raw.get("personalizationSummary") or cfg_raw.get("personalization_summary") or "",
            "alreadyAskedKeys": cfg.get("alreadyAskedKeys") or cfg_raw.get("alreadyAskedKeys") or cfg_raw.get("already_asked_keys") or [],
            "formPlan": cfg.get("formPlan") or cfg_raw.get("formPlan") or cfg_raw.get("form_plan") or [],
            "batchState": cfg.get("batchState") or cfg_raw.get("batchState") or cfg_raw.get("batch_state") or {},
            "requiredUploads": [],
            "maxSteps": 1,
            "allowedMiniTypes": ["multiple_choice"],
            "stepDataSoFar": dict(req_v2.step_data_so_far or {}),
        }
        # Put image inputs into known_answers so prompt-building can reference them.
        if req_v2.user_image:
            payload["stepDataSoFar"]["userImage"] = req_v2.user_image
        if req_v2.product_image:
            payload["stepDataSoFar"]["productImage"] = req_v2.product_image
        if req_v2.scene_image:
            payload["stepDataSoFar"]["sceneImage"] = req_v2.scene_image
        if req_v2.reference_images:
            payload["stepDataSoFar"]["referenceImages"] = req_v2.reference_images
        if req_v2.session_id:
            payload["session"] = {"sessionId": req_v2.session_id, "instanceId": req_v2.instance_id}

    start_ms = int(time.time() * 1000)
    prompt_result: Dict[str, Any] = {"ok": True, "requestId": f"image_{start_ms}"}
    prompt_text = ""
    negative_prompt = ""
    metadata = None

    if req_v1:
        # v1 always builds a prompt (unless overridden).
        prompt_result = await anyio.to_thread.run_sync(
            lambda: build_image_prompt(payload, prompt_template=req_v1.image.prompt_template)
        )
        if not prompt_result.get("ok"):
            return JSONResponse(prompt_result, status_code=500)
        prompt_spec = prompt_result.get("prompt") if isinstance(prompt_result, dict) else None
        prompt_text = prompt_spec.get("prompt") if isinstance(prompt_spec, dict) else ""
        negative_prompt = prompt_spec.get("negativePrompt") if isinstance(prompt_spec, dict) else ""
        metadata = prompt_spec.get("metadata") if isinstance(prompt_spec, dict) else None
    else:
        assert req_v2 is not None
        if req_v2.prompt and str(req_v2.prompt).strip():
            prompt_text = str(req_v2.prompt).strip()
            negative_prompt = str(req_v2.negative_prompt or "").strip()
            prompt_result = {
                "ok": True,
                "requestId": f"image_{start_ms}",
                "prompt": {
                    "prompt": prompt_text,
                    "negativePrompt": negative_prompt,
                    "styleTags": [],
                    "metadata": {"builder": "client_prompt"},
                },
            }
        else:
            prompt_result = await anyio.to_thread.run_sync(lambda: build_image_prompt(payload))
            if not prompt_result.get("ok"):
                return JSONResponse(prompt_result, status_code=500)
            prompt_spec = prompt_result.get("prompt") if isinstance(prompt_result, dict) else None
            prompt_text = prompt_spec.get("prompt") if isinstance(prompt_spec, dict) else ""
            # Allow client to override negativePrompt even when prompt is built.
            negative_prompt = str(req_v2.negative_prompt or (prompt_spec.get("negativePrompt") if isinstance(prompt_spec, dict) else "") or "").strip()
            metadata = prompt_spec.get("metadata") if isinstance(prompt_spec, dict) else None

    try:
        provider = os.getenv("IMAGE_PROVIDER") or ("mock" if req_v1 else "mock")
        num_variants = req_v1.image.num_variants if req_v1 else req_v2.num_outputs
        return_format = req_v1.image.return_format if req_v1 else req_v2.output_format
        gen = await anyio.to_thread.run_sync(
            lambda: generate_images(
                prompt=str(prompt_text or ""),
                num_variants=num_variants,
                provider=provider,
                size=(req_v1.image.size if req_v1 else None),
                return_format=return_format,
                metadata=metadata if isinstance(metadata, dict) else None,
            )
        )
    except Exception as e:
        return JSONResponse(
            {
                "ok": False,
                "requestId": prompt_result.get("requestId"),
                "error": f"Image generation failed: {str(e)}",
                "prompt": (prompt_result.get("prompt") if isinstance(prompt_result, dict) else None),
            },
            status_code=500,
        )

    total_ms = int(time.time() * 1000) - start_ms
    out_prompt = prompt_result.get("prompt") if isinstance(prompt_result, dict) else None
    use_case = str(payload.get("useCase") or "")
    # Response contract requested by the UI: {images, predictionId?, prompt?, negativePrompt?, metrics?}
    out = {
        "images": gen.images,
        "predictionId": prompt_result.get("requestId"),
        "prompt": (out_prompt.get("prompt") if isinstance(out_prompt, dict) else prompt_text) or None,
        "negativePrompt": (negative_prompt or (out_prompt.get("negativePrompt") if isinstance(out_prompt, dict) else "")) or None,
        "metrics": {**(gen.metrics or {}), "totalTimeMs": total_ms, "useCase": use_case},
    }

    # Best-effort telemetry storage (safe when Supabase isn't configured).
    try:
        session_id = req_v1.session_id if req_v1 else (req_v2.session_id or "")
        instance_id = req_v1.instance_id if req_v1 else req_v2.instance_id
        batch_id = req_v1.batch_id if req_v1 else (payload.get("batchId") or "ImageGen")
        insert_telemetry_event(
            {
                "session_id": session_id,
                "instance_id": instance_id,
                "event_type": "image_generated",
                "batch_id": batch_id,
                "payload_json": {
                    "predictionId": out.get("predictionId"),
                    "numOutputs": (req_v1.image.num_variants if req_v1 else req_v2.num_outputs),
                    "provider": (os.getenv("IMAGE_PROVIDER") or "mock"),
                    "metrics": out.get("metrics"),
                },
            }
        )
    except Exception:
        pass

    return JSONResponse(out)
