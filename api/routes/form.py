from __future__ import annotations

import time
from typing import Any, AsyncIterator, Dict, Set

import anyio
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse, StreamingResponse

from api.contract import ui_step_schema_json, schema_version
from api.models import FormRequest, MinimalFormRequest
from api.supabase_client import build_planner_payload_from_supabase
from api.utils import normalize_step_id, sse, sse_comment, sse_padding
from flow_planner import next_steps_jsonl

router = APIRouter(prefix="/api/form", tags=["form"])


def _normalize_ministeps_in_result(result: Any) -> Any:
    """Keep non-streaming output consistent with streaming normalization."""
    try:
        mini_steps = result.get("miniSteps") if isinstance(result, dict) else None
        if isinstance(mini_steps, list):
            out_steps = []
            for mini in mini_steps:
                if not isinstance(mini, dict):
                    continue
                sid = normalize_step_id(str(mini.get("id") or ""))
                if sid:
                    mini = dict(mini)
                    mini["id"] = sid
                out_steps.append(mini)
            result = dict(result) if isinstance(result, dict) else {"ok": True}
            result["miniSteps"] = out_steps
    except Exception:
        pass
    return result


@router.post("")
async def form(body: Dict[str, Any] = Body(...)) -> JSONResponse:
    """
    Non-streaming endpoint for form batch generation.
    
    **Minimal format (recommended - only requires session, currentBatch, state):**
    ```json
    {
      "session": {
        "sessionId": "abc123",
        "instanceId": "xyz789"
      },
      "currentBatch": {
        "batchId": "ContextCore",
        "batchNumber": 1,
        "satietyTarget": 0.77,
        "satietyRemaining": 0.77,
        "maxSteps": 5,
        "allowedComponentTypes": ["choice"]
      },
      "state": {
        "satietyCurrent": 0.0,
        "answers": {},
        "askedStepIds": [],
        "formPlan": null,
        "personalizationSummary": ""
      }
    }
    ```
    
    Backend automatically fetches from Supabase:
    - Prompt context (goal, business context, industry, grounding, etc.)
    - Everything else needed for DSPy
    
    **Full format (backward compatible - for direct API usage without Supabase):**
    ```json
    {
      "context": { ... },
      "state": { ... },
      "constraints": { ... },
      "grounding": { ... }
    }
    ```
    """
    # Check if this is a minimal request (has session_id or sessionId)
    if "session_id" in body or "sessionId" in body:
        # Minimal format - fetch from Supabase
        try:
            minimal = MinimalFormRequest.model_validate(body)
            payload = await build_planner_payload_from_supabase(
                session_id=minimal.session_id,
                batch_id=minimal.batch_id,
                batch_state=minimal.batch_state,
                answers=minimal.answers,
                asked_step_ids=minimal.asked_step_ids,
                form_plan=minimal.form_plan,
                personalization_summary=minimal.personalization_summary,
                goal=minimal.platform_goal,
                business_context=minimal.business_context,
                industry=minimal.industry,
                service=minimal.service,
                grounding=minimal.grounding_preview,
                max_steps_override=minimal.max_steps,
                allowed_step_types_override=minimal.allowed_step_types,
                required_uploads=minimal.required_uploads,
                items=minimal.items,
            )
            # Add request metadata
            if minimal.schema_version:
                payload["schemaVersion"] = minimal.schema_version
        except Exception as e:
            return JSONResponse(
                {"ok": False, "error": f"Failed to fetch from Supabase: {str(e)}"},
                status_code=500
            )
    else:
        # Full format - backward compatible
        request = FormRequest.model_validate(body)
        payload = request.to_legacy_dict()
    
    result = await anyio.to_thread.run_sync(lambda: next_steps_jsonl(payload, stream=False))
    result = _normalize_ministeps_in_result(result)
    return JSONResponse(result)


@router.post("/stream")
async def form_stream(body: Dict[str, Any] = Body(...)) -> StreamingResponse:
    """
    Streaming endpoint (SSE) for form batch generation.
    
    Accepts minimal format (only requires session, currentBatch, state - backend fetches rest from Supabase)
    or full format (see `/api/form` docs).
    """
    # Check if this is a minimal request (has session_id or sessionId)
    if "session_id" in body or "sessionId" in body:
        # Minimal format - fetch from Supabase
        try:
            minimal = MinimalFormRequest.model_validate(body)
            payload = await build_planner_payload_from_supabase(
                session_id=minimal.session_id,
                batch_id=minimal.batch_id,
                batch_state=minimal.batch_state,
                answers=minimal.answers,
                asked_step_ids=minimal.asked_step_ids,
                form_plan=minimal.form_plan,
                personalization_summary=minimal.personalization_summary,
                goal=minimal.platform_goal,
                business_context=minimal.business_context,
                industry=minimal.industry,
                service=minimal.service,
                grounding=minimal.grounding_preview,
                max_steps_override=minimal.max_steps,
                allowed_step_types_override=minimal.allowed_step_types,
                required_uploads=minimal.required_uploads,
                items=minimal.items,
            )
            # Add request metadata
            if minimal.schema_version:
                payload["schemaVersion"] = minimal.schema_version
        except Exception as e:
            # Return error via SSE
            async def error_gen():
                yield sse_padding(2048)
                yield sse("error", {"message": f"Failed to fetch from Supabase: {str(e)}", "type": "SupabaseError"})
            return StreamingResponse(error_gen(), media_type="text/event-stream")
    else:
        # Full format - backward compatible
        request = FormRequest.model_validate(body)
        payload = request.to_legacy_dict()
    
    async def gen() -> AsyncIterator[str]:
        seen: Set[str] = set()
        t0 = time.time()
        yield sse_padding(2048)
        yield sse("open", {"ts": int(time.time() * 1000)})

        send, recv = anyio.create_memory_object_stream[str](max_buffer_size=200)
        done = anyio.Event()

        async def _compute_and_emit() -> None:
            try:
                result = await anyio.to_thread.run_sync(lambda: next_steps_jsonl(payload, stream=False))
                mini_steps = result.get("miniSteps") if isinstance(result, dict) else []
                if not isinstance(mini_steps, list):
                    mini_steps = []

                for mini in mini_steps:
                    if not isinstance(mini, dict):
                        continue
                    sid = normalize_step_id(str(mini.get("id") or ""))
                    if sid:
                        mini = dict(mini)
                        mini["id"] = sid
                    if sid and sid in seen:
                        continue
                    if sid:
                        seen.add(sid)
                    await send.send(sse("mini_step", mini))

                meta = result if isinstance(result, dict) else {"ok": True}
                meta = dict(meta)
                meta["latencyMs_service"] = int((time.time() - t0) * 1000)
                await send.send(sse("meta", meta))
            except Exception as e:
                await send.send(sse("error", {"message": str(e), "type": type(e).__name__}))
            finally:
                done.set()
                await send.aclose()

        async def _heartbeat() -> None:
            while not done.is_set():
                await anyio.sleep(5)
                try:
                    await send.send(sse_comment(f"hb {int(time.time() * 1000)}"))
                except Exception:
                    return

        async with anyio.create_task_group() as tg:
            tg.start_soon(_compute_and_emit)
            tg.start_soon(_heartbeat)
            async with recv:
                async for chunk in recv:
                    yield chunk

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/capabilities")
async def capabilities() -> JSONResponse:
    """
    Returns the currently deployed contract version + (optionally) the JSON Schema.

    Intended for the UI repo to detect contract drift.
    """
    return JSONResponse(
        {
            "schemaVersion": schema_version(),
            "uiStepSchema": ui_step_schema_json(),
        }
    )


