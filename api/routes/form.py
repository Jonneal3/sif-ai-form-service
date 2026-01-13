from __future__ import annotations

import time
from typing import Any, AsyncIterator, Dict, Optional

import anyio
from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import ValidationError

from api.contract import ui_step_schema_json, schema_version
from api.models import MinimalFormRequest
from api.supabase_client import build_planner_payload_from_supabase, insert_model_trace
from api.utils import normalize_step_id, sse, sse_padding
from flow_planner import next_steps_jsonl, stream_next_steps_jsonl

router = APIRouter(prefix="/api/form", tags=["form"])


def _normalize_ministeps_in_result(result: Any) -> Any:
    """Normalize step ids in the result payload."""
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
async def form(request: Request, body: Dict[str, Any] = Body(...)) -> Response:
    """
    Form batch generation.

    Accepts minimal format only (session, currentBatch, state - backend fetches rest from Supabase).
    """
    # Minimal format - fetch from Supabase
    try:
        minimal = MinimalFormRequest.model_validate(body)
        payload = await build_planner_payload_from_supabase(
            session_id=minimal.session_id,
            instance_id=minimal.instance_id,
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
            max_steps_override=minimal.max_steps,
            allowed_step_types_override=minimal.allowed_step_types,
            required_uploads=minimal.required_uploads,
            items=minimal.items,
        )
        payload["currentBatch"] = minimal.current_batch.model_dump(by_alias=True)
        if minimal.request:
            payload["request"] = minimal.request.model_dump(by_alias=True)
        # Add request metadata
        if minimal.schema_version:
            payload["schemaVersion"] = minimal.schema_version
    except ValidationError as e:
        return JSONResponse(
            {"ok": False, "error": f"Invalid request: {e}"},
            status_code=422,
        )
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": f"Failed to fetch from Supabase: {str(e)}"},
            status_code=500,
        )

    wants_stream = False
    if request.query_params.get("stream", "").lower() in {"1", "true", "yes"}:
        wants_stream = True
    elif "text/event-stream" in (request.headers.get("accept") or ""):
        wants_stream = True

    if wants_stream:
        start_time = time.time()

        async def gen() -> AsyncIterator[str]:
            meta_payload: Optional[Dict[str, Any]] = None
            error_payload: Optional[Dict[str, Any]] = None
            yield sse_padding(2048)
            yield sse("open", {"ts": int(time.time() * 1000)})
            try:
                async for event in stream_next_steps_jsonl(payload):
                    if event.get("event") == "meta" and isinstance(event.get("data"), dict):
                        meta_payload = event.get("data")
                    if event.get("event") == "error" and isinstance(event.get("data"), dict):
                        error_payload = event.get("data")
                    yield sse(event["event"], event["data"])
            except Exception as e:
                yield sse("error", {"message": str(e), "type": type(e).__name__})
            finally:
                latency_ms = int((time.time() - start_time) * 1000)
                trace_payload = meta_payload or error_payload
                if trace_payload:
                    insert_model_trace(payload, trace_payload, latency_ms)

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

    start_time = time.time()
    result = await anyio.to_thread.run_sync(lambda: next_steps_jsonl(payload))
    latency_ms = int((time.time() - start_time) * 1000)
    result = _normalize_ministeps_in_result(result)
    insert_model_trace(payload, result, latency_ms)
    return JSONResponse(result)


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
