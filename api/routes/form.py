from __future__ import annotations

import os
import time
from typing import Any, AsyncIterator, Dict

import anyio
from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import ValidationError

from api.contract import ui_step_schema_json, schema_version
from api.models import MinimalFormRequest
from api.supabase_client import build_planner_payload_from_supabase
from api.utils import normalize_step_id, sse, sse_padding
from app.pipeline.pipeline import next_steps_jsonl, stream_next_steps_jsonl

router = APIRouter(prefix="/api/form", tags=["form"])


try:
    BaseExceptionGroup
except NameError:  # pragma: no cover - Python < 3.11
    BaseExceptionGroup = Exception


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


def _minimize_success_response(result: Any) -> Any:
    """
    For successful responses, return only the two frontend-required sections:
    `miniSteps` and `formPlan`.

    If an error is present, return the original payload for debuggability.
    """
    def _prune_empty(v: Any) -> Any:
        if isinstance(v, dict):
            out: Dict[str, Any] = {}
            for k, vv in v.items():
                pv = _prune_empty(vv)
                if pv is None:
                    continue
                if pv == "":
                    continue
                if isinstance(pv, dict) and not pv:
                    continue
                if isinstance(pv, list) and not pv:
                    continue
                out[k] = pv
            # Backward compat: rename deprecated `blueprint` key to `metadata`
            if "metadata" not in out and "blueprint" in out:
                out["metadata"] = out.pop("blueprint")
            return out
        if isinstance(v, list):
            out_list = []
            for item in v:
                pv = _prune_empty(item)
                if pv is None:
                    continue
                if pv == "":
                    continue
                if isinstance(pv, dict) and not pv:
                    continue
                if isinstance(pv, list) and not pv:
                    continue
                out_list.append(pv)
            return out_list
        return v

    def _minimize_form_plan(v: Any) -> Any:
        """
        Prefer the server-produced compact `formPlan` object.
        Backward-compat: if a legacy `FormPlanItem[]` list appears, collapse it to keys only.
        """
        if isinstance(v, list):
            keys: list[str] = []
            for item in v:
                if not isinstance(item, dict):
                    continue
                k = str(item.get("key") or "").strip()
                if k:
                    keys.append(k)
            return {"v": 1, "keys": keys}
        if isinstance(v, dict):
            return _prune_empty(v)
        return v

    if not isinstance(result, dict):
        return result
    if result.get("error") or result.get("ok") is False:
        return result
    mini_steps = result.get("miniSteps")
    if not isinstance(mini_steps, list):
        mini_steps = []
    # Drop null/empty fields from steps to reduce payload size, while preserving any populated fields.
    mini_steps = [
        _prune_empty(step) if isinstance(step, dict) else step for step in mini_steps
    ]
    out: Dict[str, Any] = {"formPlan": _minimize_form_plan(result.get("formPlan")), "miniSteps": mini_steps}
    # Deterministic/structural step placement instructions (uploads, CTAs, etc).
    deterministic_plan = result.get("deterministicPlacements")
    if isinstance(deterministic_plan, dict) and deterministic_plan:
        out["deterministicPlacements"] = _prune_empty(deterministic_plan)
    return out


@router.post("")
async def form(request: Request, body: Dict[str, Any] = Body(...)) -> Response:
    """
    Form batch generation.

    Accepts minimal format only (session, currentBatch, state - backend fetches rest from Supabase).

    Note: streaming (SSE) is disabled by default. Set `AI_FORM_ENABLE_STREAMING=true` to re-enable.
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
            batch_policy=getattr(minimal.state, "batch_policy", None),
            psychology_plan=getattr(minimal.state, "psychology_plan", None),
            batch_number=minimal.current_batch.batch_number,
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
        # Pass through session info for frontend verification
        payload["session"] = {
            "sessionId": minimal.session_id,
            "instanceId": minimal.instance_id
        }
    except ValidationError as e:
        import traceback
        print(f"[form] ValidationError: {e}", flush=True)
        print(f"[form] Traceback: {traceback.format_exc()}", flush=True)
        return JSONResponse(
            {"ok": False, "error": f"Invalid request: {e}"},
            status_code=422,
        )
    except Exception as e:
        import traceback
        print(f"[form] Exception in form route: {e}", flush=True)
        print(f"[form] Traceback: {traceback.format_exc()}", flush=True)
        return JSONResponse(
            {"ok": False, "error": f"Failed to fetch from Supabase: {str(e)}"},
            status_code=500,
        )

    streaming_enabled = (os.getenv("AI_FORM_ENABLE_STREAMING") or "").strip().lower() in {"1", "true", "yes"}
    wants_stream = False
    if streaming_enabled:
        if request.query_params.get("stream", "").lower() in {"1", "true", "yes"}:
            wants_stream = True
        elif "text/event-stream" in (request.headers.get("accept") or ""):
            wants_stream = True

    if wants_stream:
        async def gen() -> AsyncIterator[str]:
            yield sse_padding(2048)
            yield sse("open", {"ts": int(time.time() * 1000)})
            try:
                async for event in stream_next_steps_jsonl(payload):
                    # Keep streamed step events intact; minimize the final meta payload.
                    if event.get("event") == "meta":
                        meta = _normalize_ministeps_in_result(event.get("data"))
                        yield sse("meta", _minimize_success_response(meta))
                    else:
                        yield sse(event["event"], event["data"])
            except BaseExceptionGroup as e:
                yield sse("error", {"message": str(e), "type": type(e).__name__})
            except Exception as e:
                yield sse("error", {"message": str(e), "type": type(e).__name__})

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

    stream_requested = (
        request.query_params.get("stream", "").lower() in {"1", "true", "yes"}
        or ("text/event-stream" in (request.headers.get("accept") or ""))
    )
    headers = {"X-Streaming-Disabled": "1"} if (stream_requested and not streaming_enabled) else None

    try:
        result = await anyio.to_thread.run_sync(lambda: next_steps_jsonl(payload))
        result = _normalize_ministeps_in_result(result)
        result = _minimize_success_response(result)
        if isinstance(result, dict) and result.get("error"):
            print(f"[form] DSPy returned error: {result.get('error')}", flush=True)
        return JSONResponse(result, headers=headers)
    except Exception as e:
        import traceback
        print(f"[form] Exception in next_steps_jsonl: {e}", flush=True)
        print(f"[form] Traceback: {traceback.format_exc()}", flush=True)
        return JSONResponse(
            {"ok": False, "error": f"Internal error: {str(e)}"},
            status_code=500,
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
