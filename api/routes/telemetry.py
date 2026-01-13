from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from api.models import FeedbackEvent, TelemetryEvent
from api.supabase_client import insert_step_feedback, insert_telemetry_event

router = APIRouter(prefix="/api", tags=["telemetry"])


def _normalize_payload(payload: Optional[Dict[str, Any]], timestamp: Optional[int]) -> Optional[Dict[str, Any]]:
    if payload is None and timestamp is None:
        return None
    out: Dict[str, Any] = {}
    if isinstance(payload, dict):
        out.update(payload)
    if timestamp is not None:
        out["timestamp"] = timestamp
    return out


@router.post("/telemetry")
async def ingest_telemetry(body: Any = Body(...)) -> JSONResponse:
    items = body if isinstance(body, list) else [body]
    rows: List[Dict[str, Any]] = []

    try:
        for item in items:
            evt = TelemetryEvent.model_validate(item)
            payload_json = _normalize_payload(evt.payload, evt.timestamp)
            rows.append(
                {
                    "session_id": evt.session_id,
                    "instance_id": evt.instance_id,
                    "event_type": evt.event_type,
                    "step_id": evt.step_id,
                    "batch_id": evt.batch_id,
                    "model_request_id": evt.model_request_id,
                    "payload_json": payload_json,
                }
            )
    except ValidationError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=422)

    inserted = 0
    for row in rows:
        if insert_telemetry_event(row):
            inserted += 1

    return JSONResponse({"ok": True, "count": inserted})


@router.post("/feedback")
async def ingest_feedback(body: Any = Body(...)) -> JSONResponse:
    items = body if isinstance(body, list) else [body]
    rows: List[Dict[str, Any]] = []

    try:
        for item in items:
            evt = FeedbackEvent.model_validate(item)
            payload_json = _normalize_payload(evt.payload, evt.timestamp)
            rows.append(
                {
                    "session_id": evt.session_id,
                    "instance_id": evt.instance_id,
                    "step_id": evt.step_id,
                    "model_request_id": evt.model_request_id,
                    "source": evt.source,
                    "rating": evt.rating,
                    "vote": evt.vote,
                    "tags": evt.tags,
                    "comment": evt.comment,
                    "send_to_dataset": evt.send_to_dataset,
                    "payload_json": payload_json,
                }
            )
    except ValidationError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=422)

    inserted = 0
    for row in rows:
        if insert_step_feedback(row):
            inserted += 1

    return JSONResponse({"ok": True, "count": inserted})
