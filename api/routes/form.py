from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from api.contract import ui_step_schema_json, schema_version
from api.models import FlowNewBatchRequest
from api.routes.flow import new_batch_json, new_batch_stream

router = APIRouter(prefix="/api/form", tags=["form"])


@router.post("")
async def form(body: FlowNewBatchRequest) -> JSONResponse:
    """
    Friendly alias for the planner (non-streaming):
      POST /api/form

    Equivalent to:
      POST /flow/new-batch
    """
    return await new_batch_json(body)


@router.post("/stream")
async def form_stream(body: FlowNewBatchRequest) -> StreamingResponse:
    """
    Friendly alias for the planner (SSE):
      POST /api/form/stream

    Equivalent to:
      POST /flow/new-batch/stream
    """
    return await new_batch_stream(body)


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


