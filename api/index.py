from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Dict, Set

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from flow_planner import next_steps_jsonl

app = FastAPI(title="sif-ai-form-service", version="0.1.0")


def _normalize_step_id(step_id: str) -> str:
    """
    Canonicalize step ids to match the Next.js side:
      - underscores -> hyphens
      - preserve leading `step-` prefix
    """
    t = str(step_id or "").strip()
    if not t:
        return t
    return t.replace("_", "-")


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "service": "sif-ai-form-service", "ts": int(time.time() * 1000)}


@app.post("/flow/new-batch")
async def new_batch_json(request: Request) -> JSONResponse:
    """
    Non-streaming debug endpoint.
    """
    payload = await request.json()
    result = await anyio.to_thread.run_sync(lambda: next_steps_jsonl(payload, stream=False))
    return JSONResponse(result)


@app.post("/flow/new-batch/stream")
async def new_batch_stream(request: Request) -> StreamingResponse:
    """
    Streaming endpoint (SSE).

    Emits:
      - event: mini_step (one validated mini step object per event)
      - event: meta (final meta object)
      - event: error (best-effort error envelope)
    """
    payload = await request.json()

    async def gen() -> AsyncIterator[str]:
        seen: Set[str] = set()
        t0 = time.time()
        try:
            result = await anyio.to_thread.run_sync(lambda: next_steps_jsonl(payload, stream=False))
            mini_steps = result.get("miniSteps") if isinstance(result, dict) else []
            if not isinstance(mini_steps, list):
                mini_steps = []

            for mini in mini_steps:
                if not isinstance(mini, dict):
                    continue
                sid = _normalize_step_id(str(mini.get("id") or ""))
                if sid:
                    mini = dict(mini)
                    mini["id"] = sid
                if sid and sid in seen:
                    continue
                if sid:
                    seen.add(sid)
                yield _sse("mini_step", mini)

            meta = result if isinstance(result, dict) else {"ok": True}
            meta = dict(meta)
            meta["latencyMs_service"] = int((time.time() - t0) * 1000)
            yield _sse("meta", meta)
        except Exception as e:
            yield _sse(
                "error",
                {
                    "message": str(e),
                    "type": type(e).__name__,
                },
            )

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Best-effort hint to avoid buffering by some proxies
            "X-Accel-Buffering": "no",
        },
    )


