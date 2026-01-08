from __future__ import annotations

import time
from typing import Any, AsyncIterator, Dict, Set

import anyio
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from api.models import FlowNewBatchRequest
from api.utils import normalize_step_id, sse, sse_comment, sse_padding
from flow_planner import next_steps_jsonl

router = APIRouter(prefix="/flow", tags=["flow"])


def _normalize_ministeps_in_result(result: Any) -> Any:
    # Keep non-streaming output consistent with streaming normalization.
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


@router.post("/new-batch")
async def new_batch_json(body: FlowNewBatchRequest) -> JSONResponse:
    """
    Non-streaming debug endpoint.
    """
    payload = body.model_dump()
    result = await anyio.to_thread.run_sync(lambda: next_steps_jsonl(payload, stream=False))
    result = _normalize_ministeps_in_result(result)
    return JSONResponse(result)


@router.post("/new-batch/stream")
async def new_batch_stream(body: FlowNewBatchRequest) -> StreamingResponse:
    """
    Streaming endpoint (SSE).
    """
    payload = body.model_dump()

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


