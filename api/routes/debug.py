from __future__ import annotations

import time
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.utils import sse, sse_padding

router = APIRouter(tags=["debug"])


@router.get("/debug/stream-flush")
async def debug_stream_flush(count: int = 10, interval_ms: int = 500) -> StreamingResponse:
    """
    Flush-verification endpoint (SSE).
    """

    async def gen() -> AsyncIterator[str]:
        yield sse_padding(2048)
        yield sse("open", {"ts": int(time.time() * 1000)})
        n = max(1, min(int(count), 200))
        delay = max(10, min(int(interval_ms), 30_000)) / 1000.0
        import anyio

        for i in range(n):
            yield sse("tick", {"i": i, "ts": int(time.time() * 1000)})
            await anyio.sleep(delay)
        yield sse("done", {"ok": True, "count": n})

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


