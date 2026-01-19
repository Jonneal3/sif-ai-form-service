import asyncio
import sys


def test_safe_stream_aclose_swallows_exception_group():
    if sys.version_info < (3, 11):
        return

    from app.pipeline.form_pipeline import _safe_stream_aclose

    class DummyStream:
        async def aclose(self) -> None:
            raise BaseExceptionGroup("boom", [ValueError("inner")])

    asyncio.run(_safe_stream_aclose(DummyStream()))

