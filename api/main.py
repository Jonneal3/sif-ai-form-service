from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Dict

from dotenv import load_dotenv
from fastapi import APIRouter, Body, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def _repo_root() -> Path:
    # `api/main.py` lives at `<repo>/api/main.py`
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path() -> None:
    src = _repo_root() / "src"
    if not src.is_dir():
        return
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)


_ensure_src_on_path()

from programs.form_planner.orchestrator import next_steps_jsonl, stream_next_steps_jsonl  # noqa: E402
from programs.image_generator.orchestrator import build_image_prompt  # noqa: E402
from providers.image_generation import generate_images  # noqa: E402


def _load_contract_schema() -> Dict[str, Any]:
    root = _repo_root()
    schema_path = root / "shared" / "ai-form-contract" / "schema" / "ui_step.schema.json"
    version_path = root / "shared" / "ai-form-contract" / "schema" / "schema_version.txt"
    schema_obj: Dict[str, Any] = {}
    try:
        schema_obj = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception:
        schema_obj = {}
    try:
        schema_version = version_path.read_text(encoding="utf-8").strip()
    except Exception:
        schema_version = ""
    schema_version = schema_version or (schema_obj.get("schemaVersion") if isinstance(schema_obj, dict) else "")
    return {"schemaVersion": schema_version or "", "uiStepSchema": schema_obj}


def create_app() -> FastAPI:
    # Load `.env` + `.env.local` when present (local dev convenience).
    load_dotenv(_repo_root() / ".env", override=False)
    load_dotenv(_repo_root() / ".env.local", override=False)

    app = FastAPI(title="sif-ai-form-service")
    router = APIRouter(prefix="/v1/api")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"ok": True}

    @router.get("/form/capabilities")
    def capabilities() -> Dict[str, Any]:
        return {"ok": True, **_load_contract_schema()}

    @router.post("/form")
    async def form(request: Request, payload: Dict[str, Any] = Body(default_factory=dict)) -> Any:
        wants_stream = False
        try:
            wants_stream = bool(int(request.query_params.get("stream", "0")))
        except Exception:
            wants_stream = False
        if not wants_stream:
            accept = (request.headers.get("accept") or "").lower()
            wants_stream = "text/event-stream" in accept

        if not wants_stream:
            return JSONResponse(next_steps_jsonl(payload))

        async def _events() -> AsyncIterator[bytes]:
            async for event in stream_next_steps_jsonl(payload):
                yield (f"data: {json.dumps(event, ensure_ascii=False)}\n\n").encode("utf-8")

        return StreamingResponse(_events(), media_type="text/event-stream")

    @router.post("/image")
    def image(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
        prompt_result = build_image_prompt(payload, prompt_template=payload.get("promptTemplate"))
        if not prompt_result.get("ok"):
            return prompt_result

        prompt = (prompt_result.get("prompt") or {}).get("prompt") if isinstance(prompt_result.get("prompt"), dict) else None
        num_outputs = payload.get("numOutputs") or payload.get("num_outputs") or 1
        try:
            n = int(num_outputs)
        except Exception:
            n = 1
        n = max(1, min(8, n))
        output_format = str(payload.get("outputFormat") or payload.get("output_format") or "url")

        images = generate_images(prompt=str(prompt or ""), num_outputs=n, output_format=output_format)
        return {**prompt_result, "images": images}

    app.include_router(router)
    return app


app = create_app()

