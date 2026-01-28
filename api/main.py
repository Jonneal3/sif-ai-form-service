from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_422_UNPROCESSABLE_ENTITY, HTTP_500_INTERNAL_SERVER_ERROR


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

from programs.form_pipeline.orchestrator import next_steps_jsonl  # noqa: E402
from programs.image_generator.orchestrator import build_image_prompt  # noqa: E402
from providers.image_generation import generate_images  # noqa: E402
from schemas.api_models import ExecuteFunctionRequest, NewBatchRequest, FormResponse  # noqa: E402
from api.request_adapter import to_next_steps_payload  # noqa: E402


def _load_contract_schema() -> Dict[str, Any]:
    root = _repo_root()
    # The canonical UI-step contract lives under `shared/ai-form-ui-contract/` (symlinked in dev).
    # Keep a fallback for older layouts to avoid breaking local setups.
    schema_path = root / "shared" / "ai-form-ui-contract" / "schema" / "ui_step.schema.json"
    version_path = root / "shared" / "ai-form-ui-contract" / "schema" / "schema_version.txt"
    if not schema_path.exists():
        schema_path = root / "shared" / "ai-form-contract" / "schema" / "ui_step.schema.json"
    if not version_path.exists():
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


def _http_status_for_pipeline_response(resp: Any) -> int:
    """
    Map pipeline failures (`{ok:false,...}`) to HTTP status codes.

    - Client input problems => 400/422
    - Server misconfiguration => 500
    """
    if not isinstance(resp, dict):
        return HTTP_500_INTERNAL_SERVER_ERROR
    if resp.get("ok") is not False:
        return 200

    err = str(resp.get("error") or "").strip().lower()
    msg = str(resp.get("message") or resp.get("error") or "").strip().lower()

    # Server-side issues.
    if "dspy lm not configured" in msg or "dspy import failed" in msg or err in {"internal_error"}:
        return HTTP_500_INTERNAL_SERVER_ERROR

    # Client-side issues.
    if "missing service context" in msg or "missing services_summary" in msg:
        return HTTP_422_UNPROCESSABLE_ENTITY
    if "token budget exhausted" in msg:
        return HTTP_400_BAD_REQUEST

    # Default: treat as bad request (caller can inspect `error` for details).
    return HTTP_400_BAD_REQUEST


def create_app() -> FastAPI:
    # Load `.env` + `.env.local` when present (local dev convenience).
    load_dotenv(_repo_root() / ".env", override=False)
    load_dotenv(_repo_root() / ".env.local", override=False)

    app = FastAPI(title="sif-ai-form-service")
    router = APIRouter(prefix="/v1/api")
    compat_router = APIRouter(prefix="/api")

    _EXEC_FN_CACHE: dict[str, tuple[float, Dict[str, Any]]] = {}

    def _exec_fn_cache_get(cache_key: str) -> Optional[Dict[str, Any]]:
        if not cache_key:
            return None
        rec = _EXEC_FN_CACHE.get(cache_key)
        if not rec:
            return None
        expires_at, value = rec
        if time.time() >= float(expires_at):
            _EXEC_FN_CACHE.pop(cache_key, None)
            return None
        return value

    def _exec_fn_cache_set(cache_key: str, value: Dict[str, Any], *, ttl_sec: int) -> None:
        if not cache_key:
            return
        ttl = max(60, min(3600, int(ttl_sec or 0)))
        _EXEC_FN_CACHE[cache_key] = (time.time() + ttl, value)

    def _hash_step_data(step_data: Dict[str, Any]) -> str:
        try:
            raw = json.dumps(step_data or {}, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        except Exception:
            raw = str(step_data or "")
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    def _answered_qa_from_step_data(step_data: Dict[str, Any]) -> list[dict[str, Any]]:
        """
        Best-effort adapter: build `answeredQA[]` from `stepData{}`.
        """
        out: list[dict[str, Any]] = []
        if not isinstance(step_data, dict):
            return out
        for k, v in step_data.items():
            key = str(k or "").strip()
            if not key:
                continue
            step_id = f"step-{key.replace('_', '-').replace(' ', '-')}"
            try:
                answer = v if isinstance(v, (str, int, float, bool)) or v is None else json.dumps(v, ensure_ascii=True)
            except Exception:
                answer = str(v)
            out.append({"stepId": step_id, "question": key, "answer": answer})
            if len(out) >= 32:
                break
        return out

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = f"val_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        # Keep server logs useful without dumping full bodies.
        print(
            f"[api] 422 validation_error requestId={request_id} path={request.url.path} errors={exc.errors()}",
            flush=True,
        )
        return JSONResponse(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "ok": False,
                "error": "validation_error",
                "message": "Request body did not match expected schema.",
                "requestId": request_id,
                "details": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = f"err_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        print(f"[api] 500 internal_error requestId={request_id} path={request.url.path} err={exc!r}", flush=True)
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "ok": False,
                "error": "internal_error",
                "message": "Unhandled server error.",
                "requestId": request_id,
            },
        )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"ok": True}

    @router.get("/form/capabilities")
    def capabilities() -> Dict[str, Any]:
        return {"ok": True, **_load_contract_schema()}

    @router.post(
        "/form/{instanceId}",
        response_model=FormResponse,
        response_model_exclude_none=True,
        description=(
            "Generates the next batch of UI steps. Requires instanceId in the URL."
        ),
    )
    async def form(
        instanceId: str,
        payload: Dict[str, Any] = Body(default_factory=dict),
    ) -> Any:
        from api.openapi_contract import validate_new_batch_request, validate_new_batch_response

        try:
            parsed = NewBatchRequest.model_validate(payload)
        except ValidationError as exc:
            request_id = f"val_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            print(
                f"[api] 422 validation_error requestId={request_id} path=/v1/api/form/{instanceId} errors={exc.errors()}",
                flush=True,
            )
            return JSONResponse(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "ok": False,
                    "error": "validation_error",
                    "message": "Request body did not match expected schema.",
                    "requestId": request_id,
                    "details": exc.errors(),
                },
            )

        body = parsed.model_dump(by_alias=True, exclude_none=True)
        validate_contract = os.getenv("AI_FORM_VALIDATE_CONTRACT") == "true"
        if validate_contract:
            validate_new_batch_request(body)

        payload_dict = to_next_steps_payload(instance_id=instanceId, body=body)
        resp = next_steps_jsonl(payload_dict)

        if validate_contract:
            validate_new_batch_response(resp)

        status = _http_status_for_pipeline_response(resp)
        return JSONResponse(status_code=status, content=resp)

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

    @compat_router.post("/ai-form/{instanceId}/execute-function")
    @router.post("/ai-form/{instanceId}/execute-function")
    async def execute_function(
        instanceId: str,
        payload: Dict[str, Any] = Body(default_factory=dict),
    ) -> Any:
        request_id = f"exec_fn_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        try:
            parsed = ExecuteFunctionRequest.model_validate(payload)
        except ValidationError as exc:
            return JSONResponse(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "ok": False,
                    "error": "validation_error",
                    "message": "Request body did not match expected schema.",
                    "requestId": request_id,
                    "details": exc.errors(),
                },
            )

        body = parsed.model_dump(by_alias=True, exclude_none=True)
        session_id = str(body.get("sessionId") or "").strip()
        function_name = str(body.get("functionName") or "").strip()
        step_data = body.get("stepData") if isinstance(body.get("stepData"), dict) else {}

        if function_name != "generateInitialImage":
            return JSONResponse(
                status_code=HTTP_400_BAD_REQUEST,
                content={
                    "ok": False,
                    "error": "unknown_function",
                    "message": f"Unsupported functionName: {function_name}",
                    "requestId": request_id,
                },
            )

        cache_key = f"exec_fn:{session_id or 'none'}:{function_name}:{_hash_step_data(step_data)}"
        ttl_sec = int(os.getenv("AI_FORM_EXECUTE_FUNCTION_CACHE_TTL_SEC") or "900")
        cached = _exec_fn_cache_get(cache_key) if session_id else None
        if isinstance(cached, dict) and cached.get("ok") is True:
            return JSONResponse(status_code=200, content=cached)

        # Adapt the execute-function payload into the existing image prompt+generation input shape.
        # We use `answeredQA` as the canonical "answers so far" for context building.
        image_payload: Dict[str, Any] = {
            "sessionId": session_id,
            "instanceId": instanceId,
            "answeredQA": _answered_qa_from_step_data(step_data),
            # Pass through any optional context fields if the caller included them.
            "serviceSummary": payload.get("serviceSummary") or payload.get("service_summary"),
            "companySummary": payload.get("companySummary") or payload.get("company_summary"),
            "useCase": payload.get("useCase") or payload.get("use_case"),
            "instanceContext": payload.get("instanceContext") or payload.get("instance_context"),
            # Provide a stable batchId for prompt generation continuity.
            "batchId": f"exec-{(session_id or 'none')[:40]}",
        }

        prompt_result = build_image_prompt(image_payload, prompt_template=payload.get("promptTemplate"))
        if not prompt_result.get("ok"):
            resp = {**prompt_result, "requestId": request_id}
            return JSONResponse(status_code=_http_status_for_pipeline_response(resp), content=resp)

        prompt = (prompt_result.get("prompt") or {}).get("prompt") if isinstance(prompt_result.get("prompt"), dict) else None
        num_outputs = payload.get("numOutputs") or payload.get("num_outputs") or 1
        try:
            n = int(num_outputs)
        except Exception:
            n = 1
        n = max(1, min(8, n))
        output_format = str(payload.get("outputFormat") or payload.get("output_format") or "url")

        images = generate_images(prompt=str(prompt or ""), num_outputs=n, output_format=output_format)
        resp = {**prompt_result, "images": images, "requestId": request_id}
        if session_id:
            _exec_fn_cache_set(cache_key, resp, ttl_sec=ttl_sec)
        return JSONResponse(status_code=200, content=resp)

    app.include_router(router)
    app.include_router(compat_router)
    return app


app = create_app()
