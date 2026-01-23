from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import APIRouter, Body, FastAPI
from fastapi.responses import JSONResponse


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

from programs.batch_generator.orchestrator import next_steps_jsonl  # noqa: E402
from programs.image_generator.orchestrator import build_image_prompt  # noqa: E402
from providers.image_generation import generate_images  # noqa: E402
from schemas.api_models import FormRequest, FormResponse  # noqa: E402


def _hydrate_from_supabase_if_possible(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort "RAG-ish" hydration: resolve instance/service labels from Supabase.

    Works for both:
    - widget shape (session.instanceId + state/currentBatch)
    - service contract shape (/api/ai-form/{instanceId}/new-batch) where `instanceId` is passed in.
    """
    try:
        from api.supabase_client import fetch_instance, fetch_instance_subcategories, resolve_selected_service
    except Exception:
        return payload

    adapted = dict(payload)
    session = adapted.get("session") if isinstance(adapted.get("session"), dict) else {}
    instance_id = str(session.get("instanceId") or session.get("instance_id") or "").strip()
    if not instance_id:
        return adapted

    try:
        instance = fetch_instance(instance_id) or {}
        instance_subcategories = fetch_instance_subcategories(instance_id)

        answers = adapted.get("stepDataSoFar") if isinstance(adapted.get("stepDataSoFar"), dict) else {}
        selected_service_id = answers.get("step-service-primary") or answers.get("step-service") or None
        category_name, subcategory_name, subcategory_id = resolve_selected_service(
            selected_subcategory_id=str(selected_service_id) if selected_service_id is not None else None,
            instance_subcategories=instance_subcategories,
        )

        adapted.setdefault("instanceSubcategories", instance_subcategories)
        adapted.setdefault("instance_subcategories", instance_subcategories)
        adapted.setdefault("categoryName", category_name or adapted.get("categoryName"))
        adapted.setdefault("subcategoryName", subcategory_name or adapted.get("subcategoryName"))
        adapted.setdefault("subcategoryId", subcategory_id or adapted.get("subcategoryId"))
        adapted.setdefault("industry", category_name or adapted.get("industry") or "General")
        adapted.setdefault("service", subcategory_name or adapted.get("service") or "")
        adapted.setdefault("businessContext", instance.get("name") or adapted.get("businessContext") or "")
        adapted.setdefault("useCase", instance.get("use_case") or instance.get("useCase") or adapted.get("useCase") or None)
    except Exception:
        return adapted
    return adapted


def _normalize_form_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a few known client payload shapes into the internal shape expected by
    `programs.batch_generator.orchestrator.next_steps_jsonl`.

    Supported:
    - sif-widget `/api/ai-form/[instanceId]/new-batch` shape:
      { session, currentBatch, state: { answers, askedStepIds, ... }, request }
    """
    if not isinstance(payload, dict):
        return {}

    # Widget shape adapter.
    state = payload.get("state") if isinstance(payload.get("state"), dict) else None
    current_batch = payload.get("currentBatch") if isinstance(payload.get("currentBatch"), dict) else None
    if state and current_batch:
        adapted = dict(payload)
        adapted.setdefault("batchId", current_batch.get("batchId") or current_batch.get("batch_id"))
        adapted.setdefault("batchNumber", current_batch.get("batchNumber") or current_batch.get("batch_number"))
        adapted.setdefault("maxSteps", current_batch.get("maxSteps") or current_batch.get("max_steps"))
        # Internal names used throughout the pipeline.
        adapted.setdefault("stepDataSoFar", state.get("answers") or state.get("stepDataSoFar") or {})
        asked_step_ids = state.get("askedStepIds") or state.get("alreadyAskedKeys") or []
        # Broaden de-dupe: if the caller provides explicit rendered step ids, treat them as "asked" too.
        existing_step_ids = payload.get("existingStepIds") or payload.get("existing_step_ids") or []
        question_step_ids = payload.get("questionStepIds") or payload.get("question_step_ids") or []
        merged_asked: list[str] = []
        for seq in (asked_step_ids, existing_step_ids, question_step_ids):
            if not isinstance(seq, list):
                continue
            for v in seq:
                s = str(v or "").strip()
                if s and s not in merged_asked:
                    merged_asked.append(s)
        asked_step_ids = merged_asked
        adapted.setdefault("askedStepIds", asked_step_ids)
        # Deprecated alias (the backend historically called these "keys", but they are step ids).
        adapted.setdefault("alreadyAskedKeys", asked_step_ids)
        # Preserve nested widget context/grounding so the planner has plain-English anchors.
        # This is critical because the widget often stores UUIDs in `state.answers` that the LLM can't interpret.
        state_context = state.get("context") if isinstance(state.get("context"), dict) else {}
        if state_context:
            adapted.setdefault("businessContext", state_context.get("businessContext") or state_context.get("business_context"))
            adapted.setdefault("industry", state_context.get("industry") or state_context.get("categoryName") or state_context.get("category_name"))
            adapted.setdefault("subcategoryName", state_context.get("subcategoryName") or state_context.get("subcategory_name"))
            adapted.setdefault("service", state_context.get("subcategoryName") or state_context.get("subcategory_name"))
            adapted.setdefault("categoryName", state_context.get("categoryName") or state_context.get("category_name"))
            adapted.setdefault("subcategoryId", state_context.get("subcategoryId") or state_context.get("subcategory_id"))
            adapted.setdefault("trafficSource", state_context.get("trafficSource") or state_context.get("traffic_source"))
        if state.get("grounding") is not None:
            adapted.setdefault("grounding", state.get("grounding"))
        if state.get("answeredQA") is not None:
            adapted.setdefault("answeredQA", state.get("answeredQA"))

        adapted = _hydrate_from_supabase_if_possible(adapted)
        # Preserve batch_state if the widget ever adds it.
        adapted.setdefault("batchState", state.get("batchState") or state.get("batch_state") or {})
        return adapted

    return payload


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

    @router.post(
        "/form",
        response_model=FormResponse,
        response_model_exclude_none=True,
        description=(
            "Generates the next batch of UI steps."
        ),
    )
    async def form(payload: FormRequest = Body(default_factory=FormRequest)) -> Any:
        payload_dict = payload.model_dump(by_alias=True, exclude_none=True)
        payload_dict = _normalize_form_payload(payload_dict)
        return JSONResponse(next_steps_jsonl(payload_dict))

    @app.post(
        "/api/ai-form/{instanceId}/new-batch",
        description="Generates the next batch of UI steps (OpenAPI contract endpoint).",
    )
    async def ai_form_new_batch(instanceId: str, body: Dict[str, Any] = Body(default_factory=dict)) -> Any:
        from api.openapi_contract import validate_new_batch_request, validate_new_batch_response

        validate_new_batch_request(body)
        # Provide the instance id to the backend for Supabase hydration + context enrichment.
        payload_dict = dict(body)
        payload_dict["session"] = {
            "instanceId": instanceId,
            "sessionId": str(body.get("sessionId") or ""),
        }
        payload_dict = _hydrate_from_supabase_if_possible(payload_dict)
        payload_dict = _normalize_form_payload(payload_dict)
        resp = next_steps_jsonl(payload_dict)
        validate_new_batch_response(resp)
        return JSONResponse(resp)

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
