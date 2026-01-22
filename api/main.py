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


def _normalize_form_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a few known client payload shapes into the internal shape expected by
    `programs.batch_generator.orchestrator.next_steps_jsonl`.

    Supported:
    - Native service payload shape (already has `batchId`, `stepDataSoFar`, `alreadyAskedKeys`)
    - sif-widget `/api/ai-form/[instanceId]/new-batch` shape:
      { session, currentBatch, state: { answers, askedStepIds, ... }, request }
    """
    if not isinstance(payload, dict):
        return {}

    # Widget shape adapter.
    state = payload.get("state") if isinstance(payload.get("state"), dict) else None
    current_batch = payload.get("currentBatch") if isinstance(payload.get("currentBatch"), dict) else None
    if state and current_batch:
        from api.supabase_client import fetch_instance, fetch_instance_subcategories, resolve_selected_service

        def _batch_policy_from_form_plan(form_plan: Dict[str, Any]) -> Dict[str, Any]:
            # Current backend expects `batchPolicy` (phases/maxCalls). Some clients round-trip
            # a richer widget `formPlan` snapshot which includes `batches[]`.
            if not isinstance(form_plan, dict) or not form_plan:
                return {}
            if isinstance(form_plan.get("phases"), list) or form_plan.get("maxCalls") is not None:
                return form_plan
            batches = form_plan.get("batches") if isinstance(form_plan.get("batches"), list) else []
            phases: list[Dict[str, Any]] = []

            def _mini_type_from_component(t: Any) -> str:
                x = str(t or "").strip().lower()
                if x in ("multiple_choice", "segmented_choice", "chips_multi", "searchable_select", "image_choice_grid"):
                    return "choice"
                if x in ("text_input", "text"):
                    return "text"
                if x in ("file_upload", "file_picker", "upload"):
                    return "upload"
                return x or "choice"

            for b in batches:
                if not isinstance(b, dict):
                    continue
                bid = str(b.get("batchId") or b.get("id") or "").strip()
                if not bid:
                    continue
                raw_allowed = b.get("allowedComponentTypes") if isinstance(b.get("allowedComponentTypes"), list) else []
                allowed_mini_types: list[str] = []
                for t in raw_allowed:
                    mt = _mini_type_from_component(t)
                    if mt and mt not in allowed_mini_types:
                        allowed_mini_types.append(mt)
                focus_keys_raw = b.get("focusKeys") if isinstance(b.get("focusKeys"), list) else []
                focus_keys: list[str] = []
                for k in focus_keys_raw:
                    kk = str(k or "").strip()
                    if kk and kk not in focus_keys:
                        focus_keys.append(kk)
                phases.append(
                    {
                        "id": bid,
                        "purpose": b.get("purpose"),
                        "maxSteps": b.get("maxSteps"),
                        "allowedMiniTypes": allowed_mini_types,
                        "rigidity": b.get("rigidity"),
                        "focusKeys": focus_keys or None,
                    }
                )

            # Support both legacy (`constraints`/`stop` at top-level) and the leaner shape
            # where global config lives under `form`.
            form_section = form_plan.get("form") if isinstance(form_plan.get("form"), dict) else {}
            constraints = (
                form_section.get("constraints")
                if isinstance(form_section.get("constraints"), dict)
                else (form_plan.get("constraints") if isinstance(form_plan.get("constraints"), dict) else {})
            )
            stop = (
                form_section.get("stop")
                if isinstance(form_section.get("stop"), dict)
                else (form_plan.get("stop") if isinstance(form_plan.get("stop"), dict) else {})
            )

            max_calls = constraints.get("maxBatches")
            out: Dict[str, Any] = {
                "v": 1,
                "maxCalls": max_calls or len(phases) or 2,
                "phases": phases,
                "stopConditions": {
                    "requiredKeysComplete": stop.get("requiredComplete"),
                    "satietyTarget": stop.get("satietyTarget"),
                },
            }
            keys = form_section.get("keys") if isinstance(form_section.get("keys"), list) else form_plan.get("keys")
            if isinstance(keys, list):
                out["keys"] = keys
            return out

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

        # Optional Supabase hydration (RAG-ish grounding).
        # If the caller only provides UUIDs (e.g. `step-service-primary`) we can resolve labels here.
        try:
            session = adapted.get("session") if isinstance(adapted.get("session"), dict) else {}
            instance_id = str(session.get("instanceId") or session.get("instance_id") or "").strip()
            if instance_id:
                instance = fetch_instance(instance_id) or {}
                instance_subcategories = fetch_instance_subcategories(instance_id)

                # Resolve selected service (UUID) -> labels.
                answers = adapted.get("stepDataSoFar") if isinstance(adapted.get("stepDataSoFar"), dict) else {}
                selected_service_id = answers.get("step-service-primary") or answers.get("step-service") or None
                category_name, subcategory_name, subcategory_id = resolve_selected_service(
                    selected_subcategory_id=str(selected_service_id) if selected_service_id is not None else None,
                    instance_subcategories=instance_subcategories,
                )

                # Populate plain-English fields expected by the planner (best-effort).
                adapted.setdefault("instanceSubcategories", instance_subcategories)
                adapted.setdefault("instance_subcategories", instance_subcategories)
                adapted.setdefault("categoryName", category_name or adapted.get("categoryName"))
                adapted.setdefault("subcategoryName", subcategory_name or adapted.get("subcategoryName"))
                adapted.setdefault("subcategoryId", subcategory_id or adapted.get("subcategoryId"))
                adapted.setdefault("industry", category_name or adapted.get("industry") or "General")
                adapted.setdefault("service", subcategory_name or adapted.get("service") or "")
                adapted.setdefault("businessContext", instance.get("name") or adapted.get("businessContext") or "")
                adapted.setdefault("useCase", instance.get("use_case") or instance.get("useCase") or adapted.get("useCase") or None)

                # Also attach nested context/grounding if missing, so downstream code can rely on it.
                st = adapted.get("state") if isinstance(adapted.get("state"), dict) else {}
                if isinstance(st, dict):
                    ctx = st.get("context") if isinstance(st.get("context"), dict) else {}
                    if not ctx:
                        ctx = {}
                    ctx.setdefault("businessContext", adapted.get("businessContext"))
                    ctx.setdefault("categoryName", adapted.get("categoryName"))
                    ctx.setdefault("subcategoryName", adapted.get("subcategoryName"))
                    ctx.setdefault("subcategoryId", adapted.get("subcategoryId"))
                    st["context"] = ctx
                    if st.get("grounding") is None:
                        st["grounding"] = {
                            "version": 1,
                            "service": {
                                "categoryName": adapted.get("categoryName"),
                                "subcategoryName": adapted.get("subcategoryName"),
                                "subcategoryId": adapted.get("subcategoryId"),
                            },
                        }
                    adapted["state"] = st
        except Exception:
            pass
        # `state.formPlan` may be either a batchPolicy-like skeleton OR a richer widget snapshot.
        # Normalize it into the internal `batchPolicy` shape.
        if isinstance(state.get("formPlan"), dict) and state.get("formPlan"):
            adapted.setdefault("batchPolicy", _batch_policy_from_form_plan(state.get("formPlan") or {}))
        # Preserve batch_state if the widget ever adds it.
        adapted.setdefault("batchState", state.get("batchState") or state.get("batch_state") or {})
        return adapted

    return payload


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

    @router.post(
        "/form",
        response_model=FormResponse,
        description=(
            "Generates the next batch of UI steps.\n\n"
            "If this is a widget-style new-batch request (includes `currentBatch`) and the client did not provide a "
            "plan (`batchPolicy` or `state.formPlan`), the backend will bootstrap one and include it in the response "
            "as `formPlan`."
        ),
    )
    async def form(payload: FormRequest = Body(default_factory=FormRequest)) -> Any:
        payload_dict = payload.model_dump(by_alias=True, exclude_none=True)
        payload_dict = _normalize_form_payload(payload_dict)
        return JSONResponse(next_steps_jsonl(payload_dict))

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
