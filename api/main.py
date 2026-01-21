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

from programs.form_planner.orchestrator import next_steps_jsonl  # noqa: E402
from programs.image_generator.orchestrator import build_image_prompt  # noqa: E402
from providers.image_generation import generate_images  # noqa: E402
from schemas.api_models import FormRequest, FormResponse  # noqa: E402


def _normalize_form_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a few known client payload shapes into the internal shape expected by
    `programs.form_planner.orchestrator.next_steps_jsonl`.

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
        adapted.setdefault("alreadyAskedKeys", state.get("askedStepIds") or state.get("alreadyAskedKeys") or [])
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
