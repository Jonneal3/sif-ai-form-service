"""
Minimal DSPy Flow Planner.

Ported from `sif-widget/dspy/flow_planner.py` into this standalone service repo so we can run DSPy
without spawning a local Python subprocess from Next.js.

---

### How to read this file (DSPy beginner notes)

This module is the **bridge between HTTP** (FastAPI) and **DSPy** (prompted programs).

Data flow:
1. `api/index.py` receives a POST body (payload dict)
2. `api/index.py` calls `next_steps_jsonl(payload)`
3. `next_steps_jsonl` creates a DSPy LM + configures DSPy settings
4. `next_steps_jsonl` creates a DSPy predictor: `dspy.Predict(NextStepsJSONL)`
5. DSPy sends a request to the provider (Groq/OpenAI) via LiteLLM
6. We parse the model output, validate it with Pydantic, and return a clean JSON structure

Key DSPy concepts used here:
- **Signature** (`NextStepsJSONL`): describes *inputs* and *outputs* in a declarative way.
- **Predict** (`dspy.Predict(Signature)`): turns that signature into a callable LLM-backed function.
- **Demos**: examples attached to the predictor that guide the model (few-shot).

Why some fields are strings:
- LLM outputs are text. For reliability we ask DSPy to output JSON/JSONL **as strings**,
  then we parse + validate with Pydantic. This makes failures detectable and recoverable.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _strip_code_fences(s: str) -> str:
    import re

    if not s:
        return s
    t = s.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t, flags=re.IGNORECASE)
    return t.strip()


def _best_effort_parse_json(text: str) -> Any:
    if not text:
        return None
    t = _strip_code_fences(str(text))
    parsed = _safe_json_loads(t)
    if parsed is not None:
        return parsed
    # Fallback: find first array/object block
    import re

    m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", t)
    if not m:
        return None
    return _safe_json_loads(m.group(0))


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


def _maybe_suggest(condition: bool, instruction: str) -> None:
    """
    Best-effort `dspy.Suggest` wrapper.
    We keep this fail-open because DSPy APIs differ by version.
    """
    if condition:
        return
    try:
        import dspy  # type: ignore

        suggest = getattr(dspy, "Suggest", None)
        if callable(suggest):
            suggest(condition, instruction)
    except Exception:
        return


def next_steps_jsonl(payload: Dict[str, Any], *, stream: bool = False) -> Dict[str, Any]:
    """
    Single-call NEXT STEPS generator.

    - DSPy decides which questions to ask next based on batch_state + known_answers + grounding.
    - Output is JSONL lines (one MiniStep per line) for streaming.
    - A final meta JSON object is returned for non-stream callers.
    """
    import time as _time

    request_id = f"next_steps_{int(_time.time() * 1000)}"
    start_time = _time.time()
    schema_version = (
        payload.get("schemaVersion")
        or payload.get("schema_version")
        or _best_effort_contract_schema_version()
    )

    # 1) Decide which provider/model to use (Groq/OpenAI), based on env vars.
    #    If this returns None, we fail early with a helpful error.
    lm_cfg = _make_dspy_lm()
    if not lm_cfg:
        return {
            "ok": False,
            "error": "DSPy LM not configured",
            "requestId": request_id,
            "schemaVersion": str(schema_version) if schema_version else "0",
        }

    try:
        import dspy  # type: ignore
    except Exception:
        return {
            "ok": False,
            "error": "DSPy import failed",
            "requestId": request_id,
            "schemaVersion": str(schema_version) if schema_version else "0",
        }

    # 2) DSPy v3 uses a LiteLLM-backed LM object. Think of this as your "transport" to the model.
    llm_timeout = float(os.getenv("DSPY_LLM_TIMEOUT_SEC") or "20")
    # Higher temperature for more randomness/variety in questions
    temperature = float(os.getenv("DSPY_TEMPERATURE") or "0.7")
    max_tokens = int(os.getenv("DSPY_NEXT_STEPS_MAX_TOKENS") or "2000")

    lm = dspy.LM(
        model=lm_cfg["model"],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=llm_timeout,
        num_retries=0,
    )
    # 3) Configure DSPy (and optionally telemetry) BEFORE instantiating predictors.
    _configure_dspy(lm)

    # 4) Configure DSPy settings BEFORE creating predictor.
    #    This ensures the predictor sees the configured LM.
    try:
        if hasattr(dspy, "settings") and hasattr(dspy.settings, "configure"):
            dspy.settings.configure(lm=lm, track_usage=True)
            print("[FlowPlanner] ✅ Configured DSPy settings with LM before creating predictor", file=sys.stderr, flush=True)
        else:
            print("[FlowPlanner] ⚠️ dspy.settings.configure not available", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[FlowPlanner] ⚠️ Failed to configure DSPy settings: {e}", file=sys.stderr, flush=True)

    from modules.signatures.flow_signatures import (
        BudgetCardsUI,
        FileUploadUI,
        FormPlanItem,
        GenericUI,
        MultipleChoiceUI,
        NextStepsJSONL,
        SliderUI,
        StepCopy,
        TextInputUI,
    )

    def _validate_mini(obj: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(obj, dict):
            return None
        t = str(obj.get("type") or obj.get("componentType") or "").lower()
        try:
            if t in ["text", "text_input"]:
                out = TextInputUI.model_validate(obj).model_dump(by_alias=True)
                out["id"] = _normalize_step_id(str(out.get("id") or ""))
                return out
            if t in ["choice", "multiple_choice", "segmented_choice", "chips_multi"]:
                # Repair common LLM failure mode: missing options.
                if "options" not in obj or not obj.get("options"):
                    obj = dict(obj)
                    obj["options"] = ["Not sure"]
                out = MultipleChoiceUI.model_validate(obj).model_dump(by_alias=True)
                out["id"] = _normalize_step_id(str(out.get("id") or ""))
                return out
            if t in ["slider", "rating", "range_slider"]:
                out = SliderUI.model_validate(obj).model_dump(by_alias=True)
                out["id"] = _normalize_step_id(str(out.get("id") or ""))
                return out
            if t in ["budget_cards"]:
                out = BudgetCardsUI.model_validate(obj).model_dump(by_alias=True)
                out["id"] = _normalize_step_id(str(out.get("id") or ""))
                return out
            if t in ["upload", "file_upload", "file_picker"]:
                out = FileUploadUI.model_validate(obj).model_dump(by_alias=True)
                out["id"] = _normalize_step_id(str(out.get("id") or ""))
                return out
            
            # Fallback for other UI types found in ComponentType union
            return GenericUI.model_validate(obj).model_dump(by_alias=True)
        except Exception:
            return None

    # 5) Create the predictor AFTER configuring DSPy settings.
    #    This is the exact line where DSPy becomes an "LLM program":
    #      - Signature defines the contract (inputs/outputs)
    #      - Predict creates an object that can be called like a function
    predictor = dspy.Predict(NextStepsJSONL)

    # Attach demos if present (best-effort).
    try:
        from examples.registry import as_dspy_examples, load_examples_pack

        # Demos are stored as JSONL so they're easy to edit and version control.
        # IMPORTANT: demos must match the NextStepsJSONL signature.
        demos = as_dspy_examples(
            # IMPORTANT: demos must match the NextStepsJSONL signature, otherwise DSPy can degrade or error.
            load_examples_pack(_default_next_steps_demo_pack()),
            input_keys=[
                "platform_goal",
                "batch_id",
                "business_context",
                "industry",
                "service",
                "grounding_preview",
                "required_uploads_json",
                "personalization_summary",
                "known_answers_json",
                "already_asked_keys_json",
                "form_plan_json",
                "batch_state_json",
                "max_steps",
                "allowed_mini_types",
            ],
        )
        setattr(predictor, "demos", demos)
    except Exception:
        pass

    platform_goal = str(payload.get("platformGoal") or payload.get("platform_goal") or "")[:600]
    batch_id = str(payload.get("batchId") or payload.get("batch_id") or "ContextCore")[:40]
    business_context = str(payload.get("businessContext") or payload.get("business_context") or "")[:200]
    industry = str(payload.get("industry") or payload.get("vertical") or "General")
    service = str(payload.get("service") or payload.get("subcategoryName") or "")
    grounding_preview = str(payload.get("groundingPreview") or payload.get("grounding_preview") or "")[:2000]
    required_uploads_raw = payload.get("requiredUploads") or payload.get("required_uploads") or []
    required_uploads_json = json.dumps(required_uploads_raw if isinstance(required_uploads_raw, list) else [])[:800]
    personalization_summary = str(payload.get("personalizationSummary") or payload.get("personalization_summary") or "")[:1200]
    known_answers_json = json.dumps(payload.get("stepDataSoFar") or payload.get("knownAnswers") or {})[:2400]
    already_asked = payload.get("alreadyAskedKeys") or payload.get("alreadyAskedKeysJson") or []
    # Normalize ids to avoid underscore-vs-hyphen duplicates across systems.
    normalized_already: list[str] = []
    if isinstance(already_asked, list):
        for x in already_asked:
            t = str(x or "").strip()
            if not t:
                continue
            normalized_already.append(_normalize_step_id(t))
    already_asked_json = json.dumps(normalized_already)[:2000]
    form_plan_json = json.dumps(payload.get("formPlan") or payload.get("form_plan") or [])[:3600]
    batch_state_json = json.dumps(payload.get("batchState") or payload.get("batch_state") or {})[:2000]
    max_steps = str(payload.get("maxSteps") or payload.get("max_steps") or "4")
    allowed_mini_types = payload.get("allowedMiniTypes") or payload.get("allowed_mini_types") or []
    allowed_csv = (
        ",".join([str(x).strip() for x in allowed_mini_types if str(x).strip()])
        if isinstance(allowed_mini_types, list)
        else str(allowed_mini_types)
    )

    def _call_predictor() -> Any:
        return predictor(
            platform_goal=platform_goal,
            batch_id=batch_id,
            business_context=business_context,
            industry=industry[:80],
            service=service[:80],
            grounding_preview=grounding_preview[:2000],
            required_uploads_json=required_uploads_json,
            personalization_summary=personalization_summary,
            known_answers_json=known_answers_json,
            already_asked_keys_json=already_asked_json,
            form_plan_json=form_plan_json,
            batch_state_json=batch_state_json,
            max_steps=max_steps,
            allowed_mini_types=allowed_csv[:200],
        )

    # We keep stream mode for compatibility with older callers, but the microservice currently uses
    # stream=False and does streaming at the HTTP layer (SSE events).
    pred = _call_predictor()

    emitted: list[dict] = []
    produced_form_plan: list[dict] | None = None
    produced_copy: dict | None = None
    ready_flag: bool | None = None

    # 6) DSPy output is an object that exposes fields defined in the Signature.
    #    Here we read the *string* `mini_steps_jsonl` and parse line-by-line.
    raw_lines = getattr(pred, "mini_steps_jsonl", None) or ""

    if raw_lines:
        for line in str(raw_lines).splitlines():
            line = line.strip()
            if not line:
                continue
            obj = _best_effort_parse_json(line)
            v = _validate_mini(obj)
            if v:
                emitted.append(v)

    # 7) Parse other string fields. These are optional outputs that the model may or may not fill.
    try:
        pf = getattr(pred, "produced_form_plan_json", None) or ""
        parsed_pf = _best_effort_parse_json(str(pf))
        if isinstance(parsed_pf, list):
            produced_form_plan = []
            for it in parsed_pf[:30]:
                try:
                    produced_form_plan.append(FormPlanItem.model_validate(it).model_dump())
                except Exception:
                    continue
    except Exception:
        produced_form_plan = None

    try:
        raw_copy = getattr(pred, "must_have_copy_json", None) or ""
        parsed_copy = _best_effort_parse_json(str(raw_copy))
        if isinstance(parsed_copy, dict):
            out_copy: dict[str, Any] = {}
            for k, v in parsed_copy.items():
                try:
                    out_copy[str(k)] = StepCopy.model_validate(v).model_dump()
                except Exception:
                    continue
            produced_copy = out_copy
    except Exception:
        produced_copy = None

    try:
        raw_ready = getattr(pred, "ready_for_image_gen", None)
        ready_flag = str(raw_ready).strip().lower() == "true"
    except Exception:
        ready_flag = None

    latency_ms = int((_time.time() - start_time) * 1000)
    meta = {
        "requestId": request_id,
        "latencyMs": latency_ms,
        "schemaVersion": str(schema_version) if schema_version else "0",
        "modelUsed": lm_cfg.get("modelName") or lm_cfg.get("model"),
        "miniSteps": emitted,
        "producedFormPlan": produced_form_plan,
        "mustHaveCopy": produced_copy,
        "readyForImageGen": bool(ready_flag) if ready_flag is not None else False,
        "usage": _extract_usage_from_lm(lm, limit=1),
        "lmHistory": {"present": False},
        "ok": True,
    }

    if stream:
        return {"ok": True, "requestId": request_id}

    return meta


def _default_next_steps_demo_pack() -> str:
    """
    Prefer shared contract demos when present; otherwise fall back to repo-local examples.
    """
    env_pack = (os.getenv("DSPY_NEXT_STEPS_DEMO_PACK") or "").strip()
    if env_pack:
        return env_pack
    repo_root = Path(__file__).resolve().parent
    shared = repo_root / "shared" / "ai-form-contract" / "demos" / "next_steps_examples.jsonl"
    if shared.exists():
        return str(shared)
    return "next_steps_examples.jsonl"


def _best_effort_contract_schema_version() -> str:
    try:
        repo_root = Path(__file__).resolve().parent
        p = repo_root / "shared" / "ai-form-contract" / "schema" / "schema_version.txt"
        if p.exists():
            v = p.read_text(encoding="utf-8").strip()
            return v or "0"
    except Exception:
        pass
    return "0"


_LITELLM_CALLBACK_INSTALLED = False
_LITELLM_LAST_META: list[dict] = []


def _make_dspy_lm() -> Optional[Dict[str, str]]:
    """
    Return a LiteLLM model string for DSPy v3 (provider-prefixed), or None if not configured.
    """
    provider = (os.getenv("DSPY_PROVIDER") or "groq").lower()
    locked_model = os.getenv("DSPY_MODEL_LOCK") or "llama-3.3-70b-versatile"
    requested_model = os.getenv("DSPY_MODEL") or locked_model
    model = requested_model

    # Block 8B/instant models by default (JSON reliability / rate limit stability)
    if "8b" in model.lower() or "8-b" in model.lower() or "instant" in model.lower():
        sys.stderr.write(
            f"[DSPy] 🚫 BLOCKED: Requested DSPY_MODEL='{model}' (8B/instant). Forcing lock='{locked_model}'.\n"
        )
        model = locked_model

    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            return None
        return {"provider": "groq", "model": f"groq/{model}", "modelName": model}

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return None
        return {"provider": "openai", "model": f"openai/{model}", "modelName": model}

    return None


def _print_lm_history_if_available(lm: Any, n: int = 1) -> None:
    try:
        inspect_fn = getattr(lm, "inspect_history", None)
        if not callable(inspect_fn):
            return
        with contextlib.redirect_stdout(sys.stderr):
            inspect_fn(n=n)
    except Exception:
        return


def _configure_dspy(lm: Any) -> None:
    try:
        import dspy  # type: ignore
    except Exception:
        return

    # Telemetry is opt-in. When enabled, we try to capture token usage via LiteLLM callbacks.
    telemetry_on = os.getenv("AI_FORM_TOKEN_TELEMETRY") == "true" or os.getenv("AI_FORM_DEBUG") == "true"
    track_usage = os.getenv("DSPY_TRACK_USAGE") == "true" or telemetry_on

    global _LITELLM_CALLBACK_INSTALLED
    if telemetry_on and not _LITELLM_CALLBACK_INSTALLED:
        try:
            import litellm  # type: ignore

            def _capture_litellm_success(*args: Any, **kwargs: Any) -> None:
                try:
                    resp = None
                    if len(args) >= 2:
                        resp = args[1]
                    if resp is None:
                        resp = kwargs.get("response") or kwargs.get("completion_response") or kwargs.get("result")

                    usage = None
                    if isinstance(resp, dict):
                        usage = resp.get("usage")
                    else:
                        usage = getattr(resp, "usage", None)

                    headers = None
                    if isinstance(resp, dict):
                        headers = resp.get("response_headers") or resp.get("headers") or resp.get("_response_headers")
                    else:
                        headers = (
                            getattr(resp, "response_headers", None)
                            or getattr(resp, "headers", None)
                            or getattr(resp, "_response_headers", None)
                        )
                    headers = headers or kwargs.get("response_headers") or kwargs.get("headers")

                    _LITELLM_LAST_META.append(
                        {
                            "ts": int(time.time() * 1000),
                            "usage": usage if isinstance(usage, dict) else None,
                            "headers": headers if isinstance(headers, dict) else None,
                        }
                    )
                    if len(_LITELLM_LAST_META) > 25:
                        del _LITELLM_LAST_META[:-25]
                except Exception:
                    return

            # Install callback once (outside the callback body).
            scb = getattr(litellm, "success_callback", None)
            if isinstance(scb, list):
                if _capture_litellm_success not in scb:
                    scb.append(_capture_litellm_success)
            elif callable(scb):
                litellm.success_callback = [scb, _capture_litellm_success]  # type: ignore[attr-defined]
            else:
                litellm.success_callback = [_capture_litellm_success]  # type: ignore[attr-defined]
            sys.stderr.write("[DSPy] ✅ LiteLLM callback installed\n")
            _LITELLM_CALLBACK_INSTALLED = True
        except Exception:
            pass

    try:
        settings = getattr(dspy, "settings", None)
        settings_cfg = getattr(settings, "configure", None)
        if callable(settings_cfg):
            settings_cfg(lm=lm, track_usage=track_usage)
            return
    except Exception:
        return


def _extract_litellm_last_meta_snapshot() -> Optional[Dict[str, Any]]:
    try:
        if isinstance(_LITELLM_LAST_META, list) and len(_LITELLM_LAST_META) > 0:
            m = _LITELLM_LAST_META[-1]
            if isinstance(m, dict):
                return {"usage": m.get("usage"), "headers": m.get("headers")}
    except Exception:
        return None
    return None


def _extract_usage_from_lm(lm: Any, limit: int = 1) -> Dict[str, Any]:
    # LiteLLM usage is best-effort; prefer callback snapshot.
    snap = _extract_litellm_last_meta_snapshot()
    usage = (snap or {}).get("usage") if isinstance(snap, dict) else None
    if isinstance(usage, dict):
        return {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "calls": limit,
        }
    return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None, "calls": limit}


