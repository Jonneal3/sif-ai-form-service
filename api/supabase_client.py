"""
Supabase client for fetching form data.

Uses official Supabase Python client with realtime support.
The backend fetches all form configuration and session state
from Supabase, so clients only need to send minimal identifiers.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from supabase import create_client, Client


_client: Optional[Client] = None


def _normalize_step_id(step_id: str) -> str:
    t = str(step_id or "").strip()
    if not t:
        return t
    return t.replace("_", "-")


def _ensure_use_case_uploads(use_case: str, uploads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = str(use_case or "").strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    if "tryon" in normalized or "try_on" in normalized:
        normalized = "tryon"
    elif "scene" in normalized and "placement" in normalized:
        normalized = "scene_placement"
    if normalized not in {"tryon", "scene_placement"}:
        return uploads
    role = "userImage" if normalized == "tryon" else "sceneImage"
    step_id = "step-upload-subject" if normalized == "tryon" else "step-upload-scene"
    target_id = _normalize_step_id(step_id)
    for item in uploads:
        existing_id = _normalize_step_id(item.get("stepId") or item.get("step_id") or "")
        if existing_id == target_id or item.get("role") == role:
            return uploads
    return [{"stepId": step_id, "role": role}] + uploads


def _flatten_composite_answers(
    answers: Dict[str, Any],
    asked_step_ids: List[str],
) -> tuple[Dict[str, Any], List[str]]:
    """
    Frontend composite steps store answers as an object keyed by block.id under the parent step id.

    Backend convention:
    - If a composite answer value is a dict containing keys that look like step ids (start with `step-`),
      lift those into the top-level answers dict so DSPy and image prompting can consume them normally.
    - Also add lifted step ids to asked_step_ids so we don't re-ask nested steps.
    """
    if not isinstance(answers, dict) or not answers:
        return answers, asked_step_ids
    asked = [str(x).strip() for x in (asked_step_ids or []) if str(x).strip()]
    asked_set = set(_normalize_step_id(x) for x in asked)
    out = dict(answers)

    for parent_id, val in list(answers.items()):
        if not isinstance(val, dict):
            continue
        lifted_any = False
        for k, v in val.items():
            step_id = _normalize_step_id(str(k or ""))
            if not step_id.startswith("step-"):
                continue
            if step_id not in out:
                out[step_id] = v
            if step_id not in asked_set:
                asked.append(step_id)
                asked_set.add(step_id)
            lifted_any = True
        # Optionally mark the parent composite as asked if we lifted anything from it.
        if lifted_any:
            pid = _normalize_step_id(str(parent_id or ""))
            if pid and pid not in asked_set:
                asked.append(pid)
                asked_set.add(pid)

    return out, asked


def _env_csv(name: str) -> List[str]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _infer_phase(batch_id: str, form_plan_list: List[Dict[str, Any]]) -> str:
    """
    Normalize the high-level phase. This is intentionally coarse:
    - ContextCore: first call where we may create the plan
    - PersonalGuide: subsequent calls that follow the plan (can happen across many batches)
    """
    bid = str(batch_id or "").strip()
    if bid in {"ContextCore", "PersonalGuide"}:
        return bid
    # If we already have a plan, we're no longer in the "first batch / plan creation" phase.
    return "PersonalGuide" if form_plan_list else "ContextCore"


def _default_allowed_mini_types(phase: str, items_list: List[Dict[str, Any]]) -> List[str]:
    """
    Choose which "mini" UI types DSPy is allowed to generate.

    Defaults:
    - ContextCore: choice-only unless remaining items clearly need text/slider
    - PersonalGuide: choice + slider by default; add text_input if needed

    Env overrides (comma-separated):
    - AI_FORM_ALLOWED_MINI_TYPES_CONTEXTCORE
    - AI_FORM_ALLOWED_MINI_TYPES_PERSONALGUIDE
    """
    phase_norm = str(phase or "").strip()
    env_name = (
        "AI_FORM_ALLOWED_MINI_TYPES_CONTEXTCORE"
        if phase_norm == "ContextCore"
        else "AI_FORM_ALLOWED_MINI_TYPES_PERSONALGUIDE"
    )
    overridden = _env_csv(env_name)
    if overridden:
        return overridden

    allowed: List[str] = ["choice"]
    if phase_norm != "ContextCore":
        allowed.append("slider")

    for item in items_list or []:
        if not isinstance(item, dict):
            continue
        hint = str(item.get("component_hint") or "").strip().lower()
        if hint in {"slider", "rating", "range_slider"} and "slider" not in allowed:
            allowed.append("slider")
        if hint in {"text", "text_input"} and "text_input" not in allowed:
            allowed.append("text_input")
    return allowed


def _default_max_steps(phase: str, items_list: List[Dict[str, Any]]) -> int:
    """
    Default maxSteps per call. Caps to the number of remaining items when available.

    Env overrides:
    - AI_FORM_CONTEXTCORE_MAX_STEPS
    - AI_FORM_PERSONALGUIDE_MAX_STEPS
    """
    phase_norm = str(phase or "").strip()
    env_name = "AI_FORM_CONTEXTCORE_MAX_STEPS" if phase_norm == "ContextCore" else "AI_FORM_PERSONALGUIDE_MAX_STEPS"
    raw = (os.getenv(env_name) or "").strip()
    if raw:
        try:
            base = int(raw)
        except Exception:
            base = 5 if phase_norm == "ContextCore" else 10
    else:
        base = 5 if phase_norm == "ContextCore" else 10
    if items_list:
        try:
            return max(1, min(int(base), len(items_list)))
        except Exception:
            return base
    return base


def get_supabase_client() -> Optional[Client]:
    """Get or create Supabase client (singleton)."""
    global _client
    
    if _client is not None:
        return _client
    
    # Try NEXT_PUBLIC_SUPABASE_URL first (for consistency with Next.js), then SUPABASE_URL
    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL")
    # Use service role key for backend (has full access), fallback to anon key
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    
    if not url or not key:
        return None
    
    try:
        _client = create_client(url, key)
        return _client
    except Exception as e:
        print(f"[Supabase] Failed to create client: {e}")
        return None


async def fetch_instance_subcategories(instance_id: str) -> List[Dict[str, Any]]:
    """
    Fetch instance subcategories with joined category/subcategory info.

    Returns list of:
        {
            "id": "...",
            "subcategory": "...",
            "description": "...",
            "slug": "...",
            "category_name": "..."
        }
    """
    client = get_supabase_client()
    if not client or not instance_id:
        return []

    try:
        result = (
            client.table("instance_subcategories")
            .select(
                "category_subcategory_id, categories_subcategories ( id, subcategory, description, slug, categories ( name ) )"
            )
            .eq("instance_id", instance_id)
            .execute()
        )
        out: List[Dict[str, Any]] = []
        rows = result.data or []
        for row in rows:
            subcat = row.get("categories_subcategories") if isinstance(row, dict) else None
            if not isinstance(subcat, dict):
                continue
            category = subcat.get("categories") if isinstance(subcat.get("categories"), dict) else {}
            out.append(
                {
                    "id": subcat.get("id"),
                    "subcategory": subcat.get("subcategory"),
                    "description": subcat.get("description"),
                    "slug": subcat.get("slug"),
                    "category_name": category.get("name"),
                }
            )
        return out
    except Exception as e:
        print(f"[Supabase] Error fetching instance subcategories: {e}")
        return []


async def fetch_form_config(instance_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch form configuration from Supabase.
    
    Returns:
        {
            "platform_goal": "...",
            "business_context": "...",
            "industry": "...",
            "service": "...",
            "max_steps": 4,
            "allowed_step_types": [...],
            "required_uploads": [...],
            "instance_subcategories": [...]
        }
    """
    client = get_supabase_client()
    if not client:
        return {}
    if not instance_id:
        return {}
    
    try:
        instance_subcategories = await fetch_instance_subcategories(str(instance_id))
        instance_use_case = ""
        instance_data: Dict[str, Any] = {}
        try:
            instance_result = (
                client.table("instances")
                .select("*")
                .eq("id", str(instance_id))
                .execute()
            )
            if instance_result.data and len(instance_result.data) > 0:
                instance_data = instance_result.data[0]
                instance_use_case = instance_data.get("use_case") or instance_data.get("useCase") or ""
        except Exception as e:
            print(f"[Supabase] Error fetching instance data: {e}")
        
        return {
            "platform_goal": instance_data.get("platform_goal", ""),
            "business_context": instance_data.get("business_context", ""),
            "industry": instance_data.get("industry", "General"),
            "service": instance_data.get("service", ""),
            "max_steps": instance_data.get("max_steps", 4),
            "allowed_step_types": instance_data.get("allowed_step_types", []),
            "required_uploads": instance_data.get("required_uploads", []),
            "instance_subcategories": instance_subcategories,
            "use_case": instance_use_case,
        }
    except Exception as e:
        print(f"[Supabase] Error fetching form config: {e}")
    
    return {}


async def fetch_session_state(session_id: str) -> Dict[str, Any]:
    """
    Fetch current session state from Supabase.
    
    Returns:
        {
            "answers": {...},
            "asked_step_ids": [...],
            "form_plan": [...],
            "personalization_summary": "..."
        }
    """
    # Sessions are client-side only; no backend state.
    return {}


async def fetch_batch_state(session_id: str, batch_id: str) -> Dict[str, Any]:
    """
    Fetch batch state from Supabase.
    
    Returns:
        {
            "calls_used": 0,
            "max_calls": 2,
            "calls_remaining": 2,
            "satiety_so_far": 0,
            "satiety_remaining": 1,
            "missing_high_impact_keys": [],
            "must_have_copy_needed": false
        }
    """
    client = get_supabase_client()
    if not client:
        return _default_batch_state()
    
    try:
        # Adjust table name to match your schema
        result = (
            client.table("batch_states")
            .select("*")
            .eq("session_id", session_id)
            .eq("batch_id", batch_id)
            .execute()
        )
        
        if result.data and len(result.data) > 0:
            return result.data[0]
    except Exception as e:
        print(f"[Supabase] Error fetching batch state: {e}")
    
    return _default_batch_state()


def _default_batch_state() -> Dict[str, Any]:
    """Return default batch state if not found in Supabase."""
    return {
        "calls_used": 0,
        "max_calls": 2,
        "calls_remaining": 2,
        "satiety_so_far": 0,
        "satiety_remaining": 1,
        "missing_high_impact_keys": [],
        "must_have_copy_needed": False,
    }


async def build_planner_payload_from_supabase(
    session_id: str,
    instance_id: Optional[str],
    batch_id: str,
    batch_state: Dict[str, Any],
    answers: Dict[str, Any],
    asked_step_ids: List[str],
    form_plan: Optional[List[Dict[str, Any]]] = None,
    personalization_summary: str = "",
    goal: Optional[str] = None,
    business_context: Optional[str] = None,
    industry: Optional[str] = None,
    service: Optional[str] = None,
    max_steps_override: Optional[int] = None,
    allowed_step_types_override: Optional[List[str]] = None,
    required_uploads: Optional[List[Dict[str, Any]]] = None,
    items: Optional[List[Dict[str, Any]]] = None,
    batch_policy: Optional[Dict[str, Any]] = None,
    psychology_plan: Optional[Dict[str, Any]] = None,
    batch_number: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build full planner payload from Supabase (static data) + frontend (dynamic state).

    Always fetch form config from Supabase for authoritative fields.
    
    Frontend always provides:
    - Batch state (callsUsed, satiety, etc.)
    - Step data (answers so far)
    - Already asked keys
    - Form plan (AI-generated structure)
    - Personalization summary
    """
    # Fetch static data from Supabase for authoritative form config
    instance_subcategories: List[Dict[str, Any]] = []
    use_case = None
    form_config = await fetch_form_config(instance_id=instance_id)
    goal = form_config.get("platform_goal") or goal or ""
    business_context = form_config.get("business_context") or business_context or ""
    industry = form_config.get("industry") or industry or "General"
    service = form_config.get("service") or service or ""
    instance_subcategories = form_config.get("instance_subcategories") or []
    use_case = form_config.get("use_case") or use_case or ""
    
    if instance_subcategories:
        subcat_names = [
            str(s.get("subcategory")).strip()
            for s in instance_subcategories
            if isinstance(s, dict) and s.get("subcategory")
        ]
        if subcat_names:
            subcat_text = ", ".join(subcat_names[:40])
    
    # Use frontend-provided state (they know current state better)
    form_plan_list = form_plan or []

    # Composite answers may contain nested step answers keyed by step ids; lift them for backend use.
    answers, asked_step_ids = _flatten_composite_answers(answers or {}, asked_step_ids or [])

    # Phase resolution:
    # - If planner provided phases, select by current 1-based batch_number.
    # - Otherwise fall back to legacy ContextCore/PersonalGuide inference.
    try:
        from app.form_psychology.policy import normalize_policy, policy_for_batch

        normalized_policy = normalize_policy(batch_policy)
    except Exception:
        normalized_policy = None

    phase = _infer_phase(batch_id, form_plan_list)
    if normalized_policy and isinstance(batch_number, int) and batch_number > 0:
        try:
            phase = str(policy_for_batch(normalized_policy, batch_number).get("phaseId") or phase)
        except Exception:
            pass
    is_batch_1 = phase == "ContextCore"
    is_batch_2 = phase != "ContextCore"
    
    # Use provided requiredUploads or calculate from formPlan
    uploads_list = required_uploads or []
    if not uploads_list and is_batch_2:
        # Extract upload steps from formPlan if not provided
        for item in form_plan_list:
            if item.get("component_hint") == "file_upload":
                step_id = f"step-{item.get('key', '').replace('_', '-')}"
                role = item.get("role", "sceneImage")  # Default to sceneImage
                uploads_list.append({"stepId": step_id, "role": role})
    if is_batch_1:
        uploads_list = _ensure_use_case_uploads(use_case, uploads_list)
    # NOTE: Upload steps are deterministic UI components placed via `uiPlan` (returned by the backend),
    # so we intentionally do NOT enable `upload` generation by the LLM here.
    
    # Use provided items or calculate from formPlan
    items_list = items or []
    if not items_list and form_plan_list:
        focus_set = None
        if normalized_policy and isinstance(batch_number, int) and batch_number > 0:
            try:
                from app.form_psychology.policy import policy_for_batch

                active = policy_for_batch(normalized_policy, batch_number)
                focus_keys = active.get("focusKeys")
                if isinstance(focus_keys, list) and focus_keys:
                    focus_set = {str(k or "").strip() for k in focus_keys if str(k or "").strip()}
            except Exception:
                focus_set = None

        # Calculate items array (formPlan items for this batch, excluding already asked)
        for item in form_plan_list:
            key = item.get("key", "")
            if focus_set is not None and str(key or "").strip() not in focus_set:
                continue
            step_id = f"step-{key.replace('_', '-')}"
            
            # Skip if already asked
            if step_id in asked_step_ids or key in asked_step_ids:
                continue
            
            # Batch 1: exclude upload steps
            if is_batch_1 and item.get("component_hint") == "file_upload":
                continue
            
            # Batch 2: include all remaining items (including uploads)
            items_list.append({
                "id": step_id,
                "key": key,
                "goal": item.get("goal", ""),
                "why": item.get("why", ""),
                "priority": item.get("priority", "medium"),
                "component_hint": item.get("component_hint", "choice"),
                "importance_weight": item.get("importance_weight", 0.1),
                "expected_metric_gain": item.get("expected_metric_gain", 0.1),
            })

    # Calculate allowedMiniTypes (centralized policy, can still be overridden per-request).
    if allowed_step_types_override:
        allowed_mini_types = allowed_step_types_override
    else:
        allowed_from_policy = None
        if normalized_policy:
            try:
                from app.form_psychology.policy import policy_for_batch, policy_for_phase

                if isinstance(batch_number, int) and batch_number > 0:
                    allowed_from_policy = policy_for_batch(normalized_policy, batch_number).get("allowedMiniTypes")
                else:
                    allowed_from_policy = policy_for_phase(normalized_policy, phase).get("allowedMiniTypes")
            except Exception:
                allowed_from_policy = None
        if isinstance(allowed_from_policy, list) and allowed_from_policy:
            allowed_mini_types = allowed_from_policy
        else:
            allowed_mini_types = _default_allowed_mini_types(phase, items_list)
    
    # Calculate maxSteps
    if max_steps_override:
        max_steps = max_steps_override
    else:
        max_from_policy = None
        if normalized_policy:
            try:
                from app.form_psychology.policy import policy_for_batch, policy_for_phase

                if isinstance(batch_number, int) and batch_number > 0:
                    max_from_policy = policy_for_batch(normalized_policy, batch_number).get("maxSteps")
                else:
                    max_from_policy = policy_for_phase(normalized_policy, phase).get("maxSteps")
            except Exception:
                max_from_policy = None
        try:
            max_steps = int(max_from_policy) if max_from_policy is not None else _default_max_steps(phase, items_list)
        except Exception:
            max_steps = _default_max_steps(phase, items_list)
    
    # Use frontend-provided batchState (they know current state)
    batch_state_dict = batch_state.copy()  # Use as-is from frontend
    
    # Build payload in legacy format (for flow_planner compatibility)
    payload = {
        "batchId": phase,
        "platformGoal": goal or "",
        "businessContext": business_context or "",
        "industry": industry or "General",
        "service": service or "",
        "useCase": use_case,
        "requiredUploads": uploads_list,
        "personalizationSummary": personalization_summary,
        "stepDataSoFar": answers,
        "alreadyAskedKeys": asked_step_ids,
        "formPlan": form_plan_list,
        "batchPolicy": normalized_policy or {},
        "psychologyPlan": psychology_plan or {},
        "batchNumber": batch_number,
        "batchState": batch_state_dict,
        "maxSteps": max_steps,
        "allowedMiniTypes": allowed_mini_types,
    }
    if instance_subcategories:
        payload["instanceSubcategories"] = instance_subcategories
    
    # Add items array if present (for DSPy to know what to generate)
    if items_list:
        payload["items"] = items_list
    
    return payload


def _insert_row(table: str, row: Dict[str, Any]) -> bool:
    client = get_supabase_client()
    if not client:
        return False
    try:
        client.table(table).insert(row).execute()
        return True
    except Exception as e:
        print(f"[Supabase] Error inserting into {table}: {e}")
        return False


def insert_telemetry_event(event: Dict[str, Any]) -> bool:
    payload = event.get("payload_json") if isinstance(event, dict) else None
    row = {
        "session_id": event.get("session_id"),
        "instance_id": event.get("instance_id"),
        "event_type": event.get("event_type"),
        "step_id": event.get("step_id"),
        "batch_id": event.get("batch_id"),
        "model_request_id": event.get("model_request_id"),
        "payload_json": payload if isinstance(payload, dict) else payload,
    }
    return _insert_row("telemetry_events", row)


def insert_feedback_event(event: Dict[str, Any]) -> bool:
    payload = event.get("payload_json") if isinstance(event, dict) else None
    row = {
        "session_id": event.get("session_id"),
        "instance_id": event.get("instance_id"),
        "event_type": event.get("event_type") or "step_feedback",
        "step_id": event.get("step_id"),
        "batch_id": event.get("batch_id"),
        "model_request_id": event.get("model_request_id"),
        "payload_json": payload if isinstance(payload, dict) else payload,
    }
    return _insert_row("telemetry_events", row)
