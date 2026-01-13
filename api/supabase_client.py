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


async def fetch_form_config(session_id: str, instance_id: Optional[str] = None) -> Dict[str, Any]:
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
    
    try:
        # Fetch session (instance-backed; no forms join)
        result = client.table("sessions").select("*").eq("id", session_id).execute()
        
        if result.data and len(result.data) > 0:
            session = result.data[0]
            resolved_instance_id = instance_id or session.get("instance_id") or session.get("instanceId")
            instance_subcategories = []
            if resolved_instance_id:
                instance_subcategories = await fetch_instance_subcategories(str(resolved_instance_id))

            instance_use_case = session.get("use_case") or session.get("useCase") or ""
            instance_data: Dict[str, Any] = {}
            if resolved_instance_id:
                try:
                    instance_result = (
                        client.table("instances")
                        .select("*")
                        .eq("id", str(resolved_instance_id))
                        .execute()
                    )
                    if instance_result.data and len(instance_result.data) > 0:
                        instance_data = instance_result.data[0]
                        if not instance_use_case:
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
    client = get_supabase_client()
    if not client:
        return {}
    
    try:
        # Adjust column names to match your schema
        result = client.table("sessions").select(
            "step_data,asked_step_ids,form_plan,personalization_summary"
        ).eq("id", session_id).execute()
        
        if result.data and len(result.data) > 0:
            session = result.data[0]
            return {
                "answers": session.get("step_data", {}),
                "asked_step_ids": session.get("asked_step_ids", []),
                "form_plan": session.get("form_plan", []),
                "personalization_summary": session.get("personalization_summary", ""),
            }
    except Exception as e:
        print(f"[Supabase] Error fetching session state: {e}")
    
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
    form_config = await fetch_form_config(session_id, instance_id=instance_id)
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
    
    # Batch-specific logic
    is_batch_1 = batch_id == "ContextCore"
    is_batch_2 = batch_id == "PersonalGuide"
    
    # Calculate allowedMiniTypes based on batch
    if allowed_step_types_override:
        allowed_mini_types = allowed_step_types_override
    elif is_batch_1:
        allowed_mini_types = ["choice"]  # Batch 1: only choice questions
    else:
        allowed_mini_types = ["choice", "slider", "text"]  # Batch 2: more variety
    
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
    if uploads_list and "upload" not in allowed_mini_types and "file_upload" not in allowed_mini_types:
        allowed_mini_types = list(allowed_mini_types) + ["upload"]
    
    # Use provided items or calculate from formPlan
    items_list = items or []
    if not items_list and form_plan_list:
        # Calculate items array (formPlan items for this batch, excluding already asked)
        for item in form_plan_list:
            key = item.get("key", "")
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
    
    # Calculate maxSteps
    if max_steps_override:
        max_steps = max_steps_override
    elif is_batch_1:
        max_steps = 5  # Batch 1: fixed at 5
    else:
        # Batch 2: remaining items + upload steps
        max_steps = len(items_list) if items_list else 10
    
    # Use frontend-provided batchState (they know current state)
    batch_state_dict = batch_state.copy()  # Use as-is from frontend
    
    # Build payload in legacy format (for flow_planner compatibility)
    payload = {
        "batchId": batch_id,
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
