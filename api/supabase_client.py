"""
Supabase client for fetching form data.

Uses official Supabase Python client with realtime support.
The backend fetches all form configuration, session state, and grounding data
from Supabase, so clients only need to send minimal identifiers.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from supabase import create_client, Client


_client: Optional[Client] = None


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


async def fetch_form_config(session_id: str) -> Dict[str, Any]:
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
            "required_uploads": [...]
        }
    """
    client = get_supabase_client()
    if not client:
        return {}
    
    try:
        # Fetch session with related form (adjust table/column names to match your schema)
        result = client.table("sessions").select("*, forms(*)").eq("id", session_id).execute()
        
        if result.data and len(result.data) > 0:
            session = result.data[0]
            # Handle both direct form data and nested forms relation
            form = session.get("forms") or {}
            if isinstance(form, list) and len(form) > 0:
                form = form[0]
            
            return {
                "platform_goal": form.get("platform_goal", ""),
                "business_context": form.get("business_context", ""),
                "industry": form.get("industry", "General"),
                "service": form.get("service", ""),
                "max_steps": form.get("max_steps", 4),
                "allowed_step_types": form.get("allowed_step_types", []),
                "required_uploads": form.get("required_uploads", []),
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


async def fetch_grounding_data(session_id: str, industry: Optional[str] = None, service: Optional[str] = None) -> str:
    """
    Fetch grounding/RAG data from Supabase.
    
    Returns grounding preview string (JSON or text).
    """
    client = get_supabase_client()
    if not client:
        return ""
    
    try:
        # Adjust table name to match your schema (could be "grounding", "rag_data", etc.)
        query = client.table("grounding").select("preview,data")
        
        if industry:
            query = query.eq("industry", industry)
        if service:
            query = query.eq("service", service)
        
        result = query.execute()
        
        if result.data and len(result.data) > 0:
            grounding = result.data[0]
            return grounding.get("preview", "") or grounding.get("data", "")
    except Exception as e:
        print(f"[Supabase] Error fetching grounding data: {e}")
    
    return ""


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
    grounding: Optional[str] = None,
    max_steps_override: Optional[int] = None,
    allowed_step_types_override: Optional[List[str]] = None,
    required_uploads: Optional[List[Dict[str, Any]]] = None,
    items: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build full planner payload from Supabase (static data) + frontend (dynamic state).
    
    If frontend provides form config (platformGoal, businessContext, etc.), use it.
    Otherwise, fetch from Supabase (avoids duplicate requests if frontend already has it).
    
    Frontend always provides:
    - Batch state (callsUsed, satiety, etc.)
    - Step data (answers so far)
    - Already asked keys
    - Form plan (AI-generated structure)
    - Personalization summary
    """
    # Fetch static data from Supabase only if frontend didn't provide it
    if not all([goal, business_context, industry, service]):
        form_config = await fetch_form_config(session_id)
        goal = goal or form_config.get("platform_goal", "")
        business_context = business_context or form_config.get("business_context", "")
        industry = industry or form_config.get("industry", "General")
        service = service or form_config.get("service", "")
    else:
        form_config = {}  # Not needed if frontend provided everything
    
    # Fetch grounding only if frontend didn't provide it
    if not grounding:
        grounding = await fetch_grounding_data(session_id, industry, service)
    grounding_preview = (grounding[:2000] if grounding else "")
    
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
        "groundingPreview": grounding_preview,  # First 2000 chars (for logging)
        "requiredUploads": uploads_list,
        "personalizationSummary": personalization_summary,
        "stepDataSoFar": answers,
        "alreadyAskedKeys": asked_step_ids,
        "formPlan": form_plan_list,
        "batchState": batch_state_dict,
        "maxSteps": max_steps,
        "allowedMiniTypes": allowed_mini_types,
    }
    
    # Add items array if present (for DSPy to know what to generate)
    if items_list:
        payload["items"] = items_list
    
    return payload

