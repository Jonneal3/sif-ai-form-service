from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
_SERVICE_FROM_SUMMARY_RE = re.compile(r"^(.{3,96}?)\s+is\s+(?:an?\s+)?service\b", re.IGNORECASE)
_SERVICE_LABEL_RE = re.compile(r"\bService:\s*([^\.\n]{3,140})", re.IGNORECASE)


def _coerce_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (dict, list)):
        try:
            return json.dumps(raw, ensure_ascii=True, separators=(",", ":"))
        except Exception:
            return str(raw)
    return str(raw)


def _looks_like_uuid(text: str) -> bool:
    t = str(text or "").strip()
    return bool(t and _UUID_RE.match(t))


def _is_uuid_list(raw: Any) -> bool:
    if not isinstance(raw, list) or not raw:
        return False
    items = [str(x or "").strip() for x in raw]
    items = [x for x in items if x]
    return bool(items) and all(_looks_like_uuid(x) for x in items)


def _string_is_json_uuid_list(text: str) -> bool:
    t = str(text or "").strip()
    if not (t.startswith("[") and t.endswith("]")):
        return False
    try:
        parsed = json.loads(t)
    except Exception:
        return False
    return _is_uuid_list(parsed)


def _service_name_from_summary(text: str) -> str:
    """
    Attempt to extract a short service name from common summary formats.
    Examples:
      - "Bathroom remodeling is a service where ..." -> "Bathroom remodeling"
      - "Industry: Bathroom Remodeling. Service: Guest bath refresh." -> "Guest bath refresh"
    """
    t = str(text or "").strip()
    if not t:
        return ""
    m = _SERVICE_LABEL_RE.search(t)
    if m:
        return m.group(1).strip()
    m = _SERVICE_FROM_SUMMARY_RE.match(t)
    if m:
        return m.group(1).strip()
    return ""


def _normalize_use_case(raw: Any) -> str:
    t = str(raw or "").strip().lower().replace("_", "-")
    if t in {"tryon", "try-on"}:
        return "tryon"
    if t in {"scene", "scene-placement"}:
        return t
    return t


def _dedupe_keep_order(urls: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for u in urls:
        s = str(u or "").strip()
        if not s or s in seen:
            continue
        out.append(s)
        seen.add(s)
    return out


def extract_reference_images(payload: Dict[str, Any]) -> Tuple[List[str], Optional[str], Optional[str]]:
    """
    Normalize reference images across request shapes.

    Returns:
      - `reference_images[]` (deduped, ordered)
      - `scene_image` (best-effort)
      - `product_image` (best-effort)
    """
    scene_image = payload.get("sceneImage") or payload.get("scene_image")
    product_image = payload.get("productImage") or payload.get("product_image")

    scene = str(scene_image).strip() if isinstance(scene_image, str) and scene_image.strip() else None
    product = str(product_image).strip() if isinstance(product_image, str) and product_image.strip() else None

    ref_raw = payload.get("referenceImages") or payload.get("reference_images") or []
    refs: List[str] = []
    if isinstance(ref_raw, list):
        refs = [str(x).strip() for x in ref_raw if isinstance(x, str) and str(x).strip()]

    # Prefer explicit scene/product first when provided; many placement models expect this ordering.
    ordered = _dedupe_keep_order([scene or "", product or "", *refs])
    return ordered[:8], scene, product


def extract_negative_prompt(payload: Dict[str, Any]) -> str:
    raw = payload.get("negativePrompt") or payload.get("negative_prompt") or ""
    t = _coerce_text(raw).strip()
    t = " ".join(t.split())  # collapse whitespace/newlines
    return t[:480]


def _extract_step_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw = payload.get("stepDataSoFar") or payload.get("step_data_so_far") or {}
    return raw if isinstance(raw, dict) else {}


def _extract_style_tags(step_data: Dict[str, Any]) -> List[str]:
    raw = step_data.get("style")
    tags: List[str] = []
    if isinstance(raw, list):
        for x in raw:
            t = str(x or "").strip()
            if t:
                tags.append(t[:48])
    elif isinstance(raw, str):
        for part in raw.split(","):
            t = part.strip()
            if t:
                tags.append(t[:48])
    # de-dupe while preserving order
    out: List[str] = []
    seen: set[str] = set()
    for t in tags:
        k = t.lower()
        if k in seen:
            continue
        out.append(t)
        seen.add(k)
    return out[:12]


def _best_effort_service(step_data: Dict[str, Any], payload: Dict[str, Any]) -> str:
    def _extract_service_id() -> str:
        for k in ("service_primary", "step-service-primary", "service"):
            raw = step_data.get(k)
            if isinstance(raw, str):
                s = raw.strip()
                if _looks_like_uuid(s):
                    return s
                if _string_is_json_uuid_list(s):
                    try:
                        parsed = json.loads(s)
                    except Exception:
                        continue
                    if isinstance(parsed, list):
                        for x in parsed:
                            xs = str(x or "").strip()
                            if _looks_like_uuid(xs):
                                return xs
            if isinstance(raw, list):
                for x in raw:
                    xs = str(x or "").strip()
                    if _looks_like_uuid(xs):
                        return xs
        return ""

    def _instance_context_service_name() -> str:
        ctx = payload.get("instanceContext") if isinstance(payload.get("instanceContext"), dict) else None
        if not ctx and isinstance(payload.get("instance_context"), dict):
            ctx = payload.get("instance_context")
        if isinstance(ctx, dict):
            svc = ctx.get("service")
            if isinstance(svc, dict):
                name = str(svc.get("name") or svc.get("label") or "").strip()
                if name:
                    return name

            # If we have a service id and an id->summary mapping, derive a name from that summary.
            summaries = ctx.get("serviceSummariesBySubcategoryId") or ctx.get("service_summaries_by_subcategory_id")
            if isinstance(summaries, dict):
                service_id = _extract_service_id()
                if service_id:
                    summary = summaries.get(service_id)
                    if isinstance(summary, str) and summary.strip():
                        derived = _service_name_from_summary(summary)
                        if derived:
                            return derived

            # Back-compat: sometimes a single summary is present; derive a name from it.
            for k in ("serviceSummary", "service_summary", "services_summary", "servicesSummary"):
                v = ctx.get(k)
                if isinstance(v, str) and v.strip():
                    derived = _service_name_from_summary(v)
                    if derived:
                        return derived

            # Back-compat: sometimes `subcategories[]` exists.
            subs = ctx.get("subcategories")
            if isinstance(subs, list) and subs:
                first = subs[0]
                if isinstance(first, dict):
                    name = str(first.get("name") or first.get("label") or "").strip()
                    if name:
                        return name
                name = str(first or "").strip()
                if name and not _looks_like_uuid(name):
                    return name
        return ""

    def _step_service_primary_text() -> str:
        for k in ("step-service-primary", "service_primary", "service"):
            raw = step_data.get(k)
            if isinstance(raw, dict):
                name = str(raw.get("name") or raw.get("label") or "").strip()
                if name:
                    return name
                continue
            if isinstance(raw, list):
                # Many widgets store this as an array of service IDs; ignore if it's purely ids.
                if _is_uuid_list(raw):
                    continue
                parts: List[str] = []
                for x in raw:
                    t = str(x or "").strip()
                    if not t or _looks_like_uuid(t):
                        continue
                    parts.append(t)
                if parts:
                    return ", ".join(parts)
                continue
            if isinstance(raw, str) and _string_is_json_uuid_list(raw):
                continue
            t = _coerce_text(raw).strip()
            if t and not _looks_like_uuid(t):
                return t
        return ""

    # Prefer explicit human labels over ids.
    candidates = [
        _instance_context_service_name(),
        _step_service_primary_text(),
        payload.get("service"),
        payload.get("serviceSummary"),
        payload.get("service_summary"),
        payload.get("servicesSummary"),
        payload.get("services_summary"),
    ]
    for c in candidates:
        if isinstance(c, dict):
            name = str(c.get("name") or c.get("label") or "").strip()
            if name and not _looks_like_uuid(name):
                return name
            continue
        if _is_uuid_list(c):
            continue
        t = _coerce_text(c).strip()
        if not t:
            continue
        if _looks_like_uuid(t):
            continue
        if _string_is_json_uuid_list(t):
            continue
        derived = _service_name_from_summary(t)
        if derived:
            return derived
        # If caller passed a verbose summary, keep it but cap later.
        return t
    return ""


def _extract_location(step_data: Dict[str, Any]) -> str:
    city = step_data.get("location_city") or step_data.get("locationCity") or ""
    state = step_data.get("location_state") or step_data.get("locationState") or ""
    c = str(city or "").strip()
    s = str(state or "").strip()
    if c and s:
        return f"{c}, {s}"
    return c or s


def _kv_details(step_data: Dict[str, Any], *, skip_keys: set[str]) -> List[str]:
    """
    Best-effort: include a few extra answers as compact detail lines.
    """
    out: List[str] = []
    for k, v in step_data.items():
        if k in skip_keys:
            continue
        key = str(k or "").strip()
        if not key:
            continue
        val = _coerce_text(v).strip()
        if not val:
            continue
        if _looks_like_uuid(val):
            continue
        # Avoid dumping huge blobs.
        if len(val) > 140:
            val = val[:140].rstrip() + "…"
        out.append(f"{key}: {val}")
        if len(out) >= 10:
            break
    return out


def build_image_prompt_text(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministically build a high-quality prompt from `stepDataSoFar` + reference images.

    Returns an object shaped like ImagePromptSpec (by alias names).
    """
    step_data = _extract_step_data(payload)
    use_case = _normalize_use_case(payload.get("useCase") or payload.get("use_case"))
    reference_images, scene_image, product_image = extract_reference_images(payload)
    negative_prompt = extract_negative_prompt(payload)

    service = _best_effort_service(step_data, payload)
    location = _extract_location(step_data)
    style_tags = _extract_style_tags(step_data)
    notes = str(step_data.get("notes") or "").strip()
    budget = str(step_data.get("budget_range") or step_data.get("budgetRange") or "").strip()
    timeline = str(step_data.get("timeline") or "").strip()

    # Keep deterministic but helpful: a short, structured prompt with explicit constraints.
    lines: List[str] = []

    if use_case == "scene-placement":
        lines.append("Create a photorealistic scene-placement composite.")
        if service:
            lines.append(f"Project/service context: {service[:240]}.")
        if location:
            lines.append(f"Location context: {location[:80]}.")
        lines.append("Use the provided images as strict visual references.")
        if scene_image:
            lines.append("Scene image: use as the background; preserve camera angle and lighting.")
        if product_image:
            lines.append("Product image: place the product naturally into the scene.")
        lines.append("Match scale, perspective, and shadows; ensure seamless blending; no text/logos/watermarks.")
    else:
        subject = service[:140] if service else "home service"
        if location:
            lines.append(f"Generate a realistic image of a {subject} project in {location[:80]}.")
        else:
            lines.append(f"Generate a realistic image of a {subject} project.")

    if style_tags:
        lines.append(f"Style: {', '.join(style_tags)}.")
    if notes:
        lines.append(f"Notes: {notes[:280]}.")
    if budget:
        lines.append(f"Budget: {budget[:80]}.")
    if timeline:
        lines.append(f"Timeline: {timeline[:80]}.")

    # Include a few additional details for specificity (but avoid noisy ids).
    skip_keys = {
        "step-service-primary",
        "service_primary",
        "service",
        "location_city",
        "locationCity",
        "location_state",
        "locationState",
        "style",
        "notes",
        "budget_range",
        "budgetRange",
        "timeline",
    }
    extra = _kv_details(step_data, skip_keys=skip_keys)
    if extra:
        lines.append("Additional details:")
        lines.extend([f"- {x}" for x in extra])

    if reference_images:
        lines.append("Reference images (provided separately to the model):")
        for i, u in enumerate(reference_images[:4]):
            lines.append(f"{i+1}. {u}")

    if negative_prompt:
        lines.append(f"Avoid: {negative_prompt}.")

    prompt = "\n".join([x for x in lines if str(x).strip()]).strip()

    return {
        "prompt": prompt,
        "negativePrompt": negative_prompt,
        "styleTags": style_tags,
        "metadata": {
            "useCase": use_case,
            "location": location,
            "referenceImagesCount": len(reference_images),
            "hasSceneImage": bool(scene_image),
            "hasProductImage": bool(product_image),
        },
    }


__all__ = [
    "build_image_prompt_text",
    "extract_negative_prompt",
    "extract_reference_images",
]
