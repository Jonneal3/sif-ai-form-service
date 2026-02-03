from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from programs.form_pipeline.context_builder import build_context


_CURRENCY_RE = re.compile(r"(?i)\b(usd|cad|aud|gbp|eur)\b")


def _parse_money_token(raw: str) -> Optional[int]:
    """
    Parse a single money token to an int (USD-ish), best-effort.

    Supports: "$12,000", "12000", "12k", "12.5k".
    """
    t = str(raw or "").strip().lower()
    if not t:
        return None
    t = t.replace(",", "")
    t = t.replace("$", "")

    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*k$", t)
    if m:
        try:
            return int(float(m.group(1)) * 1000)
        except Exception:
            return None

    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)$", t)
    if m:
        try:
            return int(float(m.group(1)))
        except Exception:
            return None
    return None


def _extract_money_range(value: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract a (low, high) range from any common payload value.
    """
    if value is None:
        return None, None
    if isinstance(value, (int, float)):
        v = int(value)
        return (v, v) if v > 0 else (None, None)
    if isinstance(value, dict):
        # Common shapes: { min, max }, { low, high }, { value }, etc.
        for a, b in (("min", "max"), ("low", "high"), ("rangeLow", "rangeHigh"), ("range_low", "range_high")):
            lo = _extract_money_range(value.get(a))[0]
            hi = _extract_money_range(value.get(b))[1]
            if lo or hi:
                return lo, hi
        return _extract_money_range(value.get("value"))
    if isinstance(value, list) and value:
        # Prefer the first scalar-ish item.
        for item in value[:3]:
            lo, hi = _extract_money_range(item)
            if lo or hi:
                return lo, hi
        return None, None

    s = str(value)
    s_norm = s.replace(",", "")
    # Grab up to 2 money-ish tokens.
    toks = re.findall(r"\$?\s*\d+(?:\.\d+)?\s*k?\b", s_norm, flags=re.IGNORECASE)
    vals: List[int] = []
    for tok in toks[:4]:
        v = _parse_money_token(tok)
        if isinstance(v, int) and v > 0:
            vals.append(v)
    if not vals:
        return None, None
    vals = sorted(vals)
    if len(vals) == 1:
        return vals[0], vals[0]
    return vals[0], vals[-1]


def _extract_budget_hint(step_data: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(step_data, dict):
        return None, None

    # Look for any key that smells like a budget answer.
    for k in list(step_data.keys()):
        key = str(k or "").strip().lower()
        if not key:
            continue
        if "budget" in key or "price" in key or "cost" in key:
            lo, hi = _extract_money_range(step_data.get(k))
            if lo or hi:
                return lo, hi
    return None, None


def _extract_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None
    if isinstance(value, dict):
        return _extract_int(value.get("value"))
    if isinstance(value, list) and value:
        for item in value[:3]:
            v = _extract_int(item)
            if v:
                return v
    s = str(value or "").strip().lower()
    if not s:
        return None
    m = re.search(r"(\d{1,7})", s)
    if not m:
        return None
    try:
        n = int(m.group(1))
    except Exception:
        return None
    return n if n > 0 else None


def _extract_quantity_hints(step_data: Dict[str, Any]) -> Dict[str, int]:
    if not isinstance(step_data, dict):
        return {}
    hints: Dict[str, int] = {}
    for k, v in step_data.items():
        key = str(k or "").strip().lower()
        if not key:
            continue
        if key in {"sqft", "squarefeet", "square_feet", "squarefootage", "square_footage", "area"}:
            n = _extract_int(v)
            if n:
                hints["sqft"] = n
        elif key in {"rooms", "room_count", "bedrooms", "bathrooms"}:
            n = _extract_int(v)
            if n:
                hints[key] = n
    return hints


def _detect_currency(payload: Dict[str, Any]) -> str:
    raw = payload.get("currency") or payload.get("Currency")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().upper()[:8]

    # Try to infer from free-text fields.
    for k in ("serviceSummary", "service_summary", "companySummary", "company_summary"):
        v = payload.get(k)
        if isinstance(v, str):
            m = _CURRENCY_RE.search(v)
            if m:
                return m.group(1).upper()
    return "USD"


def _base_range_for_service(text: str) -> Tuple[int, int, str]:
    """
    Lightweight heuristic baseline ranges (USD).

    Returns (low, high, matched_rule).
    """
    t = (text or "").lower()

    rules: List[Tuple[str, Tuple[int, int]]] = [
        ("kitchen", (25000, 90000)),
        ("bath", (15000, 60000)),
        ("roof", (8000, 28000)),
        ("hvac", (5000, 16000)),
        ("landscap", (3000, 25000)),
        ("deck", (8000, 35000)),
        ("fence", (2500, 12000)),
        ("floor", (3000, 25000)),
        ("window", (4000, 22000)),
        ("paint", (2000, 14000)),
        ("pool", (30000, 130000)),
        ("patio", (6000, 40000)),
        ("driveway", (5000, 30000)),
        ("siding", (9000, 35000)),
        ("plumb", (500, 8000)),
        ("electric", (800, 12000)),
    ]
    for key, (lo, hi) in rules:
        if key in t:
            return lo, hi, key
    return 2000, 15000, "default"


def _multiplier_from_text(text: str) -> float:
    t = (text or "").lower()
    mult = 1.0

    if any(w in t for w in ("luxury", "high end", "high-end", "premium", "custom")):
        mult *= 1.35
    if any(w in t for w in ("budget", "basic", "affordable", "cheap")):
        mult *= 0.85
    if any(w in t for w in ("full", "complete", "gut", "total", "entire")):
        mult *= 1.2
    if any(w in t for w in ("partial", "refresh", "touch up", "touch-up", "minor")):
        mult *= 0.9

    return max(0.6, min(2.0, mult))


def _clamp_range(lo: int, hi: int) -> Tuple[int, int]:
    lo_i = int(max(0, lo))
    hi_i = int(max(lo_i, hi))
    return lo_i, hi_i


def estimate_pricing(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Estimate a rough pricing range (no model calls).

    Input payload is expected to match the same general shape as `next_steps_jsonl()`.
    """
    request_id = f"pricing_{int(time.time() * 1000)}"

    ctx = build_context(payload)
    services_summary = str(ctx.get("services_summary") or ctx.get("service_summary") or "").strip()
    industry = str(ctx.get("industry") or "").strip()
    service = str(ctx.get("service") or "").strip()
    if not services_summary and not industry and not service:
        return {
            "ok": False,
            "error": "Missing service context (provide serviceSummary/service_summary or industry/service).",
            "requestId": request_id,
        }

    step_data_raw = payload.get("stepDataSoFar") or payload.get("step_data_so_far") or {}
    step_data = step_data_raw if isinstance(step_data_raw, dict) else {}

    currency = _detect_currency(payload)
    basis_text = " ".join([services_summary, industry, service]).strip()
    base_lo, base_hi, matched_rule = _base_range_for_service(basis_text)

    budget_lo, budget_hi = _extract_budget_hint(step_data)
    qty = _extract_quantity_hints(step_data)

    text_hint = f"{basis_text} {step_data}".lower()
    mult = _multiplier_from_text(text_hint)

    # Light quantity scaling (only when hints exist).
    if isinstance(qty.get("sqft"), int):
        sqft = int(qty["sqft"])
        # Keep conservative: up to 3x at very large sizes.
        mult *= max(0.75, min(3.0, sqft / 600.0))
    elif isinstance(qty.get("rooms"), int):
        rooms = int(qty["rooms"])
        mult *= max(0.8, min(2.0, rooms / 3.0))

    lo = int(base_lo * mult)
    hi = int(base_hi * mult)

    lo, hi = _clamp_range(lo, hi)

    # If the user gave a budget range, nudge the estimate toward it, but don't hard-clamp.
    if isinstance(budget_lo, int) and isinstance(budget_hi, int) and budget_lo > 0 and budget_hi >= budget_lo:
        mid_est = (lo + hi) / 2.0
        mid_budget = (budget_lo + budget_hi) / 2.0
        blend = 0.35
        mid = (1.0 - blend) * mid_est + blend * mid_budget
        width = max(hi - lo, int(0.4 * mid))
        lo = int(max(0, mid - width / 2.0))
        hi = int(max(lo, mid + width / 2.0))
        lo, hi = _clamp_range(lo, hi)

    confidence = "low"
    if matched_rule != "default" and (budget_lo or budget_hi or qty):
        confidence = "medium"

    notes: List[str] = [
        "Heuristic estimate (no contractor quote).",
        f"Matched rule: {matched_rule}.",
    ]
    if budget_lo or budget_hi:
        notes.append("Incorporates budget hint from answers.")
    if qty:
        notes.append("Incorporates size/quantity hints from answers.")

    return {
        "ok": True,
        "requestId": request_id,
        "currency": currency,
        "rangeLow": int(lo),
        "rangeHigh": int(hi),
        "confidence": confidence,
        "basis": "heuristic_v1",
        "notes": notes,
        "inputs": {
            "industry": industry,
            "service": service,
            "serviceSummary": services_summary[:600],
            "budgetHint": {"low": budget_lo, "high": budget_hi},
            "quantityHints": qty,
        },
    }
