from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Set

from supabase import create_client


def _get_supabase_client():
    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL/NEXT_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY.")
    return create_client(url, key)


def _load_existing_names(path: str) -> Set[str]:
    names: Set[str] = set()
    if not os.path.exists(path):
        return names
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = (line or "").strip()
            if not t:
                continue
            try:
                obj = json.loads(t)
            except Exception:
                continue
            name = obj.get("name") if isinstance(obj, dict) else None
            if isinstance(name, str):
                names.add(name)
    return names


REQUEST_SIGNATURES = {
    "batchId",
    "platformGoal",
    "businessContext",
    "industry",
    "service",
    "maxSteps",
    "allowedMiniTypes",
    "formPlan",
}

RESPONSE_SIGNATURES = {"miniSteps", "response", "generatedSteps", "steps"}


def _build_case_name(step_id: Optional[str], request_id: str) -> str:
    sid = (step_id or "step").replace(" ", "-")
    return f"feedback_{sid}_{request_id}"


def _normalize_tags(tags: Any) -> List[str]:
    if isinstance(tags, list):
        return [str(t) for t in tags if t]
    return []


def _fetch_feedback(client, since: Optional[str], limit: Optional[int], include_negative: bool) -> List[Dict[str, Any]]:
    query = (
        client.table("telemetry_events")
        .select("*")
        .eq("event_type", "step_feedback")
        .order("created_at", desc=True)
    )
    if since:
        query = query.gte("created_at", since)
    if limit:
        query = query.limit(limit)
    result = query.execute()
    rows = result.data or []
    filtered = []
    for row in rows:
        payload = row.get("payload_json") or {}
        if not isinstance(payload, dict):
            payload = {}
        if payload.get("send_to_dataset") is True:
            filtered.append(row)
            continue
        if include_negative:
            vote = payload.get("vote")
            rating = payload.get("rating")
            if vote == "down" or (isinstance(rating, int) and rating <= 2):
                filtered.append(row)
    return filtered


def _looks_like_request(payload: Dict[str, Any]) -> bool:
    return any(key in payload for key in REQUEST_SIGNATURES)


def _looks_like_response(payload: Dict[str, Any]) -> bool:
    return any(key in payload for key in RESPONSE_SIGNATURES)


def _extract_request_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in ("request_json", "request", "requestPayload", "trace_request"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate
    if _looks_like_request(payload):
        return payload
    return None


def _extract_response_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in ("response_json", "response", "trace_response"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate
    if _looks_like_response(payload):
        return payload
    return None


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export feedback-driven eval cases from Supabase.")
    parser.add_argument("--since", help="ISO timestamp filter (created_at >= since).")
    parser.add_argument("--limit", type=int, help="Max feedback rows to consider.")
    parser.add_argument("--include-negative", action="store_true", help="Include downvotes/low ratings without send_to_dataset.")
    parser.add_argument("--out", default="eval/eval_cases.jsonl", help="Output eval cases JSONL path.")
    parser.add_argument("--failures-out", default="eval/eval_cases_failures.jsonl", help="Output failures JSONL path.")
    args = parser.parse_args()

    client = _get_supabase_client()
    feedback = _fetch_feedback(client, args.since, args.limit, args.include_negative)

    existing = _load_existing_names(args.out)
    failures_existing = _load_existing_names(args.failures_out)
    cases: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for row in feedback:
        request_id = row.get("model_request_id")
        if not isinstance(request_id, str):
            continue
        name = _build_case_name(row.get("step_id"), request_id)
        if name in existing and name in failures_existing:
            continue

        row_payload = row.get("payload_json") or {}
        if not isinstance(row_payload, dict):
            row_payload = {}

        tags = ["feedback"]
        tags.extend(_normalize_tags(row_payload.get("tags")))
        if row_payload.get("source"):
            tags.append(str(row_payload.get("source")))

        request_payload = _extract_request_payload(row_payload)
        if not isinstance(request_payload, dict):
            continue

        response_payload = _extract_response_payload(row_payload)

        expected = {
            "bad_step_id": row.get("step_id"),
            "feedback_tags": _normalize_tags(row_payload.get("tags")),
            "comment": row_payload.get("comment"),
            "source": row_payload.get("source"),
        }

        if name not in existing:
            cases.append({"name": name, "tags": tags, "payload": request_payload, "expected": expected})

        if name not in failures_existing:
            failures.append(
                {
                    "name": name,
                    "request": request_payload,
                    "response": response_payload,
                    "feedback": {
                        "session_id": row.get("session_id"),
                        "instance_id": row.get("instance_id"),
                        "step_id": row.get("step_id"),
                        "model_request_id": request_id,
                        "rating": row_payload.get("rating"),
                        "vote": row_payload.get("vote"),
                        "tags": row_payload.get("tags"),
                        "comment": row_payload.get("comment"),
                        "source": row_payload.get("source"),
                        "created_at": row.get("created_at"),
                    },
                }
            )

    _write_jsonl(args.out, cases)
    _write_jsonl(args.failures_out, failures)
    print(f"Wrote {len(cases)} eval cases and {len(failures)} failures.")


if __name__ == "__main__":
    main()
