from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Set

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


def _chunk(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _build_case_name(step_id: Optional[str], request_id: str) -> str:
    sid = (step_id or "step").replace(" ", "-")
    return f"feedback_{sid}_{request_id}"


def _normalize_tags(tags: Any) -> List[str]:
    if isinstance(tags, list):
        return [str(t) for t in tags if t]
    return []


def _fetch_feedback(client, since: Optional[str], limit: Optional[int], include_negative: bool) -> List[Dict[str, Any]]:
    query = client.table("step_feedback").select("*").order("created_at", desc=True)
    if since:
        query = query.gte("created_at", since)
    if limit:
        query = query.limit(limit)
    result = query.execute()
    rows = result.data or []
    filtered = []
    for row in rows:
        if row.get("send_to_dataset") is True:
            filtered.append(row)
            continue
        if include_negative:
            vote = row.get("vote")
            rating = row.get("rating")
            if vote == "down" or (isinstance(rating, int) and rating <= 2):
                filtered.append(row)
    return filtered


def _fetch_model_traces(client, request_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not request_ids:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for chunk in _chunk(request_ids, 50):
        result = client.table("model_trace").select("*").in_("model_request_id", chunk).execute()
        for row in result.data or []:
            req_id = row.get("model_request_id")
            if isinstance(req_id, str):
                out[req_id] = row
    return out


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
    request_ids = [row.get("model_request_id") for row in feedback if row.get("model_request_id")]
    traces = _fetch_model_traces(client, list({rid for rid in request_ids if isinstance(rid, str)}))

    existing = _load_existing_names(args.out)
    failures_existing = _load_existing_names(args.failures_out)
    cases: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for row in feedback:
        request_id = row.get("model_request_id")
        if not isinstance(request_id, str):
            continue
        trace = traces.get(request_id)
        if not trace:
            continue
        name = _build_case_name(row.get("step_id"), request_id)
        if name in existing and name in failures_existing:
            continue

        tags = ["feedback"]
        tags.extend(_normalize_tags(row.get("tags")))
        if row.get("source"):
            tags.append(str(row.get("source")))

        payload = trace.get("request_json")
        if not isinstance(payload, dict):
            continue

        expected = {
            "bad_step_id": row.get("step_id"),
            "feedback_tags": _normalize_tags(row.get("tags")),
            "comment": row.get("comment"),
            "source": row.get("source"),
        }

        if name not in existing:
            cases.append({"name": name, "tags": tags, "payload": payload, "expected": expected})

        if name not in failures_existing:
            failures.append(
                {
                    "name": name,
                    "request": payload,
                    "response": trace.get("response_json"),
                    "feedback": {
                        "session_id": row.get("session_id"),
                        "instance_id": row.get("instance_id"),
                        "step_id": row.get("step_id"),
                        "model_request_id": request_id,
                        "rating": row.get("rating"),
                        "vote": row.get("vote"),
                        "tags": row.get("tags"),
                        "comment": row.get("comment"),
                        "source": row.get("source"),
                        "created_at": row.get("created_at"),
                    },
                }
            )

    _write_jsonl(args.out, cases)
    _write_jsonl(args.failures_out, failures)
    print(f"Wrote {len(cases)} eval cases and {len(failures)} failures.")


if __name__ == "__main__":
    main()
