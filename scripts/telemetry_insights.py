#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from supabase import create_client


CHECKPOINT_DEFAULT = Path(".telemetry_checkpoint.json")
SUMMARY_DEFAULT = Path("data/telemetry_summary.json")
DROP_OFF_EVENT_TYPES = {"dropoff", "form_dropoff", "abandon", "abandoned_step", "user_left", "timeout"}
DROP_OFF_PAYLOAD_KEYS = {"dropoff", "abandoned", "cancelled", "timeout", "gave_up", "dropped"}


def _get_supabase_client():
    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL/NEXT_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY.")
    return create_client(url, key)


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_checkpoint(path: Path) -> Dict[str, Optional[str]]:
    if not path.exists():
        return {"last_created_at": None, "last_id": None}
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            return {"last_created_at": None, "last_id": None}
    return {"last_created_at": data.get("last_created_at"), "last_id": data.get("last_id")}


def _save_checkpoint(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"last_created_at": row.get("created_at"), "last_id": row.get("id")}
    path.write_text(json.dumps(checkpoint, ensure_ascii=False), encoding="utf-8")


def _rows_since_checkpoint(rows: List[Dict[str, Any]], checkpoint: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
    if not checkpoint["last_created_at"]:
        return rows
    last_ts = _parse_timestamp(checkpoint["last_created_at"])
    last_id = checkpoint.get("last_id")
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        created_at = _parse_timestamp(row.get("created_at"))
        if not created_at or not last_ts:
            filtered.append(row)
            continue
        if created_at < last_ts:
            continue
        if created_at == last_ts and last_id is not None:
            if str(row.get("id")) <= str(last_id):
                continue
        filtered.append(row)
    return filtered


def _fetch_new_rows(client, since: Optional[str], limit: int) -> List[Dict[str, Any]]:
    query = client.table("telemetry_events").select("*").order("created_at", ascending=True)
    if since:
        query = query.gte("created_at", since)
    if limit:
        query = query.limit(limit)
    result = query.execute()
    return result.data or []


def _is_dropoff(event_type: str, payload: Dict[str, Any]) -> bool:
    if event_type and event_type.lower() in DROP_OFF_EVENT_TYPES:
        return True
    status = str(payload.get("stepStatus") or payload.get("status") or "").lower()
    if status in {"abandoned", "cancelled", "failed", "dropped"}:
        return True
    for key in DROP_OFF_PAYLOAD_KEYS:
        val = payload.get(key)
        if isinstance(val, bool) and val:
            return True
        if isinstance(val, str) and val.lower() in {"true", "yes", "1"}:
            return True
    return False


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_stats: Dict[str, Dict[str, Dict[str, int]]] = {}
    dropoffs: List[Dict[str, Any]] = []
    feedback = {"count": 0, "rating_sum": 0, "rating_counts": {}, "votes": {"up": 0, "down": 0}}

    for row in rows:
        batch_id = row.get("batch_id") or "unknown"
        event_type = (row.get("event_type") or "unknown").lower()
        step_id = row.get("step_id") or "unknown"
        payload = row.get("payload_json") or {}
        if not isinstance(payload, dict):
            payload = {}

        batch_entry = batch_stats.setdefault(batch_id, {"events": defaultdict(int), "steps": {}})
        batch_entry["events"][event_type] += 1
        step_entry = batch_entry["steps"].setdefault(step_id, defaultdict(int))
        step_entry[event_type] += 1

        if event_type == "step_feedback":
            rating = payload.get("rating")
            vote = payload.get("vote")
            feedback["count"] += 1
            if isinstance(rating, int):
                feedback["rating_sum"] += rating
                key = str(rating)
                feedback["rating_counts"][key] = feedback["rating_counts"].get(key, 0) + 1
            if isinstance(vote, str):
                normalized = vote.lower()
                if normalized in feedback["votes"]:
                    feedback["votes"][normalized] += 1

        if _is_dropoff(event_type, payload):
            dropoffs.append(
                {
                    "batch_id": batch_id,
                    "step_id": step_id,
                    "event_type": event_type,
                    "timestamp": row.get("created_at"),
                    "payload": payload,
                }
            )

    return {"batch_stats": batch_stats, "dropoffs": dropoffs, "feedback": feedback, "processed": len(rows)}


def _merge_summary(existing: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    existing.setdefault("batch_stats", {})
    for batch_id, metrics in update["batch_stats"].items():
        tgt_batch = existing["batch_stats"].setdefault(batch_id, {"events": {}, "steps": {}})
        for event_type, count in metrics["events"].items():
            tgt_batch["events"][event_type] = tgt_batch["events"].get(event_type, 0) + count
        for step_id, step_metrics in metrics["steps"].items():
            tgt_step = tgt_batch["steps"].setdefault(step_id, {})
            for event_type, count in step_metrics.items():
                tgt_step[event_type] = tgt_step.get(event_type, 0) + count

    existing.setdefault("feedback", {"count": 0, "rating_sum": 0, "rating_counts": {}, "votes": {"up": 0, "down": 0}})
    fb = existing["feedback"]
    for key in ("count", "rating_sum"):
        fb[key] = fb.get(key, 0) + update["feedback"].get(key, 0)
    for rating, value in update["feedback"]["rating_counts"].items():
        fb["rating_counts"][rating] = fb["rating_counts"].get(rating, 0) + value
    for vote, value in update["feedback"]["votes"].items():
        fb["votes"][vote] = fb["votes"].get(vote, 0) + value

    existing.setdefault("dropoffs", [])
    max_dropoffs = 200
    combined = existing["dropoffs"] + update["dropoffs"]
    existing["dropoffs"] = combined[-max_dropoffs:]

    existing.setdefault("runs", [])
    existing["runs"].append(
        {
            "timestamp": _now_iso(),
            "processed": update.get("processed", 0),
            "dropoff_events": len(update["dropoffs"]),
            "feedback_events": update["feedback"]["count"],
        }
    )

    existing["last_updated"] = _now_iso()
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize telemetry into incremental insights.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_DEFAULT), help="Checkpoint path for processed telemetry.")
    parser.add_argument("--summary", default=str(SUMMARY_DEFAULT), help="Summary output path.")
    parser.add_argument("--limit", type=int, default=1000, help="Max rows to pull per run.")
    parser.add_argument("--since", help="Manually override the last-created timestamp filter.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    summary_path = Path(args.summary)
    checkpoint = _load_checkpoint(checkpoint_path)
    client = _get_supabase_client()
    since = args.since or checkpoint.get("last_created_at")

    rows = _fetch_new_rows(client, since, args.limit)
    rows = _rows_since_checkpoint(rows, checkpoint)
    if not rows:
        print("No new telemetry to summarize.")
        return

    summary = summary_path.exists()
    summary_data: Dict[str, Any] = {}
    if summary:
        with summary_path.open("r", encoding="utf-8") as f:
            try:
                summary_data = json.load(f)
            except Exception:
                summary_data = {}

    update = _summarize(rows)
    merged = _merge_summary(summary_data, update)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    _save_checkpoint(checkpoint_path, rows[-1])

    print(f"Processed {len(rows)} telemetry rows and updated {summary_path}")


if __name__ == "__main__":
    main()
