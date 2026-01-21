"""
Copy pack loader/compiler for DSPy flow planner.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple


def _compact_json(obj: Any) -> str:
    try:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=True, sort_keys=True)
    except Exception:
        return json.dumps(str(obj), separators=(",", ":"), ensure_ascii=True)


def _pack_path(pack_id: str) -> Path:
    safe = str(pack_id or "").strip()
    if not safe or re.search(r"[^a-zA-Z0-9_-]", safe):
        safe = "default_v1"
    filename = f"{safe}.json" if not safe.endswith(".json") else safe
    base = Path(__file__).resolve().parent / "packs"
    return base / filename


def load_pack(pack_id: str) -> Dict[str, Any]:
    path = _pack_path(pack_id)
    if not path.exists() and str(pack_id).strip() != "default_v1":
        path = _pack_path("default_v1")
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Copy pack must be a JSON object")
    return data


def compile_pack(pack: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    tone = pack.get("tone") or {}
    rules = pack.get("rules") or []
    banned = pack.get("banned_phrases") or []
    limits = pack.get("limits") or {}
    templates = pack.get("templates") or {}
    pack_id = str(pack.get("id") or "default_v1")
    version = str(pack.get("version") or "1")

    style_snippet = {
        "id": pack_id,
        "v": version,
        "tone": tone,
        "rules": rules,
        "banned": banned,
        "limits": limits,
        "templates": {
            "lead_capture_prompt": templates.get("lead_capture_prompt") or "",
            "privacy_reassurance": templates.get("privacy_reassurance") or "",
            "upload_reassurance": templates.get("upload_reassurance") or "",
        },
    }

    lint_config: Dict[str, Any] = {
        "limits": limits,
        "banned_phrases": banned,
        "templates": style_snippet["templates"],
        "pack_id": pack_id,
        "pack_version": version,
        "reassurance_phrases": [
            t
            for t in [
                style_snippet["templates"].get("privacy_reassurance"),
                style_snippet["templates"].get("upload_reassurance"),
            ]
            if isinstance(t, str) and t.strip()
        ],
        "risk_levels": {
            "low": [
                "multiple_choice",
                "choice",
                "segmented_choice",
                "chips_multi",
                "yes_no",
                "image_choice_grid",
                "rating",
                "slider",
                "range_slider",
                "budget_cards",
                "color_picker",
                "searchable_select",
                "pricing",
                "intro",
                "confirmation",
                "designer",
                "composite",
            ],
            "medium": ["text", "text_input"],
            "high": ["lead_capture", "email", "phone"],
            "very_high": ["upload", "file_upload", "file_picker"],
        },
    }

    return _compact_json(style_snippet), lint_config
