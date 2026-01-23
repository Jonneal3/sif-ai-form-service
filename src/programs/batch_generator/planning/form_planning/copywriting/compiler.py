from __future__ import annotations

import json
from typing import Any, Dict, Tuple


def load_pack(pack_id: str) -> Dict[str, Any]:
    pid = (pack_id or "").strip() or "default_v1"
    if pid not in {"default_v1"}:
        pid = "default_v1"
    return {
        "pack_id": pid,
        "pack_version": "1",
        "style": {
            "tone": "direct, friendly, professional",
            "question_rules": [
                "Ask one thing at a time.",
                "Use concrete nouns; avoid generic filler.",
                "Avoid parenthetical enumerations when options are present.",
                "Keep questions under ~12 words when possible.",
            ],
            "option_rules": [
                "Use parallel phrasing across options.",
                "Avoid overly broad location lists unless the service is outdoor-specific.",
                "Include 'Not sure' only when it reduces drop-off.",
            ],
        },
        "lint": {
            "require_question_mark": True,
            "max_question_chars": 120,
            "banned_question_substrings": ["(install, replace, repair)"],
        },
    }


def compile_pack(pack: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    style = pack.get("style") if isinstance(pack.get("style"), dict) else {}
    lint = pack.get("lint") if isinstance(pack.get("lint"), dict) else {}
    lint_config: Dict[str, Any] = {
        "pack_id": str(pack.get("pack_id") or "").strip() or "default_v1",
        "pack_version": str(pack.get("pack_version") or "").strip() or "1",
        **lint,
    }
    style_snippet_json = json.dumps(style, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return style_snippet_json, lint_config


__all__ = ["load_pack", "compile_pack"]

