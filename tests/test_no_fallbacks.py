from __future__ import annotations

from pathlib import Path


def test_repo_contains_no_fallback_language() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    skip_dirs = {
        ".git",
        "__pycache__",
        ".venv",
        "node_modules",
        "compiled",
        "site",
    }
    allowed_exts = {
        ".py",
        ".md",
        ".json",
        ".txt",
        ".sh",
        ".toml",
        ".yml",
        ".yaml",
    }

    hits: list[str] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix and path.suffix not in allowed_exts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "fallback" in text.lower():
            hits.append(str(path.relative_to(repo_root)))

    assert not hits, f"Remove 'fallback' references: {hits[:50]}"

