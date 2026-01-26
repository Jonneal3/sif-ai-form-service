#!/usr/bin/env python3
"""
CI guardrail: Check DSPy examples for vertical-specific leaks.

This script scans example files for forbidden vocabulary that would
bake industry-specific content into optimized DSPy programs.

Usage:
    python scripts/check_example_leaks.py [path/to/examples.jsonl]

Exit codes:
    0 - No leaks detected
    1 - Leaks detected or error
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Forbidden vocabulary that should never appear in DSPy demos.
# Keep this list focused on *vertical leaks* (not generic materials/units).
# The goal is to prevent accidentally training the program toward a specific industry/service.
FORBIDDEN_TERMS = {
    "pool",
    "patio",
    "kitchen",
    "bathroom",
    "landscaping",
    "remodel",
    "renovation",
}


def _iter_strings(obj: object, path: str = ""):
    if isinstance(obj, str):
        yield path, obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            next_path = f"{path}.{key}" if path else key
            yield from _iter_strings(v, next_path)
        return
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            next_path = f"{path}[{i}]" if path else f"[{i}]"
            yield from _iter_strings(v, next_path)
        return


def check_example_for_leaks(example: object) -> list[tuple[str, str]]:
    leaks: list[tuple[str, str]] = []
    terms = sorted(FORBIDDEN_TERMS, key=len, reverse=True)
    for field_path, text in _iter_strings(example, ""):
        lower = text.lower()
        for term in terms:
            if term in lower:
                leaks.append((field_path or "<root>", term))
    return leaks


def check_file(file_path: Path) -> tuple[bool, list[tuple[str, str, str]]]:
    """
    Check a JSONL file for leaks.

    Returns:
        (has_leaks, list of (line_num, field_path, term))
    """
    leaks: list[tuple[str, str, str]] = []
    has_leaks = False

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
                example_leaks = check_example_for_leaks(example)

                for field_path, term in example_leaks:
                    leaks.append((str(line_num), field_path, term))
                    has_leaks = True

            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}", file=sys.stderr, flush=True)
                continue

    return has_leaks, leaks


def _iter_jsonl_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted([p for p in path.rglob("*.jsonl") if p.is_file()])
    return []


def main() -> int:
    """Main entry point."""
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        # Default: check all demo packs (including per-use-case packs).
        target = Path(__file__).parent.parent / "src" / "programs" / "form_pipeline" / "demos"

    files = _iter_jsonl_files(target)
    if not files:
        print(f"Error: No .jsonl files found under: {target}", file=sys.stderr, flush=True)
        return 1

    print(f"Checking {len(files)} JSONL file(s) under {target} for vertical leaks...", flush=True)
    print(f"Forbidden terms: {', '.join(sorted(FORBIDDEN_TERMS))}", flush=True)
    print("-" * 60, flush=True)

    has_leaks = False
    leaks: list[tuple[str, str, str, str]] = []  # (file, line_num, field_path, term)
    for fp in files:
        file_has_leaks, file_leaks = check_file(fp)
        if file_has_leaks:
            has_leaks = True
            for line_num, field_path, term in file_leaks:
                leaks.append((str(fp), line_num, field_path, term))

    if has_leaks:
        print(f"\nLEAKS DETECTED: {len(leaks)} violation(s)\n", flush=True)
        for file_path, line_num, field_path, term in leaks:
            print(f"  {file_path}:{line_num} {field_path}: '{term}'", flush=True)
        print("\nFix: Remove vertical-specific language from demos.", flush=True)
        return 1
    else:
        print("\nNo leaks detected. Examples are vertical-agnostic.", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
