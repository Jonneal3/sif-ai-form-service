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

# Add src to path to import sanitize_examples
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from programs.batch_generator.examples.sanitize_examples import (
    check_example_for_leaks,
    FORBIDDEN_TERMS,
)


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


def main() -> int:
    """Main entry point."""
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        # Default: check structural examples
        file_path = (
            Path(__file__).parent.parent
            / "src"
            / "programs"
            / "batch_generator"
            / "examples"
            / "current"
            / "structural_examples.jsonl"
        )

    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr, flush=True)
        return 1

    print(f"Checking {file_path} for vertical leaks...", flush=True)
    print(f"Forbidden terms: {', '.join(sorted(FORBIDDEN_TERMS))}", flush=True)
    print("-" * 60, flush=True)

    has_leaks, leaks = check_file(file_path)

    if has_leaks:
        print(f"\n❌ LEAKS DETECTED: {len(leaks)} violation(s)\n", flush=True)
        for line_num, field_path, term in leaks:
            print(f"  Line {line_num}, {field_path}: '{term}'", flush=True)
        print(
            "\n💡 Fix: Run sanitize_examples.py or use structural_examples.jsonl",
            flush=True,
        )
        return 1
    else:
        print("\n✅ No leaks detected. Examples are vertical-agnostic.", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
