"""
Convert between pretty JSON and compact JSONL formats for DSPy examples.

Usage:
    # Pretty -> Compact (for use in DSPy)
    python convert_examples.py pretty.json compact.jsonl

    # Compact -> Pretty (for editing/reading)
    python convert_examples.py compact.jsonl pretty.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def pretty_to_compact(pretty_path: Path) -> List[Dict[str, Any]]:
    """
    Load pretty JSON format and convert to compact format.

    Pretty format has:
    - context_json as a dict (not string)
    - mini_steps_jsonl as a list (not JSONL string), OR
    - batches_json as a dict (not JSON string)

    Compact format has:
    - context_json as a JSON string
    - mini_steps_jsonl as a JSONL string (one step per line), OR
    - batches_json as a JSON string
    """
    with open(pretty_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    compact_examples: List[Dict[str, Any]] = []

    for example in examples:
        compact = {
            "meta": example.get("meta", {}),
            "inputs": {},
            "outputs": {},
        }

        # Convert inputs
        inputs = example.get("inputs", {})
        compact_inputs = inputs.copy()

        # Convert context_json from dict to JSON string
        context_json = inputs.get("context_json")
        if isinstance(context_json, dict):
            compact_inputs["context_json"] = json.dumps(context_json, separators=(",", ":"))
        else:
            compact_inputs["context_json"] = str(context_json)

        compact["inputs"] = compact_inputs

        # Convert outputs
        outputs = example.get("outputs", {})
        compact_outputs = outputs.copy()

        # Convert mini_steps_jsonl from list to JSONL string (next-steps demos).
        if "mini_steps_jsonl" in outputs:
            mini_steps = outputs.get("mini_steps_jsonl")
            if isinstance(mini_steps, list):
                compact_outputs["mini_steps_jsonl"] = "\n".join(
                    json.dumps(step, separators=(",", ":")) for step in mini_steps
                )
            else:
                compact_outputs["mini_steps_jsonl"] = str(mini_steps)

        # Convert batches_json from dict to JSON string (batch-config demos).
        if "batches_json" in outputs:
            batches_json = outputs.get("batches_json")
            if isinstance(batches_json, dict) or isinstance(batches_json, list):
                compact_outputs["batches_json"] = json.dumps(batches_json, separators=(",", ":"))
            else:
                compact_outputs["batches_json"] = str(batches_json)

        compact["outputs"] = compact_outputs
        compact_examples.append(compact)

    return compact_examples


def compact_to_pretty(compact_path: Path) -> List[Dict[str, Any]]:
    """
    Load compact JSONL format and convert to pretty format.

    Compact format has:
    - context_json as a JSON string
    - mini_steps_jsonl as a JSONL string (one step per line), OR
    - batches_json as a JSON string

    Pretty format has:
    - context_json as a dict (not string)
    - mini_steps_jsonl as a list (not JSONL string), OR
    - batches_json as a dict (not JSON string)
    """
    pretty_examples: List[Dict[str, Any]] = []

    with open(compact_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}", file=sys.stderr, flush=True)
                continue

            pretty = {
                "meta": example.get("meta", {}),
                "inputs": {},
                "outputs": {},
            }

            # Convert inputs
            inputs = example.get("inputs", {})
            pretty_inputs = inputs.copy()

            # Convert context_json from JSON string to dict
            context_json = inputs.get("context_json")
            if isinstance(context_json, str):
                try:
                    pretty_inputs["context_json"] = json.loads(context_json)
                except json.JSONDecodeError:
                    pretty_inputs["context_json"] = context_json  # Keep as-is if invalid
            else:
                pretty_inputs["context_json"] = context_json

            pretty["inputs"] = pretty_inputs

            # Convert outputs
            outputs = example.get("outputs", {})
            pretty_outputs = outputs.copy()

            # Convert mini_steps_jsonl from JSONL string to list (next-steps demos).
            if "mini_steps_jsonl" in outputs:
                mini_steps_jsonl = outputs.get("mini_steps_jsonl")
                if isinstance(mini_steps_jsonl, str):
                    steps_list: List[Dict[str, Any]] = []
                    for step_line in mini_steps_jsonl.strip().split("\n"):
                        if step_line.strip():
                            try:
                                steps_list.append(json.loads(step_line))
                            except json.JSONDecodeError:
                                pass  # Skip invalid lines
                    pretty_outputs["mini_steps_jsonl"] = steps_list
                else:
                    pretty_outputs["mini_steps_jsonl"] = mini_steps_jsonl

            # Convert batches_json from JSON string to dict (batch-config demos).
            if "batches_json" in outputs:
                batches_json = outputs.get("batches_json")
                if isinstance(batches_json, str):
                    try:
                        pretty_outputs["batches_json"] = json.loads(batches_json)
                    except json.JSONDecodeError:
                        pretty_outputs["batches_json"] = batches_json
                else:
                    pretty_outputs["batches_json"] = batches_json

            pretty["outputs"] = pretty_outputs
            pretty_examples.append(pretty)

    return pretty_examples


def write_compact(examples: List[Dict[str, Any]], output_path: Path) -> None:
    """Write examples in compact JSONL format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, separators=(",", ":")) + "\n")


def write_pretty(examples: List[Dict[str, Any]], output_path: Path) -> None:
    """Write examples in pretty JSON format."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)


def detect_format(file_path: Path) -> str:
    """Detect if file is pretty JSON or compact JSONL."""
    with open(file_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    # Try to parse as JSON (pretty format)
    try:
        data = json.loads(first_line)
        # If it's a dict with "meta" key, it's compact JSONL
        if isinstance(data, dict) and "meta" in data:
            return "compact"
    except json.JSONDecodeError:
        pass

    # Try to parse entire file as JSON array (pretty format)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list) and data and isinstance(data[0], dict) and "inputs" in data[0]:
                # Check if context_json is a dict (pretty) or string (compact)
                if isinstance(data[0].get("inputs", {}).get("context_json"), dict):
                    return "pretty"
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

    # Default: assume compact JSONL
    return "compact"


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 3:
        print(
            "Usage: python convert_examples.py <input> <output>\n"
            "  Converts between pretty JSON and compact JSONL formats.\n"
            "  Automatically detects input format.",
            file=sys.stderr,
        )
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr, flush=True)
        return 1

    # Detect format
    input_format = detect_format(input_path)
    output_format = "compact" if output_path.suffix == ".jsonl" else "pretty"

    print(f"Converting {input_format} -> {output_format}...", flush=True)

    # Convert
    if input_format == "pretty" and output_format == "compact":
        examples = pretty_to_compact(input_path)
        write_compact(examples, output_path)
        print(f"✅ Converted {len(examples)} examples to compact JSONL", flush=True)
    elif input_format == "compact" and output_format == "pretty":
        examples = compact_to_pretty(input_path)
        write_pretty(examples, output_path)
        print(f"✅ Converted {len(examples)} examples to pretty JSON", flush=True)
    else:
        print(f"Error: Unsupported conversion {input_format} -> {output_format}", file=sys.stderr, flush=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
