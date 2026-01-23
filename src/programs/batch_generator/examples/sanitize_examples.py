"""
Sanitize DSPy examples to remove vertical-specific content.

This utility ensures examples teach structure and behavior, not industry-specific content.
It replaces vertical terms with generic placeholders before examples become "demos" in DSPy.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set


# Common vertical terms to replace (add more as needed)
VERTICAL_TERMS: Dict[str, str] = {
    # Industry nouns
    "pool": "product_a",
    "patio": "product_b",
    "kitchen": "product_c",
    "bathroom": "product_d",
    "landscaping": "product_e",
    "remodel": "service_a",
    "renovation": "service_b",
    "installation": "service_c",
    "repair": "service_d",
    # Material/domain terms
    "tile": "material_a",
    "concrete": "material_b",
    "wood": "material_c",
    "stone": "material_d",
    "depth": "dimension_a",
    "width": "dimension_b",
    "length": "dimension_c",
    # Common business terms
    "estimate": "outcome_a",
    "quote": "outcome_b",
    "pricing": "outcome_c",
}

# Generic placeholder patterns
GENERIC_PATTERNS = {
    "industry": r"Vertical_\d+",
    "service": r"Service_[A-Z]",
    "product": r"Product_[A-Z]",
    "option": r"Option_[A-Z]",
    "attribute": r"attribute_[a-z]",
    "style": r"Style_[A-Z]",
    "value": r"[a-z_]+",
}

# Forbidden vocabulary that should never appear in structural examples
FORBIDDEN_TERMS: Set[str] = {
    "pool",
    "patio",
    "kitchen",
    "bathroom",
    "landscaping",
    "remodel",
    "renovation",
    "tile",
    "concrete",
    "wood",
    "stone",
    "depth",
    "width",
    "length",
    "sq ft",
    "square feet",
    "cubic",
    "gallon",
    "pound",
    "acre",
}


def sanitize_text(text: str, context: str = "") -> str:
    """
    Replace vertical-specific terms with generic placeholders.

    Args:
        text: Text to sanitize
        context: Optional context hint (e.g., "label", "question", "goal")

    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return text

    result = text
    # Replace known vertical terms
    for term, replacement in VERTICAL_TERMS.items():
        # Case-insensitive replacement, preserving word boundaries
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        result = pattern.sub(replacement, result)

    # Replace any remaining industry-specific patterns
    # This is a conservative approach - only replace if we're confident
    return result


def sanitize_context_json(context_json: str | Dict[str, Any]) -> str:
    """
    Sanitize the context_json field, replacing vertical-specific content.

    Args:
        context_json: JSON string or dict to sanitize

    Returns:
        Sanitized JSON string
    """
    if isinstance(context_json, str):
        try:
            obj = json.loads(context_json)
        except json.JSONDecodeError:
            return context_json  # Return as-is if invalid JSON
    else:
        obj = context_json

    if not isinstance(obj, dict):
        return json.dumps(obj) if isinstance(context_json, dict) else context_json

    sanitized = obj.copy()

    # Sanitize industry/service fields - use generic placeholders
    if "industry" in sanitized and sanitized["industry"]:
        if not re.match(GENERIC_PATTERNS["industry"], str(sanitized["industry"])):
            # Replace with generic vertical ID
            sanitized["industry"] = "Vertical_01"

    if "service" in sanitized and sanitized["service"]:
        if not re.match(GENERIC_PATTERNS["service"], str(sanitized["service"])):
            sanitized["service"] = "Service_A"

    # Sanitize form_plan entries
    if "form_plan" in sanitized and isinstance(sanitized["form_plan"], list):
        sanitized["form_plan"] = [
            {
                **item,
                "key": sanitize_key(item.get("key", "")),
                "goal": sanitize_text(str(item.get("goal", ""))),
                "why": sanitize_text(str(item.get("why", ""))),
            }
            for item in sanitized["form_plan"]
            if isinstance(item, dict)
        ]

    # Sanitize attribute_families
    if "attribute_families" in sanitized and isinstance(sanitized["attribute_families"], list):
        sanitized["attribute_families"] = [
            {
                **item,
                "family": sanitize_key(item.get("family", "")),
                "goal": sanitize_text(str(item.get("goal", ""))),
            }
            for item in sanitized["attribute_families"]
            if isinstance(item, dict)
        ]

    # Sanitize service_anchor_terms - replace with generic
    if "service_anchor_terms" in sanitized and isinstance(sanitized["service_anchor_terms"], list):
        sanitized["service_anchor_terms"] = ["visual", "design"]

    # Sanitize platform_goal and business_context
    if "platform_goal" in sanitized:
        sanitized["platform_goal"] = sanitize_text(str(sanitized["platform_goal"]))
    if "business_context" in sanitized:
        sanitized["business_context"] = sanitize_text(str(sanitized["business_context"]))

    return json.dumps(sanitized, separators=(",", ":"))


def sanitize_key(key: str) -> str:
    """
    Sanitize a key/identifier to use generic naming.

    Args:
        key: Key to sanitize (e.g., "project_type", "area_location")

    Returns:
        Sanitized key (e.g., "attribute_a", "attribute_b")
    """
    if not isinstance(key, str):
        return str(key)

    # If already generic (attribute_X pattern), keep it
    if re.match(r"^attribute_[a-z]$", key):
        return key

    # Replace common patterns
    key_lower = key.lower()
    for term, replacement in VERTICAL_TERMS.items():
        if term in key_lower:
            key_lower = key_lower.replace(term, replacement)

    # Convert to attribute_X pattern if needed
    if not key_lower.startswith("attribute_"):
        # Generate a stable hash-based attribute name
        import hashlib

        hash_val = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        attr_idx = chr(ord("a") + (hash_val % 26))
        return f"attribute_{attr_idx}"

    return key_lower


def sanitize_output_jsonl(jsonl: str) -> str:
    """
    Sanitize the mini_steps_jsonl output field.

    Args:
        jsonl: JSONL string to sanitize

    Returns:
        Sanitized JSONL string
    """
    if not isinstance(jsonl, str):
        return jsonl

    lines = jsonl.strip().split("\n")
    sanitized_lines = []

    for line in lines:
        if not line.strip():
            continue
        try:
            step = json.loads(line)
            if not isinstance(step, dict):
                sanitized_lines.append(line)
                continue

            sanitized_step = step.copy()

            # Sanitize question text
            if "question" in sanitized_step:
                sanitized_step["question"] = sanitize_text(str(sanitized_step["question"]))

            # Sanitize id
            if "id" in sanitized_step:
                sanitized_step["id"] = sanitize_key(str(sanitized_step["id"]).replace("step-", ""))

            # Sanitize options
            if "options" in sanitized_step and isinstance(sanitized_step["options"], list):
                sanitized_step["options"] = [
                    {
                        "label": sanitize_text(str(opt.get("label", ""))),
                        "value": sanitize_key(str(opt.get("value", ""))),
                    }
                    for opt in sanitized_step["options"]
                    if isinstance(opt, dict)
                ]

            # Sanitize placeholder
            if "placeholder" in sanitized_step:
                sanitized_step["placeholder"] = sanitize_text(str(sanitized_step["placeholder"]))

            sanitized_lines.append(json.dumps(sanitized_step, separators=(",", ":")))
        except json.JSONDecodeError:
            # If we can't parse, keep original
            sanitized_lines.append(line)

    return "\n".join(sanitized_lines)


def sanitize_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize a single example record.

    Args:
        example: Example dict with meta, inputs, outputs

    Returns:
        Sanitized example dict
    """
    sanitized = {"meta": example.get("meta", {})}

    # Sanitize inputs
    inputs = example.get("inputs", {})
    sanitized_inputs = inputs.copy()

    if "context_json" in sanitized_inputs:
        sanitized_inputs["context_json"] = sanitize_context_json(sanitized_inputs["context_json"])

    sanitized["inputs"] = sanitized_inputs

    # Sanitize outputs
    outputs = example.get("outputs", {})
    sanitized_outputs = outputs.copy()

    if "mini_steps_jsonl" in sanitized_outputs:
        sanitized_outputs["mini_steps_jsonl"] = sanitize_output_jsonl(
            sanitized_outputs["mini_steps_jsonl"]
        )

    sanitized["outputs"] = sanitized_outputs

    return sanitized


def detect_leaks(text: str) -> List[str]:
    """
    Detect forbidden vertical-specific terms in text.

    Uses word boundary matching to avoid false positives (e.g., "max_length" shouldn't match "length").

    Args:
        text: Text to check

    Returns:
        List of detected forbidden terms
    """
    if not isinstance(text, str):
        return []

    found: List[str] = []
    text_lower = text.lower()

    for term in FORBIDDEN_TERMS:
        term_lower = term.lower()
        # Use word boundary matching to avoid false positives
        # Match whole words only (not substrings)
        pattern = re.compile(rf"\b{re.escape(term_lower)}\b")
        if pattern.search(text_lower):
            found.append(term)

    return found


def check_example_for_leaks(example: Dict[str, Any]) -> List[tuple[str, str]]:
    """
    Check an example for vertical-specific leaks.

    Args:
        example: Example dict to check

    Returns:
        List of (field_path, detected_term) tuples
    """
    leaks: List[tuple[str, str]] = []

    # Check inputs.context_json
    context_json = example.get("inputs", {}).get("context_json", "")
    if context_json:
        if isinstance(context_json, str):
            try:
                context_obj = json.loads(context_json)
            except json.JSONDecodeError:
                context_obj = {}
        else:
            context_obj = context_json

        context_str = json.dumps(context_obj)
        for term in detect_leaks(context_str):
            leaks.append(("inputs.context_json", term))

    # Check outputs.mini_steps_jsonl
    jsonl = example.get("outputs", {}).get("mini_steps_jsonl", "")
    if jsonl:
        for term in detect_leaks(jsonl):
            leaks.append(("outputs.mini_steps_jsonl", term))

    return leaks


def sanitize_jsonl_file(input_path: Path, output_path: Path | None = None) -> int:
    """
    Sanitize a JSONL file of examples.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output file (default: input_path with .sanitized.jsonl)

    Returns:
        Number of examples processed
    """
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}.sanitized.jsonl"

    examples: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}", flush=True)

    sanitized_examples = [sanitize_example(ex) for ex in examples]

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in sanitized_examples:
            f.write(json.dumps(ex, separators=(",", ":")) + "\n")

    return len(sanitized_examples)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sanitize_examples.py <input.jsonl> [output.jsonl]", flush=True)
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}", flush=True)
        sys.exit(1)

    count = sanitize_jsonl_file(input_file, output_file)
    print(f"Sanitized {count} examples -> {output_file or input_file.parent / f'{input_file.stem}.sanitized.jsonl'}", flush=True)
