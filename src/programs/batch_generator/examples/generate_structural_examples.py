"""
Generate structural examples programmatically.

Creates vertical-agnostic examples that teach structure and behavior,
not industry-specific content.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List


def generate_attribute_key(index: int) -> str:
    """Generate a generic attribute key like attribute_a, attribute_b, etc."""
    return f"attribute_{chr(ord('a') + (index % 26))}"


def generate_vertical_id(index: int) -> str:
    """Generate a generic vertical ID like Vertical_01, Vertical_02, etc."""
    return f"Vertical_{index:02d}"


def generate_service_id(index: int) -> str:
    """Generate a generic service ID like Service_A, Service_B, etc."""
    return f"Service_{chr(ord('A') + (index % 26))}"


def generate_option_label(option_index: int, total: int) -> str:
    """Generate a generic option label."""
    if option_index == total - 1:
        return "Not sure"
    return f"Option {chr(ord('A') + option_index)}"


def generate_option_value(option_index: int, total: int) -> str:
    """Generate a generic option value."""
    if option_index == total - 1:
        return "not_sure"
    return f"option_{chr(ord('a') + option_index)}"


def create_basic_choice_example(example_index: int) -> Dict[str, Any]:
    """Create a basic example with two choice steps."""
    attr_a = generate_attribute_key(example_index * 2)
    attr_b = generate_attribute_key(example_index * 2 + 1)

    return {
        "meta": {
            "name": f"structural_basic_choices_{example_index}",
            "notes": "Two choice steps with deterministic ids. Teaches: valid JSONL format, option shape, required field.",
        },
        "inputs": {
            "context_json": json.dumps(
                {
                    "platform_goal": "Collect attributes for visual generation.",
                    "business_context": "Keep questions concise.",
                    "industry": generate_vertical_id(example_index + 1),
                    "service": generate_service_id(example_index),
                    "use_case": "scene",
                    "goal_intent": "visual",
                    "required_uploads": [],
                    "personalization_summary": "",
                    "known_answers": {},
                    "already_asked_keys": [],
                    "form_plan": [
                        {
                            "key": attr_a,
                            "goal": "Select primary attribute",
                            "why": "Sets foundation",
                            "component_hint": "choice",
                            "priority": "critical",
                            "importance_weight": 0.2,
                            "expected_metric_gain": 0.18,
                        },
                        {
                            "key": attr_b,
                            "goal": "Choose secondary attribute",
                            "why": "Adds detail",
                            "component_hint": "choice",
                            "priority": "high",
                            "importance_weight": 0.15,
                            "expected_metric_gain": 0.12,
                        },
                    ],
                    "batch_state": {
                        "callsRemaining": 2,
                        "callsUsed": 0,
                        "maxCalls": 2,
                        "satietySoFar": 0,
                        "satietyRemaining": 1,
                        "missingHighImpactKeys": [],
                        "mustHaveCopyNeeded": False,
                    },
                    "attribute_families": [
                        {"family": attr_a, "goal": "Primary attribute selection."},
                        {"family": attr_b, "goal": "Secondary attribute selection."},
                    ],
                    "service_anchor_terms": ["visual", "design"],
                    "items": [],
                    "instance_subcategories": [],
                },
                separators=(",", ":"),
            ),
            "batch_id": "ContextCore",
            "max_steps": 2,
            "allowed_mini_types": ["multiple_choice", "text_input"],
        },
        "outputs": {
            "mini_steps_jsonl": "\n".join(
                [
                    json.dumps(
                        {
                            "id": f"step-{attr_a}",
                            "type": "multiple_choice",
                            "question": "Which primary attribute applies?",
                            "required": True,
                            "options": [
                                {
                                    "label": generate_option_label(i, 4),
                                    "value": generate_option_value(i, 4),
                                }
                                for i in range(4)
                            ],
                        },
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        {
                            "id": f"step-{attr_b}",
                            "type": "multiple_choice",
                            "question": "What secondary attribute fits?",
                            "required": False,
                            "options": [
                                {
                                    "label": generate_option_label(i, 4),
                                    "value": generate_option_value(i, 4),
                                }
                                for i in range(4)
                            ],
                        },
                        separators=(",", ":"),
                    ),
                ]
            ),
        },
    }


def create_skip_already_asked_example(example_index: int) -> Dict[str, Any]:
    """Create an example that skips already-asked steps."""
    attr_a = generate_attribute_key(example_index * 2)
    attr_c = generate_attribute_key(example_index * 2 + 2)

    return {
        "meta": {
            "name": f"structural_skip_already_asked_{example_index}",
            "notes": "Skips step-attribute-a because already_asked_keys contains it. Teaches: respect already_asked_keys, emit only unasked steps.",
        },
        "inputs": {
            "context_json": json.dumps(
                {
                    "platform_goal": "Collect attributes for visual generation.",
                    "business_context": "Keep questions concise.",
                    "industry": generate_vertical_id(example_index + 1),
                    "service": generate_service_id(example_index),
                    "use_case": "scene",
                    "goal_intent": "visual",
                    "required_uploads": [],
                    "personalization_summary": "",
                    "known_answers": {},
                    "already_asked_keys": [f"step-{attr_a}"],
                    "form_plan": [
                        {
                            "key": attr_a,
                            "goal": "Select primary attribute",
                            "why": "Sets foundation",
                            "component_hint": "choice",
                            "priority": "critical",
                            "importance_weight": 0.2,
                            "expected_metric_gain": 0.18,
                        },
                        {
                            "key": attr_c,
                            "goal": "Capture detail attribute",
                            "why": "Adds specificity",
                            "component_hint": "text",
                            "priority": "medium",
                            "importance_weight": 0.1,
                            "expected_metric_gain": 0.08,
                        },
                    ],
                    "batch_state": {
                        "callsRemaining": 2,
                        "callsUsed": 0,
                        "maxCalls": 2,
                        "satietySoFar": 0,
                        "satietyRemaining": 1,
                        "missingHighImpactKeys": [],
                        "mustHaveCopyNeeded": False,
                    },
                    "attribute_families": [
                        {"family": attr_a, "goal": "Primary attribute selection."},
                        {"family": attr_c, "goal": "Detail attribute capture."},
                    ],
                    "service_anchor_terms": ["visual", "design"],
                    "items": [],
                    "instance_subcategories": [],
                },
                separators=(",", ":"),
            ),
            "batch_id": "ContextCore",
            "max_steps": 2,
            "allowed_mini_types": ["multiple_choice", "text_input"],
        },
        "outputs": {
            "mini_steps_jsonl": json.dumps(
                {
                    "id": f"step-{attr_c}",
                    "type": "text_input",
                    "question": "Any additional detail to include?",
                    "required": False,
                    "max_length": 120,
                    "placeholder": "Optional",
                },
                separators=(",", ":"),
            ),
        },
    }


def create_max_steps_example(example_index: int) -> Dict[str, Any]:
    """Create an example that respects max_steps=1."""
    attr_f = generate_attribute_key(example_index * 2 + 5)
    attr_g = generate_attribute_key(example_index * 2 + 6)

    return {
        "meta": {
            "name": f"structural_max_steps_one_{example_index}",
            "notes": "Respects max_steps=1. Teaches: hard limit on step count, emit only one step even if form_plan has more.",
        },
        "inputs": {
            "context_json": json.dumps(
                {
                    "platform_goal": "Collect attributes for visual generation.",
                    "business_context": "Keep questions concise.",
                    "industry": generate_vertical_id(example_index + 1),
                    "service": generate_service_id(example_index),
                    "use_case": "scene",
                    "goal_intent": "visual",
                    "required_uploads": [],
                    "personalization_summary": "",
                    "known_answers": {},
                    "already_asked_keys": [],
                    "form_plan": [
                        {
                            "key": attr_f,
                            "goal": "Choose attribute F",
                            "why": "Frames the subject",
                            "component_hint": "choice",
                            "priority": "high",
                            "importance_weight": 0.12,
                            "expected_metric_gain": 0.1,
                        },
                        {
                            "key": attr_g,
                            "goal": "Select attribute G",
                            "why": "Sets context",
                            "component_hint": "choice",
                            "priority": "medium",
                            "importance_weight": 0.08,
                            "expected_metric_gain": 0.06,
                        },
                    ],
                    "batch_state": {
                        "callsRemaining": 2,
                        "callsUsed": 0,
                        "maxCalls": 2,
                        "satietySoFar": 0,
                        "satietyRemaining": 1,
                        "missingHighImpactKeys": [],
                        "mustHaveCopyNeeded": False,
                    },
                    "attribute_families": [
                        {"family": attr_f, "goal": "Attribute F selection."},
                        {"family": attr_g, "goal": "Attribute G selection."},
                    ],
                    "service_anchor_terms": ["visual", "design"],
                    "items": [],
                    "instance_subcategories": [],
                },
                separators=(",", ":"),
            ),
            "batch_id": "ContextCore",
            "max_steps": 1,
            "allowed_mini_types": ["multiple_choice"],
        },
        "outputs": {
            "mini_steps_jsonl": json.dumps(
                {
                    "id": f"step-{attr_f}",
                    "type": "multiple_choice",
                    "question": "Which attribute F option works best?",
                    "required": True,
                    "options": [
                        {
                            "label": generate_option_label(i, 4),
                            "value": generate_option_value(i, 4),
                        }
                        for i in range(4)
                    ],
                },
                separators=(",", ":"),
            ),
        },
    }


def generate_structural_examples(count: int = 20) -> List[Dict[str, Any]]:
    """
    Generate a set of structural examples.

    Args:
        count: Number of examples to generate

    Returns:
        List of example dicts
    """
    examples: List[Dict[str, Any]] = []

    # Generate variety of example types
    for i in range(count):
        example_type = i % 3
        if example_type == 0:
            examples.append(create_basic_choice_example(i))
        elif example_type == 1:
            examples.append(create_skip_already_asked_example(i))
        else:
            examples.append(create_max_steps_example(i))

    return examples


def write_jsonl(examples: List[Dict[str, Any]], output_path: Path) -> None:
    """Write examples to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    import sys

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("structural_examples_generated.jsonl")

    examples = generate_structural_examples(count)
    write_jsonl(examples, output_path)

    print(f"Generated {len(examples)} structural examples -> {output_path}", flush=True)
