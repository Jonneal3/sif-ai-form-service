from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EXAMPLES = "src/programs/batch_generator/examples/current/batch_generator_structural_examples.jsonl"

BANNED_TERMS = {
    # Intent / product-strategy language (optimizer-poison for this compiler layer)
    "baseline",
    "pricing",
    "estimate",
    "clarify",
    "quantify",
    "scope",
    "roi",
    "signal",
    "ambiguity",
    "funnel",
    "final",
    "initial",
    "early",
    "middle",
    "late",
}

ALLOWED_COMPONENT_TYPES = {
    "multiple_choice",
    "yes_no",
    "slider",
    "range_slider",
    "file_upload",
}

ALLOWED_GUIDANCE_VERBOSITY = {"low", "medium", "high"}
ALLOWED_GUIDANCE_SPECIFICITY = {"low", "medium", "high"}
ALLOWED_GUIDANCE_BREADTH = {"broad", "focused"}

ALLOWED_CONTEXT_KEYS = {"batch_constraints", "policy", "prior_plan_preview"}
ALLOWED_CONSTRAINT_KEYS = {"maxBatches", "maxStepsPerBatch", "tokenBudgetTotal"}
ALLOWED_POLICY_KEYS = {"noUploads"}


def _iter_strings(obj: Any) -> Iterable[str]:
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                yield k
            yield from _iter_strings(v)
        return
    if isinstance(obj, list):
        for item in obj:
            yield from _iter_strings(item)


def _contains_banned_terms(obj: Any) -> list[str]:
    hits: set[str] = set()
    for s in _iter_strings(obj):
        t = s.lower()
        for term in BANNED_TERMS:
            if term in t:
                hits.add(term)
    return sorted(hits)


def _as_int(x: Any) -> int | None:
    try:
        n = int(x)
    except Exception:
        return None
    return n


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def validate_examples(path: Path) -> int:
    if not path.exists():
        _fail(f"Error: examples file not found: {path}")
        return 2

    raw = path.read_text(encoding="utf-8").splitlines()
    errors: list[str] = []

    for line_no, line in enumerate(raw, 1):
        if not line.strip():
            continue
        try:
            ex = json.loads(line)
        except Exception as e:
            errors.append(f"{path}:{line_no}: invalid JSON: {e}")
            continue

        # Banlist anywhere in record (meta/inputs/outputs).
        hits = _contains_banned_terms(ex)
        if hits:
            errors.append(f"{path}:{line_no}: contains banned terms: {', '.join(hits)}")

        inputs = ex.get("inputs") if isinstance(ex.get("inputs"), dict) else {}
        outputs = ex.get("outputs") if isinstance(ex.get("outputs"), dict) else {}

        # context_json must be JSON string that is constraint-only.
        context_json = inputs.get("context_json")
        if not isinstance(context_json, str):
            errors.append(f"{path}:{line_no}: inputs.context_json must be a JSON string")
            continue
        try:
            context = json.loads(context_json)
        except Exception as e:
            errors.append(f"{path}:{line_no}: inputs.context_json invalid JSON: {e}")
            continue
        if not isinstance(context, dict):
            errors.append(f"{path}:{line_no}: inputs.context_json must parse to an object")
            continue
        extra_ctx = set(context.keys()) - ALLOWED_CONTEXT_KEYS
        if extra_ctx:
            errors.append(f"{path}:{line_no}: context_json has disallowed keys: {sorted(extra_ctx)}")

        # batch_constraints shape
        constraints_in = context.get("batch_constraints")
        if constraints_in is not None:
            if not isinstance(constraints_in, dict):
                errors.append(f"{path}:{line_no}: context_json.batch_constraints must be an object")
            else:
                extra = set(constraints_in.keys()) - ALLOWED_CONSTRAINT_KEYS
                if extra:
                    errors.append(f"{path}:{line_no}: batch_constraints has disallowed keys: {sorted(extra)}")

        policy = context.get("policy")
        if policy is not None:
            if not isinstance(policy, dict):
                errors.append(f"{path}:{line_no}: context_json.policy must be an object")
            else:
                extra = set(policy.keys()) - ALLOWED_POLICY_KEYS
                if extra:
                    errors.append(f"{path}:{line_no}: policy has disallowed keys: {sorted(extra)}")
                if "noUploads" in policy and not isinstance(policy.get("noUploads"), bool):
                    errors.append(f"{path}:{line_no}: policy.noUploads must be boolean")

        prior_preview = context.get("prior_plan_preview")
        if prior_preview is not None:
            if not isinstance(prior_preview, dict):
                errors.append(f"{path}:{line_no}: prior_plan_preview must be an object")
            else:
                batches = prior_preview.get("batches")
                if batches is not None and not isinstance(batches, list):
                    errors.append(f"{path}:{line_no}: prior_plan_preview.batches must be an array")
                if isinstance(batches, list):
                    for i, item in enumerate(batches):
                        if not isinstance(item, dict) or set(item.keys()) != {"phaseIndex"}:
                            errors.append(
                                f"{path}:{line_no}: prior_plan_preview.batches[{i}] must be {{phaseIndex}} only"
                            )

        # batches_json must be a JSON string that conforms to the compiler schema.
        batches_json = outputs.get("batches_json")
        if not isinstance(batches_json, str):
            errors.append(f"{path}:{line_no}: outputs.batches_json must be a JSON string")
            continue
        try:
            plan = json.loads(batches_json)
        except Exception as e:
            errors.append(f"{path}:{line_no}: outputs.batches_json invalid JSON: {e}")
            continue
        if not isinstance(plan, dict):
            errors.append(f"{path}:{line_no}: outputs.batches_json must parse to an object")
            continue

        allowed_top = {"version", "constraints", "batches"}
        extra_top = set(plan.keys()) - allowed_top
        if extra_top:
            errors.append(f"{path}:{line_no}: batches_json has disallowed keys: {sorted(extra_top)}")

        constraints = plan.get("constraints")
        if not isinstance(constraints, dict):
            errors.append(f"{path}:{line_no}: batches_json.constraints must be an object")
            continue
        extra = set(constraints.keys()) - ALLOWED_CONSTRAINT_KEYS
        if extra:
            errors.append(f"{path}:{line_no}: batches_json.constraints has disallowed keys: {sorted(extra)}")

        max_batches = _as_int(constraints.get("maxBatches"))
        if max_batches is None or max_batches <= 0:
            errors.append(f"{path}:{line_no}: constraints.maxBatches must be a positive integer")
            continue

        policy_no_uploads = bool(isinstance(policy, dict) and policy.get("noUploads") is True)

        batches = plan.get("batches")
        if not isinstance(batches, list) or not batches:
            errors.append(f"{path}:{line_no}: batches_json.batches must be a non-empty array")
            continue
        if len(batches) > max_batches:
            errors.append(f"{path}:{line_no}: batches_json.batches exceeds maxBatches ({len(batches)} > {max_batches})")

        seen_phase_indexes: list[int] = []
        for idx, b in enumerate(batches):
            if not isinstance(b, dict):
                errors.append(f"{path}:{line_no}: batches[{idx}] must be an object")
                continue
            allowed_phase_keys = {"phaseIndex", "maxSteps", "allowedComponentTypes", "guidance"}
            extra = set(b.keys()) - allowed_phase_keys
            if extra:
                errors.append(f"{path}:{line_no}: batches[{idx}] has disallowed keys: {sorted(extra)}")
            phase_index = _as_int(b.get("phaseIndex"))
            if phase_index is None:
                errors.append(f"{path}:{line_no}: batches[{idx}].phaseIndex must be an integer")
            else:
                seen_phase_indexes.append(phase_index)

            max_steps = _as_int(b.get("maxSteps"))
            if max_steps is None or max_steps <= 0:
                errors.append(f"{path}:{line_no}: batches[{idx}].maxSteps must be a positive integer")

            allowed_types = b.get("allowedComponentTypes")
            if not isinstance(allowed_types, list) or not all(isinstance(x, str) for x in allowed_types):
                errors.append(f"{path}:{line_no}: batches[{idx}].allowedComponentTypes must be a string array")
            else:
                bad = [t for t in allowed_types if t not in ALLOWED_COMPONENT_TYPES]
                if bad:
                    errors.append(f"{path}:{line_no}: batches[{idx}].allowedComponentTypes has unknown types: {bad}")
                if policy_no_uploads and "file_upload" in allowed_types:
                    errors.append(f"{path}:{line_no}: policy.noUploads forbids file_upload in batches[{idx}]")

            guidance = b.get("guidance")
            if not isinstance(guidance, dict):
                errors.append(f"{path}:{line_no}: batches[{idx}].guidance must be an object")
            else:
                allowed_guidance_keys = {"verbosity", "specificity", "breadth"}
                extra = set(guidance.keys()) - allowed_guidance_keys
                if extra:
                    errors.append(f"{path}:{line_no}: batches[{idx}].guidance has disallowed keys: {sorted(extra)}")
                if guidance.get("verbosity") not in ALLOWED_GUIDANCE_VERBOSITY:
                    errors.append(f"{path}:{line_no}: batches[{idx}].guidance.verbosity invalid")
                if guidance.get("specificity") not in ALLOWED_GUIDANCE_SPECIFICITY:
                    errors.append(f"{path}:{line_no}: batches[{idx}].guidance.specificity invalid")
                if guidance.get("breadth") not in ALLOWED_GUIDANCE_BREADTH:
                    errors.append(f"{path}:{line_no}: batches[{idx}].guidance.breadth invalid")

        if seen_phase_indexes:
            expected = list(range(len(batches)))
            if sorted(seen_phase_indexes) != expected:
                errors.append(
                    f"{path}:{line_no}: phaseIndex must be sequential 0..{len(batches)-1} (got {sorted(seen_phase_indexes)})"
                )

    if errors:
        for e in errors:
            _fail(e)
        _fail(f"FAIL: {len(errors)} issue(s) found in {path}")
        return 1

    print(f"OK: {path} ({sum(1 for l in raw if l.strip())} examples)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate batch-plan DSPy demos are optimizer-safe and schema-locked.")
    ap.add_argument("--examples", default=DEFAULT_EXAMPLES, help="Path to batch-plan examples JSONL file.")
    args = ap.parse_args()
    return validate_examples(Path(args.examples))


if __name__ == "__main__":
    raise SystemExit(main())
