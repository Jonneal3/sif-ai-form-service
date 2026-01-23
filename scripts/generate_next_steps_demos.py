#!/usr/bin/env python3
from __future__ import annotations

"""
Generate DSPy demo packs for `programs.batch_generator`.

Why this exists:
- We want demos to be *vertical-agnostic* but *not toy/generic* (avoid “Option A/Category 1”).
- We want demos to stay in sync with the UI-step contract and backend constraints as they evolve.
- We want demo packs keyed by (use_case, total_batches, batch_index) so the program can attach
  the right few-shot examples for the current batch position.

Output layout:
  src/programs/batch_generator/examples/{use_case}/b{total_batches}/batch_{batch_index}.jsonl
  (and later: `.optimized.jsonl` variants produced by an offline optimizer)

Run:
  PYTHONPATH=.:src python scripts/generate_next_steps_demos.py
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dump_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=True) for r in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _step_jsonl(steps: List[dict]) -> str:
    return "\n".join(json.dumps(s, ensure_ascii=True, separators=(",", ":")) for s in steps)


def _assert_step_ids_unique(steps: List[dict]) -> None:
    seen: set[str] = set()
    for s in steps:
        sid = str(s.get("id") or "")
        if not sid:
            raise ValueError("missing step id")
        if sid in seen:
            raise ValueError(f"duplicate step id: {sid}")
        seen.add(sid)


def _validate_steps(steps: List[dict]) -> None:
    """
    Validate emitted demo steps against the current backend UI contract.
    """
    from schemas.ui_steps import (
        BudgetCardsUI,
        ColorPickerUI,
        CompositeUI,
        ConfirmationUI,
        DatePickerUI,
        DesignerUI,
        FileUploadUI,
        GalleryUI,
        IntroUI,
        LeadCaptureUI,
        MultipleChoiceUI,
        PricingUI,
        RatingUI,
        SearchableSelectUI,
        TextInputUI,
    )

    type_to_model = {
        "text": TextInputUI,
        "text_input": TextInputUI,
        "intro": IntroUI,
        "rating": RatingUI,
        "date_picker": DatePickerUI,
        "color_picker": ColorPickerUI,
        "lead_capture": LeadCaptureUI,
        "pricing": PricingUI,
        "confirmation": ConfirmationUI,
        "designer": DesignerUI,
        "file_upload": FileUploadUI,
        "upload": FileUploadUI,
        "file_picker": FileUploadUI,
        "budget_cards": BudgetCardsUI,
        "multiple_choice": MultipleChoiceUI,
        "choice": MultipleChoiceUI,
        "segmented_choice": MultipleChoiceUI,
        "chips_multi": MultipleChoiceUI,
        "yes_no": MultipleChoiceUI,
        "image_choice_grid": MultipleChoiceUI,
        "searchable_select": SearchableSelectUI,
        "composite": CompositeUI,
        "gallery": GalleryUI,
    }

    for s in steps:
        t = str(s.get("type") or "").strip().lower()
        if not t:
            raise ValueError("missing step type")
        model = type_to_model.get(t)
        if not model:
            raise ValueError(f"unsupported demo step type: {t}")
        model.model_validate(s)


@dataclass(frozen=True)
class DemoSpec:
    name: str
    context: Dict[str, Any]
    max_steps: int
    allowed_mini_types: List[str]
    steps: List[Dict[str, Any]]


def _generic_context(*, use_case: str, goal_intent: str, already_asked: list[str] | None = None) -> Dict[str, Any]:
    return {
        "platform_goal": "Pricing intake funnel for a design-first estimate.",
        "business_context": (
            "Ask like a pro designer/estimator. Keep it simple, concrete, and buyer-friendly. "
            "Use multiple choice early; vary option counts; avoid filler like Option A."
        ),
        "industry": "Generic",
        "service": "",
        "use_case": use_case,
        "goal_intent": goal_intent,
        "known_answers": {},
        "already_asked_keys": already_asked or [],
        "asked_step_ids": already_asked or [],
        "required_uploads": [],
        "choice_option_min": 4,
        "choice_option_max": 10,
        "choice_option_target": 7,
        "attribute_families": [],
        "items": [],
        "batch_state": {},
    }


def _choice_step(*, sid: str, q: str, options: list[tuple[str, str]], required: bool = True) -> dict:
    return {
        "id": sid,
        "type": "multiple_choice",
        "question": q,
        "required": required,
        "options": [{"label": lbl, "value": val} for (lbl, val) in options],
    }


def _yesno_step(*, sid: str, q: str, required: bool = True) -> dict:
    return {"id": sid, "type": "yes_no", "question": q, "required": required, "options": []}


def _upload_step(*, sid: str, q: str) -> dict:
    return {"id": sid, "type": "file_upload", "question": q, "required": False, "accept": "image/*", "max_files": 6}


def _gallery_step(*, sid: str, q: str) -> dict:
    return {"id": sid, "type": "gallery", "question": q, "required": True}


def _demo_specs_for(*, use_case: str, total_batches: int, batch_index: int) -> list[DemoSpec]:
    """
    Minimal, vertical-agnostic but realistic demos.
    NOTE: We intentionally avoid domain words that the leak-checker would flag.
    """
    last = batch_index == total_batches - 1

    # Allowed types are intentionally conservative; the orchestrator will enforce real runtime constraints.
    if batch_index == 0:
        allowed = ["multiple_choice"]
        max_steps = 3
    elif last:
        allowed = ["multiple_choice", "file_upload", "gallery"]
        max_steps = 3
    else:
        allowed = ["multiple_choice", "yes_no", "slider", "range_slider"]
        max_steps = 3

    if use_case == "tryon":
        ctx = _generic_context(use_case=use_case, goal_intent="visual")
        ctx["attribute_families"] = [
            {"family": "item_type", "goal": "What item is being tried on (category/type)."},
            {"family": "fit_silhouette", "goal": "How it should fit or drape."},
            {"family": "materials_finishes", "goal": "Material and finish preferences."},
        ]
        base_steps = [
            _choice_step(
                sid="step-item-type",
                q="What are you trying on?",
                options=[
                    ("Hat", "hat"),
                    ("Shirt", "shirt"),
                    ("Jacket", "jacket"),
                    ("Shoes", "shoes"),
                    ("Jewelry", "jewelry"),
                    ("Other", "other"),
                ],
            ),
            _choice_step(
                sid="step-fit-silhouette",
                q="How should it fit?",
                options=[
                    ("Slim", "slim"),
                    ("Regular", "regular"),
                    ("Relaxed", "relaxed"),
                    ("Oversized", "oversized"),
                    ("Not sure", "not_sure"),
                ],
            ),
            _choice_step(
                sid="step-material-pref",
                q="Any material preference?",
                options=[
                    ("Lightweight", "lightweight"),
                    ("Warm/cozy", "warm_cozy"),
                    ("Structured", "structured"),
                    ("Soft/drapey", "soft_drapey"),
                    ("Not sure", "not_sure"),
                ],
                required=False,
            ),
        ]
    else:
        # scene / scene_placement
        ctx = _generic_context(use_case=use_case, goal_intent="pricing")
        ctx["attribute_families"] = [
            {"family": "project_type", "goal": "Type of work or outcome needed (install, replace, repair)."},
            {"family": "area_location", "goal": "Which area/room/location the work applies to."},
            {"family": "area_size", "goal": "Approximate size/scale category (small/medium/large)."},
        ]
        base_steps = [
            _choice_step(
                sid="step-project-type",
                q="What type of work do you need?",
                options=[
                    ("New install", "new_install"),
                    ("Upgrade existing", "upgrade_existing"),
                    ("Repair an issue", "repair_issue"),
                    ("Replace something", "replace"),
                    ("Not sure", "not_sure"),
                    ("Other", "other"),
                ],
            ),
            _choice_step(
                sid="step-area-location",
                q="Where will the work happen?",
                options=[
                    ("Indoors", "indoors"),
                    ("Outdoors", "outdoors"),
                    ("Both", "both"),
                    ("Not sure", "not_sure"),
                ],
            ),
            _choice_step(
                sid="step-area-size",
                q="How big is the space?",
                options=[
                    ("Small", "small"),
                    ("Medium", "medium"),
                    ("Large", "large"),
                    ("Multiple areas", "multiple"),
                    ("Not sure", "not_sure"),
                ],
            ),
        ]

    specs: list[DemoSpec] = []

    # Demo 1: straight batch
    steps1 = base_steps[:max_steps]
    if last:
        # Last batch must end with Upload → Gallery.
        steps1 = (
            steps1[: max(0, max_steps - 2)]
            + [
                _upload_step(sid="step-upload", q="Upload a reference image (optional)."),
                _gallery_step(sid="step-gallery", q="Pick a design direction you like."),
            ]
        )
    specs.append(
        DemoSpec(
            name=f"{use_case}_b{total_batches}_i{batch_index}_basic",
            context=ctx,
            max_steps=max_steps,
            allowed_mini_types=allowed,
            steps=steps1,
        )
    )

    # Demo 2: show option-count variety + already-asked behavior
    ctx2 = dict(ctx)
    ctx2["already_asked_keys"] = ["step-project-type"] if use_case != "tryon" else ["step-item-type"]
    ctx2["asked_step_ids"] = ctx2["already_asked_keys"]
    steps2 = [s for s in base_steps if s["id"] not in set(ctx2["already_asked_keys"])]
    # Make one step have more options
    if steps2:
        s0 = dict(steps2[0])
        if s0.get("type") == "multiple_choice":
            s0["options"] = list(s0.get("options") or []) + [{"label": "Prefer not to say", "value": "prefer_not"}]
        steps2[0] = s0
    steps2 = steps2[:max_steps]
    if last:
        steps2 = (
            steps2[: max(0, max_steps - 2)]
            + [
                _upload_step(sid="step-upload", q="Upload a reference image (optional)."),
                _gallery_step(sid="step-gallery", q="Pick a design direction you like."),
            ]
        )
    specs.append(
        DemoSpec(
            name=f"{use_case}_b{total_batches}_i{batch_index}_skip_asked",
            context=ctx2,
            max_steps=max_steps,
            allowed_mini_types=allowed,
            steps=steps2,
        )
    )

    # Demo 3: include a yes/no in non-first batches if allowed
    if "yes_no" in allowed and not last:
        steps3 = base_steps[:2] + [_yesno_step(sid="step-urgent", q="Is this urgent?")]
        specs.append(
            DemoSpec(
                name=f"{use_case}_b{total_batches}_i{batch_index}_yesno",
                context=ctx,
                max_steps=max_steps,
                allowed_mini_types=allowed,
                steps=steps3[:max_steps],
            )
        )

    return specs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(_repo_root() / "src" / "programs" / "batch_generator" / "examples"),
        help="Output directory for demo packs",
    )
    parser.add_argument(
        "--total-batches",
        type=int,
        default=0,
        help="Override total batches; 0 uses DEFAULT_CONSTRAINTS['maxBatches']",
    )
    args = parser.parse_args()

    # Constraints
    total_batches = int(args.total_batches or 0)
    if total_batches <= 0:
        from programs.batch_generator.planning.form_planning.static_constraints import DEFAULT_CONSTRAINTS

        total_batches = int((DEFAULT_CONSTRAINTS or {}).get("maxBatches") or 3)
    total_batches = max(1, min(10, total_batches))

    out_dir = Path(args.out)
    use_cases = ["scene", "scene_placement", "tryon"]

    all_written: list[Path] = []
    for use_case in use_cases:
        for batch_index in range(total_batches):
            specs = _demo_specs_for(use_case=use_case, total_batches=total_batches, batch_index=batch_index)
            rows: list[dict] = []
            for spec in specs:
                _assert_step_ids_unique(spec.steps)
                _validate_steps(spec.steps)
                rows.append(
                    {
                        "meta": {"name": spec.name, "version": 1},
                        "inputs": {
                            "context_json": _dump_compact(spec.context),
                            "batch_id": f"batch-{batch_index + 1}",
                            "max_steps": int(spec.max_steps),
                            "allowed_mini_types": list(spec.allowed_mini_types),
                        },
                        "outputs": {"mini_steps_jsonl": _step_jsonl(spec.steps)},
                    }
                )

            pack_path = out_dir / use_case / f"b{total_batches}" / f"batch_{batch_index}.jsonl"
            _write_jsonl(pack_path, rows)
            all_written.append(pack_path)

    print(f"Wrote {len(all_written)} demo packs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

