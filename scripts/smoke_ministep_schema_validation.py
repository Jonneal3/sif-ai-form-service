#!/usr/bin/env python3
from __future__ import annotations

"""
Smoke checks for min-step schema validation (`_validate_mini`).

Run:
  PYTHONPATH=.:src python3 scripts/smoke_ministep_schema_validation.py
"""


def _ui_types() -> dict:
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
        SliderUI,
        SearchableSelectUI,
        TextInputUI,
    )

    return {
        "BudgetCardsUI": BudgetCardsUI,
        "ColorPickerUI": ColorPickerUI,
        "CompositeUI": CompositeUI,
        "ConfirmationUI": ConfirmationUI,
        "DatePickerUI": DatePickerUI,
        "DesignerUI": DesignerUI,
        "FileUploadUI": FileUploadUI,
        "GalleryUI": GalleryUI,
        "IntroUI": IntroUI,
        "LeadCaptureUI": LeadCaptureUI,
        "MultipleChoiceUI": MultipleChoiceUI,
        "PricingUI": PricingUI,
        "RatingUI": RatingUI,
        "SliderUI": SliderUI,
        "SearchableSelectUI": SearchableSelectUI,
        "TextInputUI": TextInputUI,
    }


def main() -> int:
    from programs.question_planner.renderer.validation import _validate_mini

    ui_types = _ui_types()

    # Choice: requires question + options.
    assert _validate_mini({"type": "multiple_choice", "options": [{"label": "A", "value": "a"}]}, ui_types) is None
    ok_choice = _validate_mini(
        {
            "type": "multiple_choice",
            "question": "Pick one",
            "options": [{"label": "A", "value": "a"}],
            "multi_select": True,
            "allow_other": True,
            "other_label": "Other",
        },
        ui_types,
    )
    assert isinstance(ok_choice, dict) and ok_choice.get("id") and ok_choice.get("question")

    # Rating: requires scale_min + scale_max.
    assert _validate_mini({"type": "rating", "question": "Rate it"}, ui_types) is None
    ok_rating = _validate_mini({"type": "rating", "question": "Rate it", "scale_min": 1, "scale_max": 5}, ui_types)
    assert isinstance(ok_rating, dict) and ok_rating.get("id") and ok_rating.get("question")

    # Budget cards: requires ranges.
    assert _validate_mini({"type": "budget_cards", "question": "Budget?"}, ui_types) is None
    ok_budget = _validate_mini(
        {"type": "budget_cards", "question": "Budget?", "ranges": [{"min": 0, "max": 1000, "label": "$0-$1k"}]},
        ui_types,
    )
    assert isinstance(ok_budget, dict) and ok_budget.get("id") and ok_budget.get("question")

    print("OK: min-step schema validation smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
