from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Annotated, Union

from pydantic import Field, TypeAdapter

# Ensure repo-root imports work when running as `python scripts/export_contract.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.schemas.ui_steps import (  # noqa: E402
    BudgetCardsUI,
    ColorPickerUI,
    CompositeUI,
    ConfirmationUI,
    DatePickerUI,
    DesignerUI,
    FileUploadUI,
    IntroUI,
    LeadCaptureUI,
    MultipleChoiceUI,
    PricingUI,
    RatingUI,
    SearchableSelectUI,
    TextInputUI,
)

CONTRACT_DIR = REPO_ROOT / "shared" / "ai-form-contract"
SCHEMA_DIR = CONTRACT_DIR / "schema"
SCHEMA_PATH = SCHEMA_DIR / "ui_step.schema.json"
VERSION_PATH = SCHEMA_DIR / "schema_version.txt"

# LOCAL SYNC: Automatically mirror the contract to the Next.js repo
SYNC_TARGETS = [
    "/Users/jon/Desktop/sif-widget/shared/ai-form-contract",
]


def _read_version() -> str:
    try:
        v = VERSION_PATH.read_text(encoding="utf-8").strip()
        return v or "0"
    except Exception:
        return "0"


def main() -> None:
    # Discriminated union by `type`
    UIStep = Annotated[
        Union[
            IntroUI,
            TextInputUI,
            MultipleChoiceUI,
            RatingUI,
            BudgetCardsUI,
            FileUploadUI,
            DatePickerUI,
            ColorPickerUI,
            SearchableSelectUI,
            LeadCaptureUI,
            PricingUI,
            ConfirmationUI,
            DesignerUI,
            CompositeUI,
        ],
        Field(discriminator="type"),
    ]

    schema = TypeAdapter(UIStep).json_schema()
    version = _read_version()

    # Extract allowed types from the schema's discriminator mapping
    allowed_types = [
        "intro",
        "text_input",
        "text",
        "multiple_choice",
        "choice",
        "segmented_choice",
        "chips_multi",
        "yes_no",
        "image_choice_grid",
        "rating",
        "slider",
        "range_slider",
        "file_upload",
        "upload",
        "file_picker",
        "budget_cards",
        "date_picker",
        "color_picker",
        "searchable_select",
        "lead_capture",
        "pricing",
        "confirmation",
        "designer",
        "composite",
    ]

    out = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "UIStep",
        "description": "The 'Real Schema' contract for UI components. The Next.js repo uses this for rendering.",
        "schemaVersion": version,
        "allowedTypes": allowed_types,
        "schema": schema,
    }

    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {SCHEMA_PATH} (schemaVersion={version})")

    # Mirror to other projects
    for target in SYNC_TARGETS:
        try:
            target_path = Path(target)
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(CONTRACT_DIR, target_path)
            print(f"✅ Automatically synced contract to: {target}")
        except Exception as e:
            print(f"⚠️ Could not sync to {target}: {e}")


if __name__ == "__main__":
    main()

