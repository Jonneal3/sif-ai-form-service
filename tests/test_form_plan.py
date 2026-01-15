import json

from app.form_psychology.form_plan import finalize_form_plan


def test_finalize_form_plan_skips_when_existing_plan_present():
    payload = {"batchId": "ContextCore", "currentBatch": {"batchNumber": 1}}
    context = {"form_plan": [{"key": "existing", "goal": "x", "why": "y", "component_hint": "choice", "priority": "low", "importance_weight": 0.1, "expected_metric_gain": 0.1}]}
    plan, did = finalize_form_plan(payload=payload, context=context, produced_form_plan_json="[]")
    assert plan is None
    assert did is False


def test_finalize_form_plan_skips_when_not_first_batch():
    payload = {"batchId": "PersonalGuide", "currentBatch": {"batchNumber": 2}}
    context = {"form_plan": [], "required_uploads": []}
    plan, did = finalize_form_plan(payload=payload, context=context, produced_form_plan_json="[]")
    assert plan is None
    assert did is False


def test_finalize_form_plan_merges_deterministic_uploads_and_produced_items():
    payload = {"batchId": "ContextCore", "currentBatch": {"batchNumber": 1}}
    context = {
        "form_plan": [],
        "required_uploads": [{"stepId": "step-upload-scene", "role": "sceneImage"}],
        "attribute_families": [],
    }
    produced = json.dumps(
        [
            {
                "key": "project_type",
                "goal": "Define the work type",
                "why": "Determines scope",
                "component_hint": "choice",
                "priority": "critical",
                "importance_weight": 0.2,
                "expected_metric_gain": 0.18,
            }
        ]
    )
    plan, did = finalize_form_plan(payload=payload, context=context, produced_form_plan_json=produced)
    assert did is True
    assert isinstance(plan, list)
    assert plan[0]["key"] == "upload_scene"
    assert plan[0].get("deterministic") is True
    assert plan[0].get("role") == "sceneImage"
    assert any(item.get("key") == "project_type" for item in plan)


def test_finalize_form_plan_falls_back_when_no_produced_plan():
    payload = {"batchId": "ContextCore", "currentBatch": {"batchNumber": 1}}
    context = {
        "form_plan": [],
        "required_uploads": [{"stepId": "step-upload-subject", "role": "userImage"}],
        "attribute_families": [{"family": "project_type", "goal": "Type of work needed."}],
    }
    plan, did = finalize_form_plan(payload=payload, context=context, produced_form_plan_json="")
    assert did is True
    assert isinstance(plan, list)
    assert plan[0]["key"] == "upload_subject"
    assert any(item.get("key") == "project_type" for item in plan)
