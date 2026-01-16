from app.form_planning.ui_plan import build_ui_plan


def test_ui_plan_places_upload_after_last_emitted_step_in_first_batch():
    payload = {"batchId": "ContextCore", "currentBatch": {"batchNumber": 1}}
    final_form_plan = [{"key": "project_type", "goal": "x", "why": "y", "component_hint": "choice", "priority": "critical", "importance_weight": 0.2, "expected_metric_gain": 0.1}]
    emitted = [{"id": "step-project-type", "type": "multiple_choice", "question": "q"}]
    required_uploads = [{"stepId": "step-upload-scene", "role": "sceneImage"}]

    ui_plan = build_ui_plan(
        payload=payload,
        final_form_plan=final_form_plan,
        emitted_mini_steps=emitted,
        required_uploads=required_uploads,
    )
    assert ui_plan is not None
    assert ui_plan["v"] == 1
    assert ui_plan["placements"][0]["id"] == "step-upload-scene"
    assert ui_plan["placements"][0]["position"] == "after"
    assert ui_plan["placements"][0]["anchor_step_id"] == "step-project-type"


def test_ui_plan_places_upload_at_start_when_no_emitted_steps():
    payload = {"batchId": "ContextCore", "currentBatch": {"batchNumber": 1}}
    final_form_plan = [{"key": "project_type", "goal": "x", "why": "y", "component_hint": "choice", "priority": "critical", "importance_weight": 0.2, "expected_metric_gain": 0.1}]
    required_uploads = [{"stepId": "step-upload-subject", "role": "userImage"}]

    ui_plan = build_ui_plan(
        payload=payload,
        final_form_plan=final_form_plan,
        emitted_mini_steps=[],
        required_uploads=required_uploads,
    )
    assert ui_plan is not None
    assert ui_plan["placements"][0]["position"] == "start"
    assert ui_plan["placements"][0]["anchor_step_id"] is None


def test_ui_plan_is_none_for_non_first_batch():
    payload = {"batchId": "PersonalGuide", "currentBatch": {"batchNumber": 2}}
    ui_plan = build_ui_plan(
        payload=payload,
        final_form_plan=[{"key": "x"}],
        emitted_mini_steps=[{"id": "step-x"}],
        required_uploads=[{"stepId": "step-upload-scene"}],
    )
    assert ui_plan is None
