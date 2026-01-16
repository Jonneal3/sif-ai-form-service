from app.form_planning.composite import wrap_last_step_with_upload_composite


def test_wrap_last_step_with_upload_composite_wraps_on_first_batch():
    payload = {"batchId": "ContextCore", "currentBatch": {"batchNumber": 1}}
    emitted = [
        {"id": "step-project-type", "type": "multiple_choice", "question": "What type?"},
        {"id": "step-area-location", "type": "multiple_choice", "question": "Where?"},
    ]
    required_uploads = [{"stepId": "step-upload-scene", "role": "sceneImage"}]

    out, did = wrap_last_step_with_upload_composite(
        payload=payload,
        emitted_steps=emitted,
        required_uploads=required_uploads,
    )
    assert did is True
    assert len(out) == 2
    assert out[-1]["type"] == "composite"
    assert out[-1]["blocks"][1]["id"] == "step-area-location"
    assert out[-1]["blocks"][1]["step"]["id"] == "step-area-location"
    assert any(b.get("id") == "step-upload-scene" for b in out[-1]["blocks"])


def test_wrap_last_step_with_upload_composite_noop_when_not_first_batch():
    payload = {"batchId": "PersonalGuide", "currentBatch": {"batchNumber": 2}}
    emitted = [{"id": "step-x", "type": "multiple_choice", "question": "Q"}]
    out, did = wrap_last_step_with_upload_composite(
        payload=payload,
        emitted_steps=emitted,
        required_uploads=[{"stepId": "step-upload-scene", "role": "sceneImage"}],
    )
    assert did is False
    assert out == emitted
