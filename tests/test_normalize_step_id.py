from app.pipeline.form_pipeline import _normalize_step_id


def test_normalize_step_id_underscore_to_hyphen():
    assert _normalize_step_id("step_project_goal") == "step-project-goal"
    assert _normalize_step_id("step-project-goal") == "step-project-goal"


def test_normalize_step_id_empty():
    assert _normalize_step_id("") == ""
    assert _normalize_step_id("   ") == ""
