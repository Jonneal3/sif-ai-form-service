from eval.metrics import compute_metrics


def test_metrics_fail_when_plan_has_unasked_items_and_zero_steps():
    payload = {
        "formPlan": [{"key": "project_goal"}, {"key": "style_direction"}],
        "alreadyAskedKeys": [],
        "allowedMiniTypes": ["multiple_choice"],
        "maxSteps": 2,
    }
    result = {"ok": True, "miniSteps": []}
    m = compute_metrics(payload, result)
    assert m.has_min_step_when_needed is False
    assert "needed_min_one_step_but_zero" in m.errors


def test_metrics_pass_when_plan_has_unasked_items_and_one_step():
    payload = {
        "formPlan": [{"key": "project_goal"}, {"key": "style_direction"}],
        "alreadyAskedKeys": [],
        "allowedMiniTypes": ["multiple_choice"],
        "maxSteps": 2,
    }
    result = {"ok": True, "miniSteps": [{"id": "step-project-goal", "type": "multiple_choice", "question": "x", "options": []}]}
    m = compute_metrics(payload, result)
    assert m.has_min_step_when_needed is True


