from app.form_planning.policy import parse_batch_policy_json, parse_psychology_plan_json


def test_parse_batch_policy_json_accepts_valid_json():
    policy = parse_batch_policy_json('{"v":1,"maxCalls":2,"maxStepsPerCall":{"ContextCore":5},"allowedMiniTypes":{"ContextCore":["choice"]}}')
    assert policy is not None
    assert policy["v"] == 1
    assert policy["maxCalls"] == 2


def test_parse_psychology_plan_json_accepts_valid_json():
    plan = parse_psychology_plan_json('{"v":1,"approach":"escalation_ladder","stages":[{"id":"scope","goal":"x","rules":["y"]}]}')
    assert plan is not None
    assert plan["approach"] == "escalation_ladder"
