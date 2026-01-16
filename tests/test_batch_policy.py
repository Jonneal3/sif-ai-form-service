import os

from api.supabase_client import _default_allowed_mini_types, _default_max_steps, _infer_phase
from app.form_planning.policy import normalize_policy, policy_for_batch, policy_for_phase


def test_infer_phase_uses_plan_presence_when_batch_id_unknown():
    assert _infer_phase("Anything", []) == "ContextCore"
    assert _infer_phase("Anything", [{"key": "x"}]) == "PersonalGuide"


def test_default_allowed_mini_types_driven_by_items_and_phase(monkeypatch):
    monkeypatch.delenv("AI_FORM_ALLOWED_MINI_TYPES_CONTEXTCORE", raising=False)
    monkeypatch.delenv("AI_FORM_ALLOWED_MINI_TYPES_PERSONALGUIDE", raising=False)

    items = [{"component_hint": "text"}, {"component_hint": "slider"}]
    assert set(_default_allowed_mini_types("ContextCore", items)) == {"choice", "text_input", "slider"}
    assert set(_default_allowed_mini_types("PersonalGuide", [])) == {"choice", "slider"}


def test_allowed_mini_types_env_override(monkeypatch):
    monkeypatch.setenv("AI_FORM_ALLOWED_MINI_TYPES_CONTEXTCORE", "choice,text_input")
    assert _default_allowed_mini_types("ContextCore", [{"component_hint": "slider"}]) == ["choice", "text_input"]


def test_default_max_steps_caps_to_remaining_items(monkeypatch):
    monkeypatch.delenv("AI_FORM_CONTEXTCORE_MAX_STEPS", raising=False)
    monkeypatch.delenv("AI_FORM_PERSONALGUIDE_MAX_STEPS", raising=False)
    assert _default_max_steps("ContextCore", [{"x": 1}] * 3) == 3
    assert _default_max_steps("PersonalGuide", [{"x": 1}] * 2) == 2


def test_default_max_steps_env_override(monkeypatch):
    monkeypatch.setenv("AI_FORM_PERSONALGUIDE_MAX_STEPS", "4")
    assert _default_max_steps("PersonalGuide", [{"x": 1}] * 10) == 4


def test_policy_for_phase_reads_allowed_and_max_steps():
    policy = normalize_policy(
        {
            "v": 1,
            "maxCalls": 3,
            "maxStepsPerCall": {"ContextCore": 2, "PersonalGuide": 4},
            "allowedMiniTypes": {"ContextCore": ["choice"], "PersonalGuide": ["choice", "slider"]},
        }
    )
    assert policy is not None
    assert policy_for_phase(policy, "ContextCore")["maxSteps"] == 2
    assert policy_for_phase(policy, "PersonalGuide")["allowedMiniTypes"] == ["choice", "slider"]


def test_policy_for_batch_uses_phases_when_present():
    policy = normalize_policy(
        {
            "v": 1,
            "maxCalls": 3,
            "phases": [
                {"id": "One", "maxSteps": 2, "allowedMiniTypes": ["choice"]},
                {"id": "Two", "maxSteps": 3, "allowedMiniTypes": ["choice", "slider"]},
                {"id": "Three", "maxSteps": 4, "allowedMiniTypes": ["choice", "text_input"]},
            ],
        }
    )
    assert policy is not None
    assert policy_for_batch(policy, 1)["phaseId"] == "One"
    assert policy_for_batch(policy, 2)["maxSteps"] == 3
    assert policy_for_batch(policy, 99)["phaseId"] == "Three"


def test_policy_phase_metadata_roundtrips():
    policy = normalize_policy(
        {
            "v": 1,
            "maxCalls": 2,
            "phases": [
                {"id": "A", "purpose": "p", "maxSteps": 2, "allowedMiniTypes": ["choice"], "rigidity": 0.8, "focusKeys": ["k1", "k2"]},
                {"id": "B", "purpose": "q", "maxSteps": 3, "allowedMiniTypes": ["choice", "slider"], "rigidity": 0.2, "focusKeys": []},
            ],
        }
    )
    assert policy is not None
    a = policy_for_batch(policy, 1)
    assert a["purpose"] == "p"
    assert a["rigidity"] == 0.8
    assert a["focusKeys"] == ["k1", "k2"]
