from app.pipeline.pipeline import build_image_prompt


def test_image_prompt_fallback_includes_personalization_and_some_answers():
    payload = {
        "batchId": "ContextCore",
        "platformGoal": "AI pre-design intake",
        "businessContext": "We generate AI images for early design concepts",
        "industry": "Interior Design",
        "service": "Kitchen Remodel",
        "useCase": "scene",
        "requiredUploads": [],
        "personalizationSummary": "User wants a bright, warm, modern look with natural materials.",
        "stepDataSoFar": {"step-budget": "5000", "step-space-type": "kitchen"},
        "alreadyAskedKeys": [],
        "formPlan": [],
        "batchState": {"satietySoFar": 0.7, "satietyRemaining": 0.1},
        "maxSteps": 5,
        "allowedMiniTypes": ["multiple_choice"],
    }
    result = build_image_prompt(payload)
    assert result["ok"] is True
    prompt = result["prompt"]["prompt"]
    assert "Kitchen Remodel" in prompt or "Interior Design" in prompt
    assert "User wants a bright, warm, modern look" in prompt
    assert "step-budget" in prompt or "Known preferences" in prompt
