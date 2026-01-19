from app.pipeline.pipeline import build_image_prompt


def test_image_prompt_requires_dspy(monkeypatch):
    monkeypatch.delenv("DSPY_PROVIDER", raising=False)
    monkeypatch.delenv("DSPY_MODEL", raising=False)
    monkeypatch.delenv("DSPY_MODEL_LOCK", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

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
    assert result["ok"] is False
    assert "not configured" in str(result.get("error", "")).lower()
