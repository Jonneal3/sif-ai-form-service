def test_backend_overrides_client_max_batches(monkeypatch):
    monkeypatch.setenv("AI_FORM_MAX_BATCH_CALLS", "2")

    from app.pipeline.form_pipeline import _build_context

    payload = {
        "useCase": "scene",
        "industry": "Landscaping",
        "service": "Pool Design",
        "formState": {"batchIndex": 0, "maxBatches": 99},
        "batchState": {},
    }
    ctx = _build_context(payload)
    assert ctx["batch_policy"]["maxCalls"] == 2
    assert ctx["batch_info"]["max_batches"] == 2

    from app.form_planning.form_plan import build_shared_form_plan

    plan = build_shared_form_plan(context=ctx, batch_policy=None, form_plan_items=None)
    assert plan["constraints"]["maxBatches"] == 2

