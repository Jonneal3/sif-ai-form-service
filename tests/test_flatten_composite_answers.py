from api.supabase_client import _flatten_composite_answers


def test_flatten_composite_answers_lifts_nested_step_ids_and_marks_asked():
    answers = {
        "step-composite-foo": {
            "step-project-type": "new_install",
            "step-upload-scene": {"file": "x"},
            "md-1": None,
        }
    }
    asked = []
    out_answers, out_asked = _flatten_composite_answers(answers, asked)
    assert out_answers["step-project-type"] == "new_install"
    assert out_answers["step-upload-scene"] == {"file": "x"}
    assert "step-project-type" in out_asked
    assert "step-upload-scene" in out_asked
    assert "step-composite-foo" in out_asked

