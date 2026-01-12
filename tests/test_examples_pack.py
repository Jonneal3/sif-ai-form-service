import json

from examples.registry import load_jsonl_records


def test_next_steps_examples_pack_loads_and_has_expected_keys():
    records = load_jsonl_records("next_steps_examples.jsonl")
    assert len(records) >= 1

    for r in records:
        assert isinstance(r.inputs, dict)
        assert isinstance(r.outputs, dict)
        # Signature-aligned fields we rely on for demos
        for k in [
            "context_json",
            "batch_id",
            "max_steps",
            "allowed_mini_types",
        ]:
            assert k in r.inputs
        assert "mini_steps_jsonl" in r.outputs

        # Ensure JSON fields are parseable strings (best-effort).
        assert isinstance(r.inputs["context_json"], str)
        assert isinstance(json.loads(r.inputs["context_json"]), dict)
        assert isinstance(r.inputs["allowed_mini_types"], list)
        assert isinstance(r.outputs["mini_steps_jsonl"], str)

        # `mini_steps_jsonl` should be JSON objects per line.
        for line in r.outputs["mini_steps_jsonl"].splitlines():
            line = line.strip()
            assert line
            assert isinstance(json.loads(line), dict)
