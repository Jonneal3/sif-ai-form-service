import pytest

from api.models import ImageRequest


def test_image_request_accepts_scene_placement_and_normalizes():
    req = ImageRequest.model_validate(
        {"instanceId": "abc", "useCase": "scene-placement", "numOutputs": 2, "outputFormat": "url"}
    )
    assert req.use_case == "scene-placement"
    assert req.num_outputs == 2


def test_image_request_accepts_tryon_variants():
    req = ImageRequest.model_validate({"instanceId": "abc", "useCase": "try on"})
    assert req.use_case == "tryon"


def test_image_request_rejects_bad_use_case():
    with pytest.raises(Exception):
        ImageRequest.model_validate({"instanceId": "abc", "useCase": "other"})


def test_image_request_rejects_bad_output_format():
    with pytest.raises(Exception):
        ImageRequest.model_validate({"instanceId": "abc", "useCase": "scene", "outputFormat": "png"})

