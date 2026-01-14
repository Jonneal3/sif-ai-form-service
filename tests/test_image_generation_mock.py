from modules.image_generation import generate_images


def test_mock_image_provider_returns_requested_number_of_images():
    result = generate_images(prompt="hello world", num_variants=3, provider="mock")
    assert len(result.images) == 3
    assert all(isinstance(u, str) and u.startswith("data:image/svg+xml;utf8,") for u in result.images)
    assert result.metrics["provider"] == "mock"
    assert isinstance(result.metrics["generationTimeMs"], int)

