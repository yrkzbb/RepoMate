import pytest

from repopilot.config import RepoPilotConfig


def test_from_env_requires_api_key() -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        RepoPilotConfig.from_env({})


def test_from_env_uses_defaults() -> None:
    config = RepoPilotConfig.from_env({"OPENAI_API_KEY": "test-key"})

    assert config.api_key == "test-key"
    assert config.model == "gpt-5-mini"
    assert config.base_url is None


def test_from_env_accepts_model_override() -> None:
    config = RepoPilotConfig.from_env(
        {"OPENAI_API_KEY": "test-key", "REPOPILOT_MODEL": "custom-model"}
    )

    assert config.model == "custom-model"


def test_explicit_model_overrides_blank_environment_model() -> None:
    config = RepoPilotConfig.from_env(
        {"OPENAI_API_KEY": "test-key", "REPOPILOT_MODEL": "   "},
        model_override="override-model",
    )

    assert config.model == "override-model"


def test_from_env_accepts_compatible_provider() -> None:
    config = RepoPilotConfig.from_env(
        {
            "OPENAI_API_KEY": "test-key",
            "REPOPILOT_MODEL": "provider-model",
            "REPOPILOT_BASE_URL": " https://api.example.com/v1 ",
        }
    )

    assert config.model == "provider-model"
    assert config.base_url == "https://api.example.com/v1"


def test_compatible_provider_requires_explicit_model() -> None:
    with pytest.raises(
        ValueError,
        match="REPOPILOT_MODEL is required when REPOPILOT_BASE_URL is set",
    ):
        RepoPilotConfig.from_env(
            {
                "OPENAI_API_KEY": "test-key",
                "REPOPILOT_BASE_URL": "https://api.example.com/v1",
            }
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "not-a-url",
        "https://",
        "ftp://api.example.com/v1",
        "file:///tmp/api",
        "https://api.example.com:bad/v1",
        "https://api.example.com:99999/v1",
        "https://[::1/v1",
    ],
)
def test_compatible_provider_rejects_invalid_base_url(base_url: str) -> None:
    with pytest.raises(
        ValueError,
        match="REPOPILOT_BASE_URL must be an absolute http or https URL with a host",
    ):
        RepoPilotConfig.from_env(
            {
                "OPENAI_API_KEY": "test-key",
                "REPOPILOT_MODEL": "provider-model",
                "REPOPILOT_BASE_URL": base_url,
            }
        )
