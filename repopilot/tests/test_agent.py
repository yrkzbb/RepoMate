from types import SimpleNamespace

import pytest

import repopilot.agent as agent_module
from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel, RunConfig, Runner
from repopilot.agent import answer_question, build_agent


def test_build_agent_uses_responses_for_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, *, api_key: str) -> None:
            captured["api_key"] = api_key

    monkeypatch.setattr(agent_module, "AsyncOpenAI", FakeClient)
    agent = build_agent("test-model", "test-key")

    assert agent.name == "RepoPilot"
    assert captured["api_key"] == "test-key"
    assert isinstance(agent.model, OpenAIResponsesModel)
    assert agent.model.model == "test-model"


def test_build_agent_uses_chat_completions_for_compatible_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr(agent_module, "AsyncOpenAI", FakeClient)
    agent = build_agent("provider-model", "provider-key", "https://api.example.com/v1")

    assert captured == {
        "api_key": "provider-key",
        "base_url": "https://api.example.com/v1",
    }
    assert isinstance(agent.model, OpenAIChatCompletionsModel)
    assert agent.model.model == "provider-model"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("base_url", "tracing_disabled"),
    [(None, False), ("https://api.example.com/v1", True)],
)
async def test_answer_question_returns_output_and_usage(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str | None,
    tracing_disabled: bool,
) -> None:
    usage = SimpleNamespace(
        requests=1,
        input_tokens=12,
        output_tokens=8,
        total_tokens=20,
    )
    fake_result = SimpleNamespace(
        final_output="A grounded answer",
        context_wrapper=SimpleNamespace(usage=usage),
    )

    async def fake_run(agent: object, question: str, *, run_config: RunConfig) -> object:
        assert question == "Where is the runner?"
        assert run_config.tracing_disabled is tracing_disabled
        return fake_result

    monkeypatch.setattr(agent_module, "build_agent", lambda model, api_key, base_url: object())
    monkeypatch.setattr(Runner, "run", fake_run)
    times = iter((10.0, 10.25))
    monkeypatch.setattr(agent_module, "perf_counter", lambda: next(times))

    summary = await answer_question(
        "Where is the runner?",
        "test-model",
        "test-key",
        base_url,
    )

    assert summary.answer == "A grounded answer"
    assert summary.elapsed_seconds == 0.25
    assert summary.requests == 1
    assert summary.input_tokens == 12
    assert summary.output_tokens == 8
    assert summary.total_tokens == 20


@pytest.mark.asyncio
async def test_answer_question_rejects_blank_input() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        await answer_question("   ", "test-model", "test-key")
