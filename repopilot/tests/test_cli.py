from types import SimpleNamespace

import pytest

import repopilot.cli as cli_module


@pytest.mark.asyncio
async def test_run_prints_answer_and_metrics(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("REPOPILOT_MODEL", "environment-model")
    monkeypatch.setenv("REPOPILOT_BASE_URL", "https://api.example.com/v1")

    async def fake_answer(question: str, model: str, api_key: str, base_url: str | None) -> object:
        assert (question, model, api_key, base_url) == (
            "question",
            "override-model",
            "test-key",
            "https://api.example.com/v1",
        )
        return SimpleNamespace(
            answer="answer",
            elapsed_seconds=0.125,
            requests=1,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )

    monkeypatch.setattr(cli_module, "answer_question", fake_answer)

    exit_code = await cli_module._run("question", "override-model")

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "answer" in captured.out
    assert "model: override-model" in captured.out
    assert "elapsed_seconds: 0.125" in captured.out
    assert "total_tokens: 15" in captured.out


@pytest.mark.asyncio
async def test_run_reports_configuration_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code = await cli_module._run("question", None)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "OPENAI_API_KEY is required" in captured.err


@pytest.mark.asyncio
async def test_run_reports_blank_question_as_input_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    exit_code = await cli_module._run("   ", None)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err == "Input error: question must not be empty\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_base_url",
    ["file:///tmp/api", "https://api.example.com:bad/v1"],
)
async def test_run_rejects_invalid_base_url_before_agent_call(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    invalid_base_url: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("REPOPILOT_MODEL", "provider-model")
    monkeypatch.setenv("REPOPILOT_BASE_URL", invalid_base_url)
    called = False

    async def fake_answer(question: str, model: str, api_key: str, base_url: str | None) -> object:
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(cli_module, "answer_question", fake_answer)

    exit_code = await cli_module._run("question", None)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert not called
    assert "REPOPILOT_BASE_URL must be an absolute http or https URL" in captured.err


@pytest.mark.asyncio
async def test_run_reports_agent_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def failing_answer(
        question: str, model: str, api_key: str, base_url: str | None
    ) -> object:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(cli_module, "answer_question", failing_answer)

    exit_code = await cli_module._run("question", None)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Agent run failed: provider unavailable" in captured.err
