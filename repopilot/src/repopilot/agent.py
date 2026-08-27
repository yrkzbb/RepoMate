"""The minimal RepoPilot agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from openai import AsyncOpenAI

from agents import Agent, OpenAIChatCompletionsModel, OpenAIResponsesModel, RunConfig, Runner

INSTRUCTIONS = """You are RepoPilot, a concise code-repository assistant.
In this first milestone you have no repository tools. State that limitation instead
of pretending that you inspected files. Answer general software questions clearly.
"""


@dataclass(frozen=True)
class RunSummary:
    answer: str
    elapsed_seconds: float
    requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int


def build_agent(model: str, api_key: str, base_url: str | None = None) -> Agent:
    client = (
        AsyncOpenAI(api_key=api_key, base_url=base_url)
        if base_url
        else AsyncOpenAI(api_key=api_key)
    )
    sdk_model = (
        OpenAIChatCompletionsModel(model=model, openai_client=client)
        if base_url
        else OpenAIResponsesModel(model=model, openai_client=client)
    )
    return Agent(name="RepoPilot", instructions=INSTRUCTIONS, model=sdk_model)


async def answer_question(
    question: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
) -> RunSummary:
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    started = perf_counter()
    result = await Runner.run(
        build_agent(model, api_key, base_url),
        question,
        run_config=RunConfig(tracing_disabled=base_url is not None),
    )
    elapsed_seconds = perf_counter() - started
    usage = result.context_wrapper.usage

    return RunSummary(
        answer=str(result.final_output),
        elapsed_seconds=elapsed_seconds,
        requests=usage.requests,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )
