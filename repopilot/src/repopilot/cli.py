"""Command-line interface for RepoPilot."""

from __future__ import annotations

import argparse
import asyncio
import sys

from .agent import answer_question
from .config import RepoPilotConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask the RepoPilot agent a question.")
    parser.add_argument("question", help="Question to send to RepoPilot")
    parser.add_argument("--model", help="Override REPOPILOT_MODEL")
    return parser


async def _run(question: str, model_override: str | None) -> int:
    if not question.strip():
        print("Input error: question must not be empty", file=sys.stderr)
        return 2

    try:
        config = RepoPilotConfig.from_env(model_override=model_override)
        summary = await answer_question(
            question,
            config.model,
            config.api_key,
            config.base_url,
        )
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Agent run failed: {exc}", file=sys.stderr)
        return 1

    print(summary.answer)
    print(
        "\n--- run metrics ---\n"
        f"model: {config.model}\n"
        f"elapsed_seconds: {summary.elapsed_seconds:.3f}\n"
        f"requests: {summary.requests}\n"
        f"input_tokens: {summary.input_tokens}\n"
        f"output_tokens: {summary.output_tokens}\n"
        f"total_tokens: {summary.total_tokens}"
    )
    return 0


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(asyncio.run(_run(args.question, args.model)))


if __name__ == "__main__":
    main()
