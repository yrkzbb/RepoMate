# RepoPilot

RepoPilot is a repository-analysis agent built as a hands-on OpenAI Agents SDK
learning project. Milestone 1 provides a minimal command-line agent and reports
basic run usage. Repository tools arrive in Milestone 2.

## Requirements

- Python 3.10 or newer
- `uv` (recommended) or `pip`
- An OpenAI API key, or a key from an OpenAI-compatible provider

## Setup

From this directory, create the environment and install the local SDK checkout:

```bash
uv sync --frozen --extra dev
```

If `uv` is not installed, the following `pip` setup is a convenient but
non-locked development fallback:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ..
.venv/bin/pip install -e '.[dev]'
```

Set credentials in your shell. Do not commit real keys:

```bash
export OPENAI_API_KEY="your-api-key"
export REPOPILOT_MODEL="gpt-5-mini"
```

For OpenAI-compatible providers, set the provider's API base URL and an exact
model name supported by that provider. RepoPilot uses Chat Completions whenever
`REPOPILOT_BASE_URL` is set because compatibility providers do not necessarily
implement the Responses API. SDK tracing is disabled for these runs so prompts
and outputs are not also exported to OpenAI's tracing service:

```bash
export OPENAI_API_KEY="your-provider-key"
export REPOPILOT_BASE_URL="https://api.gpt.ge/v1"
export REPOPILOT_MODEL="a-model-returned-by-the-provider"
```

Use the API address shown in your provider console; a documentation hostname is
not necessarily the request hostname. Do not send private repository content to
a provider until you have reviewed its data-handling and retention policy.

## Run

```bash
uv run --frozen repopilot "What is an agent loop?"
```

Override the configured model for one run:

```bash
uv run --frozen repopilot --model gpt-5-mini "Explain tool calling in two sentences."
```

## Test

The initial tests do not call the OpenAI API:

```bash
uv run --frozen pytest
```

With the `pip` setup, run:

```bash
.venv/bin/python -m pytest
```

## Milestones

1. Minimal agent, configuration, CLI, and usage reporting.
2. Safe repository file tools.
3. Loop limits, retries, timeouts, and repeated-call detection.
4. Handoffs and agents-as-tools experiments.
5. Sessions, tracing, and human approval.
6. Evaluation suite, documentation, and portfolio demo.
