"""Runtime configuration for RepoPilot."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class RepoPilotConfig:
    """Validated configuration read from environment variables."""

    api_key: str
    model: str = "gpt-5-mini"
    base_url: str | None = None

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        model_override: str | None = None,
    ) -> RepoPilotConfig:
        values = os.environ if environ is None else environ
        api_key = values.get("OPENAI_API_KEY", "").strip()
        base_url = values.get("REPOPILOT_BASE_URL", "").strip() or None
        configured_model = values.get("REPOPILOT_MODEL")
        raw_model = model_override if model_override is not None else configured_model
        model = raw_model.strip() if raw_model is not None else ""

        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if base_url is not None:
            try:
                parsed_base_url = urlsplit(base_url)
                hostname = parsed_base_url.hostname
                scheme = parsed_base_url.scheme
                _port = parsed_base_url.port
            except ValueError:
                hostname = None
                scheme = ""
            if scheme not in {"http", "https"} or not hostname:
                raise ValueError(
                    "REPOPILOT_BASE_URL must be an absolute http or https URL with a host"
                )
        if base_url is not None and not model:
            raise ValueError("REPOPILOT_MODEL is required when REPOPILOT_BASE_URL is set")
        if not model:
            model = "gpt-5-mini"

        return cls(api_key=api_key, model=model, base_url=base_url)
