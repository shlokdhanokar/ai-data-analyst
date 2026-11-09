"""Application settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # LLM: any OpenAI-compatible endpoint works (Groq by default, OpenAI, Ollama, vLLM, ...)
    api_key: str
    base_url: str
    model: str
    temperature: float
    request_timeout: float

    # Agent behaviour
    max_tool_steps: int
    history_turns: int
    max_rows_to_llm: int
    max_tool_output_chars: int

    # Uploads
    max_upload_mb: int
    max_files: int

    # Ops
    cache_size: int
    log_level: str
    log_dir: str

    @property
    def llm_configured(self) -> bool:
        return bool(self.api_key)


def get_settings() -> Settings:
    return Settings(
        api_key=_env("LLM_API_KEY", _env("GROQ_API_KEY", "")),
        base_url=_env("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        model=_env("LLM_MODEL", "openai/gpt-oss-120b"),
        temperature=float(_env("LLM_TEMPERATURE", "0.1")),
        request_timeout=float(_env("LLM_TIMEOUT_SECONDS", "60")),
        max_tool_steps=int(_env("MAX_TOOL_STEPS", "6")),
        history_turns=int(_env("HISTORY_TURNS", "6")),
        max_rows_to_llm=int(_env("MAX_ROWS_TO_LLM", "30")),
        max_tool_output_chars=int(_env("MAX_TOOL_OUTPUT_CHARS", "6000")),
        max_upload_mb=int(_env("MAX_UPLOAD_MB", "50")),
        max_files=int(_env("MAX_FILES", "10")),
        cache_size=int(_env("CACHE_SIZE", "128")),
        log_level=_env("LOG_LEVEL", "INFO"),
        log_dir=_env("LOG_DIR", "logs"),
    )
