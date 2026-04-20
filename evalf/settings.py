from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from evalf.schemas import MetricMode
from evalf.utils import split_csv

_ENV_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_dotenv(path: str | Path) -> None:
    """Load a dotenv-style file without overwriting existing process env vars."""
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return

    for line in dotenv_path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ENV_LINE_RE.match(line)
        if not match:
            continue
        key, value = match.groups()
        os.environ.setdefault(key, _strip_quotes(value))


def load_default_environment() -> None:
    """Load `.env.local` and `.env` with local values taking precedence."""
    load_dotenv(".env.local")
    load_dotenv(".env")


def resolve_api_key(provider: str) -> str | None:
    """Resolve the most appropriate API key env var for the selected provider."""
    provider = provider.lower()
    if provider == "openai":
        return os.getenv("EVALF_API_KEY") or os.getenv("OPENAI_API_KEY")
    if provider == "gemini":
        return os.getenv("EVALF_API_KEY") or os.getenv("GEMINI_API_KEY")
    if provider == "claude":
        return os.getenv("EVALF_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    return os.getenv("EVALF_API_KEY")


class RuntimeSettings(BaseSettings):
    """Runtime configuration sourced from explicit values, env vars, and dotenv files."""

    model_config = SettingsConfigDict(
        env_prefix="EVALF_",
        enable_decoding=False,
        extra="ignore",
        populate_by_name=True,
    )

    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    base_url: str | None = None
    api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("api_key", "EVALF_API_KEY"),
    )
    concurrency: int = 4
    request_timeout_seconds: float = Field(
        default=60.0,
        validation_alias=AliasChoices(
            "request_timeout_seconds",
            "EVALF_REQUEST_TIMEOUT_SECONDS",
            "EVALF_TIMEOUT_SECONDS",
        ),
    )
    per_sample_timeout_seconds: float | None = None
    max_retries: int = Field(default=3, ge=0, le=10)
    temperature: float = 0.0
    max_tokens: int = 800
    output_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("output_path", "EVALF_OUTPUT"),
    )
    threshold: float = 0.7
    metrics: list[str] = Field(default_factory=list)
    metric_mode: MetricMode = "pass@k"
    k: int = Field(default=1, ge=1, le=5)

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return value.lower().strip() if isinstance(value, str) else value

    @field_validator("metrics", mode="before")
    @classmethod
    def parse_metrics(cls, value: str | list[str] | tuple[str, ...] | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return split_csv(value)
        return list(value)

    @model_validator(mode="after")
    def populate_api_key(self) -> RuntimeSettings:
        if self.api_key is None:
            self.api_key = resolve_api_key(self.provider)
        return self

    @property
    def timeout_seconds(self) -> float:
        return self.request_timeout_seconds


def load_runtime_settings() -> RuntimeSettings:
    """Load runtime settings after resolving local dotenv files."""
    load_default_environment()
    return RuntimeSettings()
