"""Secrets, read only from the process environment (or a local .env file in development).

Secrets are kept apart from configuration on purpose: config is versioned in git and
reviewed; secrets are injected at runtime by a secrets manager and never committed.
`SecretStr` keeps values out of logs, reprs, and tracebacks.
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Which secret fields each integration needs. Used by the composition root and health checks.
BROKER_SECRET_FIELDS: dict[str, tuple[str, ...]] = {
    "alpaca": ("alpaca_api_key", "alpaca_secret_key"),
}
MARKET_DATA_SECRET_FIELDS: dict[str, tuple[str, ...]] = {
    "alpaca": ("alpaca_api_key", "alpaca_secret_key"),
}
LLM_SECRET_FIELDS: dict[str, tuple[str, ...]] = {
    "openai": ("openai_api_key",),
    "anthropic": ("anthropic_api_key",),
}


class MissingSecretError(RuntimeError):
    """A required secret is not present in the environment."""


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    alpaca_api_key: SecretStr | None = None
    alpaca_secret_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    # Audit database, e.g. postgresql://user:pass@host:5432/db or sqlite:///data/audit.db
    database_url: SecretStr | None = None
    live_trading_confirmed: bool = False

    def require(self, name: str) -> str:
        """Return a secret's plain value, or raise if it is not set."""
        value = getattr(self, name)
        if not isinstance(value, SecretStr) or not value.get_secret_value():
            raise MissingSecretError(f"{name.upper()} is not set")
        return value.get_secret_value()

    def missing(self, *names: str) -> list[str]:
        """Return the environment variable names that are required but empty."""
        absent: list[str] = []
        for name in names:
            value = getattr(self, name)
            if value is None or (isinstance(value, SecretStr) and not value.get_secret_value()):
                absent.append(name.upper())
        return absent
