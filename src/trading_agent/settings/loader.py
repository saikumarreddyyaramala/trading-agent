"""Load configuration: config/base.yaml deep-merged with config/<environment>.yaml."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from trading_agent.settings.models import AppConfig, Environment
from trading_agent.settings.secrets import Secrets


class ConfigError(RuntimeError):
    """Configuration is missing, malformed, or unsafe."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file must contain a mapping at the top level: {path}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict where nested mappings in `override` are merged into `base`."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def resolve_environment(explicit: str | None = None) -> Environment:
    raw = (explicit or os.environ.get("APP_ENV") or Environment.PAPER.value).strip().lower()
    try:
        return Environment(raw)
    except ValueError as exc:
        allowed = ", ".join(e.value for e in Environment)
        raise ConfigError(f"unknown environment '{raw}' (expected one of: {allowed})") from exc


def resolve_config_dir(explicit: Path | None = None) -> Path:
    return explicit or Path(os.environ.get("CONFIG_DIR", "config"))


def load_config(environment: str | None = None, config_dir: Path | None = None) -> AppConfig:
    env = resolve_environment(environment)
    directory = resolve_config_dir(config_dir)
    merged = deep_merge(_read_yaml(directory / "base.yaml"), _read_yaml(directory / f"{env}.yaml"))
    merged["environment"] = env.value
    try:
        return AppConfig.model_validate(merged)
    except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
        raise ConfigError(f"invalid configuration for '{env}':\n{exc}") from exc


def assert_live_trading_allowed(config: AppConfig, secrets: Secrets) -> None:
    """Live trading needs two independent switches: the live config AND an explicit env flag."""
    if config.is_live and not secrets.live_trading_confirmed:
        raise ConfigError(
            "live environment selected but LIVE_TRADING_CONFIRMED is not true; refusing to start"
        )
