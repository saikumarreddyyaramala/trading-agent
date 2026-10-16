"""Command-line entry point: `trading-agent <command>`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from trading_agent import __version__
from trading_agent.bootstrap import Container
from trading_agent.health import CheckStatus, overall_ok, run_healthchecks
from trading_agent.observability.logging import configure_logging
from trading_agent.settings.loader import ConfigError, load_config
from trading_agent.settings.secrets import Secrets

app = typer.Typer(
    help="Enterprise-style AI trading agent. Paper trading by default.",
    no_args_is_help=True,
    add_completion=False,
)

EnvOption = Annotated[
    str | None, typer.Option("--env", help="Environment to load: paper or live (default: APP_ENV).")
]
ConfigDirOption = Annotated[
    Path | None, typer.Option("--config-dir", help="Folder with base.yaml and <env>.yaml.")
]

_COLORS = {
    CheckStatus.PASS: typer.colors.GREEN,
    CheckStatus.WARN: typer.colors.YELLOW,
    CheckStatus.FAIL: typer.colors.RED,
    CheckStatus.SKIP: typer.colors.BRIGHT_BLACK,
}


def _container(env: str | None, config_dir: Path | None) -> Container:
    configure_logging()
    try:
        return Container(config=load_config(env, config_dir), secrets=Secrets())
    except ConfigError as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def healthcheck(
    env: EnvOption = None,
    config_dir: ConfigDirOption = None,
    skip_llm: Annotated[bool, typer.Option("--skip-llm", help="Do not call the LLM.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Check config, secrets, broker account, market clock, and LLM access."""
    container = _container(env, config_dir)
    results = run_healthchecks(
        container.config,
        container.secrets,
        broker_factory=container.build_broker,
        llm_factory=None if skip_llm else container.build_llm_gateway,
    )
    ok = overall_ok(results)

    if as_json:
        payload = {
            "ok": ok,
            "checks": [
                {"name": r.name, "status": r.status.value, "detail": r.detail} for r in results
            ],
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        for result in results:
            typer.secho(f"{result.status.value:<5}", fg=_COLORS[result.status], nl=False)
            typer.echo(f" {result.name:<16} {result.detail}")
        verdict = "Ready" if ok else "Not ready: fix the FAIL items above"
        typer.secho(f"\n{verdict}", fg=typer.colors.GREEN if ok else typer.colors.RED, bold=True)

    raise typer.Exit(code=0 if ok else 1)


@app.command("show-config")
def show_config(env: EnvOption = None, config_dir: ConfigDirOption = None) -> None:
    """Print the effective configuration (never includes secrets)."""
    container = _container(env, config_dir)
    typer.echo(json.dumps(container.config.model_dump(mode="json"), indent=2))


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
