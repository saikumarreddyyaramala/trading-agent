"""Command-line entry point: `trading-agent <command>`."""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Any

import anyio
import typer

from trading_agent import __version__
from trading_agent.agents.analyst import AnalystResult
from trading_agent.bootstrap import Container
from trading_agent.data.errors import MarketDataError
from trading_agent.domain.models import Evidence, RiskDecision, SymbolData, TradeProposal
from trading_agent.health import CheckStatus, format_age, overall_ok, run_healthchecks
from trading_agent.mcp_server.server import build_server
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
        market_data_factory=container.build_market_data,
        news_factory=container.build_news,
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


def _symbol_payload(data: SymbolData, now: datetime) -> dict[str, Any]:
    quote = data.quote
    return {
        "symbol": data.symbol,
        "name": data.asset.name,
        "exchange": data.asset.exchange,
        "tradable": data.asset.tradable,
        "quote": None
        if quote is None
        else {
            "bid": str(quote.bid_price),
            "ask": str(quote.ask_price),
            "spread_pct": f"{quote.spread_pct:.3f}",
            "timestamp": quote.timestamp.isoformat(),
            "age_seconds": round(quote.age_seconds(now)),
        },
        "last_close": str(data.last_close),
        "change_pct": f"{data.change_pct:.2f}",
        "avg_daily_volume": data.avg_daily_volume,
        "bars": len(data.daily_bars),
        "news": [
            {
                "created_at": a.created_at.isoformat(),
                "headline": a.headline,
                "source": a.source,
                "url": a.url,
            }
            for a in data.news
        ],
    }


def _print_symbol(data: SymbolData, now: datetime) -> None:
    asset = data.asset
    state = "tradable" if asset.tradable else "NOT tradable"
    typer.secho(f"{data.symbol}", bold=True, nl=False)
    typer.echo(f"  {asset.name or ''} ({asset.exchange}, {state})")
    quote = data.quote
    if quote is None:
        typer.echo("  Quote   none usable")
    else:
        typer.echo(
            f"  Quote   bid ${quote.bid_price:,.2f} / ask ${quote.ask_price:,.2f} "
            f"(spread {quote.spread_pct:.2f}%), {format_age(quote.age_seconds(now))} old"
        )
    typer.echo(
        f"  Daily   last close ${data.last_close:,.2f}, {len(data.daily_bars)}d change "
        f"{data.change_pct:+.2f}%, avg volume {data.avg_daily_volume:,}"
    )
    typer.echo(f"  News    {len(data.news)} recent article(s)")
    for article in data.news:
        typer.echo(f"          {article.created_at:%Y-%m-%d %H:%M}  {article.headline}")


@app.command("market-data")
def market_data(
    symbols: Annotated[list[str], typer.Argument(help="Ticker symbols, e.g. AAPL MSFT.")],
    env: EnvOption = None,
    config_dir: ConfigDirOption = None,
    fresh: Annotated[
        bool, typer.Option("--fresh", help="Reject symbols whose quote is stale or unusable.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Collect quotes, daily bars, and news for symbols (read-only)."""
    container = _container(env, config_dir)
    try:
        snapshot = container.build_collector().collect(symbols, require_fresh_quotes=fresh)
    except (ConfigError, ValueError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    except MarketDataError as exc:
        typer.secho(f"Market data unavailable: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    now = snapshot.collected_at
    if as_json:
        payload = {
            "collected_at": now.isoformat(),
            "symbols": [_symbol_payload(d, now) for d in snapshot.symbols.values()],
            "rejected": snapshot.rejected,
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        for data in snapshot.symbols.values():
            _print_symbol(data, now)
            typer.echo("")
        for symbol, reason in snapshot.rejected.items():
            typer.secho("REJECTED", fg=typer.colors.RED, nl=False)
            typer.echo(f"  {symbol}: {reason}")

    raise typer.Exit(code=1 if snapshot.rejected else 0)


def _print_analysis(result: AnalystResult) -> None:
    colors = {"proposed": typer.colors.GREEN, "no_trade": typer.colors.YELLOW}
    typer.secho(
        f"{result.status.upper()}",
        fg=colors.get(result.status, typer.colors.RED),
        bold=True,
        nl=False,
    )
    usage = result.usage
    typer.echo(
        f"  {result.reason}  [run {result.run_id}, {result.turns} turns, "
        f"{len(result.tool_calls)} tool calls, "
        f"{usage.input_tokens + usage.output_tokens:,} tokens, {result.duration_seconds:.1f}s]"
    )
    proposal = result.proposal
    if proposal is not None:
        if proposal.decision == "buy":
            typer.echo(
                f"Decision  BUY {proposal.symbol} near ${proposal.reference_price} "
                f"(confidence {proposal.confidence})"
            )
        else:
            typer.echo(f"Decision  NO TRADE (confidence {proposal.confidence})")
        typer.echo(f"Thesis    {proposal.thesis}")
        for item in proposal.evidence:
            typer.echo(f"Evidence  - {item.claim} [{item.source_call_id}]")
        for risk in proposal.risks:
            typer.echo(f"Risk      - {risk}")
    typer.echo("\nNo order was placed. Proposals go to the risk engine (Phase 4).")


@app.command()
def analyze(
    env: EnvOption = None,
    config_dir: ConfigDirOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Run the analyst agent: research with read-only tools and propose one trade or none.

    Calls the LLM (costs tokens). Never places an order.
    """
    container = _container(env, config_dir)
    try:
        agent = container.build_analyst()
    except ConfigError as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    result = anyio.run(agent.run)

    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        _print_analysis(result)
    raise typer.Exit(code=1 if result.status == "failed" else 0)


def _print_risk(decision: RiskDecision) -> None:
    for check in decision.checks:
        status = CheckStatus.PASS if check.passed else CheckStatus.FAIL
        typer.secho(f"{status.value:<5}", fg=_COLORS[status], nl=False)
        typer.echo(f" {check.name:<18} {check.detail}")
    plan = decision.plan
    if plan is not None:
        typer.secho("\nAPPROVED", fg=typer.colors.GREEN, bold=True)
        typer.echo(
            f"Buy {plan.quantity} {plan.symbol} (~${plan.notional_usd}), "
            f"limit ${plan.limit_price}, stop ${plan.stop_loss_price}, "
            f"target ${plan.take_profit_price}"
        )
    else:
        typer.secho("\nREJECTED", fg=typer.colors.RED, bold=True)
    typer.echo("No order was placed. Execution arrives in Phase 5.")


@app.command("risk-check")
def risk_check(
    symbol: Annotated[str, typer.Argument(help="Ticker to check, e.g. AAPL.")],
    price: Annotated[
        str, typer.Option("--price", help="Reference price, as an analyst would give.")
    ],
    env: EnvOption = None,
    config_dir: ConfigDirOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Run every risk rule on a hypothetical buy, with live data. Never places an order."""
    container = _container(env, config_dir)
    try:
        reference = Decimal(price)
        proposal = TradeProposal(
            decision="buy",
            symbol=symbol,
            thesis="Manual risk check from the command line.",
            reference_price=reference,
            confidence="low",
            evidence=(
                Evidence(claim="Manual check", source_call_id="cli"),
                Evidence(claim="Manual check", source_call_id="cli"),
            ),
            risks=("Manual check",),
        )
        engine = container.build_risk_engine()
    except (InvalidOperation, ValueError) as exc:
        typer.secho(f"Error: invalid input: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    except ConfigError as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    decision = engine.evaluate(proposal)
    if as_json:
        typer.echo(decision.model_dump_json(indent=2))
    else:
        _print_risk(decision)
    raise typer.Exit(code=0 if decision.approved else 1)


@app.command("mcp-server")
def mcp_server(env: EnvOption = None, config_dir: ConfigDirOption = None) -> None:
    """Serve the read-only research tools over MCP (stdio), for an agent host to launch."""
    container = _container(env, config_dir)
    try:
        tools = container.build_research_tools()
    except ConfigError as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    # stdout is the protocol channel from here on; logs already go to stderr.
    build_server(tools).run("stdio")


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
