"""Command-line entry point: `trading-agent <command>`."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo

import anyio
import typer

from trading_agent import __version__
from trading_agent.agents.analyst import AnalystResult
from trading_agent.agents.critic import CriticResult
from trading_agent.bootstrap import Container
from trading_agent.data.errors import MarketDataError
from trading_agent.domain.models import Evidence, RiskDecision, SymbolData, TradeProposal
from trading_agent.execution.executor import (
    EntryResult,
    ExecutionDisabledError,
    ManageResult,
)
from trading_agent.health import CheckStatus, format_age, overall_ok, run_healthchecks
from trading_agent.mcp_server.server import build_server
from trading_agent.observability.logging import configure_logging
from trading_agent.orchestration.cycle import DailyCycle, DayResult
from trading_agent.reporting.delivery import DeliveryError, post_webhook, save_report
from trading_agent.reporting.report import render_markdown
from trading_agent.settings.loader import ConfigError, load_config
from trading_agent.settings.secrets import Secrets
from trading_agent.storage.database import (
    current_revision,
    head_revision,
    make_engine,
    upgrade,
)
from trading_agent.storage.store import AuditError, AuditRecorder, AuditStore

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

db_app = typer.Typer(help="Audit database: migrations and status.", no_args_is_help=True)
app.add_typer(db_app, name="db")

_COLORS = {
    CheckStatus.PASS: typer.colors.GREEN,
    CheckStatus.WARN: typer.colors.YELLOW,
    CheckStatus.FAIL: typer.colors.RED,
    CheckStatus.SKIP: typer.colors.BRIGHT_BLACK,
}


def _engine_factory(container: Container) -> Callable[[], Any] | None:
    url = container.database_url()
    return None if url is None else (lambda: make_engine(url))


def _recorder(container: Container) -> AuditRecorder:
    try:
        return container.build_recorder()
    except ConfigError as exc:
        typer.secho(f"Audit store unavailable: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


def _audit(recorder: AuditRecorder, what: str, fn: Callable[[AuditStore], None]) -> bool:
    """Write to the audit store; a required write that fails ends the command."""
    try:
        return recorder.write(what, fn)
    except AuditError as exc:
        typer.secho(f"Audit write failed ({what}): {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


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
        engine_factory=_engine_factory(container),
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


def _print_critique(result: CriticResult) -> None:
    colors = {"approved": typer.colors.GREEN, "vetoed": typer.colors.YELLOW}
    typer.secho("\nCRITIC ", bold=True, nl=False)
    typer.secho(
        result.status.upper(), fg=colors.get(result.status, typer.colors.RED), bold=True, nl=False
    )
    usage = result.usage
    typer.echo(f"  [{result.model}, {usage.input_tokens + usage.output_tokens:,} tokens]")
    typer.echo(f"Summary   {result.summary}")
    for check in result.evidence_checks:
        mark = "supported" if check.supported else "UNSUPPORTED"
        typer.echo(f"Check     {mark:<11} {check.source_call_id}: {check.note}")
    for reason in result.reasons:
        typer.echo(f"Reason    - {reason}")
    for concern in result.concerns:
        typer.echo(f"Concern   - {concern}")


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
    recorder = _recorder(container)
    try:
        agent = container.build_analyst()
    except ConfigError as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    result = anyio.run(agent.run)
    critique: CriticResult | None = None
    proposal = result.proposal
    if result.status == "proposed" and proposal is not None and proposal.decision == "buy":
        critic = container.build_critic()
        if critic is not None:
            critique = critic.review(result.run_id, proposal, list(result.tool_calls))

    def record(store: AuditStore) -> None:
        store.start_run(result.run_id, "analyze")
        store.record_analysis(result)
        if critique is not None:
            store.record_critique(critique)
        store.finish_run(result.run_id, result.status, result.reason)

    _audit(recorder, "analysis", record)

    if as_json:
        payload = {
            "analysis": result.model_dump(mode="json"),
            "critique": None if critique is None else critique.model_dump(mode="json"),
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        _print_analysis(result)
        if critique is not None:
            _print_critique(critique)
        typer.echo("\nNo order was placed. Approved proposals go to the risk engine.")
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
    typer.echo("The risk check itself places no order.")


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
        proposal = _manual_proposal(symbol, price)
        engine = container.build_risk_engine()
    except (InvalidOperation, ValueError) as exc:
        typer.secho(f"Error: invalid input: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    except ConfigError as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    recorder = _recorder(container)
    decision = engine.evaluate(proposal)
    run_id = f"check-{uuid.uuid4().hex[:10]}"

    def record(store: AuditStore) -> None:
        store.start_run(run_id, "risk_check")
        store.record_risk_decision(run_id, decision)
        store.finish_run(run_id, "approved" if decision.approved else "rejected", proposal.thesis)

    _audit(recorder, "risk decision", record)
    if as_json:
        typer.echo(decision.model_dump_json(indent=2))
    else:
        _print_risk(decision)
    raise typer.Exit(code=0 if decision.approved else 1)


ConfirmOption = Annotated[
    bool, typer.Option("--confirm", help="Required: this places real orders on the paper account.")
]


def _manual_proposal(symbol: str, price: str) -> TradeProposal:
    return TradeProposal(
        decision="buy",
        symbol=symbol,
        thesis="Manual order from the command line.",
        reference_price=Decimal(price),
        confidence="low",
        evidence=(
            Evidence(claim="Manual", source_call_id="cli"),
            Evidence(claim="Manual", source_call_id="cli"),
        ),
        risks=("Manual",),
    )


def _print_entry(result: EntryResult) -> None:
    color = typer.colors.GREEN if result.status == "entered" else typer.colors.RED
    typer.secho(result.status.upper(), fg=color, bold=True, nl=False)
    typer.echo(f"  {result.reason}")
    if result.entry_order is not None:
        order = result.entry_order
        typer.echo(
            f"Entry  {order.status}: {order.filled_quantity}/{order.quantity} {order.symbol} "
            f"at avg ${order.filled_avg_price or '-'} (limit ${order.limit_price})"
        )
    if result.stop_order is not None:
        stop = result.stop_order
        typer.echo(
            f"Stop   {stop.status}: sell {stop.quantity} {stop.symbol} at ${stop.stop_price}"
        )


def _print_manage(results: tuple[ManageResult, ...]) -> None:
    for result in results:
        color = typer.colors.RED if result.action == "error" else typer.colors.GREEN
        typer.secho(f"{result.action.upper():<14}", fg=color, nl=False)
        typer.echo(f" {result.symbol or '':<6} {result.detail}")


def _executor_or_exit(container: Container) -> Any:
    try:
        return container.build_executor()
    except (ConfigError, ExecutionDisabledError) as exc:
        typer.secho(f"Execution unavailable: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def buy(
    symbol: Annotated[str, typer.Argument(help="Ticker to buy, e.g. AAPL.")],
    price: Annotated[str, typer.Option("--price", help="Reference price for the risk checks.")],
    confirm: ConfirmOption = False,
    env: EnvOption = None,
    config_dir: ConfigDirOption = None,
) -> None:
    """Risk-check a buy and, if approved, enter it on the PAPER account with a stop-loss."""
    if not confirm:
        typer.secho(
            "Refusing without --confirm: this places orders on the paper account. "
            "Use `risk-check` for a dry run.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    container = _container(env, config_dir)
    executor = _executor_or_exit(container)
    try:
        proposal = _manual_proposal(symbol, price)
        engine = container.build_risk_engine()
    except (InvalidOperation, ValueError) as exc:
        typer.secho(f"Error: invalid input: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    recorder = _recorder(container)
    run_id = f"manual-{uuid.uuid4().hex[:10]}"
    decision = engine.evaluate(proposal)

    def record_decision(store: AuditStore) -> None:
        store.start_run(run_id, "manual_buy")
        store.record_risk_decision(run_id, decision)

    # When the audit store is required, no record means no trade.
    _audit(recorder, "risk decision", record_decision)
    _print_risk(decision)
    if not decision.approved:
        _audit(recorder, "run result", lambda s: s.finish_run(run_id, "rejected", "risk rejected"))
        raise typer.Exit(code=1)
    typer.echo("")
    result = executor.enter(decision, run_id)
    _print_entry(result)

    def record_entry(store: AuditStore) -> None:
        orders = [o for o in (result.entry_order, result.stop_order) if o is not None]
        store.record_orders(run_id, orders)
        store.finish_run(run_id, result.status, result.reason)

    _audit(recorder, "orders", record_entry)
    raise typer.Exit(code=0 if result.status == "entered" else 1)


@app.command()
def manage(env: EnvOption = None, config_dir: ConfigDirOption = None) -> None:
    """One exit-management pass: force-close, take profit, and keep a stop on every position."""
    container = _container(env, config_dir)
    recorder = _recorder(container)
    results = _executor_or_exit(container).manage()
    _print_manage(results)
    # Record only passes that changed something; "hold" every minute would be noise.
    changes = [r for r in results if r.action not in ("flat", "hold")]

    def record(store: AuditStore) -> None:
        for result in changes:
            store.record_position_event(None, result)

    if changes:
        _audit(recorder, "position events", record)
    raise typer.Exit(code=1 if any(r.action == "error" for r in results) else 0)


@app.command()
def flatten(
    confirm: ConfirmOption = False, env: EnvOption = None, config_dir: ConfigDirOption = None
) -> None:
    """Emergency exit: cancel every open order and close every position (paper)."""
    if not confirm:
        typer.secho("Refusing without --confirm.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    container = _container(env, config_dir)
    recorder = _recorder(container)
    result = _executor_or_exit(container).flatten()
    _print_manage((result,))
    _audit(recorder, "flatten", lambda s: s.record_position_event(None, result))
    raise typer.Exit(code=1 if result.action == "error" else 0)


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


def _database_or_exit(container: Container) -> str:
    url = container.database_url()
    if url is None:
        typer.secho("DATABASE_URL is not set.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    return url


@db_app.command("upgrade")
def db_upgrade(env: EnvOption = None, config_dir: ConfigDirOption = None) -> None:
    """Create or migrate the audit database schema to the latest version."""
    container = _container(env, config_dir)
    engine = make_engine(_database_or_exit(container))
    upgrade(engine)
    typer.echo(f"Audit database ({engine.dialect.name}) is at schema {current_revision(engine)}.")


@db_app.command("status")
def db_status(env: EnvOption = None, config_dir: ConfigDirOption = None) -> None:
    """Show the audit database's schema version."""
    container = _container(env, config_dir)
    engine = make_engine(_database_or_exit(container))
    current, head = current_revision(engine), head_revision()
    state = "up to date" if current == head else "needs `trading-agent db upgrade`"
    typer.echo(f"{engine.dialect.name}: schema {current or 'none'}, latest {head} ({state})")
    raise typer.Exit(code=0 if current == head else 1)


def _store_or_exit(container: Container) -> AuditStore:
    _database_or_exit(container)
    return _recorder(container).store


@app.command()
def runs(
    limit: Annotated[int, typer.Option("--limit", min=1, max=500)] = 20,
    env: EnvOption = None,
    config_dir: ConfigDirOption = None,
) -> None:
    """List recent runs from the audit trail."""
    store = _store_or_exit(_container(env, config_dir))
    for run in store.list_runs(limit):
        typer.echo(
            f"{run.started_at:%Y-%m-%d %H:%M}  {run.run_id:<20} {run.kind:<11} "
            f"{run.status or 'running':<18} {run.summary or ''}"[:160]
        )


@app.command("show-run")
def show_run(
    run_id: Annotated[str, typer.Argument(help="Run id from `trading-agent runs`.")],
    env: EnvOption = None,
    config_dir: ConfigDirOption = None,
) -> None:
    """Everything recorded for one run: analysis, tool calls, risk checks, orders, events."""
    store = _store_or_exit(_container(env, config_dir))
    report = store.get_run(run_id)
    if report is None:
        typer.secho(f"No run {run_id}.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(json.dumps(report, indent=2, default=str))


@app.command("sync-orders")
def sync_orders(
    days: Annotated[int, typer.Option("--days", min=1, max=30)] = 1,
    env: EnvOption = None,
    config_dir: ConfigDirOption = None,
) -> None:
    """Refresh recorded orders from the broker: final statuses, fills, and exit orders."""
    container = _container(env, config_dir)
    _database_or_exit(container)
    recorder = _recorder(container)
    try:
        orders = container.build_order_port().get_orders_since(
            datetime.now(UTC) - timedelta(days=days)
        )
    except Exception as exc:
        typer.secho(f"Could not fetch orders: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    _audit(recorder, "orders", lambda s: s.record_orders(None, orders))
    typer.echo(f"Synced {len(orders)} order(s) from the last {days} day(s).")


def _print_day(result: DayResult) -> None:
    color = {"traded": typer.colors.GREEN, "no_trade": typer.colors.YELLOW}.get(
        result.status, typer.colors.RED if result.status == "error" else typer.colors.WHITE
    )
    typer.secho(f"{result.status.upper()}", fg=color, bold=True, nl=False)
    typer.echo(f"  {result.run_id}: {result.summary}")
    for step, status in result.steps.items():
        typer.echo(f"  {step:<10} {status}")


@app.command()
def run(
    once: Annotated[
        bool, typer.Option("--once", help="Run today's cycle, then exit (default: every day).")
    ] = False,
    env: EnvOption = None,
    config_dir: ConfigDirOption = None,
) -> None:
    """Run the daily trading cycle on the PAPER account, on the configured schedule.

    Pre-flight, analysis (uses LLM tokens), risk check, entry, exit management, force-close,
    and reconciliation. Safe to restart: finished steps are not repeated.
    """
    container = _container(env, config_dir)
    try:
        services = container.build_cycle_services()
    except (ConfigError, ExecutionDisabledError) as exc:
        typer.secho(f"Cannot start the cycle: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    cycle = DailyCycle(services, container.config)

    if once:
        today = datetime.now(ZoneInfo(container.config.schedule.timezone)).date()
        result = cycle.run_day(today)
        _print_day(result)
        raise typer.Exit(code=1 if result.status == "error" else 0)
    try:
        cycle.run_forever()
    except KeyboardInterrupt:
        typer.echo(
            "Stopped. Open positions keep their broker-side stop-loss. "
            "Restart `trading-agent run` to resume, or run `trading-agent manage`."
        )


@app.command("cycle-status")
def cycle_status(
    day: Annotated[str | None, typer.Option("--date", help="YYYY-MM-DD (default: today).")] = None,
    env: EnvOption = None,
    config_dir: ConfigDirOption = None,
) -> None:
    """Show a day's cycle: each step's status and detail."""
    container = _container(env, config_dir)
    store = _store_or_exit(container)
    try:
        when = (
            date.fromisoformat(day)
            if day
            else datetime.now(ZoneInfo(container.config.schedule.timezone)).date()
        )
    except ValueError as exc:
        typer.secho(f"Invalid date: {day}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    report = store.get_run(DailyCycle.run_id(when))
    if report is None:
        typer.echo(f"No cycle recorded for {when}.")
        raise typer.Exit(code=1)
    run_row = report["run"]
    typer.echo(f"{run_row['run_id']}: {run_row['status'] or 'running'}  {run_row['summary'] or ''}")
    for step in report["steps"]:
        typer.echo(f"  {step['step']:<10} {step['status']:<10} {step['detail']}")


@app.command()
def report(
    day: Annotated[str | None, typer.Option("--date", help="YYYY-MM-DD (default: today).")] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print JSON instead of Markdown.")
    ] = False,
    save: Annotated[
        bool, typer.Option("--save/--no-save", help="Write reports/<date>.md and .json.")
    ] = True,
    send: Annotated[bool, typer.Option("--send", help="Also post to REPORT_WEBHOOK_URL.")] = False,
    env: EnvOption = None,
    config_dir: ConfigDirOption = None,
) -> None:
    """The daily report: analysis, risk check, trade, P&L, and comparison with the benchmark."""
    container = _container(env, config_dir)
    zone = ZoneInfo(container.config.schedule.timezone)
    try:
        when = date.fromisoformat(day) if day else datetime.now(zone).date()
    except ValueError as exc:
        typer.secho(f"Invalid date: {day}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    if send and container.report_webhook_url() is None:
        typer.secho("REPORT_WEBHOOK_URL is not set.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    try:
        builder = container.build_report_builder(_recorder(container).store)
    except ConfigError as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    built = builder.build(when)
    markdown = render_markdown(built)
    typer.echo(built.model_dump_json(indent=2) if as_json else markdown)
    try:
        if save:
            md_path, _ = save_report(built, markdown, Path(container.config.reporting.output_dir))
            typer.secho(f"Saved {md_path}", err=True)
        url = container.report_webhook_url()
        if send and url:
            post_webhook(url, markdown)
            typer.secho("Sent to the webhook.", err=True)
    except DeliveryError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def critique(
    run_id: Annotated[str, typer.Argument(help="A run with a verified buy proposal.")],
    env: EnvOption = None,
    config_dir: ConfigDirOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Re-review a recorded proposal with the critic (uses LLM tokens). Records nothing."""
    container = _container(env, config_dir)
    store = _store_or_exit(container)
    proposal = store.get_proposal(run_id)
    if proposal is None:
        typer.secho(
            f"No verified buy proposal recorded for {run_id}.", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)
    try:
        critic = container.build_critic()
    except ConfigError as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    if critic is None:
        typer.secho(
            "The critic is disabled (critic.enabled: false).", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=2)
    result = critic.review(run_id, proposal, store.tool_calls(run_id))
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        _print_critique(result)
    raise typer.Exit(code=1 if result.status == "failed" else 0)


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
