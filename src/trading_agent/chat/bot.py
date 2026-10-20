"""The chat bot (ADR 0022): answers the operator's messages from the bot's own records.

Common questions (hi, status, positions, a day's report, trades over a period, costs, the
plan) get free answers by fixed rules. Anything else goes to the AI with the same records
as context, under a daily cap. Nothing said in chat can trade: buy and sell requests are
refused, and the one action, "pause", only stops new trades for the rest of today.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from trading_agent.chat.dates import parse_day, parse_span, span_label
from trading_agent.chat.pause import PauseStore
from trading_agent.domain.models import ChatMessage, LLMRequest, LLMResponse
from trading_agent.llm.errors import LLMError
from trading_agent.observability.logging import get_logger
from trading_agent.settings.models import AppConfig, ChatConfig

MAX_REPLY_CHARS = 1500
AI_MAX_TOKENS = 600

HELP = (
    "Happy to help! I can answer:\n"
    "• hi or status: how I'm doing right now\n"
    "• positions: what I'm holding\n"
    '• report [day]: a day\'s summary, e.g. "report last wednesday"\n'
    '• trades [period]: trades and results, e.g. "trades this week"\n'
    "• costs [period]: what running me cost\n"
    "• plan: my to-do list for the next session\n"
    "• pause: stop new trades for the rest of today\n"
    "Anything else, just ask in your own words."
)

SYSTEM_PROMPT = """\
You are an automated US stock trading bot, chatting with your owner on Telegram. Speak as
the bot itself, in the first person ("I watched...", "I didn't trade because..."). You
trade a paper (simulated) account by yourself, with code-enforced risk checks. In chat you
can only explain: you never place, change, or cancel orders, and nothing said in chat
changes what you do (the owner can send "pause" to stop new trades today).

If you are answering, you are running. "About me" describes your daily schedule; "Today"
shows what you have done so far and what comes next; "Journal" shows what you have
watched and recorded today. Use them to answer questions about yourself.

Answer from the records given with each question, and say plainly when they do not cover
it; never invent trades, prices, or numbers. Indicators and past results describe the past;
do not promise profits or give personal financial advice.

Tone: warm, friendly, and encouraging, like a helpful assistant talking with the person it
works for; never curt or cold. Use plain words, at most 6 short lines of plain text, no
Markdown tables. One emoji is fine when it fits. When the news is bad (a loss, no trade, a
problem), say it honestly and kindly, and say what happens next."""

_CHECK_IN = re.compile(
    r"^(/?status|what'?s up|sup|how(?: is|'s| was| are) (?:it going|everything|things|you"
    r"(?: doing)?)|are you (?:there|ok|okay|alive|awake|running|working|watching|on)|you there)\b"
)
_CHECK_IN_MAX_WORDS = 7
_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'})

_GREETING = re.compile(
    r"^(hi+|hel+o+|he+y+|hiya|yo|sup|gm|good (morning|afternoon|evening)|/start)\b"
    r"[\s!.,?]*\w{0,12}[\s!.?]*$"
)
_TRADE_REQUEST = re.compile(r"^(please\s+)?(buy|sell|short|close|cancel|exit|liquidate)\b")


class ChatData(Protocol):
    """Read-only records, as plain JSON-style values (Decimals as strings)."""

    def journal(self, day: date) -> dict[str, Any] | None:
        """What the market journal recorded today: snapshots, sources, notes, notebook."""
        ...

    def overview(self) -> dict[str, Any]: ...

    def day_report(self, day: date) -> dict[str, Any] | None: ...

    def trades(self, days: int) -> list[dict[str, Any]]: ...

    def costs(self, days: int) -> dict[str, Any]: ...

    def plan(self, day: date) -> dict[str, Any] | None: ...


def _dec(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _usd(value: Any, *, sign: bool = False) -> str:
    amount = _dec(value) or Decimal(0)
    text = f"${abs(amount):,.2f}"
    if sign:
        return ("-" if amount < 0 else "+") + text
    return ("-" if amount < 0 else "") + text


def _pct(value: Any) -> str:
    amount = _dec(value) or Decimal(0)
    return f"{amount:+.2f}%"


def about_text(config: AppConfig) -> str:
    """The bot's own day, from the live config, so chat can explain itself."""
    schedule, journal, risk = config.schedule, config.journal, config.risk
    slots = schedule.extra_analyses[: max(risk.max_trades_per_day - 1, 0)]
    post_end = (
        datetime.combine(date.today(), time(16, 0)) + timedelta(hours=journal.post_market_hours)
    ).time()
    notebook = (
        datetime.combine(date.today(), post_end) + timedelta(minutes=journal.notebook_delay_minutes)
    ).time()
    steps = [
        f"{journal.pre_market_start:%H:%M} I start watching and recording the market every "
        f"{journal.snapshot_minutes} minutes (pre-market is watch only; prices on the free "
        "data feed start at 08:00)",
        f"{schedule.premarket_watch:%H:%M} pre-market scan",
        f"{schedule.data_collection:%H:%M} health check",
        f"{schedule.analysis:%H:%M} AI research and the critic's review",
        f"{schedule.trade_window_start:%H:%M} the first possible buy, only if every risk "
        "check passes",
    ]
    steps += [f"{at:%H:%M} another research and buy chance" for at in slots]
    steps += [
        f"{schedule.force_close:%H:%M} I sell everything",
        f"16:00-{post_end:%H:%M} post-market, watch only",
        f"{notebook:%H:%M} my day notebook",
        f"{schedule.post_market:%H:%M} lessons, guidebook, tomorrow's plan, and the report",
    ]
    limits = (
        f"At most {risk.max_trades_per_day} trades a day, ${risk.max_position_usd:g} per "
        f"position, a {risk.stop_loss_pct:g}% stop-loss and a {risk.take_profit_pct:g}% "
        "take-profit; never outside regular hours."
    )
    return f"My day ({schedule.timezone} time): " + "; ".join(steps) + ". " + limits


def _weekday(day: date) -> str:
    return f"{day:%a} {day.isoformat()}"


class ChatBot:
    def __init__(
        self,
        data: ChatData,
        pause: PauseStore,
        config: ChatConfig,
        *,
        complete: Callable[[LLMRequest], LLMResponse] | None,
        clock: Callable[[], datetime] | None = None,
        timezone: str = "America/New_York",
        about: str = "",
        logger: Any = None,
    ) -> None:
        self._about = about
        self._data = data
        self._pause = pause
        self._config = config
        self._complete = complete
        self._clock = clock or (lambda: datetime.now(UTC))
        self._zone = ZoneInfo(timezone)
        self._log = logger or get_logger(__name__)
        self._ai_day: date | None = None
        self._ai_used = 0

    def _today(self) -> date:
        return self._clock().astimezone(self._zone).date()

    # ---------- Entry point ----------

    def reply(self, text: str) -> str:
        """Always returns a reply; never raises for bad data or a missing AI."""
        message = " ".join(text.translate(_QUOTES).split())
        lowered = message.lower()
        try:
            return self._route(message, lowered)[:MAX_REPLY_CHARS]
        except Exception as exc:
            self._log.warning("chat.data_failed", error=str(exc))
            return (
                f"Sorry, I couldn't read my records just now ({type(exc).__name__}: {exc}). "
                "Please try again in a minute."
            )

    def _route(self, message: str, lowered: str) -> str:
        today = self._today()
        if not lowered or lowered in ("help", "/help", "?", "commands", "menu"):
            return HELP
        bare = lowered.rstrip(" ?!.")
        if _CHECK_IN.match(bare) and len(bare.split()) <= _CHECK_IN_MAX_WORDS:
            return "\n".join([self._opener(bare), *self._status_lines()])
        if _GREETING.match(lowered):
            return "\n".join(
                [
                    "Hi! 👋",
                    *self._status_lines(),
                    "Ask me anything, or send help to see what I can do.",
                ]
            )
        if _TRADE_REQUEST.match(lowered):
            return (
                "I'd love to help, but I can't place trades from chat. Only the daily cycle "
                "trades, after all its safety checks. If you'd like me to hold off, send "
                '"pause" and I\'ll stop new trades for the rest of today.'
            )
        if lowered in ("pause", "/pause", "pause trading", "stop trading", "stop"):
            return self._pause_today(today)
        if lowered in ("resume", "/resume", "unpause", "start trading", "resume trading"):
            return (
                "For safety I can't restart trading from chat. A pause ends by itself "
                "tomorrow, or run `trading-agent resume` on the computer."
            )
        if re.search(r"\b(positions?|holding|holdings|own)\b", lowered):
            return self._positions()
        if re.search(r"\b(report|summary|recap)\b", lowered):
            return self._day(parse_day(lowered, today) or self._last_session(today))
        if re.search(r"\btrades\b|\bhow many trade", lowered):
            days = parse_span(lowered, today) or 7
            return self._trades(days, span_label(lowered, days))
        if re.search(r"\b(costs?|spend|spent|spending|bill|budget)\b", lowered):
            days = parse_span(lowered, today) or 1
            return self._costs(days, span_label(lowered, days))
        if re.search(r"\b(plan|to-?do)\b", lowered):
            return self._plan(today)
        return self._ask_ai(message, today)

    # ---------- Free answers ----------

    def _status_lines(self) -> list[str]:
        """Where things stand, as short natural sentences."""
        view = self._data.overview()
        lines = []
        market = view.get("market")
        if market is None:
            lines.append("I can't see the market clock right now.")
        elif market.get("is_open"):
            closes = datetime.fromisoformat(market["next_close"]).astimezone(self._zone)
            lines.append(f"The market is open until {closes:%H:%M}.")
        else:
            opens = datetime.fromisoformat(market["next_open"]).astimezone(self._zone)
            lines.append(f"The market is closed right now. It opens {opens:%a} at {opens:%H:%M}.")
        positions = [p["symbol"] for p in view.get("positions") or []]
        if not positions:
            lines.append("I have no open positions, so everything is in cash.")
        else:
            noun = "position" if len(positions) == 1 else "positions"
            lines.append(f"I'm holding {len(positions)} {noun}: {', '.join(positions)}.")
        account = view.get("account")
        if account:
            change = _dec(account.get("day_change"))
            if change is None:
                today = ""
            elif change == 0:
                today = " (unchanged today)"
            else:
                today = f" ({_usd(change, sign=True)} today)"
            lines.append(f"Account value: {_usd(account.get('equity'))}{today}.")
        try:
            journal = self._data.journal(self._today())
        except Exception:  # the journal only adds to the answer; never block it
            journal = None
        if journal and journal.get("snapshots"):
            lines.append(
                f"Today I've recorded {journal['snapshots']} market snapshots and "
                f"{journal.get('sources', 0)} news items."
            )
        if self._pause.paused_reason(self._today()):
            lines.append("New trades are paused for today.")
        if view.get("kill_switch"):
            lines.append("The kill switch is on, so no new trades.")
        for error in (view.get("errors") or [])[:2]:
            lines.append(f"(Note: {error}.)")
        return lines

    def _status_line(self) -> str:
        return "\n".join(self._status_lines())

    @staticmethod
    def _opener(check_in: str) -> str:
        if re.match(r"how(?: are you| are things|'s it going| is it going)", check_in):
            return "I'm doing well, thanks for asking! 😊"
        if re.match(r"(?:what'?s up|sup)\b", check_in):
            return "Not much, just keeping an eye on the market. 👀"
        if re.match(r"how(?: is|'s| was) everything", check_in):
            return "Everything's running smoothly. 👍"
        if re.match(r"(?:are you|you there)", check_in):
            return "Yes, I'm here and watching. 👀"
        return "Here's where things stand:"

    def _positions(self) -> str:
        rows = self._data.overview().get("positions") or []
        if not rows:
            return "No open positions right now, so everything is in cash. 💵"
        lines = ["Here's what I'm holding:"]
        lines += [
            f"{p['symbol']} {p['quantity']} sh at {_usd(p['avg_entry_price'])}, now "
            f"{_usd(p['price'])} ({_usd(p['unrealized_pnl'], sign=True)}, "
            f"{_pct(p['unrealized_pct'])}), stop {_usd(p['stop_price'])}, "
            f"target {_usd(p['target_price'])}"
            for p in rows
        ]
        return "\n".join(lines)

    @staticmethod
    def _last_session(today: date) -> date:
        day = today
        while day.weekday() >= 5:  # holidays show up as "no record"
            day -= timedelta(days=1)
        return day

    @staticmethod
    def _trade_line(trade: dict[str, Any]) -> str:
        bought = f"{trade['symbol']} bought at {_usd(trade['entry_price'])}"
        if not trade.get("closed"):
            return f"{bought}, still open"
        return (
            f"{bought}, sold at {_usd(trade['exit_price'])}: "
            f"{_usd(trade['pnl_usd'], sign=True)} ({_pct(trade['return_pct'])})"
        )

    def _day(self, day: date) -> str:
        report = self._data.day_report(day)
        trades = (report or {}).get("trades") or []
        if report is None or (report.get("cycle_status") is None and not trades):
            return (
                f"I don't have a trading record for {_weekday(day)}. The market was closed that "
                "day, or I wasn't running."
            )
        lines = [f"{_weekday(day)}: {report.get('cycle_status') or 'no cycle result'}"]
        if trades:
            closed = [t for t in trades if t.get("closed")]
            wins = sum(1 for t in closed if (_dec(t.get("pnl_usd")) or 0) > 0)
            pnl = sum((_dec(t.get("pnl_usd")) or Decimal(0) for t in closed), Decimal(0))
            noun = "trade" if len(trades) == 1 else "trades"
            lines.append(f"{len(trades)} {noun}, {wins} won, P&L {_usd(pnl, sign=True)}")
            lines += [f"• {self._trade_line(t)}" for t in trades]
        else:
            lines.append("No trades that day.")
            if report.get("cycle_summary"):
                lines.append(f"Why: {report['cycle_summary']}")
        if report.get("benchmark_pct") is not None:
            lines.append(
                f"{report.get('benchmark_symbol', 'SPY')} {_pct(report['benchmark_pct'])} that day"
            )
        for warning in (report.get("warnings") or [])[:2]:
            lines.append(f"Warning: {warning}")
        return "\n".join(lines)

    def _trades(self, days: int, label: str) -> str:
        rows = self._data.trades(days)
        if not rows:
            return f"{label}: no trades yet."
        closed = [t for t in rows if t.get("closed")]
        wins = sum(1 for t in closed if (_dec(t.get("pnl_usd")) or 0) > 0)
        pnl = sum((_dec(t.get("pnl_usd")) or Decimal(0) for t in closed), Decimal(0))
        noun = "trade" if len(rows) == 1 else "trades"
        lines = [
            f"{label}: {len(rows)} {noun} ({len(closed)} closed, {wins} won), "
            f"P&L {_usd(pnl, sign=True)}"
        ]
        for t in rows[:10]:
            day = date.fromisoformat(t["day"])
            result = (
                f"{_usd(t['pnl_usd'], sign=True)} ({_pct(t['return_pct'])})"
                if t.get("closed")
                else "still open"
            )
            lines.append(f"• {day:%a %m-%d} {t['symbol']} {result}")
        if len(rows) > 10:
            lines.append(f"…and {len(rows) - 10} more")
        return "\n".join(lines)

    def _costs(self, days: int, label: str) -> str:
        report = self._data.costs(days)
        total = report.get("total") or {}
        calls = total.get("calls", 0)
        budget = report.get("daily_budget_usd")
        tail = f" Daily AI budget: {_usd(budget)}." if budget is not None else ""
        head = f"Costs {label.lower()}"
        if total.get("total_usd") is None:
            return (
                f"{head}: unknown ({total.get('unpriced_calls', calls)} AI calls without a "
                "price; set costs.llm_prices in config/base.yaml to see dollars)." + tail
            )
        return f"{head}: {_usd(total['total_usd'])} ({calls} AI calls).{tail}"

    def _plan(self, today: date) -> str:
        plan = self._data.plan(today)
        if not plan or not plan.get("items"):
            return "No plan yet. I write one every evening after the session closes."
        day = date.fromisoformat(plan["for_day"])
        items = sorted(plan["items"], key=lambda item: item.get("priority", 3))
        lines = [f"Plan for {_weekday(day)}:"]
        lines += [f"{n}. {item['text']}" for n, item in enumerate(items, start=1)]
        return "\n".join(lines)

    def _pause_today(self, today: date) -> str:
        if self._pause.paused_reason(today):
            return "Already paused for today 👍 I'll start trading again tomorrow."
        try:
            self._pause.pause(today, by="telegram")
        except OSError as exc:
            return (
                f"Sorry, I couldn't save the pause ({exc}), so nothing changed. Please try again."
            )
        self._log.info("chat.paused", day=today.isoformat())
        return (
            f"Done ✅ I've paused new trades for the rest of today ({_weekday(today)}). Any "
            "open positions keep their stop-loss and are still managed and closed as usual. "
            "I'll start trading again tomorrow."
        )

    # ---------- AI answers ----------

    def _context(self, message: str, today: date) -> str:
        parts = [f"Now: {self._clock().astimezone(self._zone):%a %Y-%m-%d %H:%M} New York"]
        if self._about:
            parts.append(f"About me: {self._about}")
        parts.append(f"Status: {self._status_line()}")
        view = self._data.overview()
        parts.append(
            "Today: " + json.dumps({"cycle": view.get("cycle"), "next_step": view.get("next_step")})
        )
        try:
            parts.append(f"Journal: {json.dumps(self._data.journal(today))}")
        except Exception:
            parts.append("Journal: unavailable")
        day = parse_day(message, today)
        if day is not None:
            parts.append(f"Report for {day.isoformat()}: {json.dumps(self._data.day_report(day))}")
        days = parse_span(message, today) or 7
        parts.append(f"Trades, last {days} days: {json.dumps(self._data.trades(days))}")
        plan = self._data.plan(today)
        if plan:
            parts.append(f"Next plan: {json.dumps(plan)}")
        return "\n".join(parts)

    def _ask_ai(self, message: str, today: date) -> str:
        if self._complete is None or not self._config.ai_answers:
            return HELP
        if self._ai_day != today:
            self._ai_day, self._ai_used = today, 0
        cap = self._config.max_ai_answers_per_day
        if self._ai_used >= cap:
            return (
                f"I've used my {cap} AI answers today, sorry! I can still answer the basics "
                "for free: send help to see them."
            )
        request = LLMRequest(
            system=SYSTEM_PROMPT,
            messages=(
                ChatMessage(
                    role="user",
                    content=f"Records:\n{self._context(message, today)}\n\nQuestion: {message}",
                ),
            ),
            max_tokens=AI_MAX_TOKENS,
            tags={"agent": "chat"},
        )
        try:
            response = self._complete(request)
        except LLMError as exc:
            return f"Sorry, I couldn't reach the AI ({exc}) just now.\n\n{HELP}"
        self._ai_used += 1
        return response.text.strip() or HELP
