"""Chatting with the bot over Telegram (ADR 0022). Written first.

Free answers come from the bot's own records by fixed rules; anything else goes to the AI
with the same records as context, under a daily cap. Nothing said in chat can trade: the
only action is "pause", which can only make the bot safer.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from trading_agent.chat.bot import ChatBot
from trading_agent.chat.dates import parse_day, parse_span
from trading_agent.chat.pause import PauseStore
from trading_agent.domain.models import LLMRequest, LLMResponse, LLMUsage
from trading_agent.llm.errors import LLMError
from trading_agent.settings.models import ChatConfig

SATURDAY = datetime(2026, 10, 17, 16, 0, tzinfo=UTC)  # noon in New York
TODAY = date(2026, 10, 17)


# ---------- Dates ----------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("report for today", date(2026, 10, 17)),
        ("what happened yesterday", date(2026, 10, 16)),
        ("give me the report of last wednesday", date(2026, 10, 14)),
        ("wednesday report", date(2026, 10, 14)),
        ("last Friday please", date(2026, 10, 16)),
        ("report 2026-10-12", date(2026, 10, 12)),
        ("report oct 12", date(2026, 10, 12)),
        ("report 12 October", date(2026, 10, 12)),
        ("report 10/12", date(2026, 10, 12)),
        ("how are you", None),
        ("report for 2026-02-30", None),  # not a real date
    ],
)
def test_days_in_plain_words(text: str, expected: date | None) -> None:
    assert parse_day(text, TODAY) == expected


def test_a_weekday_today_means_today_but_last_means_a_week_ago() -> None:
    assert parse_day("saturday", TODAY) == TODAY
    assert parse_day("last saturday", TODAY) == date(2026, 10, 10)


def test_a_month_day_in_the_future_means_last_year() -> None:
    assert parse_day("report dec 24", TODAY) == date(2025, 12, 24)


@pytest.mark.parametrize(
    ("text", "days"),
    [
        ("how many trades this week", 6),  # Monday through Saturday
        ("trades last week", 7),
        ("last 30 days", 30),
        ("past 3 days", 3),
        ("this month", 17),
        ("trades", None),
    ],
)
def test_spans_in_plain_words(text: str, days: int | None) -> None:
    assert parse_span(text, TODAY) == days


# ---------- Fakes ----------


def _overview(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "environment": "paper",
        "kill_switch": False,
        "account": {"equity": "100012.34", "day_change": "12.40", "day_change_pct": "0.01"},
        "positions": [],
        "market": {
            "is_open": False,
            "next_open": "2026-10-19T13:30:00+00:00",
            "next_close": "2026-10-19T20:00:00+00:00",
        },
        "cycle": None,
        "errors": [],
    }
    data.update(changes)
    return data


AAPL_POSITION = {
    "symbol": "AAPL",
    "quantity": "0.4899",
    "avg_entry_price": "102.00",
    "price": "103.10",
    "unrealized_pnl": "0.54",
    "unrealized_pct": "1.08",
    "stop_price": "96.90",
    "target_price": "112.20",
}


def _trade(day: str, symbol: str, pnl: str | None, pct: str | None, closed: bool = True) -> dict:
    return {
        "day": day,
        "run_id": f"daily-{day.replace('-', '')}",
        "symbol": symbol,
        "quantity": "0.4899",
        "entry_price": "102.00",
        "exit_price": "103.10" if closed else None,
        "closed": closed,
        "pnl_usd": pnl,
        "return_pct": pct,
        "exit_via": "exit order" if closed else "still open",
    }


WEDNESDAY_TRADES = [
    _trade("2026-10-14", "AAPL", "0.54", "1.08"),
    _trade("2026-10-14", "MSFT", "0.66", "1.32"),
]


def _report(day: str, status: str | None, trades: list[dict]) -> dict[str, Any]:
    return {
        "day": day,
        "cycle_status": status,
        "cycle_summary": "entered 0.4899 AAPL at $102" if trades else None,
        "trades": trades,
        "benchmark_symbol": "SPY",
        "benchmark_pct": "0.40" if status else None,
        "excess_pct": "0.70" if status else None,
        "analysis": {"decision": "buy", "symbol": "AAPL", "confidence": "medium"}
        if trades
        else None,
        "warnings": [],
    }


class FakeData:
    def __init__(self, **kw: Any) -> None:
        self.overview_data = kw.get("overview", _overview())
        self.reports = kw.get(
            "reports", {"2026-10-14": _report("2026-10-14", "traded", WEDNESDAY_TRADES)}
        )
        self.trade_rows = kw.get("trades", WEDNESDAY_TRADES)
        self.cost_data = kw.get("costs")
        self.plan_data = kw.get("plan")
        self.journal_data = kw.get("journal")
        self.fail = kw.get("fail", False)
        self.asked: list[tuple[str, Any]] = []

    def overview(self) -> dict[str, Any]:
        self.asked.append(("overview", None))
        if self.fail:
            raise RuntimeError("database is locked")
        return self.overview_data

    def day_report(self, day: date) -> dict[str, Any] | None:
        self.asked.append(("report", day))
        return self.reports.get(day.isoformat())

    def trades(self, days: int) -> list[dict[str, Any]]:
        self.asked.append(("trades", days))
        return self.trade_rows

    def costs(self, days: int) -> dict[str, Any]:
        self.asked.append(("costs", days))
        return self.cost_data or {
            "total": {"calls": 12, "llm_usd": "0.42", "unpriced_calls": 0, "total_usd": "0.42"},
            "daily_budget_usd": "2.00",
        }

    def journal(self, day: date) -> dict[str, Any] | None:
        self.asked.append(("journal", day))
        return self.journal_data

    def plan(self, day: date) -> dict[str, Any] | None:
        self.asked.append(("plan", day))
        return self.plan_data


class FakeAI:
    def __init__(self, *outcomes: str | Exception) -> None:
        self.outcomes = list(outcomes) or ["Tuesday had no trade: the critic vetoed it."]
        self.requests: list[LLMRequest] = []

    def __call__(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResponse(
            text=outcome,
            provider="fake",
            model="fake-model",
            stop_reason="end",
            usage=LLMUsage(input_tokens=10, output_tokens=5),
            latency_ms=1,
        )


def _bot(
    tmp_path: Path,
    data: FakeData | None = None,
    ai: FakeAI | None = None,
    about: str = "",
    **config: Any,
) -> tuple[ChatBot, FakeData, FakeAI, PauseStore]:
    data = data or FakeData()
    ai = ai if ai is not None else FakeAI()
    pause = PauseStore(tmp_path / "pause.json")
    bot = ChatBot(
        data,
        pause,
        ChatConfig.model_validate(config),
        complete=ai,
        clock=lambda: SATURDAY,
        timezone="America/New_York",
        about=about,
    )
    return bot, data, ai, pause


# ---------- Free answers ----------


def test_hi_gets_a_greeting_with_the_status(tmp_path: Path) -> None:
    bot, _, ai, _ = _bot(tmp_path)
    reply = bot.reply("Hi")
    assert reply.startswith("Hi! 👋")
    assert "The market is closed right now. It opens Mon at 09:30." in reply
    assert reply.endswith("Ask me anything, or send help to see what I can do.")
    assert "I have no open positions, so everything is in cash." in reply
    assert "Account value: $100,012.34 (+$12.40 today)." in reply
    assert ai.requests == []  # free: no AI


def test_help_lists_what_it_can_do(tmp_path: Path) -> None:
    bot, _, ai, _ = _bot(tmp_path)
    reply = bot.reply("/help")
    for word in ("status", "positions", "report", "trades", "costs", "plan", "pause"):
        assert word in reply
    assert ai.requests == []


def test_start_from_telegram_is_a_greeting(tmp_path: Path) -> None:
    bot, _, _, _ = _bot(tmp_path)
    assert bot.reply("/start").startswith("Hi!")


def test_positions_with_prices_stop_and_target(tmp_path: Path) -> None:
    bot, _, _, _ = _bot(tmp_path, FakeData(overview=_overview(positions=[AAPL_POSITION])))
    reply = bot.reply("what are you holding?")
    assert reply.startswith("Here's what I'm holding:")
    assert "AAPL 0.4899 sh at $102.00, now $103.10 (+$0.54, +1.08%)" in reply
    assert "stop $96.90, target $112.20" in reply


def test_no_positions(tmp_path: Path) -> None:
    bot, _, _, _ = _bot(tmp_path)
    assert bot.reply("positions") == "No open positions right now, so everything is in cash. 💵"


def test_the_report_and_trades_of_a_day(tmp_path: Path) -> None:
    bot, data, ai, _ = _bot(tmp_path)
    reply = bot.reply("hey give me the report of last wednesday and how many trade we did")
    assert ("report", date(2026, 10, 14)) in data.asked
    assert "Wed 2026-10-14: traded" in reply
    assert "2 trades, 2 won, P&L +$1.20" in reply
    assert "AAPL bought at $102.00, sold at $103.10: +$0.54 (+1.08%)" in reply
    assert "SPY +0.40% that day" in reply
    assert ai.requests == []


def test_a_day_without_a_record(tmp_path: Path) -> None:
    bot, _, _, _ = _bot(tmp_path)
    reply = bot.reply("report for 2026-10-11")
    assert reply == (
        "I don't have a trading record for Sun 2026-10-11. The market was closed that day, "
        "or I wasn't running."
    )


def test_report_without_a_day_means_the_last_session(tmp_path: Path) -> None:
    reports = {"2026-10-16": _report("2026-10-16", "no_trade", [])}
    bot, data, _, _ = _bot(tmp_path, FakeData(reports=reports))
    reply = bot.reply("report")
    # Saturday: the last session was Friday.
    assert ("report", date(2026, 10, 16)) in data.asked
    assert "Fri 2026-10-16: no_trade" in reply
    assert "No trades that day." in reply


def test_trades_over_a_span(tmp_path: Path) -> None:
    rows = [*WEDNESDAY_TRADES, _trade("2026-10-15", "NVDA", "-0.90", "-1.80")]
    bot, data, _, _ = _bot(tmp_path, FakeData(trades=rows))
    reply = bot.reply("how many trades this week")
    assert ("trades", 6) in data.asked
    assert "This week: 3 trades (3 closed, 2 won), P&L +$0.30" in reply
    assert "Thu 10-15 NVDA -$0.90 (-1.80%)" in reply


def test_trades_default_to_the_last_week_and_mention_open_ones(tmp_path: Path) -> None:
    rows = [_trade("2026-10-16", "TSLA", None, None, closed=False)]
    bot, data, _, _ = _bot(tmp_path, FakeData(trades=rows))
    reply = bot.reply("trades")
    assert ("trades", 7) in data.asked
    assert "Last 7 days: 1 trade (0 closed, 0 won), P&L +$0.00" in reply
    assert "Fri 10-16 TSLA still open" in reply


def test_no_trades_in_a_span(tmp_path: Path) -> None:
    bot, _, _, _ = _bot(tmp_path, FakeData(trades=[]))
    assert bot.reply("trades last 30 days") == "Last 30 days: no trades yet."


def test_costs(tmp_path: Path) -> None:
    bot, data, _, _ = _bot(tmp_path)
    assert bot.reply("how much did you cost today") == (
        "Costs today: $0.42 (12 AI calls). Daily AI budget: $2.00."
    )
    assert ("costs", 1) in data.asked


def test_costs_without_prices_say_so(tmp_path: Path) -> None:
    unpriced = {
        "total": {"calls": 5, "llm_usd": None, "unpriced_calls": 5, "total_usd": None},
        "daily_budget_usd": "2.00",
    }
    bot, _, _, _ = _bot(tmp_path, FakeData(costs=unpriced))
    reply = bot.reply("costs this week")
    assert "Costs this week: unknown (5 AI calls without a price" in reply
    assert "costs.llm_prices" in reply


def test_the_plan(tmp_path: Path) -> None:
    plan = {
        "source_day": "2026-10-16",
        "for_day": "2026-10-19",
        "items": [
            {"priority": 1, "kind": "ops", "text": "Fix the thing."},
            {"priority": 2, "kind": "watch", "text": "Watch NVDA after earnings."},
        ],
    }
    bot, _, _, _ = _bot(tmp_path, FakeData(plan=plan))
    reply = bot.reply("what's the plan for monday?")
    assert "Plan for Mon 2026-10-19:" in reply
    assert "1. Fix the thing." in reply
    assert "2. Watch NVDA after earnings." in reply


def test_no_plan_yet(tmp_path: Path) -> None:
    bot, _, _, _ = _bot(tmp_path)
    assert bot.reply("plan") == "No plan yet. I write one every evening after the session closes."


# ---------- Nothing in chat can trade ----------


@pytest.mark.parametrize(
    "text", ["buy AAPL", "Sell everything", "close my positions", "short TSLA"]
)
def test_trade_requests_are_refused(tmp_path: Path, text: str) -> None:
    bot, _, ai, _ = _bot(tmp_path)
    reply = bot.reply(text)
    assert reply.startswith("I'd love to help, but I can't place trades from chat.")
    assert "pause" in reply
    assert ai.requests == []


def test_pause_stops_new_trades_for_today(tmp_path: Path) -> None:
    bot, _, _, pause = _bot(tmp_path)
    reply = bot.reply("pause")
    assert reply.startswith("Done ✅ I've paused new trades for the rest of today (Sat 2026-10-17)")
    assert "keep their stop-loss" in reply
    assert pause.paused_reason(TODAY) is not None
    assert pause.paused_reason(TODAY + timedelta(days=1)) is None  # resumes by itself
    assert bot.reply("pause").startswith("Already paused for today")


def test_status_mentions_a_pause(tmp_path: Path) -> None:
    bot, _, _, _ = _bot(tmp_path)
    bot.reply("pause")
    assert "New trades are paused for today" in bot.reply("status")


def test_resume_is_only_possible_on_the_computer(tmp_path: Path) -> None:
    bot, _, _, pause = _bot(tmp_path)
    bot.reply("pause")
    reply = bot.reply("resume")
    assert "trading-agent resume" in reply
    assert pause.paused_reason(TODAY) is not None


# ---------- AI answers ----------


def test_other_questions_go_to_the_ai_with_the_records(tmp_path: Path) -> None:
    bot, _, ai, _ = _bot(tmp_path)
    reply = bot.reply("why didn't you trade on wednesday?")
    assert reply == "Tuesday had no trade: the critic vetoed it."
    (request,) = ai.requests
    assert request.tags == {"agent": "chat"}
    assert "never place, change, or cancel orders" in request.system
    question = request.messages[-1].content
    assert "why didn't you trade on wednesday?" in question
    assert "2026-10-14" in question  # the day it asked about, from the records
    assert "The market is closed right now" in question


def test_the_ai_is_capped_per_day(tmp_path: Path) -> None:
    bot, _, ai, _ = _bot(tmp_path, ai=FakeAI("a", "b", "c"), max_ai_answers_per_day=2)
    assert bot.reply("question one?") == "a"
    assert bot.reply("question two?") == "b"
    capped = bot.reply("question three?")
    assert "2 AI answers today" in capped
    assert "sorry" in capped.lower()
    assert "help" in capped
    assert len(ai.requests) == 2


def test_ai_answers_can_be_switched_off(tmp_path: Path) -> None:
    bot, _, ai, _ = _bot(tmp_path, ai_answers=False)
    reply = bot.reply("why so quiet?")
    assert "I can answer" in reply
    assert ai.requests == []


def test_an_unreachable_ai_falls_back_to_the_free_answers(tmp_path: Path) -> None:
    bot, _, _, _ = _bot(tmp_path, ai=FakeAI(LLMError("daily LLM budget reached")))
    reply = bot.reply("tell me something clever")
    assert "couldn't reach the AI (daily LLM budget reached)" in reply
    assert "I can answer" in reply


def test_long_ai_answers_are_trimmed(tmp_path: Path) -> None:
    bot, _, _, _ = _bot(tmp_path, ai=FakeAI("word " * 2000))
    assert len(bot.reply("essay please")) <= 1500


# ---------- Never crashes ----------


def test_a_data_failure_is_explained_not_raised(tmp_path: Path) -> None:
    bot, _, _, _ = _bot(tmp_path, FakeData(fail=True))
    reply = bot.reply("status")
    assert "couldn't read my records" in reply
    assert "database is locked" in reply


def test_empty_messages_get_the_help(tmp_path: Path) -> None:
    bot, _, _, _ = _bot(tmp_path)
    assert "I can answer" in bot.reply("   ")


# ---------- Pause store ----------


def test_pause_survives_a_restart_and_expires(tmp_path: Path) -> None:
    path = tmp_path / "data" / "pause.json"
    PauseStore(path).pause(TODAY, by="telegram")
    again = PauseStore(path)
    assert again.paused_reason(TODAY) == "paused from telegram for 2026-10-17"
    assert again.paused_reason(TODAY + timedelta(days=1)) is None


def test_resume_clears_the_pause(tmp_path: Path) -> None:
    store = PauseStore(tmp_path / "pause.json")
    store.pause(TODAY, by="telegram")
    assert store.resume() is True
    assert store.paused_reason(TODAY) is None
    assert store.resume() is False


def test_a_damaged_pause_file_does_not_pause(tmp_path: Path) -> None:
    path = tmp_path / "pause.json"
    path.write_text("{not json", encoding="utf-8")
    assert PauseStore(path).paused_reason(TODAY) is None


def test_chat_config_is_validated() -> None:
    with pytest.raises(ValueError, match="max_ai_answers_per_day"):
        ChatConfig.model_validate({"max_ai_answers_per_day": -1})


@pytest.mark.parametrize(
    "text", ["Hii", "hiii!", "heyy", "Hellooo", "hey there", "gm", "Good morning bot"]
)
def test_informal_greetings_are_free(tmp_path: Path, text: str) -> None:
    bot, _, ai, _ = _bot(tmp_path)
    assert bot.reply(text).startswith("Hi!")
    assert ai.requests == []


# ---------- Tone (2026-10-19: the owner found the replies cold) ----------


def test_the_ai_is_asked_to_be_warm(tmp_path: Path) -> None:
    bot, _, ai, _ = _bot(tmp_path)
    bot.reply("how was the day?")
    system = " ".join(ai.requests[0].system.split())
    assert "warm, friendly, and encouraging" in system
    assert "never curt or cold" in system
    assert "never place, change, or cancel orders" in system  # the facts rules stay


def test_failures_apologise_and_say_what_to_do(tmp_path: Path) -> None:
    bot, _, _, _ = _bot(tmp_path, FakeData(fail=True))
    reply = bot.reply("status")
    assert reply.startswith("Sorry, I couldn't read my records just now")
    assert reply.endswith("Please try again in a minute.")


# ---------- Knowing itself (2026-10-19: answers from the owner's phone) ----------

JOURNAL = {"snapshots": 18, "sources": 455, "notes": 1, "notebook": False}


@pytest.mark.parametrize(
    "text",
    [
        "What\u2019s up ?",  # a phone's curly apostrophe and a space before "?"
        "How is everything?",
        "How was everything?",
        "Are you watching the market right ?",
        "are you there",
        "How\u2019s it going",
    ],
)
def test_casual_check_ins_get_the_free_status(tmp_path: Path, text: str) -> None:
    bot, _, ai, _ = _bot(tmp_path)
    assert "The market is closed right now. It opens Mon at 09:30." in bot.reply(text)
    assert ai.requests == []


def test_a_real_question_still_goes_to_the_ai(tmp_path: Path) -> None:
    bot, _, ai, _ = _bot(tmp_path)
    bot.reply("How is everything going with NVDA after the earnings call this week?")
    assert len(ai.requests) == 1


def test_status_says_what_the_journal_recorded_today(tmp_path: Path) -> None:
    bot, _, _, _ = _bot(tmp_path, FakeData(journal=JOURNAL))
    reply = bot.reply("are you watching the market?")
    assert "Today I've recorded 18 market snapshots and 455 news items." in reply


def test_the_ai_knows_its_schedule_and_its_day(tmp_path: Path) -> None:
    overview = _overview(
        cycle={
            "status": "running",
            "steps": [
                {
                    "step": "analysis",
                    "status": "no_trade",
                    "detail": "the analyst chose not to trade",
                }
            ],
        },
        next_step={"step": "entry", "at": "2026-10-19T09:45:00-04:00"},
    )
    bot, _, ai, _ = _bot(
        tmp_path,
        FakeData(overview=overview, journal=JOURNAL),
        about="My day (New York time): 04:00 I start watching and recording.",
    )
    bot.reply("why didn't you start at 4 am?")
    content = ai.requests[0].messages[0].content
    assert "My day (New York time): 04:00 I start watching and recording." in content
    assert "the analyst chose not to trade" in content
    assert "2026-10-19T09:45:00-04:00" in content
    assert '"snapshots": 18' in content
    system = " ".join(ai.requests[0].system.split())
    assert "Speak as the bot itself, in the first person" in system
    assert "you are running" in system


def test_about_describes_the_configured_day() -> None:
    from trading_agent.chat.bot import about_text
    from trading_agent.settings.models import AppConfig

    text = about_text(
        AppConfig.model_validate(
            {
                "risk": {"max_trades_per_day": 3},
                "schedule": {"extra_analyses": ["10:30", "12:00"]},
            }
        )
    )
    for piece in ("04:00", "08:45", "09:45", "10:30", "12:00", "15:50", "20:20", "20:30"):
        assert piece in text, piece
    assert "watch only" in text
    single = about_text(AppConfig())  # one trade a day: no extra slots
    assert "10:30" not in single


def test_journal_summary_counts_the_days_files(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from trading_agent.chat.records import journal_summary
    from trading_agent.journal.models import JournalSnapshot, SessionRecord
    from trading_agent.journal.store import JournalStore

    store = JournalStore(tmp_path / "journal")
    assert journal_summary(store, TODAY) is None
    stamp = datetime(2026, 10, 17, 12, tzinfo=UTC)
    store.save(
        SessionRecord(
            day=TODAY,
            session="pre",
            opens=stamp,
            closes=stamp,
            snapshots=[JournalSnapshot(at=stamp), JournalSnapshot(at=stamp)],
        )
    )
    assert journal_summary(store, TODAY) == {
        "snapshots": 2,
        "sources": 0,
        "notes": 0,
        "notebook": False,
    }


# ---------- Talking like a person (2026-10-19, second round of screenshots) ----------


@pytest.mark.parametrize(
    ("text", "opener"),
    [
        ("How are you ?", "I'm doing well, thanks for asking! 😊"),
        ("How\u2019s it going", "I'm doing well, thanks for asking! 😊"),
        ("What\u2019s up ?", "Not much, just keeping an eye on the market. 👀"),
        ("How is everything?", "Everything's running smoothly. 👍"),
        ("Are you watching the market right ?", "Yes, I'm here and watching. 👀"),
        ("are you there", "Yes, I'm here and watching. 👀"),
        ("status", "Here's where things stand:"),
    ],
)
def test_check_ins_open_like_a_conversation(tmp_path: Path, text: str, opener: str) -> None:
    bot, _, _, _ = _bot(tmp_path)
    reply = bot.reply(text)
    assert reply.splitlines()[0] == opener


def test_the_status_reads_as_short_lines(tmp_path: Path) -> None:
    bot, _, _, _ = _bot(tmp_path, FakeData(journal=JOURNAL))
    assert bot.reply("how are you?").splitlines() == [
        "I'm doing well, thanks for asking! 😊",
        "The market is closed right now. It opens Mon at 09:30.",
        "I have no open positions, so everything is in cash.",
        "Account value: $100,012.34 (+$12.40 today).",
        "Today I've recorded 18 market snapshots and 455 news items.",
    ]


def test_an_unchanged_day_says_so_in_words(tmp_path: Path) -> None:
    flat = _overview(account={"equity": "99999.94", "day_change": "0.00"})
    bot, _, _, _ = _bot(tmp_path, FakeData(overview=flat))
    assert "Account value: $99,999.94 (unchanged today)." in bot.reply("status")


def test_an_open_market_and_a_holding(tmp_path: Path) -> None:
    view = _overview(
        market={
            "is_open": True,
            "next_open": "2026-10-19T13:30:00+00:00",
            "next_close": "2026-10-19T20:00:00+00:00",
        },
        positions=[AAPL_POSITION],
    )
    bot, _, _, _ = _bot(tmp_path, FakeData(overview=view))
    reply = bot.reply("status")
    assert "The market is open until 16:00." in reply
    assert "I'm holding 1 position: AAPL." in reply
