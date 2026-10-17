"""The analyst agent: research with read-only MCP tools, then propose one trade or none.

The agent reaches market data only as an MCP client of the research server, so the tool
allowlist, input bounds, call budget, and audit log from Phase 2 all apply to it.

Fail safe: every way a run can go wrong (model unavailable, turn or token limit, an answer
that fails validation or evidence checks after the allowed repairs) ends in status "failed",
which callers treat exactly like no trade. This class never raises for those cases.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

import anyio
from mcp import Client
from mcp.server import MCPServer
from mcp_types import CallToolResult, TextContent
from pydantic import BaseModel, ConfigDict

from trading_agent.agents.evidence import (
    ProposalError,
    ResearchLedger,
    parse_proposal,
    verify_proposal,
)
from trading_agent.agents.prompts import PROMPT_VERSION, repair_prompt, system_prompt, task_prompt
from trading_agent.domain.models import (
    ChatMessage,
    Lesson,
    LLMRequest,
    LLMUsage,
    ToolCall,
    ToolResult,
    ToolSpec,
    TradeProposal,
)
from trading_agent.llm.errors import LLMError
from trading_agent.llm.gateway import LLMGateway
from trading_agent.observability.logging import get_logger
from trading_agent.settings.models import AppConfig

# Keeps one oversized tool result from flooding the model's context window.
MAX_TOOL_RESULT_CHARS = 20_000

RunStatus = Literal["proposed", "no_trade", "failed"]


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str
    tool: str
    arguments: dict[str, Any] | None
    is_error: bool
    # Exactly what the model was shown (truncated like it was). Lets a reviewer check claims.
    result: str | None = None


class AnalystResult(BaseModel):
    """Outcome of one run. Only status "proposed" can ever lead to an order."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: RunStatus
    reason: str
    proposal: TradeProposal | None
    prompt_version: str
    turns: int
    tool_calls: tuple[ToolCallRecord, ...]
    usage: LLMUsage
    started_at: datetime
    duration_seconds: float
    # The lessons the analyst had in context, for judging whether learning helps.
    lesson_ids: tuple[int, ...] = ()


def _result_text(result: CallToolResult) -> str:
    text = "\n".join(b.text for b in result.content if isinstance(b, TextContent))
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + "\n[truncated]"
    return text


def _arguments(call: ToolCall) -> dict[str, Any] | None:
    try:
        value = json.loads(call.arguments_json or "{}")
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class AnalystAgent:
    def __init__(
        self,
        gateway: LLMGateway,
        server: MCPServer,
        config: AppConfig,
        *,
        clock: Callable[[], datetime] | None = None,
        logger: Any = None,
        lessons: Sequence[Lesson] = (),
    ) -> None:
        self._lessons = tuple(lessons)
        self._gateway = gateway
        self._server = server
        self._config = config
        self._limits = config.analyst
        self._clock = clock or (lambda: datetime.now(UTC))
        self._log = logger or get_logger(__name__)

    async def run(self, run_id: str | None = None, context: str | None = None) -> AnalystResult:
        """One research run. `run_id` ties it to a larger run, such as the daily cycle."""
        run_id = run_id or uuid.uuid4().hex[:12]
        started_at = self._clock()
        started = time.perf_counter()
        log = self._log.bind(run_id=run_id, agent="analyst")
        ledger = ResearchLedger()
        records: list[ToolCallRecord] = []
        turns = 0

        def finish(status: RunStatus, reason: str, proposal: TradeProposal | None) -> AnalystResult:
            result = AnalystResult(
                run_id=run_id,
                status=status,
                reason=reason,
                proposal=proposal,
                prompt_version=PROMPT_VERSION,
                lesson_ids=tuple(lesson.id for lesson in self._lessons),
                turns=turns,
                tool_calls=tuple(records),
                usage=self._gateway.total_usage,
                started_at=started_at,
                duration_seconds=round(time.perf_counter() - started, 2),
            )
            log.info(
                "analyst.finished",
                status=status,
                reason=reason,
                symbol=proposal.symbol if proposal else None,
                turns=turns,
                tool_calls=len(records),
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
            )
            return result

        async with Client(self._server) as client:
            listed = await client.list_tools()
            tools = tuple(
                ToolSpec(name=t.name, description=t.description or "", input_schema=t.input_schema)
                for t in listed.tools
            )
            system = system_prompt(self._config, self._lessons)
            messages = [
                ChatMessage(
                    role="user",
                    content=task_prompt(started_at, self._config.schedule.timezone, context),
                )
            ]
            repairs_left = self._limits.repair_attempts

            while turns < self._limits.max_turns:
                usage = self._gateway.total_usage
                if usage.input_tokens + usage.output_tokens >= self._limits.max_total_tokens:
                    return finish("failed", "token budget for the run is spent", None)

                turns += 1
                request = LLMRequest(
                    system=system,
                    messages=tuple(messages),
                    tools=tools,
                    tags={"run_id": run_id, "agent": "analyst", "turn": str(turns)},
                )
                try:
                    response = await anyio.to_thread.run_sync(self._gateway.complete, request)
                except LLMError as exc:
                    return finish("failed", f"model unavailable: {exc}", None)

                calls = response.tool_calls[: self._limits.max_tool_calls_per_turn]
                messages.append(
                    ChatMessage(role="assistant", content=response.text, tool_calls=calls)
                )
                if calls:
                    results = [await self._execute(client, call, ledger, records) for call in calls]
                    messages.append(ChatMessage(role="user", tool_results=tuple(results)))
                    continue

                try:
                    proposal = parse_proposal(response.text)
                    problems = verify_proposal(
                        proposal,
                        ledger,
                        max_price_deviation_pct=self._config.risk.max_price_deviation_pct,
                    )
                except ProposalError as exc:
                    proposal, problems = None, [str(exc)]

                if proposal is not None and not problems:
                    if proposal.decision == "buy":
                        return finish("proposed", "evidence verified", proposal)
                    return finish("no_trade", "the analyst chose not to trade", proposal)

                log.warning("analyst.answer_rejected", problems=problems, repairs_left=repairs_left)
                if repairs_left == 0:
                    return finish("failed", "answer rejected: " + "; ".join(problems), None)
                repairs_left -= 1
                messages.append(ChatMessage(role="user", content=repair_prompt(problems)))

        return finish("failed", f"turn limit of {self._limits.max_turns} reached", None)

    async def _execute(
        self,
        client: Client,
        call: ToolCall,
        ledger: ResearchLedger,
        records: list[ToolCallRecord],
    ) -> ToolResult:
        arguments = _arguments(call)
        if arguments is None:
            message = "arguments must be a JSON object"
            records.append(
                ToolCallRecord(
                    call_id=call.id, tool=call.name, arguments=None, is_error=True, result=message
                )
            )
            return ToolResult(
                call_id=call.id, content=f"call_id={call.id}\n{message}", is_error=True
            )

        result = await client.call_tool(call.name, arguments)
        text = _result_text(result)
        records.append(
            ToolCallRecord(
                call_id=call.id,
                tool=call.name,
                arguments=arguments,
                is_error=result.is_error,
                result=text,
            )
        )
        if not result.is_error:
            ledger.record(call.id, call.name, arguments, result.structured_content)
        return ToolResult(
            call_id=call.id,
            content=f"call_id={call.id}\n{text}",
            is_error=result.is_error,
        )
