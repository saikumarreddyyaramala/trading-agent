"""The dashboard's HTTP API (ADR 0018): GET only, JSON, local by default.

/api/overview        account, day change, positions, market clock, today's cycle
/api/history?span=   equity chart: 1D, 1W, 1M, 3M, 1Y
/api/trades?days=    system trades; /api/performance?days= their summary
/api/news?limit=     news cards (rule-based key facts)
/api/watch?date=     pre-market and after-hours scans
/api/plan?date=      the plan for a day and the one made at its end
/api/learning        principles, lessons, playbooks, guidebook
/api/costs?days=     the cost ledger, services, and the budget
/api/activity?limit= steps, orders, exits, lessons, guidebook editions
/api/runs/{run_id}   everything recorded for one run
/api/stream          live updates (server-sent events)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from typing import Any

import anyio
from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from trading_agent.dashboard.data import DashboardData, stream

STREAM_INTERVAL_SECONDS = 3

HOME = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Trading agent</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 40rem; margin: 3rem auto">
<h1>Trading agent dashboard</h1>
<p>The interface arrives in Phase 18. The read-only API is live:</p>
<ul>
<li><a href="/api/overview">/api/overview</a></li>
<li><a href="/api/history?span=1D">/api/history?span=1D</a></li>
<li><a href="/api/news">/api/news</a></li>
<li><a href="/api/costs">/api/costs</a></li>
<li><a href="/api/learning">/api/learning</a></li>
<li><a href="/api/activity">/api/activity</a></li>
</ul></body></html>"""


class _BadRequest(ValueError):
    pass


def _int(request: Request, name: str, default: int, low: int, high: int) -> int:
    raw = request.query_params.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise _BadRequest(f"{name} must be a whole number") from exc
    if not low <= value <= high:
        raise _BadRequest(f"{name} must be between {low} and {high}")
    return value


def _day(request: Request, data: DashboardData) -> date:
    raw = request.query_params.get("date")
    if raw is None:
        return data._today()
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise _BadRequest("date must be YYYY-MM-DD") from exc


class _Headers(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["x-content-type-options"] = "nosniff"
        response.headers.setdefault("cache-control", "no-store")
        response.headers["referrer-policy"] = "no-referrer"
        return response


def create_app(data: DashboardData, *, stream_ticks: int | None = None) -> Starlette:
    def endpoint(fn: Callable[[Request], Any]) -> Callable[[Request], Any]:
        async def handle(request: Request) -> Response:
            try:
                result = await anyio.to_thread.run_sync(lambda: fn(request))
            except (_BadRequest, ValueError) as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            if result is None:
                return JSONResponse({"error": "not found"}, status_code=404)
            return JSONResponse(result)

        return handle

    async def home(request: Request) -> Response:
        return HTMLResponse(HOME)

    async def live(request: Request) -> Response:
        async def events() -> Any:
            async for event in stream(data, interval=STREAM_INTERVAL_SECONDS, ticks=stream_ticks):
                yield {"event": event["event"], "data": json.dumps(event["data"])}

        return EventSourceResponse(events(), ping=15)

    routes = [
        Route("/", home),
        Route(
            "/api/health",
            endpoint(lambda r: {"status": "ok", "environment": data._config.environment.value}),
        ),
        Route("/api/overview", endpoint(lambda r: data.overview())),
        Route("/api/history", endpoint(lambda r: data.history(r.query_params.get("span", "1D")))),
        Route("/api/trades", endpoint(lambda r: data.trades(_int(r, "days", 30, 1, 3650)))),
        Route(
            "/api/performance",
            endpoint(lambda r: data.performance(_int(r, "days", 365, 1, 3650))),
        ),
        Route("/api/news", endpoint(lambda r: data.news(_int(r, "limit", 30, 1, 100)))),
        Route("/api/watch", endpoint(lambda r: data.watch(_day(r, data)))),
        Route("/api/plan", endpoint(lambda r: data.plan(_day(r, data)))),
        Route("/api/learning", endpoint(lambda r: data.learning())),
        Route("/api/costs", endpoint(lambda r: data.costs(_int(r, "days", 30, 1, 366)))),
        Route("/api/activity", endpoint(lambda r: data.activity(_int(r, "limit", 50, 1, 500)))),
        Route("/api/runs/{run_id}", endpoint(lambda r: data.run(r.path_params["run_id"]))),
        Route("/api/stream", live),
    ]
    return Starlette(routes=routes, middleware=[Middleware(_Headers)])
