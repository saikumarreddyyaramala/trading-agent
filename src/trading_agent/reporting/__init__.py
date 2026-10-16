"""Daily reports: trade results, benchmark comparison, and delivery."""

from trading_agent.reporting.delivery import DeliveryError, post_webhook, save_report
from trading_agent.reporting.performance import Performance, TradeResult, trades_from_orders
from trading_agent.reporting.report import DailyReport, ReportBuilder, render_markdown

__all__ = [
    "DailyReport",
    "DeliveryError",
    "Performance",
    "ReportBuilder",
    "TradeResult",
    "post_webhook",
    "render_markdown",
    "save_report",
    "trades_from_orders",
]
