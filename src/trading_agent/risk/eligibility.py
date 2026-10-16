"""Which stocks the system may trade at all. Shared by the risk engine and candidate search,
so a stock the analyst is shown is one the risk engine could approve.

Each rule returns a list of human-readable issues; empty means eligible.
"""

from __future__ import annotations

import re
from decimal import Decimal

from trading_agent.domain.models import Asset
from trading_agent.settings.models import RiskConfig

# Funds, ETFs/ETNs, and leveraged or inverse products, recognised by name. Deliberately narrow:
# words such as "Trust" or "Ultra" also appear in operating companies (Northern Trust, Ultra
# Clean Holdings). Most ETFs are also excluded by exchange, since they list on NYSE Arca.
_FUND_NAME = re.compile(
    r"\b(ETF|ETN|ETP|Fund|Leveraged|Inverse|UltraPro|UltraShort)\b|\b\d+(\.\d+)?x\b",
    re.IGNORECASE,
)


def looks_like_fund(name: str | None) -> bool:
    return bool(name and _FUND_NAME.search(name))


def asset_issues(asset: Asset, risk: RiskConfig) -> list[str]:
    issues: list[str] = []
    if not asset.tradable:
        issues.append("not tradable at the broker")
    if asset.status.lower() != "active":
        issues.append(f"asset status is {asset.status}")
    if asset.exchange not in risk.allowed_exchanges:
        allowed = ", ".join(risk.allowed_exchanges)
        issues.append(f"listed on {asset.exchange or 'unknown exchange'} (allowed: {allowed})")
    if risk.exclude_funds and looks_like_fund(asset.name):
        issues.append(f"looks like a fund, ETF, or leveraged product ({asset.name})")
    return issues


def price_issues(price: Decimal, risk: RiskConfig) -> list[str]:
    if price < risk.min_price_usd:
        return [f"price ${price} is below the ${risk.min_price_usd} minimum"]
    return []


def liquidity_issues(avg_daily_volume: int, risk: RiskConfig) -> list[str]:
    if avg_daily_volume < risk.min_avg_daily_volume:
        return [
            f"average daily volume {avg_daily_volume:,} is below "
            f"the {risk.min_avg_daily_volume:,} minimum"
        ]
    return []
