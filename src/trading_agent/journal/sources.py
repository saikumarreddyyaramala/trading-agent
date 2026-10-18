"""Where the journal's "why" comes from (ADR 0023): company news and free official sources.

- Company news: Alpaca's news feed (Benzinga), for the whole market and the watched stocks.
- Federal Reserve press releases (RSS): rates, policy statements, bank rules.
- U.S. Bureau of Labor Statistics latest numbers (RSS): jobs, inflation, wages.
- The Federal Register (JSON API): presidential documents and significant rules, which is
  where "the government changed a policy" shows up.

Everything here is untrusted third-party text, kept as data. XML is parsed with defusedxml,
so a hostile feed (entity expansion and the like) is refused instead of parsed. Each source
fails on its own; `fetch_all` never raises. A paid source later is one more class with a
`fetch(since)` method.
"""

from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Literal, Protocol
from zoneinfo import ZoneInfo

import defusedxml.ElementTree as SafeET
import httpx
from defusedxml import DefusedXmlException
from pydantic import BaseModel, ConfigDict

from trading_agent.domain.ports import NewsPort

Kind = Literal["company_news", "central_bank", "economic_data", "government"]
USER_AGENT = "trading-agent/1.0 (personal market research; read-only)"
TIMEOUT_SECONDS = 15.0
MAX_TITLE = 300
MAX_SUMMARY = 500
FEDERAL_REGISTER_API = "https://www.federalregister.gov/api/v1/documents.json"
MARKET_ZONE = ZoneInfo("America/New_York")

FEEDS: dict[str, tuple[str, Kind]] = {
    "federal_reserve": ("https://www.federalreserve.gov/feeds/press_all.xml", "central_bank"),
    "bls": ("https://www.bls.gov/feed/bls_latest.rss", "economic_data"),
}


class SourceError(RuntimeError):
    """One source could not be read. The others carry on."""


class SourceItem(BaseModel):
    """One piece of evidence. Title and summary are untrusted third-party text."""

    model_config = ConfigDict(frozen=True)

    id: str
    source: str
    kind: Kind
    title: str
    summary: str
    url: str | None
    published_at: datetime | None
    symbols: tuple[str, ...] = ()


class Source(Protocol):
    name: str

    def fetch(self, since: datetime) -> list[SourceItem]: ...


def plain(text: str | None, limit: int) -> str:
    """Markup removed, whitespace collapsed, cut to `limit` characters."""
    if not text:
        return ""
    clean = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", text)).split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def _id(source: str, key: str) -> str:
    return f"{source}:{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}"


def _get(client: Any, name: str, url: str, params: Any = None) -> httpx.Response:
    http = client or httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=True)
    try:
        response = http.get(url, params=params, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        return response
    except httpx.HTTPStatusError as exc:
        raise SourceError(f"{name}: HTTP {exc.response.status_code}") from None
    except httpx.HTTPError as exc:
        raise SourceError(f"{name}: unreachable ({type(exc).__name__})") from None
    finally:
        if client is None:
            http.close()


def _newest_first(items: list[SourceItem], limit: int) -> list[SourceItem]:
    oldest = datetime.min.replace(tzinfo=UTC)
    return sorted(items, key=lambda i: i.published_at or oldest, reverse=True)[:limit]


class RssSource:
    def __init__(
        self, name: str, url: str, kind: Kind, *, client: Any = None, max_items: int = 20
    ) -> None:
        self.name = name
        self._url = url
        self._kind = kind
        self._client = client
        self._max = max_items

    def fetch(self, since: datetime) -> list[SourceItem]:
        response = _get(self._client, self.name, self._url)
        try:
            root = SafeET.fromstring(response.content)
        except (SafeET.ParseError, DefusedXmlException, ValueError) as exc:
            raise SourceError(f"could not read {self.name}: {type(exc).__name__}") from None
        items = []
        for node in root.iter("item"):
            title = plain(node.findtext("title"), MAX_TITLE)
            link = (node.findtext("link") or "").strip() or None
            if not title:
                continue
            published = None
            raw_date = node.findtext("pubDate")
            if raw_date:
                try:
                    published = parsedate_to_datetime(raw_date).astimezone(UTC)
                except (TypeError, ValueError):
                    published = None
            if published is not None and published < since:
                continue
            key = (node.findtext("guid") or "").strip() or link or title
            items.append(
                SourceItem(
                    id=_id(self.name, key),
                    source=self.name,
                    kind=self._kind,
                    title=title,
                    summary=plain(node.findtext("description"), MAX_SUMMARY),
                    url=link,
                    published_at=published,
                )
            )
        return _newest_first(items, self._max)


class FederalRegisterSource:
    """Presidential documents (executive orders, proclamations) and significant rules."""

    name = "federal_register"
    _FIELDS = ("document_number", "title", "type", "abstract", "html_url", "publication_date",
               "agencies")  # fmt: skip

    def __init__(self, *, client: Any = None, max_items: int = 20) -> None:
        self._client = client
        self._max = max_items

    def _query(self, since_day: date, extra: list[tuple[str, str]]) -> list[dict[str, Any]]:
        params = [
            ("per_page", str(self._max)),
            ("order", "newest"),
            ("conditions[publication_date][gte]", since_day.isoformat()),
            *extra,
            *(("fields[]", field) for field in self._FIELDS),
        ]
        response = _get(self._client, self.name, FEDERAL_REGISTER_API, params)
        try:
            results = response.json().get("results") or []
        except ValueError:
            raise SourceError(f"{self.name}: the answer was not JSON") from None
        return [r for r in results if isinstance(r, dict)]

    def fetch(self, since: datetime) -> list[SourceItem]:
        since_day = since.astimezone(MARKET_ZONE).date()
        documents = self._query(since_day, [("conditions[type][]", "PRESDOCU")])
        documents += self._query(
            since_day, [("conditions[type][]", "RULE"), ("conditions[significant]", "1")]
        )
        items: dict[str, SourceItem] = {}
        for doc in documents:
            number = str(doc.get("document_number") or "")
            title = plain(doc.get("title"), MAX_TITLE)
            if not number or not title:
                continue
            agencies = ", ".join(
                a["name"]
                for a in doc.get("agencies") or []
                if isinstance(a, dict) and a.get("name")
            )
            about = plain(doc.get("abstract"), MAX_SUMMARY - 120)
            summary = " ".join(
                part
                for part in (f"{doc.get('type') or 'Document'} by {agencies or 'unknown'}.", about)
                if part
            )
            published = None
            try:
                day = date.fromisoformat(str(doc.get("publication_date")))
                published = datetime.combine(day, datetime.min.time(), tzinfo=MARKET_ZONE)
            except ValueError:
                pass
            items[number] = SourceItem(
                id=f"{self.name}:{number}",
                source=self.name,
                kind="government",
                title=title,
                summary=plain(summary, MAX_SUMMARY),
                url=doc.get("html_url"),
                published_at=published,
            )
        return _newest_first(list(items.values()), self._max)


class AlpacaNewsSource:
    """Company news for the whole market, then for the watched stocks."""

    name = "company_news"

    def __init__(
        self, news: NewsPort, symbols: Callable[[], Sequence[str]], *, max_items: int = 30
    ) -> None:
        self._news = news
        self._symbols = symbols
        self._max = max_items

    def fetch(self, since: datetime) -> list[SourceItem]:
        try:
            articles = list(self._news.get_news([], since=since, limit=self._max))
            symbols = list(self._symbols())
            if symbols:
                articles += self._news.get_news(symbols, since=since, limit=self._max)
        except Exception as exc:
            raise SourceError(f"{self.name}: {exc}") from exc
        items: dict[str, SourceItem] = {}
        for article in articles:
            items.setdefault(
                article.id,
                SourceItem(
                    id=f"news:{article.id}",
                    source=article.source or self.name,
                    kind="company_news",
                    title=plain(article.headline, MAX_TITLE),
                    summary=plain(article.summary, MAX_SUMMARY),
                    url=article.url,
                    published_at=article.created_at,
                    symbols=tuple(article.symbols),
                ),
            )
        return _newest_first(list(items.values()), self._max * 2)


def fetch_all(sources: Sequence[Source], since: datetime) -> tuple[list[SourceItem], list[str]]:
    """Every source's items since `since`, and one error line per source that failed."""
    items: list[SourceItem] = []
    errors: list[str] = []
    for source in sources:
        try:
            items += source.fetch(since)
        except SourceError as exc:
            errors.append(str(exc))
        except Exception as exc:  # one broken source never stops the journal
            errors.append(f"{source.name}: {type(exc).__name__}: {exc}")
    return items, errors
