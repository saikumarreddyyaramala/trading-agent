"""Where the "why" comes from (ADR 0023): company news and free official sources.

Written first. Every source is untrusted third-party text, fetched read-only, and may fail
on its own without stopping the others. Hostile XML must be refused, not parsed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from tests.conftest import NOW, FakeNews, make_article
from trading_agent.journal.sources import (
    AlpacaNewsSource,
    FederalRegisterSource,
    RssSource,
    SourceError,
    fetch_all,
)

FED_RSS = """<?xml version="1.0" encoding="utf-8" ?>
<rss version="2.0"><channel><title>FRB: Press Release - All Releases</title>
<item>
  <title>Federal Reserve issues FOMC statement</title>
  <link><![CDATA[https://www.federalreserve.gov/newsevents/pressreleases/monetary20261015a.htm]]></link>
  <description><![CDATA[<p>The Committee decided to <b>lower</b> rates.</p>]]></description>
  <pubDate>Thu, 15 Oct 2026 18:00:00 GMT</pubDate>
  <guid>monetary20261015a</guid>
</item>
<item>
  <title>Old release</title>
  <link>https://www.federalreserve.gov/old.htm</link>
  <pubDate>Mon, 05 Oct 2026 14:00:00 GMT</pubDate>
</item>
<item>
  <title>Undated release</title>
  <link>https://www.federalreserve.gov/undated.htm</link>
</item>
</channel></rss>"""

XML_BOMB = """<?xml version="1.0"?>
<!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;">]>
<rss><channel><item><title>&lol2;</title></item></channel></rss>"""


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _serve(body: str, status: int = 200) -> httpx.Client:
    return _client(lambda request: httpx.Response(status, text=body))


# ---------- RSS ----------


def test_rss_items_since_a_time_with_plain_text() -> None:
    source = RssSource(
        "federal_reserve", "https://fed.example/rss", "central_bank", client=_serve(FED_RSS)
    )
    items = source.fetch(NOW - timedelta(days=2))
    titles = [i.title for i in items]
    assert "Federal Reserve issues FOMC statement" in titles
    assert "Old release" not in titles  # before `since`
    assert "Undated release" in titles  # no date: kept, the journal de-duplicates by id
    fomc = items[titles.index("Federal Reserve issues FOMC statement")]
    assert fomc.kind == "central_bank"
    assert fomc.source == "federal_reserve"
    assert fomc.summary == "The Committee decided to lower rates."  # markup removed
    assert fomc.published_at == datetime(2026, 10, 15, 18, 0, tzinfo=UTC)
    assert (
        fomc.url == "https://www.federalreserve.gov/newsevents/pressreleases/monetary20261015a.htm"
    )
    assert fomc.id.startswith("federal_reserve:")


def test_rss_ids_are_stable_between_fetches() -> None:
    first = RssSource("fed", "https://x", "central_bank", client=_serve(FED_RSS)).fetch(
        NOW - timedelta(days=30)
    )
    again = RssSource("fed", "https://x", "central_bank", client=_serve(FED_RSS)).fetch(
        NOW - timedelta(days=30)
    )
    assert [i.id for i in first] == [i.id for i in again]
    assert len({i.id for i in first}) == len(first)


def test_hostile_xml_is_refused() -> None:
    source = RssSource("fed", "https://x", "central_bank", client=_serve(XML_BOMB))
    with pytest.raises(SourceError, match="could not read fed"):
        source.fetch(NOW - timedelta(days=1))


def test_rss_http_errors_are_source_errors() -> None:
    source = RssSource("bls", "https://x", "economic_data", client=_serve("", status=403))
    with pytest.raises(SourceError, match="bls: HTTP 403"):
        source.fetch(NOW)


def test_rss_sends_a_user_agent() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=FED_RSS)

    RssSource("bls", "https://x", "economic_data", client=_client(handler)).fetch(NOW)
    assert "trading-agent" in seen[0].headers["user-agent"]


def test_long_text_is_cut() -> None:
    long = FED_RSS.replace("The Committee decided", "word " * 400)
    (item, *_) = RssSource("fed", "https://x", "central_bank", client=_serve(long)).fetch(
        NOW - timedelta(days=2)
    )
    assert len(item.summary) <= 500


# ---------- Federal Register ----------


def _register(handler_log: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        handler_log.append(request)
        if "PRESDOCU" in str(request.url):
            results = [
                {
                    "document_number": "2026-21432",
                    "title": "Adjusting Imports of Semiconductors",
                    "type": "Presidential Document",
                    "abstract": None,
                    "html_url": "https://www.federalregister.gov/d/2026-21432",
                    "publication_date": "2026-10-15",
                    "agencies": [{"name": "Executive Office of the President"}],
                }
            ]
        else:
            results = [
                {
                    "document_number": "2026-30001",
                    "title": "Capital Requirements for Large Banks",
                    "type": "Rule",
                    "abstract": "The agencies adopt a final rule.",
                    "html_url": "https://www.federalregister.gov/d/2026-30001",
                    "publication_date": "2026-10-15",
                    "agencies": [{"name": "Federal Reserve System"}],
                }
            ]
        return httpx.Response(200, json={"results": results})

    return _client(handler)


def test_federal_register_presidential_documents_and_significant_rules() -> None:
    log: list[httpx.Request] = []
    items = FederalRegisterSource(client=_register(log)).fetch(NOW - timedelta(hours=10))
    assert {i.title for i in items} == {
        "Adjusting Imports of Semiconductors",
        "Capital Requirements for Large Banks",
    }
    order = next(i for i in items if "Semiconductors" in i.title)
    assert order.id == "federal_register:2026-21432"
    assert order.kind == "government"
    assert "Executive Office of the President" in order.summary
    queries = [str(r.url) for r in log]
    assert any("PRESDOCU" in q for q in queries)
    assert any("RULE" in q and "significant" in q for q in queries)
    assert all("2026-10-15" in q for q in queries)  # published since the `since` day


def test_federal_register_errors_are_source_errors() -> None:
    source = FederalRegisterSource(client=_client(lambda r: httpx.Response(500)))
    with pytest.raises(SourceError, match="federal_register: HTTP 500"):
        source.fetch(NOW)


def test_federal_register_bad_json_is_a_source_error() -> None:
    source = FederalRegisterSource(client=_serve("not json"))
    with pytest.raises(SourceError, match="federal_register"):
        source.fetch(NOW)


# ---------- Company news ----------


def test_company_news_covers_the_market_and_the_watched_symbols() -> None:
    news = FakeNews([make_article("7", ("NVDA",)), make_article("8", ())])
    source = AlpacaNewsSource(news, lambda: ["NVDA", "AAPL"])
    items = source.fetch(NOW - timedelta(hours=6))
    assert {i.id for i in items} == {"news:7", "news:8"}  # de-duplicated across both calls
    assert [c["symbols"] for c in news.calls] == [(), ("NVDA", "AAPL")]
    nvda = next(i for i in items if i.id == "news:7")
    assert nvda.kind == "company_news"
    assert nvda.symbols == ("NVDA",)
    assert nvda.title == "Headline 7"


def test_company_news_failure_is_a_source_error() -> None:
    with pytest.raises(SourceError, match="company_news"):
        AlpacaNewsSource(FakeNews(fail=True), lambda: []).fetch(NOW)


# ---------- All sources ----------


def test_one_failing_source_does_not_stop_the_others() -> None:
    good = RssSource("fed", "https://x", "central_bank", client=_serve(FED_RSS))
    bad = RssSource("bls", "https://x", "economic_data", client=_serve("", status=503))
    items, errors = fetch_all([bad, good], NOW - timedelta(days=2))
    assert items
    assert errors == ["bls: HTTP 503"]


def test_an_unexpected_crash_in_a_source_is_contained() -> None:
    class _Broken:
        name = "broken"

        def fetch(self, since: datetime) -> list[Any]:
            raise RuntimeError("bug")

    items, errors = fetch_all([_Broken()], NOW)
    assert items == []
    assert errors == ["broken: RuntimeError: bug"]


def test_items_are_json_ready() -> None:
    (item, *_) = RssSource("fed", "https://x", "central_bank", client=_serve(FED_RSS)).fetch(
        NOW - timedelta(days=2)
    )
    assert json.loads(item.model_dump_json())["source"] == "fed"
