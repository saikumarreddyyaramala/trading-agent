"""The dashboard interface (ADR 0019): served safely, self-contained, and consistent."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from tests.unit.test_dashboard import _data
from trading_agent.dashboard.app import STATIC, create_app
from trading_agent.storage.store import NullAuditStore

SCRIPTS = sorted(STATIC.rglob("*.js"))


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(_data(NullAuditStore()), stream_ticks=1))


def test_the_page_is_served_with_a_strict_content_policy(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    policy = response.headers["content-security-policy"]
    assert "default-src 'self'" in policy
    assert "script-src 'self'" in policy
    assert "'unsafe-inline'" not in policy.split("style-src")[0]  # no inline scripts
    assert "frame-ancestors 'none'" in policy
    assert response.headers["x-frame-options"] == "DENY"
    assert '<script type="module" src="/static/app.js">' in response.text


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("/static/app.js", "javascript"),
        ("/static/app.css", "text/css"),
        ("/static/lib/format.js", "javascript"),
        ("/static/lib/chart.js", "javascript"),
        ("/static/lib/workflow.js", "javascript"),
        ("/static/lib/markdown.js", "javascript"),
        ("/static/favicon.svg", "image/svg+xml"),
    ],
)
def test_assets_are_served(client: TestClient, path: str, kind: str) -> None:
    response = client.get(path)
    assert response.status_code == 200, path
    assert kind in response.headers["content-type"]


def test_unknown_assets_are_404(client: TestClient) -> None:
    assert client.get("/static/nope.js").status_code == 404
    assert client.get("/static/../app.py").status_code == 404


def test_nothing_is_loaded_from_anywhere_else() -> None:
    external = re.compile(r"""(?:src|href)\s*=\s*["']https?://|import\s+.*["']https?://""")
    for path in [STATIC / "index.html", *SCRIPTS, STATIC / "app.css"]:
        text = path.read_text(encoding="utf-8")
        assert not external.search(text), f"{path.name} loads something external"
        assert "@import url(http" not in text


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_scripts_never_parse_text_as_html(script: Path) -> None:
    code = script.read_text(encoding="utf-8")
    for unsafe in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
    ):
        assert unsafe not in code, f"{script.name} uses {unsafe}"


def test_every_element_the_script_needs_exists() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([^"]+)"', html))
    code = (STATIC / "app.js").read_text(encoding="utf-8")
    wanted = set(re.findall(r"""\$\(["']#([\w-]+)["']\)""", code))
    assert wanted, "app.js should look elements up with $('#id')"
    assert wanted <= ids, f"missing in index.html: {sorted(wanted - ids)}"


def test_no_inline_scripts_or_handlers_in_the_page() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "<script>" not in html
    assert not re.search(r"\son\w+=", html)  # no onclick= and friends


def test_the_page_is_accessible_enough() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert '<html lang="en"' in html
    assert '<meta name="viewport"' in html
    assert "<main" in html
    assert 'aria-live="polite"' in html  # the live parts announce updates
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert "prefers-reduced-motion" in css
    assert ":focus-visible" in css


def test_children_are_only_replaced_through_the_null_safe_helper() -> None:
    code = (STATIC / "app.js").read_text(encoding="utf-8")
    assert code.count("replaceChildren(") == 1, "use put(node, ...) so null never prints"
