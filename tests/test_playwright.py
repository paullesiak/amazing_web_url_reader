"""Pytest suite for the Playwright helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import amazing_web_url_reader as awur


@pytest.fixture(autouse=True)
def _reset_browser_instance():
    awur.browser_instance = None


@pytest.mark.asyncio
async def test_get_browser_reuses_browser_instance():
    mock_async_playwright = AsyncMock()
    mock_browser = AsyncMock()
    mock_browser.is_connected = lambda: True
    mock_playwright = SimpleNamespace(
        chromium=SimpleNamespace(launch=AsyncMock(return_value=mock_browser)),
    )
    mock_async_playwright.start.return_value = mock_playwright

    with patch("amazing_web_url_reader.async_playwright", return_value=mock_async_playwright):
        first_browser = await awur.get_browser()
        second_browser = await awur.get_browser()

    assert first_browser is mock_browser
    assert second_browser is mock_browser
    mock_async_playwright.start.assert_called_once()
    mock_playwright.chromium.launch.assert_awaited_once_with(headless=True, args=[
        '--disable-blink-features=AutomationControlled',
        '--disable-dev-shm-usage',
        '--disable-web-security',
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-accelerated-2d-canvas',
        '--disable-gpu',
    ])


@pytest.mark.asyncio
async def test_dismiss_popups_clicks_first_visible_button():
    button = SimpleNamespace(
        is_visible=AsyncMock(return_value=True),
        click=AsyncMock(),
    )
    locator = SimpleNamespace(first=button)

    class _Page:
        def __init__(self):
            self.wait_for_timeout = AsyncMock()

        def locator(self, selector):  # noqa: ARG002
            return locator

    page = _Page()

    await awur.dismiss_popups(page)  # type: ignore[arg-type]

    assert button.is_visible.await_count >= 1
    assert button.click.await_count >= 1


@pytest.mark.asyncio
async def test_scroll_to_load_content_scrolls_then_stops():
    mock_page = AsyncMock()
    mock_page.evaluate = AsyncMock(
        side_effect=[
            1000,  # initial height
            None,  # first scroll
            2000,  # new height after first scroll
            None,  # second scroll
            2000,  # new height same as last -> break
            None,  # scroll back to top
        ]
    )

    await awur.scroll_to_load_content(mock_page)

    evaluate_calls = [invocation.args[0] for invocation in mock_page.evaluate.call_args_list]
    assert "window.scrollTo(0, document.body.scrollHeight)" in evaluate_calls
    assert evaluate_calls[-1] == "window.scrollTo(0, 0)"


@pytest.mark.asyncio
async def test_fetch_with_playwright_sets_accept_header(monkeypatch):
    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.wait_for_timeout = AsyncMock()
    mock_page.wait_for_selector = AsyncMock()
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page
    mock_page.content.return_value = "<html></html>"

    monkeypatch.setattr("amazing_web_url_reader.get_browser", AsyncMock(return_value=mock_browser))
    monkeypatch.setattr("amazing_web_url_reader.dismiss_popups", AsyncMock())
    monkeypatch.setattr("amazing_web_url_reader.scroll_to_load_content", AsyncMock())

    content = await awur.fetch_with_playwright("https://example.com", wait_time=0)

    assert content == "<html></html>"
    kwargs = mock_browser.new_context.call_args.kwargs
    headers = kwargs["extra_http_headers"]
    assert headers["Accept"].startswith("text/markdown")
    mock_page.goto.assert_awaited_once_with("https://example.com", wait_until="networkidle", timeout=30000)
