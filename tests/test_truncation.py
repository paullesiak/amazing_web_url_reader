"""Pytest suite covering truncation behavior with lightweight stubs."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TEST_URL = "https://example.com"
MOCK_HTML = "<html><body>" + ("X" * 150_000) + "</body></html>"
MOCK_MARKDOWN = "X" * 150_000


def _install_third_party_stubs() -> callable:
    """Register lightweight stand-ins for heavy dependencies."""

    saved_modules: dict[str, types.ModuleType | None] = {}

    def _register(name: str, module: types.ModuleType) -> None:
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    # Stub MCP server/types modules
    mcp_mod = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")

    class Server:
        def __init__(self, name: str):
            self.name = name

        def list_tools(self):  # noqa: D401 - decorator stub
            def deco(fn):
                return fn

            return deco

        def call_tool(self):
            def deco(fn):
                return fn

            return deco

        def get_capabilities(self, **_kwargs):
            return {}

    class InitializationOptions:  # pragma: no cover - simple stub
        def __init__(self, *_args, **_kwargs):
            pass

    class NotificationOptions:  # pragma: no cover - simple stub
        def __init__(self, *_args, **_kwargs):
            pass

    class _DummyStdioContext:
        async def __aenter__(self):
            return (None, None)

        async def __aexit__(self, exc_type, exc, tb):
            return False

    stdio_mod = types.ModuleType("mcp.server.stdio")
    stdio_mod.stdio_server = lambda: _DummyStdioContext()

    server_mod.Server = Server
    server_mod.InitializationOptions = InitializationOptions
    server_mod.NotificationOptions = NotificationOptions
    server_mod.stdio = stdio_mod

    types_mod = types.ModuleType("mcp.types")

    class Tool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class TextContent:
        def __init__(self, type: str, text: str):
            self.type = type
            self.text = text

    class ImageContent:  # pragma: no cover - unused stub
        pass

    class EmbeddedResource:  # pragma: no cover - unused stub
        pass

    types_mod.Tool = Tool
    types_mod.TextContent = TextContent
    types_mod.ImageContent = ImageContent
    types_mod.EmbeddedResource = EmbeddedResource

    _register("mcp", mcp_mod)
    _register("mcp.server", server_mod)
    _register("mcp.server.stdio", stdio_mod)
    _register("mcp.types", types_mod)

    # Stub Playwright
    playwright_mod = types.ModuleType("playwright")
    async_api_mod = types.ModuleType("playwright.async_api")

    class _AsyncPlaywright:
        async def start(self):  # pragma: no cover - not exercised in tests
            return self

        @property
        def chromium(self):
            class _Chromium:
                async def launch(self, *args, **kwargs):  # noqa: ARG002
                    raise RuntimeError("playwright launch should be stubbed in tests")

            return _Chromium()

    async def async_playwright():  # pragma: no cover - not exercised
        return _AsyncPlaywright()

    class Page:  # type hint placeholders
        pass

    class Browser:
        pass

    async_api_mod.async_playwright = async_playwright
    async_api_mod.Page = Page
    async_api_mod.Browser = Browser
    class TimeoutError(Exception):
        pass
    async_api_mod.TimeoutError = TimeoutError

    _register("playwright", playwright_mod)
    _register("playwright.async_api", async_api_mod)

    # Stub bs4
    bs4_mod = types.ModuleType("bs4")

    class BeautifulSoup:  # pragma: no cover - unused in tests
        def __init__(self, *_args, **_kwargs):
            pass

    bs4_mod.BeautifulSoup = BeautifulSoup
    _register("bs4", bs4_mod)

    def cleanup() -> None:
        for name, original in saved_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    return cleanup


@pytest.fixture(name="server_module")
def server_module_fixture():
    """Provide the server module with heavy deps stubbed out."""

    cleanup = _install_third_party_stubs()
    root_str = str(ROOT)
    added_path = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        added_path = True
    sys.modules.pop("amazing_web_url_reader", None)
    module = importlib.import_module("amazing_web_url_reader")
    yield module
    cleanup()
    sys.modules.pop("amazing_web_url_reader", None)
    if added_path and root_str in sys.path:
        sys.path.remove(root_str)


@pytest.fixture(name="stubbed_server")
def stubbed_server_fixture(server_module):
    """Replace network-heavy helpers with deterministic fakes."""

    async def fake_fetch(url, wait_for_selector=None, wait_time=0, scroll_to_bottom=False):  # noqa: ARG001
        return MOCK_HTML

    def fake_markdown(html, url=None):  # noqa: ARG001
        return MOCK_MARKDOWN

    server_module.fetch_with_playwright = fake_fetch
    server_module._html_to_markdown_advanced = fake_markdown
    return server_module


def _extract_content(res) -> str:
    payload = json.loads(res[0].text)
    return payload["content"]


def _invoke_tool(server_module, arguments):
    async def _run():
        return await server_module.handle_call_tool(
            name="read_web_url_amazing",
            arguments=arguments,
        )

    return asyncio.run(_run())


def test_default_truncation_behavior(stubbed_server):
    res = _invoke_tool(stubbed_server, {"url": TEST_URL})
    content = _extract_content(res)
    assert len(content) == 100_025
    assert content.endswith("... [content truncated]")


def test_truncation_can_be_disabled(stubbed_server):
    res = _invoke_tool(stubbed_server, {"url": TEST_URL, "truncate": False})
    content = _extract_content(res)
    assert len(content) == len(MOCK_MARKDOWN)
    assert not content.endswith("... [content truncated]")


def test_custom_max_length_is_honored(stubbed_server):
    res = _invoke_tool(
        stubbed_server,
        {"url": TEST_URL, "truncate": True, "max_length": 5_000},
    )
    content = _extract_content(res)
    assert len(content) == 5_025
    assert content.endswith("... [content truncated]")
