"""Tests covering the MCP tool handler wiring."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

import amazing_web_url_reader as awur


@pytest.mark.asyncio
async def test_handle_call_tool_success(monkeypatch):
    async def fake_fetch(url, wait_for_selector=None, wait_time=0, scroll_to_bottom=True):  # noqa: ARG001
        return "<html><body><article><p>Hello world</p></article></body></html>"

    def fake_markdown(html, url=None):  # noqa: ARG001
        return "# Rendered\n\nHello world"

    monkeypatch.setattr("amazing_web_url_reader._fetch_native_markdown", lambda url: None)
    monkeypatch.setattr(
        "amazing_web_url_reader.fetch_with_playwright", AsyncMock(side_effect=fake_fetch)
    )
    monkeypatch.setattr("amazing_web_url_reader._html_to_markdown_advanced", fake_markdown)

    result = await awur.handle_call_tool(
        name="read_web_url_amazing",
        arguments={"url": "https://example.com", "truncate": False},
    )

    payload = json.loads(result[0].text)
    assert payload["status"] == "success"
    assert payload["content"].startswith("# Rendered")
    assert payload["render_method"] == "playwright"
    assert payload["summarization"]["enabled"] is False


@pytest.mark.asyncio
async def test_handle_call_tool_validates_url(monkeypatch):
    monkeypatch.setattr("amazing_web_url_reader.fetch_with_playwright", AsyncMock())

    result = await awur.handle_call_tool("read_web_url_amazing", {"url": "ftp://invalid"})
    payload = json.loads(result[0].text)
    assert payload["status"] == "error"
    assert "http://" in payload["error"]


@pytest.mark.asyncio
async def test_handle_call_tool_uses_native_markdown(monkeypatch):
    fetch_with_playwright = AsyncMock()

    monkeypatch.setattr(
        "amazing_web_url_reader._fetch_native_markdown",
        lambda url: (
            "# Native\n\nHello world",
            {
                "source_content_type": "text/markdown; charset=utf-8",
                "markdown_tokens": 42,
            },
        ),
    )
    monkeypatch.setattr("amazing_web_url_reader.fetch_with_playwright", fetch_with_playwright)

    result = await awur.handle_call_tool(
        name="read_web_url_amazing",
        arguments={"url": "https://example.com", "truncate": False},
    )

    payload = json.loads(result[0].text)
    assert payload["status"] == "success"
    assert payload["content"].startswith("# Native")
    assert payload["render_method"] == "native_markdown"
    assert payload["source_content_type"] == "text/markdown; charset=utf-8"
    assert payload["markdown_tokens"] == 42
    fetch_with_playwright.assert_not_called()


@pytest.mark.asyncio
async def test_handle_call_tool_summarization(monkeypatch):
    async def fake_fetch(url, wait_for_selector=None, wait_time=0, scroll_to_bottom=True):  # noqa: ARG001
        return "<html><body><article>" + ("A" * 2000) + "</article></body></html>"

    def fake_markdown(html, url=None):  # noqa: ARG001
        return "A" * 2000

    monkeypatch.setattr("amazing_web_url_reader._fetch_native_markdown", lambda url: None)
    monkeypatch.setattr(
        "amazing_web_url_reader.fetch_with_playwright", AsyncMock(side_effect=fake_fetch)
    )
    monkeypatch.setattr("amazing_web_url_reader._html_to_markdown_advanced", fake_markdown)
    monkeypatch.setattr(
        "amazing_web_url_reader._summarize_single_pass", lambda text, tokens, host, model: "summary"
    )

    result = await awur.handle_call_tool(
        name="read_web_url_amazing",
        arguments={
            "url": "https://example.com",
            "use_ollama_summarization": True,
            "summary_target_tokens": 100,
            "ollama_host": "http://localhost:11434",
            "ollama_model": "fake",
        },
    )

    payload = json.loads(result[0].text)
    assert payload["summarization"]["used"] is True
    assert payload["content"] == "summary"


@pytest.mark.asyncio
async def test_handle_call_tool_defaults_ollama_host_to_localhost(monkeypatch):
    async def fake_fetch(url, wait_for_selector=None, wait_time=0, scroll_to_bottom=True):  # noqa: ARG001
        return "<html><body><article>" + ("A" * 2000) + "</article></body></html>"

    def fake_markdown(html, url=None):  # noqa: ARG001
        return "A" * 2000

    observed: dict[str, str] = {}

    def fake_summarize(text, tokens, host, model):  # noqa: ARG001
        observed["host"] = host
        return "summary"

    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setattr("amazing_web_url_reader._fetch_native_markdown", lambda url: None)
    monkeypatch.setattr(
        "amazing_web_url_reader.fetch_with_playwright", AsyncMock(side_effect=fake_fetch)
    )
    monkeypatch.setattr("amazing_web_url_reader._html_to_markdown_advanced", fake_markdown)
    monkeypatch.setattr("amazing_web_url_reader._summarize_single_pass", fake_summarize)

    result = await awur.handle_call_tool(
        name="read_web_url_amazing",
        arguments={
            "url": "https://example.com",
            "use_ollama_summarization": True,
            "summary_target_tokens": 100,
            "ollama_model": "fake",
        },
    )

    payload = json.loads(result[0].text)
    assert observed["host"] == "http://localhost:11434"
    assert payload["summarization"]["host"] == "http://localhost:11434"
    assert payload["content"] == "summary"


@pytest.mark.asyncio
async def test_handle_call_tool_rejects_unknown_tool():
    with pytest.raises(ValueError):
        await awur.handle_call_tool("unknown", {})


@pytest.mark.asyncio
async def test_handle_call_tool_truncates_content(monkeypatch):
    async def fake_fetch(url, wait_for_selector=None, wait_time=0, scroll_to_bottom=True):  # noqa: ARG001
        return "<html><body><article>" + ("A" * 200) + "</article></body></html>"

    def fake_markdown(html, url=None):  # noqa: ARG001
        return "A" * 200

    monkeypatch.setattr("amazing_web_url_reader._fetch_native_markdown", lambda url: None)
    monkeypatch.setattr(
        "amazing_web_url_reader.fetch_with_playwright", AsyncMock(side_effect=fake_fetch)
    )
    monkeypatch.setattr("amazing_web_url_reader._html_to_markdown_advanced", fake_markdown)

    result = await awur.handle_call_tool(
        name="read_web_url_amazing",
        arguments={"url": "https://example.com", "max_length": 10, "truncate": True},
    )

    payload = json.loads(result[0].text)
    assert payload["content"].endswith("[content truncated]")


@pytest.mark.asyncio
async def test_handle_list_tools_declares_reader_tool():
    tools = await awur.handle_list_tools()
    tool_names = {tool.name for tool in tools}
    assert "read_web_url_amazing" in tool_names


@pytest.mark.asyncio
async def test_cleanup_closes_browser_instance():
    fake_browser = AsyncMock()
    awur.browser_instance = fake_browser

    await awur.cleanup()

    fake_browser.close.assert_awaited_once()
    assert awur.browser_instance is None
