"""Pytest suite for utility functions in amazing_web_url_reader.py."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from amazing_web_url_reader import (
    _chunk_text_by_chars,
    _estimate_tokens_from_text,
    _filter_content_noise,
    _get_preferred_accept_header,
    _html_to_markdown_advanced,
    _ollama_generate,
    _summarize_map_reduce,
    _summarize_single_pass,
    get_site_specific_selectors,
    get_smart_timeout,
)


def test_get_preferred_accept_header_default():
    """Test that the default accept header is returned when no env var is set."""
    with patch.dict(os.environ, {}, clear=True):
        header = _get_preferred_accept_header()
        assert header == "text/markdown, text/plain;q=0.99, text/html;q=0.9, */*;q=0.8"


def test_get_preferred_accept_header_with_env_var():
    """Test that the accept header is returned from the env var when set."""
    with patch.dict(os.environ, {"AMAZING_READER_ACCEPT_HEADER": "application/json"}, clear=True):
        header = _get_preferred_accept_header()
        assert header == "application/json"


def test_estimate_tokens_from_text():
    """Test the token estimation function."""
    assert _estimate_tokens_from_text("") == 0
    assert _estimate_tokens_from_text("one two three four") == 4
    assert _estimate_tokens_from_text("    ") == 1


def test_chunk_text_by_chars():
    """Test the text chunking function."""
    text = "0123456789"
    chunks = _chunk_text_by_chars(text, 3)
    assert chunks == ["012", "345", "678", "9"]


def test_get_smart_timeout():
    """Test the smart timeout function."""
    assert get_smart_timeout("https://www.nytimes.com/some/article") == 45000
    assert get_smart_timeout("https://twitter.com/user/status") == 45000
    assert get_smart_timeout("https://www.reddit.com/r/some/post") == 45000
    assert get_smart_timeout("https://www.foxnews.com/story") == 45000
    assert get_smart_timeout("https://example.com") == 30000


def test_get_site_specific_selectors():
    """Test the site-specific selector function."""
    assert get_site_specific_selectors("https://www.cnn.com/article") == [
        ".article__content",
        ".zn-body__paragraph",
        "#body-text",
    ]
    assert get_site_specific_selectors("https://www.example.com") == []
    assert get_site_specific_selectors("https://edition.cnn.com/2020") == [
        ".article__content",
        ".zn-body__paragraph",
        "#body-text",
    ]


def test_filter_content_noise():
    """Test the content noise filtering function."""
    assert _filter_content_noise("hello world") == "hello world"
    assert _filter_content_noise("Guest favorite Guest favorite") == ""
    assert _filter_content_noise("We use cookies to improve your experience.") == ""
    assert (
        _filter_content_noise("Share on Facebook Share on Twitter")
        == ""
    )
    assert _filter_content_noise("Home > Section > Article Home > Section > Article") == ""


def test_html_to_markdown_advanced_strips_non_content_nodes():
    html = """
    <html>
        <head>
            <script>console.log('nope');</script>
        </head>
        <body>
            <nav>navigation</nav>
            <article id="body-text">
                <h1>Headline</h1>
                <p>Paragraph <strong>content</strong>.</p>
            </article>
        </body>
    </html>
    """
    markdown = _html_to_markdown_advanced(html, url="https://example.com")
    assert "Headline" in markdown
    assert "navigation" not in markdown
    assert "console" not in markdown


def test_html_to_markdown_handles_lists_tables_and_inline_elements():
    html = """
    <body>
        <h2>Heading</h2>
        <p>Paragraph with <strong>bold</strong>, <em>italic</em>, <code>x = 1</code>, and
        <a href="https://example.com">links</a>.</p>
        <ul><li>alpha</li><li>beta</li></ul>
        <ol><li>first</li></ol>
        <blockquote><p>Quote line</p></blockquote>
        <pre><code class="language-py">print('hi')</code></pre>
        <table>
            <tr><th>A</th><th>B</th></tr>
            <tr><td>1</td><td>2</td></tr>
        </table>
        <img src="/img.png" alt="alt text" />
    </body>
    """

    markdown = _html_to_markdown_advanced(html)
    assert "## Heading" in markdown
    assert "**bold**" in markdown
    assert "*italic*" in markdown
    assert "`x = 1`" in markdown
    assert "[links](https://example.com)" in markdown
    assert "- alpha" in markdown and "1. first" in markdown
    assert "| A | B |" in markdown
    assert "![alt text](/img.png)" in markdown


@patch("amazing_web_url_reader.urllib_request.urlopen")
def test_ollama_generate(mock_urlopen):
    """Test the Ollama generate function."""
    # Given
    mock_urlopen.return_value.__enter__.return_value.read.return_value = (
        b'{"response": "summarized text"}'
    )

    # When
    response = _ollama_generate("http://localhost:11434", "test-model", "prompt")

    # Then
    assert response == "summarized text"


@patch("amazing_web_url_reader.urllib_request.urlopen")
def test_summarize_single_pass(mock_urlopen):
    """Test the single-pass summarization function."""
    # Given
    mock_urlopen.return_value.__enter__.return_value.read.return_value = (
        b'{"response": "summarized text"}'
    )

    # When
    summary = _summarize_single_pass("long text", 10, "http://localhost:11434", "test-model")

    # Then
    assert summary == "summarized text"


@patch("amazing_web_url_reader.urllib_request.urlopen")
def test_summarize_map_reduce(mock_urlopen):
    """Test the map-reduce summarization function."""
    # Given
    mock_urlopen.return_value.__enter__.return_value.read.return_value = (
        b'{"response": "summarized chunk"}'
    )
    text = "long text " * 2000

    # When
    summary, debug_info = _summarize_map_reduce(text, 100, "http://localhost:11434", "test-model")

    # Then
    assert "summarized chunk" in summary
    assert debug_info["num_chunks"] > 1
    assert mock_urlopen.call_count > 1
