#!/usr/bin/env python3
"""Amazing Web URL Reader - MCP Server with full browser rendering via Playwright.

This is an advanced MCP (Model Context Protocol) server that uses Playwright to
render web pages in a real browser before converting them to markdown. This allows
it to handle JavaScript-heavy sites, SPAs, and dynamically loaded content that
simple HTTP requests would miss.

Features:
---------
- Full browser rendering with Playwright (Chromium)
- JavaScript execution support
- Waits for dynamic content to load
- Handles SPAs and client-side rendered pages
- Auto-scrolling to trigger lazy-loaded content
- Cookie banner and popup dismissal
- Advanced content extraction with rendered DOM
- Clean markdown conversion

Installation and Usage:
----------------------
1. Save this file as `amazing_web_url_reader.py`

2. First time setup (Playwright needs to download browser):
   ```bash
   uv run amazing_web_url_reader.py --install-browser
   # or equivalently:
   uvx playwright install chromium
   ```

3. Add to your Claude Desktop config (~/Library/Application Support/Claude/claude_desktop_config.json on macOS):
   ```json
   {
     "mcpServers": {
       "amazing-web-reader": {
         "command": "uv",
         "args": ["run", "path/to/amazing_web_url_reader.py"],
         "env": {
           "OLLAMA_HOST": "http://localhost:11434",
           "OLLAMA_MODEL": "gpt-oss:20b",
           "AMAZING_READER_SUMMARY_TOKENS": "2000"
         }
       }
     }
   }
   ```

   Notes:
   - The `env` block is optional. If omitted, the server behaves as before (no summarization).
   - Set `OLLAMA_MODEL` and `AMAZING_READER_SUMMARY_TOKENS` to enable summarization by default.
   - `OLLAMA_HOST` defaults to `http://localhost:11434` if not provided.

4. The tool will be available in your MCP client as `read_web_url_amazing`

Tool Usage Example:
------------------
Once configured, the LLM can use it like:
- "Read this React app: https://react.dev"
- "Get content from this SPA: https://example-spa.com"
- "Fetch this JavaScript-heavy page: https://modern-web-app.com"

Advanced Options:
----------------
The tool accepts these parameters:
- url: The web URL to read (required)
- wait_for_selector: CSS selector to wait for before reading (optional)
- wait_time: Additional milliseconds to wait after page load (optional, default: 2000)
- scroll_to_bottom: Whether to scroll to bottom to trigger lazy loading (optional, default: true)
- truncate: Whether to truncate the output (optional, default: true)
- max_length: Max characters when truncating (optional, default: 100000)

Summarization (optional):
- use_ollama_summarization: Enable Ollama-based LLM summarization (default: false)
- summary_target_tokens: Target summary size in tokens (approximate)
- ollama_host: Ollama host URL (e.g., http://localhost:11434)
- ollama_model: Model name to use (e.g., gpt-oss:20b)

Environment variables:
- OLLAMA_HOST, OLLAMA_MODEL: Defaults for host/model
- AMAZING_READER_SUMMARY_TOKENS: Default target tokens
- AMAZING_READER_ACCEPT_HEADER: Optional override for the Accept header when fetching pages

Requirements (automatically handled by uv):
--------------------------------------------
- mcp>=1.0.0
- playwright>=1.40.0
- beautifulsoup4>=4.9.0
- lxml>=4.9.0

Notes:
------
- First run will download Chromium browser (~130MB)
- Rendering takes longer than simple HTTP requests but handles modern web apps
- Truncation is enabled by default with a high limit; set `truncate` to false to return full content
- Automatically dismisses common cookie banners and popups

License: Apache
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mcp>=1.0.0",
#   "playwright>=1.40.0",
#   "beautifulsoup4>=4.9.0",
#   "lxml>=4.9.0",
# ]
# ///

import asyncio
import json
import logging
import re
import sys
import os
import stat as _stat
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse
from urllib import request as urllib_request
from urllib.error import URLError, HTTPError

import mcp.server.stdio
import mcp.types as types
from bs4 import BeautifulSoup
from mcp.server import InitializationOptions, NotificationOptions, Server
from playwright.async_api import async_playwright, Page, Browser, TimeoutError as PlaywrightTimeout

class _MCPLogHandler(logging.Handler):
    """Mirror Python logging records to MCP logMessage notifications when possible."""

    _LEVEL_MAP: dict[int, str] = {
        logging.DEBUG: "debug",
        logging.INFO: "info",
        logging.WARNING: "warning",
        logging.ERROR: "error",
        logging.CRITICAL: "critical",
    }

    def emit(self, record: logging.LogRecord) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        try:
            session = server.request_context.session  # type: ignore[attr-defined]
        except LookupError:
            return

        level = self._LEVEL_MAP.get(record.levelno, "info")
        message = self.format(record)

        async def _send():
            try:
                await session.send_log_message(level=level, data=message, logger=record.name)
            except Exception:
                pass

        loop.create_task(_send())


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(_MCPLogHandler())
# Opt-in debug logging when MCP_DEBUG is set
if os.environ.get("MCP_DEBUG"):
    logger.setLevel(logging.DEBUG)

# Low-level server (stdio)
server = Server("amazing-web-url-reader")

# Global browser instance for reuse
browser_instance: Optional[Browser] = None

# Preferred MIME types for downstream LLM consumption
DEFAULT_ACCEPT_HEADER = "text/markdown, text/plain;q=0.99, text/html;q=0.9, */*;q=0.8"


def _get_preferred_accept_header() -> str:
    """Return the Accept header, allowing overrides via env."""
    header = os.environ.get("AMAZING_READER_ACCEPT_HEADER")
    if header and header.strip():
        return header.strip()
    return DEFAULT_ACCEPT_HEADER


# ---------- Summarization helpers (Ollama) ----------

def _estimate_tokens_from_text(text: str) -> int:
    """Rough token estimate. Approx 4 chars/token for English text."""
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def _ollama_generate(
    host: str,
    model: str,
    prompt: str,
    system: Optional[str] = None,
    options: Optional[dict] = None,
) -> str:
    """Call Ollama's /api/generate endpoint and return the response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    if options:
        payload["options"] = options

    data = json.dumps(payload).encode("utf-8")
    url = host.rstrip("/") + "/api/generate"
    req = urllib_request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")

    with urllib_request.urlopen(req, timeout=120) as resp:
        body = resp.read()
        parsed = json.loads(body.decode("utf-8"))
        return parsed.get("response", "")


def _summarize_single_pass(text: str, target_tokens: int, host: str, model: str) -> str:
    system = (
        "You are a precise summarization assistant. Preserve key facts, numbers,"
        " names, code blocks, and structure. Avoid fluff. Maintain markdown formatting."
    )
    prompt = (
        "Summarize the following content into approximately "
        f"{target_tokens} tokens. If the content is already concise, return it as-is.\n\n"
        "Output should be well-structured markdown with headings and bullet points where helpful.\n\n"
        "Content begins below:\n\n"
        "<CONTENT>\n" + text + "\n</CONTENT>\n"
    )
    return _ollama_generate(host=host, model=model, prompt=prompt, system=system)


def _chunk_text_by_chars(text: str, chunk_chars: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        chunks.append(text[start:end])
        start = end
    return chunks


def _summarize_map_reduce(text: str, target_tokens: int, host: str, model: str) -> Tuple[str, dict]:
    """Summarize large text via map-reduce style (chunk then combine). Returns (summary, debug_info)."""
    debug = {"stage": "map-reduce", "num_chunks": 0}
    # Assume conservative context window. Use chunks of ~3000 tokens (~12000 chars)
    chunk_token_budget = 3000
    chunk_chars = chunk_token_budget * 4
    chunks = _chunk_text_by_chars(text, chunk_chars)
    debug["num_chunks"] = len(chunks)

    # Per-chunk target so the final combiner has good material
    per_chunk_target = max(300, int(target_tokens * 1.5 / max(1, len(chunks))))

    summaries: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        prompt = (
            f"You are summarizing chunk {i} of {len(chunks)}. "
            f"Produce an accurate, non-lossy summary in about {per_chunk_target} tokens.\n\n"
            "Use markdown, keep key details, and preserve any code snippets.\n\n"
            "Chunk content:\n\n<CHUNK>\n" + ch + "\n</CHUNK>\n"
        )
        part = _ollama_generate(host=host, model=model, prompt=prompt, system=None)
        summaries.append(part.strip())

    combined = "\n\n".join(summaries)
    final = _summarize_single_pass(combined, target_tokens, host=host, model=model)
    return final, debug


async def get_browser() -> Browser:
    """Get or create a browser instance."""
    global browser_instance
    if browser_instance is None or not browser_instance.is_connected():
        playwright = await async_playwright().start()
        browser_instance = await playwright.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--disable-web-security',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-accelerated-2d-canvas',
                '--disable-gpu'
            ]
        )
    return browser_instance


async def dismiss_popups(page: Page) -> None:
    """Try to dismiss common cookie banners and popups."""
    try:
        # Common cookie banner selectors
        dismiss_selectors = [
            'button:has-text("Accept")',
            'button:has-text("Accept all")',
            'button:has-text("Accept cookies")',
            'button:has-text("Agree")',
            'button:has-text("Got it")',
            'button:has-text("OK")',
            'button:has-text("I agree")',
            'button:has-text("Dismiss")',
            'button:has-text("Close")',
            '[aria-label="Close"]',
            '[aria-label="Dismiss"]',
            '.cookie-consent button',
            '#cookie-banner button',
            '.cookie-notice button',
        ]
        
        for selector in dismiss_selectors:
            try:
                button = page.locator(selector).first
                if await button.is_visible(timeout=100):
                    await button.click(timeout=500)
                    await page.wait_for_timeout(500)
                    break
            except:
                continue
    except Exception as e:
        logger.debug(f"Popup dismissal attempt failed (non-critical): {e}")


async def scroll_to_load_content(page: Page) -> None:
    """Scroll page to trigger lazy-loaded content."""
    try:
        # Get initial height
        last_height = await page.evaluate("document.body.scrollHeight")
        
        # Scroll down in chunks
        for _ in range(3):  # Max 3 scroll attempts
            # Scroll to bottom
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1000)
            
            # Check if new content loaded
            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        # Scroll back to top for consistent reading
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(500)
    except Exception as e:
        logger.debug(f"Scrolling failed (non-critical): {e}")


def get_smart_timeout(url: str) -> int:
    """Return a conservative navigation timeout based on URL complexity."""
    parsed = urlparse(url)
    base = 45000
    if parsed.query or parsed.path.count("/") > 3:
        base += 15000
    return base


async def _maybe_wait_for_network_idle(page: Page, timeout_ms: int = 15000) -> None:
    """Best-effort wait for networkidle, falling back after the budget expires."""
    if timeout_ms <= 0:
        return
    try:
        await page.wait_for_load_state("networkidle", timeout=timeout_ms)
        return
    except PlaywrightTimeout:
        logger.info(
            "Page never reached networkidle within %sms; continuing with rendered DOM",
            timeout_ms,
        )
    # Give 'load' a short chance before returning control
    try:
        await page.wait_for_load_state("load", timeout=max(2000, timeout_ms // 2))
    except PlaywrightTimeout:
        logger.debug("Load state also timed out; proceeding regardless")


async def fetch_with_playwright(url: str, wait_for_selector: Optional[str] = None, 
                               wait_time: int = 2000, scroll_to_bottom: bool = True) -> str:
    """Fetch and render URL using Playwright browser."""
    browser = await get_browser()
    context = await browser.new_context(
        viewport={'width': 1920, 'height': 1080},
        user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        extra_http_headers={
            'Accept': _get_preferred_accept_header(),
        }
    )
    page = await context.new_page()
    
    try:
        # Navigate to the page with smart timeout
        timeout = get_smart_timeout(url)
        await page.goto(url, wait_until='domcontentloaded', timeout=timeout)
        
        # Wait for specific selector if provided
        if wait_for_selector:
            try:
                await page.wait_for_selector(wait_for_selector, timeout=10000)
            except:
                logger.warning(f"Selector '{wait_for_selector}' not found, continuing anyway")
        
        await _maybe_wait_for_network_idle(page)
        # Additional wait time for dynamic content
        await page.wait_for_timeout(wait_time)
        
        # Try to dismiss popups
        await dismiss_popups(page)
        
        # Scroll to load lazy content if requested
        if scroll_to_bottom:
            await scroll_to_load_content(page)
        
        # Remove unnecessary elements before getting content
        await page.evaluate("""
            // Remove common non-content elements
            const selectorsToRemove = [
                'script', 'style', 'noscript', 'iframe', 
                '.advertisement', '.ads', '.ad-container',
                '.cookie-banner', '.cookie-notice', '.gdpr-banner',
                '.newsletter-signup', '.popup', '.modal',
                'header nav', 'footer nav'
            ];
            
            selectorsToRemove.forEach(selector => {
                document.querySelectorAll(selector).forEach(el => el.remove());
            });
            
            // Remove elements with common ad/tracking classes
            document.querySelectorAll('[class*="cookie"], [class*="banner"], [id*="cookie"], [id*="banner"]')
                .forEach(el => {
                    if (el.offsetHeight < 200) el.remove();
                });
        """)
        
        # Get the fully rendered HTML
        content = await page.content()
        
        return content
        
    finally:
        await context.close()


def get_site_specific_selectors(url: str) -> list:
    """Get optimal content selectors for specific sites based on URL domain."""
    domain_selectors = {
        # News sites
        'cnn.com': ['.article__content', '.zn-body__paragraph', '#body-text'],
        'bbc.com': ['[data-component="text-block"]', '.story-body', '.story-body__inner'],
        'nytimes.com': ['[data-testid="ArticleBody"]', 'section[name="articleBody"]', '.StoryBodyCompanionColumn'],
        'reuters.com': ['[data-testid="ArticleBody"]', '.ArticleBodyWrapper', '.StandardArticleBody_body'],
        'theguardian.com': ['[data-gu-name="body"]', '.content__article-body', '#maincontent .content__article-body'],
        'washingtonpost.com': ['[data-qa="article-body"]', '.article-body', '#article-body'],
        'wsj.com': ['[data-module="ArticleBody"]', '.wsj-snippet-body', '.articleLead-container'],
        'bloomberg.com': ['[data-module="BodyText"]', '.body-content', '.fence-body'],
        'ft.com': ['[data-trackable="story-body"]', '.article__content-body', '.n-content-body'],
        'usatoday.com': ['[data-module="ArticleBody"]', '.article-wrap', '.gnt_ar_b'],
        
        # Tech sites
        'techcrunch.com': ['.entry-content', '.article-content', '.post-block'],
        'arstechnica.com': ['.article-content', '.post-content', 'section.article-guts'],
        'theverge.com': ['.duet--article--article-body-component', '.c-entry-content', '.l-wrapper'],
        'wired.com': ['.ArticleBodyWrapper', '.article__chunks', '.content-header'],
        'engadget.com': ['.article-text', '.o-article_body', '.article-content'],
        
        # Social/Community
        'reddit.com': ['[data-testid="comment"]', '.usertext-body', 'div[data-click-id="text"]'],
        'stackoverflow.com': ['.question .js-post-body', '.answer .js-post-body', '.post-text'],
        'medium.com': ['article section', '.pw-post-body-paragraph', '.graf'],
        
        # Documentation
        'docs.python.org': ['.body', '.document', '.section'],
        'developer.mozilla.org': ['#wikiArticle', '.main-page-content', 'article'],
        'github.com': ['.markdown-body', '.js-wiki-content', 'readme-toc'],
        
        # General content sites
        'wikipedia.org': ['#mw-content-text', '.mw-parser-output', '#bodyContent'],
        'youtube.com': ['#description', '#meta-contents', '.content'],
    }
    
    # Extract domain from URL
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Check for exact matches first
        if domain in domain_selectors:
            return domain_selectors[domain]
        
        # Check for subdomain matches (e.g., edition.cnn.com -> cnn.com)
        for site_domain, selectors in domain_selectors.items():
            if domain.endswith('.' + site_domain) or domain == site_domain:
                return selectors
                
    except Exception:
        pass
    
    # Return empty list if no specific selectors found
    return []


def _filter_content_noise(content: str) -> str:
    """Remove repetitive noise patterns from content."""
    if not content:
        return content
    
    # Common noise patterns to filter out
    noise_patterns = [
        # CNN repetitive numbered articles pattern (more aggressive)
        r'info\s*The top.*?preferences\.',
        
        # Generic repetitive phrases
        r'(\b\w+\s+favorite\b)\s+\1',  # "Guest favorite Guest favorite"
        
        # Cookie banner remnants (often incomplete after JS removal)
        r'(?:We use cookies|This site uses cookies|By continuing|Accept all cookies).*?(?:\.|$)',
        
        # Common admin/footer repetition
        r'(\b(?:copyright|all rights reserved|privacy policy|terms of service)\b.*?)\s+\1',
        
        # Social media share buttons text
        r'(?:Share on|Follow us on|Connect with us).*?(?:Facebook|Twitter|LinkedIn|Instagram)',
        
        # Newsletter signup remnants
        r'(?:Subscribe|Sign up|Newsletter|Email updates).*?(?:newsletter|updates|alerts)',
        
        # Navigation breadcrumb repetition
        r'(\bHome\s*>\s*.*?)\s+\1',
    ]
    
    # Apply filters
    filtered_content = content
    for pattern in noise_patterns:
        filtered_content = re.sub(pattern, '', filtered_content, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove excessive whitespace that might result from filtering
    filtered_content = re.sub(r'\n\s*\n\s*\n', '\n\n', filtered_content)
    filtered_content = re.sub(r'[ \t]+', ' ', filtered_content)
    
    return filtered_content.strip()


def _html_to_markdown_advanced(html_content: str, url: str = None) -> str:
    """Convert rendered HTML to markdown with advanced processing."""
    soup = BeautifulSoup(html_content, 'lxml')
    
    # Remove script, style, and other non-content elements
    for element in soup(
        ['script', 'style', 'nav', 'header', 'footer', 'aside', 'form',
         'button', 'input', 'select', 'textarea', 'svg', 'canvas']
    ):
        element.decompose()
    
    # Remove empty divs and spans
    for element in soup.find_all(['div', 'span']):
        if not element.get_text(strip=True):
            element.decompose()
    
    # Find main content area - try site-specific selectors first
    main_content = None
    
    # Get site-specific selectors if URL is provided
    site_selectors = []
    if url:
        site_selectors = get_site_specific_selectors(url)
    
    # Combine site-specific and generic selectors
    all_selectors = site_selectors + ['main', 'article', '[role="main"]', '#main', '.main-content', '#content', '.content']
    
    for selector in all_selectors:
        if isinstance(selector, str):
            if selector.startswith('[') or selector.startswith('#') or selector.startswith('.'):
                main_content = soup.select_one(selector)
            else:
                main_content = soup.find(selector)
        if main_content:
            break
    
    # If no main content found, use body
    if not main_content:
        main_content = soup.find('body') or soup
    
    # Convert to markdown
    markdown_parts: List[str] = []
    _process_element_advanced(main_content, markdown_parts, 0)
    
    # Join and clean up
    markdown = '\n'.join(markdown_parts)
    
    # Clean up excessive newlines
    lines = []
    empty_count = 0
    for line in markdown.split('\n'):
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
            empty_count = 0
        else:
            empty_count += 1
            if empty_count <= 1:
                lines.append('')
    
    result = '\n'.join(lines).strip()
    
    # Ensure we have some content
    if not result:
        # Fallback to basic text extraction
        result = soup.get_text(separator='\n', strip=True)
    
    # Apply content noise filtering
    result = _filter_content_noise(result)
    
    return result


def _process_element_advanced(element, markdown_parts: List[str], depth: int) -> None:
    """Process HTML element with advanced handling for rendered content."""
    if element.name is None:
        text = str(element).strip()
        if text and len(text) > 1:
            markdown_parts.append(text)
        return
    
    # Skip hidden elements
    if element.get('style'):
        if 'display:none' in element.get('style').replace(' ', '') or \
           'visibility:hidden' in element.get('style').replace(' ', ''):
            return
    
    # Handle different HTML elements
    if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        level = int(element.name[1])
        text = element.get_text(strip=True)
        if text:
            markdown_parts.append(f"{'#' * level} {text}")
            markdown_parts.append("")
    
    elif element.name == 'p':
        paragraph_parts: List[str] = []
        _process_inline_elements_advanced(element, paragraph_parts)
        if paragraph_parts:
            markdown_parts.append(' '.join(paragraph_parts))
            markdown_parts.append("")
    
    elif element.name in ['ul', 'ol']:
        markdown_parts.append("")
        list_items = element.find_all('li', recursive=False)
        for i, li in enumerate(list_items):
            text = li.get_text(strip=True)
            if text:
                if element.name == 'ul':
                    markdown_parts.append(f"- {text}")
                else:
                    markdown_parts.append(f"{i + 1}. {text}")
        markdown_parts.append("")
    
    elif element.name == 'blockquote':
        text = element.get_text(strip=True)
        if text:
            markdown_parts.append("")
            for line in text.split('\n'):
                if line.strip():
                    markdown_parts.append(f"> {line.strip()}")
            markdown_parts.append("")
    
    elif element.name in ['pre', 'code']:
        if element.name == 'pre':
            code_elem = element.find('code')
            if code_elem:
                text = code_elem.get_text()
                # Try to detect language from class
                lang = ''
                if code_elem.get('class'):
                    classes = code_elem.get('class')
                    for cls in classes:
                        if 'language-' in cls:
                            lang = cls.replace('language-', '')
                            break
            else:
                text = element.get_text()
                lang = ''
            
            if text:
                markdown_parts.append("")
                markdown_parts.append(f"```{lang}")
                markdown_parts.append(text.strip())
                markdown_parts.append("```")
                markdown_parts.append("")
        else:
            text = element.get_text()
            if text and element.parent.name != 'pre':
                markdown_parts.append(f"`{text}`")
    
    elif element.name == 'table':
        # Basic table support
        rows = element.find_all('tr')
        if rows:
            markdown_parts.append("")
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_text = ' | '.join(cell.get_text(strip=True) for cell in cells)
                    markdown_parts.append(f"| {row_text} |")
                    # Add header separator after first row if it has th elements
                    if row == rows[0] and row.find('th'):
                        separator = ' | '.join('---' for _ in cells)
                        markdown_parts.append(f"| {separator} |")
            markdown_parts.append("")
    
    elif element.name == 'img':
        alt = element.get('alt', '')
        src = element.get('src', '')
        if src:
            markdown_parts.append(f"![{alt}]({src})")
    
    elif element.name == 'a':
        text = element.get_text(strip=True)
        href = element.get('href', '')
        if text and href and not href.startswith('javascript:'):
            markdown_parts.append(f"[{text}]({href})")
        elif text:
            markdown_parts.append(text)
    
    elif element.name in ['div', 'section', 'article', 'main', 'body', 'span']:
        # Process children for container elements
        for child in element.children:
            _process_element_advanced(child, markdown_parts, depth + 1)
    
    else:
        # For other elements, try to extract meaningful text
        text = element.get_text(strip=True)
        if text and len(text) > 2:
            markdown_parts.append(text)


def _process_inline_elements_advanced(element, parts: List[str]) -> None:
    """Process inline elements with advanced handling."""
    for child in element.children:
        if child.name is None:
            text = str(child).strip()
            if text:
                parts.append(text)
        elif child.name in {'strong', 'b'}:
            text = child.get_text(strip=True)
            if text:
                parts.append(f"**{text}**")
        elif child.name in {'em', 'i'}:
            text = child.get_text(strip=True)
            if text:
                parts.append(f"*{text}*")
        elif child.name == 'code':
            text = child.get_text()
            if text:
                parts.append(f"`{text}`")
        elif child.name == 'a':
            text = child.get_text(strip=True)
            href = child.get('href', '')
            if text and href and not href.startswith('javascript:'):
                parts.append(f"[{text}]({href})")
            elif text:
                parts.append(text)
        elif child.name == 'br':
            parts.append('\n')
        else:
            text = child.get_text(strip=True)
            if text:
                parts.append(text)


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="read_web_url_amazing",
            description="Read web content using a full browser (Playwright) to handle JavaScript, SPAs, and dynamic content, then convert to markdown",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Web URL to read (must start with http:// or https://)",
                    },
                    "wait_for_selector": {
                        "type": "string",
                        "description": "Optional CSS selector to wait for before reading content",
                    },
                    "wait_time": {
                        "type": "integer",
                        "description": "Additional milliseconds to wait after page load (default: 2000)",
                        "default": 2000,
                    },
                    "scroll_to_bottom": {
                        "type": "boolean",
                        "description": "Whether to scroll to bottom to trigger lazy-loaded content (default: true)",
                        "default": True,
                    },
                    "truncate": {
                        "type": "boolean",
                        "description": "Whether to truncate the output (default: true)",
                        "default": True,
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Max characters when truncating (default: 100000)",
                        "default": 100000,
                    },
                    "use_ollama_summarization": {
                        "type": "boolean",
                        "description": "Enable Ollama-based summarization to target token length (default: false)",
                        "default": False
                    },
                    "summary_target_tokens": {
                        "type": "integer",
                        "description": "Approximate target token length for summary (e.g., 2000)"
                    },
                    "ollama_host": {
                        "type": "string",
                        "description": "Ollama host URL (e.g., http://localhost:11434). Defaults to OLLAMA_HOST env or http://localhost:11434"
                    },
                    "ollama_model": {
                        "type": "string",
                        "description": "Ollama model name (e.g., gpt-oss:20b). Defaults to OLLAMA_MODEL env if set"
                    }
                },
                "required": ["url"],
            },
        )
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any]
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution."""
    if name != "read_web_url_amazing":
        raise ValueError(f"Unknown tool: {name}")
    
    url = arguments.get("url")
    wait_for_selector = arguments.get("wait_for_selector")
    wait_time = arguments.get("wait_time", 2000)
    scroll_to_bottom = arguments.get("scroll_to_bottom", True)
    truncate = arguments.get("truncate", True)
    max_length = arguments.get("max_length", 100000)
    use_ollama_summarization = arguments.get("use_ollama_summarization", False)
    summary_target_tokens = arguments.get("summary_target_tokens")
    ollama_host_arg = arguments.get("ollama_host")
    ollama_model_arg = arguments.get("ollama_model")
    
    try:
        # Validate URL
        if not url or not isinstance(url, str):
            raise ValueError("url must be a non-empty string")
        
        url = url.strip()
        if not url:
            raise ValueError("url cannot be empty")
        
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError("url must start with http:// or https://")
        
        # Validate URL format
        try:
            parsed = urlparse(url)
            if not parsed.netloc:
                raise ValueError(f"URL is malformed: '{url}'. Must include a valid hostname.")
        except Exception as e:
            raise ValueError(f"URL is malformed: '{url}'. {str(e)}")
        
        logger.info(f"Reading web URL with Playwright: {url}")
        
        # Fetch with Playwright
        html_content = await fetch_with_playwright(
            url, 
            wait_for_selector=wait_for_selector,
            wait_time=wait_time,
            scroll_to_bottom=scroll_to_bottom
        )
        
        # Convert to markdown
        markdown_content = _html_to_markdown_advanced(html_content, url)

        # Summarization (optional)
        meta = {
            "summarization": {
                "enabled": False,
                "used": False,
                "strategy": None,
                "model": None,
                "host": None,
                "target_tokens": None,
                "original_estimated_tokens": _estimate_tokens_from_text(markdown_content),
                "final_estimated_tokens": None,
                "fallback_reason": None,
            }
        }

        # Resolve env defaults for summarization
        env_ollama_host = os.environ.get("OLLAMA_HOST")
        env_ollama_model = os.environ.get("OLLAMA_MODEL")
        env_summary_tokens = os.environ.get("AMAZING_READER_SUMMARY_TOKENS")

        effective_host = ollama_host_arg or env_ollama_host or "http://ollamalb.local:11434"
        effective_model = ollama_model_arg or env_ollama_model or "gpt-oss:20b"
        if summary_target_tokens is None and env_summary_tokens:
            try:
                summary_target_tokens = int(env_summary_tokens)
            except Exception:
                summary_target_tokens = None

        summarization_enabled = bool(use_ollama_summarization or (summary_target_tokens and effective_model))
        meta["summarization"]["enabled"] = summarization_enabled
        meta["summarization"]["model"] = effective_model
        meta["summarization"]["host"] = effective_host
        meta["summarization"]["target_tokens"] = summary_target_tokens

        if summarization_enabled and effective_model and summary_target_tokens and summary_target_tokens > 0:
            try:
                orig_tokens = meta["summarization"]["original_estimated_tokens"]
                force = bool(use_ollama_summarization)
                should_summarize = force or (orig_tokens > summary_target_tokens)
                if should_summarize:
                    meta["summarization"]["used"] = True
                    if orig_tokens <= 8000:
                        meta["summarization"]["strategy"] = "single-pass"
                        summary = _summarize_single_pass(
                            markdown_content, summary_target_tokens, effective_host, effective_model
                        )
                    else:
                        meta["summarization"]["strategy"] = "map-reduce"
                        summary, _dbg = _summarize_map_reduce(
                            markdown_content, summary_target_tokens, effective_host, effective_model
                        )

                    summary = summary.strip()
                    summary_tokens = _estimate_tokens_from_text(summary)
                    if (
                        summary_tokens < summary_target_tokens
                        and orig_tokens <= summary_target_tokens
                        and orig_tokens > summary_tokens
                    ):
                        meta["summarization"]["fallback_reason"] = (
                            "summary_shorter_than_target_and_original_under_target"
                        )
                        final_content = markdown_content
                    else:
                        final_content = summary
                    markdown_content = final_content
                    meta["summarization"]["final_estimated_tokens"] = _estimate_tokens_from_text(markdown_content)
            except Exception as e:
                logger.warning(f"Summarization failed; returning original content. Error: {e}")
                meta["summarization"]["fallback_reason"] = f"error: {e}"
        else:
            meta["summarization"]["final_estimated_tokens"] = meta["summarization"]["original_estimated_tokens"]
        
        # Optional truncation (configurable; disabled by default)
        try:
            if truncate:
                ml = int(max_length)
                if ml > 0 and len(markdown_content) > ml:
                    markdown_content = markdown_content[:ml] + "\n\n... [content truncated]"
        except Exception:
            # If max_length is invalid, do not truncate
            pass
        
        # Return the content
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "url": url,
                        "content": markdown_content,
                        "format": "markdown",
                        "status": "success",
                        "render_method": "playwright",
                        "wait_for_selector": wait_for_selector,
                        "wait_time": wait_time,
                        "scroll_to_bottom": scroll_to_bottom,
                        "truncate": truncate,
                        "max_length": max_length,
                        **meta,
                    },
                    indent=2
                )
            )
        ]
    
    except ValueError as e:
        logger.warning(f"Invalid input: {e}")
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": str(e),
                        "url": url if "url" in locals() else None,
                        "status": "error"
                    },
                    indent=2
                )
            )
        ]
    except Exception as e:
        logger.error(f"Web URL reading failed for {url}: {e}", exc_info=True)
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": f"Web URL reading failed: {str(e)}",
                        "url": url,
                        "status": "error"
                    },
                    indent=2
                )
            )
        ]


async def install_browser():
    """Install Playwright browser."""
    print("Installing Playwright browser (Chromium)...")
    import subprocess
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"])
    print("Browser installed successfully!")


async def cleanup():
    """Cleanup browser instance on shutdown."""
    global browser_instance
    if browser_instance:
        await browser_instance.close()
        browser_instance = None


async def main():
    """Main entry point for the MCP server."""
    # Check if we need to install browser
    if len(sys.argv) > 1 and sys.argv[1] == "--install-browser":
        await install_browser()
        return
    
    try:
        logger.debug("MCP server main starting")
        try:
            stdin_isatty = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
        except Exception:
            stdin_isatty = None
        try:
            stdout_isatty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        except Exception:
            stdout_isatty = None
        try:
            fd0 = os.fstat(0)
            fd1 = os.fstat(1)
            fd0_mode = fd0.st_mode
            fd1_mode = fd1.st_mode
            fd0_kind = (
                "tty" if _stat.S_ISCHR(fd0_mode) else "fifo" if _stat.S_ISFIFO(fd0_mode) else "file" if _stat.S_ISREG(fd0_mode) else str(fd0_mode)
            )
            fd1_kind = (
                "tty" if _stat.S_ISCHR(fd1_mode) else "fifo" if _stat.S_ISFIFO(fd1_mode) else "file" if _stat.S_ISREG(fd1_mode) else str(fd1_mode)
            )
        except Exception as e:
            fd0_kind = fd1_kind = f"err:{e}"
        logger.debug(
            "stdio details: stdin_isatty=%s stdout_isatty=%s fd0=%s fd1=%s ENV={TERM=%s SHELL=%s}",
            stdin_isatty,
            stdout_isatty,
            fd0_kind,
            fd1_kind,
            os.environ.get("TERM"),
            os.environ.get("SHELL"),
        )
        # Run the server using stdin/stdout
        logger.debug("Entering stdio_server context")
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.debug("Entered stdio_server; starting server.run")
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="amazing-web-url-reader",
                    server_version="2.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
            logger.debug("server.run returned")
    finally:
        await cleanup()



if __name__ == "__main__":
    import asyncio
    from mcp.server import InitializationOptions
    
    asyncio.run(main())
