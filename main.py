"""Entry point that proxies to the Playwright-based MCP server."""

import asyncio

from amazing_web_url_reader import main as run_server


def main() -> None:
    """Run the async server entry point inside an event loop."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
