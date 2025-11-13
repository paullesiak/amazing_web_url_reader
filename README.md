# Amazing Web URL Reader

Playwright-backed MCP server that renders modern web apps in Chromium, strips noise, converts the DOM to Markdown, and (optionally) summarizes the result for downstream LLM tooling. By default we send `Accept: text/markdown, text/plain;q=0.99, text/html;q=0.9, */*;q=0.8`; override via `AMAZING_READER_ACCEPT_HEADER` if your client needs something else.

## Quick Start
1. Install the Playwright Chromium binary once:
   ```bash
   uvx --from git+https://github.com/paullesiak/amazing_web_url_reader python amazing_web_url_reader.py --install-browser
   ```
2. Point your MCP-aware agent at the same `uvx` command (see examples below). The executable automatically stays current with the repository’s main branch.

## Configuration Options
| Variable | Required? | Default | Purpose |
| --- | --- | --- | --- |
| `AMAZING_READER_ACCEPT_HEADER` | optional | `text/markdown, text/plain;q=0.99, text/html;q=0.9, */*;q=0.8` | Advertise a different `Accept` header to the target site. |
| `AMAZING_READER_SUMMARY_TOKENS` | optional | unset (no summarization) | Global fallback token target when `use_ollama_summarization` is true but the request omits `summary_target_tokens`. |
| `OLLAMA_HOST` | optional | `http://ollamalb.local:11434` | Default host for the summarization backend. |
| `OLLAMA_MODEL` | optional | `gpt-oss:20b` | Default LLM for summarization. |
| `MCP_DEBUG` | optional | unset | Set to `1` for verbose logging (stdio wiring, tracing, etc.). |

Tool arguments (`read_web_url_amazing`) mirror the MCP schema: `url` (required) plus optional `wait_for_selector`, `wait_time`, `scroll_to_bottom`, `truncate`, `max_length`, `use_ollama_summarization`, `summary_target_tokens`, `ollama_host`, and `ollama_model`.

## MCP Agent Examples

### OpenAI Codex (`configuration.toml`)
```toml
[[mcp_servers]]
name = "amazing-web-url-reader"
type = "stdio"
command = "uvx"
args = [
  "--from",  # required: tell uvx to pull straight from GitHub
  "git+https://github.com/paullesiak/amazing_web_url_reader",
  "python",
  "amazing_web_url_reader.py"  # use main.py if you prefer the shim; both work
]

[mcp_servers.env]
# Optional overrides — delete the ones you don't need.
AMAZING_READER_ACCEPT_HEADER = "text/markdown, text/plain;q=0.99, text/html;q=0.9, */*;q=0.8"
AMAZING_READER_SUMMARY_TOKENS = "1500"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gpt-oss:20b"
MCP_DEBUG = "0"
```

### Claude Code (`mcpServers` JSON)
```json
{
  "mcpServers": {
    "amazing-web-reader": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/paullesiak/amazing_web_url_reader",
        "python",
        "main.py"
      ],
      "env": {
        "AMAZING_READER_SUMMARY_TOKENS": "0",
        "AMAZING_READER_ACCEPT_HEADER": "text/markdown, text/plain;q=0.99, text/html;q=0.9, */*;q=0.8",
        "OLLAMA_HOST": "http://localhost:11434",
        "OLLAMA_MODEL": "gpt-oss:20b"
      }
    }
  }
}
```

These snippets demonstrate every environment variable the server understands; feel free to strip them back to only the ones you care about. Re-running your agent automatically fetches the latest code from GitHub via `uvx`.
