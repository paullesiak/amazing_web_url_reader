"""Pytest equivalents of the former smoke-test scripts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "amazing_web_url_reader.py"


def _send_message(proc: subprocess.Popen, obj) -> None:
    line = obj if isinstance(obj, str) else json.dumps(obj, separators=(",", ":"))
    assert proc.stdin is not None
    proc.stdin.write(line + "\n")
    proc.stdin.flush()


def _recv_message(proc: subprocess.Popen) -> str:
    assert proc.stdout is not None
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("server exited before responding")
    return line.strip()


def _build_initialize_request(id: int = 1) -> dict:
    from mcp.types import LATEST_PROTOCOL_VERSION

    return {
        "jsonrpc": "2.0",
        "id": id,
        "method": "initialize",
        "params": {
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "raw-stdio-test", "version": "0.0.1"},
        },
    }


def _build_tools_list_request(id: int = 2) -> dict:
    return {"jsonrpc": "2.0", "id": id, "method": "tools/list"}


def test_raw_stdio_handshake(tmp_path):
    assert SERVER.exists(), f"Missing server entry point: {SERVER}"

    env = os.environ.copy()
    env.setdefault("AMAZING_READER_SUMMARY_TOKENS", "0")
    env.setdefault("MCP_DEBUG", "1")

    proc = subprocess.Popen(
        [sys.executable, str(SERVER)],
        cwd=str(ROOT),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    try:
        init = _build_initialize_request()
        _send_message(proc, init)
        init_resp = json.loads(_recv_message(proc))
        assert init_resp.get("result", {}).get("serverInfo", {}).get("name") == "amazing-web-url-reader"

        _send_message(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        _send_message(proc, _build_tools_list_request())
        tools_resp = json.loads(_recv_message(proc))
        tool_names = {tool["name"] for tool in tools_resp.get("result", {}).get("tools", [])}
        assert "read_web_url_amazing" in tool_names

    finally:
        if proc.stdin:
            proc.stdin.close()
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        stderr_output = proc.stderr.read() if proc.stderr else ""
        if proc.returncode and stderr_output:
            (tmp_path / "raw-handshake.stderr").write_text(stderr_output)
