"""Pytest suite for main.py."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from main import main as main_entry_point


@patch("main.run_server")
def test_main_entry_point(mock_run_server):
    """Test that the main entry point calls run_server."""
    # Given
    loop = asyncio.new_event_loop()
    mock_run_server.return_value = loop.create_future()
    mock_run_server.return_value.set_result(None)

    # When
    main_entry_point()

    # Then
    mock_run_server.assert_called_once()
    loop.close()
