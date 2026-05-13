"""Singleton Anthropic client — shared across all daemon calls to avoid repeated connection setup."""
from __future__ import annotations

import os

import anthropic

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    return _client
