"""Shared MCP tool registration (audit structural #5).

Proves register_tools() binds exactly the seven vault tools onto a FastMCP
instance (the stdio-style consumption pattern), that the registered wrappers
delegate to the shared *_impl bodies with the bound deps, and that all three
server modules' registration paths import cleanly.
"""
from __future__ import annotations

import asyncio
import importlib

import fastmcp
import pytest

from alfred.mcp.tools import ToolDeps, register_tools

EXPECTED_TOOLS = {
    "vault_query",
    "vault_search",
    "vault_entity_lookup",
    "vault_read_record",
    "vault_status",
    "vault_api_cost",
    "vault_feedback",
}


class _StubStateStore:
    """Just enough surface for vault_status_impl / vault_api_cost_impl."""

    def __init__(self):
        class _State:
            api_calls_date = "2026-07-13"
            api_calls_today = 7
            api_cost_usd_today = 0.123456

        self.state = _State()

    def load(self):
        return self.state

    def file_count(self):
        return 42

    def embedded_count(self):
        return 40

    def chunk_count(self):
        return 100

    def cluster_count(self):
        return 5

    def wiki_page_count(self):
        return 3


def _mcp_with_tools() -> tuple[fastmcp.FastMCP, ToolDeps]:
    deps = ToolDeps(cfg=object(), state_store=_StubStateStore(), engine=object())
    mcp = fastmcp.FastMCP("alfred-test")
    register_tools(mcp, deps)
    return mcp, deps


def test_register_tools_binds_the_seven_tool_names():
    mcp, _ = _mcp_with_tools()
    tools = asyncio.run(mcp.list_tools())
    assert {t.name for t in tools} == EXPECTED_TOOLS


def test_registered_tools_carry_docstrings():
    """The wrappers' docstrings are the MCP contract clients see."""
    mcp, _ = _mcp_with_tools()
    tools = asyncio.run(mcp.list_tools())
    for tool in tools:
        assert tool.description, f"{tool.name} registered without a description"


def test_registered_wrapper_delegates_to_impl_with_bound_deps():
    """Calling the registered vault_status must hit the bound state store."""
    mcp, _ = _mcp_with_tools()
    tool = asyncio.run(mcp.get_tool("vault_status"))
    result = tool.fn()
    assert result == {
        "files_tracked": 42,
        "files_embedded": 40,
        "chunks": 100,
        "clusters": 5,
        "wiki_pages": 3,
    }


def test_vault_api_cost_wrapper_delegates():
    mcp, _ = _mcp_with_tools()
    tool = asyncio.run(mcp.get_tool("vault_api_cost"))
    result = tool.fn()
    assert result == {
        "date": "2026-07-13",
        "calls_today": 7,
        "cost_usd_today": 0.123456,
    }


def test_vault_feedback_rejects_invalid_signal():
    """Shared body keeps the +1/-1 contract without touching the state store."""
    mcp, _ = _mcp_with_tools()
    tool = asyncio.run(mcp.get_tool("vault_feedback"))
    result = tool.fn(path="notes/x.md", signal=0)
    assert result == {"error": "signal must be 1 (helpful) or -1 (not helpful)"}


@pytest.mark.parametrize(
    "module",
    [
        "alfred.mcp.tools",
        "alfred.mcp.defaults",
        "alfred.mcp.server",
        "alfred.mcp.server_http",
        "alfred.mcp.meta_server",
    ],
)
def test_server_registration_paths_import_cleanly(module):
    importlib.import_module(module)
