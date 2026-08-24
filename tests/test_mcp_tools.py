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

from alfred.mcp.defaults import build_query_options, validate_result_count
from alfred.mcp.tools import ToolDeps, register_tools, vault_query_impl, vault_search_impl

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


# ---------------------------------------------------------------------------
# Numeric bounds validation (top_k / limit) — P1-05
# ---------------------------------------------------------------------------


class _StubQueryResult:
    def __init__(self):
        self.hits = []
        self.answer = "stub answer"
        self.wiki_hit = None


class _StubEngine:
    """Enough surface for vault_query_impl once validation has passed."""

    def query(self, query, opts):
        assert opts.top_k == 5  # the in-range value should reach the engine unchanged
        return _StubQueryResult()


class _StubCfg:
    def __init__(self, vault_path):
        self.vault_path = vault_path
        self.ignore_dirs = []


def test_validate_result_count_rejects_negative():
    with pytest.raises(ValueError, match="top_k"):
        validate_result_count(-5, param_name="top_k")


def test_validate_result_count_rejects_excessively_large():
    with pytest.raises(ValueError, match="top_k"):
        validate_result_count(100_000, param_name="top_k")


def test_validate_result_count_passes_through_in_range_value():
    assert validate_result_count(5, param_name="top_k") == 5


def test_build_query_options_rejects_negative_top_k():
    with pytest.raises(ValueError):
        build_query_options(top_k=-1)


def test_build_query_options_rejects_excessive_top_k():
    with pytest.raises(ValueError):
        build_query_options(top_k=999)


def test_build_query_options_normal_top_k_passes_through():
    opts = build_query_options(top_k=5)
    assert opts.top_k == 5


def test_vault_query_impl_rejects_negative_top_k():
    """A negative top_k must raise, not silently mis-slice downstream."""
    with pytest.raises(ValueError, match="top_k"):
        vault_query_impl(_StubEngine(), "hello", top_k=-5)


def test_vault_query_impl_rejects_excessive_top_k():
    with pytest.raises(ValueError, match="top_k"):
        vault_query_impl(_StubEngine(), "hello", top_k=10_000)


def test_vault_query_impl_normal_top_k_passes_through():
    result = vault_query_impl(_StubEngine(), "hello", top_k=5)
    assert result["answer"] == "stub answer"
    assert result["sources"] == []


def test_registered_vault_query_rejects_negative_top_k():
    """Same guard, exercised through the FastMCP-registered wrapper."""
    mcp, _ = _mcp_with_tools()
    tool = asyncio.run(mcp.get_tool("vault_query"))
    with pytest.raises(ValueError, match="top_k"):
        tool.fn(query="hello", top_k=-5)


def test_vault_search_impl_rejects_negative_limit(tmp_path):
    """A negative limit must raise, not silently truncate from the wrong end."""
    cfg = _StubCfg(tmp_path)
    with pytest.raises(ValueError, match="limit"):
        vault_search_impl(cfg, query=None, limit=-5)


def test_vault_search_impl_rejects_excessive_limit(tmp_path):
    cfg = _StubCfg(tmp_path)
    with pytest.raises(ValueError, match="limit"):
        vault_search_impl(cfg, query=None, limit=100_000)


def test_vault_search_impl_normal_limit_passes_through(tmp_path):
    (tmp_path / "note.md").write_text("hello world", encoding="utf-8")
    cfg = _StubCfg(tmp_path)
    results = vault_search_impl(cfg, query=None, limit=10)
    assert isinstance(results, list)
    assert len(results) <= 10


def test_registered_vault_search_rejects_negative_limit():
    mcp, _ = _mcp_with_tools()
    tool = asyncio.run(mcp.get_tool("vault_search"))
    with pytest.raises(ValueError, match="limit"):
        tool.fn(limit=-1)
