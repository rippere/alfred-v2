"""Meta server numeric-bounds validation (P1-05).

The meta server's ``vault_query`` tool derives an effective top_k (falling
back to the configured ``final_top_k`` when the caller passes 0) and uses
it directly to slice the reranked results — a path that does not go
through ``build_query_options``, so it needs its own guard. This proves
that guard rejects out-of-range input instead of silently mis-slicing.
"""
from __future__ import annotations

import asyncio

import fastmcp
import pytest
import yaml

from alfred.mcp.meta_server import run_meta_server


def _register_meta_mcp(tmp_path, monkeypatch) -> fastmcp.FastMCP:
    """Run run_meta_server() with mcp.run() stubbed out, capturing the built app.

    Zero vaults configured — engines/vault_cfgs are both empty lists — so
    registration doesn't need any real vault data on disk; the guard we're
    testing fires before any vault is touched.
    """
    config_path = tmp_path / "config-meta.yaml"
    config_path.write_text(
        yaml.safe_dump({"vaults": [], "meta_server": {"top_k_per_vault": 5, "final_top_k": 8}})
    )

    captured: dict[str, fastmcp.FastMCP] = {}

    def fake_run(self, *args, **kwargs):
        captured["mcp"] = self

    monkeypatch.setattr(fastmcp.FastMCP, "run", fake_run)
    run_meta_server(config_path)
    return captured["mcp"]


def test_meta_vault_query_rejects_negative_top_k(tmp_path, monkeypatch):
    mcp = _register_meta_mcp(tmp_path, monkeypatch)
    tool = asyncio.run(mcp.get_tool("vault_query"))
    with pytest.raises(ValueError, match="top_k"):
        asyncio.run(tool.fn(query="hello", top_k=-5))


def test_meta_vault_query_rejects_excessive_top_k(tmp_path, monkeypatch):
    mcp = _register_meta_mcp(tmp_path, monkeypatch)
    tool = asyncio.run(mcp.get_tool("vault_query"))
    with pytest.raises(ValueError, match="top_k"):
        asyncio.run(tool.fn(query="hello", top_k=10_000))


def test_meta_vault_query_normal_top_k_passes_through(tmp_path, monkeypatch):
    mcp = _register_meta_mcp(tmp_path, monkeypatch)
    tool = asyncio.run(mcp.get_tool("vault_query"))
    result = asyncio.run(tool.fn(query="hello", top_k=5))
    assert result["hits"] == []
    assert result["vaults_queried"] == []
