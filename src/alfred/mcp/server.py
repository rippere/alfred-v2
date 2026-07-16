"""Alfred MCP stdio server — exposes vault tools to Claude Code and other MCP clients.

Tool bodies live in alfred.mcp.tools (shared with the HTTP and meta servers);
this module only owns the stdio transport specifics.
"""
from __future__ import annotations

from pathlib import Path

import structlog

log = structlog.get_logger()


def run_server(config_path: Path) -> None:
    """Start the FastMCP stdio server. Blocks until EOF."""
    try:
        import fastmcp
    except ImportError:
        raise SystemExit("fastmcp not installed. Run: uv pip install fastmcp")

    from alfred.config import AlfredConfig
    from alfred.mcp.tools import ToolDeps, register_tools
    from alfred.query.engine import QueryEngine
    from alfred.store.state import StateStore

    cfg = AlfredConfig.load(config_path)
    state_store = StateStore(cfg.state_path)
    state_store.load()
    engine = QueryEngine(cfg)

    mcp = fastmcp.FastMCP("alfred")
    register_tools(mcp, ToolDeps(cfg=cfg, state_store=state_store, engine=engine))

    mcp.run(transport="stdio")


if __name__ == "__main__":
    import os
    _config = os.environ.get("ALFRED_CONFIG") or str(Path(__file__).parents[3] / "config.yaml")
    run_server(Path(_config))
