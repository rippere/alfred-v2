"""Alfred MCP HTTP/SSE server — localhost-only transport for Claude Code clients.

This module exposes the same vault tools as server.py but over HTTP/SSE on
127.0.0.1:8765.  Binding to loopback prevents accidental LAN exposure.

Optional auth: set ALFRED_HTTP_TOKEN in the environment to require a Bearer
token in the Authorization header.  When the variable is unset the server
allows unauthenticated connections (preserving local-only behaviour).

The stdio server (server.py) for the desktop remains unchanged.

Tool bodies live in alfred.mcp.tools (shared with the stdio and meta
servers); this module only owns the HTTP transport + Bearer-auth specifics.
"""
from __future__ import annotations

import os
from pathlib import Path

import structlog

log = structlog.get_logger()


def _make_auth_middleware(token: str | None):
    """Return a Starlette middleware class that validates the Bearer token.

    When *token* is None, the middleware is a transparent pass-through so
    that existing unauthenticated callers keep working.
    """
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response

    class BearerAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if token is None:
                return await call_next(request)
            auth_header = request.headers.get("Authorization", "")
            if auth_header == f"Bearer {token}":
                return await call_next(request)
            return Response("Unauthorized", status_code=401)

    return BearerAuthMiddleware


def run_server(config_path: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Start the FastMCP HTTP/SSE server. Blocks until interrupted."""
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

    # Optional Bearer-token authentication.
    # If ALFRED_HTTP_TOKEN is set, attach the middleware to the underlying
    # Starlette app before starting.  FastMCP exposes the raw ASGI app via
    # .app or ._app depending on the version — try both.
    http_token = os.environ.get("ALFRED_HTTP_TOKEN")
    if http_token:
        AuthMiddleware = _make_auth_middleware(http_token)
        try:
            raw_app = getattr(mcp, "app", None) or getattr(mcp, "_app", None)
            if raw_app is not None:
                raw_app.add_middleware(AuthMiddleware)
        except Exception as e:
            # Middleware attachment failed — fall through and start without auth
            # (safe because we're bound to loopback). Warn: a token was configured
            # but isn't being enforced, which is a security-relevant surprise.
            log.warning("mcp.auth_middleware_attach_failed", error=str(e))

    mcp.run(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    _config = os.environ.get("ALFRED_CONFIG") or str(Path(__file__).parents[3] / "config.yaml")
    run_server(Path(_config))
