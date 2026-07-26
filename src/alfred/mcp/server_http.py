"""Alfred MCP HTTP server — streamable-http transport for Claude Code clients.

This module exposes the same vault tools as server.py but over HTTP, by default
on 127.0.0.1:8765.  ALFRED_HTTP_HOST/ALFRED_HTTP_PORT override that; the default
stays loopback so widening the bind is always a deliberate act, and a
non-loopback host without ALFRED_HTTP_TOKEN is refused outright.

The deployed unit binds the Tailscale interface address specifically (never
0.0.0.0) so the vault is reachable from the tailnet but not from the LAN.

The app is mounted at /mcp (fastmcp's streamable-http default). There is no
/query route — RUNBOOK.md and scripts/content-brief.sh still reference one and
have been getting 404s independently of the transport change.

Optional auth: set ALFRED_HTTP_TOKEN in the environment to require a Bearer
token in the Authorization header.  When the variable is unset the server
allows unauthenticated connections (preserving local-only behaviour).

If the token IS set, enforcement is all-or-nothing: the server verifies the
auth middleware is really in the ASGI stack and exits rather than serve
unauthenticated.  A configured-but-unenforced token is a false safety
guarantee — it passes every config-level check while the door stands open.

The stdio server (server.py) for the desktop remains unchanged.

Tool bodies live in alfred.mcp.tools (shared with the stdio and meta
servers); this module only owns the HTTP transport + Bearer-auth specifics.
"""
from __future__ import annotations

import hmac
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
            # Constant-time: a short-circuiting == leaks the token byte-by-byte via timing.
            if hmac.compare_digest(auth_header, f"Bearer {token}"):
                return await call_next(request)
            return Response("Unauthorized", status_code=401)

    return BearerAuthMiddleware


def _attach_auth_middleware(app, token: str) -> None:
    """Attach Bearer auth to *app*, or refuse to start.

    A token that is configured but not enforced is worse than no token at all:
    every config- and grep-level check passes while the server stays wide open.
    So this does not treat "no exception was raised" as proof of attachment —
    it reads the ASGI stack back and confirms the middleware actually landed.
    Any failure is fatal by design.
    """
    AuthMiddleware = _make_auth_middleware(token)
    try:
        app.add_middleware(AuthMiddleware)
    except Exception as e:  # noqa: BLE001 — any failure here must be fatal
        raise SystemExit(
            "ALFRED_HTTP_TOKEN is set but the Bearer-auth middleware could not be "
            f"attached ({e!r}). Refusing to start an unauthenticated server."
        ) from e

    attached = [mw.cls for mw in getattr(app, "user_middleware", [])]
    if AuthMiddleware not in attached:
        raise SystemExit(
            "ALFRED_HTTP_TOKEN is set but the Bearer-auth middleware is absent from "
            f"the ASGI stack after add_middleware() (stack: {[c.__name__ for c in attached]}). "
            "Refusing to start an unauthenticated server."
        )


def run_server(config_path: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Start the FastMCP HTTP/SSE server. Blocks until interrupted."""
    try:
        import fastmcp
        import uvicorn
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

    # Build the ASGI app explicitly rather than letting mcp.run() own it, so the
    # auth middleware can be attached to a real app and verified before we bind.
    app = mcp.http_app(transport="streamable-http")

    # Optional Bearer-token authentication. Unset -> open (documented mode). But SET is
    # all-or-nothing: a set-but-empty token must never degrade to open, because that is
    # exactly how it happens in practice — Environment="ALFRED_HTTP_TOKEN=${SECRET}" with
    # SECRET unset expands to "" and the operator believes the door is locked.
    http_token = os.environ.get("ALFRED_HTTP_TOKEN")
    if http_token is not None and not http_token.strip():
        raise SystemExit(
            "ALFRED_HTTP_TOKEN is set but empty/whitespace. Refusing to start: a "
            "configured-but-blank token reads as 'auth on' to every config check while "
            "serving unauthenticated. Unset it to run open, or give it a real value."
        )
    if http_token:
        _attach_auth_middleware(app, http_token)  # fatal if it cannot be enforced
        log.info("mcp.auth_enabled", host=host, port=port)
    else:
        log.warning("mcp.auth_disabled", host=host, port=port)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    _config = os.environ.get("ALFRED_CONFIG") or str(Path(__file__).parents[3] / "config.yaml")
    # Host/port are overridable so the unit can bind the tailnet interface for
    # laptop access. Defaults stay loopback: widening the bind has to be a
    # deliberate act, and it must be paired with ALFRED_HTTP_TOKEN — otherwise
    # the vault's whole query surface is served to the network unauthenticated.
    _host = os.environ.get("ALFRED_HTTP_HOST", "127.0.0.1")
    _port = int(os.environ.get("ALFRED_HTTP_PORT", "8765"))
    if _host != "127.0.0.1" and not os.environ.get("ALFRED_HTTP_TOKEN"):
        raise SystemExit(
            f"ALFRED_HTTP_HOST={_host} is non-loopback but ALFRED_HTTP_TOKEN is unset. "
            "Refusing to serve the vault to the network without auth."
        )
    run_server(Path(_config), host=_host, port=_port)
