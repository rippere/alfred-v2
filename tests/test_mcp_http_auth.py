"""MCP HTTP Bearer-auth enforcement (regression: inert auth middleware).

The original code reached for the raw ASGI app via ``mcp.app`` / ``mcp._app``.
On fastmcp 3.2.4 FastMCP has neither (only ``.http_app``), so the getattr chain
returned None, ``add_middleware`` was silently skipped, and no exception fired —
setting ALFRED_HTTP_TOKEN was a no-op and the server ran UNAUTHENTICATED while
every config- and grep-level check passed.

These tests drive run_server() itself (with uvicorn.run patched out so nothing
binds a port) and assert on the app it actually hands to the server, so they
cover the real wiring rather than a reimplementation of it.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from alfred.config import AlfredConfig
from alfred.mcp import server_http

TOKEN = "test-secret-token"

INIT_BODY = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "regression-probe", "version": "1"},
    },
}
INIT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Minimal on-disk config; run_server() loads via AlfredConfig.load()."""
    vault = tmp_path / "vault"
    vault.mkdir()
    (tmp_path / "data").mkdir()  # AlfredConfig.load() creates this in production
    cfg_path = tmp_path / "config-test.yaml"
    cfg_path.write_text(
        "vault:\n"
        f"  path: {vault}\n"
        f"data_dir: {tmp_path / 'data'}\n"
    )
    AlfredConfig.load(cfg_path)  # fail fast here if the fixture config is bad
    return cfg_path


#: Never the production port. alfred-mcp-http.service owns 127.0.0.1:8765 and is a
#: live dependency of Ben's MCP clients. These tests bind nothing (uvicorn.run is
#: patched out), but an explicit off-production port is passed anyway so that no
#: bug in the code under test can ever reach for 8765 via run_server's default.
SAFE_PORT = 8799


@pytest.fixture
def served_app(monkeypatch, config_path):
    """Run run_server() up to the point of binding; return the app it built.

    uvicorn.run is patched to capture-and-return instead of blocking, so no port
    is ever bound. Belt-and-braces: fastmcp's own serving paths are stubbed too,
    so a regression that goes back to mcp.run() fails loudly here instead of
    quietly binding a real socket.
    """

    def _serve(config_path_arg=config_path, port=SAFE_PORT, **kwargs):
        captured = {}

        import fastmcp
        import uvicorn

        def fake_run(app, **run_kwargs):
            captured["app"] = app
            captured["kwargs"] = run_kwargs

        def _no_bind(*a, **kw):
            raise AssertionError(
                "run_server() tried to serve via FastMCP.run()/run_async() instead of "
                "handing an explicitly-built app to uvicorn.run(). That path binds a "
                "real socket and cannot carry verified auth middleware."
            )

        monkeypatch.setattr(uvicorn, "run", fake_run)
        monkeypatch.setattr(fastmcp.FastMCP, "run", _no_bind)
        monkeypatch.setattr(fastmcp.FastMCP, "run_http_async", _no_bind)

        server_http.run_server(config_path_arg, port=port, **kwargs)
        assert "app" in captured, (
            "run_server() never handed an app to uvicorn.run — the auth "
            "middleware cannot have been attached to the served app."
        )
        assert captured["kwargs"].get("port") == port, (
            f"run_server(port={port}) served on {captured['kwargs'].get('port')!r}; "
            "the port argument is being ignored."
        )
        return captured["app"]

    return _serve


def test_wrong_bearer_token_is_rejected(monkeypatch, served_app):
    """The regression: a wrong token must 401, not sail through to the MCP handler."""
    monkeypatch.setenv("ALFRED_HTTP_TOKEN", TOKEN)
    app = served_app()

    with TestClient(app) as client:
        resp = client.post(
            "/mcp",
            json=INIT_BODY,
            headers={**INIT_HEADERS, "Authorization": "Bearer WRONG-TOKEN"},
        )

    assert resp.status_code == 401, (
        f"expected 401 for a wrong Bearer token, got {resp.status_code}. "
        "ALFRED_HTTP_TOKEN is set but not enforced — the server is unauthenticated."
    )


def test_missing_auth_header_is_rejected(monkeypatch, served_app):
    monkeypatch.setenv("ALFRED_HTTP_TOKEN", TOKEN)
    app = served_app()

    with TestClient(app) as client:
        resp = client.post("/mcp", json=INIT_BODY, headers=INIT_HEADERS)

    assert resp.status_code == 401


def test_correct_bearer_token_is_accepted(monkeypatch, served_app):
    """Auth must not be so eager it locks out legitimate callers."""
    monkeypatch.setenv("ALFRED_HTTP_TOKEN", TOKEN)
    app = served_app()

    with TestClient(app) as client:
        resp = client.post(
            "/mcp",
            json=INIT_BODY,
            headers={**INIT_HEADERS, "Authorization": f"Bearer {TOKEN}"},
        )

    assert resp.status_code == 200, f"correct token was rejected: {resp.status_code}"
    assert '"serverInfo"' in resp.text


def test_auth_middleware_is_actually_in_the_asgi_stack(monkeypatch, served_app):
    """Guard the specific failure mode: the middleware silently never attaching."""
    monkeypatch.setenv("ALFRED_HTTP_TOKEN", TOKEN)
    app = served_app()

    names = [mw.cls.__name__ for mw in app.user_middleware]
    assert "BearerAuthMiddleware" in names, (
        f"BearerAuthMiddleware missing from the ASGI stack: {names}"
    )


def test_no_token_configured_leaves_server_open(monkeypatch, served_app):
    """Unset token preserves the documented unauthenticated local-only behaviour."""
    monkeypatch.delenv("ALFRED_HTTP_TOKEN", raising=False)
    app = served_app()

    names = [mw.cls.__name__ for mw in app.user_middleware]
    assert "BearerAuthMiddleware" not in names

    with TestClient(app) as client:
        resp = client.post("/mcp", json=INIT_BODY, headers=INIT_HEADERS)
    assert resp.status_code == 200


def test_unattachable_middleware_is_fatal_not_silent(monkeypatch, config_path):
    """The lesson of the bug: a token that cannot be enforced must kill startup.

    Simulates add_middleware() being a no-op (exactly what the old getattr-chain
    bug amounted to) and asserts we exit rather than serve unauthenticated.
    """
    monkeypatch.setenv("ALFRED_HTTP_TOKEN", TOKEN)

    class _NoOpApp:
        user_middleware: list = []

        def add_middleware(self, cls, **kwargs):
            pass  # silently does nothing, like the old code path

    with pytest.raises(SystemExit) as excinfo:
        server_http._attach_auth_middleware(_NoOpApp(), TOKEN)

    assert "Refusing to start an unauthenticated server" in str(excinfo.value)


def test_add_middleware_raising_is_fatal_not_silent(monkeypatch):
    """The old code caught this and started anyway with only a warning."""

    class _BoomApp:
        user_middleware: list = []

        def add_middleware(self, cls, **kwargs):
            raise RuntimeError("cannot add middleware after startup")

    with pytest.raises(SystemExit) as excinfo:
        server_http._attach_auth_middleware(_BoomApp(), TOKEN)

    assert "Refusing to start an unauthenticated server" in str(excinfo.value)


def test_set_but_empty_token_is_fatal_not_open(monkeypatch, served_app):
    """A configured-but-blank token must never degrade to an open server.

    Environment="ALFRED_HTTP_TOKEN=${SECRET}" with SECRET unset expands to "".
    A truthiness check treats that as "no auth configured" and serves wide open
    while every config-level check reports auth is on — the same class of silent
    false guarantee this module exists to prevent.
    """
    monkeypatch.setenv("ALFRED_HTTP_TOKEN", "")
    with pytest.raises(SystemExit) as excinfo:
        served_app()
    assert "empty" in str(excinfo.value).lower()


def test_whitespace_only_token_is_fatal_not_open(monkeypatch, served_app):
    monkeypatch.setenv("ALFRED_HTTP_TOKEN", "   ")
    with pytest.raises(SystemExit) as excinfo:
        served_app()
    assert "empty" in str(excinfo.value).lower()


def test_token_comparison_is_constant_time(monkeypatch, served_app):
    """Guards against a regression back to `==`, which leaks the token via timing."""
    import inspect

    src = inspect.getsource(server_http._make_auth_middleware)
    assert "compare_digest" in src, "Bearer comparison must be constant-time"

    # And it must still actually work.
    monkeypatch.setenv("ALFRED_HTTP_TOKEN", TOKEN)
    with TestClient(served_app()) as client:
        ok = client.post(
            "/mcp", json=INIT_BODY, headers={**INIT_HEADERS, "Authorization": f"Bearer {TOKEN}"}
        )
        bad = client.post(
            "/mcp", json=INIT_BODY, headers={**INIT_HEADERS, "Authorization": "Bearer wrong"}
        )
    assert ok.status_code == 200
    assert bad.status_code == 401
