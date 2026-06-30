"""Alfred MCP HTTP/SSE server — localhost-only transport for Claude Code clients.

This module exposes the same vault tools as server.py but over HTTP/SSE on
127.0.0.1:8765.  Binding to loopback prevents accidental LAN exposure.

Optional auth: set ALFRED_HTTP_TOKEN in the environment to require a Bearer
token in the Authorization header.  When the variable is unset the server
allows unauthenticated connections (preserving local-only behaviour).

The stdio server (server.py) for the desktop remains unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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
    from alfred.query.engine import QueryEngine, QueryOptions
    from alfred.store.state import StateStore
    from alfred.core.vault_ops import vault_search as _vault_search, vault_read

    cfg = AlfredConfig.load(config_path)
    state_store = StateStore(cfg.state_path)
    state_store.load()
    engine = QueryEngine(cfg)

    mcp = fastmcp.FastMCP("alfred")

    @mcp.tool()
    def vault_query(
        query: str,
        top_k: int = 8,
        synthesis: bool = True,
    ) -> dict[str, Any]:
        """Full RAG query against the personal knowledge vault.

        Args:
            query: Natural language question or topic
            top_k: Number of chunks to retrieve (default 8)
            synthesis: Include LLM synthesis of results (default True)
        """
        opts = QueryOptions(
            top_k=top_k,
            use_hopfield=True,
            use_graph=True,
            include_synthesis=synthesis,
            include_inbox=False,
        )
        result = engine.query(query, opts)
        sources = [
            {"path": h.rel_path, "type": h.record_type, "score": round(h.rerank_score or h.score, 4)}
            for h in result.hits
        ]
        return {
            "answer": result.answer or "",
            "sources": sources,
            "wiki_hit": result.wiki_hit.rel_path if result.wiki_hit else None,
        }

    @mcp.tool()
    def vault_search(
        query: str | None = None,
        record_type: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Text search the vault. Can filter by type or status.

        Args:
            query: Optional text substring to search for in file content
            record_type: Optional type filter (e.g. 'person', 'project', 'note')
            status: Optional status filter (e.g. 'active', 'done')
            limit: Max results to return (default 20)
        """
        results = _vault_search(
            cfg.vault_path,
            grep_pattern=query,
            ignore_dirs=cfg.ignore_dirs,
        )
        if record_type:
            results = [r for r in results if r.get("type") == record_type]
        if status:
            results = [r for r in results if r.get("status") == status]
        return results[:limit]

    @mcp.tool()
    def vault_entity_lookup(entity_name: str) -> dict[str, Any]:
        """Look up a wiki entity page by name.

        Args:
            entity_name: Name of the entity (person, concept, project, etc.)
        """
        state_store.load()
        state = state_store.state
        key = entity_name.lower()
        page = state.wiki_pages.get(key)
        if not page:
            return {"found": False, "entity": entity_name}

        try:
            rec = vault_read(cfg.vault_path, page.rel_path)
            body = rec["body"]
        except Exception as e:
            log.debug("mcp.entity_body_read_failed", path=page.rel_path, error=str(e))
            body = ""

        return {
            "found": True,
            "entity": page.entity_name,
            "type": page.entity_type,
            "path": page.rel_path,
            "known_facts": page.known_facts,
            "related": page.related,
            "sources": page.sources,
            "body": body[:2000],
        }

    @mcp.tool()
    def vault_read_record(rel_path: str) -> dict[str, Any]:
        """Read a vault record by its relative path.

        Args:
            rel_path: Relative path within the vault (e.g. 'people/Alice.md')
        """
        try:
            rec = vault_read(cfg.vault_path, rel_path)
            return {
                "path": rec["path"],
                "frontmatter": rec["frontmatter"],
                "body": rec["body"][:4000],
            }
        except Exception as e:
            return {"error": str(e), "path": rel_path}

    @mcp.tool()
    def vault_status() -> dict[str, Any]:
        """Return current vault statistics."""
        state_store.load()
        return {
            "files_tracked": state_store.file_count(),
            "files_embedded": state_store.embedded_count(),
            "chunks": state_store.chunk_count(),
            "clusters": state_store.cluster_count(),
            "wiki_pages": state_store.wiki_page_count(),
        }

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
