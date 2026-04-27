"""Alfred MCP stdio server — exposes vault tools to Claude Code and other MCP clients."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def run_server(config_path: Path) -> None:
    """Start the FastMCP stdio server. Blocks until EOF."""
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
        except Exception:
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

    mcp.run(transport="stdio")


if __name__ == "__main__":
    import os
    _config = os.environ.get("ALFRED_CONFIG") or str(Path(__file__).parents[3] / "config.yaml")
    run_server(Path(_config))
