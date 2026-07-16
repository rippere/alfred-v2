"""Shared MCP tool implementations for Alfred's three MCP surfaces.

``server.py`` (stdio), ``server_http.py`` (streamable-http), and
``meta_server.py`` (multi-vault fan-out) expose the same underlying vault
operations. This module holds the single implementation of each tool body,
parameterized by its dependencies (config / state store / query engine), so
the servers register the shared code instead of reimplementing it — the
byte-for-byte copy-paste across the three servers had already drifted once
(AUDIT-2026-07-13, quick win #9 / structural #5).

Two consumption patterns:

- Single-vault servers (stdio, HTTP) call ``register_tools(mcp, deps)``,
  which binds all seven tools onto a FastMCP instance. The registered
  wrappers' signatures and docstrings are part of the MCP contract —
  clients see them — so they are preserved verbatim from the originals.
- The meta server resolves dependencies per vault (its engine-resolver
  loop over config-meta.yaml) and calls the ``*_impl`` functions directly,
  layering its own fan-out / vault-tagging / re-ranking semantics on top.
  The impls therefore take their deps as explicit arguments rather than
  assuming one global engine.

Retrieval defaults (top_k, use_hopfield, ...) live in ``alfred.mcp.defaults``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from alfred.mcp.defaults import DEFAULT_TOP_K, build_query_options, validate_result_count

log = structlog.get_logger()


@dataclass
class ToolDeps:
    """Dependencies a single-vault MCP server binds its tools to."""

    cfg: Any  # AlfredConfig
    state_store: Any  # StateStore
    engine: Any  # QueryEngine


# ---------------------------------------------------------------------------
# Tool implementations (plain functions, explicit deps)
# ---------------------------------------------------------------------------


def vault_query_impl(
    engine: Any,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    synthesis: bool = True,
) -> dict[str, Any]:
    """Full RAG query against one vault's QueryEngine."""
    opts = build_query_options(top_k=top_k, include_synthesis=synthesis)
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


def vault_search_impl(
    cfg: Any,
    query: str | None = None,
    record_type: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Text search one vault, with optional type/status filters."""
    from alfred.core.vault_ops import vault_search as _vault_search

    validate_result_count(limit, param_name="limit")

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


def vault_entity_lookup_impl(
    cfg: Any,
    state_store: Any,
    entity_name: str,
) -> dict[str, Any]:
    """Look up a wiki entity page by name in one vault's state."""
    from alfred.core.vault_ops import vault_read

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


def vault_read_record_impl(cfg: Any, rel_path: str) -> dict[str, Any]:
    """Read a vault record by its relative path."""
    from alfred.core.vault_ops import vault_read

    try:
        rec = vault_read(cfg.vault_path, rel_path)
        return {
            "path": rec["path"],
            "frontmatter": rec["frontmatter"],
            "body": rec["body"][:4000],
        }
    except Exception as e:
        return {"error": str(e), "path": rel_path}


def vault_status_impl(state_store: Any) -> dict[str, Any]:
    """Return current statistics for one vault's state store."""
    state_store.load()
    return {
        "files_tracked": state_store.file_count(),
        "files_embedded": state_store.embedded_count(),
        "chunks": state_store.chunk_count(),
        "clusters": state_store.cluster_count(),
        "wiki_pages": state_store.wiki_page_count(),
    }


def vault_api_cost_impl(state_store: Any) -> dict[str, Any]:
    """Return today's API call count and estimated cost from state.json."""
    state_store.load()
    state = state_store.state
    return {
        "date": state.api_calls_date or "no calls recorded",
        "calls_today": state.api_calls_today,
        "cost_usd_today": round(state.api_cost_usd_today, 6),
    }


def vault_feedback_impl(
    cfg: Any,
    state_store: Any,
    path: str,
    signal: int,
    query: str = "",
) -> dict[str, Any]:
    """Record user feedback (+1/-1) on a retrieved vault record."""
    import json
    from datetime import datetime, timezone

    from alfred.core.models import MemoryStrength

    if signal not in (1, -1):
        return {"error": "signal must be 1 (helpful) or -1 (not helpful)"}

    state_store.load()
    state = state_store.state

    ms = state.memory.get(path)
    if ms is None:
        ms = MemoryStrength(rel_path=path)
        state.memory[path] = ms

    if signal == 1:
        ms.update()
    else:
        ms.stability = max(0.1, ms.stability - 0.5)

    state_store.save()

    # Append to feedback log
    try:
        log_path = cfg.data_dir / "query_log.jsonl"
        entry = {
            "type": "feedback",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": path,
            "signal": signal,
            "query": query,
            "stability_after": round(ms.stability, 3),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

    return {
        "path": path,
        "signal": signal,
        "stability": round(ms.stability, 3),
        "access_count": ms.access_count,
    }


# ---------------------------------------------------------------------------
# Registration for single-vault servers (stdio + HTTP)
# ---------------------------------------------------------------------------


def register_tools(mcp: Any, deps: ToolDeps) -> None:
    """Bind the seven single-vault tools onto a FastMCP instance.

    The wrappers' names, signatures, and docstrings are the MCP contract
    that clients depend on — keep them verbatim; only the bodies delegate.
    """

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
        return vault_query_impl(deps.engine, query, top_k=top_k, synthesis=synthesis)

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
        return vault_search_impl(
            deps.cfg, query=query, record_type=record_type, status=status, limit=limit
        )

    @mcp.tool()
    def vault_entity_lookup(entity_name: str) -> dict[str, Any]:
        """Look up a wiki entity page by name.

        Args:
            entity_name: Name of the entity (person, concept, project, etc.)
        """
        return vault_entity_lookup_impl(deps.cfg, deps.state_store, entity_name)

    @mcp.tool()
    def vault_read_record(rel_path: str) -> dict[str, Any]:
        """Read a vault record by its relative path.

        Args:
            rel_path: Relative path within the vault (e.g. 'people/Alice.md')
        """
        return vault_read_record_impl(deps.cfg, rel_path)

    @mcp.tool()
    def vault_status() -> dict[str, Any]:
        """Return current vault statistics."""
        return vault_status_impl(deps.state_store)

    @mcp.tool()
    def vault_api_cost() -> dict[str, Any]:
        """Return today's Anthropic API call count and estimated cost.

        Reads the daily counters persisted in state.json. Cost is estimated
        using claude-sonnet-4-6 pricing: $3.00/M input, $0.30/M cached input,
        $15.00/M output tokens.
        """
        return vault_api_cost_impl(deps.state_store)

    @mcp.tool()
    def vault_feedback(path: str, signal: int, query: str = "") -> dict[str, Any]:
        """Record user feedback on a retrieved vault record.

        signal: 1 = helpful/relevant, -1 = not helpful/irrelevant.
        Updates the memory strength for the record so future queries
        surface (or suppress) it accordingly.
        path: relative vault path as returned by vault_query (e.g. 'sessions/my-note.md')
        query: optional — the query that surfaced this result (for audit log)
        """
        return vault_feedback_impl(deps.cfg, deps.state_store, path, signal, query=query)
