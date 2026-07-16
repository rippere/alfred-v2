"""Alfred Meta MCP server — fans out queries to all configured vault instances.

Loads vault configs from a config-meta.yaml file, fires parallel queries
against each individual vault's QueryEngine, merges results, re-ranks with
FlashRank, and returns the top-k across all vaults.

Each result includes a 'vault' field indicating which vault it came from.

Usage (stdio):
    python -m alfred.mcp.meta_server --config /path/to/config-meta.yaml
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import yaml


def _load_meta_config(config_path: Path) -> dict:
    """Load and parse config-meta.yaml."""
    raw = yaml.safe_load(config_path.read_text())
    return raw


def _build_engines(meta_raw: dict) -> list[tuple[str, Any]]:
    """Instantiate a QueryEngine for every configured vault. Returns [(name, engine)]."""
    from alfred.config import AlfredConfig
    from alfred.query.engine import QueryEngine

    engines = []
    for vault_entry in meta_raw.get("vaults", []):
        name = vault_entry["name"]
        cfg_path = Path(vault_entry["config"])
        try:
            cfg = AlfredConfig.load(cfg_path)
            engine = QueryEngine(cfg)
            engines.append((name, engine))
        except Exception as e:
            print(f"[warn] Could not load vault '{name}' from {cfg_path}: {e}", file=sys.stderr)
    return engines


def _query_one_vault(
    vault_name: str,
    engine: Any,
    query: str,
    top_k: int,
    synthesis: bool,
) -> list[dict[str, Any]]:
    """Run a query against a single vault engine. Returns list of hit dicts."""
    from alfred.mcp.defaults import build_query_options

    try:
        opts = build_query_options(top_k=top_k, include_synthesis=synthesis)
        result = engine.query(query, opts)
        hits = []
        for h in result.hits:
            hits.append({
                "vault": vault_name,
                "path": h.rel_path,
                "type": h.record_type or "unknown",
                "score": round(h.rerank_score or h.score, 4),
                "chunk": getattr(h, "chunk_index", 0),
            })
        return hits
    except Exception as e:
        print(f"[warn] Query failed for vault '{vault_name}': {e}", file=sys.stderr)
        return []


def _rerank_combined(
    hits: list[dict[str, Any]],
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Re-rank combined hits from all vaults using FlashRank, return top_k."""
    try:
        from flashrank import Ranker, RerankRequest

        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp")
        passages = [{"id": i, "text": f"{h['vault']}:{h['path']}"} for i, h in enumerate(hits)]
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        reranked = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
        return [
            {**hits[r["id"]], "rerank_score": round(r["score"], 4)}
            for r in reranked
        ]
    except Exception:
        # FlashRank unavailable or failed — fall back to original scores
        deduped = {(h["vault"], h["path"]): h for h in hits}
        sorted_hits = sorted(deduped.values(), key=lambda h: h["score"], reverse=True)
        return sorted_hits[:top_k]


async def _parallel_query(
    engines: list[tuple[str, Any]],
    query: str,
    top_k_per_vault: int,
    synthesis: bool,
) -> list[dict[str, Any]]:
    """Fire all vault queries in parallel using asyncio.gather."""
    loop = asyncio.get_event_loop()

    async def run_one(vault_name: str, engine: Any) -> list[dict[str, Any]]:
        return await loop.run_in_executor(
            None,
            _query_one_vault,
            vault_name,
            engine,
            query,
            top_k_per_vault,
            synthesis,
        )

    results = await asyncio.gather(*[run_one(name, eng) for name, eng in engines])
    combined = []
    for vault_hits in results:
        combined.extend(vault_hits)
    return combined


async def _parallel_search(
    vault_cfgs: list[tuple[str, Any]],  # (name, cfg)
    query: str | None,
    record_type: str | None,
    status: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Fire vault_search across all vaults in parallel."""
    from alfred.mcp.tools import vault_search_impl

    async def run_one(vault_name: str, cfg: Any) -> list[dict[str, Any]]:
        loop = asyncio.get_event_loop()

        def _search():
            results = vault_search_impl(
                cfg,
                query=query,
                record_type=record_type,
                status=status,
                limit=limit,
            )
            for r in results:
                r["vault"] = vault_name
            return results

        return await loop.run_in_executor(None, _search)

    results = await asyncio.gather(*[run_one(name, cfg) for name, cfg in vault_cfgs])
    combined = []
    for vault_results in results:
        combined.extend(vault_results)
    return combined[:limit]


def run_meta_server(meta_config_path: Path) -> None:
    """Start the FastMCP meta stdio server. Blocks until EOF."""
    try:
        import fastmcp
    except ImportError:
        raise SystemExit("fastmcp not installed. Run: uv pip install fastmcp")

    from alfred.config import AlfredConfig
    from alfred.store.state import StateStore

    meta_raw = _load_meta_config(meta_config_path)
    meta_cfg = meta_raw.get("meta_server", {})
    top_k_per_vault = meta_cfg.get("top_k_per_vault", 5)
    final_top_k = meta_cfg.get("final_top_k", 8)

    # Build engines for query
    engines = _build_engines(meta_raw)

    # Build (name, cfg) pairs for search and entity_lookup
    vault_cfgs: list[tuple[str, Any]] = []
    for vault_entry in meta_raw.get("vaults", []):
        name = vault_entry["name"]
        cfg_path = Path(vault_entry["config"])
        try:
            cfg = AlfredConfig.load(cfg_path)
            vault_cfgs.append((name, cfg))
        except Exception as e:
            print(f"[warn] Skipping vault '{name}' for search: {e}", file=sys.stderr)

    mcp = fastmcp.FastMCP("alfred-meta")

    @mcp.tool()
    async def vault_query(
        query: str,
        top_k: int = 8,
        synthesis: bool = False,
    ) -> dict[str, Any]:
        """Full RAG query across ALL knowledge vaults (ai-systems, neuroscience, finance, personal).

        Fires parallel queries against every vault, merges and re-ranks results using
        FlashRank, returns the top-k most relevant hits across all domains.

        Args:
            query: Natural language question or topic
            top_k: Total number of results to return across all vaults (default 8)
            synthesis: Include LLM synthesis of results (default False — expensive)
        """
        from alfred.mcp.defaults import validate_result_count

        # top_k=0 is a sentinel meaning "use the server's final_top_k
        # default" (preserved pre-existing behavior); any other value is
        # validated against the shared 1-100 range before it's used to
        # slice the reranked results, so a negative value can't silently
        # mis-slice and a huge value can't blow up the rerank pass.
        effective_top_k = top_k if top_k else final_top_k
        validate_result_count(effective_top_k, param_name="top_k")

        all_hits = await _parallel_query(engines, query, top_k_per_vault, synthesis)
        ranked = _rerank_combined(all_hits, query, effective_top_k)
        return {
            "hits": ranked,
            "vaults_queried": [name for name, _ in engines],
            "total_candidates": len(all_hits),
        }

    @mcp.tool()
    async def vault_search(
        query: str | None = None,
        record_type: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Text search across ALL vaults. Can filter by type or status.

        Each result includes a 'vault' field indicating which vault it came from.

        Args:
            query: Optional text substring to search for in file content
            record_type: Optional type filter (e.g. 'person', 'project', 'note')
            status: Optional status filter (e.g. 'active', 'done')
            limit: Max results per vault (default 20)
        """
        return await _parallel_search(vault_cfgs, query, record_type, status, limit)

    @mcp.tool()
    def vault_entity_lookup(entity_name: str) -> list[dict[str, Any]]:
        """Look up a wiki entity by name across ALL vaults.

        Returns matches from every vault, each tagged with its source vault.

        Args:
            entity_name: Name of the entity (person, concept, project, etc.)
        """
        from alfred.mcp.tools import vault_entity_lookup_impl
        from alfred.store.state import StateStore

        all_results = []

        for vault_name, cfg in vault_cfgs:
            try:
                res = vault_entity_lookup_impl(cfg, StateStore(cfg.state_path), entity_name)
                if not res["found"]:
                    continue
                all_results.append({"vault": vault_name, **res})
            except Exception as e:
                print(f"[warn] entity_lookup failed for vault '{vault_name}': {e}", file=sys.stderr)

        if not all_results:
            return [{"found": False, "entity": entity_name}]
        return all_results

    @mcp.tool()
    def vault_status() -> dict[str, Any]:
        """Return statistics for all configured vaults."""
        from alfred.mcp.tools import vault_status_impl
        from alfred.store.state import StateStore

        per_vault = {}
        for vault_name, cfg in vault_cfgs:
            try:
                stats = vault_status_impl(StateStore(cfg.state_path))
                per_vault[vault_name] = {"vault_path": str(cfg.vault_path), **stats}
            except Exception as e:
                per_vault[vault_name] = {"error": str(e)}

        return {
            "vaults": per_vault,
            "engines_loaded": [name for name, _ in engines],
        }

    mcp.run(transport="stdio")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Alfred Meta MCP server")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[4] / "config-meta.yaml",
        help="Path to config-meta.yaml",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[error] config-meta.yaml not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    run_meta_server(args.config)


if __name__ == "__main__":
    main()
