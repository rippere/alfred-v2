"""Centralized retrieval defaults for the MCP surfaces.

One place to tune QueryOptions for ``vault_query`` across all three MCP
servers (stdio, HTTP, and the meta server's per-vault fan-out). Before this
module existed the same literals were copy-pasted into each server and had
already started to drift — tune them here and every surface follows.
"""
from __future__ import annotations

from typing import Any

# Number of chunks retrieved per query (per vault, for the meta server).
DEFAULT_TOP_K = 8

# Retrieval-pipeline toggles.
USE_HOPFIELD = True
USE_GRAPH = True
INCLUDE_INBOX = False


def build_query_options(
    top_k: int = DEFAULT_TOP_K,
    include_synthesis: bool = True,
) -> Any:
    """Construct a QueryOptions carrying the centralized retrieval defaults.

    Import is deferred so that importing this module stays cheap (matches
    the lazy-import style of the server modules).
    """
    from alfred.query.engine import QueryOptions

    return QueryOptions(
        top_k=top_k,
        use_hopfield=USE_HOPFIELD,
        use_graph=USE_GRAPH,
        include_synthesis=include_synthesis,
        include_inbox=INCLUDE_INBOX,
    )
