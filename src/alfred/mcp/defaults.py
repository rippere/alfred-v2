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

# Valid inclusive range for caller-supplied result-count arguments
# (top_k, limit, top_k_per_vault) across the MCP tool layer. A negative
# value silently mis-slices (e.g. ``results[:limit]`` with ``limit=-5``
# truncates from the wrong end instead of raising); an unbounded value is
# a local resource-exhaustion and Anthropic-API-cost-amplification lever
# (top_k flows into QueryEngine.query() and, in the meta server, is fanned
# out across every configured vault plus a FlashRank rerank pass).
MIN_RESULT_COUNT = 1
MAX_RESULT_COUNT = 100


def validate_result_count(value: int, *, param_name: str) -> int:
    """Validate a caller-supplied result-count argument (top_k/limit/etc).

    Raises ``ValueError`` with a clear message for out-of-range input
    rather than silently clamping (which hides the problem from the
    caller) or passing it straight through (which lets negative values
    mis-slice results and lets huge values amplify cost/resource use).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{param_name} must be an integer, got {value!r}")
    if not (MIN_RESULT_COUNT <= value <= MAX_RESULT_COUNT):
        raise ValueError(
            f"{param_name} must be between {MIN_RESULT_COUNT} and "
            f"{MAX_RESULT_COUNT} (got {value})"
        )
    return value


def build_query_options(
    top_k: int = DEFAULT_TOP_K,
    include_synthesis: bool = True,
) -> Any:
    """Construct a QueryOptions carrying the centralized retrieval defaults.

    Validates ``top_k`` (see ``validate_result_count``) before building
    the options — this is the single enforcement point for every MCP
    surface that queries a vault (stdio/HTTP single-vault tools and the
    meta server's per-vault fan-out).

    Import of QueryOptions is deferred so that importing this module
    stays cheap (matches the lazy-import style of the server modules).
    """
    validate_result_count(top_k, param_name="top_k")

    from alfred.query.engine import QueryOptions

    return QueryOptions(
        top_k=top_k,
        use_hopfield=USE_HOPFIELD,
        use_graph=USE_GRAPH,
        include_synthesis=include_synthesis,
        include_inbox=INCLUDE_INBOX,
    )
