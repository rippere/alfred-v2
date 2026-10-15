"""Vault-knowledge -> brief synthesis for the CRM bridge.

Given a contact already resolved to a vault `person/*.md` entity (see
`alfred.bridge.resolve.EntityMatch`), this module:

  1. Queries the vault's real `QueryEngine.query()` for context relevant to
     that entity — directly, not via the `vault_query` MCP wrapper
     (`alfred.mcp.tools.vault_query_impl`). The bridge runs as a plain
     library call from a Typer CLI, not over MCP, so there is nothing to
     gain from routing through the MCP tool layer; calling the engine
     directly also lets this module ask for retrieval without the engine's
     built-in generic Q&A synthesis (`include_synthesis=False`), since a CRM
     brief needs its own prompt, not "answer this query" framing.
  2. Synthesizes a short 2-4 sentence brief from that context on the vault's
     configured local backend (`alfred.core.local_llm.complete(**cfg.llm)`,
     the same path every daemon and `alfred.query.synth` use). Retrieved vault text never goes to
     a cloud LLM: the former `anthropic_client.get_client()` call here was the
     last cloud path left after the local-LLM swap, and it would have failed
     open (main-vault context to Anthropic) the moment API credits returned.
     If the local backend is down, `LocalLLMUnavailable` propagates — there is
     no fallback ladder.

If the query returns no relevant context at all, `synthesize_entity_brief`
returns `None` — "nothing worth writing" is a normal, silent outcome for a
contact with no related vault knowledge, not an error.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from alfred.bridge import config as C
from alfred.bridge.resolve import EntityMatch

SYSTEM_PROMPT = (
    "You are Alfred, a personal knowledge assistant helping prepare a CRM "
    "note about a contact. Using ONLY the provided vault context, write a "
    "short brief of 2 to 4 sentences: concrete, specific facts and history "
    "useful right before a call or meeting with this person. No greetings, "
    "no headers, no bullet points, no meta-commentary about the source "
    "material — plain prose only. If the context is thin, write 1-2 "
    "sentences rather than padding with generic filler."
)


@dataclass(frozen=True)
class Brief:
    """A synthesized brief, ready to hand to `alfred.bridge.notes.post_note`."""

    entity_rel_path: str
    entity_name: str
    text: str
    note_hash: str
    source_paths: list[str]


def build_query_text(entity: EntityMatch) -> str:
    """The natural-language query sent to the vault for this entity."""
    return f"What do we know about {entity.name}? Relevant history, context, and facts."


def query_vault_context(engine: Any, entity: EntityMatch, *, top_k: int = 6) -> Any:
    """Run the real `QueryEngine` (not the MCP wrapper) for this entity.

    Returns the engine's `QueryResult` unchanged — callers read `.context`
    and `.sources` off it. Synthesis is switched off (`include_synthesis=
    False`): the engine's built-in generic-Q&A synthesis path
    (`alfred.query.synth`) is not the brief we want, so `_call_llm` below
    does its own LLM call over the raw retrieved context instead.
    """
    from alfred.query.engine import QueryOptions

    opts = QueryOptions(top_k=top_k, include_synthesis=False)
    return engine.query(build_query_text(entity), opts)


def compute_brief_hash(entity_rel_path: str, brief_text: str) -> str:
    """Stable idempotency hash from (entity id, brief content).

    Content-derived, not random/time-based, so re-running the pipeline
    against unchanged vault knowledge always reproduces the same hash and
    `alfred.bridge.notes.has_matching_hash` recognizes "already posted"
    across runs/restarts with no local state file — the CRM's existing
    notes are the only source of truth for what's already been posted.
    """
    digest = hashlib.sha256(f"{entity_rel_path}\n{brief_text}".encode("utf-8")).hexdigest()
    return digest[:12]


def _call_llm(
    entity: EntityMatch, context: str, *, llm: Mapping[str, Any], max_tokens: int
) -> str:
    """One brief completion on the vault's configured backend (`cfg.llm`).

    `llm` carries api/base_url/model/local_only, so a local-only vault keeps
    its loopback-Ollama guard here exactly as on every daemon path. Raises
    LocalLLMUnavailable.
    """
    from alfred.core.local_llm import complete

    return complete(
        SYSTEM_PROMPT,
        f"Contact: {entity.name}\n\nVault context:\n\n{context}",
        max_tokens=max_tokens,
        **llm,
    ).strip()


def synthesize_entity_brief(
    engine: Any,
    entity: EntityMatch,
    *,
    llm: Mapping[str, Any],
    top_k: int = 6,
    max_tokens: int = C.BRIEF_MAX_TOKENS,
) -> Optional[Brief]:
    """Query the vault for `entity`, then synthesize a brief if there's
    anything worth saying.

    Returns `None` (never calls the LLM) when the vault query returns no
    usable context — a contact with no related vault knowledge is the
    expected common case, not a failure, and the caller should simply skip
    posting for that contact.

    `llm` is the vault's backend (`cfg.llm`: loopback Ollama by default, the
    Spark only where a satellite config opts in). It is required so no caller
    can fall back to a cloud default by omission.
    """
    result = query_vault_context(engine, entity, top_k=top_k)
    context = (getattr(result, "context", "") or "").strip()
    if not context:
        return None

    text = _call_llm(entity, context, llm=llm, max_tokens=max_tokens)
    if not text:
        return None

    source_paths = sorted({s.rel_path for s in getattr(result, "sources", [])})
    note_hash = compute_brief_hash(entity.rel_path, text)
    return Brief(
        entity_rel_path=entity.rel_path,
        entity_name=entity.name,
        text=text,
        note_hash=note_hash,
        source_paths=source_paths,
    )
