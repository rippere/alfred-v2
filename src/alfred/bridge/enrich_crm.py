"""Orchestration entry point for the CRM bridge: `alfred bridge enrich`.

Flow, per active deal:
    1. List active NovaCRM deals (GET .../deals, closed stages excluded
       client-side — see `alfred.bridge.config.CLOSED_DEAL_STAGES`).
    2. For each deal's single contact (`Deal.contact_id`), fetch the full
       contact record (GET .../contacts/{id}) to get its email.
    3. Resolve the contact to a vault entity by email
       (`alfred.bridge.resolve.resolve_contact_to_entity`) — email-exact-match
       only; no match is dropped, never guessed.
    4. For a match, synthesize a brief from vault knowledge
       (`alfred.bridge.brief.synthesize_entity_brief`) — `None` means "nothing
       worth writing", a normal outcome, not an error.
    5. For a brief, post it as a note (`alfred.bridge.notes.post_note`) with
       `dry_run` defaulting to True. The note is posted on the DEAL (not the
       contact): Pattern A is framed as *deal* enrichment, the deal is what
       drove inclusion in this run, and per-deal idempotency means a contact
       who sits on several deals gets the same brief attached to each deal
       page rather than a single global note the operator has to hunt for.

Every decision (matched / no-match, brief synthesized / skipped,
would-post / skipped-duplicate / posted / error) is logged via structlog and
recorded in the returned `EnrichSummary.decisions` list, so a dry run can be
audited without re-running anything.

Read (GET) calls always happen, even in dry-run — resolution, synthesis, and
the duplicate check are exactly what a dry run is supposed to surface. Only
the note-creation POST is gated by `dry_run` (see `alfred.bridge.notes.
post_note`). If required credentials are absent, nothing is fetched at all —
`run_enrich` returns an all-zero summary rather than guessing at deal data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import structlog

from alfred.bridge import config as C
from alfred.bridge.brief import synthesize_entity_brief
from alfred.bridge.notes import post_note
from alfred.bridge.resolve import resolve_contact_to_entity
from alfred.ledger.push import _AuthError, _authenticate, ensure_env_scaffold

log = structlog.get_logger()


@dataclass
class EnrichSummary:
    """Counts + per-contact decision log for one `run_enrich` invocation."""

    deals_seen: int = 0
    contacts_seen: int = 0
    matched: int = 0
    skipped_no_match: int = 0
    briefs_synthesized: int = 0
    skipped_no_brief: int = 0
    would_post: int = 0
    posted: int = 0
    skipped_duplicate: int = 0
    errors: int = 0
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, int]:
        """Counts only (no decisions) — the shape the CLI table renders."""
        return {
            "deals_seen": self.deals_seen,
            "contacts_seen": self.contacts_seen,
            "matched": self.matched,
            "skipped_no_match": self.skipped_no_match,
            "briefs_synthesized": self.briefs_synthesized,
            "skipped_no_brief": self.skipped_no_brief,
            "would_post": self.would_post,
            "posted": self.posted,
            "skipped_duplicate": self.skipped_duplicate,
            "errors": self.errors,
        }


def fetch_active_deals(
    api_url: str, workspace_id: str, token: str, *, timeout: float = 30.0
) -> list[dict]:
    """GET .../deals, filtered client-side to exclude closed_won/closed_lost.

    There is no server-side "active" filter (stage is a free string column)
    — see `alfred.bridge.config.CLOSED_DEAL_STAGES`.
    """
    resp = httpx.get(
        f"{api_url}/workspaces/{workspace_id}/deals",
        headers={"Authorization": f"Bearer {token}"},
        params={"limit": 500},
        timeout=timeout,
    )
    resp.raise_for_status()
    deals = resp.json()
    return [d for d in deals if d.get("stage") not in C.CLOSED_DEAL_STAGES]


def fetch_contact(
    api_url: str, workspace_id: str, contact_id: str, token: str, *, timeout: float = 30.0
) -> Optional[dict]:
    """GET .../contacts/{id}. Returns None on 404 (contact vanished/bad id)."""
    resp = httpx.get(
        f"{api_url}/workspaces/{workspace_id}/contacts/{contact_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def run_enrich(
    cfg: Any,
    engine: Any,
    *,
    dry_run: bool = True,
    top_k: int = 6,
    timeout: float = 30.0,
) -> EnrichSummary:
    """Run one full enrichment pass. `cfg` is an `AlfredConfig`, `engine` a
    `QueryEngine` (both passed in rather than constructed here, so tests can
    supply fakes without touching the vault/model config).

    Degrade behavior mirrors `alfred.ledger.push.push_snapshot`: missing
    credentials or a failed auth grant short-circuits before any deal/contact
    GET is made, returning an all-zero (or errors=1) summary rather than a
    partial/garbled run.
    """
    summary = EnrichSummary()

    ensure_env_scaffold()
    env = C.load_push_env()
    missing = [k for k in C.PUSH_ENV_KEYS if not env.get(k)]
    if missing:
        log.warning("bridge.enrich.no_creds", missing=missing)
        return summary

    token = _authenticate(env, timeout=timeout)
    if token is None or isinstance(token, _AuthError):
        detail = token.detail if isinstance(token, _AuthError) else "no access_token returned"
        log.error("bridge.enrich.auth_failed", detail=detail)
        summary.errors += 1
        return summary

    api_url = env["LEDGER_API_URL"].rstrip("/")
    workspace_id = env["LEDGER_WORKSPACE_ID"]

    try:
        deals = fetch_active_deals(api_url, workspace_id, token, timeout=timeout)
    except httpx.HTTPError as e:
        log.error("bridge.enrich.list_deals_failed", error=str(e))
        summary.errors += 1
        return summary

    summary.deals_seen = len(deals)
    log.info("bridge.enrich.deals_found", count=len(deals), dry_run=dry_run)

    for deal in deals:
        deal_id = deal.get("id")
        contact_id = deal.get("contact_id")
        if not contact_id:
            log.debug("bridge.enrich.deal_skipped_no_contact", deal_id=deal_id)
            continue

        try:
            contact = fetch_contact(api_url, workspace_id, contact_id, token, timeout=timeout)
        except httpx.HTTPError as e:
            log.error(
                "bridge.enrich.fetch_contact_failed",
                deal_id=deal_id, contact_id=contact_id, error=str(e),
            )
            summary.errors += 1
            continue

        summary.contacts_seen += 1
        if contact is None:
            log.warning(
                "bridge.enrich.contact_not_found", deal_id=deal_id, contact_id=contact_id,
            )
            summary.errors += 1
            continue

        email = contact.get("email")
        name = contact.get("name")
        decision: dict[str, Any] = {
            "deal_id": deal_id, "contact_id": contact_id, "contact_name": name,
        }

        match = resolve_contact_to_entity(
            cfg.vault_path, email=email, name=name, ignore_dirs=C.VAULT_IGNORE_DIRS,
        )
        if match is None:
            log.info(
                "bridge.enrich.no_match", deal_id=deal_id, contact_id=contact_id,
                contact_name=name, email=email,
            )
            summary.skipped_no_match += 1
            decision["outcome"] = "no_match"
            summary.decisions.append(decision)
            continue

        summary.matched += 1
        decision["entity"] = match.rel_path
        log.info(
            "bridge.enrich.matched", deal_id=deal_id, contact_id=contact_id,
            entity=match.rel_path, confidence=match.confidence,
        )

        brief_obj = synthesize_entity_brief(engine, match, top_k=top_k)
        if brief_obj is None:
            log.info(
                "bridge.enrich.no_brief", deal_id=deal_id, contact_id=contact_id,
                entity=match.rel_path,
            )
            summary.skipped_no_brief += 1
            decision["outcome"] = "no_brief"
            summary.decisions.append(decision)
            continue

        summary.briefs_synthesized += 1
        decision["brief_text"] = brief_obj.text
        decision["note_hash"] = brief_obj.note_hash
        log.info(
            "bridge.enrich.brief_synthesized", deal_id=deal_id, entity=match.rel_path,
            hash=brief_obj.note_hash, sources=brief_obj.source_paths,
        )

        status, detail = post_note(
            "deals", deal_id, brief_obj.text, brief_obj.note_hash,
            dry_run=dry_run, timeout=timeout,
        )
        decision["post_status"] = status
        decision["post_detail"] = detail

        if status == "dry":
            summary.would_post += 1
            decision["outcome"] = "would_post"
            log.info("bridge.enrich.would_post", deal_id=deal_id, entity=match.rel_path, detail=detail)
        elif status == "skip":
            summary.skipped_duplicate += 1
            decision["outcome"] = "skipped_duplicate"
            log.info("bridge.enrich.skipped_duplicate", deal_id=deal_id, entity=match.rel_path, detail=detail)
        elif status == "ok":
            summary.posted += 1
            decision["outcome"] = "posted"
            log.info("bridge.enrich.posted", deal_id=deal_id, entity=match.rel_path, detail=detail)
        else:
            summary.errors += 1
            decision["outcome"] = "error"
            log.error("bridge.enrich.post_error", deal_id=deal_id, entity=match.rel_path, detail=detail)

        summary.decisions.append(decision)

    log.info("bridge.enrich.done", **summary.as_dict())
    return summary
