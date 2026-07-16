"""Post/list notes on NovaCRM deals & contacts — the bridge's write surface.

Auth + dry-run-degrade behavior is reused verbatim from `alfred.ledger.push`
(same Supabase bot user, same env file, same `_authenticate` password-grant
flow — see that module's docstring for the two-step flow). `alfred.bridge`
never redefines its own credential set (see `alfred.bridge.config`'s
docstring), so cross-importing `_authenticate` / `_AuthError` /
`ensure_env_scaffold` from `alfred.ledger.push` is the correct mirror here,
not a duplicate — there is exactly one Supabase identity involved.

This module adds only what's CRM-bridge-specific on top of that shared auth:
listing/posting notes on the two target endpoints, the idempotency-marker
dedup check, and the dry-run split that's part of Pattern A's spec.

Endpoints (see recon — apps/api/app/routers/{deals,contacts}.py):
    GET  {API_URL}/workspaces/{WORKSPACE_ID}/deals/{deal_id}/notes
    POST {API_URL}/workspaces/{WORKSPACE_ID}/deals/{deal_id}/notes
    GET  {API_URL}/workspaces/{WORKSPACE_ID}/contacts/{contact_id}/notes
    POST {API_URL}/workspaces/{WORKSPACE_ID}/contacts/{contact_id}/notes
Both pairs share the same request/response shape:
    POST body:  {"body": str, "author": str | None}   (CreateDealNoteRequest /
                                                         CreateContactNoteRequest)
    response:   {"id", "workspace_id", "<deal|contact>_id", "body", "author",
                 "created_at"}                          (*NoteResponse)

IMPORTANT — the dry_run split only ever guards the POST:
    - `dry_run=True` (the default, and the only mode exercised by any
      automation in this build pass) still authenticates and GETs existing
      notes, because the idempotency check ("would this be a duplicate?")
      is part of what dry-run is supposed to show the user. It just never
      calls `create_note` / issues a POST.
    - `dry_run=False` is the future `--push` path. It is implemented here
      for completeness but must not be exercised against the live API in
      this pass (no test or manual run in this repo calls it with real
      credentials).
    - If required credentials are absent, both modes degrade to a "dry"
      result with zero network calls at all (same trigger point as
      `ledger.push.push_snapshot`).
"""

from __future__ import annotations

from typing import Literal, Optional

import httpx

from alfred.bridge import config as C
from alfred.ledger.push import _AuthError, _authenticate, ensure_env_scaffold

# Same six keys as `alfred.ledger.push._REQUIRED` — duplicated as a local
# guard list (matching that module's own convention) rather than imported,
# since it's a private name; must stay in sync with `alfred.bridge.config`'s
# `PUSH_ENV_KEYS` (itself an alias of `alfred.ledger.config.PUSH_ENV_KEYS`).
_REQUIRED = (
    "LEDGER_SUPABASE_URL",
    "LEDGER_SUPABASE_ANON_KEY",
    "LEDGER_API_URL",
    "LEDGER_WORKSPACE_ID",
    "LEDGER_EMAIL",
    "LEDGER_PASSWORD",
)

TargetType = Literal["deals", "contacts"]


def build_note_body(brief_text: str, note_hash: str) -> str:
    """Append the idempotency marker to a brief's prose.

    e.g. "...brief prose...\\n\\n<!-- alfred:brief hash=abcdef123456 -->"
    """
    marker = f"{C.NOTE_MARKER_PREFIX}{note_hash}{C.NOTE_MARKER_SUFFIX}"
    return f"{brief_text.strip()}\n\n{marker}"


def build_note_payload(body: str, *, author: Optional[str] = "alfred") -> dict:
    """CreateDealNoteRequest / CreateContactNoteRequest shape: {"body", "author"}."""
    return {"body": body, "author": author}


def has_matching_hash(existing_notes: list[dict], note_hash: str) -> bool:
    """True if any existing note body already embeds this hash's marker."""
    marker = f"{C.NOTE_MARKER_PREFIX}{note_hash}{C.NOTE_MARKER_SUFFIX}"
    return any(marker in (note.get("body") or "") for note in existing_notes)


def _notes_url(env: dict, target_type: TargetType, target_id: str) -> str:
    api = env["LEDGER_API_URL"].rstrip("/")
    ws = env["LEDGER_WORKSPACE_ID"]
    return f"{api}/workspaces/{ws}/{target_type}/{target_id}/notes"


def fetch_existing_notes(
    target_type: TargetType,
    target_id: str,
    env: dict,
    token: str,
    *,
    timeout: float = 30.0,
) -> list[dict]:
    """GET .../notes — read existing notes for the dedup check."""
    resp = httpx.get(
        _notes_url(env, target_type, target_id),
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def create_note(
    target_type: TargetType,
    target_id: str,
    payload: dict,
    env: dict,
    token: str,
    *,
    timeout: float = 30.0,
) -> httpx.Response:
    """POST .../notes — the actual write. Callers must never invoke this
    when `dry_run` is True; `post_note` below is the only sanctioned caller.
    """
    return httpx.post(
        _notes_url(env, target_type, target_id),
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=timeout,
    )


def post_note(
    target_type: TargetType,
    target_id: str,
    brief_text: str,
    note_hash: str,
    *,
    dry_run: bool = True,
    author: Optional[str] = "alfred",
    timeout: float = 30.0,
) -> tuple[str, str]:
    """Post (or simulate posting) one brief as a note. Returns (status, detail).

    status:
        "ok"    — POST accepted (2xx); only possible when dry_run is False.
        "dry"   — dry_run True, OR required creds are missing — nothing was
                  written (no POST was made either way).
        "skip"  — a note with this hash's marker already exists on the
                  target — not reposted, regardless of dry_run.
        "error" — auth, list, or post failure (detail has the reason).
    """
    ensure_env_scaffold()
    env = C.load_push_env()
    body = build_note_body(brief_text, note_hash)
    payload = build_note_payload(body, author=author)
    target_desc = f"{target_type}/{target_id}"

    missing = [k for k in _REQUIRED if not env.get(k)]
    if missing:
        return "dry", (
            f"DRY RUN (missing {', '.join(missing)}) — would post to "
            f"{target_desc}: {body[:200]!r}"
        )

    token = _authenticate(env, timeout=timeout)
    if token is None:
        return "error", "auth failed (no access_token returned)"
    if isinstance(token, _AuthError):
        return "error", f"auth failed: {token.detail}"

    try:
        existing = fetch_existing_notes(target_type, target_id, env, token, timeout=timeout)
    except httpx.HTTPError as e:
        return "error", f"failed to list existing notes on {target_desc}: {e}"

    if has_matching_hash(existing, note_hash):
        return "skip", f"already posted (hash={note_hash}) on {target_desc} — not reposting"

    if dry_run:
        return "dry", f"DRY RUN — would post to {target_desc}: {body[:200]!r}"

    try:
        resp = create_note(target_type, target_id, payload, env, token, timeout=timeout)
    except httpx.HTTPError as e:
        return "error", f"post request to {target_desc} failed: {e}"

    if resp.is_success:
        return "ok", f"posted note to {target_desc} (HTTP {resp.status_code})"
    return "error", f"post to {target_desc} HTTP {resp.status_code}: {resp.text[:200]}"
