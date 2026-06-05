"""Push a day's KPI snapshot to a NovaCRM workspace via Supabase auth.

Flow:
  1. Supabase password grant:
       POST {SUPABASE_URL}/auth/v1/token?grant_type=password
       headers: apikey: {SUPABASE_ANON_KEY}
       json: {email, password}                       -> access_token
  2. PUT {API_URL}/workspaces/{WORKSPACE_ID}/kpi/{date}
       headers: Authorization: Bearer {access_token}
       json: {"snapshots": [{domain, metric, value, meta}, ...]}

Credentials come from ``~/.config/alfred-ledger/env`` (or the process env). If
any required credential is missing, this becomes a DRY RUN: it prints a payload
summary and returns status ``"dry"`` — no network call is made. Real secrets are
never written by this module; only ``env.example`` is scaffolded.
"""

from __future__ import annotations

import json
from typing import Optional

import httpx

from alfred.ledger import config as C

_REQUIRED = (
    "LEDGER_SUPABASE_URL",
    "LEDGER_SUPABASE_ANON_KEY",
    "LEDGER_API_URL",
    "LEDGER_WORKSPACE_ID",
    "LEDGER_EMAIL",
    "LEDGER_PASSWORD",
)

_ENV_EXAMPLE_BODY = """\
# Alfred ledger → NovaCRM push credentials.
# Copy to `env` (same dir) and fill in real values. Never commit real secrets.
# If any value is missing, `alfred ledger ... --push` runs as a dry run.

LEDGER_SUPABASE_URL=https://YOUR-PROJECT.supabase.co
LEDGER_SUPABASE_ANON_KEY=your-supabase-anon-key
LEDGER_API_URL=https://your-novacrm-api.example.com
LEDGER_WORKSPACE_ID=your-workspace-uuid
LEDGER_EMAIL=you@example.com
LEDGER_PASSWORD=your-password
"""


def ensure_env_scaffold() -> None:
    """Create the config dir + env.example template (never overwrites real env)."""
    C.ENV_DIR.mkdir(parents=True, exist_ok=True)
    if not C.ENV_EXAMPLE.exists():
        C.ENV_EXAMPLE.write_text(_ENV_EXAMPLE_BODY)


def _payload_summary(date: str, snapshots: list[dict]) -> str:
    by_domain: dict[str, int] = {}
    for s in snapshots:
        by_domain[s["domain"]] = by_domain.get(s["domain"], 0) + 1
    parts = ", ".join(f"{d}:{n}" for d, n in sorted(by_domain.items()))
    return f"{date} — {len(snapshots)} metrics ({parts})"


def _to_snapshots(rows: list[dict]) -> list[dict]:
    """Project DB/collector rows into the API's snapshot shape.

    meta arrives as a dict (live collect), a JSON string (sqlite round-trip),
    or None — the API requires a dict, so normalize all three.
    """
    out = []
    for r in rows:
        meta = r.get("meta")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (ValueError, TypeError):
                meta = {}
        if not isinstance(meta, dict):
            meta = {}
        out.append(
            {"domain": r["domain"], "metric": r["metric"], "value": r["value"], "meta": meta}
        )
    return out


def push_snapshot(
    date: str,
    rows: list[dict],
    *,
    timeout: float = 30.0,
) -> tuple[str, str]:
    """Push one day's snapshot. Returns (status, detail).

    status: "ok"   — server accepted (2xx)
            "dry"  — missing creds; printed summary only, no network call
            "error"— auth or push failed (detail has the reason)
    """
    ensure_env_scaffold()
    env = C.load_push_env()
    snapshots = _to_snapshots(rows)
    summary = _payload_summary(date, snapshots)

    missing = [k for k in _REQUIRED if not env.get(k)]
    if missing:
        return "dry", f"DRY RUN (missing {', '.join(missing)}) — would push {summary}"

    if not snapshots:
        return "dry", f"no metrics to push for {date}"

    # ── 1. Supabase password grant ───────────────────────────────────────────
    token = _authenticate(env, timeout=timeout)
    if token is None:
        return "error", "auth failed (no access_token returned)"
    if isinstance(token, _AuthError):
        return "error", f"auth failed: {token.detail}"

    # ── 2. PUT the snapshot ──────────────────────────────────────────────────
    url = f"{env['LEDGER_API_URL'].rstrip('/')}/workspaces/{env['LEDGER_WORKSPACE_ID']}/kpi/{date}"
    try:
        resp = httpx.put(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json={"snapshots": snapshots},
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        return "error", f"push request failed: {e}"

    if resp.is_success:
        return "ok", f"pushed {summary} (HTTP {resp.status_code})"
    return "error", f"push HTTP {resp.status_code}: {resp.text[:200]}"


class _AuthError:
    def __init__(self, detail: str) -> None:
        self.detail = detail


def _authenticate(env: dict[str, str], *, timeout: float):
    """Return an access token string, None, or _AuthError."""
    auth_url = f"{env['LEDGER_SUPABASE_URL'].rstrip('/')}/auth/v1/token?grant_type=password"
    try:
        resp = httpx.post(
            auth_url,
            headers={
                "apikey": env["LEDGER_SUPABASE_ANON_KEY"],
                "Content-Type": "application/json",
            },
            json={"email": env["LEDGER_EMAIL"], "password": env["LEDGER_PASSWORD"]},
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        return _AuthError(str(e))
    if not resp.is_success:
        return _AuthError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        return resp.json().get("access_token")
    except ValueError:
        return _AuthError("non-JSON auth response")
