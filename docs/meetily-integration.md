# Meetily → Alfred integration

**Goal:** fold [Meetily](https://github.com/Zackriya-Solutions/meeting-minutes)
(a local-first meeting-notes tool: audio capture → Whisper/Parakeet transcription
→ pluggable-LLM summary) into Alfred so meetings become first-class, queryable
memory alongside sessions, projects, and topics.

## The core idea: Meetily is a *source*, not a fork

Meetily recently became a self-contained **Rust/Tauri desktop app** (its old
Python/FastAPI backend is archived). Forking or embedding that app into Alfred
would be a large, brittle undertaking and buys nothing — Alfred is a *memory*
system, Meetily is a *capture* tool.

Both Meetily builds persist everything to a **local SQLite database**. That file
is the clean, stable integration seam. Alfred reads it (read-only) and drops
`type: meeting` markdown into `inbox/`; from there **the entire existing Alfred
pipeline runs unchanged**:

```
 Meetily.app ──writes──▶ meeting_minutes.db (SQLite)
                              │
        alfred meetily sync (this module, read-only)
                              │  renders type: meeting markdown
                              ▼
                    <vault>/inbox/meetily-*.md
                              │
             Curator daemon (already runs every 10s)
                              │  files by declared type
                              ▼
                    <vault>/meeting/<slug>.md
                              │
   Surveyor (embed) · Distiller (learnings) · Consolidator (project links)
                              │
              vault_query / vault_search / MCP tools · HTTP API
```

Nothing in the daemon runner, query engine, or MCP surface has to change. We add
one record type and one small producer module — mirroring the existing `ledger`
sub-app precedent.

## Meetily's data model (the read surface)

Three SQLite tables (`backend/app/db.py`, unchanged in the Tauri build):

| table | key columns Alfred reads |
|---|---|
| `meetings` | `id`, `title`, `created_at`, `updated_at` |
| `transcripts` | `meeting_id`, `transcript`, `summary`, `action_items`, `key_points`, `duration` |
| `summary_processes` | `meeting_id`, `status`, `result` (JSON structured summary) |

## Field mapping (Meetily → Alfred `type: meeting`)

| Meetily | Alfred frontmatter / body |
|---|---|
| `meetings.title` | `name` |
| `meetings.created_at` | `created`, `meeting_date` |
| `meetings.id` | `meetily_id` (dedup key — never ingested twice) |
| `transcripts.duration` (summed) | `duration_min` |
| `summary_processes.result` present? | `status: summarized` else `captured` |
| `summary` / `key_points` / `action_items` | body `## Summary` / `## Key Points` / `## Action Items` |
| `summary_processes.result` (JSON) | body sections (CamelCase keys → headings) |
| `transcripts.transcript` (stitched) | body `## Transcript` (truncated to 6k chars) |

Rendered example lives in the module's smoke test; the note carries
`source: meetily` and `tags: [meeting]` so it's filterable everywhere.

## What was scaffolded

```
src/alfred/meetily/
  __init__.py     package + design overview
  config.py       Meetily DB path resolution ($MEETILY_DB / config / probe)
  reader.py       read-only SQLite reader → Meeting dataclass (schema-drift tolerant)
  record.py       Meeting → type: meeting markdown (frontmatter + body)
  ingest.py       write new meetings to inbox/, JSON sidecar for idempotency
  cli.py          `alfred meetily sync|status` Typer sub-app

src/alfred/core/schema.py   + "meeting" type (KNOWN_TYPES, STATUS_BY_TYPE,
                              TYPE_DIRECTORY, aliases) — additive, non-breaking
src/alfred/cli.py           + app.add_typer(meetily_app, name="meetily")
config.yaml                 + meetily.db_path example block
deploy/systemd/             meetily-sync.service + .timer (poll every 15 min)
```

`meeting` statuses: `captured → summarized → reviewed → archived`.

## Usage

```bash
alfred meetily status                  # is the DB reachable? how many meetings, how many pending?
alfred meetily sync --dry-run          # preview what would be ingested
alfred meetily sync                    # ingest new meetings into the vault inbox
alfred meetily sync --since 2026-06-01 # backfill from a date

# then, unchanged:
alfred query "action items from the Q3 planning meeting" --synthesis
# or in Claude Code:  vault_search record_type=meeting status=summarized
```

Point it at your DB with `$MEETILY_DB` or `meetily.db_path` in `config.yaml`.
Run continuously via the systemd timer (`deploy/systemd/meetily-sync.timer`).

## Why this shape

- **Zero new moving parts in the hot path.** Curator/Surveyor/Distiller already
  do filing, embedding, and learning-extraction. Meetings inherit all of it.
- **Idempotent + read-only.** Alfred never writes Meetily's DB; a JSON sidecar of
  synced `meetily_id`s makes re-runs safe.
- **Matches an existing precedent.** Structurally identical to `alfred ledger`
  (Typer sub-app + systemd timer + SQLite source), so it's easy to reason about.

## Phase 2 ideas (not built)

1. **Auto project-linking for meetings** — the Curator already links *sessions*
   to project hubs by name match (`_link_session_to_project`); extend it to
   `type: meeting` so a meeting mentioning "Tribe v2" links to that project hub.
2. **Action-items → tasks** — spin `## Action Items` into real `type: task`
   records with `related: [[meeting/...]]`.
3. **Context-aware summaries (bidirectional)** — before Meetily summarises, feed
   it relevant vault context via Alfred's HTTP API so summaries are grounded in
   what you already know. Requires a Meetily-side hook; larger effort.
4. **Distiller awareness** — teach the nightly Distiller that meetings are a
   high-signal source for decisions/assumptions.
```

<!-- Note: bump docs/RUNBOOK.md with a "Meeting capture" row once this is live. -->
