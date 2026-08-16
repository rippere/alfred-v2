# Alfred v2

Personal agentic knowledge infrastructure. Self-hosted. Always on.

Alfred watches a set of Obsidian-style vaults, ingests everything dropped into their
inboxes (Claude Code session handoffs, notes, clippings), keeps the vaults clean and
indexed, and makes the whole corpus queryable in plain English via MCP tools and an
HTTP API. It runs unattended as systemd user services.

## Architecture: 5 daemons per vault

Each vault runs one `alfred up` process hosting five APScheduler-driven daemons
(`src/alfred/daemons/`):

| Daemon | Role | Cadence |
|---|---|---|
| **Surveyor** | Re-embeds changed files into LanceDB (dense + BM25 index) | every 60s |
| **Curator** | Processes inbox drops: classifies, routes, files into the vault | every 10s |
| **Janitor** | Structural lint/autofix, stub backfill, dedup, session archival | every 4h |
| **Distiller** | Extracts durable learnings from raw captures (LLM, costed) | nightly 2am |
| **Consolidator** | Cluster labeling and synthesis pages across the vault | every 30 min |

## Multi-vault layout

One config file + one data dir + one systemd unit per vault. Live vaults:

| Vault | Content | Config |
|---|---|---|
| Main (ai-systems) | Software/AI/projects | `config.yaml` |
| Neuroscience | Study notes, papers | `config-neuroscience.yaml` |
| Finance | Markets, trading | `config-finance.yaml` |
| Personal | Personal records | `config-personal.yaml` |
| Employment | Job/work records | `config-employment.yaml` |
| Content *(dormant)* | Content creation | `config-content.yaml` (no service; excluded from meta fan-out) |

`config-meta.yaml` defines the cross-vault fan-out used by the meta MCP server
(spawned per-session via `.mcp.json`), which merges top-k results across vaults.

Query surfaces (`src/alfred/mcp/`): `server.py` (stdio MCP), `server_http.py`
(HTTP API on :8765), `meta_server.py` (cross-vault MCP).

## Quickstart

```bash
cd /home/rippere/alfred-v2
uv pip install --python .venv/bin/python -e .

# Run one vault in the foreground
.venv/bin/alfred up --config config.yaml

# Query it
.venv/bin/alfred query "what did I learn about agent architecture last week"
```

In production each vault runs as a systemd user unit (`alfred.service`,
`alfred-neuroscience.service`, ...) with `alfred-mcp-http.service` for the HTTP API
and `alfred-watchdog.timer` for self-healing. See the runbook for operations.
Set `ALFRED_HTTP_TOKEN` to require Bearer auth on the HTTP API — see the runbook's
"HTTP API authentication" section for its three modes (unset/set/set-but-blank).

## Docs

- [RUNBOOK.md](RUNBOOK.md) — day-to-day operations: services, querying, feedback, recovery
- [AUDIT-2026-07-13.md](AUDIT-2026-07-13.md) — structural audit: known gaps and prioritized roadmap
- [DISTILL-REDESIGN.md](DISTILL-REDESIGN.md) — ACTIVE plan: distill organ + two-lane vault (reconciled 2026-07-13; Phases 1–4 pending)
