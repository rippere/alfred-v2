# Alfred v2 — Session Handoff

**Project:** alfred-v2
**Location:** `/home/rippere/alfred-v2/`
**Vault:** `/mnt/external/obsidian-vault/`
**Last updated:** 2026-04-27

## Current State

All 5 daemons running (PID in `data/alfred.pid`).
- 774 files tracked, 12 inbox files curator-processed
- Inbox: empty (all files processed)
- Hyprland autostart: wired to `start.sh` (old personal-alfred commented out)

## Recent Commits

```
b99ca33 fix: validate and correct LLM-returned statuses before vault_create
9aff7a2 fix: use asyncio.to_thread for blocking Anthropic API calls in daemons
bb9449d fix: daemon fork, add anthropic dep, add start.sh
59ed140 fix: MCP server main entry point and vault_search main entry point and vault_search name collision
7d29861 feat: initial alfred-v2 implementation
```

## Start/Stop

```bash
# Start (from project root)
env -u CLAUDECODE bash /home/rippere/alfred-v2/start.sh

# Or background daemon:
rm -f data/alfred.pid && nohup .venv/bin/alfred up >> data/alfred.log 2>&1 &

# Stop
.venv/bin/alfred down
```

## Known Issues / Next Steps

- Janitor reports 713 files with issues (mostly stub/missing fields) — LLM enrichment will
  run automatically on next deep sweep (24h cycle)
- Distiller hasn't run yet (24h interval from first start)
- MCP server registered as `alfred-vault` in Claude Code settings

## Architecture

- **Surveyor**: embed+cluster (HDBSCAN/Leiden), 10-min cycle
- **Janitor**: structural scan + autofix + LLM enrichment (1h/24h)
- **Curator**: inbox→vault records via Anthropic API, 10s poll
- **Distiller**: extract learnings from vault records, 24h
- **Consolidator**: cluster summaries, 15min
- **Query**: hybrid BM25+dense, Hopfield refinement, spreading activation, FlashRank reranking
- **MCP**: FastMCP stdio server (`alfred mcp` or `python -m alfred.mcp.server`)
