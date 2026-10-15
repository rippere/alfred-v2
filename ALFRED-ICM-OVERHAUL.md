# Alfred ICM Overhaul — Handoff (Phase 1 done, Phase 2/3 queued)

**Owner:** Ben · **Started:** 2026-08-20 · **Repo:** ~/alfred-v2
**Source of truth for this work.** Read this first in any continuation session or auto-dev pickup.

## Goal
Keep Alfred as agent-only infra but reorganize on ICM principles (navigate-then-read, small stable
catalogs, exhaust physically outside the queryable set) for organizational control + reliable query.
Diagnostic + hardening (2026-08-20) established the numbers; Phase 1 (destructive reclaim) is complete.

## Phase 1 — DONE & VERIFIED (2026-08-20)
- Root cause fixed: `.stignore` now excludes `.git` from Syncthing (was replicating .git → corruption + bloat).
- git: **1.3 G → 503 M** (`git prune-packed` + removed 1,657 `.sync-conflict-*` objects). HEAD readable, 0 garbage.
- LanceDB main store: **33 G → 600 M** (`Table.optimize(cleanup_older_than=0, delete_unverified=True)`), 34,300→~30 versions, 197k rows preserved, vector search verified.
- inbox/processed graveyard: **8,757 → 0**.
- Vault total **1.7 G → 670 M**. All services restored (syncthing, alfred-mcp-http, alfred, 11 timers). Retrieval proven via MCP (`project/novacrm.md` @ 0.999).
- **Backups (keep until confident, ~35 G):** `/home/rippere/alfred-phase1-backup-20260820/obsidian-vault.tar` (1.6 G) + `/mnt/external/backups/alfred-phase1-20260820/lancedb-main.tar` (33 G) + `state.json`.

## Loose ends from Phase 1
- **git corruption residual:** ~437 MiB of loose objects can't be repacked until 1,371 missing objects (incl. commit `eaee51d9`) are restored from a Syncthing peer, then `git repack -ad`. Repo works fine as-is. Low priority.
- **Peer devices** (benderman-laptop/desktop) still sync their own `.git` — apply the same `.stignore` there or they can re-seed conflicts.

## Phase 2 — token trim (REVERSIBLE; low risk)
- [ ] Rewrite `briefing-generate.sh` → two-tier pointer-mode: small stable shelf-index injected eagerly (~1.5–2K tok), detail (session records, git activity) written to disk & read on demand. Copies the MEMORY.md catalog pattern. **Real payoff ~5,977 tok** (Recent Session Records 3,835 + Git Activity 2,142). Keep old generator for rollback.
- NOTE: ECC 271-skill listing is NOT the problem — harness injects only ~2,450 tok (7 of 271 get descriptions). Do NOT touch plugin config.

## Phase 3 — ICM retrieval redesign — **GO** (77% navigate-first hit rate; band 70–82%)
- [ ] **Route session/ run-logs OUT of the ingested set** — surveyor currently embeds crm-self-healer/kpi/watchdog exhaust (68% of session/). Move to a `run/` dir excluded from distiller `source_types` in config.yaml. Fixes write-amplification (root of regrowth).
- [ ] **Add periodic `optimize` to the vault janitor** — `Table.optimize(cleanup_older_than=timedelta(hours=24), delete_unverified=True)` on the main store, so Lance can't silently rebuild to 33 G. Requires `pylance` (installed: 10.0.0) in `.venv`.
- [ ] **Generate `VAULT_DESK.md`** (context map / "the desk") — MUST be generator-maintained, not hand-written, and MUST replace the 3 rotted root files (Home.md, "Start Here.md", VAULT_CONTEXT.md). Sketch in the diagnostic. No live per-shelf counts unless the generator fills them.
- [ ] **Shelf-scoped `vault_query`** — add a `record_type`/folder `where`-predicate (metadata is on every chunk); make Hopfield/graph/Ebbinghaus associative stages OPT-IN "explore mode". **Do NOT hard-disable associative** — protects the 23% semantic tail (~30 recurring cross-project queries).
- [ ] **Entity-page freshness + sync-conflict dedup (IN SCOPE, not optional)** — 41% of navigable queries currently land in raw transcripts because entity pages are stale (`project/novacrm.md` untouched since 2026-06-22; `person/Ben Rippere.md` 868 B; duplicate forks for novacrm/tribe-social/sector-flow/alfred-v2). Without this the effective hit-rate collapses to ~59%.

## Verification gate for Phase 3 retrieval changes
Before shipping the `vault_query` predicate + stage-gating: run the 131 logged queries {stages ON/OFF} × {shelf-filter/none}, score relevance. If stages don't measurably hurt or navigation wouldn't have answered real queries, that sub-change is a no-op — ship only what the A/B supports. (Query-distribution analysis already says GO overall.)

## Full detail
Diagnostic + hardened-plan-delta live in this session's workflow transcripts; ask to republish as an Artifact.
