# alfred-v2 Upkeep-Reduction Roadmap

Date: 2026-07-24 · Branch audited: `claude/roadmap-phase-2` · All paths relative to `/home/rippere/alfred-v2` unless absolute.

---

## 1. TL;DR

The July incident chain (watchdog alert storm → vault pollution → curator budget burn → 659k-line log spam → starved rotation → 32GB LanceDB bloat) traces to **four latent code defects and one process gap**, all now root-caused and verified:

1. **No LanceDB maintenance code exists anywhere** — 32GB of quadratically-growing version manifests accumulated in 8 days (`data/lancedb/_versions/`).
2. **`StateStore.save()` aliases its merge base** (`state.py:254-255`) — every `janitor_sweeps`/`distiller_runs` append since 07-16 was silently dropped.
3. **APScheduler jobs registered without `misfire_grace_time` inherit a 1-second default** — log rotation never executed after 07-16 while the file grew to 189MB.
4. **Alert paths have no rate limit and alert content flows into the knowledge base** — 3,586 watchdog files became 1,487 Lance rows, 1,759 graph nodes, and a 799-member pure-noise cluster.
5. **37 commits of hardening sit unmerged on `claude/roadmap-phase-2`**; 5 branches exist only on this disk.

The fleet-level failure (dead legacy unit alerting 288×/day for 7 days) was **fixed this morning**. Everything below is either a one-time repair (~32GB reclaim, one merge session, one purge) or a durable mechanism riding existing timers/ticks. **Estimated recurring manual work eliminated after dedupe across themes: ~20–25 h/month, with zero new recurring manual steps.** Six decisions needed from Ben (§6); the single highest-leverage one-liner is `loginctl enable-linger rippere`.

---

## 2. Fixed today (before this doc)

- Half-done systemd migration completed: legacy `alfred-{finance,neuroscience,meta}.service` removed (backups in `.trash-migration/`), finance moved to `alfred@finance`.
- Fleet green: `alfred.service` + `alfred@{personal,finance,neuroscience,employment}` + `alfred-mcp-http.service` all active.
- Watchdog alert loop stopped (was 288 alerts/day for 7 days; last legacy-unit alert 15:18).
- Roster (`scripts/alfred-roster.sh`) now emits template units.
- Concurrent mechanical cleanup emptied `inbox/processed/` of watchdog copies (curated records still present — see purge, §4 rank 9).

---

## 3. Verified findings

| Finding | Verdict | Evidence (one line) |
|---|---|---|
| lance-bloat | PARTIAL (core confirmed, causes corrected) | 32GB `_versions/` (55,340 manifests, each ~762KB re-listing all 55k fragments — quadratic) accumulated 07-16→07-24; **zero** compact/cleanup call sites in the codebase; driver is vault churn (watchdog spam embedded 2,799×, one file 353×), not per-tick writes |
| log-bomb | CONFIRMED | 659,070 `api_budget_exhausted` lines (652,013 from curator retrying its backlog every 10s); rotation job skipped every hourly slot since 07-16 because `runner.py:267` omits `misfire_grace_time` (APScheduler default = 1s) while the event loop lags minutes |
| state-freeze | PARTIAL (mechanism corrected, proven by repro) | `save()` sets `self._base_raw = merged` while `_decode_state` passes containers by reference — post-save appends mutate the merge base and are dropped from disk AND memory; `janitor_sweeps` frozen at 07-16 despite sweeps running today; side effect: spurious `distiller_catchup` burns API budget every restart |
| sigkill-corruption | PARTIAL | 8/134 stops since 06-20 hit `TimeoutStopSec=30` → SIGKILL (only when a surveyor tick is in flight; `scheduler.shutdown(wait=True)` has no cancellation point); the two Jul-16 quarantines were **unclean reboots**, not stop-path kills — separate failure class, mitigated by existing auto-quarantine |
| stranded-work | PARTIAL (topology corrected) | 37 commits on `claude/roadmap-phase-2` unmerged (daemons DO run them via editable install); 9 work-carrying branches of which 3 are one stacked lineage (8⊃6⊃1); 5 branches local-only = single-disk risk; 4 branches dead; local master 2 behind origin |
| vault-pollution | CONFIRMED | 3,586 marker-tagged files; 1,763 curated records, 1,487/28,241 Lance rows (5.3%), 1,759/23,796 graph nodes (7.4%), cluster `semantic_490` = 799/799 watchdog; `_alert_inbox` has no rate limit, dedup, or cooldown |

---

## 4. Solves ranked by leverage

Effort: **S** ≤ 2h · **M** ≤ 1 day · **L** 1–2 days. Savings are per month unless marked one-time. Deduped across the five theme designs (curator guard, retention tick, and watchdog rate-limit each appeared in 2–3 themes; counted once here).

### Tier 0 — one-liners and one-time repairs (do first)

| # | Solve | Saves | Effort | Integration point |
|---|---|---|---|---|
| 0a | `loginctl enable-linger rippere` | caps unbounded tail risk (fleet dead after logout = 2–6h per incident) | 1 cmd | — (verified `Linger=no` today) |
| 0b | One-time cold Lance compaction: quiesce `alfred.service`+mcp, `cp -al` snapshot, `compact_files()` + `cleanup_old_versions(1d)`, verify `count_rows()==28,221`, restart | **~32GB reclaimed**, one-time | S | `.venv/bin/python` against `data/lancedb`, table `vault_v2` per fix direction |
| 0c | Branch merge session: ff master to phase-2 → merge PRs #7/#6/#5 → push+merge crm-bridge → one conflict pass on `fix/mcp-auth-middleware-inert` stack → delete 4 dead branches → restart fleet | closes 37-commit gap + 5-branch single-disk risk, one-time | M (~2h) | git + `gh`; only step 4 has conflicts (8 files, drop fb49224, keep 5297c88) |
| 0d | Watchdog-record purge: archive 1,763 curated records to `_archived/ops-noise/`, delete embeddings/state/graph nodes, kill cluster `semantic_490` | −1,487 Lance rows, −1,759 graph nodes; retrieval quality; one-time | M | new janitor `_ops_noise_sweep()` + CLI `alfred purge-ops-noise --dry-run`; reuses `janitor.py:414` `_delete_embeddings` |

### Tier 1 — root-cause code fixes (highest recurring leverage per line changed)

| # | Solve | Saves | Effort | Integration point |
|---|---|---|---|---|
| 1 | **State-freeze fix** — `self._base_raw = asdict(self._state)` after re-decode; copy containers in `_decode_state`; cap lists in `_merge_pipeline_state`; regression test | ~1 h/mo + stops spurious catchup API burn; **prerequisite for #5, #7, #12, #13** | S | `src/alfred/store/state.py:save` (255), `_decode_state` (47–49), `_merge_pipeline_state` (after 176) |
| 2 | **Scheduler misfire fix + budget backoff** — `job_defaults={'misfire_grace_time':300}`; throttle sole WARN site to once/15min/daemon; curator/distiller break loop on first budget rejection; pause-until-midnight | ~1.5 h/mo; kills the 347k-lines/day spam class | S–M | `runner.py:102,267-274,202-209`; `state.py:375-389`; `curator.py:86`; `distiller.py:234` |
| 3 | **Curator transient-source guard** — files tagged `alfred:source alfred_watchdog` (or `lancedb_quarantine`) route to processed/ops, never classified, never embedded | ~2 h/mo; makes any future storm structurally unable to reach LanceDB/graph | S | `curator.py:_ingest_file` (after line 120) + `config.py` `curator_transient_sources` field |
| 4 | **Alert lib + rate limit** — `scripts/alfred-alert-lib.sh` `alert_once <key> <cooldown>`: one rolling note per incident signature in `~/.local/state/alfred/alerts/`, signal-only ntfy push, `alert_clear` on recovery; convert watchdog `_alert_inbox`, heartbeat-check, alert-inbox | ~1.5 h/mo; 288 files/day → ≤4 pushes/day/service | M | new `scripts/alfred-alert-lib.sh`; `alfred-watchdog.sh:123-141,150`; `alfred-heartbeat-check.sh:35-51` |
| 5 | **Janitor Lance optimize tick** — `LanceDBStore.optimize(days)` + daily 3am cron tick, off-loop via `asyncio.to_thread`; quarantine-dir pruning | ~2–3 h/mo; prevents recurrence in all 6 vaults | M | `lancedb_store.py` (new method near `delete_file:381`); `janitor.py` new tick; `runner.py:_add` with `GRACE_DAILY`; keys `janitor.lance_cleanup_days=7` |
| 6 | **Cooperative shutdown cancel** — set `_stop` on SIGTERM, check between files in surveyor/janitor/curator/distiller loops; bounded 20s drain; `TimeoutStopSec=120` in both units | ~0.5–1 h/mo; removes stop-path Lance corruption class | M | `runner.py:279-306`; `surveyor.py:_process_diff` (116, 130); `deploy/systemd/alfred{,@}.service` |

### Tier 2 — durable machinery (ride existing timers/ticks)

| # | Solve | Saves | Effort | Integration point |
|---|---|---|---|---|
| 7 | Retention tick — `inbox/processed/` TTL 30d → `_archived/`, `curator_processed` prune 90d | ~1 h/mo | S | new `janitor.retention_tick`; **gated on #1** (dict deletions no-op under aliasing); keys `janitor.processed_ttl_days` |
| 8 | Logrotate systemd timer (defense in depth for #2) — daily `logrotate` copytruncate over `data{,-*}/alfred.log`, `OnFailure=alfred-alert@%n` | included in #2's 1.5 h/mo; cannot be starved by the event loop | S | new `deploy/systemd/alfred-logrotate.{service,timer}` |
| 9 | `alfred status` rewrite + `core/health.py` — drop dead milvus checks; per-vault Lance `_versions/` size (scandir, never `list_versions()`), log size, liveness, budget, sweep-tail age; `--fleet` rollup | ~1.5 h/mo; would have caught both July incidents in a day | M | `cli.py:31-86` (delete 54, 76, 83); new `src/alfred/core/health.py` |
| 10 | Weekly `alfred health-report` — systemd timer + oneshot, one inbox note (tagged transient so #3 keeps it out of the corpus); includes branch-age sentinel | ~3 h/mo + every failure mode visible ≤7 days | M | new CLI cmd + `deploy/systemd/alfred-health-report.{service,timer}`; depends on #1, #9 |
| 11 | Game-guard freeze-don't-stop — `systemctl --user freeze/thaw` for daemons (ollama still stops); pull script into repo | ~0.5 h/mo; instant GPU-free at game launch | S | `~/.local/bin/ollama-game-guard.sh:enforce_paused/resume_all`; rollback env `OGG_PAUSE_VERB=stop` |
| 12 | state.json daily backup — creator never existed; add validated snapshot before the existing prune | ~0.5 h/mo EV (next corruption = file copy, not log archaeology) | S | `runner.py:_housekeeping` (before prune at 256); rides #2's grace fix |
| 13 | Lint backlog drain — per-daemon API caps (`curator:350/distiller:75/janitor:75`), LINK001 deterministic autofix via stem index, FM003 schema decision, STUB001 via funded deep sweep | ~3–4 h/mo (kills quarterly "vault cleanup day") | M–L | `state.py:can_make_api_call`; `janitor.py:_autofix` (269); `core/schema.py`; re-baseline **after** 0d purge |
| 14 | Unit single-source-of-truth — copy 4 installed-only units into repo, symlink installs via `deploy/install-units.sh`, watchdog drift sentinel (`NeedDaemonReload`, non-symlinked, legacy-unit coexistence) | ~1 h/mo; July's drift class detected in 5 min instead of 7 days | M | new `deploy/install-units.sh`; `alfred-watchdog.sh` `_check_drift()`; reconcile `alfred-mcp-http.service` with 0c step 4 |
| 15 | Janitor dedup staged enablement — `dedup_mode: off/dry_run/apply`, archive-not-delete, 20-merge circuit breaker, comparisons off-loop | ~1 h/mo | M | `janitor.py:_dedup_sweep` (433–583, gate mutations at 527–555); `runner.py:148-149` |

### Tier 3 — small hardening (batch into adjacent PRs)

| # | Solve | Saves | Effort | Integration point |
|---|---|---|---|---|
| 16 | Ledger push loud-once — stamp-gated exit 1 on credential-missing dry-run; `ok` clears stamp | ~0.5 h/mo; blindness bounded to 24h | S | `ledger/push.py:104-109`; `ledger/cli.py:69-79` |
| 17 | `OnFailure=` + `StartLimitBurst=5` on core units; game-guard-aware `alfred-alert-inbox.sh` | ~0.5 h/mo; core-daemon death = push in seconds | S | `deploy/systemd/alfred{,@}.service` `[Unit]`; `scripts/alfred-alert-inbox.sh` |
| 18 | Tier-2 healer allowlist + 6h cooldown — roster-derived gate, ≤4 `claude -p` calls/day/service | ~0.25 h/mo + hard API-spend cap | S | `alfred-watchdog.sh:170-185` |
| 19 | create-vault self-enablement + watchdog `is-enabled` convergence | ~0.5 h/mo; retires the 2-for-2 vault-add failure | S–M | `cli.py:293-439` (after 427); watchdog roster loop |
| 20 | Watchdog linger check (escalate-once) + install-time gates | keeps 0a from silently regressing | S | `alfred-watchdog.sh` (after line 22) |
| 21 | Content vault decommission (pending decision, §6) | ~0.5 h/mo + 5-not-6 configs on every fleet rollout | S (15 min) | `config-content.yaml` → `deploy/_decommissioned/`; RUNBOOK tombstone; fully reversible |

---

## 5. Suggested sequencing

### Days 0–7 (one-time repairs + root causes; ~2–3 focused sessions)

1. **Day 0:** `loginctl enable-linger rippere` (0a). Decisions from §6 confirmed.
2. **Day 0–1:** One-time cold compaction (0b) — 32GB back the same day.
3. **Day 1:** Land **#1 state-freeze + #2 misfire/backoff + #12 backups** as one `runner.py`/`state.py` PR with regression tests; fleet restart. Acceptance: `jq '.janitor_sweeps[-1]' data/state.json` newer than restart within 2h; `api_budget_exhausted` count for a full exhausted day is O(daemons).
4. **Day 2:** **#3 curator guard + #4 alert lib/rate-limit** (the two layers that make storms structurally impossible). Then **0d purge** (dry-run report → apply) and re-baseline the lint sweep.
5. **Day 3–5:** **0c merge session** (~2h supervised; step 4 conflict pass on the mcp-auth stack is the only thinking part). Restart fleet after merge (alfred@neuroscience runs pre-merge code since Jul 16).
6. **Day 5–7:** **#5 optimize tick + #8 logrotate timer**; observe one nightly cycle (`grep janitor.optimized data*/alfred.log`).

### Days 8–30

- **#6 cooperative shutdown** + unit TimeoutStopSec bump; verify a restart during a busy tick shows `surveyor.tick_aborted_shutdown`, no `stop-sigterm timed out`.
- **#14 unit symlinks + drift sentinel** (reconciling mcp-http against the merged code), then **#11 game-guard freeze** through the new install path.
- **#7 retention tick** (now safe — #1 merged), **#9 status rewrite**.
- Tier-3 batch: #16–#20 (one watchdog edit batch + one small units PR).

### Days 31–60

- **#10 weekly health report** (needs #1 + #9); enable timer, tune thresholds after first manual run.
- **#13 lint backlog drain**: per-daemon caps → FM003 schema decision → LINK001 dry-run week → apply. Target `files_with_issues` < 100 by day 60.
- **#15 dedup staged rollout**: dry-run 2 weeks on main vault → apply → fleet-wide.
- **#21 content vault decommission** execution (pending §6 decision).
- Exit criteria for the program: weekly report green 2 consecutive weeks; `_versions/` bounded; zero manual upkeep steps performed in the final 2 weeks.

---

## 6. Decisions needed from Ben

| # | Decision | Recommendation | Notes |
|---|---|---|---|
| 1 | Delete `data.backup-2026-05-04` (47MB, Milvus-era) | **Delete** | Pre-LanceDB artifact; nothing reads it. Also candidates: stale `vault_v2_backup_20260607.parquet` + May-12 state snapshots (superseded by #12's rolling backups) |
| 2 | Purge the 1,763 ingested watchdog records (files → `_archived/`, −1,487 Lance rows, −1,759 graph nodes, kill cluster `semantic_490`) | **Approve** | Archive-not-delete; dry-run report first; fully reversible (re-embed from markdown). 5.3% of every query's corpus is pure ops noise today |
| 3 | Content vault: wire up vs decommission | **Decommission** | Dormant since 06-22, index empty, zero usage signal in 5 weeks; re-wiring later is the same 15 minutes. Keeping it half-alive is the worst option |
| 4 | Merge `claude/roadmap-phase-2` → master (+ merge plan in 0c) | **Approve** | ff-only, conflict-free; the daemons already run this code — master is what's stale. Also approves pushing the 5 local-only branches |
| 5 | Lance version-history cleanup: `cleanup_older_than=1d` one-time, 7d recurring | **Approve** | Nothing in the codebase uses time-travel/`list_versions`/`checkout` (verified zero call sites); worst case is a full re-embed from markdown (hours, local) |
| 6 | `loginctl enable-linger rippere` | **Run it** | May prompt polkit — the one true external gate. Without it the entire fleet dies at last-logout. Watchdog check (#20) prevents silent regression |

One additional low-stakes input: an ntfy topic name for `~/.config/alfred/ntfy-topic` (alert-lib pushes stay silently disabled until it exists; local rolling notes work regardless).

---

## 7. Appendix — per-theme design detail (condensed)

### A. Storage (self-cleaning stores)

- **Lance optimize**: `LanceDBStore.optimize(cleanup_older_than_days)` wrapping `self._tbl.optimize(cleanup_older_than=timedelta(...))` with the existing `_looks_like_corruption` classification → quarantine path (`lancedb_store.py:195`). Janitor tick (not runner `_housekeeping` — it just failed for 8 days; not a standalone timer — cross-process compaction against a live writer), registered via `runner.py:_add(..., "cron", hour=3, misfire_grace_time=GRACE_DAILY)`. Off-loop via `asyncio.to_thread`. Config: `janitor.lance_cleanup_days=7`, `janitor.quarantine_retention_days=30`, `janitor.optimize_enabled` kill switch. Quarantine pruning cannot blind the `_MAX_QUARANTINES_PER_DAY` breaker (`lancedb_store.py:43`, rolling 24h).
- **Log/budget**: blanket `job_defaults={'misfire_grace_time':300, 'coalesce':True}` at `runner.py:102`; explicit `GRACE_HOURLY` on housekeeping (`runner.py:267-274`), `GRACE_FAST` on `periodic_save`. Throttle in `StateStore.can_make_api_call` (`state.py:375-389`): warn on state transition + once/900s/daemon, else debug. Curator (`curator.py:86`): one `tick_skipped_budget` line, frontmatter-typed files still process. Distiller (`distiller.py:234`): break batch. Pause-until-midnight reuses `api_paused_until` (`state.py:350`). Logrotate timer: `size 50M / rotate 2 / copytruncate` (fd is O_APPEND, `runner.py:212-216`).
- **Retention**: `janitor.retention_tick` → `inbox/processed/` older than 30d → `_archived/inbox-processed/YYYY-MM/` (implements DISTILL-REDESIGN §4.4 exactly); `curator_processed` keys older than 90d or path-gone pruned (5,639 today, unbounded). **Hard gate: state-freeze fix first** — dict deletions resurrect from `theirs` under aliasing.
- **Backups**: `_housekeeping` step 2.5 — parse-validated `state.json` snapshot via tmp+`os.replace`, 24h cadence, existing prune keeps 3. Restore runbook: stop → copy → rm stale `.lock` → start; ≤24h embed-state delta self-heals via surveyor md5 diff.

### B. Alerting (no crying wolf)

- **`scripts/alfred-alert-lib.sh`** — `alert_once <key> <cooldown> <title> <detail>`: stamp at `~/.local/state/alfred/alerts/<key>.stamp`, rolling detail note (fixed filename per signature — accumulation structurally impossible), signal-only ntfy per the hard rule, `alert_clear` on recovery. Sourced by watchdog, heartbeat-check, alert-inbox.
- **Watchdog**: `_alert_inbox` → `alert_once "svc-down:$svc" 21600`; recovery hook clears stamp at loop line 150. Heartbeat-check same conversion.
- **Curator guard**: `curator_transient_sources` config field; marker match in first 500 bytes → move to processed, `return True` (marks processed, no retry, no LLM call, no record). Covers `lancedb_quarantine` notes for free.
- **OnFailure on core units**: `OnFailure=alfred-alert@%n.service` + `StartLimitIntervalSec=600` / `StartLimitBurst=5` — fires only when 5 restarts/10min exhaust = escalate-once at the systemd layer. `alfred-alert-inbox.sh` gains game-guard suppression (copies `_game_active()` from `alfred-watchdog.sh:29-38`).
- **Drift + tier-2**: `_check_drift()` in the watchdog (legacy-unit coexistence — exactly the July mechanism; installed-vs-`deploy/systemd` diff; 24h cooldown per signature). Tier-2: allowlist file at `~/.local/state/alfred/tier2-allowlist` with roster-derived finance fallback; 6h stamp cooldown caps `claude -p` spend at ≤4/day/service.

### C. Lifecycle (fleet survives everything)

- **Shutdown**: reuse existing `BaseDaemon._stop` (`base.py:36`); set it in `_handle_signal` (`runner.py:279`); check per-file in `surveyor._process_diff` loops (116, 130), `_recluster`, janitor sweeps, curator/distiller loops; `scheduler.shutdown(wait=False)` + 20s bounded drain replacing `runner.py:306`; final `state_store.save()` unchanged. Units: `TimeoutStopSec=120` both, `OOMPolicy=stop`+`MemoryMax=3G` added to `alfred@.service` (migration-parity comment retired). Note: Jul-16 corruption class = unclean reboot; mitigation remains the existing quarantine handler.
- **Game-guard**: `freeze`/`thaw` for user units (GPU is held by ollama, not the daemons); `FreezerState` gate replaces `is-active`; watchdog treats frozen as healthy. Rollback lever `OGG_PAUSE_VERB=stop`. Pull script into `deploy/`, symlink from `~/.local/bin`.
- **Units**: repo completeness first (4 installed-only units have no repo copy — unrecoverable if home dir lost; `alfred-mcp-http.service` already drifted), then `deploy/install-units.sh` (quarantine-never-delete, symlink, `systemd-analyze verify`, assert linger), then watchdog sentinel (`not-symlinked` + `NeedDaemonReload` checks, stamp keyed on drift-set hash).
- **create-vault**: `--enable/--no-enable` option runs `systemctl --user enable --now alfred@<name>` after scaffold; watchdog convergence block enables any roster unit reporting not-enabled (registered-but-unenabled vaults currently get tier-1 started but die at reboot). Invariant: in `config-meta.yaml` ⇒ monitored, game-paused, running, boot-persistent. Decommission order: remove from meta first (documented in RUNBOOK).

### D. Hygiene (knowledge base stays clean)

- **Purge**: janitor `_ops_noise_sweep()` (CLI one-shot + daily no-op tick); detection via `hygiene.noise_globs` (`*alfred-watchdog-alert-*`, `*.sync-conflict-*`, ...) + marker; per file: rename to `_archived/ops-noise/` → `_delete_embeddings` (`janitor.py:414`); one graph pass via new `graph.py:remove_paths()`; archive-not-delete is load-bearing — `_build_stem_index` (`janitor.py:192-210`) still resolves wikilinks, so no LINK001 storm. Back up `graph.pkl` first.
- **Dedup**: `janitor.dedup_mode: off|dry_run|apply` replacing the bool; `dupe.unlink()` (line 541) → archive to `_archived/dedup/`; `dedup_max_merges_per_sweep=20` circuit breaker; SequenceMatcher off-loop before apply mode. Staged: 2 weeks dry-run on main vault → apply → fleet.
- **Lint drain**: the 2,247 sweep number was watchdog-dominated (744 files on disk post-cleanup: FM003 518, LINK001 346); `autofixed: 0` is structural — LINK001/STUB001 have no autofix path and the deep sweep starves on budget. Fix: per-daemon caps in `can_make_api_call`; LINK001 stem-index rewrite (`difflib.get_close_matches` cutoff 0.92, dry-run gated); FM003 = one-time `STATUS_BY_TYPE` extension in `core/schema.py` from a histogram report; stall check emits one `type: task` note per 7 days.
- **Content vault**: decommission per §6; briefs/drafts worth keeping → main-vault `inbox/` (typed drops cost no API).

### E. Observability (nothing invisible for 8 days again)

- **`core/health.py`**: `collect_health(cfg)` — Lance `_versions/` bytes/count via `os.scandir` (never `table.list_versions()`, which hung >5 min on the bloated table), log size vs 50/100MB, PID+unit liveness (extract from `cli.py:up` 102–141), budget fields, sweep-tail age (RED >26h — the check that catches the next state-freeze in a day). `collect_fleet()` shells out to `alfred-roster.sh` for the roster; fleet API-spend rollup from `data*/state.json` + `ledger.db` (closes the ROADMAP Phase 2 item). Thresholds as module constants, no config keys.
- **State-freeze fix**: three edits + `test_state_merge_aliasing.py` (50-full list, two append+save cycles, both entries on disk; dict-add-after-save). Deferred separately: stale-reference class (`save()` rebinding `self._state` orphans refs across awaits in distiller/curator).
- **Ledger loud-once**: split `push_snapshot` statuses (`dry_no_creds` vs `dry_empty`, no detail string-matching); stamp keyed on missing-key set; exit 1 → existing `OnFailure` → one note; `ok` clears stamp (re-arms).
- **Branch sentinel**: `stranded_branches(repo)` in `health.py` — unique commits >0 + >14d = stranded; no upstream = at-risk; scope alfred-v2 + optionally `ledger/config.py:GIT_REPOS`; optional `repo.stranded_branches` ledger metric. SessionEnd hook push failures (currently a private log line at `process-session.sh:244-266`) optionally get a daily-stamped inbox note.
- **Weekly report**: `alfred health-report` CLI + `alfred-health-report.{service,timer}` (`OnCalendar=Mon 07:30`, `Persistent=true`, `OnFailure=alfred-alert@%n`); one markdown note tagged `alfred:source alfred_health` (transient — never embedded); every collector sub-section individually try/excepted so a sick system still produces a report; exit non-zero only on generation failure.

---

*Verification doctrine for every item above: "done" = observed working (fresh state.json append, a rotated log, a green fleet cycle, a reclaimed `du`), never asserted. Each solve lists its acceptance check inline.*
