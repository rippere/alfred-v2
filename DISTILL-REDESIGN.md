# Alfred Redesign — The Distill Organ & Two-Lane Vault

**Status:** ACTIVE — reconciled 2026-07-13 against the live tree (post AUDIT-2026-07-13
phase-1 hardening). The decisions in §7 remain locked; the core plan (Phases 1–4) is
still unexecuted, still coherent, and still worth executing — see the reconciliation
note below before starting.
**Date:** 2026-06-20 (original) · 2026-07-13 (status reconciliation)
**Goal:** Make the vault simple and worth opening for a human, *without* losing (in fact while improving) its function as queryable AI memory.

---

## Reconciliation note — what changed since 2026-06-20 (read before executing)

Verified against the code and the live main vault on 2026-07-13
(AUDIT-2026-07-13.md structural item #9):

**Already done elsewhere — drop from this plan:**
- **Phase 0 / §4.4 log rotation** — implemented in `src/alfred/runner.py`'s hourly
  housekeeping job (copytruncate rotation + retention caps on rotated generations
  and `state.json.backup-*` snapshots), deployed fleet-wide in phase-1 hardening
  (commit 7bcbd36). `data/alfred.log` is now capped; the 105 MB figure in §1 is history.
- **Distiller cadence** — the distiller now runs nightly (cron 2am + one-shot
  catch-up) in **all six** vaults, and a `distiller.mode: scheduled | on_demand`
  config plus `DistillerDaemon.trigger_sweep()` already exist — §4.1's
  "trigger on session-end" idea has a ready hook point.

**Superseded terminology — mechanism still valid, names are not:**
- Every **Milvus** reference is now **LanceDB** (`src/alfred/store/lancedb_store.py`);
  the migration happened in May. §4.1's dedup-on-write should query the surveyor's
  LanceDB index; §6 step 4's re-index likewise.
- The vault already has a janitor-managed **`_archived/`** dir (absorbed/completed
  sessions ≥90 days → `_archived/session/`). The plan's `_archive/` lane should
  standardize on `_archived/` rather than introduce a second archive root.
- Cited line numbers in §4 have drifted slightly (e.g. `_EXTRACT_SYSTEM` is now
  `distiller.py:90`); treat them as landmarks, not coordinates.

**Still unimplemented — the actual remaining plan:** two-lane layout (§3.1),
bi-modal insight notes + 7-type taxonomy (§3.2–3.3, §4.1), janitor
resolve-to-archive for the orphan/dupe backlog (§4.2), `Home.md` + weekly digest
(§4.3), `inbox/processed/` TTL purge + bi-temporal supersede (§4.4), the one-time
migration (§6). `insight` is still absent from `core/schema.py` `KNOWN_TYPES`.

**The diagnosed problem has worsened since June:** `inbox/processed/` grew from
1,935 to **5,039** files; `session/` sits at 5,198. There is still no `raw/` lane,
no `garden/`, no `Home.md`. §1's disease is live, not historical.

**New context since this was written:**
- The **Employment vault** (added 2026-06-22) postdates this doc — include it in
  any rollout that touches shared daemon code.
- The **Content vault** is dormant (config + data dir, no service, excluded from
  meta fan-out) pending AUDIT quick-win #22's wire-up/decommission decision —
  exclude it until that resolves.

---

## 1. The problem, in numbers

Audit of `/mnt/external/obsidian-vault/` (main):

| Signal | Value |
|---|---|
| Total files | 11,502 (134 MB) |
| Growth rate (last 7d) | ~524 files/day — **8.4× the 90-day baseline** |
| Machine session-noise (`session/` + `inbox/processed/` + dumps) | ~1,900 files (**35%**) |
| Actual human knowledge (`decision`+`synthesis`+`assumption`+`topic`+`note`) | ~800 files |
| Files the janitor already flagged as orphan/broken/dup | **1,532** (tagged, never resolved) |
| `inbox/processed/` (graveyard, never cleared since May 13) | 1,935 files |
| `data/alfred.log` | **105 MB** (no rotation) |

**Root cause:** Alfred has a strong **capture** stage and a strong **index/query** stage, but **no distill stage in the middle**. Everything is append-only; nothing is ever metabolized into human-facing insight or garbage-collected. The browsability pain and the bloat are the same disease.

**Validated by research:** an agent with 2,400 raw records scored **13%** task accuracy; pruned to 248 curated memories it scored **39%**. More raw logs make retrieval *worse*. So distilling helps the AI function too — this is not a cosmetic trade-off.

---

## 2. Design principles (convergent across Mem0, MemGPT/Letta, Zep, basic-memory, BASB, Zettelkasten)

1. **Two lanes.** A `raw/` lane (machine ground-truth, fully queryable, *never browsed*) and a `garden/` lane (distilled, titled, linked, the *only* thing a human sees).
2. **Distill, don't hoard.** Each session → a handful of typed atomic insights, not a 60 KB transcript.
3. **Bi-modal notes.** One markdown file serves both: frontmatter for machine retrieval, prose for human reading. This dissolves the "two competing jobs in one vault" tension.
4. **Dedup on write.** ADD / UPDATE / DELETE(supersede) / NOOP against existing notes — never blind-append.
5. **Salience gates the human surface.** An importance score decides what reaches *you* vs. what stays machine-only vs. what is dropped.
6. **Supersede, never hard-delete.** `valid_until` closes a stale fact; the raw trail is preserved for citation.
7. **Bounded surfaces.** Home dashboard + weekly digest are capped in size so they can't become new graveyards.

---

## 3. Target architecture

### 3.1 Vault layout (the two lanes)

```
obsidian-vault/
  Home.md                  ← opens on launch; the ONLY front door (Phase 1)
  garden/                  ← HUMAN LANE (browsable, ~800 → grows slowly)
    insights/YYYY/MM/      ← typed atomic insights (bi-modal notes)  [NEW]
    moc/                   ← Maps of Content: topic entry points      [NEW]
    digest/                ← weekly/monthly rollups                   [NEW]
    decision/ note/ project/ person/ org/ ...   ← existing curated types, kept
  raw/                     ← MACHINE LANE (queryable, hidden from human view)
    session/               ← was session/            (moved)
    inbox/                 ← was inbox/ + inbox/processed/ (moved)
    ai-dialogue/ ...
  _archive/                ← superseded / resolved-orphan cold storage
```

`raw/` and `_archive/` are added to Obsidian's **Excluded files** (Settings → Files & Links) so they vanish from search-as-you-type, quick-switcher, and graph — **but they stay on disk and stay in Milvus/BM25, so AI queries are unaffected.** That is the key move: human invisibility ≠ machine invisibility.

### 3.2 The bi-modal insight note (the core new artifact)

```markdown
---
type: insight
insight_type: DECISION        # DECISION|COMMITMENT|OPEN_QUESTION|LEARNING|ENTITY_FACT|PREFERENCE|RISK
created: 2026-06-20
importance: 8                 # 1-10, LLM-assigned at extraction
status: open                  # open | resolved | superseded
valid_until: null             # set (not deleted) when superseded
entities: [crm-agentic, postgres]
source: "[[raw/session/session-crm-agentic-2026-06-20-ff5d7524]]"
tags: [architecture, database]
---

## Use Postgres, not MongoDB, for CRM deal storage

Settled on Postgres as primary store for deal records. The deal schema is
stable enough that document flexibility buys nothing; JSON columns cover the
rare unstructured fields.

**Owner:** Ben · **Open:** does migration need a data-freeze window?
```

- **Human** opens it → reads the title + 3 sentences, follows the wikilink.
- **AI** retrieves it → embeds the prose, filters on `insight_type`/`status`/`entities`, cites `source`.
- **Title is an assertion**, not a topic — that's what makes it browsable and reusable.

### 3.3 The 7-type taxonomy (fixed enum — the LLM may not invent types)

| Type | What | Decay |
|---|---|---|
| `DECISION` | a settled choice | resolves/expires |
| `COMMITMENT` | action item w/ owner + deadline | resolves |
| `OPEN_QUESTION` | unresolved, needs a future answer | closes on answer |
| `LEARNING` | durable transferable claim | slow |
| `ENTITY_FACT` | fact about a named entity | faster (facts change) |
| `PREFERENCE` | how Ben likes things done | slow |
| `RISK` | something that could block progress | resolves |

---

## 4. Concrete Alfred changes (grounded in current code)

### 4.1 Extend the distiller — `src/alfred/daemons/distiller.py`

The distiller is the seed of the distill organ. Today (lines 89–101) it extracts generic "learnings" (`title/body/tags`) and appends them to machine `topic/` files via `vault_append_to_topic` (line 272). Changes:

- **Replace the extraction prompt** (`_EXTRACT_SYSTEM`, line 89) with the 7-type taxonomy + a required `importance` 1–10 score + `entities` list.
- **New writer** `vault_write_insight()` in `core/vault_ops.py` → emits bi-modal notes into `garden/insights/YYYY/MM/` instead of (or alongside) topic appends.
- **Dedup-on-write:** before writing, embed the candidate and query the existing insights via the **surveyor's Milvus index** (already built). On cosine > 0.80 → one LLM call decides UPDATE (merge) vs. NOOP; 0.60–0.80 → flag for review; < 0.60 → ADD. This reuses infra that already exists.
- **Salience gate:** `importance >= 6` → write to `garden/` (human-visible). `< 6` → write machine-only metadata / skip. Tune threshold empirically.
- **Trigger on session-end, not 30-day-stale.** Hook the distiller to fire on the session-end hook (compaction-style reflection) so insights are fresh. Keep the daily sweep as a backstop. (`DISTILL_INTERVAL`, `STALE_DAYS` lines 20–21 become the backstop, not the primary path.)

### 4.2 Promote the janitor from "flag" to "resolve" — `src/alfred/daemons/janitor.py`

It already *detects* the 1,532 orphans/dupes. Add an action stage (behind a config flag, with a grace period): move resolved-orphan / confirmed-duplicate files to `_archive/` instead of leaving them tagged in place. Archive-never-delete.

### 4.3 New: rollup + Home surface

- **`Home.md`** (static + Dataview): active projects, "what changed this week," 3 resurfaced garden notes, an `inbox/` count as a guilt gauge. Three blocks, not thirty folders.
- **Weekly digest** (`garden/digest/`): a new small daemon task or `tick()` that queries `status: open AND created >= 7d` across `garden/insights/`, narrates a **delta** ("decided / open / new"), **hard-capped at ~10 items**. Assembled *from* insights — disposable, regenerable.

### 4.4 GC / lifecycle (stops the regrowth)

- `inbox/processed/` → TTL purge to `_archive/` after N days (config).
- **Log rotation** for `data/alfred.log` (105 MB now) — `logging.handlers.RotatingFileHandler` or logrotate. Quick, independent win.
- Bi-temporal supersede: UPDATE sets old note's `valid_until` instead of creating a duplicate (kills the "4× vault-janitor decision files" pattern).

### 4.5 Schema additions — `src/alfred/core/schema.py`

Add `insight` to `KNOWN_TYPES` (line 4) and `TYPE_DIRECTORY` (line 47) → `garden/insights`. Add an `insight_type` controlled vocab. Add `lane` (raw|garden) as an optional field.

---

## 5. Phased rollout — safe & reversible first

| Phase | What | Risk | Reversible? |
|---|---|---|---|
| **0. Safety** | Backup vault + `state.json`; add log rotation; truncate `alfred.log` | none | n/a |
| **1. Two lanes + Home** | Move `session/`+`inbox/` → `raw/`; Obsidian-exclude `raw/`+`_archive/`; build `Home.md`. **No engine logic changes.** | low | yes — move back |
| **2. Distill organ** | Extend distiller: 7-type, salience, dedup-on-write, bi-modal output → `garden/insights/` | medium | yes — disable daemon |
| **3. Rollup + review** | Weekly digest; MOCs; optional review queue | low | yes |
| **4. GC/lifecycle** | janitor resolve-to-archive; `processed/` TTL; bi-temporal supersede | medium | archive (not delete) |

**Phase 1 alone** gives you a calm, browsable vault *this week* — it's pure file moves + an Obsidian setting, no risk to the engine. Everything after is incremental and individually revertible.

---

## 6. One-time migration of the existing 11.5K files

1. **Backfill move:** `session/`, `inbox/`, `inbox/processed/` → `raw/` (preserve paths). Re-point surveyor watch roots.
2. **Resolve the backlog:** run the upgraded janitor once to archive the 1,532 flagged orphans/dupes → `_archive/`.
3. **Seed the garden:** run the upgraded distiller over the **high-value backlog only** (existing `decision/`, `synthesis/`, `project/`, recent sessions) to populate `garden/insights/`. Do **not** distill all 11.5K — salience-gate it.
4. **Re-index** Milvus/BM25 against new paths (surveyor handles via filesystem diff).

Net: you open Obsidian to `Home.md` → ~800 curated notes that grow slowly and deliberately. I keep querying everything, faster, because the signal is no longer buried.

---

## 7. Decisions (locked 2026-06-20)

1. **Lane separation mechanism:** ✅ **Physical move** into `raw/`. Surveyor watch roots re-pointed; Obsidian-exclude `raw/` + `_archive/`.
2. **Salience threshold:** ✅ **6/10**. `importance >= 6` → `garden/` (human-visible); below → machine-only/skip.
3. **Review queue:** ✅ **None — trust the salience gate.** Insights flow automatically into `garden/`. Add review only if quality disappoints later.
4. **Backlog distill depth:** ✅ **Recent + high-value first** (existing `decision/`, `synthesis/`, `project/`, recent sessions). No full 11.5K sweep.
```
