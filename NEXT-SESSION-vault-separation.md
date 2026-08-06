# Next session: separate the human vault from the machine substrate

**Written:** 2026-08-06 · **Branch:** `claude/restore-ingestion-local-llm` (4 commits, unpushed, 358 tests green)

---

## The goal, in Ben's words

> "The vault is primarily 'machine exhaust' because the system is so contingent on it. I'd
> like it to be in a spot where I am also able to use it for purposes other than session
> feedback."

So this is **not** a culling project and **not** a PKM taxonomy migration. Alfred genuinely
needs the exhaust — sessions are its feed. The goal is that the same vault serves two
consumers with different needs:

| Consumer | Needs | Today |
|---|---|---|
| Alfred | every session, event, inbox drop, synthesis | ✅ working |
| Ben | ~1,400 hand-authored files, browsable without search | ❌ buried under 90% machine output |

**Separation, not deletion.** Same bytes on disk, two views. Retention/culling is one means
to that end (Step 4), not the end itself.

---

## Verified numbers — measured 2026-08-06, not estimated

Vault: `/mnt/external/obsidian-vault`, **19,165 .md files**, 1.4 GB. Alfred indexes 7,153.

```
7274 inbox/     7216 session/    1183 event/     821 synthesis/
 503 _archived/  451 note/        415 task/      397 decision/
 253 assumption/ 148 topic/       128 run/        74 wiki/
```

- **~90.2% machine-generated**; human-authored ≈ 1,878 files (9.8%), `note/` alone 451 (2.3%).
- Two agents (an architect and an adversary) working independently converged on this ratio.
  It is the single most decision-relevant fact.

**Storage — the 28 GB is NOT embeddings:**
```
data/lancedb/vault_v2.lance/_versions       25 G   (36,312 manifests, never compacted)
data/lancedb/vault_v2.lance/data           1.3 G   ← actual vectors
data/lancedb/vault_v2.lance/_transactions  123 M
```
Any eviction policy reclaims from the 1.3 GB. The 25 GB needs **LanceDB compaction + version
pruning**, which is untouched. Also: all 7,153 files were re-embedded within the last 20 days
— find out what triggered a full-corpus re-embed.

---

## Live bugs — verified, unrelated to any restructure. Fix these first.

### Bug 1 — `janitor._infer_type` is already wrong (`janitor.py:347-352`)

It inverts `TYPE_DIRECTORY`, which has collisions. Reproduced:

```
session/x.md   -> 'ai-dialogue'   ← wrong
topic/x.md     -> 'learn'         ← wrong (legacy-only type)
decision/x.md  -> ''              ← infers nothing
note/x.md      -> 'note'          ← correct
```

Repro:
```bash
.venv/bin/python -c "from alfred.core.schema import TYPE_DIRECTORY; \
d={v:k for k,v in TYPE_DIRECTORY.items()}; \
print(d.get('session'), d.get('topic'), repr(d.get('decision','')))"
```

Impact: FM001 autofix has been mislabeling/skipping the two largest content types. Fix by
walking path parts and resolving collisions explicitly, not by dict inversion.

### Bug 2 — wikilink graph is split-brain (`graph.py:225-234`)

`add_edges_from_wikilinks` adds sources as `note/foo.md` but targets as raw link text
`note/foo`. Measured on `data/graph.pkl`:

```
nodes 25,757   edges 99,479
.md nodes 16,968   non-.md 8,789   duplicates 6,897
out-degree of duplicate nodes: [0,0,0,0,0]
```

6,897 files exist as two nodes; the link-text node is a dead end, so graph spreading
activation dies at hop 2 (`engine.py:400-419`), and the synthesized
`chunk_id=f"{rel_path}::chunk_00"` matches nothing in LanceDB. **Graph-hop recall is roughly
halved today.** Fix: normalize non-`.md` nodes to `<node>.md` when a source node exists;
merge. ~6,897 merges.

### Bug 3 — index concentration

**22 files hold 92,327 of 113,230 chunks = 81.5% of the entire vector index.**

```
note/ecc-procedural-instincts-821.md   4,566 chunks
note/ecc-procedural-instincts-818.md   4,550 chunks
note/ecc-procedural-instincts-810.md   4,507 chunks
```

0.1% of files, 81.5% of the index — these ~30k-line dumps almost certainly crowd out
retrieval for everything else. They should not be single vault notes. Decide: split,
exclude from indexing, or move out of the vault.

---

## The plan, in order

### Step 1 — Phase 0 separation. Nothing moves. (½ day, 5-minute rollback)

The cheapest possible test of whether navigation was ever the real problem.

1. Set `.obsidian/app.json` → `userIgnoreFilters` (currently `None`) to hide the machine
   substrate from Obsidian's UI only: `inbox/`, `session/`, `event/`, `run/`, `_archived/`,
   `topic/`, `synthesis/`, `task/`, `input/`, `assumption/`, `constraint/`,
   `contradiction/`, `process/`, `wiki/`.
   → Ben sees ~9 folders. Alfred sees all 20. **Zero bytes moved.**
2. Add `Home.md` at vault root, pinned as startup note. Every list on it is a Base, so it
   cannot go stale. Include an **`## Unfiled`** count — the one number that honestly
   reports whether the system is being maintained.
3. **Live on it for two weeks before doing anything else.**

Rollback: revert one JSON key, delete one note.

### Step 2 — Fix Bugs 1 and 2

Self-contained, valuable regardless of what happens to folders. Add regression tests.

### Step 3 — Deal with the 22 monster files (Bug 3)

Biggest single lever on retrieval quality. Cheap to test: exclude, re-query, compare.

### Step 4 — File-level retention for the exhaust

**This is the actual fix for "the vault is mostly machine exhaust."**

Today the vault has no file retention at all: `session/` (7,216), `event/` (1,183) and
`inbox/processed/` (7,270) grow forever. The Ebbinghaus retention sweep landed today
(commit `d001bee`) evicts *embeddings* — not files.

Model it on `janitor._archive_sessions` (`janitor.py:758`), which already does this
correctly for sessions and calls `_delete_embeddings` (`:810`) so no orphan vectors are
left. Extend to `event/` and `inbox/processed/`. Keep it **opt-in and dry-run-first**, same
as the forget sweep.

### Step 5 — Only then reconsider folders

If Steps 1-4 fixed it, stop. The architect's full domain-first design is in this session's
history if it's needed.

---

## Do NOT do these

- **Do not move files naively.** `surveyor._compute_diff` (`surveyor.py:85-115`) keys state
  on `rel_path` with no move detection: a move = `deleted` + `new` = full re-embed.
  Measured from `data/alfred.log`: 0.022 s/chunk idle → **113,230 chunks ≈ 42 min**, up to
  ~4.4 h under load, during which moved files are absent from search. A repath script
  (rewrite Lance row-ids via `query_all`/`upsert_many`, rekey `state.json`, relabel
  `graph.pkl`, **re-sync md5** so the surveyor sees no diff) is *mandatory infrastructure*,
  not an optimization.
- **Do not move the machine substrate, ever.** Hardcoded paths that break silently:
  - `curator.py:76` — `vault_path / "inbox"`; rename ⇒ ingestion stops with **no error log**
  - `janitor.py:768-770` — `session/`, `_archived/session`; archival stops silently
  - `vault_ops.py:160` — `topic_dir`; distiller creates duplicate topics forever
  - `vault_ops.py:307` — `proj_dir = vault_path / "project"`; session→project linking breaks
    (⚠️ breaks even the architect's own Phase 1, which moved `project/`)
  - `consolidator.py:272` — `synthesis/` hardcoded
  - `provenance.py:17` — `DAEMON_OUTPUT_PREFIXES` is a **path-prefix** guard; move
    `synthesis/` and the consolidator starts synthesizing its own output
- **Do not add human/domain folders to `config.ignore_dirs`** — that content must stay indexed.
- `bridge/resolve.py:40` `PERSON_GLOB = "person/*.md"` breaks if `person/` ever moves.

**Useful fact if moves ever happen:** `.obsidian/app.json` has `alwaysUpdateLinks: true`
(verified), so dragging folders *inside Obsidian* rewrites all inbound wikilinks atomically.
Never use `mv`.

---

## Landed 2026-08-06 (context for what's already true)

| Commit | What |
|---|---|
| `560f4f9` | `StateStore.save()` aliasing — the merged dict was both in-memory state *and* the next save's merge base, so `curator_processed` / `distiller_runs` / `janitor_sweeps` writes were silently dropped on any instance that saved twice (i.e. the daemon) |
| `f0dd2b1` | Merged `claude/alfred-crm-bridge-pattern-a` — CRM contact resolution, note posting, brief synthesis, `enrich_crm` + CLI |
| `2a36bf6` | `alfred.core.failures` — 21 of 23 silent handlers now count into `state.error_counts`; `alfred status` shows a "swallowed errors" row |
| `d001bee` | Ebbinghaus retention: `MemoryStrength.retrievability()` + janitor forget sweep + `alfred forget` (dry-run default). **Off by default; reclaims nothing today** — see storage note above |

Corrections to earlier assumptions, for the record: there were **zero** bare `except:` in the
tree (23 silent handlers, 0 bare); the three "unlanded" hardening fixes were already
ancestors of this branch; and the 28 GB is version history, not embeddings.

Stashed: auto-generated `HANDOFF.md` churn (`git stash pop` to recover).

---

## Open questions for Ben

1. **Which failure do you actually hit?** "Can't locate a note I know exists" (structure) ·
   "Don't know if it exists" (retrieval) · "Found one, don't know if there are five more"
   (consolidation) · **"Vault is full of things I never wrote"** (separation — current
   working assumption).
2. What are the `ecc-procedural-instincts-*.md` files, and do they belong in the vault?
3. What triggered the full-corpus re-embed in the last 20 days?
4. Is Zotero in use? If you read papers, that pipeline is the highest-leverage academic
   addition and is independent of every folder decision.
5. OK to run LanceDB compaction on the live store to reclaim the 25 GB?

Full PKM research report:
`/mnt/external/obsidian-vault/inbox/research-pkm-second-brain-2026-08-06.md`
