# Audit — local-LLM migration, 2026-07-26

_13-agent adversarial sweep over the commits that replaced the Anthropic →
OpenRouter → Ollama chain with a single local backend. 4 audit lenses
(regression, silent-failure, residual-cloud, model-fitness); every finding then
handed to a verifier told to refute it. 7 confirmed, 1 refuted, 27 reported but
NOT adversarially verified — those are marked inline and should be treated as
leads, not facts._

## Status since the audit ran

**Item 1 (surveyor deletes vectors when the embedder is down) is FIXED** —
`1dacd3a`. `OllamaEmbedder.embed()` now raises `EmbeddingBackendUnavailable` on
retry exhaustion instead of returning `None`, and `_process_diff` returns before
the delete-and-record block. Regression tests in
`tests/test_surveyor_embedder_down.py`. Note the audit slightly overstated the
trigger: the retry budget is 62s and `ollama-game-guard` already orders both
edges correctly, so it does not fire on an ordinary gaming pause — it needs
ollama to die independently for over a minute while the daemon runs.

**The "shutdown fix has never executed" item is CONFIRMED, and it invalidates
`fc026eb`.** APScheduler's `AsyncIOExecutor.shutdown` ignores `wait` entirely
— its own comment reads *"There is no way to honor wait=True without converting
this method into a coroutine method"* — so `scheduler.shutdown(wait=True)` was
never what overran `TimeoutStopSec=30`. The 15s bound is harmless but fixes
nothing. Likeliest real cause: event-loop starvation from `_structural_sweep`
calling `frontmatter.load()` synchronously across 17k files inside an async job,
which prevents the SIGTERM handler from ever running. The unit that died had
consumed 2h49m CPU over 2h43m wall.

Everything else below is **unactioned**.

---

## Fix before the inbox drains

1. **Surveyor records files as indexed with zero vectors when the embedder is down, after deleting their existing vectors.** `src/alfred/embed/ollama.py:50` returns `None` for both "chunk too long" and "backend dead"; `src/alfred/surveyor.py:149-150` `continue`s, `rows` is empty, line 194-204 deletes every old chunk_id as stale, line 207 writes `FileState(md5=current, chunk_ids=[])`. `_compute_diff` never revisits it. Fix: add `EmbeddingBackendUnavailable`, raise it after retry exhaustion (keep `return None` only for the genuine `input length exceeds` skip), and in `_process_diff` catch it and `return` before any delete or state write. This is the one finding that destroys data rather than deferring work.

2. **The curator's 10-second tick is an unbounded retry loop for every non-ingest outcome.** `src/alfred/daemons/curator.py:176` (unknown type) and `:162` (empty classification) both `return False`, and `state.curator_processed[...]` is written only `if ingested` (`:103-105`). Pre-fix each retry was free (no key → no call); now every retry is a real GPU inference. Fix: land unknown types on `note` at `:176`, and add an attempt counter in `_process_inbox` that dead-letters a file after N failures. Without this the drain competes with itself and with the surveyor.

3. **`_enrich_file` writes any >20-char model output straight into a real vault record, unstamped.** `src/alfred/daemons/janitor.py:425`. A STUB_RECORD is by definition context-free, so a refusal ("I don't have enough information…", 88 chars) clears the gate and replaces the body. `provenance.is_daemon_generated` doesn't match it, so it reads as human-authored, gets re-embedded, and — worse — `_deep_sweep` clears STUB001 and the now-longer body never re-trips the <50 check, so it is permanent. Fix: reject refusal-shaped and >2000-char output, and pass `set_fields={"generated_by": "llm"}` to `vault_edit`.

4. **The distiller stamps `last_distilled` when the model's JSON was unusable.** `src/alfred/daemons/distiller.py:167` stamps whenever `_distill_file` returns without raising; `:257-262` returns `0` for `JSONDecodeError` and for a non-list — identical to a legitimate empty extraction. That record is then excluded for `STALE_DAYS = 30`. A 7B model hits wrong-shape output materially more than the cloud model did. Fix: return `None` from the two failure branches and `continue` without stamping.

5. **A corrupt `graph.pkl` is silently replaced by a 2-node graph.** `src/alfred/store/graph.py:184` returns `False` on any exception without setting `self._g`; `_graph()` then lazily builds an empty DiGraph and the surveyor's per-file `load → add_edges → save` (`surveyor.py:214-222`) commits it. Live graph is 24,122 nodes / 92,021 edges, and `build_from_vault` has no caller, so there is no rebuild path — the ~51k wikilink edges would be permanently lost. Fix: set a `_load_failed` flag on an unpickle failure of an *existing* file and have `save()` refuse to write; also replace that bare `except Exception: pass` with a log.

6. **Synthesis sends up to 60k chars to a model whose context window is never declared.** `query/context.py:14` is `MAX_CONTEXT_CHARS = 60_000`; `core/local_llm.py:60` sets only `num_predict` — `num_ctx` appears nowhere in `src/`. Ollama silently truncates to its default window, dropping most of the retrieved context and possibly the system prompt, and returns a clean 200. This never bit before because Ollama was the never-exercised last resort. Fix: pass an explicit `num_ctx` in `local_llm` and lower `MAX_CONTEXT_CHARS` to fit it.

7. **Cluster labels become file paths with no slugification.** `src/alfred/daemons/consolidator.py:185` and `:252` do `"-".join(label.lower().split())[:60]`. `LABEL: AI/ML research notes` yields `synthesis/ai/ml-research-notes.md` and `mkdir(parents=True)` silently creates a bogus subdirectory; the no-LABEL fallback (`:394-398`) stores `raw[:50]` — e.g. "Sure! Here is a descriptive label for this cluste" — as the filename. Fix: reuse `curator._slugify`.

8. **`classification["type"]` is used unguarded and crashes on a list.** `curator.py:171`: `['session'] in KNOWN_TYPES` → `TypeError: unhashable type`, `correct_type(['session'])` → `AttributeError`. The generic handler logs it and the file rejoins the retry loop from item 2. The sibling `tags` field is already guarded at `:194-195`, so this is an inconsistency, not a design choice. Fix: normalise in `_classify` before returning.

9. **(unverified) `_find_richest_topic_by_tag` can redirect a topic append into a Syncthing conflict copy.** `core/vault_ops.py:169` — the conflict filter was added to `vault_search` (`:294`) and `vault_context` (`:327`) but not here, and "richest" means most lines, which the janitor's own data says favours the conflict copy in 9 of 129 pairs. 54 conflict files live in `topic/` today. Fix: `if is_sync_conflict(md_file): continue` — `is_sync_conflict` is already imported at `:18`.

## Fix soon

- **(unverified) A truncated distiller sweep still records a run**, so `runner.py:168-172` sees a fresh timestamp and skips the `distiller_catchup` job — the sweep cancels its own recovery. Worse than written: `distilled_count` increments for files that never reach the backend (short body, daemon record, read failure), so one skippable file ahead of the first real one is enough. Not permanent — the 02:00 cron still fires and `last_distilled` genuinely resumes — but you lose the opportunistic drain-on-restart. `distiller.py:190`.
- **(unverified) Two consolidator call sites were never migrated.** `consolidator.py:316-326` and `:380-403` still post raw to `/api/generate` and `except Exception: return ""`, and `_synthesis_pass` counts those as successes (`consolidator.synthesized clusters=5` during a total outage). The block labelled "Fallback 1" is a second call to the same dead endpoint. Route both through `local_llm.complete`.
- **(unverified) `janitor.py:413` byte-cap raises instead of truncating**: slices the `str` then calls `.decode` on it. `AttributeError` on the exact line meant to prevent oversized prompts; the file is retried on every deep sweep forever. One-character fix — slice `prompt_bytes`, not `prompt`.
- **(unverified) State entry popped even when the vector delete failed** — `janitor.py:444` and `surveyor.py:125`. Orphaned chunks stay in LanceDB, still rank in `search()`, consume top_k slots, and are unreachable by the ghost-prune (which iterates `state.files`). Only pop on success.
- **(unverified) Wiki writer has three separate ordering bugs**: `ensure_page` swallows every `VaultError` and records the page as created (`writer.py:66`); `enrich_page` mutates `page.known_facts` before the `vault_edit` and returns `True` even when it raised (`:98`), and the `f not in page.known_facts` filter then makes the loss permanent; `:128` iterates a bare string into single-character facts and `[[A]]`-style wikilinks. `enrich_page` has no callers in `src/` today, which is the only reason this is not blocking.
- **(unverified) `_check_file` returns `[]` on unparseable frontmatter** (`janitor.py:230`), and `_structural_sweep` clears `open_issues` before repopulating — so the one file with genuinely broken YAML is the only one the linter reports as clean. Emit a distinct PARSE001 code.
- **(unverified) Session archival unlinks a live file on a name collision** without comparing content (`janitor.py:656`), and slug reuse is reachable once the original leaves `session/`. Compare hashes; disambiguate like `curator.py:253-256` already does.
- **(unverified) Conflict copies persist as graph nodes forever** — neither deletion path calls `GraphStore.remove_file` (which has no callers), and `engine._spread_activate` (`engine.py:409`) injects any activated rel_path as a hit with no index-membership check, reading content straight off disk. 427 such nodes live today. Add the `remove_file` calls plus an `is_sync_conflict` skip in the activation loop.
- **The whole api_budget subsystem is orphaned** — `can_make_api_call` / `record_api_call` / `budget_remaining` have zero callers after 82cd5d8, yet the config keys still load and are allowlisted in `_CONSUMED`. Downstream, `mcp/tools.py:156` (`vault_api_cost`, live on the running HTTP server) and `ledger/sources.py:298` read counters nothing writes. The 0.0 is currently *correct*, so this is cleanup, not a lie — delete the subsystem, the two `api_budget` YAML blocks, the ledger metric, and the `claude-sonnet-4-6` pricing sentence in the tool docstring. Also: `tests/test_config_consumed.py` only asserts a YAML leaf mutates *some* dataclass field, never that the field is *read* — which is exactly why this passed the dead-key guard. Strengthen it.
- **The shutdown fix has never executed.** `alfred.service` is `failed (Result: timeout)` from a SIGKILL 5.5h *before* the `SHUTDOWN_GRACE_S` commit landed, `NRestarts=0`. Separately, the diagnosis is wrong: `AsyncIOScheduler.shutdown()` dispatches via `call_soon_threadsafe` and returns in ~4ms, and `AsyncIOExecutor.shutdown` discards `wait` entirely — so it was never what overran `TimeoutStopSec=30`. Instrument the actual stop path (task-group cancel, `state_store.save`, surveyor teardown) before re-fixing; the likeliest real cause is blocking sync work inline in a job coroutine stalling the loop so `await stop_event.wait()` never resumes.
- **v1 still runs the original bug.** `/mnt/external/Projects/personal-alfred/alfred/query.py:297,306` still has the Anthropic → OpenRouter → Ollama chain with `e.__class__.__name__`, the hardcoded `claude-sonnet-4-6`, and the deprecated `x-ai/grok-4.1-fast` slug (also defaulted in `alfred/surveyor/config.py:60` and `quickstart.py:319`). Both keys are live in the shared `.env`, so both branches are entered on every v1 query.
- **`git worktree remove .claude/worktrees/wf_7d16400e-fec-7`** — branch `spike/single-process`, 13 days stale, contains all six pre-migration Anthropic call sites and four config files with the dead model ids. It is a merge-time reintroduction risk and roughly half the raw grep hits in this audit came from it.

## Model choice

**Set `ollama.llm_model` to `qwen2.5:7b-instruct` for the classifier path, and — more importantly — constrain the output at decode time by passing Ollama's `format` a JSON *schema* with `"type": {"enum": [...KNOWN_TYPES]}` instead of the bare `"json"` string.** `format: "json"` guarantees syntax only, not shape, which is the single root cause behind three of the findings above (out-of-vocab types, list-valued `type`, string-instead-of-array facts) — schema-constrained decoding makes all three structurally impossible regardless of model. Within that, qwen2.5:7b-instruct is a strictly better schema-follower than `mistral:latest` (7B v0.3) at the same VRAM footprint, so it also cuts the wrong-shape rate on the unconstrained paths (distiller extraction, cluster labels) that a schema is harder to write for.

## Not worth fixing

- `core/anthropic_client.py` — zero importers; the only real change worth making is raising on an empty `ANTHROPIC_API_KEY` instead of `.get(..., "")`, and that costs more than it saves while nothing calls it.
- The shared `.env` symlink into the v1 tree — the two dead keys are set and ignored by v2; do not touch it until v1's `query.py` fallback chain is fixed, or you break v1 silently.
- `pm_agent.py` and its two uninstalled unit files — dormant, and it crashes loudly rather than silently if ever enabled.
- The 12 subagent `model: claude-sonnet-4-6` frontmatter entries and the two secret-redaction regexes containing `ANTHROPIC_API_KEY` — non-callers; stripping them breaks working timers or leaks a real key into the vault.
- `janitor.py:365` clearing STUB_RECORD after a no-op enrichment — self-healing, the hourly structural sweep re-adds it before the next deep sweep.
- `writer.py:109` in-try import referenced by the `except` clause — real NameError shape, but `enrich_page` has no callers.
- `runner.py:324-333` `if scope.cancelled_caught:` — dead branch. Checked and dismissed: the `anyio` claim about `abandon_on_cancel` is technically correct but the wrapped call returns in ~4ms, so the branch can never fire either way. The 15s bound harms nothing; leave it and chase the real overrun instead.
- `scripts/_archive/phase1b_merge_drafts.py` — inert either way.
