"""Orphan reaper — delete vector rows whose source note is gone for good.

An *orphan* is a row in the vector store that no longer corresponds to
anything: its rel_path has no entry in ``state.files`` AND its ``.md`` is
absent from the vault.  Both halves are required, and the reason is the
2026-08-07 incident chain.

Why not "absent from state.files" alone — the tempting definition, and the
one that would destroy data.  ``state.files`` is a *lagging* index of the
vault, not a mirror of it.  The surveyor persists state every 25 files
(b659e9c), so at any instant up to 25 freshly-embedded files have rows in the
store and no state entry; a crash widens that window arbitrarily.  Measured
against the live store on 2026-08-07, that window was 2,892 files / 45,352
rows wide — 29% of the index — and every one of those files still existed in
the vault.  "Not in state" means "possibly not recorded yet".  It is not
evidence of anything.

Why not "absent from the vault" alone.  A file can be temporarily unreadable
— /mnt/external not mounted yet, a sync in flight — while state.files still
legitimately tracks it.  Vault-absence is only trustworthy when state agrees
the file is gone too.

Why every path is apostrophe-normalised.  ``core/vault.py::_safe_chunk_id``
does ``rel_path.replace("'", "")`` before minting the chunk id, so rel_path
is NOT recoverable from an id: 57 vault files contain apostrophes.
Un-normalised, such a file looks absent from state (state holds the real
name) AND absent from the vault (we would stat the stripped name), so BOTH
safety checks pass and a live file gets reaped.  Normalising all three sides
can only ever cause a path to be RETAINED — "Alfred's" and "Alfreds" collapse
together — never deleted.  This is the highest-severity path in the module.

Deletion goes through ``LanceDBStore.delete_ids`` in 500-id batches rather
than a prefix predicate, deliberately: a prefix built from the raw rel_path
re-inserts the apostrophe that the id never had, and would match zero rows
while the caller believed it had deleted the file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from alfred.core.failures import record_failure

log = structlog.get_logger()

# Absolute floor on the vault walk before any reap is allowed. The ratio guard
# alone cannot catch a broken mount when state is also small, and the failure
# mode is deleting live data — so this is a blunt "the vault cannot plausibly
# be this small" check. The real vault holds ~19,500 .md files.
_MIN_VAULT_FILES = 0   # off by default; the empty-state abort is the load-bearing guard

# Anchored, so a non-conforming id surfaces loudly instead of being rsplit
# into a bogus rel_path that then passes both safety checks and gets reaped.
# ``.+`` is greedy, so a rel_path that itself contains "::" splits on the
# LAST "::chunk_NN" — the only interpretation that round-trips _safe_chunk_id.
CHUNK_ID_RE = re.compile(r"^(?P<path>.+)::chunk_(?P<idx>\d+)$")

# Cap on retained malformed ids: they are diagnostics, and an id column that
# is entirely malformed must not become an unbounded list in memory — the
# whole point of the streaming scan is that peak does not track row count.
MAX_MALFORMED_SAMPLES = 50


def normalize_rel_path(rel_path: str) -> str:
    """Mirror ``core/vault.py::_safe_chunk_id``'s lossy transform.

    Kept as a read-side normalisation rather than fixing _safe_chunk_id: that
    would change every id already minted, and five call sites read ids back
    with a bare rsplit (lancedb_store.py, milvus.py, query/engine.py,
    query/context.py, daemons/surveyor.py) plus every chunk_id persisted in
    state.json.  Migrating those is a separate, larger change.
    """
    return rel_path.replace("'", "")


@dataclass
class ReapPlan:
    """What a reap would do.  Produced without touching anything."""

    store_rows: int = 0
    store_paths: int = 0
    # (normalized rel_path, row count) for paths absent from BOTH state and vault.
    orphans: list[tuple[str, int]] = field(default_factory=list)
    # Orphan paths skipped because they alone exceed the per-sweep row cap.
    deferred: list[tuple[str, int]] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)
    malformed_count: int = 0
    # Paths with store rows and no state entry whose .md IS present — the
    # 25-file window at scale.  Reported so the number stays visible; never reaped.
    untracked_but_present: int = 0
    untracked_but_present_rows: int = 0
    aborted: str | None = None

    @property
    def reapable_rows(self) -> int:
        return sum(n for _, n in self.orphans)

    @property
    def reapable_paths(self) -> int:
        return len(self.orphans)


def scan_store_paths(store, batch_size: int = 4096) -> tuple[dict[str, int], list[str], int]:
    """Stream the id column and count rows per normalised rel_path.

    Returns ``(counts, malformed_samples, malformed_count)``.  Memory is
    bounded by the number of *distinct paths*, not by row count — the whole
    id column is never held.
    """
    counts: dict[str, int] = {}
    malformed: list[str] = []
    malformed_count = 0
    for cid in store.iter_ids(batch_size=batch_size):
        m = CHUNK_ID_RE.match(cid)
        if m is None:
            malformed_count += 1
            if len(malformed) < MAX_MALFORMED_SAMPLES:
                malformed.append(cid)
            continue
        key = normalize_rel_path(m.group("path"))
        counts[key] = counts.get(key, 0) + 1
    return counts, malformed, malformed_count


def collect_ids(store, paths: set[str], batch_size: int = 4096) -> dict[str, list[str]]:
    """Second streaming pass: gather the actual ids belonging to *paths*.

    Two passes rather than one so pass 1 never holds 157k id strings.  The
    scan costs ~2 s; re-running it is cheaper and safer than trusting a
    checkpoint that may have gone stale against a store the surveyor is
    still writing to.
    """
    out: dict[str, list[str]] = {p: [] for p in paths}
    if not paths:
        return out
    for cid in store.iter_ids(batch_size=batch_size):
        m = CHUNK_ID_RE.match(cid)
        if m is None:
            continue
        key = normalize_rel_path(m.group("path"))
        bucket = out.get(key)
        if bucket is not None:
            bucket.append(cid)
    return out


def vault_rel_paths(vault_path: Path) -> set[str]:
    """Every ``.md`` in the vault, rel to vault root, apostrophe-normalised."""
    return {
        normalize_rel_path(str(p.relative_to(vault_path)))
        for p in vault_path.rglob("*.md")
    }


def build_plan(
    store,
    state,
    vault_path: Path,
    max_rows: int = 5000,
    batch_size: int = 4096,
    min_vault_files: int = _MIN_VAULT_FILES,
) -> ReapPlan:
    """Compute what is reapable.  Pure: reads only, mutates nothing.

    ``state`` is a ``PipelineState`` (the ``.state`` of a StateStore).
    """
    plan = ReapPlan()

    # Abort guards.  An unmounted /mnt/external turns "absent from the vault"
    # into "delete everything", which is the single way this module could
    # destroy the index, so the walk is sanity-checked before it is trusted.
    if not vault_path.exists() or not vault_path.is_dir():
        plan.aborted = f"vault path missing: {vault_path}"
        log.warning("reap.aborted", reason=plan.aborted)
        return plan

    vault_paths = vault_rel_paths(vault_path)
    if not vault_paths:
        plan.aborted = f"vault walk returned 0 .md files under {vault_path}"
        log.warning("reap.aborted", reason=plan.aborted)
        return plan

    state_paths = {normalize_rel_path(k) for k in state.files}
    # An empty state must reap NOTHING, not everything. StateStore._read_raw
    # returns {} for a *missing* state.json — silently, no exception — so an
    # unreadable or not-yet-created state is indistinguishable here from "no
    # files tracked". With state_paths empty, every store path fails the
    # not-in-state test and becomes an orphan candidate, and the ratio guard
    # below is arithmetically vacuous: 0.9 * 0 == 0, and len(vault_paths) < 0
    # is never true. Proven by execution: empty state plus a vault walk
    # returning 1 of 100 live notes reaped 99 live files (297 rows) and left
    # plan.aborted unset.
    if not state_paths:
        plan.aborted = "state.json tracks 0 files — refusing to treat the whole store as orphaned"
        log.warning("reap.aborted", reason=plan.aborted)
        return plan
    # An absolute floor, in ADDITION to the ratio guard below, for deployments
    # whose state is small enough that a proportional check is weak. Defaults
    # to 0 (off) because the empty-state abort above already covers the case
    # that actually destroyed data; production sets it via config.
    if min_vault_files and len(vault_paths) < min_vault_files:
        plan.aborted = (
            f"vault walk found only {len(vault_paths)} .md files — refusing to reap "
            f"against a vault that small (partial mount?)"
        )
        log.warning("reap.aborted", reason=plan.aborted)
        return plan
    # Alfred only ever embeds files it found in the vault, so a vault holding
    # far fewer notes than state tracks means the walk is lying (partial
    # mount, sync in progress) — not that the notes were deleted.
    if len(vault_paths) < 0.9 * len(state_paths):
        plan.aborted = (
            f"vault walk found {len(vault_paths)} .md but state tracks "
            f"{len(state_paths)} files — refusing to treat that as deletion"
        )
        log.warning("reap.aborted", reason=plan.aborted)
        return plan

    counts, malformed, malformed_count = scan_store_paths(store, batch_size=batch_size)
    plan.malformed = malformed
    plan.malformed_count = malformed_count
    plan.store_paths = len(counts)
    plan.store_rows = sum(counts.values())
    if malformed_count:
        record_failure(
            "reap.malformed_chunk_id",
            error=ValueError(f"{malformed_count} ids did not match {CHUNK_ID_RE.pattern}"),
            count=malformed_count,
        )

    candidates: list[tuple[str, int]] = []
    for path, n in counts.items():
        if path in state_paths:
            continue
        if path in vault_paths:
            # The 25-file window (or a crash mid-embed).  Live data.
            plan.untracked_but_present += 1
            plan.untracked_but_present_rows += n
            continue
        candidates.append((path, n))

    # Largest first: orphan mass is heavily concentrated (30 paths hold 80% of
    # the live index), so the cap should buy back the most rows per sweep.
    candidates.sort(key=lambda pn: (-pn[1], pn[0]))

    budget = max_rows
    for path, n in candidates:
        if n > max_rows:
            # A single path bigger than the whole cap is deferred rather than
            # silently blowing through the bound.  Conservative on purpose:
            # raise --max-rows deliberately instead of having the cap mean
            # nothing.  Reported so it cannot sit unnoticed forever.
            plan.deferred.append((path, n))
            continue
        if n > budget:
            plan.deferred.append((path, n))
            continue
        plan.orphans.append((path, n))
        budget -= n

    return plan


def execute_plan(
    store,
    state,
    vault_path: Path,
    plan: ReapPlan,
    batch_size: int = 4096,
    delete_batch: int = 500,
) -> int:
    """Delete the planned orphan rows.  Returns rows actually deleted.

    Every path is re-checked against state and the vault immediately before
    its delete, not just at plan time.  The surveyor writes state every 25
    files while this runs, so a path that was orphan-shaped during the scan
    can legitimately have become tracked before we get to it; deleting it
    then would wipe chunks the surveyor had just written.
    """
    if plan.aborted or not plan.orphans:
        return 0

    state_paths = {normalize_rel_path(k) for k in state.files}
    # Re-walk the vault into the SAME normalized key space the plan uses.
    # `path` here is already apostrophe-stripped (normalize_rel_path, mirroring
    # core/vault.py's chunk-id construction), so the old `(vault_path /
    # path).exists()` stat looked for "Alfreds note.md" when the real file is
    # "Alfred's note.md" — the guard could never fire for the 57 apostrophe
    # files in this vault. Proven by execution: an apostrophe note restored
    # between plan and apply had all 4 of its fresh rows deleted, while the
    # non-apostrophe control correctly logged reap.skip_now_present.
    live_paths = vault_rel_paths(vault_path)
    planned = {p for p, _ in plan.orphans}
    ids_by_path = collect_ids(store, planned, batch_size=batch_size)

    deleted = 0
    for path, _ in plan.orphans:
        if path in state_paths:
            log.info("reap.skip_now_tracked", path=path)
            continue
        if path in live_paths:
            log.info("reap.skip_now_present", path=path)
            continue
        ids = ids_by_path.get(path) or []
        if not ids:
            continue
        try:
            deleted += store.delete_ids(ids, batch=delete_batch)
        except Exception as e:
            # One unreapable path must not abort the sweep, but a silently
            # swallowed vector delete is exactly how orphans were created in
            # the first place, so it is counted.
            record_failure("reap.vector_delete_failed", error=e, path=path)
    return deleted
