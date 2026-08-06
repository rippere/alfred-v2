"""The deletion half of Ebbinghaus: decay score → retention action.

Before this, Ebbinghaus existed only as a retrieval-ranking nudge — memory
decayed a hit's *score* but nothing ever acted on the decay, so the vector
store only ever grew (28 GB of lancedb against a 1.4 GB vault).

The load-bearing test here is
test_forgotten_file_is_not_re_embedded_by_surveyor_diff: eviction is only
worth anything if it sticks. The surveyor decides what to embed by diffing
vault md5s against state.files, so evicting a file by *popping* its state
entry makes it reappear as "new" on the next tick and re-embed immediately —
which is how the store grew back. The entry must be kept, md5 intact, with
its vectors cleared.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from alfred.core.models import FileState, MemoryStrength, RETENTION_SCALE_DAYS
from alfred.daemons.janitor import JanitorDaemon
from alfred.store.state import StateStore

NOW = datetime(2026, 8, 6, tzinfo=timezone.utc)


class _FakeStore:
    """Stand-in for LanceDBStore recording what was asked to be deleted."""

    def __init__(self, fail_on: set[str] | None = None):
        self.deleted: list[tuple[str, list]] = []
        self.fail_on = fail_on or set()

    def delete_file(self, rel_path, chunk_ids=None):
        if rel_path in self.fail_on:
            raise RuntimeError("vector store unavailable")
        self.deleted.append((rel_path, list(chunk_ids or [])))


class _Cfg:
    janitor_forget_enabled = True
    janitor_forget_retrievability = 0.02
    janitor_forget_min_age_days = 180
    janitor_forget_max_per_sweep = 500
    vault_path = None


def _janitor(tmp_path, store=None, **cfg_overrides):
    state = StateStore(tmp_path / "state.json")
    state.load()
    cfg = _Cfg()
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)
    cfg.vault_path = tmp_path / "vault"
    j = JanitorDaemon(cfg, state, asyncio.Queue(), store=store or _FakeStore())
    return j, state


def _add_file(state, rel_path, *, days_since_embed, chunks=3, accessed_days_ago=None,
              access_count=1, forgotten=""):
    embedded = (NOW - timedelta(days=days_since_embed)).isoformat()
    state.state.files[rel_path] = FileState(
        md5=f"md5-{rel_path}",
        last_embedded=embedded,
        chunk_ids=[f"{rel_path}#{i}" for i in range(chunks)],
        forgotten=forgotten,
    )
    if accessed_days_ago is not None:
        ms = MemoryStrength(rel_path=rel_path, access_count=access_count)
        ms.last_accessed = (NOW - timedelta(days=accessed_days_ago)).isoformat()
        ms.stability = 1.0
        state.state.memory[rel_path] = ms


# ── the decay score ───────────────────────────────────────────────────────────

def test_retrievability_never_accessed_is_zero():
    """No retrieval evidence is the weakest possible memory, not a neutral
    one — otherwise never-queried files could never become candidates."""
    assert MemoryStrength(rel_path="a.md").retrievability(NOW) == 0.0


def test_retrievability_decays_toward_zero_with_time():
    ms = MemoryStrength(rel_path="a.md", access_count=1, stability=1.0)
    ms.last_accessed = NOW.isoformat()
    fresh = ms.retrievability(NOW)

    ms.last_accessed = (NOW - timedelta(days=RETENTION_SCALE_DAYS)).isoformat()
    one_scale = ms.retrievability(NOW)

    ms.last_accessed = (NOW - timedelta(days=RETENTION_SCALE_DAYS * 10)).isoformat()
    far = ms.retrievability(NOW)

    assert fresh == pytest.approx(1.0)
    assert one_scale == pytest.approx(1 / 2.718281828, rel=1e-3), "R should be 1/e at one scale"
    assert far < 0.001
    assert fresh > one_scale > far


def test_higher_stability_decays_slower():
    """A file read 50 times must survive far longer than one read once —
    that is the entire point of stability."""
    weak = MemoryStrength(rel_path="a.md", access_count=1, stability=1.0)
    strong = MemoryStrength(rel_path="b.md", access_count=50, stability=5.0)
    ago = (NOW - timedelta(days=200)).isoformat()
    weak.last_accessed = strong.last_accessed = ago
    assert strong.retrievability(NOW) > weak.retrievability(NOW)


def test_score_modifier_still_bounded_and_independent():
    """retrievability() must not have turned into score_modifier(): ranking
    stays clamped at 0.5 so a stale file ranks lower but never vanishes."""
    ms = MemoryStrength(rel_path="a.md", access_count=1, stability=1.0)
    ms.last_accessed = (NOW - timedelta(days=10_000)).isoformat()
    assert ms.score_modifier() == 0.5
    assert ms.retrievability(NOW) < 0.001


# ── candidate selection ───────────────────────────────────────────────────────

def test_old_never_accessed_file_is_a_candidate(tmp_path):
    j, state = _janitor(tmp_path)
    _add_file(state, "notes/cold.md", days_since_embed=400)
    got = [c["rel_path"] for c in j.forget_candidates(NOW)]
    assert got == ["notes/cold.md"]


def test_recently_embedded_file_is_protected_regardless_of_decay(tmp_path):
    """Cold-start guard: with no query history every file looks unaccessed,
    so without the age floor the first sweep would evict everything."""
    j, state = _janitor(tmp_path)
    _add_file(state, "notes/new.md", days_since_embed=10)
    assert j.forget_candidates(NOW) == []


def test_recently_read_file_is_protected(tmp_path):
    j, state = _janitor(tmp_path)
    _add_file(state, "notes/hot.md", days_since_embed=400, accessed_days_ago=1)
    assert j.forget_candidates(NOW) == []


def test_already_forgotten_and_unembedded_files_are_skipped(tmp_path):
    j, state = _janitor(tmp_path)
    _add_file(state, "notes/gone.md", days_since_embed=400, forgotten=NOW.isoformat())
    _add_file(state, "notes/nochunks.md", days_since_embed=400, chunks=0)
    assert j.forget_candidates(NOW) == []


def test_candidates_ordered_coldest_first(tmp_path):
    """The per-sweep cap slices this list, so ordering decides what actually
    gets evicted — it must be the coldest, not an arbitrary slice."""
    j, state = _janitor(tmp_path)
    _add_file(state, "notes/warmish.md", days_since_embed=400,
              accessed_days_ago=300, access_count=1)
    _add_file(state, "notes/frozen.md", days_since_embed=400)
    got = [c["rel_path"] for c in j.forget_candidates(NOW)]
    assert got[0] == "notes/frozen.md", f"coldest file must sort first, got {got}"


def test_unparsable_embed_timestamp_is_counted_not_swallowed(tmp_path):
    from alfred.core.failures import peek_failures, reset_failures

    reset_failures()
    j, state = _janitor(tmp_path)
    _add_file(state, "notes/bad.md", days_since_embed=400)
    state.state.files["notes/bad.md"].last_embedded = "not-a-timestamp"

    assert j.forget_candidates(NOW) == []
    assert peek_failures().get("janitor.forget_timestamp_unparsable") == 1
    reset_failures()


# ── the retention action ──────────────────────────────────────────────────────

def test_forget_evicts_vectors_but_keeps_the_state_entry(tmp_path):
    store = _FakeStore()
    j, state = _janitor(tmp_path, store=store)
    _add_file(state, "notes/cold.md", days_since_embed=400, chunks=3)

    asyncio.run(j._forget_sweep())

    assert store.deleted == [("notes/cold.md", [f"notes/cold.md#{i}" for i in range(3)])]
    fs = state.state.files["notes/cold.md"]
    assert fs.chunk_ids == []
    assert fs.last_embedded == ""
    assert fs.forgotten, "eviction must be stamped so it isn't reconsidered every sweep"
    assert fs.md5 == "md5-notes/cold.md", "md5 must survive — it is what stops re-embedding"


def test_forgotten_file_is_not_re_embedded_by_surveyor_diff(tmp_path):
    """The load-bearing property. The surveyor computes new/changed/deleted by
    comparing on-disk md5s against state.files. A forgotten file must appear
    in NONE of those buckets — otherwise it re-embeds on the next tick and the
    storage comes straight back, which is exactly what used to happen."""
    store = _FakeStore()
    j, state = _janitor(tmp_path, store=store)
    _add_file(state, "notes/cold.md", days_since_embed=400)
    asyncio.run(j._forget_sweep())

    # Reproduce _compute_diff's comparison against an unchanged vault file.
    current = {"notes/cold.md": "md5-notes/cold.md"}
    known = state.state.files
    new = [r for r in current if r not in known]
    changed = [r for r in current if r in known and current[r] != known[r].md5]
    deleted = [r for r in known if r not in current]

    assert new == [], "forgotten file reappeared as new — it would be re-embedded"
    assert changed == [], "forgotten file looks changed — it would be re-embedded"
    assert deleted == [], "forgotten file looks deleted — its state would be dropped"


def test_editing_a_forgotten_file_brings_it_back(tmp_path):
    """Forgetting must be reversible by normal use, or it is data loss."""
    j, state = _janitor(tmp_path)
    _add_file(state, "notes/cold.md", days_since_embed=400)
    asyncio.run(j._forget_sweep())

    current = {"notes/cold.md": "md5-EDITED"}
    known = state.state.files
    changed = [r for r in current if r in known and current[r] != known[r].md5]
    assert changed == ["notes/cold.md"], "an edited forgotten file must re-embed"


def test_sweep_is_a_no_op_when_disabled(tmp_path):
    store = _FakeStore()
    j, state = _janitor(tmp_path, store=store, janitor_forget_enabled=False)
    _add_file(state, "notes/cold.md", days_since_embed=400)

    asyncio.run(j._forget_sweep())
    asyncio.run(j.forget_tick())

    assert store.deleted == []
    assert state.state.files["notes/cold.md"].chunk_ids != []


def test_per_sweep_cap_limits_evictions(tmp_path):
    store = _FakeStore()
    j, state = _janitor(tmp_path, store=store, janitor_forget_max_per_sweep=2)
    for i in range(5):
        _add_file(state, f"notes/c{i}.md", days_since_embed=400 + i)

    asyncio.run(j._forget_sweep())
    assert len(store.deleted) == 2, "cap must bound a single sweep"
    remaining = [p for p, fs in state.state.files.items() if fs.chunk_ids]
    assert len(remaining) == 3


def test_vector_delete_failure_leaves_state_untouched(tmp_path):
    """If the vectors could not actually be deleted, the file must NOT be
    marked forgotten — otherwise state claims reclaimed space that is still
    on disk, and nothing will ever retry it."""
    from alfred.core.failures import reset_failures

    reset_failures()
    store = _FakeStore(fail_on={"notes/cold.md"})
    j, state = _janitor(tmp_path, store=store)
    _add_file(state, "notes/cold.md", days_since_embed=400)

    asyncio.run(j._forget_sweep())

    fs = state.state.files["notes/cold.md"]
    assert fs.chunk_ids != [], "state was cleared even though the vectors survived"
    assert fs.forgotten == ""
    # Asserted on persisted state, not the in-process counter: the sweep ends
    # in save_state(), which drains the counters into error_counts by design.
    assert state.state.error_counts.get("janitor.forget_vector_delete_failed") == 1
    reset_failures()


def test_sweep_result_persists_across_processes(tmp_path):
    j, state = _janitor(tmp_path)
    _add_file(state, "notes/cold.md", days_since_embed=400)
    asyncio.run(j._forget_sweep())

    reborn = StateStore(tmp_path / "state.json")
    reborn.load()
    fs = reborn.state.files["notes/cold.md"]
    assert fs.forgotten, "eviction must survive the process that made it"
    assert fs.chunk_ids == []
    assert fs.md5 == "md5-notes/cold.md"
