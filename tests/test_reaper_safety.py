"""Data-loss guards for the orphan reaper.

Every test here corresponds to a path PROVEN by execution to delete live data
before it was fixed. tests/test_orphan_reaper.py covers the happy path; these
cover the ways the reaper destroys the vault's index, which is the only failure
mode that actually matters for this module.

B1 — an empty state.json made every store path an orphan candidate, and the
     ratio guard was arithmetically vacuous (0.9 * 0 == 0, and len(x) < 0 is
     never true). StateStore returns {} for a MISSING state file, silently.
     Proven: empty state + a vault walk returning 1 of 100 live notes reaped
     99 live files / 297 rows without aborting.
B2 — the pre-delete vault re-check stat'd an apostrophe-STRIPPED path (chunk
     ids strip "'", mirroring core/vault.py), so for the 57 apostrophe files in
     this vault it looked for "Alfreds note.md" and could never fire. Proven:
     an apostrophe note restored between plan and apply had all 4 fresh rows
     deleted; the non-apostrophe control correctly skipped.
"""
from __future__ import annotations

from pathlib import Path

from alfred.core.models import FileState
from alfred.store.reaper import build_plan, execute_plan, normalize_rel_path
from alfred.store.state import StateStore


class FakeStore:
    """Vector store stub: holds ids, records what was deleted."""

    def __init__(self, ids: list[str]) -> None:
        self._ids = list(ids)
        self.deleted: list[str] = []

    def count(self) -> int:
        return len(self._ids)

    def iter_ids(self, batch_size: int = 4096):
        yield from self._ids

    def delete_ids(self, chunk_ids, batch: int = 500) -> int:
        self.deleted.extend(chunk_ids)
        drop = set(chunk_ids)
        self._ids = [i for i in self._ids if i not in drop]
        return len(chunk_ids)


def _vault(tmp_path: Path, names: list[str]) -> Path:
    v = tmp_path / "vault"
    (v / "note").mkdir(parents=True, exist_ok=True)
    for n in names:
        (v / n).parent.mkdir(parents=True, exist_ok=True)
        (v / n).write_text("---\ntype: note\n---\nbody\n", encoding="utf-8")
    return v


def _ids_for(names: list[str], per_file: int = 3) -> list[str]:
    return [
        f"{normalize_rel_path(n)}::chunk_{i:02d}"
        for n in names for i in range(per_file)
    ]


def _state(tmp_path: Path, tracked: list[str]) -> StateStore:
    st = StateStore(tmp_path / "state.json")
    st.load()
    for n in tracked:
        st.state.files[n] = FileState(md5="m", chunk_ids=[f"{n}::chunk_00"])
    return st


# ── B1: empty / unreadable state must reap NOTHING ───────────────────────────

def test_empty_state_aborts_instead_of_reaping_everything(tmp_path):
    """The whole store looks orphaned when state is empty. That must abort."""
    names = [f"note/live-{i:03d}.md" for i in range(150)]
    vault = _vault(tmp_path, names)
    store = FakeStore(_ids_for(names))
    st = _state(tmp_path, tracked=[])          # state.json tracks nothing

    plan = build_plan(store, st.state, vault)

    assert plan.aborted, "empty state must abort, not treat every row as an orphan"
    assert "0 files" in plan.aborted
    assert plan.orphans == []
    assert execute_plan(store, st.state, vault, plan) == 0
    assert store.deleted == []


def test_missing_state_file_reaps_nothing(tmp_path):
    """StateStore returns {} for a MISSING file with no exception — the reaper
    must not read that as 'everything is an orphan'."""
    names = [f"note/live-{i:03d}.md" for i in range(150)]
    vault = _vault(tmp_path, names)
    store = FakeStore(_ids_for(names))
    st = StateStore(tmp_path / "does-not-exist.json")
    st.load()

    plan = build_plan(store, st.state, vault)

    assert plan.aborted
    assert store.deleted == []


def test_tiny_vault_walk_aborts(tmp_path):
    """A partial mount returns almost nothing. An absolute floor catches that
    even when state is small enough for the ratio guard to pass."""
    vault = _vault(tmp_path, ["note/only-one.md"])
    tracked = [f"note/tracked-{i}.md" for i in range(2)]
    store = FakeStore(_ids_for([f"note/gone-{i}.md" for i in range(5)]))
    st = _state(tmp_path, tracked)

    plan = build_plan(store, st.state, vault)

    assert plan.aborted, "a vault walk this small must never authorise a reap"
    assert store.deleted == []


# ── B2: the pre-delete re-check must see files the plan cannot ───────────────

def test_apostrophe_file_restored_before_apply_is_not_reaped(tmp_path):
    """chunk ids strip apostrophes, so the re-check must compare in the SAME
    normalized key space — not stat the stripped path, which never exists."""
    live = [f"note/live-{i:03d}.md" for i in range(150)]
    apostrophe = "note/Alfred's note.md"
    vault = _vault(tmp_path, live)
    store = FakeStore(_ids_for(live) + _ids_for([apostrophe], per_file=4))
    st = _state(tmp_path, live)

    plan = build_plan(store, st.state, vault)
    assert not plan.aborted
    assert normalize_rel_path(apostrophe) in {p for p, _ in plan.orphans}, "orphan at plan time"

    # The file comes back between plan and apply — exactly the race the
    # re-check exists for.
    (vault / apostrophe).write_text("---\ntype: note\n---\nrestored\n", encoding="utf-8")
    deleted = execute_plan(store, st.state, vault, plan)

    assert deleted == 0, "a file present at apply time must never be reaped"
    assert store.deleted == []


def test_plain_file_restored_before_apply_is_not_reaped(tmp_path):
    """Control: the non-apostrophe case was already correct and must stay so."""
    live = [f"note/live-{i:03d}.md" for i in range(150)]
    gone = "note/plain-gone.md"
    vault = _vault(tmp_path, live)
    store = FakeStore(_ids_for(live) + _ids_for([gone]))
    st = _state(tmp_path, live)

    plan = build_plan(store, st.state, vault)
    (vault / gone).write_text("---\ntype: note\n---\nrestored\n", encoding="utf-8")

    assert execute_plan(store, st.state, vault, plan) == 0
    assert store.deleted == []


def test_a_genuine_orphan_is_still_reaped(tmp_path):
    """The guards must not make the reaper useless — a row whose file is gone
    AND whose state entry is gone is exactly what this exists to remove."""
    live = [f"note/live-{i:03d}.md" for i in range(150)]
    gone = "note/really-gone.md"
    vault = _vault(tmp_path, live)
    store = FakeStore(_ids_for(live) + _ids_for([gone]))
    st = _state(tmp_path, live)

    plan = build_plan(store, st.state, vault)
    deleted = execute_plan(store, st.state, vault, plan)

    assert deleted == 3
    assert all(d.startswith("note/really-gone.md::") for d in store.deleted)


def test_tracked_but_absent_file_is_not_reaped(tmp_path):
    """Still in state.json = the surveyor believes it is indexed. Not an orphan
    even if the .md is momentarily missing (sync in flight)."""
    live = [f"note/live-{i:03d}.md" for i in range(150)]
    tracked_absent = "note/tracked-but-absent.md"
    vault = _vault(tmp_path, live)
    store = FakeStore(_ids_for(live) + _ids_for([tracked_absent]))
    st = _state(tmp_path, live + [tracked_absent])

    plan = build_plan(store, st.state, vault)

    assert normalize_rel_path(tracked_absent) not in {p for p, _ in plan.orphans}
    assert execute_plan(store, st.state, vault, plan) == 0


def test_embedded_but_not_yet_state_saved_is_never_reaped(tmp_path):
    """The 25-file window from b659e9c: vectors are committed before state is
    persisted, so a just-embedded file is legitimately in the store, absent
    from state, and PRESENT in the vault. Reaping it would delete live data."""
    live = [f"note/live-{i:03d}.md" for i in range(150)]
    fresh = "note/just-embedded.md"
    vault = _vault(tmp_path, live + [fresh])
    store = FakeStore(_ids_for(live) + _ids_for([fresh]))
    st = _state(tmp_path, live)               # fresh not yet saved to state

    plan = build_plan(store, st.state, vault)

    assert normalize_rel_path(fresh) not in {p for p, _ in plan.orphans}
    assert plan.untracked_but_present >= 1, "must be counted and reported, not reaped"
    assert execute_plan(store, st.state, vault, plan) == 0
