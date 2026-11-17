"""The reaper's job is to delete rows; these tests exist to prove what it will NOT delete.

The whole module is one guard against repeating 2026-08-07. On that day the
store held 157,229 rows against 123,307 chunk_ids in state.json, and the
obvious reading of that 34k gap — "rows state.json doesn't know about are
garbage" — was wrong: 2,892 of those files / 45,352 of those rows were live
notes sitting in the surveyor's 25-file save window, still present in the
vault. A reaper built to the obvious premise deletes 29% of the index.

So the dangerous cases are tested hardest and first:
  * embedded-but-not-yet-state-saved must survive (test_*_25_file_window*)
  * apostrophes, which make rel_path unrecoverable from a chunk id and make
    BOTH naive safety checks pass on a live file
  * "::" inside a rel_path, which breaks a bare rsplit
  * an empty store and a store past the cap, which are the boring ends of the
    range where an off-by-one deletes everything or nothing
"""
from __future__ import annotations

import asyncio

import pytest

from alfred.core.models import FileState
from alfred.core.vault import _safe_chunk_id
from alfred.daemons.janitor import JanitorDaemon
from alfred.store.reaper import (
    CHUNK_ID_RE,
    build_plan,
    execute_plan,
    normalize_rel_path,
    scan_store_paths,
)
from alfred.store.state import StateStore


class _FakeStore:
    """In-memory stand-in for LanceDBStore's id column.

    Only ``iter_ids`` and ``delete_ids`` are used by the reaper, and both are
    modelled faithfully — including delete batching, so a test can assert the
    predicate never grows past the configured term count.
    """

    def __init__(self, ids: list[str]):
        self.ids = list(ids)
        self.delete_batches: list[list[str]] = []

    def iter_ids(self, batch_size: int = 4096):
        # Yield in chunks so a caller depending on eager materialisation fails.
        for i in range(0, len(self.ids), max(1, batch_size)):
            yield from self.ids[i:i + batch_size]

    def delete_ids(self, chunk_ids, batch: int = 500):
        chunk_ids = list(chunk_ids)
        for i in range(0, len(chunk_ids), batch):
            group = chunk_ids[i:i + batch]
            self.delete_batches.append(group)
            for cid in group:
                if cid in self.ids:
                    self.ids.remove(cid)
        return len(chunk_ids)

    def count(self) -> int:
        return len(self.ids)


class _Cfg:
    janitor_reap_enabled = True
    janitor_reap_max_rows_per_sweep = 5000
    janitor_reap_scan_batch_size = 4096
    janitor_reap_delete_batch = 500
    vault_path = None


def _default_cfg(tmp_path):
    """A real AlfredConfig with nothing but the two required paths set."""
    from alfred.config import AlfredConfig
    return AlfredConfig(vault_path=tmp_path / "vault", data_dir=tmp_path / "data")


def _ids_for(rel_path: str, n: int) -> list[str]:
    """Mint ids exactly the way the embed path does (lossy apostrophe strip)."""
    return [_safe_chunk_id(rel_path, i) for i in range(n)]


def _vault(tmp_path, *rel_paths: str):
    vault = tmp_path / "vault"
    for rp in rel_paths:
        f = vault / rp
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("# note\n")
    vault.mkdir(parents=True, exist_ok=True)
    return vault


def _state(tmp_path, tracked: dict[str, list[str]]):
    store = StateStore(tmp_path / "state.json")
    store.load()
    for rel_path, chunk_ids in tracked.items():
        store.state.files[rel_path] = FileState(
            md5="x", last_embedded="2026-08-01T00:00:00+00:00", chunk_ids=list(chunk_ids)
        )
    return store


# ── The 25-file window: the case that must never regress ────────────────────


def test_embedded_but_not_yet_state_saved_is_never_reaped(tmp_path):
    """The incident case. Rows exist, state.json has no entry, note is present.

    The surveyor persists state every 25 files, so this is the *normal*
    steady state for anything embedded since the last save — not an anomaly.
    """
    vault = _vault(tmp_path, "note/fresh.md", "note/tracked.md")
    store = _FakeStore(_ids_for("note/fresh.md", 12) + _ids_for("note/tracked.md", 3))
    state = _state(tmp_path, {"note/tracked.md": _ids_for("note/tracked.md", 3)})

    plan = build_plan(store, state.state, vault, max_rows=5000)

    assert plan.orphans == []
    assert plan.untracked_but_present == 1
    assert plan.untracked_but_present_rows == 12
    assert execute_plan(store, state.state, vault, plan) == 0
    assert len(store.ids) == 15


def test_window_file_survives_even_if_state_is_completely_empty(tmp_path):
    """A crash before the first save leaves state.json empty. Everything present
    in the vault must still survive — the vault, not state, is the authority
    on whether a note exists."""
    vault = _vault(tmp_path, "note/a.md", "note/b.md")
    store = _FakeStore(_ids_for("note/a.md", 5) + _ids_for("note/b.md", 5))
    state = _state(tmp_path, {"note/_nonempty.md": ["note/_nonempty.md::chunk_00"]})

    plan = build_plan(store, state.state, vault, max_rows=5000)

    assert plan.orphans == []
    assert plan.untracked_but_present_rows == 10


def test_path_that_becomes_tracked_between_plan_and_apply_is_skipped(tmp_path):
    """Plan and apply are seconds apart and the surveyor writes state in
    between. Apply re-checks; it does not trust the plan."""
    vault = _vault(tmp_path, "note/keep.md")
    store = _FakeStore(_ids_for("note/gone.md", 4))
    state = _state(tmp_path, {"note/_nonempty.md": ["note/_nonempty.md::chunk_00"]})

    plan = build_plan(store, state.state, vault, max_rows=5000)
    assert plan.reapable_rows == 4

    # Surveyor finishes its batch and records the file.
    state.state.files["note/gone.md"] = FileState(
        md5="x", last_embedded="", chunk_ids=_ids_for("note/gone.md", 4)
    )

    assert execute_plan(store, state.state, vault, plan) == 0
    assert len(store.ids) == 4


def test_path_that_reappears_in_the_vault_between_plan_and_apply_is_skipped(tmp_path):
    vault = _vault(tmp_path, "note/keep.md")
    store = _FakeStore(_ids_for("note/gone.md", 4))
    state = _state(tmp_path, {"note/_nonempty.md": ["note/_nonempty.md::chunk_00"]})
    plan = build_plan(store, state.state, vault, max_rows=5000)
    assert plan.reapable_rows == 4

    (vault / "note" / "gone.md").write_text("# restored from sync\n")

    assert execute_plan(store, state.state, vault, plan) == 0


# ── Apostrophes: rel_path is not recoverable from an id ─────────────────────


def test_apostrophe_file_present_in_vault_is_never_reaped(tmp_path):
    """core/vault.py::_safe_chunk_id strips ' when minting ids, so the stored
    path is 'Alfreds…' while the real file is 'Alfred's…'. Un-normalised, the
    file looks absent from state AND absent from the vault — both safety
    checks pass and a live note gets deleted. 57 vault files are exposed."""
    real = "synthesis/alfred's-assistant-tasks.md"
    vault = _vault(tmp_path, real)
    store = _FakeStore(_ids_for(real, 6))
    state = _state(tmp_path, {real: _ids_for(real, 6)})

    # Precondition: the id really is lossy, or this test proves nothing.
    assert "'" not in store.ids[0]
    assert store.ids[0].startswith("synthesis/alfreds-assistant-tasks.md::")

    plan = build_plan(store, state.state, vault, max_rows=5000)
    assert plan.orphans == []
    assert plan.untracked_but_present == 0  # matched via state, not the vault


def test_apostrophe_file_present_in_vault_but_absent_from_state_is_never_reaped(tmp_path):
    """Same file, now also inside the 25-file window. The vault check has to
    normalise too, or this is the exact double-failure that deletes it."""
    real = "synthesis/ben's-primary-barrier.md"
    vault = _vault(tmp_path, real)
    store = _FakeStore(_ids_for(real, 6))
    state = _state(tmp_path, {"note/_nonempty.md": ["note/_nonempty.md::chunk_00"]})

    plan = build_plan(store, state.state, vault, max_rows=5000)
    assert plan.orphans == []
    assert plan.untracked_but_present_rows == 6


def test_apostrophe_file_genuinely_deleted_is_reaped(tmp_path):
    """The flip side: normalisation must not make apostrophe files unreapable
    forever, or the safety fix becomes a leak."""
    vault = _vault(tmp_path, "note/other.md")
    store = _FakeStore(_ids_for("synthesis/ben's-gone.md", 3))
    state = _state(tmp_path, {"note/_nonempty.md": ["note/_nonempty.md::chunk_00"]})

    plan = build_plan(store, state.state, vault, max_rows=5000)
    assert plan.reapable_rows == 3
    assert execute_plan(store, state.state, vault, plan) == 3
    assert store.ids == []


def test_normalize_is_idempotent_and_collapses_only_apostrophes():
    assert normalize_rel_path("a'b.md") == "ab.md"
    assert normalize_rel_path(normalize_rel_path("a'b.md")) == "ab.md"
    assert normalize_rel_path("a-b.md") == "a-b.md"


# ── "::" inside a rel_path ──────────────────────────────────────────────────


def test_rel_path_containing_double_colon_parses_on_the_last_marker(tmp_path):
    """A bare rsplit on '::' is fine, but a *left*-anchored or non-greedy match
    is not: 'note/a::b.md::chunk_00' must yield 'note/a::b.md', not 'note/a'.
    Getting this wrong invents a rel_path that exists nowhere and therefore
    passes both safety checks."""
    weird = "note/a::b.md"
    m = CHUNK_ID_RE.match(_safe_chunk_id(weird, 0))
    assert m is not None
    assert m.group("path") == weird

    vault = _vault(tmp_path, weird, "note/other.md")
    store = _FakeStore(_ids_for(weird, 4))
    state = _state(tmp_path, {"note/_nonempty.md": ["note/_nonempty.md::chunk_00"]})

    plan = build_plan(store, state.state, vault, max_rows=5000)
    assert plan.orphans == []  # the file is present in the vault
    assert plan.untracked_but_present_rows == 4


def test_double_colon_path_genuinely_missing_is_reaped_whole(tmp_path):
    vault = _vault(tmp_path, "note/other.md")
    store = _FakeStore(_ids_for("note/a::b.md", 4))
    state = _state(tmp_path, {"note/_nonempty.md": ["note/_nonempty.md::chunk_00"]})

    plan = build_plan(store, state.state, vault, max_rows=5000)
    assert plan.orphans == [("note/a::b.md", 4)]
    assert execute_plan(store, state.state, vault, plan) == 4


def test_malformed_ids_are_reported_and_never_reaped(tmp_path):
    """An id that does not match '<path>::chunk_NN' has unknown provenance.
    Guessing a rel_path for it would hand it to the delete path with both
    safety checks passing on a fabricated name."""
    vault = _vault(tmp_path, "note/other.md")
    store = _FakeStore(["garbage", "note/x.md::chunkNN", "note/x.md::chunk_"])
    state = _state(tmp_path, {"note/_nonempty.md": ["note/_nonempty.md::chunk_00"]})

    counts, samples, n = scan_store_paths(store)
    assert counts == {}
    assert n == 3
    assert samples == ["garbage", "note/x.md::chunkNN", "note/x.md::chunk_"]

    plan = build_plan(store, state.state, vault, max_rows=5000)
    assert plan.orphans == []
    assert plan.malformed_count == 3
    assert execute_plan(store, state.state, vault, plan) == 0
    assert len(store.ids) == 3


# ── Boring ends of the range ────────────────────────────────────────────────


def test_empty_store_reaps_nothing(tmp_path):
    vault = _vault(tmp_path, "note/a.md")
    store = _FakeStore([])
    state = _state(tmp_path, {"note/a.md": _ids_for("note/a.md", 2)})

    plan = build_plan(store, state.state, vault, max_rows=5000)
    assert plan.store_rows == 0
    assert plan.orphans == []
    assert execute_plan(store, state.state, vault, plan) == 0


def test_store_larger_than_cap_defers_the_remainder(tmp_path):
    """The cap is blast radius. Whatever it cannot take this sweep must be
    reported, not silently dropped — a cap that reads as 'that was all there
    was' is how a backlog stays invisible."""
    vault = _vault(tmp_path, "note/keep.md")
    ids = []
    for i in range(10):
        ids += _ids_for(f"note/gone{i}.md", 10)
    store = _FakeStore(ids)
    state = _state(tmp_path, {"note/_nonempty.md": ["note/_nonempty.md::chunk_00"]})

    plan = build_plan(store, state.state, vault, max_rows=35)
    assert plan.reapable_rows <= 35
    assert plan.reapable_paths == 3
    assert len(plan.deferred) == 7
    deleted = execute_plan(store, state.state, vault, plan)
    assert deleted == 30
    assert len(store.ids) == 70

    # Idempotent: a second sweep picks up where this one stopped.
    plan2 = build_plan(store, state.state, vault, max_rows=35)
    assert plan2.reapable_rows == 30


def test_single_path_bigger_than_the_cap_is_deferred_not_truncated(tmp_path):
    """Partially reaping one file would leave a half-deleted record that no
    consistency check names. Conservative choice: defer it and say so."""
    vault = _vault(tmp_path, "note/keep.md")
    store = _FakeStore(_ids_for("note/huge.md", 4000))
    state = _state(tmp_path, {"note/_nonempty.md": ["note/_nonempty.md::chunk_00"]})

    plan = build_plan(store, state.state, vault, max_rows=500)
    assert plan.orphans == []
    assert plan.deferred == [("note/huge.md", 4000)]
    assert execute_plan(store, state.state, vault, plan) == 0


def test_delete_predicate_is_batched(tmp_path):
    """The 2026-08-07 OOM was one ~3,700-term IN clause planned against 10,274
    fragments. Term count per call is the thing that must stay bounded."""
    vault = _vault(tmp_path, "note/keep.md")
    store = _FakeStore(_ids_for("note/gone.md", 1200))
    state = _state(tmp_path, {"note/_nonempty.md": ["note/_nonempty.md::chunk_00"]})

    plan = build_plan(store, state.state, vault, max_rows=5000)
    execute_plan(store, state.state, vault, plan, delete_batch=500)

    assert [len(b) for b in store.delete_batches] == [500, 500, 200]


def test_scan_does_not_materialise_the_id_column(tmp_path):
    """Peak memory must not track row count. Proxy assertion: the scan holds
    one dict keyed by distinct path, and iter_ids is consumed lazily."""
    consumed = []

    class _Lazy(_FakeStore):
        def iter_ids(self, batch_size: int = 4096):
            for cid in self.ids:
                consumed.append(cid)
                yield cid

    store = _Lazy([f"note/n{i % 3}.md::chunk_{i:02d}" for i in range(300)])
    counts, _, _ = scan_store_paths(store)
    assert set(counts) == {"note/n0.md", "note/n1.md", "note/n2.md"}
    assert sum(counts.values()) == 300
    assert len(consumed) == 300


# ── Abort guards ────────────────────────────────────────────────────────────


def test_missing_vault_aborts_without_deleting(tmp_path):
    """An unmounted /mnt/external turns 'absent from the vault' into 'delete
    everything'. This is the failure mode that ends the index."""
    store = _FakeStore(_ids_for("note/a.md", 5))
    state = _state(tmp_path, {"note/_nonempty.md": ["note/_nonempty.md::chunk_00"]})

    plan = build_plan(store, state.state, tmp_path / "not-there", max_rows=5000)
    assert plan.aborted
    assert plan.orphans == []
    assert execute_plan(store, state.state, tmp_path / "not-there", plan) == 0
    assert len(store.ids) == 5


def test_empty_vault_walk_aborts(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    store = _FakeStore(_ids_for("note/a.md", 5))
    state = _state(tmp_path, {"note/_nonempty.md": ["note/_nonempty.md::chunk_00"]})

    plan = build_plan(store, state.state, vault, max_rows=5000)
    assert "0 .md" in (plan.aborted or "")
    assert len(store.ids) == 5


def test_vault_much_smaller_than_state_aborts(tmp_path):
    """A partial mount shows *some* files. Alfred only embeds what it found in
    the vault, so a vault holding far fewer notes than state tracks means the
    walk is lying, not that the notes were deleted."""
    vault = _vault(tmp_path, "note/a.md")
    tracked = {f"note/n{i}.md": _ids_for(f"note/n{i}.md", 1) for i in range(20)}
    store = _FakeStore(_ids_for("note/gone.md", 3))
    state = _state(tmp_path, tracked)

    plan = build_plan(store, state.state, vault, max_rows=5000)
    assert "refusing" in (plan.aborted or "")
    assert len(store.ids) == 3


# ── Janitor wiring ──────────────────────────────────────────────────────────


def _janitor(tmp_path, store, state, **overrides):
    cfg = _Cfg()
    cfg.vault_path = tmp_path / "vault"
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return JanitorDaemon(cfg, state, asyncio.Queue(), store=store)


def test_reap_sweep_is_off_by_default(tmp_path):
    """Default-off is the contract: this deletes rows state.json does not even
    track, so no other consistency check would notice a bug here."""
    assert _default_cfg(tmp_path).janitor_reap_enabled is False

    _vault(tmp_path, "note/keep.md")
    store = _FakeStore(_ids_for("note/gone.md", 3))
    state = _state(tmp_path, {"note/_nonempty.md": ["note/_nonempty.md::chunk_00"]})
    j = _janitor(tmp_path, store, state, janitor_reap_enabled=False)

    asyncio.run(j.reap_tick())
    assert len(store.ids) == 3


def test_reap_sweep_deletes_only_true_orphans(tmp_path):
    _vault(tmp_path, "note/present.md")
    store = _FakeStore(_ids_for("note/present.md", 4) + _ids_for("note/gone.md", 3))
    state = _state(tmp_path, {"note/_nonempty.md": ["note/_nonempty.md::chunk_00"]})
    j = _janitor(tmp_path, store, state)

    asyncio.run(j.reap_tick())

    assert len(store.ids) == 4
    assert all(i.startswith("note/present.md::") for i in store.ids)


def test_reap_plan_matches_what_the_sweep_does(tmp_path):
    """Dry run and sweep must call the same function, the way forget does —
    a preview that can disagree with the action is worse than no preview."""
    _vault(tmp_path, "note/present.md")
    store = _FakeStore(_ids_for("note/present.md", 4) + _ids_for("note/gone.md", 3))
    state = _state(tmp_path, {"note/_nonempty.md": ["note/_nonempty.md::chunk_00"]})
    j = _janitor(tmp_path, store, state)

    plan = j.reap_plan()
    assert plan.orphans == [("note/gone.md", 3)]
    assert len(store.ids) == 7  # planning changed nothing

    asyncio.run(j.reap_tick())
    assert len(store.ids) == 4


def test_reap_tick_swallows_store_errors_but_counts_them(tmp_path):
    from alfred.core.failures import peek_failures

    _vault(tmp_path, "note/keep.md")

    class _Boom(_FakeStore):
        def delete_ids(self, chunk_ids, batch: int = 500):
            raise RuntimeError("vector store unavailable")

    store = _Boom(_ids_for("note/gone.md", 3))
    state = _state(tmp_path, {"note/_nonempty.md": ["note/_nonempty.md::chunk_00"]})
    j = _janitor(tmp_path, store, state)

    asyncio.run(j.reap_tick())  # must not raise
    assert (peek_failures().get("reap.vector_delete_failed", 0)
            + state.state.error_counts.get("reap.vector_delete_failed", 0)) >= 1


def test_reap_max_rows_override_is_honoured(tmp_path):
    _vault(tmp_path, "note/keep.md")
    ids = []
    for i in range(5):
        ids += _ids_for(f"note/gone{i}.md", 10)
    store = _FakeStore(ids)
    state = _state(tmp_path, {"note/_nonempty.md": ["note/_nonempty.md::chunk_00"]})
    j = _janitor(tmp_path, store, state)

    assert j.reap_plan(max_rows=20).reapable_rows == 20
    assert j.reap_plan().reapable_rows == 50


def test_config_yaml_keys_reach_the_dataclass(tmp_path):
    """A janitor_* key that never leaves the YAML is dead config — the exact
    class of bug tests/test_config_consumed.py exists to catch."""
    import yaml
    from alfred.config import AlfredConfig

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "vault": {"path": str(tmp_path / "vault")},
        "data_dir": str(tmp_path / "data"),
        "janitor": {
            "reap_enabled": True,
            "reap_max_rows_per_sweep": 77,
            "reap_scan_batch_size": 128,
            "reap_delete_batch": 42,
        },
    }))
    cfg = AlfredConfig.load(path)
    assert cfg.janitor_reap_enabled is True
    assert cfg.janitor_reap_max_rows_per_sweep == 77
    assert cfg.janitor_reap_scan_batch_size == 128
    assert cfg.janitor_reap_delete_batch == 42


# ── CLI, end to end against a real LanceDB table in tmp_path ────────────────


def _cli_fixture(tmp_path):
    """A real (tiny) LanceDB store + config, entirely inside tmp_path."""
    import yaml
    from alfred.store.lancedb_store import LanceDBStore

    vault = _vault(tmp_path, "note/present.md")
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)

    store = LanceDBStore(uri=str(data / "lancedb"), collection="vault_v2", dims=8)
    rows = []
    for rel, n in (("note/present.md", 3), ("note/gone.md", 2)):
        for i in range(n):
            rows.append({
                "chunk_id": _safe_chunk_id(rel, i), "dense": [0.1] * 8,
                "record_type": "note", "name": rel, "chunk_index": i,
            })
    store.upsert_many(rows)

    from alfred.core.models import FileState
    from alfred.store.state import StateStore
    _st = StateStore(data / "state.json")
    _st.load()
    _st.state.files["note/present.md"] = FileState(
        md5="m", chunk_ids=[_safe_chunk_id("note/present.md", i) for i in range(3)]
    )
    _st.save()

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "vault": {"path": str(vault)},
        "data_dir": str(data),
        "surveyor": {"embed_dims": 8},
        "vector_store": "lancedb",
    }))
    return cfg_path, store


def test_cli_reap_dry_run_changes_nothing(tmp_path):
    """Dry-run-by-default is the whole safety posture of this command."""
    from typer.testing import CliRunner
    from alfred.cli import app

    cfg_path, store = _cli_fixture(tmp_path)
    res = CliRunner().invoke(app, ["reap", "-c", str(cfg_path)])
    assert res.exit_code == 0, res.output
    assert "note/gone.md" in res.output
    assert "Dry run" in res.output
    assert store.count() == 5


def test_cli_reap_apply_deletes_only_the_orphan(tmp_path):
    from typer.testing import CliRunner
    from alfred.cli import app

    cfg_path, store = _cli_fixture(tmp_path)
    res = CliRunner().invoke(app, ["reap", "-c", str(cfg_path), "--apply"])
    assert res.exit_code == 0, res.output
    assert "reaped" in res.output

    # Re-open rather than reusing the fixture handle: a LanceDB table object
    # is pinned to the version it was opened at, so an existing handle keeps
    # reporting pre-delete rows. Same reason the daemons must not be asked to
    # confirm a reap they did not perform themselves.
    from alfred.store.lancedb_store import LanceDBStore
    store = LanceDBStore(
        uri=str(tmp_path / "data" / "lancedb"), collection="vault_v2", dims=8
    )
    remaining = sorted(store.iter_ids())
    assert remaining == sorted(_ids_for("note/present.md", 3))


def test_cli_reap_json_is_machine_readable(tmp_path):
    import json as _json
    from typer.testing import CliRunner
    from alfred.cli import app

    cfg_path, store = _cli_fixture(tmp_path)
    res = CliRunner().invoke(app, ["reap", "-c", str(cfg_path), "--json"])
    assert res.exit_code == 0, res.output
    payload = _json.loads(res.output)
    assert payload["applied"] is False
    assert payload["reapable_rows"] == 2
    assert payload["rows"] == [{"rel_path": "note/gone.md", "chunks": 2}]
    assert store.count() == 5


@pytest.mark.parametrize("cid,expected", [
    ("note/a.md::chunk_00", "note/a.md"),
    ("a/b/c.md::chunk_99", "a/b/c.md"),
    ("note/a::b.md::chunk_07", "note/a::b.md"),
])
def test_chunk_id_regex_round_trips(cid, expected):
    assert CHUNK_ID_RE.match(cid).group("path") == expected
