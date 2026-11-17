"""Tests for LanceDBStore's corruption-vs-transient classification and the
quarantine circuit breaker.

``LanceDBStore.__init__`` auto-heals a corrupt table by quarantining it aside
and recreating an empty one, but only when the failure to open genuinely
looks like on-disk corruption (``_looks_like_corruption``).  A transient or
operational failure (permissions, disk full, a schema mismatch) must re-raise
untouched instead of nuking a table that isn't actually broken.  And even a
genuine corruption must stop self-healing once ``_MAX_QUARANTINES_PER_DAY``
has been hit within a rolling 24h window, since silently recreating forever
would shred the store on a persistent fault (bad disk, a Lance bug).

This file covers the four behaviors not exercised by
``test_lancedb_quarantine_lock.py`` (which is about cross-process races, not
classification or the breaker):

1. A transient exception does NOT trigger quarantine.
2. A genuine corruption exception DOES trigger quarantine.
3. The quarantine dir is named as expected (``.quarantine-corrupt-<stamp>``).
4. The circuit breaker trips after ``_MAX_QUARANTINES_PER_DAY`` quarantines in
   a day and stops attempting recreation.
"""
from __future__ import annotations

import re
from pathlib import Path

import lancedb
import pytest

from alfred.store.lancedb_store import (
    _MAX_QUARANTINES_PER_DAY,
    _looks_like_corruption,
    LanceDBStore,
)

_QUARANTINE_DIR_RE = re.compile(r"^\.quarantine-corrupt-\d{8}-\d{6}(-\d+)?$")


def _make_corrupted_table(tmp_path: Path, collection: str, dims: int = 4) -> str:
    """Build a real LanceDB table, then truncate its latest manifest to
    reproduce the exact zero-byte-manifest corruption this module heals.

    Mirrors the fixture in ``test_lancedb_quarantine_lock.py``.
    """
    uri = str(tmp_path)
    seed = LanceDBStore(uri, collection=collection, dims=dims)
    seed.upsert("a", [0.1, 0.2, 0.3, 0.4][:dims], {}, "note", "a.md", 0)
    assert seed.was_recreated is False

    tbl_dir = tmp_path / f"{collection}.lance"
    manifests = sorted((tbl_dir / "_versions").glob("*.manifest"))
    latest = max(manifests, key=lambda p: p.stat().st_mtime)
    latest.write_bytes(b"")

    # Sanity: the corruption actually reproduces and trips the same
    # marker-based detection LanceDBStore uses.
    db = lancedb.connect(uri)
    with pytest.raises(Exception) as excinfo:
        db.open_table(collection)
    assert _looks_like_corruption(excinfo.value)

    return uri


class _FlakyDB:
    """Wraps a real LanceDB connection but raises a caller-supplied exception
    on the first ``open_table`` call, delegating everything else (and
    subsequent calls) to the real connection.  Used to simulate a transient/
    operational failure that is not on-disk corruption.
    """

    def __init__(self, real_db, exc: Exception) -> None:
        self._real = real_db
        self._exc = exc
        self._raised = False

    def table_names(self):
        return self._real.table_names()

    def open_table(self, name):
        if not self._raised:
            self._raised = True
            raise self._exc
        return self._real.open_table(name)

    def create_table(self, *args, **kwargs):
        return self._real.create_table(*args, **kwargs)


def test_transient_exception_does_not_trigger_quarantine(tmp_path, monkeypatch):
    """A non-corruption failure to open (permissions, disk full, ...) must
    re-raise untouched — no move, no recreate, table dir left exactly as-is.
    """
    uri = str(tmp_path)
    seed = LanceDBStore(uri, collection="transient_tbl", dims=4)
    seed.upsert("a", [0.1, 0.2, 0.3, 0.4], {}, "note", "a.md", 0)
    assert seed.was_recreated is False

    table_dir = tmp_path / "transient_tbl.lance"
    assert table_dir.is_dir()

    transient_exc = RuntimeError("Permission denied: disk quota exceeded")
    assert not _looks_like_corruption(transient_exc)

    real_connect = lancedb.connect

    def fake_connect(u):
        return _FlakyDB(real_connect(u), transient_exc)

    monkeypatch.setattr(lancedb, "connect", fake_connect)

    with pytest.raises(RuntimeError, match="Permission denied"):
        LanceDBStore(uri, collection="transient_tbl", dims=4)

    # No quarantine happened: the original table dir is untouched and no
    # quarantine dir was created.
    assert table_dir.is_dir()
    assert not list(tmp_path.glob(".quarantine-corrupt-*"))


def test_genuine_corruption_triggers_quarantine(tmp_path):
    """A real corrupt-manifest failure must be auto-healed: quarantined aside
    and recreated empty, with ``was_recreated`` set so callers know to
    re-embed.
    """
    uri = _make_corrupted_table(tmp_path, collection="corrupt_tbl")

    store = LanceDBStore(uri, collection="corrupt_tbl", dims=4)

    assert store.was_recreated is True
    assert store.count() == 0

    quarantine_dirs = [p for p in Path(uri).glob(".quarantine-corrupt-*") if p.is_dir()]
    assert len(quarantine_dirs) == 1


def test_quarantine_dir_is_named_as_expected(tmp_path):
    """The quarantine destination must match the documented
    ``.quarantine-corrupt-<YYYYMMDD-HHMMSS>[-N]`` naming scheme."""
    uri = _make_corrupted_table(tmp_path, collection="named_tbl")

    LanceDBStore(uri, collection="named_tbl", dims=4)

    quarantine_dirs = [p for p in Path(uri).glob(".quarantine-corrupt-*") if p.is_dir()]
    assert len(quarantine_dirs) == 1
    assert _QUARANTINE_DIR_RE.match(quarantine_dirs[0].name), quarantine_dirs[0].name


def test_circuit_breaker_trips_after_max_quarantines_per_day(tmp_path):
    """Once ``_MAX_QUARANTINES_PER_DAY`` quarantines have happened within the
    last 24h, a further genuine corruption must re-raise instead of
    quarantining again — the breaker stops attempting recreation so a
    persistent fault doesn't shred the store on every restart.
    """
    uri = _make_corrupted_table(tmp_path, collection="breaker_tbl")
    base = Path(uri)

    # Pre-seed the breaker window with MAX fake quarantine dirs, all recent
    # (fresh mkdir gives them a current mtime, well inside the 24h cutoff).
    for i in range(_MAX_QUARANTINES_PER_DAY):
        (base / f".quarantine-corrupt-fake-{i}").mkdir()

    table_dir = base / "breaker_tbl.lance"
    assert table_dir.is_dir()

    with pytest.raises(Exception) as excinfo:
        LanceDBStore(uri, collection="breaker_tbl", dims=4)

    # It's the original corruption error re-raised, not something new.
    assert _looks_like_corruption(excinfo.value)

    # No new quarantine happened: dir count unchanged and the real corrupt
    # table dir was never moved.
    quarantine_dirs = [p for p in base.glob(".quarantine-corrupt-*") if p.is_dir()]
    assert len(quarantine_dirs) == _MAX_QUARANTINES_PER_DAY
    assert table_dir.is_dir()
