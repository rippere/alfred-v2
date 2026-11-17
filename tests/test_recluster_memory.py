"""The periodic recluster must not scale memory with chunks or with files².

Regression for 2026-11-16: ``SurveyorDaemon._recluster`` loaded every chunk's
vector via ``query_all`` as Python lists of Python floats (~2 GB at 83k rows,
~25 GB at the 1.07M rows the store once reached) and then ran HDBSCAN with
``metric="cosine"``, which sklearn can only do by brute force — an n x n
distance matrix over every file. alfred.service sat throttled at its 6 GB
MemoryHigh for as long as a recluster ran.

The fix loads one representative vector per file straight into a float32
matrix (``file_vectors``) and clusters unit-normalised vectors with Euclidean
distance, which orders pairs exactly like cosine and lets HDBSCAN use a tree.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from alfred.config import AlfredConfig
from alfred.core.models import ClusterState, FileState
from alfred.daemons.surveyor import SurveyorDaemon, cluster_labels
from alfred.store.lancedb_store import LanceDBStore
from alfred.store.types import representative_chunks

DIMS = 4


def _row(path: str, idx: int, vec) -> dict:
    return {"chunk_id": f"{path}::chunk_{idx:02d}", "dense": list(vec),
            "record_type": "note", "name": path, "chunk_index": idx}


# ── representative_chunks ─────────────────────────────────────────────────────

def test_representative_is_lowest_chunk_index_with_null_last():
    best = representative_chunks([
        ("a.md::chunk_03", 3),
        ("a.md::chunk_00", 0),
        ("a.md::chunk_01", 1),
        ("b.md::chunk_05", 5),
        ("b.md::chunk_02", 2),
        ("c.md::x", None),
        ("c.md::chunk_07", 7),
        ("d.md::only", None),
    ])
    assert best["a.md"] == (0, "a.md::chunk_00")
    assert best["b.md"] == (2, "b.md::chunk_02")
    assert best["c.md"][1] == "c.md::chunk_07"
    assert best["d.md"][1] == "d.md::only"


def test_representative_splits_on_last_separator_only():
    best = representative_chunks([("dir/odd::name.md::chunk_00", 0)])
    assert list(best) == ["dir/odd::name.md"]


# ── LanceDBStore.file_vectors ─────────────────────────────────────────────────

def test_file_vectors_empty_store(tmp_path):
    store = LanceDBStore(str(tmp_path / "db"), dims=DIMS)
    paths, vecs = store.file_vectors()
    assert paths == []
    assert vecs.shape == (0, DIMS) and vecs.dtype == np.float32


def test_file_vectors_one_row_per_file_prefers_chunk_zero(tmp_path):
    store = LanceDBStore(str(tmp_path / "db"), dims=DIMS)
    # Separate upserts -> separate fragments, and chunk 0 deliberately not
    # first in storage order (the old "first-seen" rule picked chunk 2 here).
    store.upsert_many([_row("b.md", 2, [9, 9, 9, 9])])
    store.upsert_many([_row("b.md", 0, [2, 0, 0, 0]), _row("b.md", 1, [8, 8, 8, 8])])
    store.upsert_many([_row("a.md", 0, [1, 0, 0, 0]), _row("a.md", 1, [7, 7, 7, 7])])
    # No chunk 0 at all: falls back to its lowest chunk_index.
    store.upsert_many([_row("c.md", 4, [6, 6, 6, 6]), _row("c.md", 3, [0, 0, 3, 0])])

    paths, vecs = store.file_vectors(batch_size=1)

    assert paths == ["a.md", "b.md", "c.md"]
    assert vecs.dtype == np.float32 and vecs.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(
        vecs, np.array([[1, 0, 0, 0], [2, 0, 0, 0], [0, 0, 3, 0]], dtype=np.float32)
    )


def test_file_vectors_matches_query_all_chunk_zero(tmp_path):
    """Same vectors as the query_all path would give for each file's chunk 0."""
    rng = np.random.default_rng(0)
    store = LanceDBStore(str(tmp_path / "db"), dims=DIMS)
    for f in range(30):
        store.upsert_many([_row(f"n/{f}.md", i, rng.normal(size=DIMS)) for i in range(3)])

    paths, vecs = store.file_vectors(batch_size=7)
    expected = {
        r["id"].rsplit("::", 1)[0]: r["embedding"]
        for r in store.query_all(output_fields=["id", "embedding"])
        if r["id"].endswith("::chunk_00")
    }
    assert paths == sorted(expected)
    np.testing.assert_allclose(vecs, np.array([expected[p] for p in paths]), rtol=0)


# ── cluster_labels ────────────────────────────────────────────────────────────

def _two_directions(n_per: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.array([1.0, 0, 0, 0]) + rng.normal(scale=0.02, size=(n_per, DIMS))
    b = np.array([0, 0, 1.0, 0]) + rng.normal(scale=0.02, size=(n_per, DIMS))
    # Wildly different magnitudes within each direction: Euclidean on the raw
    # vectors finds no structure at all (everything is noise); cosine, via the
    # unit normalisation, must see exactly two groups.
    scale = np.geomspace(0.01, 100.0, n_per)[:, None]
    return np.vstack([a * scale, b * scale]).astype(np.float32)


def test_cluster_labels_groups_by_direction_not_magnitude():
    x = _two_directions()
    labels = cluster_labels(x, min_cluster_size=3, min_samples=2)
    a, b = labels[:12], labels[12:]
    assert (a != -1).all() and (b != -1).all()
    assert len(set(a)) == 1 and len(set(b)) == 1 and a[0] != b[0]


def test_cluster_labels_does_not_mutate_input_and_marks_degenerate_rows_noise():
    x = np.vstack([_two_directions(), np.zeros((1, DIMS)), np.full((1, DIMS), np.nan)])
    x = x.astype(np.float32)
    before = x.copy()
    labels = cluster_labels(x, min_cluster_size=3, min_samples=2)
    np.testing.assert_array_equal(np.isnan(x), np.isnan(before))
    np.testing.assert_array_equal(np.nan_to_num(x), np.nan_to_num(before))
    assert labels[-2] == -1 and labels[-1] == -1
    assert len(labels) == len(x)


def test_cluster_labels_never_builds_a_brute_force_distance_matrix(monkeypatch):
    """The fix is only a fix if HDBSCAN takes the tree path."""
    import sklearn.cluster._hdbscan.hdbscan as hdb

    def _boom(*a, **kw):
        raise AssertionError("HDBSCAN fell back to the n x n brute-force path")

    monkeypatch.setattr(hdb, "_hdbscan_brute", _boom)
    labels = cluster_labels(_two_directions(), min_cluster_size=3, min_samples=2)
    assert len(set(labels) - {-1}) == 2


def test_cluster_labels_isolated_matches_in_process():
    """The subprocess path (which keeps the MST off the daemon's GIL) must
    return exactly what the in-process call does."""
    from alfred.daemons.surveyor import cluster_labels_isolated

    x = _two_directions(seed=3)
    np.testing.assert_array_equal(
        cluster_labels_isolated(x, 3, 2), cluster_labels(x, 3, 2)
    )


def test_cluster_labels_too_few_rows_is_all_noise():
    labels = cluster_labels(np.ones((1, DIMS), dtype=np.float32), 2, 1)
    assert labels.tolist() == [-1]


# ── Surveyor wiring ───────────────────────────────────────────────────────────

class _VectorStore:
    def __init__(self, paths, vecs):
        self._paths, self._vecs = paths, vecs

    def file_vectors(self):
        return list(self._paths), self._vecs

    def query_all(self, *a, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("recluster must not load every chunk via query_all")


def _daemon(tmp_path: Path, store) -> SurveyorDaemon:
    from alfred.store.state import StateStore

    vault = tmp_path / "vault"
    vault.mkdir()
    cfg = AlfredConfig(vault_path=vault, data_dir=tmp_path / "data")
    cfg.hdbscan_min_cluster_size = 3
    cfg.hdbscan_min_samples = 2
    state = StateStore(tmp_path / "state.json")
    state.load()
    return SurveyorDaemon(cfg, state, asyncio.Queue(), store=store)


def test_recluster_uses_file_vectors_and_writes_cluster_state(tmp_path):
    x = _two_directions()
    paths = [f"a{i}.md" for i in range(12)] + [f"b{i}.md" for i in range(12)]
    d = _daemon(tmp_path, _VectorStore(paths, x))
    st = d.state.state
    for p in paths:
        st.files[p] = FileState(md5="x")
    st.clusters["semantic_0"] = ClusterState(cluster_id=0, label=["kept"], last_labeled="t")

    asyncio.run(d._recluster())

    groups = {}
    for p in paths:
        groups.setdefault(p[0], set()).add(st.files[p].semantic_cluster_id)
    assert len(groups["a"]) == 1 and len(groups["b"]) == 1
    assert groups["a"] != groups["b"] and -1 not in groups["a"] | groups["b"]
    members = sorted(sorted(c.member_files) for k, c in st.clusters.items()
                     if k in {f"semantic_{i}" for i in groups["a"] | groups["b"]})
    assert members == [sorted(paths[:12]), sorted(paths[12:])]
    # Existing label carried over by key, as before.
    assert st.clusters["semantic_0"].label == ["kept"]


def test_recluster_skips_when_store_has_too_few_files(tmp_path):
    d = _daemon(tmp_path, _VectorStore(["a.md"], np.ones((1, DIMS), np.float32)))
    asyncio.run(d._recluster())
    assert d.state.state.clusters == {}


@pytest.mark.parametrize("n", [0])
def test_recluster_empty_store(tmp_path, n):
    d = _daemon(tmp_path, _VectorStore([], np.empty((n, DIMS), np.float32)))
    asyncio.run(d._recluster())
    assert d.state.state.clusters == {}


# ── subprocess deadline / failure handling ────────────────────────────────────

def test_cluster_labels_isolated_times_out_and_kills_the_child():
    """No result within the deadline -> TimeoutError, and no child left behind.

    0.01 s is far below spawn + import time, so the deadline always fires
    while the child is still alive.
    """
    import multiprocessing

    from alfred.daemons.surveyor import cluster_labels_isolated

    with pytest.raises(TimeoutError):
        cluster_labels_isolated(_two_directions(), 3, 2, timeout_s=0.01)
    assert [p for p in multiprocessing.active_children() if p.name == "alfred-hdbscan"] == []


def test_cluster_labels_isolated_surfaces_child_errors():
    from alfred.daemons.surveyor import cluster_labels_isolated

    with pytest.raises(RuntimeError, match="ValueError"):
        cluster_labels_isolated(np.array([["a", "b"]]), 3, 2, timeout_s=120)


def test_recluster_timeout_is_config_backed_and_logged_as_cluster_failed(tmp_path, monkeypatch):
    from structlog.testing import capture_logs

    import alfred.daemons.surveyor as surveyor_mod

    seen: dict = {}

    def _hang(vectors, mcs, ms, timeout_s):
        seen["timeout_s"] = timeout_s
        raise TimeoutError("HDBSCAN subprocess gave no result within 12s; killed")

    monkeypatch.setattr(surveyor_mod, "cluster_labels_isolated", _hang)
    x = _two_directions()
    d = _daemon(tmp_path, _VectorStore([f"f{i}.md" for i in range(len(x))], x))
    d.cfg.recluster_timeout_s = 12.5

    with capture_logs() as logs:
        asyncio.run(d._recluster())

    assert seen["timeout_s"] == 12.5
    assert any(e["event"] == "surveyor.cluster_failed" and "12s" in e["error"] for e in logs)
    assert d.state.state.clusters == {}


def test_recluster_timeout_loads_from_yaml(tmp_path):
    import yaml

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "vault": {"path": str(tmp_path / "vault")},
        "surveyor": {"recluster_timeout_s": 42},
    }))
    assert AlfredConfig.load(cfg_path).recluster_timeout_s == 42.0
    cfg_path.write_text(yaml.safe_dump({"vault": {"path": str(tmp_path / "vault")}}))
    assert AlfredConfig.load(cfg_path).recluster_timeout_s == 900.0
