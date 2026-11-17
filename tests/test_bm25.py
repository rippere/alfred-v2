"""Crash-safety tests for BM25Store (audit P1-03).

save() must go through a temp-file + os.replace() so a mid-write kill can
never leave a truncated pickle behind — matching StateStore.save() and
GraphStore.save(). load() must treat a corrupt/truncated pickle as a
recoverable "not found" state (return False) rather than raising, matching
the existing contract callers rely on (`if not store.load(): raise ...`).
"""
from __future__ import annotations

import os

import pytest

from alfred.store.bm25 import BM25Store


@pytest.fixture
def bm25_path(tmp_path):
    return tmp_path / "bm25_index.pkl"


def _fitted_store(path) -> BM25Store:
    store = BM25Store(path)
    store.fit_and_store(
        ["alpha beta gamma", "beta gamma delta", "gamma delta epsilon"],
        ["chunk-1", "chunk-2", "chunk-3"],
    )
    return store


def test_load_returns_false_on_corrupt_pickle(bm25_path):
    """Garbage bytes must not propagate a raw UnpicklingError out of load()."""
    bm25_path.write_bytes(b"not a pickle at all \x00\x01\x02")
    store = BM25Store(bm25_path)
    assert store.load() is False
    assert not store.is_fitted


def test_load_returns_false_on_truncated_pickle(bm25_path):
    """A pickle truncated mid-write (simulating a killed process) must also be recoverable."""
    store = _fitted_store(bm25_path)
    store.save()
    full = bm25_path.read_bytes()
    bm25_path.write_bytes(full[: len(full) // 2])

    fresh = BM25Store(bm25_path)
    assert fresh.load() is False
    assert not fresh.is_fitted


def test_load_missing_file_still_returns_false(bm25_path):
    store = BM25Store(bm25_path)
    assert store.load() is False


def test_save_then_load_roundtrip(bm25_path):
    store = _fitted_store(bm25_path)
    store.save()

    fresh = BM25Store(bm25_path)
    assert fresh.load() is True
    assert fresh.is_fitted
    assert fresh.has_corpus
    assert fresh.vocab_size() == store.vocab_size()

    # No orphaned tmp file left behind.
    assert not bm25_path.with_name(bm25_path.name + ".tmp").exists()


def test_save_uses_tmp_file_then_atomic_replace(bm25_path, monkeypatch):
    """If os.replace() fails after the tmp write, the original file must be untouched."""
    store = _fitted_store(bm25_path)
    store.save()
    original_bytes = bm25_path.read_bytes()

    # Refit with different data so a second save would produce different bytes.
    store.fit_and_store(["zzz yyy xxx"], ["chunk-9"])

    real_replace = os.replace

    def failing_replace(*args, **kwargs):
        raise OSError("simulated crash between tmp-write and replace")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError):
        store.save()
    monkeypatch.setattr(os, "replace", real_replace)

    # Original file must be untouched — no partial/truncated overwrite occurred.
    assert bm25_path.read_bytes() == original_bytes

    # The tmp file should exist (write succeeded, only the replace failed) and
    # must be a complete, loadable pickle of the new data — proof the write
    # itself was not the truncated one.
    tmp_path = bm25_path.with_name(bm25_path.name + ".tmp")
    assert tmp_path.exists()
    tmp_store = BM25Store(tmp_path)
    assert tmp_store.load() is True
