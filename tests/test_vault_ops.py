"""Regression tests for core/vault_ops.py path-boundary check and write-lock coverage.

Covers two bug classes fixed together:
1. `_resolve()` used a raw string-prefix test to enforce vault containment, which a
   sibling directory whose name has the vault dir's name as a prefix (e.g.
   `vault-finance` vs `vault-finance-backup-20260513-151423`) could bypass.
2. `vault_append_to_topic()` and `vault_delete()` performed their
   existence-check-through-write / existence-check-through-unlink sequences without
   holding the module's `_write_lock`, unlike `vault_create`/`vault_edit`/`vault_move` —
   a TOCTOU window that could silently lose concurrent writes.
"""
from __future__ import annotations

import threading

import pytest

from alfred.core.vault_ops import (
    VaultError,
    vault_append_to_topic,
    vault_create,
    vault_delete,
    vault_edit,
    vault_read,
)

N_THREADS = 8
OPS_PER_THREAD = 25


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "vault-finance"
    v.mkdir()
    return v


def test_resolve_rejects_prefix_sibling_directory(vault):
    """A sibling dir whose name prefixes the vault dir's name must not be reachable.

    Mirrors the real deployment collision: /mnt/external/vault-finance vs
    /mnt/external/vault-finance-backup-20260513-151423 — both exist side by side,
    and the buggy `str.startswith()` check let a relative path resolve into the
    sibling because its string representation starts with the vault's string path.
    """
    sibling = vault.parent / (vault.name + "-backup-20260513-151423")
    sibling.mkdir()
    secret = sibling / "secret.md"
    secret.write_text("---\ntype: note\n---\nsecret contents\n", encoding="utf-8")

    # Relative path that walks out of the vault and into the prefix-colliding sibling.
    escaping_rel_path = f"../{sibling.name}/secret.md"

    with pytest.raises(VaultError, match="Path traversal denied"):
        vault_read(vault, escaping_rel_path)


def test_resolve_allows_paths_actually_inside_vault(vault):
    vault_create(vault, "note", "hello")
    result = vault_read(vault, "note/hello.md")
    assert result["path"] == "note/hello.md"


def test_resolve_rejects_dotdot_escape(vault):
    with pytest.raises(VaultError, match="Path traversal denied"):
        vault_read(vault, "../outside.md")


def test_vault_append_to_topic_concurrent_no_lost_writes(vault):
    """Two+ threads appending to the same topic file must not lose either write."""
    errors: list[BaseException] = []

    def append_worker(tid: int) -> None:
        try:
            for i in range(OPS_PER_THREAD):
                vault_append_to_topic(
                    vault,
                    "shared-topic",
                    f"Insight t{tid}-{i}",
                    f"body from thread {tid} op {i}",
                    tags=["shared"],
                )
        except BaseException as e:  # noqa: BLE001 — collect everything for the assertion
            errors.append(e)

    threads = [threading.Thread(target=append_worker, args=(tid,)) for tid in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"exceptions during concurrent appends: {errors!r}"

    result = vault_read(vault, "topic/shared-topic.md")
    body = result["body"]
    for tid in range(N_THREADS):
        for i in range(OPS_PER_THREAD):
            assert f"Insight t{tid}-{i}" in body, f"lost write from thread {tid} op {i}"


def test_vault_edit_concurrent_no_lost_writes(vault):
    """Two threads editing the same file's list field concurrently must not lose either write."""
    vault_create(vault, "note", "shared", set_fields={"tags": []})
    errors: list[BaseException] = []

    def edit_worker(tid: int) -> None:
        try:
            for i in range(OPS_PER_THREAD):
                vault_edit(vault, "note/shared.md", append_fields={"tags": f"t{tid}-{i}"})
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=edit_worker, args=(tid,)) for tid in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"exceptions during concurrent edits: {errors!r}"

    result = vault_read(vault, "note/shared.md")
    tags = result["frontmatter"]["tags"]
    expected = {f"t{tid}-{i}" for tid in range(N_THREADS) for i in range(OPS_PER_THREAD)}
    missing = expected - set(tags)
    assert not missing, f"lost writes — {len(missing)} tags missing, e.g. {sorted(missing)[:5]}"


def test_vault_delete_removes_file(vault):
    vault_create(vault, "note", "to-delete")
    result = vault_delete(vault, "note/to-delete.md")
    assert result["deleted"] is True
    assert not (vault / "note" / "to-delete.md").exists()


def test_vault_delete_missing_raises(vault):
    with pytest.raises(VaultError, match="File not found"):
        vault_delete(vault, "note/does-not-exist.md")
