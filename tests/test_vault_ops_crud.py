"""Behavioral coverage for core/vault_ops.py create/edit/move/delete operations.

Phase 1 (see tests/test_vault_ops.py) covers the path-traversal boundary check
and write-lock races. This file covers the actual CRUD behavior of those same
operations — happy path plus at least one failure/edge case each — which
Phase 1 did not touch. `vault_delete`'s happy path + missing-file case are
already covered in test_vault_ops.py, so they are not repeated here.
"""
from __future__ import annotations

import pytest

from alfred.core.vault_ops import (
    VaultError,
    vault_create,
    vault_edit,
    vault_move,
    vault_read,
)


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    return v


# ── vault_create ────────────────────────────────────────────────────────────

def test_vault_create_writes_expected_frontmatter_and_default_body(vault):
    result = vault_create(vault, "note", "hello-world", set_fields={"tags": ["a"]})
    assert result["path"] == "note/hello-world.md"

    rec = vault_read(vault, "note/hello-world.md")
    fm = rec["frontmatter"]
    assert fm["type"] == "note"
    assert fm["name"] == "hello-world"
    assert fm["tags"] == ["a"]
    assert "created" in fm
    # frontmatter.load() strips the single trailing newline vault_create writes.
    assert rec["body"] == "# hello-world"


def test_vault_create_custom_body_overrides_default(vault):
    vault_create(vault, "note", "custom", body="Hand-written body.\n")
    rec = vault_read(vault, "note/custom.md")
    assert rec["body"] == "Hand-written body."


def test_vault_create_uses_type_specific_name_field(vault):
    # NAME_FIELD_BY_TYPE maps "conversation" -> "subject" instead of "name".
    vault_create(vault, "conversation", "topic-x")
    rec = vault_read(vault, "session/topic-x.md")
    assert rec["frontmatter"]["subject"] == "topic-x"
    assert "name" not in rec["frontmatter"]


def test_vault_create_duplicate_raises(vault):
    vault_create(vault, "note", "dup")
    with pytest.raises(VaultError, match="Already exists"):
        vault_create(vault, "note", "dup")


def test_vault_create_unknown_type_raises(vault):
    with pytest.raises(VaultError, match="Unknown type"):
        vault_create(vault, "not-a-real-type", "x")
    # Nothing should have been written to disk for an unknown type.
    assert not any(vault.rglob("*.md"))


def test_vault_create_invalid_status_raises(vault):
    with pytest.raises(VaultError, match="Invalid status"):
        vault_create(vault, "task", "y", set_fields={"status": "not-a-real-status"})


# ── vault_edit ──────────────────────────────────────────────────────────────

def test_vault_edit_set_fields_and_body_replace(vault):
    vault_create(vault, "note", "editme")
    result = vault_edit(
        vault,
        "note/editme.md",
        set_fields={"status": "active"},
        body_replace="New body text.",
    )
    assert set(result["fields_changed"]) >= {"status", "body"}

    rec = vault_read(vault, "note/editme.md")
    assert rec["frontmatter"]["status"] == "active"
    assert rec["body"] == "New body text."


def test_vault_edit_append_fields_converts_existing_scalar_to_list(vault):
    """append_fields on a field that already holds a bare scalar (not a list)
    must upgrade it to a list containing both values, per the documented
    fallback branch in vault_edit (`fm[k] = [existing, v]`)."""
    vault_create(vault, "note", "scalarfield", set_fields={"related": "solo-value"})
    vault_edit(vault, "note/scalarfield.md", append_fields={"related": "second-value"})

    rec = vault_read(vault, "note/scalarfield.md")
    assert rec["frontmatter"]["related"] == ["solo-value", "second-value"]


def test_vault_edit_append_fields_dedupes_within_existing_list(vault):
    vault_create(vault, "note", "listfield", set_fields={"tags": ["x"]})
    vault_edit(vault, "note/listfield.md", append_fields={"tags": "x"})
    rec = vault_read(vault, "note/listfield.md")
    assert rec["frontmatter"]["tags"] == ["x"]  # not duplicated


def test_vault_edit_body_append_adds_separator(vault):
    vault_create(vault, "note", "appendme", body="First line.")
    vault_edit(vault, "note/appendme.md", body_append="Second section.")
    rec = vault_read(vault, "note/appendme.md")
    assert rec["body"] == "First line.\n\nSecond section."


def test_vault_edit_missing_file_raises(vault):
    with pytest.raises(VaultError, match="File not found"):
        vault_edit(vault, "note/does-not-exist.md", set_fields={"status": "active"})


# ── vault_move ──────────────────────────────────────────────────────────────

def test_vault_move_happy_path(vault):
    vault_create(vault, "note", "movable")
    result = vault_move(vault, "note/movable.md", "note/moved.md")
    assert result == {"from": "note/movable.md", "to": "note/moved.md"}
    assert not (vault / "note" / "movable.md").exists()
    assert (vault / "note" / "moved.md").exists()


def test_vault_move_creates_destination_parent_dirs(vault):
    vault_create(vault, "note", "src")
    vault_move(vault, "note/src.md", "note/nested/dest.md")
    assert (vault / "note" / "nested" / "dest.md").exists()


def test_vault_move_source_missing_raises(vault):
    with pytest.raises(VaultError, match="Source not found"):
        vault_move(vault, "note/nope.md", "note/dest.md")


def test_vault_move_destination_exists_raises(vault):
    vault_create(vault, "note", "one")
    vault_create(vault, "note", "two")
    with pytest.raises(VaultError, match="Destination exists"):
        vault_move(vault, "note/one.md", "note/two.md")
    # Neither file should have been touched by the failed move.
    assert (vault / "note" / "one.md").exists()
    assert (vault / "note" / "two.md").exists()
