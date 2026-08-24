"""Behavioral coverage for JanitorDaemon's deterministic autofix stage
(_autofix), which fixes issue codes FM001-FM004 without an LLM call:

  FM001 MISSING_REQUIRED_FIELD — backfills missing `type`/`created` (and name
        field) from directory location / file mtime / filename.
  FM002 INVALID_TYPE_VALUE     — corrects known type typos via correct_type().
  FM003 INVALID_STATUS_VALUE   — corrects known status typos via
        correct_status().
  FM004 INVALID_FIELD_TYPE     — wraps a scalar value in a list for fields
        declared list-typed in LIST_FIELDS.

tests/test_janitor.py already covers vector-store embedding cleanup on
archive/dedup and the daemon's `store` constructor requirement — this file
covers the separate autofix stage only.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from alfred.config import AlfredConfig
from alfred.core.vault_ops import vault_read
from alfred.daemons.janitor import IssueCode, JanitorDaemon
from alfred.store.state import StateStore


def _make_daemon(tmp_path: Path) -> JanitorDaemon:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    cfg = AlfredConfig(vault_path=vault_path, data_dir=tmp_path / "data")
    state = StateStore(tmp_path / "state.json")
    state.load()
    events: asyncio.Queue = asyncio.Queue()
    return JanitorDaemon(cfg, state, events, store=None)


def _autofix_file(daemon: JanitorDaemon, vault_path: Path, rel_path: str) -> list[str]:
    """Run _check_file -> _autofix for a single file, as _structural_sweep does."""
    file_issues = daemon._check_file(vault_path, rel_path)
    issues = {rel_path: [{"code": i.code, "message": i.message} for i in file_issues]}
    return asyncio.run(daemon._autofix(issues, vault_path))


def test_fm001_backfills_missing_type_and_created(tmp_path):
    """A file with no `type`/`created` frontmatter gets both inferred: type
    from its parent directory (via TYPE_DIRECTORY reverse-lookup), created
    from the file's mtime."""
    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    note_dir = vault_path / "note"
    note_dir.mkdir()
    fp = note_dir / "bare.md"
    fp.write_text("---\n---\nJust a body, no frontmatter fields.\n", encoding="utf-8")

    fixed = _autofix_file(daemon, vault_path, "note/bare.md")

    assert fixed == ["note/bare.md"]
    rec = vault_read(vault_path, "note/bare.md")
    fm = rec["frontmatter"]
    assert fm["type"] == "note"
    assert "created" in fm and fm["created"]
    assert fm["name"] == "bare"


def test_fm002_corrects_known_type_typo(tmp_path):
    """A recognized type typo (plural 'notes') is corrected via correct_type()."""
    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    note_dir = vault_path / "note"
    note_dir.mkdir()
    fp = note_dir / "typo.md"
    fp.write_text("---\ntype: notes\ncreated: '2026-01-01'\n---\nBody text.\n", encoding="utf-8")

    fixed = _autofix_file(daemon, vault_path, "note/typo.md")

    assert fixed == ["note/typo.md"]
    rec = vault_read(vault_path, "note/typo.md")
    assert rec["frontmatter"]["type"] == "note"


def test_fm002_leaves_unrecognized_type_unfixed(tmp_path):
    """A type typo with no entry in _TYPE_CORRECTIONS must NOT be silently
    guessed at — the file stays flagged/untouched rather than getting a wrong
    type written."""
    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    note_dir = vault_path / "note"
    note_dir.mkdir()
    fp = note_dir / "bogus.md"
    # `name` is set explicitly so the FM001 name-backfill branch can't fire and
    # mask whether the type itself was (wrongly) autofixed.
    fp.write_text(
        "---\ntype: totally-bogus-type\ncreated: '2026-01-01'\nname: bogus\n---\nBody text.\n",
        encoding="utf-8",
    )

    fixed = _autofix_file(daemon, vault_path, "note/bogus.md")

    assert fixed == []
    rec = vault_read(vault_path, "note/bogus.md")
    assert rec["frontmatter"]["type"] == "totally-bogus-type"


def test_fm003_corrects_known_status_typo(tmp_path):
    """An invalid status with a known correction ('in-progress' -> 'active'
    for type 'task') gets rewritten to the valid value."""
    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    task_dir = vault_path / "task"
    task_dir.mkdir()
    fp = task_dir / "job.md"
    fp.write_text(
        "---\ntype: task\ncreated: '2026-01-01'\nstatus: in-progress\n---\nDo the thing.\n",
        encoding="utf-8",
    )

    fixed = _autofix_file(daemon, vault_path, "task/job.md")

    assert fixed == ["task/job.md"]
    rec = vault_read(vault_path, "task/job.md")
    assert rec["frontmatter"]["status"] == "active"


def test_fm003_leaves_unrecognized_status_unfixed(tmp_path):
    """A status with no known correction for its type is left as-is rather
    than guessed."""
    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    task_dir = vault_path / "task"
    task_dir.mkdir()
    fp = task_dir / "job2.md"
    # `name` is set explicitly so the FM001 name-backfill branch can't fire and
    # mask whether the status itself was (wrongly) autofixed.
    fp.write_text(
        "---\ntype: task\ncreated: '2026-01-01'\nstatus: totally-not-a-status\nname: job2\n---\nDo the thing.\n",
        encoding="utf-8",
    )

    fixed = _autofix_file(daemon, vault_path, "task/job2.md")

    assert fixed == []
    rec = vault_read(vault_path, "task/job2.md")
    assert rec["frontmatter"]["status"] == "totally-not-a-status"


def test_fm004_wraps_scalar_list_field_in_a_list(tmp_path):
    """A LIST_FIELDS field (e.g. `tags`) stored as a bare scalar is wrapped
    in a single-element list."""
    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    note_dir = vault_path / "note"
    note_dir.mkdir()
    fp = note_dir / "scalar-tag.md"
    fp.write_text(
        "---\ntype: note\ncreated: '2026-01-01'\ntags: solo-tag\n---\nBody.\n",
        encoding="utf-8",
    )

    fixed = _autofix_file(daemon, vault_path, "note/scalar-tag.md")

    assert fixed == ["note/scalar-tag.md"]
    rec = vault_read(vault_path, "note/scalar-tag.md")
    assert rec["frontmatter"]["tags"] == ["solo-tag"]


def test_fm004_project_field_scalar_is_not_wrapped(tmp_path):
    """`project` is a documented exception in both _check_file and _autofix:
    a bare string value is accepted as-is, not flagged/wrapped, even though
    `project` is in LIST_FIELDS.

    The file also carries a scalar `tags` field, a genuine FM004 violation,
    so the note actually clears _autofix's per-file issue-code gate and the
    FM004 loop body runs. Without that second field, `project` would be the
    file's only non-list-typed field; _check_file's own project exception
    means no FM004 issue is ever raised, the file never clears the gate, and
    the loop body's project-exception check is never reached at all --
    proving nothing about _autofix's own copy of the exception.
    """
    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    note_dir = vault_path / "note"
    note_dir.mkdir()
    fp = note_dir / "proj-scalar.md"
    fp.write_text(
        "---\ntype: note\ncreated: '2026-01-01'\nproject: solo-project\ntags: solo-tag\n---\nBody.\n",
        encoding="utf-8",
    )

    file_issues = daemon._check_file(vault_path, "note/proj-scalar.md")
    assert not any(
        i.code == IssueCode.INVALID_FIELD_TYPE.value and "project" in i.message
        for i in file_issues
    )
    assert any(
        i.code == IssueCode.INVALID_FIELD_TYPE.value and "tags" in i.message
        for i in file_issues
    )

    fixed = _autofix_file(daemon, vault_path, "note/proj-scalar.md")
    # The loop body ran (tags got wrapped), proving it reached the point
    # where the project exception applies -- not merely skipped via the gate.
    assert fixed == ["note/proj-scalar.md"]
    rec = vault_read(vault_path, "note/proj-scalar.md")
    assert rec["frontmatter"]["tags"] == ["solo-tag"]
    assert rec["frontmatter"]["project"] == "solo-project"


def test_autofix_only_touches_files_with_deterministic_issue_codes(tmp_path):
    """A file whose only issue is a non-autofixable code (e.g. a broken
    wikilink, LINK001) must be left completely untouched by _autofix."""
    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    note_dir = vault_path / "note"
    note_dir.mkdir()
    fp = note_dir / "haslink.md"
    original = "---\ntype: note\ncreated: '2026-01-01'\n---\nSee [[nonexistent-target]] for more.\n"
    fp.write_text(original, encoding="utf-8")

    fixed = _autofix_file(daemon, vault_path, "note/haslink.md")

    assert fixed == []
    assert fp.read_text(encoding="utf-8") == original
