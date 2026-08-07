"""Regression coverage for JanitorDaemon._infer_type and the schema map it
reads (DIRECTORY_TO_TYPE).

_infer_type used to build its directory -> type map by inverting TYPE_DIRECTORY
with `{v: k for k, v in TYPE_DIRECTORY.items()}`. TYPE_DIRECTORY is many-to-one,
so that inversion is lossy: it keeps whichever type happened to be declared
LAST for a directory. Measured on the real map before the fix:

    session/x.md   -> 'ai-dialogue'   (legacy read-path-only type)
    topic/x.md     -> 'learn'         (legacy read-path-only type)
    decision/x.md  -> ''              (decision/ is not a TYPE_DIRECTORY value)
    note/x.md      -> 'note'          (correct, no collision)

session/ and topic/ are the two largest content directories in the vault, so
FM001 autofix was stamping the wrong `type` on them and inferring nothing at
all for the epistemic directories that exist on disk but consolidate to topic/
on write.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from alfred.config import AlfredConfig
from alfred.core.schema import DIRECTORY_TO_TYPE, KNOWN_TYPES, TYPE_DIRECTORY
from alfred.core.vault_ops import vault_read
from alfred.daemons.janitor import JanitorDaemon
from alfred.store.state import StateStore


def _make_daemon(tmp_path: Path) -> JanitorDaemon:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    cfg = AlfredConfig(vault_path=vault_path, data_dir=tmp_path / "data")
    state = StateStore(tmp_path / "state.json")
    state.load()
    return JanitorDaemon(cfg, state, asyncio.Queue(), store=None)


# ── the map itself ───────────────────────────────────────────────────────────

def test_directory_to_type_prefers_the_self_named_type_over_declaration_order():
    """The collision-resolved inverse must pick the type whose own name equals
    the directory, not the last one declared. This is the exact defect: a naive
    inversion yields ai-dialogue for session/ and learn for topic/."""
    naive = {v: k for k, v in TYPE_DIRECTORY.items()}
    assert naive["session"] == "ai-dialogue"      # documents the old behavior
    assert naive["topic"] == "learn"

    assert DIRECTORY_TO_TYPE["session"] == "session"
    assert DIRECTORY_TO_TYPE["topic"] == "topic"


def test_directory_to_type_keeps_unambiguous_renamed_directories():
    """Directories whose name differs from the type have exactly one
    contributing type, so they resolve unambiguously."""
    assert DIRECTORY_TO_TYPE["ideas"] == "idea"
    assert DIRECTORY_TO_TYPE["drafts"] == "script"
    assert DIRECTORY_TO_TYPE["hooks"] == "hook"


def test_directory_to_type_covers_every_directory_in_type_directory():
    assert set(DIRECTORY_TO_TYPE) == set(TYPE_DIRECTORY.values())


def test_directory_to_type_only_yields_known_types():
    assert set(DIRECTORY_TO_TYPE.values()) <= KNOWN_TYPES


def test_directory_to_type_is_order_independent():
    """Rebuilding from a reversed declaration order must give the same map —
    the whole point is that the result no longer depends on ordering."""
    from alfred.core import schema

    original = dict(schema.TYPE_DIRECTORY)
    try:
        schema.TYPE_DIRECTORY = dict(reversed(list(original.items())))
        assert schema._build_directory_to_type() == DIRECTORY_TO_TYPE
    finally:
        schema.TYPE_DIRECTORY = original


# ── _infer_type ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "rel_path,expected",
    [
        # The four paths measured as broken/correct before the fix.
        ("session/x.md", "session"),
        ("topic/x.md", "topic"),
        ("decision/x.md", "decision"),
        ("note/x.md", "note"),
        # Other directories that exist on disk but consolidate into topic/ on
        # write, so they are absent from TYPE_DIRECTORY's values entirely.
        ("assumption/x.md", "assumption"),
        ("constraint/x.md", "constraint"),
        ("contradiction/x.md", "contradiction"),
        ("input/x.md", "input"),
        # Directories whose name differs from the type.
        ("ideas/x.md", "idea"),
        ("drafts/x.md", "script"),
        ("hooks/x.md", "hook"),
        # Plain one-to-one directories.
        ("project/x.md", "project"),
        ("person/x.md", "person"),
        ("event/x.md", "event"),
        ("synthesis/x.md", "synthesis"),
        ("wiki/x.md", "wiki"),
        ("run/x.md", "run"),
    ],
)
def test_infer_type_resolves_each_vault_directory(tmp_path, rel_path, expected):
    assert _make_daemon(tmp_path)._infer_type(rel_path) == expected


def test_infer_type_never_returns_a_legacy_only_type(tmp_path):
    """ai-dialogue and learn are read-path compatibility shims that are never
    written. Inference must not resurrect them for any vault directory."""
    daemon = _make_daemon(tmp_path)
    inferred = {daemon._infer_type(f"{d}/x.md") for d in TYPE_DIRECTORY.values()}
    assert not inferred & {"ai-dialogue", "learn"}


def test_infer_type_walks_nested_directories_nearest_first(tmp_path):
    """Nesting must resolve to the nearest meaningful directory, so archived
    sessions are still sessions and a dated subfolder does not hide the type."""
    daemon = _make_daemon(tmp_path)
    assert daemon._infer_type("_archived/session/2026/x.md") == "session"
    assert daemon._infer_type("session/2026-08/x.md") == "session"


def test_infer_type_returns_empty_rather_than_guessing(tmp_path):
    """No resolvable directory means leave `type` unset — a wrong stamp is
    worse than an absent one, because FM002/FM003 then key off it."""
    daemon = _make_daemon(tmp_path)
    assert daemon._infer_type("root-level.md") == ""
    assert daemon._infer_type("inbox/processed/x.md") == ""
    assert daemon._infer_type("") == ""


def test_infer_type_handles_windows_separators(tmp_path):
    assert _make_daemon(tmp_path)._infer_type("session\\x.md") == "session"


# ── end-to-end through FM001 autofix ─────────────────────────────────────────

def _autofix_file(daemon: JanitorDaemon, vault_path: Path, rel_path: str) -> list[str]:
    file_issues = daemon._check_file(vault_path, rel_path)
    issues = {rel_path: [{"code": i.code, "message": i.message} for i in file_issues]}
    return asyncio.run(daemon._autofix(issues, vault_path))


@pytest.mark.parametrize(
    "directory,expected", [("session", "session"), ("topic", "topic"), ("decision", "decision")]
)
def test_autofix_stamps_the_correct_type_on_the_big_directories(tmp_path, directory, expected):
    """The user-visible consequence: a type-less note in session/ or topic/ came
    out of autofix labelled ai-dialogue or learn, and one in decision/ came out
    with no type at all."""
    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    (vault_path / directory).mkdir()
    (vault_path / directory / "bare.md").write_text(
        "---\n---\nBody with no frontmatter fields.\n", encoding="utf-8"
    )

    rel = f"{directory}/bare.md"
    assert _autofix_file(daemon, vault_path, rel) == [rel]
    assert vault_read(vault_path, rel)["frontmatter"]["type"] == expected
