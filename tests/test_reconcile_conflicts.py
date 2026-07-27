"""scripts/reconcile_conflicts.py — classification and the destructive gate.

The safety property under test is narrow and absolute: nothing outside the
IDENTICAL set is ever deleted, and nothing at all is deleted without
--apply-identical.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "reconcile_conflicts",
    Path(__file__).resolve().parents[1] / "scripts" / "reconcile_conflicts.py",
)
rc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rc)


def _write(p: Path, body: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


CONFLICT = "note.sync-conflict-20260622-145746-ZO3SA2G.md"


def test_live_counterpart_strips_the_marker():
    p = Path("/v/decision") / CONFLICT
    assert rc.live_counterpart(p) == Path("/v/decision/note.md")


def test_identical_bodies(tmp_path):
    body = "---\ntype: note\n---\nSame content.\n"
    _write(tmp_path / "note.md", body)
    conflict = _write(tmp_path / CONFLICT, body)

    assert rc.classify(conflict) == (rc.IDENTICAL, 1.0)


def test_frontmatter_only_difference_still_counts_as_identical(tmp_path):
    """A differing `modified:` stamp is not a content difference."""
    _write(tmp_path / "note.md", "---\nmodified: '2026-01-01'\n---\nSame content.\n")
    conflict = _write(tmp_path / CONFLICT, "---\nmodified: '2026-06-22'\n---\nSame content.\n")

    assert rc.classify(conflict)[0] == rc.IDENTICAL


def test_diverging_bodies(tmp_path):
    _write(tmp_path / "note.md", "---\ntype: note\n---\nOriginal content here.\n")
    conflict = _write(tmp_path / CONFLICT, "---\ntype: note\n---\nCompletely other text.\n")

    category, ratio = rc.classify(conflict)
    assert category == rc.DIFFERS
    assert 0.0 <= ratio < 1.0


def test_missing_original_is_orphaned(tmp_path):
    conflict = _write(tmp_path / CONFLICT, "---\ntype: note\n---\nOnly copy left.\n")
    assert rc.classify(conflict) == (rc.ORPHANED, None)


def test_non_markdown_is_not_a_vault_record(tmp_path):
    conflict = _write(
        tmp_path / "app.sync-conflict-20260622-145746-ZO3SA2G.json", "{}"
    )
    assert rc.classify(conflict) == (rc.NON_MD, None)


def test_dry_run_deletes_nothing(tmp_path, monkeypatch, capsys):
    body = "---\ntype: note\n---\nSame content.\n"
    _write(tmp_path / "note.md", body)
    conflict = _write(tmp_path / CONFLICT, body)
    orphan = _write(tmp_path / "gone.sync-conflict-20260622-145746-ZO3SA2G.md", "x" * 40)

    monkeypatch.setattr(
        "sys.argv",
        ["reconcile_conflicts.py", "--root", str(tmp_path),
         "--report", str(tmp_path / "report.md")],
    )
    assert rc.main() == 0

    assert conflict.exists() and orphan.exists()
    assert "Dry run" in capsys.readouterr().out


def test_apply_identical_removes_only_identical(tmp_path, monkeypatch):
    """The whole safety contract, in one assertion set."""
    body = "---\ntype: note\n---\nSame content.\n"
    _write(tmp_path / "a.md", body)
    identical = _write(tmp_path / "a.sync-conflict-20260622-145746-ZO3SA2G.md", body)

    _write(tmp_path / "b.md", "---\ntype: note\n---\nOriginal.\n")
    differs = _write(tmp_path / "b.sync-conflict-20260622-145746-ZO3SA2G.md",
                     "---\ntype: note\n---\nDivergent text entirely.\n")

    orphan = _write(tmp_path / "c.sync-conflict-20260622-145746-ZO3SA2G.md",
                    "---\ntype: note\n---\nNo counterpart.\n")

    monkeypatch.setattr(
        "sys.argv",
        ["reconcile_conflicts.py", "--root", str(tmp_path),
         "--report", str(tmp_path / "report.md"), "--apply-identical"],
    )
    assert rc.main() == 0

    assert not identical.exists(), "the byte-identical copy should be gone"
    assert differs.exists(), "DIFFERS must never be batch-deleted"
    assert orphan.exists(), "ORPHANED must never be batch-deleted"
    assert (tmp_path / "a.md").exists(), "a live original must never be touched"
    assert (tmp_path / "b.md").exists()


def test_report_lists_differs_lowest_similarity_first(tmp_path):
    _write(tmp_path / "b.md", "---\n---\nOriginal text that is fairly long here.\n")
    _write(tmp_path / "b.sync-conflict-20260622-145746-ZO3SA2G.md",
           "---\n---\nOriginal text that is fairly long HERE.\n")   # near-identical
    _write(tmp_path / "c.md", "---\n---\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n")
    _write(tmp_path / "c.sync-conflict-20260622-145746-ZO3SA2G.md",
           "---\n---\nzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz\n")            # very different

    report = rc.render_report(tmp_path, rc.scan(tmp_path))
    rows = [ln for ln in report.splitlines() if ln.startswith("| 0.")]

    assert len(rows) == 2
    first = float(rows[0].split("|")[1])
    second = float(rows[1].split("|")[1])
    assert first < second, "most divergent must sort first — it needs eyes most"
