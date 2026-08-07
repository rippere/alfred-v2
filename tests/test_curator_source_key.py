"""Behavioral coverage for the curator's recurring-feed upsert.

A feed that drops a fresh full snapshot on every sync marks it with
`<!-- alfred:source <key> -->` (or a frontmatter `source_key`). That key is the
identity of the FEED, not of one drop, so each snapshot must REPLACE the
feed's record.

Nothing consumed the marker before this, and the record filename came from the
drop's H1. The ECC instinct bridge titled its heading "ECC Procedural Instincts
(N)" with N the instinct count, so every change to N slugged to a new filename
and minted another ~1.2 MB record: 24 near-identical snapshots holding 92,327 of
122,911 chunks — 81.5% of the entire vector index — for a single feed.

Stabilising the heading alone is NOT a fix: vault_create then raises
"Already exists", the fallback mints one suffixed twin, and every sync after
that hits "Already exists" twice and lands in `curator.duplicate_skip`, which
silently discards the update forever. Both halves are covered here.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from alfred.config import AlfredConfig
from alfred.core.vault_ops import vault_read
from alfred.daemons.curator import (
    CuratorDaemon,
    _extract_source_key,
    _find_by_source_key,
)
from alfred.store.state import StateStore


def _make_daemon(tmp_path: Path) -> CuratorDaemon:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    (vault_path / "inbox").mkdir()
    cfg = AlfredConfig(vault_path=vault_path, data_dir=tmp_path / "data")
    state = StateStore(tmp_path / "state.json")
    state.load()
    return CuratorDaemon(cfg, state, asyncio.Queue())


def _drop(daemon: CuratorDaemon, body: str, *, name: str = "feed.md") -> None:
    """Write an inbox drop and run one curator pass over it."""
    (daemon.cfg.vault_path / "inbox" / name).write_text(body, encoding="utf-8")
    asyncio.run(daemon._process_inbox())


def _snapshot(count: int, *, heading: str = "# ECC Procedural Instincts") -> str:
    return (
        "---\ntype: note\nsource_key: ecc-instincts\n---\n"
        "<!-- alfred:source ecc-instincts -->\n\n"
        f"{heading}\n\n**Instincts:** {count}\n\nBody for snapshot {count}.\n"
    )


def _notes(daemon: CuratorDaemon) -> list[str]:
    return sorted(p.name for p in (daemon.cfg.vault_path / "note").glob("*.md"))


# ── key extraction ───────────────────────────────────────────────────────────

def test_extract_source_key_reads_the_html_marker():
    assert _extract_source_key({}, "<!-- alfred:source ecc-instincts -->\nbody") == "ecc-instincts"


def test_extract_source_key_tolerates_whitespace():
    assert _extract_source_key({}, "<!--alfred:source   feed-1  -->") == "feed-1"


def test_extract_source_key_prefers_frontmatter():
    """An explicit declaration beats the inline convention."""
    assert _extract_source_key({"source_key": "explicit"}, "<!-- alfred:source inline -->") == "explicit"


@pytest.mark.parametrize("fm,body", [({}, "no marker here"), ({"source_key": "  "}, "")])
def test_extract_source_key_absent(fm, body):
    assert _extract_source_key(fm, body) == ""


def test_find_by_source_key_matches_and_misses(tmp_path):
    vault = tmp_path / "vault"
    (vault / "note").mkdir(parents=True)
    (vault / "note" / "feed.md").write_text(
        "---\ntype: note\nsource_key: ecc-instincts\n---\nbody\n", encoding="utf-8"
    )
    (vault / "note" / "other.md").write_text("---\ntype: note\n---\nbody\n", encoding="utf-8")

    assert _find_by_source_key(vault, "note", "ecc-instincts") == "note/feed.md"
    assert _find_by_source_key(vault, "note", "nope") is None
    assert _find_by_source_key(vault, "project", "ecc-instincts") is None


def test_find_by_source_key_ignores_a_match_in_the_body(tmp_path):
    """Only the frontmatter block counts — these records embed instinct YAML in
    fenced blocks, so a body-wide scan would match the wrong thing."""
    vault = tmp_path / "vault"
    (vault / "note").mkdir(parents=True)
    (vault / "note" / "decoy.md").write_text(
        "---\ntype: note\n---\n```\nsource_key: ecc-instincts\n```\n", encoding="utf-8"
    )
    assert _find_by_source_key(vault, "note", "ecc-instincts") is None


# ── the regression: repeated syncs must not accumulate records ───────────────

def test_repeated_snapshots_update_one_record(tmp_path):
    daemon = _make_daemon(tmp_path)

    _drop(daemon, _snapshot(830), name="a.md")
    _drop(daemon, _snapshot(834), name="b.md")
    _drop(daemon, _snapshot(841), name="c.md")

    assert _notes(daemon) == ["ecc-procedural-instincts.md"], "one feed, one record"
    rec = vault_read(daemon.cfg.vault_path, "note/ecc-procedural-instincts.md")
    assert "**Instincts:** 841" in rec["body"], "record holds the LATEST snapshot"
    assert "**Instincts:** 830" not in rec["body"]
    assert rec["frontmatter"]["source_key"] == "ecc-instincts"


def test_a_changing_heading_no_longer_mints_a_new_record(tmp_path):
    """The exact historical failure: the count lived in the H1, so each sync
    slugged to a new filename. source_key identity must override that."""
    daemon = _make_daemon(tmp_path)

    _drop(daemon, _snapshot(830, heading="# ECC Procedural Instincts (830)"), name="a.md")
    _drop(daemon, _snapshot(834, heading="# ECC Procedural Instincts (834)"), name="b.md")

    assert len(_notes(daemon)) == 1


def test_a_stable_heading_alone_would_have_gone_stale(tmp_path):
    """Guards the half-fix — stabilising the heading WITHOUT source_key.

    The collision fallback suffixes with `inbox_file.stem[-8:]`, and a recurring
    feed always writes the same inbox filename (the bridge writes
    `ecc-instincts.md` every hour), so that suffix is constant too. Drop 1
    creates the record, drop 2 mints one suffixed twin, and drop 3 onward
    collides twice and lands in `curator.duplicate_skip` — silently discarded.
    The content then never updates again."""
    daemon = _make_daemon(tmp_path)
    no_key = "---\ntype: note\n---\n\n# Stable Title\n\nrevision {n}\n"

    for n in (1, 2, 3):
        _drop(daemon, no_key.format(n=n), name="recurring-feed.md")

    names = _notes(daemon)
    assert len(names) == 2, "a twin, not an update — this is the old behavior"
    bodies = [vault_read(daemon.cfg.vault_path, f"note/{n}")["body"] for n in names]
    assert not any("revision 3" in b for b in bodies), "third drop was silently dropped"


def test_drops_without_a_source_key_are_unaffected(tmp_path):
    """Ordinary notes must keep creating separate records."""
    daemon = _make_daemon(tmp_path)

    _drop(daemon, "---\ntype: note\nname: alpha\n---\nfirst\n", name="a.md")
    _drop(daemon, "---\ntype: note\nname: beta\n---\nsecond\n", name="b.md")

    assert _notes(daemon) == ["alpha.md", "beta.md"]


def test_distinct_feeds_get_distinct_records(tmp_path):
    daemon = _make_daemon(tmp_path)

    _drop(daemon, "---\ntype: note\n---\n<!-- alfred:source feed-a -->\n\n# Feed A\n\nx\n", name="a.md")
    _drop(daemon, "---\ntype: note\n---\n<!-- alfred:source feed-b -->\n\n# Feed B\n\ny\n", name="b.md")

    assert _notes(daemon) == ["feed-a.md", "feed-b.md"]


def test_first_snapshot_stamps_source_key_into_frontmatter(tmp_path):
    """The marker is an HTML comment in the body; identity has to be promoted
    to frontmatter or the next sync cannot find this record."""
    daemon = _make_daemon(tmp_path)
    _drop(daemon, "---\ntype: note\n---\n<!-- alfred:source feed-a -->\n\n# Feed A\n\nx\n")

    rec = vault_read(daemon.cfg.vault_path, "note/feed-a.md")
    assert rec["frontmatter"]["source_key"] == "feed-a"


def test_updated_drop_is_moved_to_processed(tmp_path):
    """The upsert branch returns early, so it must still archive its input —
    otherwise the same drop is re-ingested every tick."""
    daemon = _make_daemon(tmp_path)
    _drop(daemon, _snapshot(830), name="a.md")
    _drop(daemon, _snapshot(834), name="b.md")

    processed = daemon.cfg.vault_path / "inbox" / "processed"
    assert (processed / "b.md").exists()
    assert not (daemon.cfg.vault_path / "inbox" / "b.md").exists()
