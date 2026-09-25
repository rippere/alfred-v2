"""A keyed feed's drop updates that feed's vault record in place.

The ECC instinct bridge drops a full ~1.3 MB snapshot into inbox/ whenever the
instinct set changes, marked `<!-- alfred:source ecc-instincts -->` with
`source_key: ecc-instincts` in frontmatter and `type: reference`, which the
schema does not know. Nothing honoured the key, and the model classified each
sync afresh, so the feed became ~40 near-identical records across ten type
directories (note/, drafts/, process/, topic/, task/, ...), then fell into
`created_with_suffix` / `duplicate_skip` once every name was taken. Each copy
is ~5.9K embedded chunks.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import frontmatter

from alfred.config import AlfredConfig
from alfred.daemons.curator import CuratorDaemon
from alfred.store.state import StateStore

_MARKER = "<!-- alfred:source ecc-instincts -->"


def _make_daemon(tmp_path: Path) -> CuratorDaemon:
    vault_path = tmp_path / "vault"
    (vault_path / "inbox").mkdir(parents=True)
    cfg = AlfredConfig(vault_path=vault_path, data_dir=tmp_path / "data")
    state = StateStore(tmp_path / "state.json")
    state.load()
    return CuratorDaemon(cfg, state, asyncio.Queue())


def _drop(daemon: CuratorDaemon, snapshot: str) -> None:
    (daemon.cfg.vault_path / "inbox" / "ecc-instincts.md").write_text(
        "---\ntype: reference\nstatus: active\nsource: ecc-procedural-learning\n"
        "source_key: ecc-instincts\ntags: [ecc, instincts]\n---\n"
        f"{_MARKER}\n\n# ECC Procedural Instincts\n\n{snapshot}\n",
        encoding="utf-8",
    )


def _classify_as(monkeypatch, *types: str) -> list[str]:
    """The model's type for an unknown `reference` drop, one per sync."""
    calls: list[str] = []
    answers = iter(types)

    def _complete_json(*a, **kw):
        calls.append("classify")
        return {"type": next(answers), "name": "ecc-procedural-instincts"}

    monkeypatch.setattr("alfred.daemons.curator.complete_json", _complete_json)
    return calls


def _feed_records(vault_path: Path) -> list[str]:
    return sorted(
        str(p.relative_to(vault_path))
        for p in vault_path.glob("*/ecc-procedural-instincts*.md")
        if p.parent.name != "inbox"
    )


def test_second_sync_updates_the_record_instead_of_a_suffixed_copy(tmp_path, monkeypatch):
    daemon = _make_daemon(tmp_path)
    classify_calls = _classify_as(monkeypatch, "note", "note")

    _drop(daemon, "snapshot one")
    asyncio.run(daemon._process_inbox())
    _drop(daemon, "snapshot two")
    asyncio.run(daemon._process_inbox())

    records = _feed_records(daemon.cfg.vault_path)
    assert records == ["note/ecc-procedural-instincts.md"], records
    post = frontmatter.load(str(daemon.cfg.vault_path / records[0]))
    assert "snapshot two" in post.content and "snapshot one" not in post.content
    assert post.metadata["source_key"] == "ecc-instincts"
    assert len(classify_calls) == 1, "an update needs no classification"


def test_sync_classified_as_another_type_still_updates_the_same_record(tmp_path, monkeypatch):
    """The live failure mode: each sync's type came back different."""
    daemon = _make_daemon(tmp_path)
    _classify_as(monkeypatch, "note", "process", "task")

    for n in range(3):
        _drop(daemon, f"snapshot {n}")
        asyncio.run(daemon._process_inbox())

    assert _feed_records(daemon.cfg.vault_path) == ["note/ecc-procedural-instincts.md"]


def test_existing_copies_resolve_to_the_stamped_one(tmp_path, monkeypatch):
    """A vault that already holds copies: the drop updates the one stamped
    with source_key in frontmatter, whatever directory it sits in."""
    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    for rel, fm in (
        ("note/ecc-procedural-instincts.md", "type: note\n"),
        ("drafts/ecc-procedural-instincts-838.md", "type: script\nsource_key: ecc-instincts\n"),
        ("process/ecc-procedural-instincts.md", "type: process\n"),
    ):
        (vault_path / rel).parent.mkdir(exist_ok=True)
        (vault_path / rel).write_text(f"---\n{fm}---\n{_MARKER}\n\nold\n", encoding="utf-8")
    _classify_as(monkeypatch)  # must not be called

    _drop(daemon, "fresh snapshot")
    asyncio.run(daemon._process_inbox())

    assert "fresh snapshot" in (vault_path / "drafts/ecc-procedural-instincts-838.md").read_text()
    assert "fresh snapshot" not in (vault_path / "note/ecc-procedural-instincts.md").read_text()
    assert len(_feed_records(vault_path)) == 3  # nothing new minted


def test_same_name_without_the_key_is_not_taken_over(tmp_path):
    from alfred.daemons.curator import _find_feed_record

    vault_path = tmp_path / "vault"
    (vault_path / "note").mkdir(parents=True)
    (vault_path / "note" / "ecc-procedural-instincts-notes.md").write_text(
        "---\ntype: note\n---\nMy own notes about instincts.\n", encoding="utf-8"
    )

    assert _find_feed_record(vault_path, "ecc-instincts", "ecc-procedural-instincts", []) is None


def test_ignored_directories_are_not_searched(tmp_path):
    from alfred.daemons.curator import _find_feed_record

    vault_path = tmp_path / "vault"
    (vault_path / "_archived").mkdir(parents=True)
    (vault_path / "_archived" / "ecc-procedural-instincts.md").write_text(
        f"---\nsource_key: ecc-instincts\n---\n{_MARKER}\n", encoding="utf-8"
    )

    assert _find_feed_record(
        vault_path, "ecc-instincts", "ecc-procedural-instincts", ["_archived"]
    ) is None


def test_a_drop_is_never_its_own_record(tmp_path):
    """With the default ignore_dirs (inbox/processed only), a feed whose drop
    is named after its slug would otherwise match itself in inbox/."""
    from alfred.daemons.curator import _find_feed_record

    vault_path = tmp_path / "vault"
    (vault_path / "inbox").mkdir(parents=True)
    (vault_path / "inbox" / "ecc-procedural-instincts.md").write_text(
        f"---\nsource_key: ecc-instincts\n---\n{_MARKER}\n", encoding="utf-8"
    )

    assert _find_feed_record(
        vault_path, "ecc-instincts", "ecc-procedural-instincts", ["inbox/processed"]
    ) is None
