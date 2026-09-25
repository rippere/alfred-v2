"""`<!-- alfred:source X -->` is a provenance tag, not a record identity.

About 20 producers stamp it on their drops as a category: claude_code_hook
(process-session.sh, pre-compact.sh), retro_ingest, claude_code_conversation,
crm-flywheel, project-sync and more. Thousands of different sessions share one
key, and many share a title prefix ("Brief me on latest work", the
non-interactive-mode preamble). Twice now (36fe293, then 9d430b3) the curator
learned to read the marker as "this drop replaces the record with the same
key and slug", and different sessions overwrote each other. Both were
reverted. These tests keep every such drop its own record.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from alfred.config import AlfredConfig
from alfred.daemons.curator import CuratorDaemon
from alfred.store.state import StateStore


def _make_daemon(tmp_path: Path) -> CuratorDaemon:
    vault_path = tmp_path / "vault"
    (vault_path / "inbox").mkdir(parents=True)
    cfg = AlfredConfig(vault_path=vault_path, data_dir=tmp_path / "data")
    # The live ignore list (config-base.yaml), not the dataclass default.
    cfg.ignore_dirs = ["inbox", "wiki", "_archived", "_templates", "_bases", ".obsidian"]
    state = StateStore(tmp_path / "state.json")
    state.load()
    return CuratorDaemon(cfg, state, asyncio.Queue())


def _session_drop(daemon: CuratorDaemon, sid: str, title: str, body: str, *, key: str) -> None:
    """process-session.sh's drop shape: frontmatter type session, then the marker."""
    (daemon.cfg.vault_path / "inbox" / f"session-rippere-2026-09-24-{sid}.md").write_text(
        "---\ntype: session\nstatus: active\nproject: \"\"\n"
        f"session_id: \"{sid}0000\"\nturns: 3\ncreated: \"2026-09-24\"\ntags: []\n---\n"
        f"<!-- alfred:source {key} -->\n\n# {title}\n\n{body}\n",
        encoding="utf-8",
    )
    asyncio.run(daemon._process_inbox())


def _records(vault_path: Path) -> dict[str, str]:
    return {
        str(p.relative_to(vault_path)): p.read_text(encoding="utf-8")
        for p in vault_path.glob("*/*.md")
        if p.parent.name != "inbox"
    }


def test_sessions_sharing_a_provenance_key_and_title_prefix_stay_separate(tmp_path, monkeypatch):
    daemon = _make_daemon(tmp_path)
    monkeypatch.setattr(
        "alfred.daemons.curator.complete_json",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("typed drops need no LLM")),
    )

    _session_drop(daemon, "aaaa1111", "Brief me on latest work",
                  "SESSION-A decided to ship the ledger fix.", key="claude_code_hook")
    _session_drop(daemon, "bbbb2222", "Brief me on latest work",
                  "SESSION-B planned the Scout bobber wiring.", key="claude_code_hook")
    _session_drop(daemon, "cccc3333", "Brief me",
                  "SESSION-C was a short one.", key="claude_code_hook")

    records = _records(daemon.cfg.vault_path)
    session_records = sorted(r for r in records if r.startswith("session/"))
    assert len(session_records) == 3, session_records
    for marker in ("SESSION-A", "SESSION-B", "SESSION-C"):
        holders = [r for r, text in records.items() if marker in text]
        assert len(holders) == 1, f"{marker} is in {holders}"
    # Each record holds exactly one session's body: nothing was replaced.
    for rel in session_records:
        assert sum(m in records[rel] for m in ("SESSION-A", "SESSION-B", "SESSION-C")) == 1, rel
    # Nothing is stamped as if it were a feed identity.
    assert not any("source_key" in text for text in records.values())
    # Every raw drop is archived, none left behind or dropped as a duplicate.
    assert not list((daemon.cfg.vault_path / "inbox").glob("*.md"))
    assert len(list((daemon.cfg.vault_path / "inbox" / "processed").glob("*.md"))) == 3


def test_classified_drops_sharing_a_key_do_not_overwrite_an_existing_record(tmp_path, monkeypatch):
    """The retro_ingest shape: no frontmatter type, the same preamble as title,
    and an older record already in the vault carrying the same marker."""
    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    title = "IMPORTANT you are running in non-interactive print mode you must"
    slug = "important-you-are-running-in-non-interactive-print-mode-you-must"
    (vault_path / "session").mkdir()
    old = vault_path / "session" / f"{slug}.md"
    old.write_text(
        "---\ntype: session\nname: old\n---\n<!-- alfred:source retro_ingest -->\n\n"
        f"# {title}\n\nOLD-SESSION body from last month.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "alfred.daemons.curator.complete_json",
        lambda *a, **kw: {"type": "session", "name": slug},
    )

    for n, body in enumerate(("NEW-ONE audited the ledger.", "NEW-TWO fixed the reaper.")):
        (vault_path / "inbox" / f"retro-{n}.md").write_text(
            f"<!-- alfred:source retro_ingest -->\n\n# {title}\n\n{body}\n", encoding="utf-8"
        )
        asyncio.run(daemon._process_inbox())

    assert "OLD-SESSION body" in old.read_text(encoding="utf-8"), "the older record was overwritten"
    records = _records(vault_path)
    for marker in ("OLD-SESSION", "NEW-ONE", "NEW-TWO"):
        holders = [r for r, text in records.items() if marker in text]
        assert len(holders) == 1, f"{marker} is in {holders}"
