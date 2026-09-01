"""Write-path coverage for CuratorDaemon._ingest_file / _process_inbox.

Curator is the only daemon that writes new vault records from untrusted
inbox input (LLM classification + frontmatter), and it had zero test
coverage before this file. These tests pin the behaviors that make the
write path safe to run unattended: known-type fast path, legacy type
aliasing, no-classification skip (file stays in inbox, not silently
dropped), duplicate-slug fallback/dedup, and processed-state bookkeeping.
"""
from __future__ import annotations

import asyncio

import frontmatter

from alfred.config import AlfredConfig
from alfred.daemons.curator import CuratorDaemon
from alfred.store.state import StateStore


def _make_daemon(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    vault_path = tmp_path / "vault"
    (vault_path / "inbox").mkdir(parents=True)
    cfg = AlfredConfig(vault_path=vault_path, data_dir=tmp_path / "data")
    state = StateStore(tmp_path / "state.json", cfg=cfg)
    state.load()
    daemon = CuratorDaemon(cfg=cfg, state=state, events=asyncio.Queue())
    return daemon, vault_path


def _write_inbox(vault_path, name: str, fm: dict, body: str = "Some body text.\n"):
    fp = vault_path / "inbox" / name
    post = frontmatter.Post(body, **fm)
    fp.write_text(frontmatter.dumps(post) + "\n", encoding="utf-8")
    return fp


def test_ingest_known_type_creates_record_and_moves_to_processed(tmp_path, monkeypatch):
    daemon, vault_path = _make_daemon(tmp_path, monkeypatch)
    inbox_file = _write_inbox(vault_path, "note1.md", {"type": "note", "name": "My Note"})
    processed_dir = vault_path / "inbox" / "processed"
    processed_dir.mkdir(exist_ok=True)

    ingested = asyncio.run(daemon._ingest_file(inbox_file, processed_dir))

    assert ingested is True
    created = vault_path / "note" / "my-note.md"
    assert created.exists()
    post = frontmatter.load(str(created))
    assert post.metadata["type"] == "note"
    assert not inbox_file.exists()
    assert (processed_dir / "note1.md").exists()


def test_ingest_legacy_type_alias_normalizes_to_session(tmp_path, monkeypatch):
    daemon, vault_path = _make_daemon(tmp_path, monkeypatch)
    inbox_file = _write_inbox(vault_path, "conv1.md", {"type": "conversation", "name": "Old Chat"})
    processed_dir = vault_path / "inbox" / "processed"
    processed_dir.mkdir(exist_ok=True)

    ingested = asyncio.run(daemon._ingest_file(inbox_file, processed_dir))

    assert ingested is True
    created = vault_path / "session" / "old-chat.md"
    assert created.exists()
    assert frontmatter.load(str(created)).metadata["type"] == "session"


def test_ingest_unknown_type_without_api_key_skips_and_leaves_file(tmp_path, monkeypatch):
    daemon, vault_path = _make_daemon(tmp_path, monkeypatch)
    # No `type` field and no ANTHROPIC_API_KEY -> _classify short-circuits to None.
    inbox_file = _write_inbox(vault_path, "raw1.md", {}, body="Just some raw notes.\n")
    processed_dir = vault_path / "inbox" / "processed"
    processed_dir.mkdir(exist_ok=True)

    ingested = asyncio.run(daemon._ingest_file(inbox_file, processed_dir))

    assert ingested is False
    # A skipped file must stay in the inbox -- silently dropping unclassifiable
    # input would lose data with no record it ever existed.
    assert inbox_file.exists()
    assert not any(processed_dir.iterdir())


def test_ingest_duplicate_slug_retries_with_session_id_suffix(tmp_path, monkeypatch):
    daemon, vault_path = _make_daemon(tmp_path, monkeypatch)
    (vault_path / "note").mkdir(parents=True)
    (vault_path / "note" / "dup.md").write_text(
        frontmatter.dumps(frontmatter.Post("existing", type="note", name="dup")) + "\n"
    )

    inbox_file = _write_inbox(
        vault_path, "dup2.md",
        {"type": "note", "name": "dup", "session_id": "abcdef1234567890"},
    )
    processed_dir = vault_path / "inbox" / "processed"
    processed_dir.mkdir(exist_ok=True)

    ingested = asyncio.run(daemon._ingest_file(inbox_file, processed_dir))

    assert ingested is True
    fallback = vault_path / "note" / "dup-abcdef12.md"
    assert fallback.exists()
    assert not inbox_file.exists()


def test_ingest_full_duplicate_skips_write_but_marks_processed(tmp_path, monkeypatch):
    """Both the primary slug and the session_id-suffixed fallback already exist --
    curator must not raise, and must still move the file out of inbox so it
    isn't retried forever on every poll."""
    daemon, vault_path = _make_daemon(tmp_path, monkeypatch)
    (vault_path / "note").mkdir(parents=True)
    (vault_path / "note" / "dup.md").write_text(
        frontmatter.dumps(frontmatter.Post("existing", type="note", name="dup")) + "\n"
    )
    (vault_path / "note" / "dup-abcdef12.md").write_text(
        frontmatter.dumps(frontmatter.Post("existing2", type="note", name="dup")) + "\n"
    )

    inbox_file = _write_inbox(
        vault_path, "dup3.md",
        {"type": "note", "name": "dup", "session_id": "abcdef1234567890"},
    )
    processed_dir = vault_path / "inbox" / "processed"
    processed_dir.mkdir(exist_ok=True)

    ingested = asyncio.run(daemon._ingest_file(inbox_file, processed_dir))

    assert ingested is True
    assert not inbox_file.exists()
    assert (processed_dir / "dup3.md").exists()


def test_ingest_malformed_frontmatter_ingests_raw_text_without_raising(tmp_path, monkeypatch):
    daemon, vault_path = _make_daemon(tmp_path, monkeypatch)
    processed_dir = vault_path / "inbox" / "processed"
    processed_dir.mkdir(exist_ok=True)
    inbox_file = vault_path / "inbox" / "bad.md"
    # Unterminated frontmatter delimiter -- python-frontmatter raises on this.
    inbox_file.write_text("---\ntype: [unterminated\nSome body\n", encoding="utf-8")

    # No API key -> classification returns None -> skip, but must not raise.
    ingested = asyncio.run(daemon._ingest_file(inbox_file, processed_dir))

    assert ingested is False
    assert inbox_file.exists()


def test_process_inbox_skips_already_processed_files(tmp_path, monkeypatch):
    daemon, vault_path = _make_daemon(tmp_path, monkeypatch)
    _write_inbox(vault_path, "seen.md", {"type": "note", "name": "Seen"})
    rel = "inbox/seen.md"
    daemon.state.state.curator_processed[rel] = "2026-01-01T00:00:00+00:00"

    asyncio.run(daemon._process_inbox())

    # Already-marked-processed files must not be re-ingested or moved.
    assert (vault_path / "inbox" / "seen.md").exists()
    assert not (vault_path / "note" / "seen.md").exists()


def test_process_inbox_ingests_new_file_and_records_state(tmp_path, monkeypatch):
    daemon, vault_path = _make_daemon(tmp_path, monkeypatch)
    _write_inbox(vault_path, "fresh.md", {"type": "note", "name": "Fresh"})

    asyncio.run(daemon._process_inbox())

    assert (vault_path / "note" / "fresh.md").exists()
    assert "inbox/fresh.md" in daemon.state.state.curator_processed
    assert not (vault_path / "inbox" / "fresh.md").exists()
