"""Distiller: a VaultError on topic append must be observable, not silently swallowed.

Regression coverage for the per-learning loop inside `_distill_file`. Because
that loop runs after a successful, already-billed Anthropic API call, a
`VaultError` from `vault_append_to_topic` (malformed YAML in the target topic
file, a path-traversal rejection, lock contention, a race with a concurrent
janitor delete, ...) must not cause the extracted learning to vanish with zero
trace. It should be logged, counted, and must not abort the remaining
learnings in the same call.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from structlog.testing import capture_logs

from alfred.config import AlfredConfig
from alfred.core.vault_ops import VaultError
from alfred.daemons.distiller import DistillerDaemon
from alfred.store.state import StateStore


class _FakeUsage:
    input_tokens = 10
    output_tokens = 5
    cache_read_input_tokens = 0


class _FakeResponse:
    def __init__(self, payload: list[dict]) -> None:
        self.usage = _FakeUsage()
        self.content = [type("Block", (), {"text": json.dumps(payload)})()]


class _FakeMessages:
    def __init__(self, payload: list[dict]) -> None:
        self._payload = payload

    def create(self, **kwargs):
        return _FakeResponse(self._payload)


class _FakeClient:
    def __init__(self, payload: list[dict]) -> None:
        self.messages = _FakeMessages(payload)


def _make_daemon(tmp_path: Path) -> DistillerDaemon:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    cfg = AlfredConfig(vault_path=vault_path, data_dir=tmp_path / "data")
    state = StateStore(tmp_path / "state.json")
    state.load()
    events: asyncio.Queue = asyncio.Queue()
    return DistillerDaemon(cfg, state, events)


def test_topic_append_failure_is_logged_counted_and_does_not_abort(tmp_path, monkeypatch):
    daemon = _make_daemon(tmp_path)

    long_body = "x" * 300  # clears MIN_BODY_LEN
    monkeypatch.setattr(
        "alfred.daemons.distiller.vault_read",
        lambda vault_path, rel_path: {
            "path": rel_path,
            "frontmatter": {"type": "note"},
            "body": long_body,
        },
    )

    learnings = [
        {"title": "first-learning-fails", "body": "insight one", "tags": ["misc"]},
        {"title": "second-learning-succeeds", "body": "insight two", "tags": ["misc"]},
    ]
    monkeypatch.setattr(
        "alfred.daemons.distiller.complete",
        lambda *a, **kw: json.dumps(learnings),
    )

    calls: list[str] = []

    def _fake_append_to_topic(vault_path, topic_slug, title, body_text, tags=None, source=None):
        calls.append(title)
        if title == "first-learning-fails":
            raise VaultError("malformed YAML in topic/misc.md")
        return {"path": f"topic/{topic_slug}.md"}

    monkeypatch.setattr(
        "alfred.daemons.distiller.vault_append_to_topic",
        _fake_append_to_topic,
    )

    with capture_logs() as logs:
        created = asyncio.run(daemon._distill_file(daemon.cfg.vault_path, "inbox/note.md"))

    # (b) processing continued to the second learning after the first raised.
    assert calls == ["first-learning-fails", "second-learning-succeeds"]
    # Only the successful append counts as "created".
    assert created == 1

    # (a) the warning log fired with the expected fields.
    warnings = [e for e in logs if e.get("log_level") == "warning"
                and e.get("event") == "distiller.topic_append_failed"]
    assert len(warnings) == 1
    assert warnings[0]["path"] == "inbox/note.md"
    assert warnings[0]["title"] == "first-learning-fails"
    assert "malformed YAML" in warnings[0]["error"]

    # (c) the failure is observable via the counter, not just absent from output.
    assert daemon.failed_appends_this_tick == 1
    # The daemon did not crash — no error-level log or exception propagated.
    assert not [e for e in logs if e.get("log_level") == "error"]


def test_topic_append_success_does_not_increment_failure_counter(tmp_path, monkeypatch):
    daemon = _make_daemon(tmp_path)

    long_body = "y" * 300
    monkeypatch.setattr(
        "alfred.daemons.distiller.vault_read",
        lambda vault_path, rel_path: {
            "path": rel_path,
            "frontmatter": {"type": "note"},
            "body": long_body,
        },
    )
    learnings = [{"title": "clean-learning", "body": "insight", "tags": ["misc"]}]
    monkeypatch.setattr("alfred.daemons.distiller.complete", lambda *a, **kw: json.dumps(learnings))
    monkeypatch.setattr(
        "alfred.daemons.distiller.vault_append_to_topic",
        lambda vault_path, topic_slug, title, body_text, tags=None, source=None: {
            "path": f"topic/{topic_slug}.md"
        },
    )

    created = asyncio.run(daemon._distill_file(daemon.cfg.vault_path, "inbox/note.md"))

    assert created == 1
    assert daemon.failed_appends_this_tick == 0


def test_tick_normal_path_distills_stale_file_end_to_end(tmp_path, monkeypatch):
    """Exercise the actual `tick()` APScheduler entry point end to end:
    `tick()` -> `_distill_sweep()` -> `_distill_file()`, with only the
    Anthropic client and vault I/O mocked, against a real state.files entry
    marked stale (never distilled)."""
    from alfred.core.models import FileState

    daemon = _make_daemon(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    long_body = "z" * 300
    monkeypatch.setattr(
        "alfred.daemons.distiller.vault_read",
        lambda vault_path, rel_path: {
            "path": rel_path,
            "frontmatter": {"type": "note"},
            "body": long_body,
        },
    )
    learnings = [{"title": "tick-learning", "body": "insight from tick", "tags": ["misc"]}]
    monkeypatch.setattr("alfred.daemons.distiller.complete", lambda *a, **kw: json.dumps(learnings))

    appended: list[str] = []

    def _fake_append_to_topic(vault_path, topic_slug, title, body_text, tags=None, source=None):
        appended.append(title)
        return {"path": f"topic/{topic_slug}.md"}

    monkeypatch.setattr("alfred.daemons.distiller.vault_append_to_topic", _fake_append_to_topic)

    daemon.state.state.files["inbox/note.md"] = FileState(md5="abc123")

    asyncio.run(daemon.tick())

    assert appended == ["tick-learning"]
    fs = daemon.state.state.files["inbox/note.md"]
    assert fs.last_distilled  # stamped after successful distill
    assert len(daemon.state.state.distiller_runs) == 1
    assert daemon.state.state.distiller_runs[0]["learn_records_created"] == 1


def test_tick_exception_is_caught_and_logged_not_propagated(tmp_path, monkeypatch):
    """`tick()` must swallow any exception raised inside `_distill_sweep()`
    and log it rather than letting it propagate."""
    daemon = _make_daemon(tmp_path)

    async def _boom() -> None:
        raise RuntimeError("simulated distiller sweep failure")

    monkeypatch.setattr(daemon, "_distill_sweep", _boom)

    with capture_logs() as logs:
        asyncio.run(daemon.tick())  # must not raise

    errors = [e for e in logs if e.get("log_level") == "error"
              and e.get("event") == "distiller.tick_error"]
    assert len(errors) == 1
    assert "simulated distiller sweep failure" in errors[0]["error"]


def _stub_vault(monkeypatch, appended: list[str] | None = None) -> None:
    monkeypatch.setattr(
        "alfred.daemons.distiller.vault_read",
        lambda vault_path, rel_path: {
            "path": rel_path,
            "frontmatter": {"type": "note"},
            "body": "w" * 300,
        },
    )

    def _fake_append_to_topic(vault_path, topic_slug, title, body_text, tags=None, source=None):
        if appended is not None:
            appended.append(title)
        return {"path": f"topic/{topic_slug}.md"}

    monkeypatch.setattr("alfred.daemons.distiller.vault_append_to_topic", _fake_append_to_topic)


def test_items_object_reply_is_unwrapped(tmp_path, monkeypatch):
    """json_mode constrains the reply to a JSON object, so the model answers
    {"items": [...]} — the old `isinstance(list)` check scored every such
    reply as learned=0."""
    daemon = _make_daemon(tmp_path)
    appended: list[str] = []
    _stub_vault(monkeypatch, appended)
    reply = {"items": [{"title": "wrapped-learning", "body": "insight", "tags": ["misc"]}]}
    monkeypatch.setattr("alfred.daemons.distiller.complete", lambda *a, **kw: json.dumps(reply))

    created = asyncio.run(daemon._distill_file(daemon.cfg.vault_path, "inbox/note.md"))

    assert created == 1
    assert appended == ["wrapped-learning"]


def test_unreadable_json_shape_is_counted_not_silent(tmp_path, monkeypatch):
    daemon = _make_daemon(tmp_path)
    _stub_vault(monkeypatch)
    reply = {"title": "a-lone-object", "body": "no items key", "tags": []}
    monkeypatch.setattr("alfred.daemons.distiller.complete", lambda *a, **kw: json.dumps(reply))

    with capture_logs() as logs:
        created = asyncio.run(daemon._distill_file(daemon.cfg.vault_path, "inbox/note.md"))

    assert created == 0
    assert [e for e in logs if e.get("event") == "distiller.unexpected_json_shape"]


def test_sweep_spanning_another_save_is_recorded_on_disk(tmp_path, monkeypatch):
    """The live failure: every sweep ran through the 5-min periodic save, so
    its stamps and run entry landed in a detached copy of the state. After a
    restart (fresh StateStore) the last run must be this sweep, which is what
    runner.py's catch-up reads, and no file may be stale again."""
    from alfred.core.models import FileState

    daemon = _make_daemon(tmp_path)
    _stub_vault(monkeypatch)
    reply = {"items": [{"title": "t", "body": "b", "tags": ["misc"]}]}
    monkeypatch.setattr("alfred.daemons.distiller.complete", lambda *a, **kw: json.dumps(reply))
    for name in ("a.md", "b.md", "c.md"):
        daemon.state.state.files[name] = FileState(md5=name)
    daemon.state.save()

    async def _sleep_while_another_job_saves(_seconds: float) -> None:
        daemon.state.save()

    monkeypatch.setattr("alfred.daemons.distiller.asyncio.sleep", _sleep_while_another_job_saves)

    asyncio.run(daemon.tick())

    restarted = StateStore(daemon.state.path)
    restarted.load()
    runs = restarted.state.distiller_runs
    assert len(runs) == 1, f"sweep not recorded on disk: {runs!r}"
    assert runs[0]["files_scanned"] == 3
    assert runs[0]["learn_records_created"] == 3
    stale = [p for p, fs in restarted.state.files.items() if not fs.last_distilled]
    assert not stale, f"last_distilled lost for {stale}"


def test_sweep_with_nothing_stale_still_records_a_run(tmp_path):
    """A quiet vault must not look overdue to the startup catch-up."""
    daemon = _make_daemon(tmp_path)

    asyncio.run(daemon.tick())

    assert len(daemon.state.state.distiller_runs) == 1
    assert daemon.state.state.distiller_runs[0]["files_scanned"] == 0


def test_sweep_deferred_before_any_progress_records_no_run(tmp_path, monkeypatch):
    """Backend down on the first file: nothing was done, so the catch-up must
    still see the vault as overdue and retry after the next start."""
    from alfred.core.local_llm import LocalLLMUnavailable
    from alfred.core.models import FileState

    daemon = _make_daemon(tmp_path)
    _stub_vault(monkeypatch)

    def _down(*a, **kw):
        raise LocalLLMUnavailable("connection refused")

    monkeypatch.setattr("alfred.daemons.distiller.complete", _down)
    daemon.state.state.files["a.md"] = FileState(md5="a")

    asyncio.run(daemon.tick())

    assert daemon.state.state.distiller_runs == []
    assert not daemon.state.state.files["a.md"].last_distilled
