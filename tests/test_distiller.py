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
from datetime import UTC, datetime
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


def test_request_too_large_skips_the_file_and_the_sweep_goes_on(tmp_path, monkeypatch):
    """An oversized request (the Spark's 400, or an answer cut at max_tokens)
    gets the same answer next sweep. It must not stop the sweep the way a down
    backend does, and must not leave the file stale to be re-sent tomorrow."""
    from alfred.core.local_llm import LocalLLMRequestTooLarge
    from alfred.core.models import FileState

    daemon = _make_daemon(tmp_path)
    appended: list[str] = []
    _stub_vault(monkeypatch, appended)
    monkeypatch.setattr("alfred.daemons.distiller.asyncio.sleep", _no_sleep)
    calls: list[str] = []

    def _complete(system, user, **kw):
        calls.append(user)
        if "a.md" in user:
            raise LocalLLMRequestTooLarge("400: over the window")
        return json.dumps({"items": [{"title": "from-b", "body": "b", "tags": ["misc"]}]})

    monkeypatch.setattr("alfred.daemons.distiller.complete", _complete)
    for name in ("a.md", "b.md"):
        daemon.state.state.files[name] = FileState(md5=name)

    with capture_logs() as logs:
        asyncio.run(daemon.tick())

    assert len(calls) == 2, "the sweep stopped at the oversized file"
    assert appended == ["from-b"]
    assert daemon.state.state.files["a.md"].last_distilled, "a.md would be re-sent next sweep"
    assert [e for e in logs if e.get("event") == "distiller.request_too_large"
            and e.get("path") == "a.md"]

    calls.clear()
    asyncio.run(daemon.tick())
    assert calls == []


async def _no_sleep(_seconds: float) -> None:
    return None


# ── Per-sweep cap: the {items} fix turns writes on for ~15K stale files ──────


def _stale_files(daemon: DistillerDaemon, names, last_distilled: str = "") -> None:
    from alfred.core.models import FileState

    for name in names:
        daemon.state.state.files[name] = FileState(md5=name, last_distilled=last_distilled)


def test_sweep_distills_at_most_the_cap_and_the_rest_waits(tmp_path, monkeypatch):
    """The main vault had ~15K unstamped files, each worth up to 3 topic
    appends: without a cap the first sweep would write ~45K appends in one
    night. The cap bounds files, LLM calls and appends per sweep."""
    daemon = _make_daemon(tmp_path)
    daemon.cfg.distiller_max_files_per_sweep = 2
    appended: list[str] = []
    _stub_vault(monkeypatch, appended)
    monkeypatch.setattr("alfred.daemons.distiller.asyncio.sleep", _no_sleep)
    calls: list[str] = []

    def _complete(system, user, **kw):
        calls.append(user)
        items = [{"title": f"t{n}", "body": "b", "tags": ["misc"]} for n in range(3)]
        return json.dumps({"items": items})

    monkeypatch.setattr("alfred.daemons.distiller.complete", _complete)
    _stale_files(daemon, [f"f{n}.md" for n in range(5)])

    with capture_logs() as logs:
        asyncio.run(daemon.tick())

    assert len(calls) == 2
    assert len(appended) == 6, "3 appends per file, 2 files"
    stamped = sorted(p for p, fs in daemon.state.state.files.items() if fs.last_distilled)
    assert stamped == ["f0.md", "f1.md"]
    run = daemon.state.state.distiller_runs[-1]
    assert (run["files_scanned"], run["learn_records_created"], run["stale_remaining"]) == (2, 6, 3)
    assert [e for e in logs if e.get("event") == "distiller.sweep_capped"
            and e.get("cap") == 2 and e.get("stale") == 5]

    calls.clear()
    asyncio.run(daemon.tick())
    asyncio.run(daemon.tick())
    assert len(calls) == 3, "the next sweeps take the remaining 2, then 1"
    assert all(fs.last_distilled for fs in daemon.state.state.files.values())
    assert daemon.state.state.distiller_runs[-1]["stale_remaining"] == 0


def test_capped_sweep_takes_never_distilled_then_oldest_first(tmp_path, monkeypatch):
    """With a cap, dict order would re-take the same leading files each time
    they went stale again, and the tail of a 15K-file vault would never be
    reached. Never-distilled files go first, then the oldest stamp."""
    daemon = _make_daemon(tmp_path)
    daemon.cfg.distiller_max_files_per_sweep = 3
    _stub_vault(monkeypatch)
    monkeypatch.setattr("alfred.daemons.distiller.asyncio.sleep", _no_sleep)
    seen: list[str] = []

    def _complete(system, user, **kw):
        seen.append(next(line for line in user.splitlines() if line.startswith("File: "))[6:])
        return json.dumps({"items": []})

    monkeypatch.setattr("alfred.daemons.distiller.complete", _complete)
    _stale_files(daemon, ["stale-newer.md"], "2026-07-01T00:00:00+00:00")
    _stale_files(daemon, ["stale-older.md"], "2026-05-01T00:00:00+00:00")
    _stale_files(daemon, ["fresh.md"], datetime.now(UTC).isoformat())
    _stale_files(daemon, ["never-a.md", "never-b.md"])

    asyncio.run(daemon.tick())

    assert seen == ["never-a.md", "never-b.md", "stale-older.md"]


def test_default_cap_is_200_and_config_sets_it(tmp_path):
    import pytest
    import yaml

    daemon = _make_daemon(tmp_path)
    assert daemon.cfg.distiller_max_files_per_sweep == 200

    base = {"distiller": {"mode": "scheduled", "max_files_per_sweep": 200}}
    (tmp_path / "config-base.yaml").write_text(yaml.safe_dump(base))
    vault = {"vault": {"path": str(tmp_path / "v")}, "data_dir": "./d"}
    path = tmp_path / "config-x.yaml"
    path.write_text(yaml.safe_dump(vault))
    assert AlfredConfig.load(path).distiller_max_files_per_sweep == 200

    path.write_text(yaml.safe_dump({**vault, "distiller": {"max_files_per_sweep": 50}}))
    assert AlfredConfig.load(path).distiller_max_files_per_sweep == 50

    for bad in (0, -1, "lots", True):
        path.write_text(yaml.safe_dump({**vault, "distiller": {"max_files_per_sweep": bad}}))
        with pytest.raises(ValueError, match="max_files_per_sweep"):
            AlfredConfig.load(path)


def test_real_base_config_caps_the_distiller_at_200():
    cfg = AlfredConfig.load(Path(__file__).resolve().parents[1] / "config.yaml")
    assert cfg.distiller_mode == "scheduled"
    assert cfg.distiller_max_files_per_sweep == 200
