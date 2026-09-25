"""Behavioral coverage for ConsolidatorDaemon's public tick() entry points
(label_tick / stubs_tick) — previously untested beyond APScheduler job
registration metadata (tests/test_scheduler_jobs.py).

Covers one normal-tick path per responsibility exercised here:
  - label_tick(): a real cluster gets labeled through the actual Ollama ->
    Anthropic fallback chain, with the Ollama HTTP call mocked to fail (no
    local Ollama in test env) and the Anthropic client mocked to succeed —
    proving the fallback, not just the happy path of either provider alone.
  - stubs_tick(): a real person/org vault record gets a wiki stub page
    created via real (tmp_path) vault I/O — no LLM involved in stub
    creation.
And one exception-handling path: label_tick() swallowing and logging an
internal exception rather than propagating it, matching the contract already
proven for the other daemons' tick() wrappers.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from structlog.testing import capture_logs

from alfred.config import AlfredConfig
from alfred.core.models import ClusterState, FileState
from alfred.daemons.consolidator import ConsolidatorDaemon
from alfred.store.state import StateStore


def _make_daemon(tmp_path: Path) -> ConsolidatorDaemon:
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    cfg = AlfredConfig(vault_path=vault_path, data_dir=tmp_path / "data")
    state = StateStore(tmp_path / "state.json")
    state.load()
    events: asyncio.Queue = asyncio.Queue()
    return ConsolidatorDaemon(cfg, state, events)


class _FakeUsage:
    input_tokens = 8
    output_tokens = 4
    cache_read_input_tokens = 0


class _FakeAnthropicResponse:
    def __init__(self, text: str) -> None:
        self.usage = _FakeUsage()
        self.content = [type("Block", (), {"text": text})()]


class _FakeAnthropicMessages:
    def __init__(self, text: str) -> None:
        self._text = text

    def create(self, **kwargs):
        return _FakeAnthropicResponse(self._text)


class _FakeAnthropicClient:
    def __init__(self, text: str) -> None:
        self.messages = _FakeAnthropicMessages(text)


async def _failing_post(self, url, **kwargs):
    raise httpx.ConnectError("simulated: no local Ollama running")


def test_label_tick_normal_path_labels_cluster_via_local_llm_fallback(
    tmp_path, monkeypatch
):
    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    note_dir = vault_path / "note"
    note_dir.mkdir()
    member_paths = []
    for i in range(3):
        rel = f"note/member-{i}.md"
        (vault_path / "note" / f"member-{i}.md").write_text(
            f"---\ntype: note\nname: member-{i}\n---\nBody {i}.\n", encoding="utf-8"
        )
        member_paths.append(rel)

    daemon.state.state.clusters["semantic_0"] = ClusterState(
        cluster_id=0,
        cluster_type="semantic",
        member_files=member_paths,
    )

    # Ollama (the primary path) is unreachable in the test environment —
    # mock the HTTP call to fail fast rather than actually attempting network I/O.
    monkeypatch.setattr(httpx.AsyncClient, "post", _failing_post)
    # Fallback 1 is now the local backend, bound at import time in the
    # consolidator's namespace — patch it there.
    monkeypatch.setattr(
        "alfred.daemons.consolidator.complete",
        lambda *a, **kw: "fallback cluster label",
    )

    asyncio.run(daemon.label_tick())

    cluster = daemon.state.state.clusters["semantic_0"]
    assert cluster.label == ["fallback cluster label"]
    assert cluster.last_labeled


def test_label_tick_exception_is_caught_and_logged_not_propagated(tmp_path, monkeypatch):
    """label_tick() must swallow any exception raised inside _label_pass()
    and log it rather than letting it propagate — this is what makes it safe
    to register directly as an APScheduler job function."""
    daemon = _make_daemon(tmp_path)

    async def _boom(vault_path) -> None:
        raise RuntimeError("simulated consolidator label pass failure")

    monkeypatch.setattr(daemon, "_label_pass", _boom)

    with capture_logs() as logs:
        asyncio.run(daemon.label_tick())  # must not raise

    errors = [e for e in logs if e.get("log_level") == "error"
              and e.get("event") == "consolidator.label_tick_error"]
    assert len(errors) == 1
    assert "simulated consolidator label pass failure" in errors[0]["error"]


def test_stubs_tick_normal_path_creates_wiki_page(tmp_path):
    """stubs_tick() (the real APScheduler job function) must create a wiki
    stub page for a person/org record via real vault I/O — no LLM needed."""
    daemon = _make_daemon(tmp_path)
    vault_path = daemon.cfg.vault_path
    person_dir = vault_path / "person"
    person_dir.mkdir()
    rel_path = "person/jane-doe.md"
    (vault_path / "person" / "jane-doe.md").write_text(
        "---\ntype: person\nname: Jane Doe\n---\nColleague.\n", encoding="utf-8"
    )
    daemon.state.state.files[rel_path] = FileState(md5="abc123")

    asyncio.run(daemon.stubs_tick())

    assert "jane doe" in daemon.state.state.wiki_pages
    page = daemon.state.state.wiki_pages["jane doe"]
    assert page.rel_path == "wiki/jane-doe.md"
    assert (vault_path / "wiki" / "jane-doe.md").exists()


# ── Membership-gated labeling and synthesis ────────────────────────────────


class _FakeGenerate:
    """Ollama /api/generate stand-in: counts label and synthesis prompts,
    and can run a hook mid-call (the surveyor reclustering meanwhile)."""

    def __init__(self) -> None:
        self.labels: list[str] = []
        self.syntheses: list[str] = []
        self.during_call = None

    async def post(self, client, url, json=None, **kwargs):
        prompt = json["prompt"]
        if prompt.startswith("You are synthesizing"):
            self.syntheses.append(prompt)
            text = "## Insight\nA synthesis.\n\n## Evidence\nE.\n\n## Implications\nI.\n\n## Applicability\nA."
        else:
            self.labels.append(prompt)
            text = "LABEL: alpha topic\nSUMMARY: One sentence."
        if self.during_call is not None:
            hook, self.during_call = self.during_call, None
            hook()
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={"response": text}, request=request)


def _cluster_daemon(tmp_path, monkeypatch, n_members: int = 3):
    daemon = _make_daemon(tmp_path)
    daemon.cfg.data_dir.mkdir()  # the synthesis pass locks graph.pkl there
    vault_path = daemon.cfg.vault_path
    (vault_path / "note").mkdir()
    members = []
    for i in range(n_members):
        (vault_path / "note" / f"m{i}.md").write_text(
            f"---\ntype: note\nname: m{i}\n---\nHuman-written body {i}.\n", encoding="utf-8"
        )
        members.append(f"note/m{i}.md")
    daemon.state.state.clusters["semantic_0"] = ClusterState(
        cluster_id=0, cluster_type="semantic", member_files=list(members)
    )
    fake = _FakeGenerate()

    async def _post(self, url, **kwargs):
        return await fake.post(self, url, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    monkeypatch.setattr("alfred.daemons.consolidator.asyncio.sleep", _no_sleep)
    return daemon, fake, members


async def _no_sleep(_seconds: float) -> None:
    return None


def _rebuild_cluster(daemon, key: str = "semantic_0") -> None:
    """What reclustering (or any save that detached the pass's objects) used
    to do mid-call: a new ClusterState object for the same members."""
    old = daemon.state.state.clusters[key]
    daemon.state.state.clusters[key] = ClusterState(
        cluster_id=old.cluster_id,
        cluster_type="semantic",
        label=list(old.label),
        member_files=list(old.member_files),
        last_labeled=old.last_labeled,
        consolidated_chunk_id=old.consolidated_chunk_id,
    )


def test_unchanged_cluster_is_not_relabeled_after_a_day(tmp_path, monkeypatch):
    daemon, fake, _ = _cluster_daemon(tmp_path, monkeypatch)

    asyncio.run(daemon.label_tick())
    cluster = daemon.state.state.clusters["semantic_0"]
    assert cluster.label == ["alpha topic"]
    cluster.last_labeled = "2026-01-01T00:00:00+00:00"  # long past the old 24h guard

    asyncio.run(daemon.label_tick())

    assert len(fake.labels) == 1, "an unchanged membership was relabelled"


def test_label_lands_on_the_cluster_rebuilt_mid_call(tmp_path, monkeypatch):
    """The live loop: the surveyor rebuilt the cluster while the model was
    answering, the label went into the old object, and every pass paid for
    the same label again."""
    daemon, fake, _ = _cluster_daemon(tmp_path, monkeypatch)
    fake.during_call = lambda: _rebuild_cluster(daemon)

    asyncio.run(daemon.label_tick())

    assert daemon.state.state.clusters["semantic_0"].label == ["alpha topic"]
    asyncio.run(daemon.label_tick())
    assert len(fake.labels) == 1


def test_changed_membership_is_relabeled(tmp_path, monkeypatch):
    daemon, fake, members = _cluster_daemon(tmp_path, monkeypatch, n_members=4)
    daemon.state.state.clusters["semantic_0"].member_files = members[:3]

    asyncio.run(daemon.label_tick())
    daemon.state.state.clusters["semantic_0"].member_files = members  # a file joined
    asyncio.run(daemon.label_tick())

    assert len(fake.labels) == 2


def test_renumbered_members_keep_their_label_without_a_call(tmp_path, monkeypatch):
    daemon, fake, members = _cluster_daemon(tmp_path, monkeypatch)
    asyncio.run(daemon.label_tick())

    clusters = daemon.state.state.clusters
    # HDBSCAN renumbered: the same members now sit under semantic_7, and
    # semantic_0 carries someone else's (smaller) membership.
    clusters["semantic_7"] = ClusterState(cluster_id=7, member_files=list(members))
    clusters["semantic_0"].member_files = ["note/other.md"]

    asyncio.run(daemon.label_tick())

    assert clusters["semantic_7"].label == ["alpha topic"]
    assert len(fake.labels) == 1


def test_placeholder_label_is_replaced_once_the_model_answers(tmp_path, monkeypatch):
    """Backend paused: the cluster gets a name from its paths, but not a
    stamped one — so it does not keep "note" for good."""
    daemon, fake, _ = _cluster_daemon(tmp_path, monkeypatch)
    monkeypatch.setattr(httpx.AsyncClient, "post", _failing_post)

    def _down(*a, **kw):
        from alfred.core.local_llm import LocalLLMUnavailable

        raise LocalLLMUnavailable("connection refused")

    monkeypatch.setattr("alfred.daemons.consolidator.complete", _down)
    asyncio.run(daemon.label_tick())
    assert daemon.state.state.clusters["semantic_0"].label == ["note"]

    async def _post(self, url, **kwargs):
        return await fake.post(self, url, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)
    asyncio.run(daemon.label_tick())
    assert daemon.state.state.clusters["semantic_0"].label == ["alpha topic"]


def test_unchanged_cluster_is_not_resynthesized_when_its_page_ages(tmp_path, monkeypatch):
    import os
    import time as _time

    daemon, fake, _ = _cluster_daemon(tmp_path, monkeypatch)
    asyncio.run(daemon.label_tick())
    asyncio.run(daemon.synthesis_tick())
    page = daemon.cfg.vault_path / daemon.state.state.clusters["semantic_0"].consolidated_chunk_id
    assert page.exists() and len(fake.syntheses) == 1

    old = _time.time() - 40 * 86400
    os.utime(page, (old, old))
    asyncio.run(daemon.synthesis_tick())

    assert len(fake.syntheses) == 1, "an unchanged cluster was re-synthesized"


def test_synthesis_path_lands_on_the_cluster_rebuilt_mid_call(tmp_path, monkeypatch):
    """The repeat-synthesis loop: the cluster pointed at a >30-day-old page,
    so it was ranked; the new page's path went into an object the surveyor had
    replaced meanwhile; next pass it was ranked again, 40+ times a day."""
    import os
    import time as _time

    daemon, fake, _ = _cluster_daemon(tmp_path, monkeypatch)
    asyncio.run(daemon.label_tick())
    stale = daemon.cfg.vault_path / "synthesis" / "older-name.md"
    stale.parent.mkdir()
    stale.write_text("---\ntype: synthesis\n---\nOld.\n", encoding="utf-8")
    old = _time.time() - 40 * 86400
    os.utime(stale, (old, old))
    daemon.state.state.clusters["semantic_0"].consolidated_chunk_id = "synthesis/older-name.md"
    fake.during_call = lambda: _rebuild_cluster(daemon)

    asyncio.run(daemon.synthesis_tick())
    asyncio.run(daemon.synthesis_tick())

    assert len(fake.syntheses) == 1
    assert daemon.state.state.clusters["semantic_0"].consolidated_chunk_id == "synthesis/alpha-topic.md"


def test_synthesis_prompt_is_capped(tmp_path, monkeypatch):
    """The Spark answers 400 over 32K tokens (input + output), where Ollama
    silently truncated. Eight long notes must not add up past the cap."""
    import re

    daemon, fake, _ = _cluster_daemon(tmp_path, monkeypatch)
    # ~20K estimated tokens each: prose, plus the digit-heavy kind that runs
    # at 1.5 chars per real token.
    entries = [(f"note/big{i}.md", f"big{i}", ("word " * 12_000) + ("1234 " * 2_000)) for i in range(8)]

    asyncio.run(daemon._call_synthesis_llm("alpha topic", entries))

    estimate = len(re.findall(r"[^\W\d_]{1,5}|\d|\S", fake.syntheses[0]))
    assert estimate <= 24_000 + 500, estimate  # cap + the prompt's own wording
    assert fake.syntheses[0].count("[…truncated]") == 8


def test_fit_to_budget_keeps_short_bodies_whole():
    from alfred.daemons.consolidator import _approx_tokens, _fit_to_budget

    short = ("note/s.md", "s", "a short note")
    long_a = ("note/a.md", "a", "alpha " * 5_000)
    long_b = ("note/b.md", "b", "bravo " * 5_000)

    fitted, before = _fit_to_budget([short, long_a, long_b], 1_000)

    assert before > 1_000
    assert fitted[0] == short
    marker = _approx_tokens("\n[…truncated]")
    assert sum(_approx_tokens(body) for _, _, body in fitted) <= 1_000 + 2 * marker
    assert fitted[1][2].endswith("[…truncated]") and fitted[2][2].endswith("[…truncated]")
