"""A local-only vault refuses Ollama cloud models.

The carve-out's URL check keeps requests on loopback, but benderman's Ollama
(0.32.1, cloud features on) forwards a cloud model's prompt to ollama.com. So:

- config-employment.yaml pins its models, and a cloud model named in
  config-base.yaml cannot reach it;
- check_local_only refuses a cloud model *name* (at load and every use);
- before each send (cached a few minutes) the local Ollama is asked via
  /api/show whether the model runs remotely, which catches a cloud model under
  a local alias; the daemon refuses to start on one.

The transport tests drive a real HTTP server on loopback (conftest.FakeOllama).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from alfred.config import AlfredConfig, LocalOnlyViolation, is_ollama_cloud_model
from alfred.core import ollama_guard
from alfred.core.failures import peek_failures, reset_failures
from alfred.core.local_llm import LocalLLMRemoteModel, LocalLLMUnavailable, complete
from alfred.embed.ollama import (
    EmbeddingBackendUnavailable,
    EmbeddingModelRefused,
    OllamaEmbedder,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
QWEN = "orcarouter/Qwen3.8-27B-Uncensored:q5_K_M"
CLOUD_REPLY = {"remote_model": "gpt-oss:120b", "remote_host": "https://ollama.com:443",
               "details": {"format": ""}}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    for name in ("SPARK_API_KEY", "SPARK_BASE_URL", "SPARK_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("alfred.config.SPARK_ENV_PATH", tmp_path / "no-spark-env")
    monkeypatch.setattr("alfred.embed.ollama.asyncio.sleep", _instant_sleep)
    reset_failures()
    yield
    reset_failures()


async def _instant_sleep(_delay):
    return None


def _employment(tmp_path: Path, base_ollama: dict | None = None, drop=(), vault: dict | None = None):
    """The real employment config and base in tmp_path, vault pointed at tmp."""
    base = yaml.safe_load((REPO_ROOT / "config-base.yaml").read_text())
    base["ollama"].update(base_ollama or {})
    (tmp_path / "config-base.yaml").write_text(yaml.safe_dump(base))
    raw = yaml.safe_load((REPO_ROOT / "config-employment.yaml").read_text())
    raw["vault"]["path"] = str(tmp_path / "vault-employment")
    for key in drop:
        raw.pop(key, None)
    for key, value in (vault or {}).items():
        raw[key] = {**raw.get(key, {}), **value}
    path = tmp_path / "config-employment.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


# ── names ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", [
    "gpt-oss:120b-cloud", "glm-4.6:cloud", "embeddinggemma:cloud", "qwen3-coder:480b-cloud",
    "kimi-k2:1t-cloud", "foo-cloud:latest", "foo-cloud", "GPT-OSS:20B-CLOUD",
])
def test_cloud_model_names_are_recognised(name):
    assert is_ollama_cloud_model(name)


@pytest.mark.parametrize("name", [
    QWEN, "nomic-embed-text", "nomic-embed-text:latest", "qwen3:32b", "mistral:latest",
    "cloudy-model:7b", "",
])
def test_local_model_names_are_not(name):
    assert not is_ollama_cloud_model(name)


# ── config: pins and the name check ─────────────────────────────────────────


def test_a_cloud_model_in_the_base_does_not_reach_employment(tmp_path):
    """The boundary probe's case 1: config-base.yaml names cloud models. The
    employment vault keeps its own pinned models and still loads."""
    cfg = AlfredConfig.load(_employment(
        tmp_path, {"llm_model": "gpt-oss:120b-cloud", "embed_model": "embeddinggemma:cloud"},
    ))

    assert cfg.local_only
    assert cfg.llm["model"] == cfg.ollama_llm_model == QWEN
    assert cfg.ollama_embed_model == "nomic-embed-text"


def test_employment_pins_are_todays_fleet_models():
    """Pinning changed nothing: the pinned models are the base's models."""
    base = yaml.safe_load((REPO_ROOT / "config-base.yaml").read_text())["ollama"]
    pinned = yaml.safe_load((REPO_ROOT / "config-employment.yaml").read_text())["ollama"]
    assert pinned["llm_model"] == base["llm_model"] == QWEN
    assert pinned["embed_model"] == base["embed_model"] == "nomic-embed-text"


def test_deleted_model_pins_and_a_cloud_base_fail_the_load(tmp_path):
    """Pins deleted (a bad merge) while the base names a cloud model: the URL
    is still loopback, so only the name check stands between the vault and
    ollama.com. The load refuses."""
    path = _employment(
        tmp_path, {"llm_model": "gpt-oss:120b-cloud", "embed_model": "embeddinggemma:cloud"},
        drop=("ollama",),
    )
    with pytest.raises(LocalOnlyViolation, match="cloud model"):
        AlfredConfig.load(path)


@pytest.mark.parametrize("vault", [
    pytest.param({"llm": {"model": "gpt-oss:120b-cloud"}}, id="llm.model"),
    pytest.param({"ollama": {"llm_model": "glm-4.6:cloud"}}, id="ollama.llm_model"),
    pytest.param({"ollama": {"embed_model": "embeddinggemma:cloud"}}, id="ollama.embed_model"),
])
def test_local_only_refuses_a_cloud_model_named_in_its_own_file(tmp_path, vault):
    with pytest.raises(LocalOnlyViolation, match="cloud model"):
        AlfredConfig.load(_employment(tmp_path, vault=vault))


def test_a_runtime_model_change_is_refused_at_the_call_site(tmp_path):
    cfg = AlfredConfig.load(_employment(tmp_path))
    cfg.ollama_llm_model = "gpt-oss:120b-cloud"
    with pytest.raises(LocalOnlyViolation):
        _ = cfg.llm

    cfg = AlfredConfig.load(_employment(tmp_path))
    cfg.ollama_embed_model = "embeddinggemma:cloud"
    with pytest.raises(LocalOnlyViolation):
        _ = cfg.embed_base_url


def test_other_vaults_may_still_name_any_model(tmp_path):
    """The name check is the carve-out's, not a fleet rule."""
    base = yaml.safe_load((REPO_ROOT / "config-base.yaml").read_text())
    base["ollama"]["llm_model"] = "gpt-oss:120b-cloud"
    (tmp_path / "config-base.yaml").write_text(yaml.safe_dump(base))
    raw = yaml.safe_load((REPO_ROOT / "config-personal.yaml").read_text())
    raw["vault"]["path"] = str(tmp_path / "vault-personal")
    (tmp_path / "config-personal.yaml").write_text(yaml.safe_dump(raw))

    assert AlfredConfig.load(tmp_path / "config-personal.yaml").llm["model"] == "gpt-oss:120b-cloud"


# ── completions: ask /api/show before sending ───────────────────────────────


def test_local_only_completion_refuses_a_remote_model_and_sends_nothing(fake_ollama):
    """A cloud model under a local-looking alias: the name passes, /api/show
    says remote_host, and the prompt is never sent."""
    ollama = fake_ollama()
    ollama.show_reply = CLOUD_REPLY

    with pytest.raises(LocalLLMRemoteModel) as err:
        complete("s", "SYNTHETIC PROMPT", base_url=ollama.url, model="qwen:local", local_only=True)

    assert isinstance(err.value, LocalLLMUnavailable)  # callers defer, nothing is dropped
    assert isinstance(err.value, LocalOnlyViolation)
    assert ollama.paths == ["/api/show"]
    assert ollama.requests[0][1] == {"model": "qwen:local"}
    assert peek_failures().get("local_llm.remote_model_refused") == 1


def test_local_only_completion_checks_once_then_sends(fake_ollama):
    ollama = fake_ollama()

    assert complete("s", "u", base_url=ollama.url, model=QWEN, local_only=True) == "ok"
    assert complete("s", "u", base_url=ollama.url, model=QWEN, local_only=True) == "ok"

    # Cached for VERIFY_TTL_S: one check, two sends.
    assert ollama.paths == ["/api/show", "/api/chat", "/api/chat"]


def test_the_cached_answer_expires(fake_ollama, monkeypatch):
    ollama = fake_ollama()
    complete("s", "u", base_url=ollama.url, model=QWEN, local_only=True)
    monkeypatch.setattr(ollama_guard, "VERIFY_TTL_S", 0.0)
    ollama.show_reply = CLOUD_REPLY  # the alias was repointed at a cloud model

    with pytest.raises(LocalLLMRemoteModel):
        complete("s", "u", base_url=ollama.url, model=QWEN, local_only=True)
    assert ollama.paths == ["/api/show", "/api/chat", "/api/show"]


def test_an_unanswerable_check_is_unavailable_and_not_cached(fake_ollama):
    ollama = fake_ollama()
    ollama.show_status = 404

    with pytest.raises(LocalLLMUnavailable) as err:
        complete("s", "u", base_url=ollama.url, model="missing", local_only=True)
    assert not isinstance(err.value, LocalLLMRemoteModel)
    assert ollama.paths == ["/api/show"]

    ollama.show_status = 200
    assert complete("s", "u", base_url=ollama.url, model="missing", local_only=True) == "ok"
    assert ollama.paths == ["/api/show", "/api/show", "/api/chat"]


def test_ollama_down_is_unavailable(tmp_path):
    with pytest.raises(LocalLLMUnavailable):
        complete("s", "u", base_url="http://127.0.0.1:9", model=QWEN, local_only=True, timeout=2)


def test_a_cloud_name_is_refused_without_asking(fake_ollama):
    ollama = fake_ollama()
    with pytest.raises(LocalLLMRemoteModel):
        complete("s", "u", base_url=ollama.url, model="gpt-oss:120b-cloud", local_only=True)
    assert ollama.paths == []


def test_flag_off_vaults_do_not_ask(fake_ollama):
    """Only local-only vaults pay for the check; the others send as before."""
    ollama = fake_ollama()
    ollama.show_reply = CLOUD_REPLY
    assert complete("s", "u", base_url=ollama.url, model=QWEN) == "ok"
    assert ollama.paths == ["/api/chat"]


def test_local_only_never_takes_the_openai_path():
    with pytest.raises(LocalOnlyViolation):
        complete("s", "u", base_url="http://127.0.0.1:11434", model="m", api="openai",
                 local_only=True)


# ── embeddings ──────────────────────────────────────────────────────────────


def test_local_only_embedder_refuses_a_remote_model(fake_ollama):
    ollama = fake_ollama()
    ollama.show_reply = CLOUD_REPLY
    embedder = OllamaEmbedder(ollama.url, "embed:alias", local_only=True)

    async def _run():
        try:
            return await embedder.embed("SYNTHETIC CHUNK")
        finally:
            await embedder.close()

    with pytest.raises(EmbeddingModelRefused) as err:
        asyncio.run(_run())
    assert isinstance(err.value, EmbeddingBackendUnavailable)  # surveyor leaves state alone
    assert ollama.paths == ["/api/show"]  # refused at once, not retried, nothing sent
    assert peek_failures().get("ollama.embed_remote_model_refused") == 1


def test_local_only_embedder_checks_then_embeds(fake_ollama):
    ollama = fake_ollama()
    embedder = OllamaEmbedder(ollama.url, "nomic-embed-text", local_only=True)

    async def _run():
        try:
            return [await embedder.embed("a"), await embedder.embed("b")]
        finally:
            await embedder.close()

    assert asyncio.run(_run()) == [[0.1, 0.2, 0.3]] * 2
    assert ollama.paths == ["/api/show", "/api/embeddings", "/api/embeddings"]


def test_embedder_retries_an_unanswerable_check_then_gives_up(fake_ollama):
    ollama = fake_ollama()
    ollama.show_status = 500
    embedder = OllamaEmbedder(ollama.url, "nomic-embed-text", local_only=True)

    async def _run():
        try:
            return await embedder.embed("a")
        finally:
            await embedder.close()

    with pytest.raises(EmbeddingBackendUnavailable) as err:
        asyncio.run(_run())
    assert not isinstance(err.value, EmbeddingModelRefused)
    assert "/api/embeddings" not in ollama.paths


def test_query_engine_embedder_refuses_a_remote_model(fake_ollama):
    from alfred.query.engine import _SyncEmbedder

    ollama = fake_ollama()
    ollama.show_reply = CLOUD_REPLY
    embedder = _SyncEmbedder(url=f"{ollama.url}/api/embeddings", model="embed:alias",
                             base_url=ollama.url, local_only=True)

    with pytest.raises(LocalOnlyViolation):
        embedder.embed("SYNTHETIC QUERY")
    assert ollama.paths == ["/api/show"]


def test_employment_call_sites_carry_local_only(tmp_path):
    """The real employment config reaches every embedder and complete() with
    local_only set, so each of them asks before sending."""
    from alfred.daemons.surveyor import SurveyorDaemon
    from alfred.query.engine import QueryEngine

    cfg = AlfredConfig.load(_employment(tmp_path))

    assert cfg.llm["local_only"] is True
    assert SurveyorDaemon(cfg, None, asyncio.Queue(), store=None)._get_embedder().local_only
    assert QueryEngine(cfg)._get_embedder().local_only


# ── daemon start ────────────────────────────────────────────────────────────


def _local_cfg(tmp_path, url: str) -> AlfredConfig:
    cfg = AlfredConfig.load(_employment(tmp_path))
    # Point the loaded config at the fake (its port is not 11434, so this is
    # set after load, as a stand-in for the real loopback Ollama).
    cfg.ollama_base_url = url
    cfg.llm_base_url = url
    return cfg


def test_daemon_refuses_to_start_on_a_remote_model(tmp_path, fake_ollama, monkeypatch):
    from alfred.runner import refuse_remote_models

    ollama = fake_ollama()
    ollama.show_reply = CLOUD_REPLY
    cfg = _local_cfg(tmp_path, ollama.url)
    monkeypatch.setattr(AlfredConfig, "check_local_only", lambda self: None)

    with pytest.raises(LocalOnlyViolation, match="runs remotely"):
        refuse_remote_models(cfg)


def test_daemon_starts_when_ollama_cannot_be_asked(tmp_path, monkeypatch):
    """ollama-game-guard stops Ollama for games; the daemon still starts and
    every send asks again first."""
    from alfred.runner import refuse_remote_models

    cfg = _local_cfg(tmp_path, "http://127.0.0.1:9")
    monkeypatch.setattr(AlfredConfig, "check_local_only", lambda self: None)

    refuse_remote_models(cfg)  # no raise


def test_start_check_is_skipped_for_other_vaults(tmp_path, monkeypatch):
    from alfred.runner import refuse_remote_models

    def _boom(*a, **k):
        raise AssertionError("asked Ollama for a vault that is not local-only")

    monkeypatch.setattr(ollama_guard, "check_model_runs_here", _boom)
    cfg = AlfredConfig(vault_path=tmp_path, data_dir=tmp_path / "data")
    refuse_remote_models(cfg)


def test_run_daemons_makes_the_start_check(tmp_path, fake_ollama, monkeypatch):
    """`alfred up` on a local-only vault whose model runs remotely stops before
    any daemon or job exists."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    from alfred.runner import run_daemons

    class _Started(Exception):
        pass

    class _StubVectorStore:
        was_recreated = False

        def __init__(self, *args, **kwargs):
            pass

    def _fake_start(self, *args, **kwargs):
        raise _Started

    monkeypatch.setattr(AsyncIOScheduler, "start", _fake_start)
    monkeypatch.setattr("alfred.store.lancedb_store.LanceDBStore", _StubVectorStore)
    ollama = fake_ollama()
    ollama.show_reply = CLOUD_REPLY
    cfg = _local_cfg(tmp_path, ollama.url)
    monkeypatch.setattr(AlfredConfig, "check_local_only", lambda self: None)

    with pytest.raises(LocalOnlyViolation, match="runs remotely"):
        asyncio.run(run_daemons(cfg))
