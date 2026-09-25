"""The `llm:` block: which backend completions go to.

Default is today's Ollama path, and the real configs must resolve to it until
someone flips llm.api. The employment vault is pinned to the local Ollama no
matter what config-base.yaml says (the Betson carve-out).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alfred.config import AlfredConfig, LocalOnlyViolation

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIGS = [
    "config.yaml",
    "config-content.yaml",
    "config-employment.yaml",
    "config-finance.yaml",
    "config-neuroscience.yaml",
    "config-personal.yaml",
]


@pytest.fixture(autouse=True)
def _no_real_spark_env(tmp_path, monkeypatch):
    """Keep the machine's real ~/.config/spark/env and SPARK_* out of every test."""
    for name in ("SPARK_API_KEY", "SPARK_BASE_URL", "SPARK_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("alfred.config.SPARK_ENV_PATH", tmp_path / "no-spark-env")


def _write(tmp_path: Path, base: dict, vault: dict, name: str = "config-test.yaml") -> Path:
    (tmp_path / "config-base.yaml").write_text(yaml.safe_dump(base))
    vault = {"vault": {"path": str(tmp_path / "vault")}, "data_dir": "./data", **vault}
    path = tmp_path / name
    path.write_text(yaml.safe_dump(vault))
    return path


def _copy_real(
    tmp_path: Path,
    name: str,
    base_llm: dict | None = None,
    base_ollama_url: str | None = None,
    drop: tuple[str, ...] = (),
    out_name: str | None = None,
    vault_path: str | None = None,
) -> Path:
    """A real vault config and the real base next to it, in tmp_path. The vault
    path is pointed at tmp_path so nothing reads a real vault; base_llm, when
    given, replaces the base's llm block (the fleet flip), and base_ollama_url
    the base's ollama.base_url (embeddings moved too). `drop` deletes top-level
    keys from the vault file, as a bad merge or revert would."""
    base = yaml.safe_load((REPO_ROOT / "config-base.yaml").read_text())
    if base_llm is not None:
        base["llm"] = base_llm
    if base_ollama_url is not None:
        base["ollama"]["base_url"] = base_ollama_url
    (tmp_path / "config-base.yaml").write_text(yaml.safe_dump(base))
    raw = yaml.safe_load((REPO_ROOT / name).read_text())
    raw["vault"]["path"] = vault_path or str(tmp_path / f"vault-{name}")
    for key in drop:
        raw.pop(key, None)
    out = tmp_path / (out_name or name)
    out.write_text(yaml.safe_dump(raw))
    return out


@pytest.mark.parametrize("config_name", REAL_CONFIGS)
def test_real_configs_stay_on_todays_ollama(config_name):
    """Merging this change must not move a single call off the local Ollama."""
    cfg = AlfredConfig.load(REPO_ROOT / config_name)

    assert cfg.llm["api"] == "ollama"
    assert cfg.llm["model"] == cfg.ollama_llm_model == "orcarouter/Qwen3.8-27B-Uncensored:q5_K_M"
    if config_name == "config-employment.yaml":
        assert cfg.local_only
        assert cfg.llm["base_url"] == cfg.embed_base_url == "http://127.0.0.1:11434"
    else:
        assert not cfg.local_only
        assert cfg.llm["base_url"] == cfg.embed_base_url == "http://localhost:11434"


def test_openai_takes_base_url_and_model_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARK_BASE_URL", "https://spark.test:8000/v1/")
    monkeypatch.setenv("SPARK_MODEL", "qwen3-30b")
    monkeypatch.setenv("SPARK_API_KEY", "sk-secret")

    cfg = AlfredConfig.load(_write(tmp_path, {"llm": {"api": "openai"}}, {}))

    assert cfg.llm == {
        "api": "openai",
        "base_url": "https://spark.test:8000/v1",
        "model": "qwen3-30b",
        "api_key_env": "SPARK_API_KEY",
        "local_only": False,
    }
    # The key is read per call from the variable named here, never held.
    assert "sk-secret" not in repr(cfg)
    # Embeddings are untouched by the flag.
    assert cfg.ollama_base_url == "http://localhost:11434"
    assert cfg.ollama_embed_model == "nomic-embed-text"


def test_openai_falls_back_to_the_spark_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / "spark-env"
    env_file.write_text(
        "# written by the spark setup\n"
        "SPARK_BASE_URL=https://spark-file.test:8000/v1\n"
        "export SPARK_MODEL=\"qwen3-30b\"\n"
        "SPARK_API_KEY=sk-file\n"
    )
    monkeypatch.setattr("alfred.config.SPARK_ENV_PATH", env_file)

    cfg = AlfredConfig.load(_write(tmp_path, {"llm": {"api": "openai"}}, {}))

    assert cfg.llm_base_url == "https://spark-file.test:8000/v1"
    assert cfg.llm_model == "qwen3-30b"


def test_explicit_block_values_win_over_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARK_BASE_URL", "https://env.test/v1")
    monkeypatch.setenv("SPARK_MODEL", "env-model")

    cfg = AlfredConfig.load(_write(
        tmp_path,
        {"llm": {"api": "openai"}},
        {"llm": {"base_url": "https://pinned.test/v1", "model": "pinned", "api_key_env": "OTHER_KEY"}},
    ))

    assert cfg.llm == {
        "api": "openai",
        "base_url": "https://pinned.test/v1",
        "model": "pinned",
        "api_key_env": "OTHER_KEY",
        "local_only": False,
    }


def test_openai_with_nowhere_to_send_fails_the_load(tmp_path):
    with pytest.raises(ValueError, match="SPARK_BASE_URL"):
        AlfredConfig.load(_write(tmp_path, {"llm": {"api": "openai"}}, {}))


def test_misspelled_api_fails_the_load(tmp_path):
    with pytest.raises(ValueError, match="llm.api"):
        AlfredConfig.load(_write(tmp_path, {"llm": {"api": "OpenAI"}}, {}))


SPARK_OLLAMA = "http://spark.test:11434"


@pytest.mark.parametrize("flip, ollama_url", [
    pytest.param(
        {"api": "openai", "api_key_env": "SPARK_API_KEY"}, None,
        id="base-sets-api-only",
    ),
    # Someone writes the Spark URL and model into the base instead of the env.
    pytest.param(
        {"api": "openai", "base_url": "https://spark.test:8000/v1",
         "model": "qwen3-30b", "api_key_env": "SPARK_API_KEY"}, None,
        id="base-also-sets-url-and-model",
    ),
    # And moves the fleet's embeddings off-box as well.
    pytest.param(
        {"api": "openai", "base_url": "https://spark.test:8000/v1",
         "model": "qwen3-30b", "api_key_env": "SPARK_API_KEY"}, SPARK_OLLAMA,
        id="base-also-moves-ollama-base-url",
    ),
])
def test_employment_stays_local_when_the_fleet_flips(tmp_path, monkeypatch, flip, ollama_url):
    """The Betson carve-out: config-base.yaml says openai, the employment vault
    still resolves to the local Ollama — URL and model both, even when the base
    names a model that Ollama doesn't have — and its embeddings stay there too
    when the base's ollama.base_url moves."""
    monkeypatch.setenv("SPARK_BASE_URL", "https://spark.test:8000/v1")
    monkeypatch.setenv("SPARK_MODEL", "qwen3-30b")

    employment = AlfredConfig.load(
        _copy_real(tmp_path, "config-employment.yaml", flip, ollama_url)
    )
    personal = AlfredConfig.load(_copy_real(tmp_path, "config-personal.yaml", flip, ollama_url))

    assert employment.local_only
    assert employment.llm == {
        "api": "ollama",
        "base_url": "http://127.0.0.1:11434",
        "model": "orcarouter/Qwen3.8-27B-Uncensored:q5_K_M",
        "local_only": True,
    }
    assert employment.embed_base_url == "http://127.0.0.1:11434"
    assert personal.llm["api"] == "openai"
    assert personal.llm["base_url"] == "https://spark.test:8000/v1"
    assert personal.embed_base_url == (ollama_url or "http://localhost:11434")


@pytest.mark.parametrize("flip", [
    pytest.param({"api": "openai", "api_key_env": "SPARK_API_KEY"}, id="base-sets-api-only"),
    pytest.param(
        {"api": "openai", "base_url": "https://spark.test:8000/v1",
         "model": "qwen3-30b", "api_key_env": "SPARK_API_KEY"},
        id="base-also-sets-url-and-model",
    ),
])
def test_main_vault_stays_on_ollama_when_the_base_flips(tmp_path, monkeypatch, flip):
    """config.yaml pins llm to ollama until the pre-flip gates close: the main
    vault holds twin-provenance records. A base flip (the old step-15
    instructions) moves the satellites and leaves main where it is."""
    monkeypatch.setenv("SPARK_BASE_URL", "https://spark.test:8000/v1")
    monkeypatch.setenv("SPARK_MODEL", "qwen3-30b")

    main = AlfredConfig.load(_copy_real(tmp_path, "config.yaml", flip))
    personal = AlfredConfig.load(_copy_real(tmp_path, "config-personal.yaml", flip))

    assert main.llm == {
        "api": "ollama",
        "base_url": "http://localhost:11434",
        "model": "orcarouter/Qwen3.8-27B-Uncensored:q5_K_M",
        "local_only": False,
    }
    assert personal.llm["api"] == "openai"


def _flip_one_vault(tmp_path: Path, name: str) -> Path:
    """The documented step-15 enablement: `llm: {api: openai}` in that vault's
    own file, base untouched."""
    path = _copy_real(tmp_path, name)
    raw = yaml.safe_load(path.read_text())
    raw["llm"] = {"api": "openai"}
    path.write_text(yaml.safe_dump(raw))
    return path


def test_the_documented_enablement_moves_one_satellite_only(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARK_BASE_URL", "https://spark.test:8000/v1")
    monkeypatch.setenv("SPARK_MODEL", "qwen3-30b")

    personal = AlfredConfig.load(_flip_one_vault(tmp_path, "config-personal.yaml"))
    assert personal.llm == {
        "api": "openai",
        "base_url": "https://spark.test:8000/v1",
        "model": "qwen3-30b",
        "api_key_env": "SPARK_API_KEY",
        "local_only": False,
    }
    assert personal.embed_base_url == "http://localhost:11434"
    for name in REAL_CONFIGS:
        if name != "config-personal.yaml":
            assert AlfredConfig.load(_copy_real(tmp_path, name)).llm["api"] == "ollama", name


def test_employment_refuses_the_documented_enablement(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARK_BASE_URL", "https://spark.test:8000/v1")
    monkeypatch.setenv("SPARK_MODEL", "qwen3-30b")
    with pytest.raises(LocalOnlyViolation):
        AlfredConfig.load(_flip_one_vault(tmp_path, "config-employment.yaml"))


def test_employment_chat_and_embeddings_hit_loopback_when_base_llm_and_ollama_flip(
    tmp_path, monkeypatch
):
    """Both halves of the base flipped (llm to the Spark, ollama.base_url to
    another host). Drive the real call sites and record where each request goes:
    a completion, the surveyor's embedder and the query engine's embedder."""
    import httpx

    from alfred.core.local_llm import complete
    from alfred.daemons.surveyor import SurveyorDaemon
    from alfred.query.engine import QueryEngine

    monkeypatch.setenv("SPARK_BASE_URL", "https://spark.test:8000/v1")
    monkeypatch.setenv("SPARK_MODEL", "qwen3-30b")
    flip = {"api": "openai", "base_url": "https://spark.test:8000/v1",
            "model": "qwen3-30b", "api_key_env": "SPARK_API_KEY"}
    cfg = AlfredConfig.load(
        _copy_real(tmp_path, "config-employment.yaml", flip, SPARK_OLLAMA)
    )
    seen: list[str] = []

    def _post(url, json=None, timeout=None, trust_env=True):
        seen.append(url)
        body = {"message": {"content": "ok"}, "embedding": [0.0]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", _post)

    assert complete("s", "u", **cfg.llm) == "ok"
    QueryEngine(cfg)._get_embedder().embed("synthetic text")
    import asyncio

    surveyor = SurveyorDaemon(cfg, None, asyncio.Queue(), store=None)

    # Each model is checked with the local Ollama (/api/show) before its
    # first send; the checks go to loopback too.
    assert seen == [
        "http://127.0.0.1:11434/api/show",
        "http://127.0.0.1:11434/api/chat",
        "http://127.0.0.1:11434/api/show",
        "http://127.0.0.1:11434/api/embeddings",
    ]
    assert surveyor._get_embedder().url == "http://127.0.0.1:11434/api/embeddings"


# ── local_only: the carve-out fails closed ──────────────────────────────────


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:11434", "http://127.0.0.1:11434/", "http://localhost:11434",
    "http://[::1]:11434",
])
def test_local_only_accepts_loopback_ollama(tmp_path, url):
    cfg = AlfredConfig.load(_write(
        tmp_path, {}, {"local_only": True, "ollama": {"base_url": url}, "llm": {"base_url": url}},
    ))
    assert cfg.local_only
    assert cfg.llm["base_url"] == url and cfg.embed_base_url == url


@pytest.mark.parametrize("url", [
    "http://spark.test:11434",          # another host on Ollama's port
    "http://127.0.0.2:11434",           # loopback range, not the loopback address
    "http://127.0.0.1:8000",            # a local vLLM is not the local Ollama
    "http://localhost",                 # no port: 80
    "https://127.0.0.1:11434",
    "http://user@127.0.0.1:11434",
    "http://127.0.0.1:11434/v1",
    "http://127.0.0.1.spark.test:11434",
])
@pytest.mark.parametrize("where", ["llm", "ollama"])
def test_local_only_refuses_any_other_url(tmp_path, url, where):
    vault = {"local_only": True, where: {"base_url": url}}
    with pytest.raises(LocalOnlyViolation, match="local-only vault"):
        AlfredConfig.load(_write(tmp_path, {}, vault))


def test_local_only_refuses_the_openai_api_even_on_loopback(tmp_path):
    vault = {"local_only": True,
             "llm": {"api": "openai", "base_url": "http://127.0.0.1:11434", "model": "m"}}
    with pytest.raises(LocalOnlyViolation, match="llm.api"):
        AlfredConfig.load(_write(tmp_path, {}, vault))


def test_local_only_is_checked_before_the_spark_env_is_read(tmp_path, monkeypatch):
    """No SPARK_* anywhere: a local-only vault flipped to openai must fail as
    local-only, not as 'no Spark URL' (which reads the Spark env first)."""
    with pytest.raises(LocalOnlyViolation):
        AlfredConfig.load(_write(tmp_path, {"llm": {"api": "openai"}}, {"local_only": True}))


@pytest.mark.parametrize("drop", [
    pytest.param(("llm",), id="llm-block-deleted"),
    pytest.param(("llm", "ollama"), id="both-pins-deleted"),
    pytest.param(("llm", "ollama", "local_only"), id="pins-and-flag-deleted"),
])
def test_employment_fails_closed_when_its_pins_are_deleted(tmp_path, monkeypatch, drop):
    """Before local_only, deleting the llm block from config-employment.yaml
    (a bad merge, a revert) sent the vault to the Spark with no error. Now the
    load refuses, even with the flag itself deleted: the file name forces it."""
    monkeypatch.setenv("SPARK_BASE_URL", "https://spark.test:8000/v1")
    monkeypatch.setenv("SPARK_MODEL", "qwen3-30b")

    with pytest.raises(LocalOnlyViolation):
        AlfredConfig.load(_copy_real(
            tmp_path, "config-employment.yaml", {"api": "openai"}, SPARK_OLLAMA, drop=drop,
        ))


def test_employment_vault_path_is_local_only_under_any_config_name(tmp_path, monkeypatch):
    """A copy of the employment config under another name, pins and flag
    deleted, still refuses: /mnt/external/vault-employment is local-only."""
    monkeypatch.setenv("SPARK_BASE_URL", "https://spark.test:8000/v1")
    monkeypatch.setenv("SPARK_MODEL", "qwen3-30b")

    with pytest.raises(LocalOnlyViolation):
        AlfredConfig.load(_copy_real(
            tmp_path, "config-employment.yaml", {"api": "openai"},
            drop=("llm", "ollama", "local_only"), out_name="config-work.yaml",
            vault_path="/mnt/external/vault-employment",
        ))
    with pytest.raises(LocalOnlyViolation):
        AlfredConfig.load(_copy_real(
            tmp_path, "config-employment.yaml", {"api": "openai"},
            drop=("llm", "ollama", "local_only"), out_name="config-twin.yaml",
            vault_path="/mnt/external/Employment/Betson/betson-gameroom-twin/vault",
        ))


@pytest.mark.parametrize("vault_path", [
    "/mnt/external",
    "/mnt",
    "/mnt/external/Employment/..",
    str(Path.home()),
])
def test_a_vault_that_contains_a_local_only_root_is_local_only(tmp_path, monkeypatch, vault_path):
    """The path check works both ways: a vault whose tree takes in the
    employment vault, the twin repo or ~/betson-it-review is local-only, so a
    base flip to the Spark fails its load instead of moving it."""
    monkeypatch.setenv("SPARK_BASE_URL", "https://spark.test:8000/v1")
    monkeypatch.setenv("SPARK_MODEL", "qwen3-30b")
    path = _write(tmp_path, {"llm": {"api": "openai"}}, {"vault": {"path": vault_path}})

    with pytest.raises(LocalOnlyViolation):
        AlfredConfig.load(path)

    # And with the base on ollama it loads, local-only, on loopback.
    path = _write(tmp_path, {}, {"vault": {"path": vault_path},
                                 "ollama": {"base_url": "http://127.0.0.1:11434"}})
    assert AlfredConfig.load(path).local_only


@pytest.mark.parametrize("vault_path", [
    "/mnt/external/obsidian-vault", "/mnt/external/vault-personal", "/mnt/external/Employed",
    "/mnt/external/vault-employment-archive",
])
def test_neighbours_of_a_local_only_root_are_not_local_only(tmp_path, vault_path):
    path = _write(tmp_path, {}, {"vault": {"path": vault_path}})
    assert not AlfredConfig.load(path).local_only


def test_a_runtime_change_is_refused_at_the_call_site(tmp_path):
    """load() is not the only gate: cfg.llm and cfg.embed_base_url re-check,
    so code that edits a loaded config cannot route a local-only vault off-box."""
    cfg = AlfredConfig.load(_copy_real(tmp_path, "config-employment.yaml"))
    cfg.llm_api = "openai"
    cfg.llm_base_url = "https://spark.test:8000/v1"
    with pytest.raises(LocalOnlyViolation):
        _ = cfg.llm

    cfg = AlfredConfig.load(_copy_real(tmp_path, "config-employment.yaml"))
    cfg.ollama_base_url = SPARK_OLLAMA
    with pytest.raises(LocalOnlyViolation):
        _ = cfg.embed_base_url


def test_other_vaults_are_not_local_only(tmp_path):
    for name in REAL_CONFIGS:
        if name == "config-employment.yaml":
            continue
        assert not AlfredConfig.load(_copy_real(tmp_path, name)).local_only, name
