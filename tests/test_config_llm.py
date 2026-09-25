"""The `llm:` block: which backend completions go to.

Default is today's Ollama path, and the real configs must resolve to it until
someone flips llm.api. The employment vault is pinned to the local Ollama no
matter what config-base.yaml says (the Betson carve-out).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alfred.config import AlfredConfig

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


def _copy_real(tmp_path: Path, name: str, base_llm: dict | None = None) -> Path:
    """A real vault config and the real base next to it, in tmp_path. The vault
    path is pointed at tmp_path so nothing reads a real vault; base_llm, when
    given, replaces the base's llm block (the fleet flip)."""
    base = yaml.safe_load((REPO_ROOT / "config-base.yaml").read_text())
    if base_llm is not None:
        base["llm"] = base_llm
    (tmp_path / "config-base.yaml").write_text(yaml.safe_dump(base))
    raw = yaml.safe_load((REPO_ROOT / name).read_text())
    raw["vault"]["path"] = str(tmp_path / f"vault-{name}")
    out = tmp_path / name
    out.write_text(yaml.safe_dump(raw))
    return out


@pytest.mark.parametrize("config_name", REAL_CONFIGS)
def test_real_configs_stay_on_todays_ollama(config_name):
    """Merging this change must not move a single call off the local Ollama."""
    cfg = AlfredConfig.load(REPO_ROOT / config_name)

    assert cfg.llm["api"] == "ollama"
    assert cfg.llm["model"] == cfg.ollama_llm_model == "orcarouter/Qwen3.8-27B-Uncensored:q5_K_M"
    if config_name == "config-employment.yaml":
        assert cfg.llm["base_url"] == "http://127.0.0.1:11434"
    else:
        assert cfg.llm["base_url"] == cfg.ollama_base_url == "http://localhost:11434"


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
    }


def test_openai_with_nowhere_to_send_fails_the_load(tmp_path):
    with pytest.raises(ValueError, match="SPARK_BASE_URL"):
        AlfredConfig.load(_write(tmp_path, {"llm": {"api": "openai"}}, {}))


def test_misspelled_api_fails_the_load(tmp_path):
    with pytest.raises(ValueError, match="llm.api"):
        AlfredConfig.load(_write(tmp_path, {"llm": {"api": "OpenAI"}}, {}))


def test_employment_stays_local_when_the_fleet_flips(tmp_path, monkeypatch):
    """The Betson carve-out: config-base.yaml says openai, the employment vault
    still resolves to the local Ollama — URL and model both."""
    monkeypatch.setenv("SPARK_BASE_URL", "https://spark.test:8000/v1")
    monkeypatch.setenv("SPARK_MODEL", "qwen3-30b")
    flip = {"api": "openai", "api_key_env": "SPARK_API_KEY"}

    employment = AlfredConfig.load(_copy_real(tmp_path, "config-employment.yaml", flip))
    personal = AlfredConfig.load(_copy_real(tmp_path, "config-personal.yaml", flip))

    assert employment.llm == {
        "api": "ollama",
        "base_url": "http://127.0.0.1:11434",
        "model": "orcarouter/Qwen3.8-27B-Uncensored:q5_K_M",
    }
    assert personal.llm["api"] == "openai"
    assert personal.llm["base_url"] == "https://spark.test:8000/v1"
