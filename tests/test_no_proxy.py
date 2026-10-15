"""Proxy settings cannot carry vault text off loopback.

httpx honours HTTP_PROXY / ALL_PROXY by default and has no localhost
exemption, so a proxy variable in a daemon's environment (or in a .env next
to its config, which load_env copies into os.environ) would send local-only
completions and embedding text to the proxy host while the loopback URL check
still passed. Every Ollama client now ignores the environment's proxies, and
load_env skips *_PROXY keys.

Real sockets: one FakeOllama is the target, another stands in for the proxy.
"""
from __future__ import annotations

import asyncio
import os

import pytest
import yaml

from alfred.config import AlfredConfig
from alfred.core import ollama_guard
from alfred.core.local_llm import complete
from alfred.embed.ollama import OllamaEmbedder
from alfred.query.engine import _SyncEmbedder

PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


@pytest.fixture
def proxied(fake_ollama, monkeypatch):
    """(target, proxy): every proxy variable points at `proxy`."""
    target, proxy = fake_ollama(), fake_ollama()
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name in PROXY_VARS:
        monkeypatch.setenv(name, proxy.url)
    return target, proxy


def test_the_environment_would_proxy_loopback_without_the_fix(proxied):
    """The premise: with trust_env on, httpx sends a loopback request to the proxy."""
    import httpx

    target, proxy = proxied
    httpx.post(f"{target.url}/api/chat", json={"model": "m"}, timeout=5)
    assert proxy.paths == [f"{target.url}/api/chat"]
    assert target.paths == []


def test_completion_ignores_proxy_variables(proxied):
    target, proxy = proxied
    assert complete("s", "SYNTHETIC", base_url=target.url, model="m") == "ok"
    assert complete("s", "SYNTHETIC", base_url=target.url, model="m", local_only=True) == "ok"
    assert proxy.paths == []
    assert target.paths == ["/api/chat", "/api/show", "/api/chat"]


def test_surveyor_embedder_ignores_proxy_variables(proxied):
    target, proxy = proxied
    embedder = OllamaEmbedder(target.url, "nomic-embed-text", local_only=True)

    async def _run():
        try:
            return await embedder.embed("SYNTHETIC CHUNK")
        finally:
            await embedder.close()

    assert asyncio.run(_run()) == [0.1, 0.2, 0.3]
    assert proxy.paths == []
    assert target.paths == ["/api/show", "/api/embeddings"]


def test_query_embedder_ignores_proxy_variables(proxied):
    target, proxy = proxied
    embedder = _SyncEmbedder(url=f"{target.url}/api/embeddings", model="nomic-embed-text",
                             base_url=target.url, local_only=True)

    assert embedder.embed("SYNTHETIC QUERY") == [0.1, 0.2, 0.3]
    assert proxy.paths == []
    assert target.paths == ["/api/show", "/api/embeddings"]


def test_model_check_ignores_proxy_variables(proxied):
    target, proxy = proxied
    ollama_guard.check_model_runs_here(target.url, "nomic-embed-text")
    assert proxy.paths == []
    assert target.paths == ["/api/show"]


def test_load_env_skips_proxy_keys(tmp_path, monkeypatch):
    for name in PROXY_VARS + ("NO_PROXY", "no_proxy", "ALFRED_TEST_KEEP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("alfred.config.SPARK_ENV_PATH", tmp_path / "no-spark-env")
    (tmp_path / ".env").write_text(
        "HTTP_PROXY=http://proxy.test:3128\n"
        "https_proxy=http://proxy.test:3128\n"
        "ALL_PROXY=socks5://proxy.test:1080\n"
        "NO_PROXY=\n"
        "ALFRED_TEST_KEEP=kept\n"
    )
    cfg_path = tmp_path / "config-test.yaml"
    cfg_path.write_text(yaml.safe_dump({"vault": {"path": str(tmp_path / "vault")}}))

    AlfredConfig.load(cfg_path)

    for name in ("HTTP_PROXY", "https_proxy", "ALL_PROXY", "NO_PROXY"):
        assert name not in os.environ, name
    assert os.environ["ALFRED_TEST_KEEP"] == "kept"
    monkeypatch.delenv("ALFRED_TEST_KEEP")
