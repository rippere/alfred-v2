"""Meta server numeric-bounds validation (P1-05).

The meta server's ``vault_query`` tool derives an effective top_k (falling
back to the configured ``final_top_k`` when the caller passes 0) and uses
it directly to slice the reranked results — a path that does not go
through ``build_query_options``, so it needs its own guard. This proves
that guard rejects out-of-range input instead of silently mis-slicing.
"""
from __future__ import annotations

import asyncio

import fastmcp
import pytest
import yaml

from alfred.mcp.meta_server import run_meta_server


def _register_meta_mcp(tmp_path, monkeypatch) -> fastmcp.FastMCP:
    """Run run_meta_server() with mcp.run() stubbed out, capturing the built app.

    Zero vaults configured — engines/vault_cfgs are both empty lists — so
    registration doesn't need any real vault data on disk; the guard we're
    testing fires before any vault is touched.
    """
    config_path = tmp_path / "config-meta.yaml"
    config_path.write_text(
        yaml.safe_dump({"vaults": [], "meta_server": {"top_k_per_vault": 5, "final_top_k": 8}})
    )

    captured: dict[str, fastmcp.FastMCP] = {}

    def fake_run(self, *args, **kwargs):
        captured["mcp"] = self

    monkeypatch.setattr(fastmcp.FastMCP, "run", fake_run)
    run_meta_server(config_path)
    return captured["mcp"]


def test_meta_vault_query_rejects_negative_top_k(tmp_path, monkeypatch):
    mcp = _register_meta_mcp(tmp_path, monkeypatch)
    tool = asyncio.run(mcp.get_tool("vault_query"))
    with pytest.raises(ValueError, match="top_k"):
        asyncio.run(tool.fn(query="hello", top_k=-5))


def test_meta_vault_query_rejects_excessive_top_k(tmp_path, monkeypatch):
    mcp = _register_meta_mcp(tmp_path, monkeypatch)
    tool = asyncio.run(mcp.get_tool("vault_query"))
    with pytest.raises(ValueError, match="top_k"):
        asyncio.run(tool.fn(query="hello", top_k=10_000))


def test_meta_vault_query_normal_top_k_passes_through(tmp_path, monkeypatch):
    mcp = _register_meta_mcp(tmp_path, monkeypatch)
    tool = asyncio.run(mcp.get_tool("vault_query"))
    result = asyncio.run(tool.fn(query="hello", top_k=5))
    assert result["hits"] == []
    assert result["vaults_queried"] == []


# ── Per-vault LLM backend (the Betson carve-out) ────────────────────────────


class _OneHitStore:
    """LanceDBStore stand-in: every search returns the one synthetic note."""

    def search(self, dense_vec, sparse_vec, top_k, include_inbox=False):
        from alfred.store.types import SearchHit

        return [SearchHit(
            chunk_id="notes/alpha.md::chunk_00", rel_path="notes/alpha.md",
            score=0.9, record_type="note", name="alpha",
        )]

    def count(self) -> int:
        return 0


class _FlatEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1] * 768


def test_meta_server_synthesis_follows_each_vaults_own_llm_block(tmp_path, monkeypatch):
    """_build_engines loads each vault's own config file, so with config-base
    flipped to the Spark, a meta-server synthesis over the employment vault
    still goes to the local Ollama and the personal vault's goes to the Spark."""
    from pathlib import Path

    import httpx
    import openai

    from alfred.core import local_llm
    from alfred.mcp.meta_server import _build_engines, _query_one_vault

    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("SPARK_BASE_URL", "https://spark.test:8000/v1")
    monkeypatch.setenv("SPARK_MODEL", "qwen3-30b")
    monkeypatch.setattr("alfred.config.SPARK_ENV_PATH", tmp_path / "no-spark-env")

    # The real base, flipped the way step 15 will flip it, next to the real
    # vault files with their vaults pointed at synthetic tmp dirs.
    base = yaml.safe_load((repo / "config-base.yaml").read_text())
    base["llm"]["api"] = "openai"
    (tmp_path / "config-base.yaml").write_text(yaml.safe_dump(base))
    entries = []
    for name, file in (("employment", "config-employment.yaml"), ("personal", "config-personal.yaml")):
        raw = yaml.safe_load((repo / file).read_text())
        vault = tmp_path / f"vault-{name}"
        (vault / "notes").mkdir(parents=True)
        (vault / "notes" / "alpha.md").write_text("---\ntype: note\n---\nSynthetic note body.\n")
        raw["vault"]["path"] = str(vault)
        (tmp_path / file).write_text(yaml.safe_dump(raw))
        entries.append({"name": name, "config": str(tmp_path / file)})

    engines = dict(_build_engines({"vaults": entries}))
    assert set(engines) == {"employment", "personal"}
    for engine in engines.values():
        engine._store = _OneHitStore()
        engine._embedder = _FlatEmbedder()
    monkeypatch.setattr(
        "alfred.embed.reranker.rerank", lambda query, hits, texts, top_n: hits[:top_n]
    )

    ollama_urls: list[str] = []

    def _ollama_post(url, json=None, timeout=None):
        ollama_urls.append(url)
        return httpx.Response(
            200, json={"message": {"content": "local answer"}}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx, "post", _ollama_post)

    spark_urls: list[str] = []

    def _spark(request: httpx.Request) -> httpx.Response:
        spark_urls.append(str(request.url))
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "qwen3-30b",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "spark answer"}}],
        })

    monkeypatch.setattr(local_llm, "_openai_client", lambda base_url, api_key, timeout: openai.OpenAI(
        base_url=base_url, api_key=api_key, max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(_spark)),
    ))

    assert _query_one_vault("employment", engines["employment"], "q", 3, True)
    assert ollama_urls == ["http://127.0.0.1:11434/api/chat"]
    assert spark_urls == []

    assert _query_one_vault("personal", engines["personal"], "q", 3, True)
    assert spark_urls == ["https://spark.test:8000/v1/chat/completions"]
    assert ollama_urls == ["http://127.0.0.1:11434/api/chat"]
