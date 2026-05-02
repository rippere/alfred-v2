from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def load_env(config_path: Path) -> None:
    """Load .env from the same directory as config.yaml."""
    env_path = Path(config_path).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key not in __import__("os").environ:
            __import__("os").environ[key] = val


@dataclass
class AlfredConfig:
    vault_path: Path
    data_dir: Path
    ignore_dirs: list[str] = field(default_factory=lambda: [
        "inbox/processed", "wiki", "_archived", "_templates", "_bases",
    ])

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_llm_model: str = "mistral:latest"
    embed_dims: int = 768

    # Milvus
    milvus_collection: str = "vault_v2"

    # Janitor
    janitor_sweep_interval_s: int = 3600
    janitor_deep_interval_h: int = 24
    janitor_max_bytes_per_call: int = 8000

    # Surveyor
    hdbscan_min_cluster_size: int = 2
    hdbscan_min_samples: int = 1
    leiden_resolution: float = 1.0

    # Query
    default_top_k: int = 8
    hopfield_iterations: int = 3
    hopfield_beta: float = 2.0
    graph_hops: int = 2
    graph_decay: float = 0.5

    # Wiki
    wiki_dir: str = "wiki"
    wiki_max_chars: int = 8000

    # Consolidator
    consolidator_min_interval_s: int = 1800

    # Synthesis
    anthropic_model: str = "claude-sonnet-4-6"
    openrouter_model: str = "x-ai/grok-4.1-fast"

    @property
    def milvus_uri(self) -> str:
        return str(self.data_dir / "milvus.db")

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def graph_path(self) -> Path:
        return self.data_dir / "graph.pkl"

    @property
    def bm25_path(self) -> Path:
        return self.data_dir / "bm25_index.pkl"

    @classmethod
    def load(cls, path: Path | str = "config.yaml") -> "AlfredConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        raw = yaml.safe_load(path.read_text())

        vault_path = Path(raw["vault"]["path"]).expanduser()
        data_dir_raw = raw.get("data_dir", "./data")
        # Resolve relative to config file location
        data_dir = (path.parent / data_dir_raw).resolve()
        data_dir.mkdir(parents=True, exist_ok=True)

        load_env(path)
        cfg = cls(vault_path=vault_path, data_dir=data_dir)

        # Ollama overrides
        if ol := raw.get("ollama"):
            cfg.ollama_base_url = ol.get("base_url", cfg.ollama_base_url)
            cfg.ollama_embed_model = ol.get("embed_model", cfg.ollama_embed_model)
            cfg.ollama_llm_model = ol.get("llm_model", cfg.ollama_llm_model)

        # Vault ignore dirs
        if v := raw.get("vault"):
            cfg.ignore_dirs = v.get("ignore_dirs", cfg.ignore_dirs)

        # Surveyor
        if s := raw.get("surveyor"):
            cfg.hdbscan_min_cluster_size = s.get("hdbscan_min_cluster_size", cfg.hdbscan_min_cluster_size)
            cfg.hdbscan_min_samples = s.get("hdbscan_min_samples", cfg.hdbscan_min_samples)
            cfg.embed_dims = s.get("embed_dims", cfg.embed_dims)

        # Query
        if q := raw.get("query"):
            cfg.default_top_k = q.get("top_k", cfg.default_top_k)
            cfg.hopfield_beta = q.get("hopfield_beta", cfg.hopfield_beta)

        # Janitor
        if j := raw.get("janitor"):
            cfg.janitor_sweep_interval_s = j.get("sweep_interval_s", cfg.janitor_sweep_interval_s)
            cfg.janitor_max_bytes_per_call = j.get("max_bytes_per_call", cfg.janitor_max_bytes_per_call)

        # Synthesis models
        if syn := raw.get("synthesis"):
            cfg.anthropic_model = syn.get("anthropic_model", cfg.anthropic_model)
            cfg.openrouter_model = syn.get("openrouter_model", cfg.openrouter_model)

        return cfg
