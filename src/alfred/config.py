from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import structlog
import yaml

log = structlog.get_logger()

# Shared-defaults file merged UNDER each vault config (vault file wins).
BASE_CONFIG_NAME = "config-base.yaml"

# Completion backends the `llm:` block may name.
LLM_APIS = ("ollama", "openai")

# KEY=VALUE lines shared by every Spark client on this machine (tools/spark,
# hooks). Read when a SPARK_* variable is not in the environment.
SPARK_ENV_PATH = Path.home() / ".config" / "spark" / "env"

# Every YAML key path AlfredConfig.load() actually consumes. Keys present in a
# merged config but absent here are dead weight — load() warns on them (the
# check that would have caught `distiller.stale_days` / `consolidator.*`).
_CONSUMED_KEYS: frozenset[tuple[str, ...]] = frozenset({
    ("vault", "path"),
    ("vault", "ignore_dirs"),
    ("data_dir",),
    ("ollama", "base_url"),
    ("ollama", "embed_model"),
    ("ollama", "llm_model"),
    ("llm", "api"),
    ("llm", "base_url"),
    ("llm", "model"),
    ("llm", "api_key_env"),
    ("surveyor", "hdbscan_min_cluster_size"),
    ("surveyor", "hdbscan_min_samples"),
    ("surveyor", "embed_dims"),
    ("vector_store",),           # plain-string form
    ("vector_store", "backend"),  # mapping form
    ("query", "top_k"),
    ("query", "hopfield_beta"),
    ("query", "bm25_only"),
    ("janitor", "sweep_interval_s"),
    ("janitor", "deep_interval_h"),
    ("janitor", "max_bytes_per_call"),
    ("janitor", "dedup_enabled"),
    ("janitor", "forget_enabled"),
    ("janitor", "forget_retrievability"),
    ("janitor", "forget_min_age_days"),
    ("janitor", "forget_max_per_sweep"),
    ("janitor", "reap_enabled"),
    ("janitor", "reap_max_rows_per_sweep"),
    ("janitor", "reap_scan_batch_size"),
    ("janitor", "reap_delete_batch"),
    ("distiller", "mode"),
    ("distiller", "max_files_per_sweep"),
    ("api_budget", "max_calls_per_day"),
    ("api_budget", "warn_at_calls"),
})


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; override wins. Non-dict values (incl. lists such
    as vault.ignore_dirs) are replaced wholesale, never concatenated."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _warn_unused_keys(raw: dict, source: Path) -> None:
    """structlog-warn on any merged YAML leaf key that load() never consumes.

    Non-fatal by design: an unknown key means silent drift (set but ignored),
    which deserves a log line, not a crashed vault daemon."""
    def _walk(node, prefix: tuple[str, ...]) -> None:
        if isinstance(node, dict) and node:
            for key, value in node.items():
                _walk(value, prefix + (str(key),))
        elif prefix not in _CONSUMED_KEYS:
            log.warning(
                "config_unused_key",
                key=".".join(prefix),
                file=str(source),
                hint="set in YAML but consumed by no AlfredConfig field",
            )
    _walk(raw, ())


def spark_env(name: str) -> str | None:
    """A Spark client setting: the environment first, then SPARK_ENV_PATH.

    Blank counts as unset. Never logged or stored on AlfredConfig — the file
    holds SPARK_API_KEY, which local_llm reads per call instead.
    """
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        lines = SPARK_ENV_PATH.read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, val = line.partition("=")
        if sep and key.strip() == name:
            return val.strip().strip('"').strip("'") or None
    return None


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
    ollama_llm_model: str = "orcarouter/Qwen3.8-27B-Uncensored:q5_K_M"
    embed_dims: int = 768

    # Completions backend (the `llm:` block) for every daemon call and query
    # synthesis. Embeddings stay on the ollama_* fields whatever this says.
    #   "ollama": today's path. A blank base_url/model means ollama_base_url /
    #     ollama_llm_model, so an absent block changes nothing.
    #   "openai": an OpenAI-compatible server (the DGX Spark's vLLM). load()
    #     fills a blank base_url/model from SPARK_BASE_URL / SPARK_MODEL. The
    #     key is read from the variable api_key_env names, per call, and is
    #     never held here.
    llm_api: str = "ollama"
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key_env: str = "SPARK_API_KEY"

    # Milvus
    milvus_collection: str = "vault_v2"

    # Janitor
    janitor_sweep_interval_s: int = 3600
    janitor_deep_interval_h: int = 24
    janitor_max_bytes_per_call: int = 8000
    janitor_dedup_enabled: bool = False
    # Retention (forgetting) sweep. OFF by default: it evicts vectors, and an
    # eviction policy that turns itself on unannounced against 28 GB of
    # embeddings is not something a config default should decide for you.
    # `alfred forget` (dry-run by default) is the way to see what it would do.
    janitor_forget_enabled: bool = False
    # Evict below this Ebbinghaus retrievability. 0.02 ≈ untouched for ~1 year
    # at stability 1.0 — deliberately deep in the tail.
    janitor_forget_retrievability: float = 0.02
    # Never evict a file embedded more recently than this, regardless of
    # retrievability. Guards the cold-start case where nothing has been
    # queried yet and every file therefore looks unaccessed.
    janitor_forget_min_age_days: int = 180
    # Cap per sweep so a first run can't evict the entire store in one pass.
    janitor_forget_max_per_sweep: int = 500
    # Orphan reap sweep. OFF by default for the same reason as forget: it
    # deletes vectors, and unlike forget it deletes rows state.json does not
    # even know about, so a bug here is invisible to every other consistency
    # check. `alfred reap` (dry-run by default) is how you look first.
    janitor_reap_enabled: bool = False
    # Rows per sweep. The cap is blast radius, not memory — delete batching
    # bounds memory independently. Deliberately small: on the live store the
    # true reapable count measured 5 rows, so a sweep wanting thousands is
    # itself the signal that something changed.
    janitor_reap_max_rows_per_sweep: int = 5000
    # Enumeration batch ceiling. Lance will not merge a batch across
    # fragments, so on a fragmented store the effective batch is far smaller
    # and this knob barely moves peak memory (measured flat 183-184 MB from
    # 1024 to 200000). Left configurable only for a future compacted store.
    janitor_reap_scan_batch_size: int = 4096
    # Predicate terms per delete call. Peak scales as fragments x terms
    # (~3.6e-4 MB/pair); 500 x the live 10,274 fragments is ~1.9 GB, which
    # fits the 4G cap the daemons run under. Do not raise without measuring.
    janitor_reap_delete_batch: int = 500

    # Distiller
    distiller_mode: str = "on_demand"   # "scheduled" | "on_demand"
    # Stale files distilled per sweep, oldest stamp first. Each can append up
    # to 3 learnings to topic/ notes, so this bounds a sweep's vault writes
    # (and its LLM calls). The main vault had ~15K unstamped files when the
    # {items} fix landed: 200 a night works through them in ~75 nights
    # instead of ~45K appends in one.
    distiller_max_files_per_sweep: int = 200

    # API budget
    api_max_calls_per_day: int = 500
    api_warn_at_calls: int = 400

    # Surveyor
    hdbscan_min_cluster_size: int = 2
    hdbscan_min_samples: int = 1
    leiden_resolution: float = 1.0

    # Vector store backend: "lancedb" (default, no file-lock) or "milvus" (legacy)
    vector_store: str = "lancedb"

    # Query
    default_top_k: int = 8
    hopfield_iterations: int = 3
    hopfield_beta: float = 2.0
    graph_hops: int = 2
    graph_decay: float = 0.5
    # DEPRECATED / UNSUPPORTED: bm25_only enables a lightweight query mode that
    # skips Milvus/Ollama entirely and searches only the stored BM25 corpus at
    # bm25_path. That corpus is a frozen, point-in-time snapshot — it is built
    # solely by the archived one-off script scripts/_archive/phase4_rebuild_milvus.py
    # and is never refreshed by the live surveyor pipeline, so any vault content
    # added or changed after that snapshot is invisible to this path and it will
    # silently drift stale over time. LanceDBStore (the live vector_store backend)
    # also documents that it accepts-but-ignores sparse/BM25 vectors — LanceDB
    # retrieval is dense-only. Wiring BM25 into the live pipeline so this flag
    # reflects current vault state is a real feature (tracked separately, out of
    # scope for this fix) and requires a deliberate product decision; until then,
    # treat bm25_only as unsupported/deprecated rather than a viable production
    # retrieval mode. QueryEngine.query() logs a runtime warning whenever this is
    # enabled.
    bm25_only: bool = False   # lightweight mode: skip Milvus/Ollama, use stored BM25 corpus

    # Wiki
    wiki_dir: str = "wiki"
    wiki_max_chars: int = 8000

    # Consolidator
    consolidator_min_interval_s: int = 1800

    # Synthesis runs on the local model (ollama_llm_model above). The former
    # anthropic_model / openrouter_model keys were removed with the cloud chain.

    @property
    def llm(self) -> dict:
        """Backend keyword arguments for local_llm.complete() / complete_json()."""
        if self.llm_api == "ollama":
            return {
                "api": "ollama",
                "base_url": self.llm_base_url or self.ollama_base_url,
                "model": self.llm_model or self.ollama_llm_model,
            }
        return {
            "api": self.llm_api,
            "base_url": self.llm_base_url,
            "model": self.llm_model,
            "api_key_env": self.llm_api_key_env,
        }

    @property
    def milvus_uri(self) -> str:
        return str(self.data_dir / "milvus.db")

    @property
    def lancedb_uri(self) -> str:
        """Directory used by LanceDB (no file-lock contention)."""
        return str(self.data_dir / "lancedb")

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

        raw = yaml.safe_load(path.read_text()) or {}

        # Deep-merge shared defaults from config-base.yaml sitting next to the
        # loaded file, lowest precedence first:
        #   dataclass defaults <- config-base.yaml <- the vault's own file
        # The vault file always wins.
        base_path = path.parent / BASE_CONFIG_NAME
        if base_path.is_file() and path.resolve() != base_path.resolve():
            base_raw = yaml.safe_load(base_path.read_text()) or {}
            raw = _deep_merge(base_raw, raw)

        _warn_unused_keys(raw, path)

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

        # Completions backend
        if lm := raw.get("llm"):
            cfg.llm_api = lm.get("api", cfg.llm_api)
            cfg.llm_base_url = lm.get("base_url") or cfg.llm_base_url
            cfg.llm_model = lm.get("model") or cfg.llm_model
            cfg.llm_api_key_env = lm.get("api_key_env", cfg.llm_api_key_env)
        if cfg.llm_api not in LLM_APIS:
            # A typo must not quietly fall back to either backend.
            raise ValueError(f"{path}: llm.api is {cfg.llm_api!r}; expected one of {LLM_APIS}")
        if cfg.llm_api == "openai":
            cfg.llm_base_url = (cfg.llm_base_url or spark_env("SPARK_BASE_URL") or "").rstrip("/")
            cfg.llm_model = cfg.llm_model or spark_env("SPARK_MODEL") or ""
            if not (cfg.llm_base_url and cfg.llm_model):
                raise ValueError(
                    f"{path}: llm.api is openai but no base_url/model: set llm.base_url and "
                    f"llm.model, or SPARK_BASE_URL and SPARK_MODEL (environment or {SPARK_ENV_PATH})"
                )

        # Vault ignore dirs
        if v := raw.get("vault"):
            cfg.ignore_dirs = v.get("ignore_dirs", cfg.ignore_dirs)

        # Surveyor
        if s := raw.get("surveyor"):
            cfg.hdbscan_min_cluster_size = s.get("hdbscan_min_cluster_size", cfg.hdbscan_min_cluster_size)
            cfg.hdbscan_min_samples = s.get("hdbscan_min_samples", cfg.hdbscan_min_samples)
            cfg.embed_dims = s.get("embed_dims", cfg.embed_dims)

        # Vector store backend
        if vs := raw.get("vector_store"):
            cfg.vector_store = vs if isinstance(vs, str) else vs.get("backend", cfg.vector_store)

        # Query
        if q := raw.get("query"):
            cfg.default_top_k = q.get("top_k", cfg.default_top_k)
            cfg.hopfield_beta = q.get("hopfield_beta", cfg.hopfield_beta)
            cfg.bm25_only = q.get("bm25_only", cfg.bm25_only)

        # Janitor
        if j := raw.get("janitor"):
            cfg.janitor_sweep_interval_s = j.get("sweep_interval_s", cfg.janitor_sweep_interval_s)
            cfg.janitor_deep_interval_h = j.get("deep_interval_h", cfg.janitor_deep_interval_h)
            cfg.janitor_max_bytes_per_call = j.get("max_bytes_per_call", cfg.janitor_max_bytes_per_call)
            cfg.janitor_dedup_enabled = j.get("dedup_enabled", cfg.janitor_dedup_enabled)
            cfg.janitor_forget_enabled = j.get("forget_enabled", cfg.janitor_forget_enabled)
            cfg.janitor_forget_retrievability = j.get(
                "forget_retrievability", cfg.janitor_forget_retrievability
            )
            cfg.janitor_forget_min_age_days = j.get(
                "forget_min_age_days", cfg.janitor_forget_min_age_days
            )
            cfg.janitor_forget_max_per_sweep = j.get(
                "forget_max_per_sweep", cfg.janitor_forget_max_per_sweep
            )
            cfg.janitor_reap_enabled = j.get("reap_enabled", cfg.janitor_reap_enabled)
            cfg.janitor_reap_max_rows_per_sweep = j.get(
                "reap_max_rows_per_sweep", cfg.janitor_reap_max_rows_per_sweep
            )
            cfg.janitor_reap_scan_batch_size = j.get(
                "reap_scan_batch_size", cfg.janitor_reap_scan_batch_size
            )
            cfg.janitor_reap_delete_batch = j.get(
                "reap_delete_batch", cfg.janitor_reap_delete_batch
            )

        # Distiller mode
        if d := raw.get("distiller"):
            cfg.distiller_mode = d.get("mode", cfg.distiller_mode)
            cfg.distiller_max_files_per_sweep = d.get(
                "max_files_per_sweep", cfg.distiller_max_files_per_sweep
            )
        cap = cfg.distiller_max_files_per_sweep
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            # No "0 = unlimited": an uncapped sweep is the write burst this
            # setting exists to prevent. Set a large number to mean it.
            raise ValueError(
                f"{path}: distiller.max_files_per_sweep must be a positive integer, got {cap!r}"
            )

        # API budget
        if b := raw.get("api_budget"):
            cfg.api_max_calls_per_day = b.get("max_calls_per_day", cfg.api_max_calls_per_day)
            cfg.api_warn_at_calls = b.get("warn_at_calls", cfg.api_warn_at_calls)

        return cfg
