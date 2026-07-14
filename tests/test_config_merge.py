"""Base-config merge semantics (audit structural #2).

Proves the three-layer precedence — dataclass defaults <- config-base.yaml <-
vault file — and that lists are replaced wholesale, base absence is harmless,
and unused keys warn without failing the load.

Second half: the effective-config invariant for the six REAL vault configs at
the repo root — each must load without error and land on the expected merged
values (per-vault deviations win, base fills the gaps, dataclass defaults
backstop what neither file sets).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from alfred.config import AlfredConfig, _deep_merge

REPO_ROOT = Path(__file__).resolve().parents[1]

BASE = """\
ollama:
  llm_model: base-model:latest
vault:
  ignore_dirs: [wiki, _archived]
janitor:
  sweep_interval_s: 11111
  deep_interval_h: 24
query:
  top_k: 8
"""

VAULT = """\
vault:
  path: {vault_path}
  ignore_dirs: [only-this-one]
data_dir: ./data
janitor:
  sweep_interval_s: 22222
"""


def _write(tmp_path: Path, base: str | None) -> Path:
    if base is not None:
        (tmp_path / "config-base.yaml").write_text(base)
    cfg_path = tmp_path / "config-test.yaml"
    (tmp_path / "vault").mkdir()
    cfg_path.write_text(VAULT.format(vault_path=tmp_path / "vault"))
    return cfg_path


def test_vault_file_wins_and_base_fills_gaps(tmp_path):
    cfg = AlfredConfig.load(_write(tmp_path, BASE))
    # Vault file wins over base
    assert cfg.janitor_sweep_interval_s == 22222
    # Base wins over dataclass default (default is 24, base sets it too — use
    # sweep partner key deep_interval_h which only base sets vs default 24;
    # llm_model is the unambiguous one: default "mistral:latest")
    assert cfg.ollama_llm_model == "base-model:latest"
    assert cfg.default_top_k == 8
    # Dataclass default survives where neither file sets a key
    assert cfg.hopfield_beta == 2.0


def test_lists_replaced_wholesale_not_concatenated(tmp_path):
    cfg = AlfredConfig.load(_write(tmp_path, BASE))
    assert cfg.ignore_dirs == ["only-this-one"]


def test_missing_base_is_harmless(tmp_path):
    cfg = AlfredConfig.load(_write(tmp_path, None))
    assert cfg.janitor_sweep_interval_s == 22222
    assert cfg.ollama_llm_model == "mistral:latest"  # dataclass default


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2]}
    override = {"a": {"y": 3}, "b": [9]}
    merged = _deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 3}, "b": [9]}
    assert base == {"a": {"x": 1, "y": 2}, "b": [1, 2]}
    assert override == {"a": {"y": 3}, "b": [9]}


def test_unused_key_warns_but_load_succeeds(tmp_path, capsys):
    cfg_path = _write(tmp_path, BASE)
    cfg_path.write_text(
        cfg_path.read_text() + "distiller:\n  mode: on_demand\n  stale_days: 30\n"
    )
    cfg = AlfredConfig.load(cfg_path)  # must NOT raise
    assert cfg.distiller_mode == "on_demand"
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "config_unused_key" in out
    assert "distiller.stale_days" in out


# ---------------------------------------------------------------------------
# Effective-config invariant for the six real vault configs
# ---------------------------------------------------------------------------

BASE_IGNORE_DIRS = ["inbox", "wiki", "_archived", "_templates", "_bases", ".obsidian"]

# Per-config expected AlfredConfig field values AFTER the base merge. Only
# fields with a deliberate per-vault story are listed; fleet-wide invariants
# are asserted separately for every config.
EXPECTED: dict[str, dict[str, object]] = {
    "config.yaml": {
        "vault_path": Path("/mnt/external/obsidian-vault"),
        "ignore_dirs": BASE_IGNORE_DIRS,           # from base
        "api_max_calls_per_day": 500,              # from base
        "api_warn_at_calls": 400,
        "janitor_sweep_interval_s": 14400,         # from base
        "janitor_deep_interval_h": 24,
        "default_top_k": 8,
        "embed_dims": 768,                         # pinned in the vault file
    },
    "config-content.yaml": {
        "vault_path": Path("/mnt/external/vault-content"),
        # Vault file replaces the base list wholesale
        "ignore_dirs": ["published", "_archived", "_templates", ".obsidian"],
        # Content vault keeps its intentionally tighter budget after the merge
        "api_max_calls_per_day": 100,
        "api_warn_at_calls": 80,
        "janitor_sweep_interval_s": 7200,          # per-vault deviation
        "janitor_deep_interval_h": 48,             # per-vault deviation
        "default_top_k": 6,                        # per-vault deviation
    },
    "config-employment.yaml": {
        "vault_path": Path("/mnt/external/vault-employment"),
        "ignore_dirs": BASE_IGNORE_DIRS,
        "api_max_calls_per_day": 500,
        "api_warn_at_calls": 400,
        "janitor_sweep_interval_s": 14400,
        "janitor_deep_interval_h": 24,
        "default_top_k": 8,
    },
    "config-finance.yaml": {
        "vault_path": Path("/mnt/external/vault-finance"),
        "ignore_dirs": BASE_IGNORE_DIRS,
        "api_max_calls_per_day": 500,
        "api_warn_at_calls": 400,
        "janitor_sweep_interval_s": 14400,
        "janitor_deep_interval_h": 24,
        "default_top_k": 8,
    },
    "config-neuroscience.yaml": {
        "vault_path": Path("/mnt/external/vault-neuroscience"),
        "ignore_dirs": BASE_IGNORE_DIRS,
        "api_max_calls_per_day": 500,
        "api_warn_at_calls": 400,
        "janitor_sweep_interval_s": 14400,
        "janitor_deep_interval_h": 24,
        "default_top_k": 8,
    },
    "config-personal.yaml": {
        "vault_path": Path("/mnt/external/vault-personal"),
        # Wholesale replacement: base list plus the extra published-content dir
        "ignore_dirs": BASE_IGNORE_DIRS + ["content/published"],
        "api_max_calls_per_day": 500,
        "api_warn_at_calls": 400,
        "janitor_sweep_interval_s": 14400,
        "janitor_deep_interval_h": 24,
        "default_top_k": 8,
    },
}

# Data dir each config must resolve to (relative to the repo root).
EXPECTED_DATA_DIR = {
    "config.yaml": "data",
    "config-content.yaml": "data-content",
    "config-employment.yaml": "data-employment",
    "config-finance.yaml": "data-finance",
    "config-neuroscience.yaml": "data-neuroscience",
    "config-personal.yaml": "data-personal",
}


@pytest.mark.parametrize("config_name", sorted(EXPECTED))
def test_real_config_loads_with_expected_effective_values(config_name):
    src = REPO_ROOT / config_name
    assert src.exists(), f"{config_name} missing from repo root"

    cfg = AlfredConfig.load(src)  # must not raise

    for field_name, expected in EXPECTED[config_name].items():
        actual = getattr(cfg, field_name)
        assert actual == expected, (
            f"{config_name}: {field_name} = {actual!r}, expected {expected!r}"
        )

    assert cfg.data_dir == (REPO_ROOT / EXPECTED_DATA_DIR[config_name]).resolve()

    # Fleet-wide invariants every vault must inherit (base or dataclass default)
    assert cfg.vector_store == "lancedb"
    assert cfg.distiller_mode == "scheduled"          # from config-base.yaml
    assert cfg.janitor_dedup_enabled is False         # gated until dry-run tested
    assert cfg.ollama_llm_model == "mistral:latest"
    assert cfg.ollama_embed_model == "nomic-embed-text"
    assert cfg.anthropic_model == "claude-sonnet-4-6"
    assert cfg.hopfield_beta == 2.0
    # Dataclass default backstop: no file sets consolidator cadence
    assert cfg.consolidator_min_interval_s == 1800
