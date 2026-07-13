"""Dead-config-key guard (audit quick win #6 / structural #6c).

Loads each real vault config-*.yaml via AlfredConfig.load and proves every key
set in the YAML is actually consumed: for each leaf key, a perturbed copy of
the config is loaded and at least one AlfredConfig dataclass field must change
versus the unperturbed baseline. A key whose perturbation changes nothing is
dead (e.g. the former `distiller.stale_days` / `consolidator.synthesis_batch`)
and fails the test by name.
"""
from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import pytest
import yaml

from alfred.config import AlfredConfig, _deep_merge

REPO_ROOT = Path(__file__).resolve().parents[1]

VAULT_CONFIGS = [
    "config.yaml",
    "config-content.yaml",
    "config-employment.yaml",
    "config-finance.yaml",
    "config-neuroscience.yaml",
    "config-personal.yaml",
]


def _leaf_paths(node, prefix=()):
    """Yield (key_path_tuple, value) for every scalar/list leaf in the YAML."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _leaf_paths(v, prefix + (k,))
    else:
        yield prefix, node


def _perturb(key_path: tuple[str, ...], value):
    """Produce a same-typed but different value for the key."""
    if key_path == ("data_dir",):
        # Keep it relative so the perturbed dir lands inside tmp_path, never
        # alongside the live data dirs.
        return "./__sentinel_data__"
    if isinstance(value, bool):  # bool before int — bool is an int subclass
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return value + "-__sentinel__"
    if isinstance(value, list):
        return list(value) + ["__sentinel__"]
    return "__sentinel__"


def _set_in(d: dict, key_path: tuple[str, ...], value) -> None:
    for k in key_path[:-1]:
        d = d[k]
    d[key_path[-1]] = value


def _snapshot(cfg: AlfredConfig) -> dict:
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(AlfredConfig)}


@pytest.mark.parametrize("config_name", VAULT_CONFIGS)
def test_every_yaml_key_is_consumed(config_name, tmp_path):
    src = REPO_ROOT / config_name
    assert src.exists(), f"{config_name} missing from repo root"
    # Vault configs are slim overrides merged over config-base.yaml at load
    # time. Test the MERGED document so base keys are covered by the dead-key
    # guard too. No base file is copied into tmp_path, so load() sees exactly
    # this merged doc — same effective config as production.
    base = yaml.safe_load((REPO_ROOT / "config-base.yaml").read_text())
    raw = _deep_merge(base, yaml.safe_load(src.read_text()))

    # Load from a tmp copy so relative data_dir resolves inside tmp_path and
    # the test never touches the live data-*/ dirs.
    work = tmp_path / config_name
    work.write_text(yaml.safe_dump(raw))
    baseline = _snapshot(AlfredConfig.load(work))

    dead: list[str] = []
    for key_path, value in _leaf_paths(raw):
        perturbed = copy.deepcopy(raw)
        _set_in(perturbed, key_path, _perturb(key_path, value))
        work.write_text(yaml.safe_dump(perturbed))
        cfg2 = _snapshot(AlfredConfig.load(work))
        if cfg2 == baseline:
            dead.append(".".join(key_path))

    assert not dead, (
        f"{config_name}: dead config key(s) — set in YAML but consumed by no "
        f"AlfredConfig field: {dead}. Wire them up in AlfredConfig.load() or "
        f"delete them from the YAML."
    )
