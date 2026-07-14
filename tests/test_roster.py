"""Roster helper (audit structural #3 — single source of truth).

Runs scripts/alfred-roster.sh against a fixture repo (its own config-meta.yaml
+ vault configs) and a fixture $HOME, proving:

- unit-name derivation: config.yaml -> alfred; config-<x>.yaml -> alfred-<x>
  while a named unit file exists, alfred@<x> once only the template is
  installed, alfred-<x> as the legacy default when neither exists;
- watchdog mode emits "<unit>\t<data_dir>/alfred.pid" per vault;
- the failure contract: any parse/read failure exits non-zero with ZERO
  stdout (all-or-nothing — callers fall back to their hardcoded rosters).

HOME is overridden per test so the real ~/.config/systemd/user on this host
can never influence the outcome.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "alfred-roster.sh"

META = """\
vaults:
  - name: main
    config: {repo}/config.yaml
  - name: alpha
    config: {repo}/config-alpha.yaml
  - name: beta
    config: {repo}/config-beta.yaml
"""

VAULT_CFG = """\
vault:
  path: /nonexistent/vault-{name}
data_dir: ./data-{name}
"""


def _make_repo(tmp_path: Path) -> Path:
    """Fixture repo: config-meta.yaml, three vault configs, and a .venv/bin/python
    wrapper exec-ing the test interpreter (the script runs $REPO/.venv/bin/python;
    a plain symlink would lose the venv's site-packages, so wrap instead)."""
    repo = tmp_path / "repo"
    (repo / ".venv" / "bin").mkdir(parents=True)
    wrapper = repo / ".venv" / "bin" / "python"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    (repo / "config-meta.yaml").write_text(META.format(repo=repo))
    (repo / "config.yaml").write_text(VAULT_CFG.format(name="main"))
    (repo / "config-alpha.yaml").write_text(VAULT_CFG.format(name="alpha"))
    (repo / "config-beta.yaml").write_text(VAULT_CFG.format(name="beta"))
    return repo


def _make_home(tmp_path: Path, unit_files: list[str]) -> Path:
    home = tmp_path / "home"
    unit_dir = home / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    for name in unit_files:
        (unit_dir / name).write_text("[Unit]\n")
    return home


def _run(repo: Path, home: Path, mode: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "ALFRED_REPO": str(repo), "HOME": str(home)}
    return subprocess.run(
        [str(SCRIPT), mode], env=env, capture_output=True, text=True, timeout=30
    )


def test_script_exists_and_is_executable():
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)


def test_units_legacy_default_no_unit_files(tmp_path):
    """Neither named units nor the template installed -> alfred-<x>."""
    repo = _make_repo(tmp_path)
    home = _make_home(tmp_path, [])
    res = _run(repo, home, "units")
    assert res.returncode == 0, res.stderr
    assert res.stdout.splitlines() == ["alfred", "alfred-alpha", "alfred-beta"]


def test_units_template_instances_after_migration(tmp_path):
    """Only alfred@.service installed -> template instances; main stays alfred."""
    repo = _make_repo(tmp_path)
    home = _make_home(tmp_path, ["alfred@.service"])
    res = _run(repo, home, "units")
    assert res.returncode == 0, res.stderr
    assert res.stdout.splitlines() == ["alfred", "alfred@alpha", "alfred@beta"]


def test_units_named_unit_wins_over_template(tmp_path):
    """Pre-migration named unit takes precedence over the installed template,
    per-vault — a half-migrated fleet gets a mixed (correct) roster."""
    repo = _make_repo(tmp_path)
    home = _make_home(tmp_path, ["alfred@.service", "alfred-alpha.service"])
    res = _run(repo, home, "units")
    assert res.returncode == 0, res.stderr
    assert res.stdout.splitlines() == ["alfred", "alfred-alpha", "alfred@beta"]


def test_watchdog_mode_emits_unit_and_pid_file(tmp_path):
    repo = _make_repo(tmp_path)
    home = _make_home(tmp_path, [])
    res = _run(repo, home, "watchdog")
    assert res.returncode == 0, res.stderr
    lines = res.stdout.splitlines()
    assert len(lines) == 3
    expected = {
        "alfred": (repo / "data-main").resolve() / "alfred.pid",
        "alfred-alpha": (repo / "data-alpha").resolve() / "alfred.pid",
        "alfred-beta": (repo / "data-beta").resolve() / "alfred.pid",
    }
    got = dict(line.split("\t") for line in lines)
    assert got == {unit: str(pid) for unit, pid in expected.items()}


def test_missing_meta_fails_silently(tmp_path):
    """No config-meta.yaml -> non-zero exit, zero stdout (fallback contract)."""
    repo = _make_repo(tmp_path)
    (repo / "config-meta.yaml").unlink()
    home = _make_home(tmp_path, [])
    res = _run(repo, home, "units")
    assert res.returncode != 0
    assert res.stdout == ""


def test_empty_vault_list_fails_silently(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "config-meta.yaml").write_text("vaults: []\n")
    home = _make_home(tmp_path, [])
    res = _run(repo, home, "units")
    assert res.returncode != 0
    assert res.stdout == ""


def test_one_unreadable_vault_config_yields_no_partial_roster(tmp_path):
    """All-or-nothing: if a single vault config is unreadable in watchdog
    mode, the whole roster fails with zero stdout — never a partial list."""
    repo = _make_repo(tmp_path)
    (repo / "config-beta.yaml").unlink()
    home = _make_home(tmp_path, [])
    res = _run(repo, home, "watchdog")
    assert res.returncode != 0
    assert res.stdout == ""


def test_unknown_mode_rejected(tmp_path):
    repo = _make_repo(tmp_path)
    home = _make_home(tmp_path, [])
    res = _run(repo, home, "bogus")
    assert res.returncode != 0
    assert res.stdout == ""
    assert "usage" in res.stderr.lower()
