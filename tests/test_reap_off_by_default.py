"""The orphan reaper deletes vector rows, so it must stay opt-in.

`alfred reap` is dry-run unless --apply (pinned in test_orphan_reaper.py);
these pin the other half: the janitor's scheduled sweep is off by default and
no shipped vault config turns it on. Enabling it is a deliberate, reviewed
config change made after a dry run, never a side effect of a merge.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from alfred.config import AlfredConfig

REPO = Path(__file__).resolve().parents[1]


def test_reap_sweep_disabled_in_config_defaults():
    cfg = AlfredConfig(vault_path=Path("/v"), data_dir=Path("/d"))
    assert cfg.janitor_reap_enabled is False


@pytest.mark.parametrize("cfg_file", sorted(p.name for p in REPO.glob("config*.yaml")))
def test_no_shipped_config_enables_reap_sweep(cfg_file):
    raw = yaml.safe_load((REPO / cfg_file).read_text()) or {}
    janitor = raw.get("janitor") or {}
    assert janitor.get("reap_enabled", False) is False, cfg_file


def test_reap_tick_is_a_noop_when_disabled(monkeypatch):
    from alfred.daemons.janitor import JanitorDaemon

    def _boom(*a, **k):
        raise AssertionError("reap must not run while janitor.reap_enabled is false")

    monkeypatch.setattr("alfred.store.reaper.execute_plan", _boom)
    monkeypatch.setattr("alfred.store.reaper.build_plan", _boom)

    jd = JanitorDaemon.__new__(JanitorDaemon)
    jd.cfg = SimpleNamespace(janitor_reap_enabled=False)
    jd.log = SimpleNamespace(error=_boom, warning=_boom, info=lambda *a, **k: None)
    jd.reap_plan = _boom

    asyncio.run(jd.reap_tick())
