"""Scheduler job registration (audit structural #4 — janitor/consolidator split).

Drives runner.run_daemons() up to (but not past) scheduler.start() with a
temp config and a stubbed vector store, then introspects the captured
APScheduler instance: the split janitor/consolidator responsibilities must be
registered as independent jobs with distinct ids, max_instances=1,
coalesce=True, and cadence-scaled misfire grace — and the legacy monolithic
job ids must be gone. No daemon ever actually runs.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from alfred.config import AlfredConfig
from alfred.runner import run_daemons


class _SchedulerCaptured(Exception):
    """Sentinel raised from the patched scheduler.start() to halt run_daemons
    after all jobs are registered but before anything runs or blocks."""


class _StubVectorStore:
    """Stands in for LanceDBStore — never touches disk or lancedb."""

    was_recreated = False

    def __init__(self, *args, **kwargs):
        pass


CFG_TEMPLATE = """\
vault:
  path: {vault_path}
data_dir: ./data
janitor:
  dedup_enabled: {dedup}
"""


def _registered_jobs(tmp_path: Path, monkeypatch, dedup: bool) -> dict:
    """Run run_daemons() to the brink of scheduler start; return {{job_id: Job}}."""
    (tmp_path / "vault").mkdir()
    cfg_path = tmp_path / "config-test.yaml"
    cfg_path.write_text(
        CFG_TEMPLATE.format(vault_path=tmp_path / "vault", dedup=str(dedup).lower())
    )
    cfg = AlfredConfig.load(cfg_path)

    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    captured = {}

    def _fake_start(self, *args, **kwargs):
        captured["scheduler"] = self
        raise _SchedulerCaptured

    monkeypatch.setattr(AsyncIOScheduler, "start", _fake_start)
    # run_daemons imports LanceDBStore lazily from this module attribute
    monkeypatch.setattr("alfred.store.lancedb_store.LanceDBStore", _StubVectorStore)

    with pytest.raises(_SchedulerCaptured):
        asyncio.run(run_daemons(cfg))

    scheduler = captured["scheduler"]
    return {job.id: job for job in scheduler.get_jobs()}


def test_split_jobs_registered_with_distinct_ids(tmp_path, monkeypatch):
    jobs = _registered_jobs(tmp_path, monkeypatch, dedup=True)

    # Janitor: four independent responsibilities, one job each
    janitor_ids = {
        "janitor_structural_tick",
        "janitor_deep_tick",
        "janitor_session_archive_tick",
        "janitor_dedup_tick",
    }
    # Consolidator: three independent responsibilities
    consolidator_ids = {
        "consolidator_label_tick",
        "consolidator_synthesis_tick",
        "consolidator_stubs_tick",
    }
    missing = (janitor_ids | consolidator_ids) - jobs.keys()
    assert not missing, f"split jobs not registered: {sorted(missing)}"

    # The pre-split monolithic jobs must be gone
    for legacy_id in ("janitor_tick", "consolidator_tick"):
        assert legacy_id not in jobs, f"legacy monolithic job {legacy_id} still registered"

    # Distinctness is implied by dict keys, but assert the count explicitly:
    # each responsibility has exactly one job object of its own.
    split_jobs = [jobs[i] for i in janitor_ids | consolidator_ids]
    assert len({id(j) for j in split_jobs}) == len(split_jobs)


def test_split_jobs_never_self_overlap_and_coalesce(tmp_path, monkeypatch):
    jobs = _registered_jobs(tmp_path, monkeypatch, dedup=True)
    for job_id, job in jobs.items():
        if not job_id.startswith(("janitor_", "consolidator_")):
            continue
        assert job.max_instances == 1, f"{job_id}: max_instances={job.max_instances}"
        assert job.coalesce is True, f"{job_id}: coalesce={job.coalesce}"


def test_misfire_grace_scales_with_cadence(tmp_path, monkeypatch):
    """A slow weekly dedup must tolerate hours of slippage; the sub-hourly
    consolidator jobs must not (the blanket-300s regression the split fixed)."""
    jobs = _registered_jobs(tmp_path, monkeypatch, dedup=True)
    expected_grace = {
        "janitor_structural_tick": 900,        # hourly-ish sweep
        "janitor_deep_tick": 3600,             # daily
        "janitor_session_archive_tick": 3600,  # daily
        "janitor_dedup_tick": 21600,           # weekly
        "consolidator_label_tick": 300,        # 30-min cadence
        "consolidator_synthesis_tick": 300,
        "consolidator_stubs_tick": 300,
    }
    for job_id, grace in expected_grace.items():
        assert jobs[job_id].misfire_grace_time == grace, (
            f"{job_id}: misfire_grace_time={jobs[job_id].misfire_grace_time}, "
            f"expected {grace}"
        )


def test_dedup_job_gated_on_config_flag(tmp_path, monkeypatch):
    jobs = _registered_jobs(tmp_path, monkeypatch, dedup=False)
    assert "janitor_dedup_tick" not in jobs
    # The other three janitor responsibilities are unconditional
    assert "janitor_structural_tick" in jobs
    assert "janitor_deep_tick" in jobs
    assert "janitor_session_archive_tick" in jobs


def test_other_daemons_and_infra_jobs_present(tmp_path, monkeypatch):
    jobs = _registered_jobs(tmp_path, monkeypatch, dedup=True)
    for job_id in (
        "surveyor_tick",
        "surveyor_recluster",
        "curator_tick",
        "periodic_save",
        "housekeeping",
    ):
        assert job_id in jobs, f"{job_id} missing from registration"
