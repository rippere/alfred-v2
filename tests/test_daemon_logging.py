"""Structlog coverage for daemons + the budget-guard store (regression for 9700ad2).

StateStore's budget-guard logger used to be a stdlib `logging.getLogger` called
with structlog-style kwargs (daemon=, calls_today=, cost_usd=, limit=) — stdlib
rejects unknown kwargs with a TypeError, which silently killed ~1.5k
distiller.file_error/run before it was caught (see src/alfred/store/state.py).
These tests pin every daemon's logger (via BaseDaemon, shared by all five) and
the module-level `log` each daemon file exports to structlog, so a future
regression back to stdlib logging fails loudly instead of at 1.5k errors/run.
"""
from __future__ import annotations

import asyncio
import importlib

import pytest
from structlog.testing import capture_logs

from alfred.daemons.base import BaseDaemon
from alfred.store.state import StateStore

# surveyor has no module-level `log` (only self.log via BaseDaemon) — covered
# by test_base_daemon_logger_accepts_structured_kwargs instead.
DAEMON_MODULES_WITH_MODULE_LOGGER = [
    "alfred.daemons.consolidator",
    "alfred.daemons.curator",
    "alfred.daemons.distiller",
    "alfred.daemons.janitor",
]


def test_base_daemon_logger_accepts_structured_kwargs():
    """self.log (used by every daemon) must be a structlog logger, not stdlib."""
    async def make_daemon() -> BaseDaemon:
        return BaseDaemon(cfg=None, state=None, events=asyncio.Queue())

    daemon = asyncio.run(make_daemon())
    with capture_logs() as cap:
        daemon.log.warning("daemon.event_queue_full", kind="tick", dropped_total=3)
    assert cap == [
        {
            "daemon": "base",
            "kind": "tick",
            "dropped_total": 3,
            "event": "daemon.event_queue_full",
            "log_level": "warning",
        }
    ]


@pytest.mark.parametrize("module_name", DAEMON_MODULES_WITH_MODULE_LOGGER)
def test_daemon_module_logger_is_structlog(module_name):
    """Every daemon module's module-level `log` must be structlog, mirroring
    BaseDaemon.log — a stdlib logger here would raise on the first structured
    call (e.g. curator.project_index_skip(path=..., error=...))."""
    module = importlib.import_module(module_name)
    log = module.log
    with capture_logs() as cap:
        log.warning("test.regression_probe", daemon=module_name, detail="x")
    assert cap == [
        {
            "daemon": module_name,
            "detail": "x",
            "event": "test.regression_probe",
            "log_level": "warning",
        }
    ]


def test_state_store_budget_guard_logger_accepts_structured_kwargs(tmp_path):
    """Exercise the exact code path fixed in 9700ad2: can_make_api_call() logs
    with daemon=/calls_today=/limit= kwargs once the daily budget is exhausted."""
    store = StateStore(tmp_path / "state.json", cfg=None)
    store.load()
    store._state.api_calls_today = 10_000  # force budget_remaining() <= 0

    with capture_logs() as cap:
        allowed = store.can_make_api_call(daemon="distiller")

    assert allowed is False
    assert cap == [
        {
            "daemon": "distiller",
            "calls_today": 10_000,
            "cost_usd": 0.0,
            "limit": 500,
            "event": "alfred.api_budget_exhausted",
            "log_level": "warning",
        }
    ]
