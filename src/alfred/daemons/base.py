"""BaseDaemon and DaemonEvent — shared daemon infrastructure."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import anyio
import structlog

from alfred.config import AlfredConfig
from alfred.store.state import StateSaveTimeout, StateStore

# Upper bound on a routine daemon save. Without one, a process stuck holding
# the state.json flock (e.g. a frozen MCP query server) would park the save's
# worker thread forever and hold StateStore._lock, queueing every later save,
# including the shutdown one, behind it.
DAEMON_SAVE_TIMEOUT_S = 60.0


@dataclass
class DaemonEvent:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)


class BaseDaemon:
    name: str = "base"

    def __init__(
        self,
        cfg: AlfredConfig,
        state: StateStore,
        events: asyncio.Queue[DaemonEvent],
    ) -> None:
        self.cfg = cfg
        self.state = state
        self.events = events
        self.log = structlog.get_logger(daemon=self.name)
        self._stop = anyio.Event()
        self.events_dropped: int = 0
        self._queue_full_last_warn: float = float("-inf")

    async def run(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        self._stop.set()

    def emit(self, kind: str, **payload: Any) -> None:
        try:
            self.events.put_nowait(DaemonEvent(kind=kind, payload=payload))
        except asyncio.QueueFull:
            # Events are best-effort, but dropping them silently hides
            # backpressure — count every drop and warn at most once a minute.
            self.events_dropped += 1
            now = time.monotonic()
            if now - self._queue_full_last_warn >= 60.0:
                self._queue_full_last_warn = now
                self.log.warning(
                    "daemon.event_queue_full",
                    kind=kind,
                    dropped_total=self.events_dropped,
                )

    async def save_state(self) -> None:
        # Off-loop: the blocking read/merge/dump/write runs in a worker thread.
        # The surveyor calls this every 25 files, and the old synchronous
        # save() froze the whole event loop each time, long enough under
        # memory pressure that SIGTERM wasn't even handled before systemd's
        # SIGKILL.
        # A timeout leaves state.json as it was and the changes in memory; the
        # next save retries them. Keep the tick going rather than abort it.
        try:
            await self.state.async_save(timeout=DAEMON_SAVE_TIMEOUT_S)
        except StateSaveTimeout as e:
            self.log.warning("daemon.save_state_timeout", timeout_s=DAEMON_SAVE_TIMEOUT_S, error=str(e))
