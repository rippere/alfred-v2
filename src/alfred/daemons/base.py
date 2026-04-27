"""BaseDaemon and DaemonEvent — shared daemon infrastructure."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import anyio
import structlog

from alfred.config import AlfredConfig
from alfred.store.state import StateStore


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

    async def run(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        self._stop.set()

    def emit(self, kind: str, **payload: Any) -> None:
        try:
            self.events.put_nowait(DaemonEvent(kind=kind, payload=payload))
        except asyncio.QueueFull:
            pass   # non-blocking; events are best-effort

    async def save_state(self) -> None:
        self.state.save()
