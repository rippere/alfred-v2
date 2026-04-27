"""Alfred daemon runner — starts all daemons under a single anyio task group."""
from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import Any

import anyio
import structlog

log = structlog.get_logger()


async def run_daemons(cfg, only: set[str] | None = None) -> None:
    """Start selected (or all) daemons. Runs until SIGINT/SIGTERM."""
    from alfred.store.milvus import MilvusStore
    from alfred.store.state import StateStore
    from alfred.daemons.surveyor import SurveyorDaemon
    from alfred.daemons.janitor import JanitorDaemon
    from alfred.daemons.curator import CuratorDaemon
    from alfred.daemons.distiller import DistillerDaemon
    from alfred.daemons.consolidator import ConsolidatorDaemon

    events: asyncio.Queue = asyncio.Queue(maxsize=500)
    state_store = StateStore(cfg.state_path)
    state_store.load()

    milvus = MilvusStore(uri=cfg.milvus_uri, embed_dims=cfg.embed_dims, collection=cfg.milvus_collection)

    all_daemons = {}
    all_daemons["surveyor"] = SurveyorDaemon(cfg, state_store, events, milvus)
    all_daemons["janitor"] = JanitorDaemon(cfg, state_store, events)
    all_daemons["curator"] = CuratorDaemon(cfg, state_store, events)
    all_daemons["distiller"] = DistillerDaemon(cfg, state_store, events)
    all_daemons["consolidator"] = ConsolidatorDaemon(cfg, state_store, events)

    active = {k: v for k, v in all_daemons.items() if only is None or k in only}
    log.info("alfred.starting", daemons=list(active.keys()))

    # Update last_run in state
    from datetime import datetime, timezone
    state_store.state.last_run = datetime.now(timezone.utc).isoformat()
    state_store.save()

    stop_event = anyio.Event()

    def _handle_signal():
        log.info("alfred.shutdown_signal")
        stop_event.set()
        for d in active.values():
            d._stop.set()

    async def _event_loop():
        """Drain and log events from the shared queue."""
        while not stop_event.is_set():
            try:
                event = events.get_nowait()
                log.debug("alfred.event", kind=event.kind, **event.payload)
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.5)

    async with anyio.create_task_group() as tg:
        # Register signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _handle_signal)

        tg.start_soon(_event_loop)
        for daemon in active.values():
            tg.start_soon(daemon.run)

    log.info("alfred.stopped")
