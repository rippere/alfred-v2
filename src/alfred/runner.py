"""Alfred daemon runner — APScheduler-based single-process scheduler.

Replaces the original asyncio.gather() approach with APScheduler's
AsyncIOScheduler.  All daemons share one StateStore, one MilvusStore/LanceDBStore,
and one asyncio.Lock (via StateStore._lock) — eliminating the cross-process race
condition that existed when each systemd service held its own vector store connection.

Each daemon's periodic work is exposed as a ``tick()`` / ``structural_tick()``
/ ``deep_tick()`` one-shot coroutine.  APScheduler fires them on the right
interval; the daemon classes themselves are unchanged in terms of logic.
"""
from __future__ import annotations

import asyncio
import signal
from datetime import datetime, timezone

import anyio
import structlog

log = structlog.get_logger()


async def run_daemons(cfg, only: set[str] | None = None) -> None:
    """Start selected (or all) daemons via APScheduler. Runs until SIGINT/SIGTERM."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    from alfred.daemons.consolidator import ConsolidatorDaemon
    from alfred.daemons.curator import CuratorDaemon
    from alfred.daemons.distiller import DistillerDaemon
    from alfred.daemons.janitor import JanitorDaemon
    from alfred.daemons.surveyor import SurveyorDaemon
    from alfred.store.state import StateStore

    # ── Shared infrastructure (single instance, single vector store lock) ──────
    events: asyncio.Queue = asyncio.Queue(maxsize=500)
    state_store = StateStore(cfg.state_path, cfg=cfg)
    state_store.load()

    # Vector store selection: lancedb (preferred) or milvus (legacy)
    if getattr(cfg, "vector_store", "milvus") == "lancedb":
        from alfred.store.lancedb_store import LanceDBStore
        vector_store = LanceDBStore(
            uri=getattr(cfg, "lancedb_uri", str(cfg.data_dir / "lancedb")),
            collection=cfg.milvus_collection,
            dims=cfg.embed_dims,
        )
        log.info("alfred.store", backend="lancedb")
    else:
        from alfred.store.milvus import MilvusStore
        vector_store = MilvusStore(
            uri=cfg.milvus_uri,
            embed_dims=cfg.embed_dims,
            collection=cfg.milvus_collection,
        )
        log.info("alfred.store", backend="milvus", uri=cfg.milvus_uri)

    # ── Daemon instances ───────────────────────────────────────────────────────
    surveyor = SurveyorDaemon(cfg, state_store, events, vector_store)
    janitor = JanitorDaemon(cfg, state_store, events)
    curator = CuratorDaemon(cfg, state_store, events)
    distiller = DistillerDaemon(cfg, state_store, events)
    consolidator = ConsolidatorDaemon(cfg, state_store, events)

    all_daemons = {
        "surveyor": surveyor,
        "janitor": janitor,
        "curator": curator,
        "distiller": distiller,
        "consolidator": consolidator,
    }
    active_names = (
        set(all_daemons.keys()) if only is None else (only & set(all_daemons.keys()))
    )
    log.info("alfred.starting", scheduler="apscheduler", daemons=sorted(active_names))

    # ── Update last_run in state ───────────────────────────────────────────────
    state_store.state.last_run = datetime.now(timezone.utc).isoformat()
    state_store.save()

    # ── Build APScheduler ──────────────────────────────────────────────────────
    scheduler = AsyncIOScheduler()

    def _add(daemon_name: str, func, trigger: str, **trigger_kwargs) -> None:
        """Register a scheduler job only when its daemon is selected."""
        if daemon_name not in active_names:
            return
        scheduler.add_job(
            func,
            trigger,
            id=f"{daemon_name}_{func.__name__}",
            max_instances=1,          # never overlap a slow job with itself
            coalesce=True,            # if late, run once not many times
            misfire_grace_time=300,   # tolerate 5-minute slippage
            **trigger_kwargs,
        )

    # ── Surveyor: poll every 60s, recluster every 30min ───────────────────────
    _add("surveyor", surveyor.tick, "interval", seconds=60)
    _add("surveyor", surveyor.recluster, "interval", seconds=1800)

    # ── Janitor: structural sweep on config interval, LLM deep sweep per config
    sweep_s = int(getattr(cfg, "janitor_sweep_interval_s", 3600))
    deep_h = int(getattr(cfg, "janitor_deep_interval_h", 24))
    _add("janitor", janitor.structural_tick, "interval", seconds=sweep_s)
    _add("janitor", janitor.deep_tick, "interval", hours=deep_h)
    if getattr(cfg, "janitor_dedup_enabled", False):
        _add("janitor", janitor.dedup_tick, "interval", weeks=1)

    # ── Distiller: every 24h from daemon start (not fixed clock time) ────────────
    # Interval-based so it fires during waking hours regardless of when the machine
    # was last on, rather than at a hardcoded 2am that may be missed entirely.
    if "distiller" in active_names:
        if getattr(cfg, "distiller_mode", "on_demand") == "scheduled":
            _add("distiller", distiller.tick, "interval", hours=24)
        else:
            log.info("distiller.scheduled_skipped", reason="distiller_mode=on_demand")

    # ── Consolidator: every consolidator_min_interval_s seconds ───────────────
    consolidate_s = int(getattr(cfg, "consolidator_min_interval_s", 1800))
    _add("consolidator", consolidator.tick, "interval", seconds=consolidate_s)

    # ── Curator: inbox poll every 10s ─────────────────────────────────────────
    _add("curator", curator.tick, "interval", seconds=10)

    # ── Periodic state save every 5 minutes ───────────────────────────────────
    async def _periodic_save() -> None:
        try:
            await state_store.async_save()
            log.debug("alfred.periodic_save")
        except Exception as e:
            log.warning("alfred.periodic_save_failed", error=str(e))

    scheduler.add_job(
        _periodic_save,
        "interval",
        seconds=300,
        id="periodic_save",
        max_instances=1,
        coalesce=True,
    )

    # ── Event drain task ───────────────────────────────────────────────────────
    stop_event = anyio.Event()

    def _handle_signal() -> None:
        log.info("alfred.shutdown_signal")
        stop_event.set()

    async def _event_drain() -> None:
        """Drain and log events from the shared queue."""
        while not stop_event.is_set():
            try:
                event = events.get_nowait()
                log.debug("alfred.event", kind=event.kind, **event.payload)
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.5)

    # ── Start scheduler and block until signal ─────────────────────────────────
    scheduler.start()
    log.info("alfred.scheduler_started", jobs=len(scheduler.get_jobs()))

    async with anyio.create_task_group() as tg:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _handle_signal)

        tg.start_soon(_event_drain)
        await stop_event.wait()
        tg.cancel_scope.cancel()

    # ── Graceful shutdown ──────────────────────────────────────────────────────
    scheduler.shutdown(wait=True)
    log.info("alfred.scheduler_stopped")

    # Final state save
    try:
        state_store.save()
        log.info("alfred.final_save")
    except Exception as e:
        log.warning("alfred.final_save_failed", error=str(e))

    # Surveyor teardown: close embedder HTTP session
    if "surveyor" in active_names:
        try:
            await surveyor.teardown()
        except Exception as e:
            log.warning("alfred.surveyor_teardown_failed", error=str(e))

    log.info("alfred.stopped")
