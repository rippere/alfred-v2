"""Alfred daemon runner — APScheduler-based single-process scheduler.

Replaces the original asyncio.gather() approach with APScheduler's
AsyncIOScheduler.  All daemons share one StateStore, one MilvusStore/LanceDBStore,
and one asyncio.Lock (via StateStore._lock) — eliminating the cross-process race
condition that existed when each systemd service held its own vector store connection.

Each daemon responsibility is exposed as its own one-shot coroutine
(``tick()``, ``structural_tick()``, ``deep_tick()``, ``dedup_tick()``,
``session_archive_tick()``, ``label_tick()``, ``synthesis_tick()``,
``stubs_tick()``) and registered as a separate APScheduler job with its own
id, ``max_instances=1``, ``coalesce=True``, and a cadence-scaled
``misfire_grace_time`` — so a slow pass in one responsibility (e.g. the
weekly dedup) cannot block the scheduling of any other.
"""
from __future__ import annotations

import asyncio
import signal
from datetime import datetime, timedelta, timezone

import anyio
import structlog

log = structlog.get_logger()

# Session archival cadence (janitor.session_archive_tick). Not config-driven:
# the 90-day age threshold lives in the janitor; the sweep just has to run
# often enough to keep up, and daily is plenty. (config.py is owned by another
# workstream — new intervals live here as constants per the split design.)
SESSION_ARCHIVE_INTERVAL_H = 24


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
        if getattr(vector_store, "was_recreated", False):
            # A corrupt table was quarantined and recreated empty.  The surveyor
            # diffs vault md5s against state.files and would see no change —
            # leaving search permanently empty.  Clear embed state so the next
            # tick re-embeds the whole corpus into the fresh table.
            cleared = len(state_store.state.files)
            state_store.state.files.clear()
            state_store.state.clusters.clear()
            state_store.save()
            log.warning("alfred.store_recreated_reembed", cleared_files=cleared)
    else:
        from alfred.store.milvus import MilvusStore
        vector_store = MilvusStore(
            uri=cfg.milvus_uri,
            embed_dims=cfg.embed_dims,
            collection=cfg.milvus_collection,
        )
        log.info("alfred.store", backend="milvus", uri=cfg.milvus_uri)

    # ── Daemon instances ───────────────────────────────────────────────────────
    surveyor = SurveyorDaemon(cfg, state_store, events, store=vector_store)
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

    # Per-job misfire grace: how long past its slot a job may still fire.
    # Scaled to the job's cadence — a daily/weekly job losing its slot to a
    # busy loop should still run hours later, while a 60s poll should not.
    GRACE_FAST = 300        # sub-hourly jobs: tolerate 5-minute slippage
    GRACE_HOURLY = 900      # hourly jobs: tolerate 15 minutes
    GRACE_DAILY = 3600      # daily jobs: tolerate 1 hour
    GRACE_WEEKLY = 21600    # weekly jobs: tolerate 6 hours

    def _add(
        daemon_name: str,
        func,
        trigger: str,
        misfire_grace_time: int = GRACE_FAST,
        **trigger_kwargs,
    ) -> None:
        """Register a scheduler job only when its daemon is selected."""
        if daemon_name not in active_names:
            return
        scheduler.add_job(
            func,
            trigger,
            id=f"{daemon_name}_{func.__name__}",
            max_instances=1,          # never overlap a slow job with itself
            coalesce=True,            # if late, run once not many times
            misfire_grace_time=misfire_grace_time,
            **trigger_kwargs,
        )

    # ── Surveyor: poll every 60s, recluster every 30min ───────────────────────
    _add("surveyor", surveyor.tick, "interval", seconds=60)
    _add("surveyor", surveyor.recluster, "interval", seconds=1800)

    # ── Janitor: four independently-scheduled responsibilities ────────────────
    # Each gets its own job id + grace so a slow pass in one (e.g. the O(n²)
    # weekly dedup) can never block the hourly structural lint or vice versa.
    sweep_s = int(getattr(cfg, "janitor_sweep_interval_s", 3600))
    deep_h = int(getattr(cfg, "janitor_deep_interval_h", 24))
    _add("janitor", janitor.structural_tick, "interval",
         misfire_grace_time=GRACE_HOURLY, seconds=sweep_s)
    _add("janitor", janitor.deep_tick, "interval",
         misfire_grace_time=GRACE_DAILY, hours=deep_h)
    _add("janitor", janitor.session_archive_tick, "interval",
         misfire_grace_time=GRACE_DAILY, hours=SESSION_ARCHIVE_INTERVAL_H)
    if getattr(cfg, "janitor_dedup_enabled", False):
        _add("janitor", janitor.dedup_tick, "interval",
             misfire_grace_time=GRACE_WEEKLY, weeks=1)

    # ── Distiller: nightly at 2am, with startup catch-up ─────────────────────────
    # cron, not interval: interval-24h restarts its countdown on every daemon
    # restart, and this daemon restarts far more often than daily (game-guard
    # pauses, debugging) — the 24h mark was never reached and the distiller
    # produced ZERO scheduled runs after 2026-05-14. The catch-up one-shot covers
    # the original concern behind the interval choice (machine off/paused at 2am):
    # if the last run is >36h stale, run once shortly after startup instead.
    if "distiller" in active_names:
        if getattr(cfg, "distiller_mode", "on_demand") == "scheduled":
            _add("distiller", distiller.tick, "cron", hour=2)
            last_run_ts: str | None = None
            try:
                runs = getattr(state_store.load(), "distiller_runs", None) or []
                last_run_ts = runs[-1].get("timestamp") if runs else None
                stale = last_run_ts is None or (
                    datetime.now(timezone.utc) - datetime.fromisoformat(last_run_ts)
                ) > timedelta(hours=36)
            except Exception:
                stale = True
            if stale:
                scheduler.add_job(
                    distiller.tick,
                    "date",
                    run_date=datetime.now(timezone.utc) + timedelta(minutes=10),
                    id="distiller_catchup",
                )
                log.info("distiller.catchup_scheduled", last_run=last_run_ts)
        else:
            log.info("distiller.scheduled_skipped", reason="distiller_mode=on_demand")

    # ── Consolidator: three independently-scheduled responsibilities ──────────
    # Labeling, synthesis, and wiki-stub generation communicate only through
    # persisted state.clusters (synthesis skips unlabeled clusters and picks
    # them up next interval), so a slow LLM synthesis batch can no longer
    # delay labeling or stub generation. All keep the legacy 30-min cadence.
    consolidate_s = int(getattr(cfg, "consolidator_min_interval_s", 1800))
    _add("consolidator", consolidator.label_tick, "interval", seconds=consolidate_s)
    _add("consolidator", consolidator.synthesis_tick, "interval", seconds=consolidate_s)
    _add("consolidator", consolidator.stubs_tick, "interval", seconds=consolidate_s)

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

    # ── Hourly housekeeping: log rotation + retention ──────────────────────────
    # The daemon's stdout/stderr is an O_APPEND fd pointing at data/alfred.log
    # (see cli.py `up --daemon`), so we cannot rename the inode out from under
    # the running process. copytruncate — copy contents aside, then truncate in
    # place — keeps the existing fd valid: because of O_APPEND every write seeks
    # to end-of-file atomically, so the next log line lands at the new offset 0.
    #
    # Retention: at most ``log_backups`` rotated generations (alfred.log.1..N)
    # are kept — anything older is deleted — and only the newest
    # ``state_backups`` state.json.backup-* snapshots survive each sweep.
    def _housekeeping(
        max_bytes: int = 50 * 1024 * 1024,
        log_backups: int = 2,
        state_backups: int = 3,
    ) -> None:
        import shutil

        # 1. copytruncate rotation when alfred.log exceeds max_bytes.
        log_path = cfg.data_dir / "alfred.log"
        try:
            if log_path.exists() and log_path.stat().st_size >= max_bytes:
                for i in range(log_backups - 1, 0, -1):
                    src = cfg.data_dir / f"alfred.log.{i}"
                    if src.exists():
                        src.replace(cfg.data_dir / f"alfred.log.{i + 1}")
                shutil.copy2(log_path, cfg.data_dir / "alfred.log.1")
                with open(log_path, "r+") as f:
                    f.truncate(0)
                log.info("alfred.log_rotated", max_bytes=max_bytes)
        except Exception as e:
            log.warning("alfred.log_rotate_failed", error=str(e))

        # 2. Delete rotated generations beyond the retention cap.
        try:
            for old in cfg.data_dir.glob("alfred.log.*"):
                suffix = old.name.rsplit(".", 1)[-1]
                if suffix.isdigit() and int(suffix) > log_backups:
                    old.unlink()
                    log.info("alfred.log_pruned", path=str(old))
        except Exception as e:
            log.warning("alfred.log_prune_failed", error=str(e))

        # 3. Keep only the newest state.json.backup-* snapshots.  Names embed
        # a YYYYMMDD-HHMMSS timestamp, so lexical order is chronological.
        try:
            backups = sorted(cfg.data_dir.glob("state.json.backup-*"))
            for old in backups[: max(0, len(backups) - state_backups)]:
                old.unlink()
                log.info("alfred.state_backup_pruned", path=str(old))
        except Exception as e:
            log.warning("alfred.state_backup_prune_failed", error=str(e))

    async def _housekeeping_job() -> None:
        # stat()/copy can touch a large file — keep it off the event loop.
        await anyio.to_thread.run_sync(_housekeeping)

    scheduler.add_job(
        _housekeeping_job,
        "interval",
        hours=1,
        id="housekeeping",
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
