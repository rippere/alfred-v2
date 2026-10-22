from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
from dataclasses import fields
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import anyio
import anyio.to_thread

from alfred.core.failures import drain_failures, restore_failures
from alfred.core.models import (
    ClusterState,
    FileState,
    MemoryStrength,
    PipelineState,
    WikiPage,
)

import structlog

# Use structlog (like every other module) — the budget-guard warnings below pass
# structured kwargs (daemon=, calls_today=, ...). A stdlib logger rejects those with
# "Logger._log() got an unexpected keyword argument 'daemon'", which since the Jun-5
# budget guards + distiller-cron fix turned into ~1.5k distiller.file_error/run.
_log = structlog.get_logger(__name__)

# Anthropic claude-sonnet-4-6 pricing (USD per token)
_PRICE_INPUT = 3.00 / 1_000_000
_PRICE_CACHE_READ = 0.30 / 1_000_000
_PRICE_OUTPUT = 15.00 / 1_000_000

# Recognized API failure signatures → pause duration in seconds.  When a
# daemon reports an error matching one of these substrings (via
# record_api_failure), can_make_api_call() returns False for every daemon
# until the pause expires — retrying is pointless until the underlying
# condition (e.g. account credit) changes.
_FAILURE_SIGNATURES: tuple[tuple[str, float], ...] = (
    ("credit balance is too low", 3600.0),
    ("balance is too low", 3600.0),
)


def _decode_state(raw: dict) -> PipelineState:
    state = PipelineState(
        version=raw.get("version", 1),
        last_run=raw.get("last_run", ""),
        curator_processed=raw.get("curator_processed", {}),
        distiller_runs=raw.get("distiller_runs", []),
        janitor_sweeps=raw.get("janitor_sweeps", []),
        api_calls_today=raw.get("api_calls_today", 0),
        api_calls_date=raw.get("api_calls_date", ""),
        api_cost_usd_today=raw.get("api_cost_usd_today", 0.0),
        api_paused_until=raw.get("api_paused_until", ""),
        error_counts=raw.get("error_counts", {}),
    )
    for rel_path, f in raw.get("files", {}).items():
        state.files[rel_path] = FileState(**{
            k: v for k, v in f.items()
            if k in FileState.__dataclass_fields__
        })
    for key, c in raw.get("clusters", {}).items():
        state.clusters[key] = ClusterState(**{
            k: v for k, v in c.items()
            if k in ClusterState.__dataclass_fields__
        })
    for rel_path, m in raw.get("memory", {}).items():
        state.memory[rel_path] = MemoryStrength(**{
            k: v for k, v in m.items()
            if k in MemoryStrength.__dataclass_fields__
        })
    for key, w in raw.get("wiki_pages", {}).items():
        state.wiki_pages[key] = WikiPage(**{
            k: v for k, v in w.items()
            if k in WikiPage.__dataclass_fields__
        })
    return state


# --- Cross-process merge-on-save --------------------------------------------
#
# state.json is constructed independently by up to six call sites (runner,
# query engine, cli, meta_server, server, server_http) — the daemon and any
# long-lived MCP/CLI query process each hold their own in-memory StateStore
# built from whatever state.json looked like at their own load() time. With a
# plain "serialize the raw dict and os.replace()" save(), a query process that
# loaded hours ago and merely bumps an API-budget counter would, on its next
# save, blow away every embed/cluster/wiki-page write the daemon made in the
# meantime — the file lock alone only prevents torn/interleaved writes, it
# does nothing to stop this last-writer-wins data loss.
#
# save() below does a real three-way merge (base = what this instance loaded,
# theirs = freshest on-disk content re-read under the lock, mine = this
# instance's current in-memory state) so independent additions from both
# sides survive, not just whichever process happened to save last.
def _merge_dict_field(key: str, base: dict, theirs: dict, mine: dict) -> dict:
    """Union-merge a dict-valued field: keys unknown to `mine` are kept from
    `theirs` (another writer's addition, e.g. a newly embedded file or
    cluster); keys `mine` changed relative to `base` override theirs.

    Keys `mine` *removed* relative to `base` (e.g. janitor ghost-file pruning
    or dedup deletion) are removed from the merged result too — unless
    `theirs` independently changed that same key relative to `base` since
    this instance loaded, i.e. a genuine concurrent edit-vs-delete conflict.
    We resolve that conflict by keeping theirs's value: a live concurrent
    edit wins over a stale delete, on the judgment that silently discarding
    another writer's fresh update is worse than a deleted entry occasionally
    reappearing (it will simply be deleted again on the next sweep)."""
    theirs_d = theirs.get(key, {}) or {}
    mine_d = mine.get(key, {}) or {}
    base_d = base.get(key, {}) or {}
    merged = dict(theirs_d)
    for k, v in mine_d.items():
        if v != base_d.get(k):
            merged[k] = v
    for k in base_d:
        if k in mine_d or k not in merged:
            continue
        if theirs_d.get(k) == base_d.get(k):
            del merged[k]
    return merged


def _merge_list_field(key: str, base: dict, theirs: dict, mine: dict) -> list:
    """Append-merge a log-like list field: entries `mine` added since `base`
    (and not already present in `theirs`) are appended, so two writers
    recording distiller/janitor runs concurrently both survive."""
    theirs_l = list(theirs.get(key, []) or [])
    mine_l = mine.get(key, []) or []
    base_l = base.get(key, []) or []
    for entry in mine_l:
        if entry not in base_l and entry not in theirs_l:
            theirs_l.append(entry)
    return theirs_l


def _merge_scalar(key: str, base: dict, theirs: dict, mine: dict):
    """Changed-since-base wins; otherwise keep the freshest on-disk value."""
    mine_v = mine.get(key)
    if mine_v != base.get(key):
        return mine_v
    return theirs.get(key)


def _merge_counters(base: dict, theirs: dict, mine: dict) -> tuple[int, float]:
    """Merge api_calls_today / api_cost_usd_today additively (CRDT-style
    counter) when both sides agree on api_calls_date, so concurrent
    record_api_call() bumps from two processes both land instead of one
    clobbering the other. Falls back to changed-wins across a day rollover,
    where a delta merge could double count."""
    b_date = base.get("api_calls_date", "")
    t_date = theirs.get("api_calls_date", "")
    m_date = mine.get("api_calls_date", "")
    if b_date == t_date == m_date:
        calls = theirs.get("api_calls_today", 0) + (
            mine.get("api_calls_today", 0) - base.get("api_calls_today", 0)
        )
        cost = theirs.get("api_cost_usd_today", 0.0) + (
            mine.get("api_cost_usd_today", 0.0) - base.get("api_cost_usd_today", 0.0)
        )
        return calls, cost
    calls = _merge_scalar("api_calls_today", base, theirs, mine)
    cost = _merge_scalar("api_cost_usd_today", base, theirs, mine)
    return calls, cost


def _merge_error_counts(base: dict, theirs: dict, mine: dict) -> dict[str, int]:
    """Additively merge the swallowed-failure counters, CRDT-style, for the
    same reason as _merge_counters: two processes each swallowing an error
    must produce a total of two, not one. A plain dict-field merge would take
    whichever side changed the key and discard the other's increment.

    Unlike the API counters there is no date rollover to guard — error_counts
    is monotonic for the life of the state file."""
    theirs_d = theirs.get("error_counts", {}) or {}
    mine_d = mine.get("error_counts", {}) or {}
    base_d = base.get("error_counts", {}) or {}
    merged = dict(theirs_d)
    for key, mine_v in mine_d.items():
        delta = mine_v - base_d.get(key, 0)
        if delta:
            merged[key] = merged.get(key, 0) + delta
    return merged


def _merge_pipeline_state(base: dict, theirs: dict, mine: dict) -> dict:
    """Three-way merge of the on-disk PipelineState dict shape. `base` is the
    snapshot this StateStore instance last loaded, `theirs` is the freshest
    on-disk content (re-read under the file lock, may include another
    process's writes since `base`), `mine` is this instance's current
    in-memory state."""
    merged = dict(theirs)
    for f in ("version", "last_run", "last_dedup", "api_paused_until", "api_calls_date"):
        merged[f] = _merge_scalar(f, base, theirs, mine)
    for f in ("files", "clusters", "memory", "wiki_pages", "curator_processed"):
        merged[f] = _merge_dict_field(f, base, theirs, mine)
    for f in ("distiller_runs", "janitor_sweeps"):
        merged[f] = _merge_list_field(f, base, theirs, mine)
    merged["api_calls_today"], merged["api_cost_usd_today"] = _merge_counters(base, theirs, mine)
    merged["error_counts"] = _merge_error_counts(base, theirs, mine)
    return merged



def _consume_orphaned_save_error(task: "asyncio.Future") -> None:
    """Done-callback for async_save's shielded task: if the awaiting caller was
    cancelled, nobody retrieves the task's exception, so log it here instead of
    letting asyncio report "exception was never retrieved"."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        _log.debug("state.save_task_error", error=str(exc), kind=type(exc).__name__)


# --- Snapshot / fold-back helpers -------------------------------------------
#
# Why these exist: save() used to be one synchronous block on the event loop
# (flock, read ~1.5 MB, decode, two full asdict() passes over ~21k FileState
# entries, merge, json.dumps(indent=2), write, replace). Under the daemon's
# memory pressure that block ran past systemd's 30 s TimeoutStopSec on every
# stop since 2026-10-13, and BaseDaemon.save_state() ran it every 25 surveyor
# files, freezing the loop each time.
#
# The split that makes an off-loop save race-free:
#   1. SNAPSHOT on the event-loop thread, synchronously (no await), so no
#      coroutine can mutate state mid-copy. Every in-process mutator of
#      PipelineState runs on the loop thread; the janitor's reaper worker
#      threads only read it.
#   2. WRITE in a worker thread: it touches only the snapshot, the base and
#      freshly-read disk data, never the live state.
#   3. FOLD back on the loop thread, in place, key by key: changes the merge
#      took from disk (another process's writes) are applied only where the
#      loop has not touched that key since the snapshot. The live
#      PipelineState, its dicts and its record objects are never rebound.
#
# The in-place fold also fixes a data-loss bug. The old save() did
# `self._state = _decode_state(merged)`, which orphaned every
# `state = self.state.state` reference a daemon held across an await. The
# surveyor holds one for a whole diff and saves every 25 files, so only the
# first 25 files of each tick reached state.json, and the next tick
# re-embedded everything else again.

_MISSING = object()

# PipelineState fields that map a key to a record dataclass.
_RECORD_MAPS: dict[str, type] = {
    "files": FileState,
    "clusters": ClusterState,
    "memory": MemoryStrength,
    "wiki_pages": WikiPage,
}
_RECORD_FIELDS: dict[str, tuple[str, ...]] = {
    name: tuple(f.name for f in fields(cls)) for name, cls in _RECORD_MAPS.items()
}
_STATE_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(PipelineState))
_COUNTER_FIELDS = ("api_calls_today", "api_cost_usd_today", "api_calls_date")
_LOG_LIST_FIELDS = ("distiller_runs", "janitor_sweeps")

_LOCK_POLL_S = 0.05

# In-flight async_save tasks. Held here so a save whose caller was cancelled
# can't be garbage-collected mid-flight.
_SAVE_TASKS: set = set()


class StateSaveTimeout(TimeoutError):
    """A deadline-bounded save gave up. Nothing was replaced, so state.json still
    holds the previous complete save."""


def _copy_value(v):
    """Copy JSON-shaped containers, so the snapshot shares nothing mutable with
    the live state and the live state shares nothing with the merge base."""
    if isinstance(v, list):
        return [_copy_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _copy_value(x) for k, x in v.items()}
    return v


def _record_to_raw(rec, names: tuple[str, ...]) -> dict:
    return {n: _copy_value(getattr(rec, n)) for n in names}


def _state_to_raw(state: PipelineState) -> dict:
    """Same result as dataclasses.asdict(state), at a fraction of the cost.
    asdict() recurses generically and deep-copies every leaf. Records here hold
    only scalars and lists/dicts of scalars, so a typed copy is enough."""
    raw: dict = {}
    for name in _STATE_FIELDS:
        value = getattr(state, name)
        if name in _RECORD_MAPS:
            rnames = _RECORD_FIELDS[name]
            raw[name] = {k: _record_to_raw(r, rnames) for k, r in value.items()}
        else:
            raw[name] = _copy_value(value)
    return raw


def _flock_until(fd: int, deadline: float | None) -> None:
    if deadline is None:
        fcntl.flock(fd, fcntl.LOCK_EX)
        return
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise StateSaveTimeout("timed out waiting for the state.json lock")
            time.sleep(_LOCK_POLL_S)


def _check_deadline(deadline: float | None, stage: str) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise StateSaveTimeout(f"save deadline passed before {stage}")


def _fold_delta(snap: dict, merged: dict) -> dict:
    """What the merge changed relative to the snapshot: for map fields, the keys
    that differ, and for everything else, True. Computed in the worker thread so
    the loop-side fold only visits what actually changed. When the daemon is
    the only writer, that is usually nothing."""
    delta: dict = {}
    for name in _STATE_FIELDS:
        m, s = merged.get(name), snap.get(name)
        if isinstance(m, dict) and isinstance(s, dict):
            keys = [k for k, v in m.items() if s.get(k, _MISSING) != v]
            keys.extend(k for k in s if k not in m)
            if keys:
                delta[name] = keys
        elif m != s:
            delta[name] = True
    return delta


class StateStore:
    def __init__(self, path: Path, cfg=None) -> None:
        self.path = path
        self._cfg = cfg  # optional AlfredConfig for api_warn_at_calls
        self._state: PipelineState = PipelineState()
        # Serializes async saves within one event loop: snapshot, write and
        # fold-back must not interleave with another save's.
        self._lock: asyncio.Lock = asyncio.Lock()
        # Cross-process exclusive lock, held around the read-modify-write in
        # save(). A sidecar ".lock" file (not state.json itself) so the lock
        # is independent of state.json's own atomic tmp+replace lifecycle.
        # flock() serializes both separate processes AND separate instances
        # within one process (each open() call gets its own file description,
        # which flock arbitrates between) — see test_state_lock.py.
        self._lock_path = path.with_name(path.name + ".lock")
        # Snapshot of what this instance last loaded/saved, in PipelineState's
        # asdict() shape — the merge "base" for the next save(). Starts as an
        # empty-state baseline so a save() before any load() still merges
        # correctly against a freshly-read disk file rather than clobbering it.
        # Never shares a mutable object with self._state (see _fold_merged).
        self._base_raw: dict = _state_to_raw(_decode_state({}))

    def _read_raw(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}

    def load(self) -> PipelineState:
        raw = self._read_raw()
        self._state = _decode_state(raw)
        self._base_raw = _state_to_raw(self._state)
        # Reset daily API counters if date rolled over since last save
        today = date.today().isoformat()
        if self._state.api_calls_date != today:
            self._state.api_calls_today = 0
            self._state.api_cost_usd_today = 0.0
            self._state.api_calls_date = today
        return self._state

    # -- save pipeline: snapshot (loop) -> write (any thread) -> fold (loop) --

    def _take_snapshot(self) -> tuple[dict, dict]:
        """Copy the live state. Must run on the thread that owns the state (the
        event loop, in the daemon) with no await in between, which is what
        makes it atomic with respect to every mutator.

        Folds in any failures swallowed since the last save first, so a handler
        whose body is `pass` still leaves a durable count. They are drained
        (read and zeroed) so repeated saves can't double count. The caller
        restores them if the write fails, so a disk error doesn't also erase
        the record of the earlier errors."""
        drained = drain_failures()
        for _key, _n in drained.items():
            self._state.error_counts[_key] = self._state.error_counts.get(_key, 0) + _n
        return _state_to_raw(self._state), drained

    def _unfold_failures(self, drained: dict) -> None:
        """Undo _take_snapshot's fold after a failed or timed-out write. The
        drained counts go back into the in-process counters for the next
        save, and come back out of the in-memory error_counts. Previously
        they stayed in both, so the next successful save counted them twice."""
        counts = self._state.error_counts
        for key, n in drained.items():
            left = counts.get(key, 0) - n
            if left > 0:
                counts[key] = left
            else:
                counts.pop(key, None)
        restore_failures(drained)

    def _write_merged(
        self, snap: dict, base: dict, deadline: float | None = None
    ) -> tuple[dict, dict]:
        """The blocking half of a save. Safe to run in a worker thread: it reads
        only `snap` and `base`, which nothing else mutates, plus the file on
        disk.

        Takes a cross-process flock on a sidecar lock file, re-reads the
        freshest on-disk state under that lock, three-way merges it against
        what this instance last loaded or saved (`base`) and what it holds now
        (`snap`), as described at _merge_pipeline_state, and writes the
        merged result. The write goes to a temp file in the same directory,
        then os.replace()s it into place. A mid-write kill (OOM, SIGKILL) or
        a missed deadline can never leave a truncated state.json behind: the
        old file stays intact until the rename.

        With a `deadline` (time.monotonic() value), it never waits on the lock
        past it and never replaces the file after it. Instead it raises
        StateSaveTimeout and leaves the previous state.json untouched.

        Returns (merged, delta) for _fold_merged."""
        _check_deadline(deadline, "starting")
        lock_fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            _flock_until(lock_fd, deadline)
            try:
                theirs = _state_to_raw(_decode_state(self._read_raw()))
                merged = _merge_pipeline_state(base, theirs, snap)
                # Compact: nothing reads state.json line-by-line, and
                # indent=2 roughly doubles both the dump time and the bytes.
                payload = json.dumps(merged, separators=(",", ":"))

                tmp = self.path.with_name(self.path.name + ".tmp")
                try:
                    _check_deadline(deadline, "writing")
                    tmp.write_text(payload)
                    _check_deadline(deadline, "replacing state.json")
                    os.replace(tmp, self.path)
                except BaseException:
                    tmp.unlink(missing_ok=True)
                    raise
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
        return merged, _fold_delta(snap, merged)

    def _fold_merged(self, snap: dict, merged: dict, delta: dict) -> None:
        """Fold what the merge pulled in from disk back into the live state, in
        place, then make `merged` the base for the next save.

        Runs on the state-owning thread. For each key the merge changed
        relative to `snap`:
          * if the live value still equals the snapshot, the loop hasn't
            touched it since, so take the merged value (another writer's
            change, or its deletion);
          * otherwise keep the live value. It is a fresher local edit, and
            since it differs from the new base, the next save writes it.
        Counters merge additively, matching _merge_counters and
        _merge_error_counts.

        Records are updated in place (setattr on the existing object), dicts
        are mutated in place and lists are slice-assigned. Every reference a
        daemon is holding stays live."""
        st = self._state
        for name, keys in delta.items():
            if name in _RECORD_MAPS:
                self._fold_records(name, keys, snap[name], merged[name])
            elif name == "error_counts":
                live = st.error_counts
                for k in keys:
                    d = merged[name].get(k, 0) - snap[name].get(k, 0)
                    if d:
                        live[k] = live.get(k, 0) + d
            elif name in _COUNTER_FIELDS:
                continue  # handled together below
            elif name in _LOG_LIST_FIELDS:
                live = getattr(st, name)
                s_list, m_list = snap[name], merged[name]
                if live == s_list:
                    live[:] = _copy_value(m_list)
                else:
                    added = [e for e in live if e not in s_list and e not in m_list]
                    live[:] = _copy_value(m_list) + _copy_value(added)
            elif isinstance(keys, list):  # plain str->scalar map (curator_processed)
                live = getattr(st, name)
                s_map, m_map = snap[name], merged[name]
                for k in keys:
                    if live.get(k, _MISSING) != s_map.get(k, _MISSING):
                        continue
                    m_v = m_map.get(k, _MISSING)
                    if m_v is _MISSING:
                        live.pop(k, None)
                    else:
                        live[k] = _copy_value(m_v)
            else:  # scalar
                if getattr(st, name) == snap[name]:
                    setattr(st, name, merged[name])

        if any(f in delta for f in _COUNTER_FIELDS):
            s_date, m_date = snap["api_calls_date"], merged["api_calls_date"]
            if st.api_calls_date == s_date:
                if m_date == s_date:
                    st.api_calls_today += merged["api_calls_today"] - snap["api_calls_today"]
                    st.api_cost_usd_today += (
                        merged["api_cost_usd_today"] - snap["api_cost_usd_today"]
                    )
                else:
                    # Another writer rolled the day over; adopt its counters.
                    st.api_calls_date = m_date
                    st.api_calls_today = merged["api_calls_today"]
                    st.api_cost_usd_today = merged["api_cost_usd_today"]
            # else: the loop rolled the day over itself during the save. Keep the
            # local counters; the next save's _merge_counters reconciles them.

        self._base_raw = merged

    def _fold_records(self, name: str, keys: list, s_map: dict, m_map: dict) -> None:
        live = getattr(self._state, name)
        rnames = _RECORD_FIELDS[name]
        cls = _RECORD_MAPS[name]
        for k in keys:
            rec = live.get(k, _MISSING)
            live_raw = _MISSING if rec is _MISSING else _record_to_raw(rec, rnames)
            if live_raw != s_map.get(k, _MISSING):
                continue  # touched since the snapshot: keep the local edit
            m_v = m_map.get(k, _MISSING)
            if m_v is _MISSING:
                live.pop(k, None)
            elif rec is _MISSING:
                live[k] = cls(**{f: _copy_value(v) for f, v in m_v.items() if f in rnames})
            else:
                for f in rnames:
                    if f in m_v:
                        setattr(rec, f, _copy_value(m_v[f]))

    def save(self, *, timeout: float | None = None) -> None:
        """Synchronous save, for callers that have no event loop (CLI, startup,
        query processes). Inside the daemon's loop, use `await async_save()`:
        it does the same work without blocking the loop. Don't call this on
        the loop while an async_save() of the same store is in flight.

        This means a save from one process (e.g. a long-lived query server
        merely bumping an API-budget counter) can no longer silently clobber
        concurrent writes from another (e.g. the daemon recording new embeds
        or cluster state). Both survive the merge, see _write_merged.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        snap, drained = self._take_snapshot()
        try:
            merged, delta = self._write_merged(snap, self._base_raw, deadline)
        except BaseException:
            self._unfold_failures(drained)
            raise
        self._fold_merged(snap, merged, delta)

    async def async_save(self, *, timeout: float | None = None) -> None:
        """Event-loop-safe save: the snapshot and fold-back run on the loop, and
        the lock wait, disk read, merge, dump and write run in a worker thread.

        With `timeout`, the save gives up (StateSaveTimeout) instead of waiting
        on the cross-process lock or replacing the file past the deadline, and
        state.json keeps its previous contents.

        The save runs as its own shielded task. A caller cancelled mid-save
        (the scheduler cancels in-flight jobs on shutdown) stops waiting, but
        the save itself still finishes and folds back. Dropping the fold after
        a completed write would leave a stale merge base, which double counts
        the API counters on the next save."""
        deadline = None if timeout is None else time.monotonic() + timeout
        task = asyncio.ensure_future(self._async_save_locked(deadline))
        _SAVE_TASKS.add(task)  # strong ref: the loop only keeps weak ones
        task.add_done_callback(_SAVE_TASKS.discard)
        task.add_done_callback(_consume_orphaned_save_error)
        await asyncio.shield(task)

    async def _async_save_locked(self, deadline: float | None) -> None:
        async with self._lock:
            snap, drained = self._take_snapshot()
            try:
                merged, delta = await anyio.to_thread.run_sync(
                    self._write_merged, snap, self._base_raw, deadline,
                    abandon_on_cancel=False,
                )
            except BaseException:
                self._unfold_failures(drained)
                raise
            self._fold_merged(snap, merged, delta)


    @property
    def state(self) -> PipelineState:
        return self._state

    # Convenience counters for status display
    def file_count(self) -> int:
        return len(self._state.files)

    def embedded_count(self) -> int:
        return sum(1 for f in self._state.files.values() if f.last_embedded)

    def cluster_count(self) -> int:
        return len(self._state.clusters)

    def wiki_page_count(self) -> int:
        return len(self._state.wiki_pages)

    def chunk_count(self) -> int:
        return sum(len(f.chunk_ids) for f in self._state.files.values())

    def record_api_call(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> None:
        """Record a single Anthropic API call and accumulate cost estimate.

        Resets the daily counters if the stored date differs from today.
        Logs a warning if the call count hits api_warn_at_calls.

        Args:
            input_tokens: Non-cached input tokens consumed.
            output_tokens: Output tokens generated.
            cached_tokens: Cache-read input tokens (charged at reduced rate).
        """
        today = date.today().isoformat()
        state = self._state

        # Reset daily counters on date rollover
        if state.api_calls_date != today:
            state.api_calls_today = 0
            state.api_cost_usd_today = 0.0
            state.api_calls_date = today

        state.api_calls_today += 1
        state.api_cost_usd_today += (
            input_tokens * _PRICE_INPUT
            + cached_tokens * _PRICE_CACHE_READ
            + output_tokens * _PRICE_OUTPUT
        )

        warn_threshold = (
            self._cfg.api_warn_at_calls if self._cfg is not None else 400
        )
        if state.api_calls_today >= warn_threshold:
            _log.warning(
                "alfred.api_budget_warning",
                calls_today=state.api_calls_today,
                cost_usd=round(state.api_cost_usd_today, 4),
            )

    def budget_remaining(self) -> int:
        """Return API calls remaining today. Resets counter if date rolled over."""
        today = date.today().isoformat()
        state = self._state
        if state.api_calls_date != today:
            state.api_calls_today = 0
            state.api_cost_usd_today = 0.0
            state.api_calls_date = today
        limit = self._cfg.api_max_calls_per_day if self._cfg is not None else 500
        return max(0, limit - state.api_calls_today)

    def record_api_failure(self, error: str, daemon: str = "") -> None:
        """Match an API error against recognized failure signatures and pause.

        On a match (e.g. Anthropic credit exhaustion) a pause-until timestamp
        is recorded in state, so can_make_api_call() gates ALL daemons — not
        just the caller — for the pause duration.  Unrecognized errors are
        ignored: transient faults should keep retrying on their normal cadence.
        """
        for signature, pause_s in _FAILURE_SIGNATURES:
            if signature in error:
                until = datetime.now(timezone.utc) + timedelta(seconds=pause_s)
                self._state.api_paused_until = until.isoformat()
                _log.warning(
                    "alfred.api_paused",
                    daemon=daemon,
                    signature=signature,
                    resume_at=self._state.api_paused_until,
                )
                return

    def _is_paused(self, daemon: str = "") -> bool:
        """Return True while a recorded failure pause is still in effect."""
        paused_until = self._state.api_paused_until
        if not paused_until:
            return False
        try:
            until = datetime.fromisoformat(paused_until)
        except ValueError:
            self._state.api_paused_until = ""
            return False
        if datetime.now(timezone.utc) < until:
            _log.debug("alfred.api_paused_skip", daemon=daemon, resume_at=paused_until)
            return True
        self._state.api_paused_until = ""
        return False

    def can_make_api_call(self, daemon: str = "") -> bool:
        """Return False and log when daily budget is exhausted or a failure pause is active."""
        if self._is_paused(daemon):
            return False
        remaining = self.budget_remaining()
        if remaining <= 0:
            _log.warning(
                "alfred.api_budget_exhausted",
                daemon=daemon,
                calls_today=self._state.api_calls_today,
                cost_usd=round(self._state.api_cost_usd_today, 4),
                limit=self._cfg.api_max_calls_per_day if self._cfg else 500,
            )
            return False
        return True
