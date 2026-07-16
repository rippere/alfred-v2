from __future__ import annotations

import asyncio
import fcntl
import json
import os
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

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
    cluster); keys `mine` changed relative to `base` override theirs."""
    theirs_d = theirs.get(key, {}) or {}
    mine_d = mine.get(key, {}) or {}
    base_d = base.get(key, {}) or {}
    merged = dict(theirs_d)
    for k, v in mine_d.items():
        if v != base_d.get(k):
            merged[k] = v
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
    return merged


class StateStore:
    def __init__(self, path: Path, cfg=None) -> None:
        self.path = path
        self._cfg = cfg  # optional AlfredConfig for api_warn_at_calls
        self._state: PipelineState = PipelineState()
        # Guards concurrent writes from multiple scheduler jobs running in the
        # same event loop.  Acquire before any bulk mutation of state.files or
        # state.clusters, and before every save().
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
        self._base_raw: dict = asdict(_decode_state({}))

    def _read_raw(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {}

    def load(self) -> PipelineState:
        raw = self._read_raw()
        self._state = _decode_state(raw)
        self._base_raw = asdict(self._state)
        # Reset daily API counters if date rolled over since last save
        today = date.today().isoformat()
        if self._state.api_calls_date != today:
            self._state.api_calls_today = 0
            self._state.api_cost_usd_today = 0.0
            self._state.api_calls_date = today
        return self._state

    def save(self) -> None:
        """Synchronous save — caller must hold self._lock when called from async context.

        Acquires a cross-process flock on a sidecar lock file, re-reads the
        freshest on-disk state under that lock, three-way merges it against
        what this instance loaded and what it now holds in memory (see
        _merge_pipeline_state), and writes the merged result. Writing itself
        still goes to a temp file in the same directory then os.replace()s it
        into place, so a mid-write kill (OOM, SIGKILL) can never leave a
        truncated state.json behind — the old file stays intact until the
        rename.

        This means a save from one process (e.g. a long-lived query server
        merely bumping an API-budget counter) can no longer silently clobber
        concurrent writes from another (e.g. the daemon recording new embeds
        or cluster state) — both survive the merge.
        """
        lock_fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                theirs_raw = self._read_raw()
                theirs = asdict(_decode_state(theirs_raw))
                mine = asdict(self._state)
                merged = _merge_pipeline_state(self._base_raw, theirs, mine)

                tmp = self.path.with_name(self.path.name + ".tmp")
                tmp.write_text(json.dumps(merged, indent=2))
                os.replace(tmp, self.path)

                # Fold the merge result back into memory so this instance
                # reflects the other writer's changes too, and so the next
                # save() diffs against this save rather than a stale base.
                self._state = _decode_state(merged)
                self._base_raw = merged
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

    async def async_save(self) -> None:
        """Async-safe save — acquires the state lock before writing."""
        async with self._lock:
            self.save()

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
