from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import date
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


class StateStore:
    def __init__(self, path: Path, cfg=None) -> None:
        self.path = path
        self._cfg = cfg  # optional AlfredConfig for api_warn_at_calls
        self._state: PipelineState = PipelineState()
        # Guards concurrent writes from multiple scheduler jobs running in the
        # same event loop.  Acquire before any bulk mutation of state.files or
        # state.clusters, and before every save().
        self._lock: asyncio.Lock = asyncio.Lock()

    def load(self) -> PipelineState:
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self._state = _decode_state(raw)
        else:
            self._state = PipelineState()
        # Reset daily API counters if date rolled over since last save
        today = date.today().isoformat()
        if self._state.api_calls_date != today:
            self._state.api_calls_today = 0
            self._state.api_cost_usd_today = 0.0
            self._state.api_calls_date = today
        return self._state

    def save(self) -> None:
        """Synchronous save — caller must hold self._lock when called from async context."""
        self.path.write_text(json.dumps(asdict(self._state), indent=2))

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

    def can_make_api_call(self, daemon: str = "") -> bool:
        """Return False and log when daily budget is exhausted."""
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
