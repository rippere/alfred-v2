"""DistillerDaemon — scan vault records → extract learnings → create learn/ records."""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import frontmatter
import structlog

from alfred.core.failures import record_failure
from alfred.core.local_llm import LocalLLMRequestTooLarge, LocalLLMUnavailable, complete
from alfred.core.provenance import is_daemon_generated
from alfred.core.vault_ops import VaultError, vault_append_to_topic, vault_read
from alfred.daemons.base import BaseDaemon

log = structlog.get_logger()

DISTILL_INTERVAL = 86400.0   # run once per day
STALE_DAYS = 30              # re-distill file if not distilled in 30 days
MIN_BODY_LEN = 200           # skip files with trivial bodies

# Canonical topic slug map — variant tags → canonical slug.
# Prevents the distiller from spawning duplicate topic files for the same concept.
# Keys are raw tag strings that Claude might emit; values are the canonical slug
# of the richest existing topic file.
_TAG_CANONICAL: dict[str, str] = {
    # --- ai / agents / llm cluster ---
    "agent":                   "ai-agents",
    "agents":                  "ai-agents",
    "agentic":                 "ai-agents",
    "agentic-ai":              "ai-agents",
    "agentic-systems":         "ai-agents",
    "agentic-workflow":        "ai-agents",
    "agentic-builds":          "ai-agents",
    "agentic-coding":          "ai-agents",
    "agent-systems":           "ai-agents",
    "agent-setup":             "ai-agents",
    "agent-orchestration":     "ai-agents",
    "agent-design":            "ai-agents",
    "agent-architecture":      "ai-agents",
    "multi-agent":             "ai-agents",
    "llm-systems":             "llm",
    "llm-architecture":        "llm",
    "llm-pipelines":           "llm",
    "llm-workflows":           "llm",
    "local-llm":               "llm",
    "local-ai":                "llm",
    "local-inference":         "llm",
    "local-ml":                "llm",
    "foundation-models":       "llm",
    # ai-systems → ai (most content)
    "ai-systems":              "ai",
    "ai-systems-design":       "ai",
    "artificial-intelligence": "ai",
    # --- knowledge cluster ---
    "knowledge-systems":       "knowledge-management",
    "knowledge-organization":  "knowledge-management",
    "knowledge-architecture":  "knowledge-management",
    "knowledge-quality":       "knowledge-management",
    "knowledge-integrity":     "knowledge-management",
    "knowledge-capture":       "knowledge-management",
    "knowledge-distillation":  "knowledge-management",
    "knowledge-extraction":    "knowledge-management",
    "knowledge-transfer":      "knowledge-management",
    "pkm":                     "knowledge-management",
    "personal-knowledge-management": "knowledge-management",
    "second-brain":            "knowledge-management",
    # --- workflow cluster ---
    "workflows":               "workflow",
    "workflow-automation":     "workflow",
    "workflow-design":         "workflow",
    "workflow-evolution":      "workflow",
    "workflow-optimization":   "workflow",
    "workflow-orchestration":  "workflow",
    "workflow-sequencing":     "workflow",
    # --- architecture / system-design cluster ---
    "systems-design":          "system-design",
    "software-design":         "software-architecture",
    "software-engineering":    "software-architecture",
    # --- graph cluster ---
    "knowledge-graphs":        "knowledge-graph",
    # --- knowledge-graph → system-design (graph is a design tool) ---
    # (kept separate — they're distinct enough)
}


_EXTRACT_SYSTEM = """\
You are a personal knowledge distiller. Read vault records and extract the most \
reusable, transferable insights or lessons — things worth remembering and reviewing later.

Output a JSON object with one key, "items", holding a list of learning objects (max 3). \
Each learning object must have:
  "title": short descriptive slug (3-6 words, lowercase, hyphens)
  "body": 2-3 sentences capturing the insight or lesson
  "tags": list of 1-3 topic tags

If there is nothing worth extracting (the record is purely factual/reference with no lessons), \
output: {"items": []}

Respond with only the JSON object. No prose."""

_EXTRACT_USER_TEMPLATE = """\
Record type: {rec_type}
File: {rel_path}
Frontmatter: {fm_summary}
Body:
{body}"""


class DistillerDaemon(BaseDaemon):
    name = "distiller"

    def __init__(self, cfg, state, events) -> None:
        super().__init__(cfg, state, events)
        self._last_run = 0.0  # epoch 0 ensures first run fires immediately
        # Failed topic-append count for the current/most-recent sweep. Reset at
        # the start of each _distill_sweep() call so it reflects just that tick.
        # Not persisted — surfaced via the distiller.sweep_done log line so a
        # human/monitor can notice non-zero failures (audit: dropped writes).
        self.failed_appends_this_tick = 0

    async def run(self) -> None:
        self.log.info("distiller.start", mode=getattr(self.cfg, "distiller_mode", "scheduled"))
        try:
            while not self._stop.is_set():
                # on_demand mode: never auto-run; wait for explicit trigger via trigger_sweep()
                if getattr(self.cfg, "distiller_mode", "scheduled") == "on_demand":
                    await asyncio.sleep(60.0)
                    continue
                now = time.time()
                if now - self._last_run > DISTILL_INTERVAL:
                    await self._distill_sweep()
                    self._last_run = now
                await asyncio.sleep(120.0)
        finally:
            await self.save_state()
            self.log.info("distiller.stopped")

    async def trigger_sweep(self) -> None:
        """Manually trigger a distill sweep regardless of mode."""
        await self._distill_sweep()
        self._last_run = time.time()

    async def tick(self) -> None:
        """One-shot distill sweep — called by APScheduler (daily at 2am when scheduled)."""
        try:
            await self._distill_sweep()
        except Exception as e:
            self.log.error("distiller.tick_error", error=str(e))

    async def _distill_sweep(self) -> None:
        vault_path = self.cfg.vault_path
        state = self.state.state
        now_iso = datetime.now(timezone.utc).isoformat()
        distilled_count = 0
        learn_count = 0
        deferred = False
        self.failed_appends_this_tick = 0

        stale = [
            (rel_path, fs)
            for rel_path, fs in list(state.files.items())
            # daemon output — never re-distill
            if not is_daemon_generated(rel_path, generated_by=fs.__dict__.get("generated_by"))
            and _is_stale(fs.last_distilled)
        ]
        # Never-distilled first, then the oldest stamp. With a cap, dict order
        # would take the same leading files again each time they went stale
        # and never reach the rest of the vault.
        stale.sort(key=lambda item: item[1].last_distilled or "")
        cap = max(1, int(getattr(self.cfg, "distiller_max_files_per_sweep", 200)))
        batch = stale[:cap]
        if len(stale) > cap:
            # Up to 3 topic appends per file, so an uncapped first sweep over
            # the main vault's ~15K unstamped files would be ~45K writes in
            # one night. The rest waits for the next sweeps, oldest first.
            self.log.info("distiller.sweep_capped", cap=cap, stale=len(stale))

        for rel_path, fs in batch:
            try:
                created = await self._distill_file(vault_path, rel_path)
                fs.last_distilled = now_iso
                learn_count += created
                distilled_count += 1
                if distilled_count % 50 == 0:
                    await self.save_state()
                    self.log.debug("distiller.incremental_save", files=distilled_count)
                await asyncio.sleep(1.5)
            except LocalLLMRequestTooLarge as e:
                # Retrying sends the same request, so stamp it like a file
                # with nothing to distill: it comes back when it goes stale.
                self.log.warning("distiller.request_too_large", path=rel_path, error=str(e))
                fs.last_distilled = now_iso
            except LocalLLMUnavailable as e:
                # Stop the sweep instead of retrying a dead backend once per
                # file. last_distilled is only stamped on success (line
                # above), so everything not yet reached stays stale and the
                # next sweep resumes from here.
                self.log.warning(
                    "distiller.backend_unavailable",
                    error=str(e),
                    deferred_from=rel_path,
                    distilled_before_stop=distilled_count,
                )
                deferred = True
                break
            except Exception as e:
                self.log.warning("distiller.file_error", path=rel_path, error=str(e))

        # Every finished sweep is a run, including one that found nothing
        # stale — runner.py's startup catch-up reads the last entry, and a
        # quiet vault must not look overdue. A sweep that hit a dead backend
        # before doing anything is not a run, so the catch-up retries it.
        stale_remaining = sum(1 for _, fs in stale if _is_stale(fs.last_distilled))
        if distilled_count or not deferred:
            state.distiller_runs.append({
                "timestamp": now_iso,
                "files_scanned": distilled_count,
                "learn_records_created": learn_count,
                "stale_remaining": stale_remaining,
            })
            if len(state.distiller_runs) > 30:
                state.distiller_runs = state.distiller_runs[-30:]
            self.log.info(
                "distiller.sweep_done",
                files=distilled_count,
                learned=learn_count,
                failed_appends=self.failed_appends_this_tick,
                stale_remaining=stale_remaining,
            )
            await self.save_state()

    async def _distill_file(self, vault_path: Path, rel_path: str) -> int:
        try:
            rec = vault_read(vault_path, rel_path)
        except Exception as e:
            self.log.warning("distiller.distill_read_failed", path=rel_path, error=str(e))
            return 0

        fm = rec["frontmatter"]
        body = rec["body"]
        rec_type = fm.get("type", "")

        if not body or len(body.strip()) < MIN_BODY_LEN:
            return 0
        if is_daemon_generated(record_type=rec_type, generated_by=fm.get("generated_by")):
            return 0  # daemon output — LLM-generated content must never feed back into distillation

        from datetime import date as _date, datetime as _datetime

        class _DateEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, (_date, _datetime)):
                    return o.isoformat()
                return super().default(o)

        fm_summary = json.dumps(
            {k: v for k, v in fm.items() if v},
            indent=2,
            cls=_DateEncoder,
        )
        user_text = _EXTRACT_USER_TEMPLATE.format(
            rec_type=rec_type or "unknown",
            rel_path=rel_path,
            fm_summary=fm_summary[:500],
            body=body[:2000],
        )

        def _call() -> str:
            return complete(
                _EXTRACT_SYSTEM,
                user_text,
                **self.cfg.llm,
                json_mode=True,
                max_tokens=512,
            )

        # LocalLLMUnavailable propagates to the sweep, which defers the rest of
        # the batch rather than recording every record as "nothing to distill".
        raw = (await asyncio.to_thread(_call)).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return 0
        learnings = _unwrap_items(parsed)
        if learnings is None:
            # json_mode constrains the reply to a JSON *object*, so the bare
            # array the old prompt asked for never came back and every file
            # scored learned=0 without a trace. Count any shape we still
            # can't read, so a regression shows up in error_counts.
            record_failure(
                "distiller.unexpected_json_shape",
                path=rel_path,
                shape=type(parsed).__name__,
            )
            return 0

        state = self.state.state
        created = 0
        for item in learnings[:3]:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "")
            body_text = item.get("body", "")
            tags = item.get("tags", [])
            if not title or not body_text:
                continue
            try:
                tag_list = tags if isinstance(tags, list) else []
                topic_slug = _tag_to_slug(tag_list[0] if tag_list else "misc")
                result = vault_append_to_topic(
                    vault_path,
                    topic_slug,
                    title,
                    body_text,
                    tags=tag_list,
                    source=rel_path,
                )
                if rel_path in state.files:
                    state.files[rel_path].learn_records_created.append(result["path"])
                created += 1
                self.log.debug("distiller.appended_topic", path=result["path"], title=title)
            except VaultError as e:
                # This runs after a successful, already-billed Anthropic API call —
                # losing the write here silently would burn spend for nothing and
                # leave zero trace. Log it and count it so it's observable instead.
                self.failed_appends_this_tick += 1
                self.log.warning(
                    "distiller.topic_append_failed",
                    path=rel_path,
                    title=title,
                    error=str(e),
                )

        return created


def _unwrap_items(parsed) -> list | None:
    """The learnings list from a {"items": [...]} reply, or a bare list.

    None means the reply had neither shape — not "nothing to learn", which is
    an empty list.
    """
    if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
        return parsed["items"]
    if isinstance(parsed, list):
        return parsed
    return None


def _tag_to_slug(tag: str) -> str:
    import re as _re
    slug = tag.lower().strip()
    slug = _re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-") or "misc"
    # Resolve variant slugs to their canonical counterpart so the distiller
    # doesn't proliferate near-duplicate topic files.
    return _TAG_CANONICAL.get(slug, slug)


def _is_stale(last_distilled: str) -> bool:
    if not last_distilled:
        return True
    try:
        last = datetime.fromisoformat(last_distilled)
        days = (datetime.now(timezone.utc) - last).total_seconds() / 86400
        return days >= STALE_DAYS
    except Exception as e:
        log.debug("distiller.stale_parse_failed", value=last_distilled, error=str(e))
        return True
