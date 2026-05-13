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

from alfred.core.anthropic_client import get_client
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

Output a JSON array of learning objects (max 3). Each object must have:
  "title": short descriptive slug (3-6 words, lowercase, hyphens)
  "body": 2-3 sentences capturing the insight or lesson
  "tags": list of 1-3 topic tags

If there is nothing worth extracting (the record is purely factual/reference with no lessons), \
output an empty JSON array: []

Respond with only the JSON array. No prose."""

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
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            self.log.info("distiller.skip", reason="no ANTHROPIC_API_KEY")
            return

        vault_path = self.cfg.vault_path
        state = self.state.state
        now_iso = datetime.now(timezone.utc).isoformat()
        distilled_count = 0
        learn_count = 0

        for rel_path, fs in list(state.files.items()):
            if rel_path.startswith(("learn/", "topic/", "synthesis/")):
                continue  # daemon output — never re-distill
            if fs.__dict__.get("generated_by") == "llm":
                continue  # skip any file explicitly marked as LLM-generated
            if _is_stale(fs.last_distilled):
                try:
                    created = await self._distill_file(vault_path, rel_path)
                    fs.last_distilled = now_iso
                    learn_count += created
                    distilled_count += 1
                    if distilled_count % 50 == 0:
                        await self.save_state()
                        self.log.debug("distiller.incremental_save", files=distilled_count)
                    await asyncio.sleep(1.5)
                except Exception as e:
                    self.log.warning("distiller.file_error", path=rel_path, error=str(e))

        if distilled_count:
            state.distiller_runs.append({
                "timestamp": now_iso,
                "files_scanned": distilled_count,
                "learn_records_created": learn_count,
            })
            if len(state.distiller_runs) > 30:
                state.distiller_runs = state.distiller_runs[-30:]
            self.log.info("distiller.sweep_done", files=distilled_count, learned=learn_count)
            await self.save_state()

    async def _distill_file(self, vault_path: Path, rel_path: str) -> int:
        try:
            rec = vault_read(vault_path, rel_path)
        except Exception:
            return 0

        fm = rec["frontmatter"]
        body = rec["body"]
        rec_type = fm.get("type", "")

        if not body or len(body.strip()) < MIN_BODY_LEN:
            return 0
        if rec_type in ("learn", "topic", "synthesis"):
            return 0  # daemon output — never re-distill
        if fm.get("generated_by") == "llm":
            return 0  # LLM-generated content must not feed back into distillation

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

        client = get_client()

        def _call():
            resp = client.messages.create(
                model=self.cfg.anthropic_model,
                max_tokens=512,
                system=[{
                    "type": "text",
                    "text": _EXTRACT_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_text}],
            )
            return resp

        resp = await asyncio.to_thread(_call)
        usage = resp.usage
        self.state.record_api_call(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=getattr(usage, "cache_read_input_tokens", 0),
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            learnings = json.loads(raw)
        except json.JSONDecodeError:
            return 0
        if not isinstance(learnings, list):
            return 0

        state = self.state.state
        created = 0
        for item in learnings[:3]:
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
            except VaultError:
                pass

        return created


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
    except Exception:
        return True
