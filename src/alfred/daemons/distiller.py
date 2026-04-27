"""DistillerDaemon — scan vault records → extract learnings → create learn/ records."""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import frontmatter
import structlog

from alfred.core.vault_ops import VaultError, vault_create, vault_read
from alfred.daemons.base import BaseDaemon

log = structlog.get_logger()

DISTILL_INTERVAL = 86400.0   # run once per day
STALE_DAYS = 7               # re-distill file if not distilled in 7 days
MIN_BODY_LEN = 200           # skip files with trivial bodies


_EXTRACT_PROMPT = """\
You are a personal knowledge distiller. Read the following vault record and extract the most \
reusable, transferable insights or lessons from it — things worth remembering and reviewing later.

Record type: {rec_type}
File: {rel_path}
Frontmatter: {fm_summary}
Body:
{body}

Output a JSON array of learning objects (max 3). Each object must have:
  "title": short descriptive slug (3-6 words, lowercase, hyphens)
  "body": 2-3 sentences capturing the insight or lesson
  "tags": list of 1-3 topic tags

If there is nothing worth extracting (the record is purely factual/reference with no lessons), \
output an empty JSON array: []

Respond with only the JSON array. No prose."""


class DistillerDaemon(BaseDaemon):
    name = "distiller"

    def __init__(self, cfg, state, events) -> None:
        super().__init__(cfg, state, events)
        self._last_run = 0.0

    async def run(self) -> None:
        self.log.info("distiller.start")
        try:
            while not self._stop.is_set():
                now = asyncio.get_event_loop().time()
                if now - self._last_run > DISTILL_INTERVAL:
                    await self._distill_sweep()
                    self._last_run = now
                await asyncio.sleep(120.0)
        finally:
            await self.save_state()
            self.log.info("distiller.stopped")

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
            if _is_stale(fs.last_distilled):
                try:
                    created = await self._distill_file(vault_path, rel_path)
                    fs.last_distilled = now_iso
                    learn_count += created
                    distilled_count += 1
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
        if rec_type in ("learn",):
            return 0  # don't distill learn records from other learn records

        fm_summary = json.dumps({k: v for k, v in fm.items() if v}, indent=2)
        prompt = _EXTRACT_PROMPT.format(
            rec_type=rec_type or "unknown",
            rel_path=rel_path,
            fm_summary=fm_summary[:500],
            body=body[:2000],
        )

        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=self.cfg.anthropic_model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
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
                result = vault_create(
                    vault_path,
                    "learn",
                    title,
                    set_fields={
                        "tags": tags if isinstance(tags, list) else [],
                        "source": rel_path,
                    },
                    body=body_text,
                )
                # Track learn record in source file's state
                if rel_path in state.files:
                    state.files[rel_path].learn_records_created.append(result["path"])
                created += 1
                self.log.debug("distiller.created_learn", path=result["path"])
            except VaultError:
                pass

        return created


def _is_stale(last_distilled: str) -> bool:
    if not last_distilled:
        return True
    try:
        last = datetime.fromisoformat(last_distilled)
        days = (datetime.now(timezone.utc) - last).total_seconds() / 86400
        return days >= STALE_DAYS
    except Exception:
        return True
