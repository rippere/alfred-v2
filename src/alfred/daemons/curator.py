"""CuratorDaemon — inbox watch → LLM classification → vault record creation."""
from __future__ import annotations

import asyncio
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import frontmatter
import structlog

from alfred.core.schema import KNOWN_TYPES, TYPE_DIRECTORY
from alfred.core.vault_ops import VaultError, vault_create, vault_move
from alfred.daemons.base import BaseDaemon

log = structlog.get_logger()

WATCH_INTERVAL = 10.0   # poll inbox every 10 seconds


_CLASSIFY_PROMPT = """\
You are classifying a personal knowledge vault inbox note.
Given the following note content, respond with a JSON object (and nothing else).

Fields to extract:
- "type": one of {types} (required)
- "name": a short slug-friendly title (required, use the note title or summarize in a few words)
- "status": appropriate initial status for the type (optional)
- "tags": list of relevant topic tags (optional, max 5)
- "summary": 1-2 sentence description (optional, used as body if the note is short)

Note content:
---
{content}
---

Respond with only a JSON object on a single line. No prose, no markdown fences."""


class CuratorDaemon(BaseDaemon):
    name = "curator"

    async def run(self) -> None:
        self.log.info("curator.start")
        try:
            while not self._stop.is_set():
                await self._process_inbox()
                await asyncio.sleep(WATCH_INTERVAL)
        finally:
            await self.save_state()
            self.log.info("curator.stopped")

    async def _process_inbox(self) -> None:
        inbox_path = self.cfg.vault_path / "inbox"
        if not inbox_path.exists():
            return

        processed_dir = inbox_path / "processed"
        processed_dir.mkdir(exist_ok=True)

        state = self.state.state

        for md_file in sorted(inbox_path.glob("*.md")):
            rel_str = str(md_file.relative_to(self.cfg.vault_path)).replace("\\", "/")
            if rel_str in state.curator_processed:
                continue
            try:
                await self._ingest_file(md_file, processed_dir)
                state.curator_processed[rel_str] = datetime.now(timezone.utc).isoformat()
            except Exception as e:
                self.log.warning("curator.ingest_error", path=rel_str, error=str(e))

    async def _ingest_file(self, inbox_file: Path, processed_dir: Path) -> None:
        try:
            post = frontmatter.load(str(inbox_file))
            fm = dict(post.metadata)
            body = post.content
        except Exception:
            fm = {}
            body = inbox_file.read_text(encoding="utf-8", errors="replace")

        content_preview = (body[:2000]).strip()
        if fm:
            fm_str = json.dumps({k: v for k, v in fm.items() if v}, indent=2)
            content_preview = f"Frontmatter:\n{fm_str}\n\nBody:\n{content_preview}"

        classification = await self._classify(content_preview)
        if not classification:
            self.log.info("curator.skip_no_classification", path=inbox_file.name)
            return

        rec_type = classification.get("type", "")
        if rec_type not in KNOWN_TYPES:
            self.log.info("curator.skip_unknown_type", type=rec_type, path=inbox_file.name)
            return

        name = classification.get("name") or inbox_file.stem
        # sanitize name for filesystem
        safe_name = _slugify(name)

        set_fields: dict = {}
        if status := classification.get("status"):
            set_fields["status"] = status
        if tags := classification.get("tags"):
            set_fields["tags"] = tags if isinstance(tags, list) else [tags]

        record_body = body.strip() or classification.get("summary", "")

        try:
            result = vault_create(
                self.cfg.vault_path,
                rec_type,
                safe_name,
                set_fields=set_fields,
                body=record_body or None,
            )
            self.log.info("curator.created", path=result["path"], source=inbox_file.name)
        except VaultError as e:
            if "Already exists" in str(e):
                # Deduplicate: skip silently
                self.log.debug("curator.duplicate_skip", path=inbox_file.name)
            else:
                raise

        # Move to processed/
        dest = processed_dir / inbox_file.name
        if not dest.exists():
            inbox_file.rename(dest)
        else:
            inbox_file.unlink()

        self.emit("curator_ingested", source=inbox_file.name, type=rec_type)

    async def _classify(self, content: str) -> dict | None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None

        prompt = _CLASSIFY_PROMPT.format(
            types=", ".join(sorted(KNOWN_TYPES)),
            content=content[:3000],
        )

        try:
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=self.cfg.anthropic_model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(raw)
        except Exception as e:
            self.log.warning("curator.classify_error", error=str(e))
            return None


def _slugify(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:80] or "untitled"
