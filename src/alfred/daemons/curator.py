"""CuratorDaemon — inbox watch → LLM classification → vault record creation."""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import frontmatter
import structlog

from alfred.core.schema import KNOWN_TYPES, STATUS_BY_TYPE, TYPE_DIRECTORY, correct_status, correct_type
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

Routing rules:
- AI conversation sessions, Claude Code handoffs, chat logs → type: "session"
- Beliefs, limits, system constraints → type: "assumption" or "constraint" (stored in topic/)
- Contradictions, conflicts → type: "contradiction" (stored in topic/)
- Choices, trade-offs, architectural decisions → type: "decision" (stored in topic/)

Note content:
---
{content}
---

Respond with only a JSON object on a single line. No prose, no markdown fences."""


def _json_default(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class CuratorDaemon(BaseDaemon):
    name = "curator"
    _classify_paused_until: float = 0.0

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
                ingested = await self._ingest_file(md_file, processed_dir)
                if ingested:
                    state.curator_processed[rel_str] = datetime.now(timezone.utc).isoformat()
            except Exception as e:
                self.log.warning("curator.ingest_error", path=rel_str, error=str(e))

    async def _ingest_file(self, inbox_file: Path, processed_dir: Path) -> bool:
        try:
            post = frontmatter.load(str(inbox_file))
            fm = dict(post.metadata)
            body = post.content
        except Exception:
            fm = {}
            body = inbox_file.read_text(encoding="utf-8", errors="replace")

        content_preview = (body[:2000]).strip()
        if fm:
            fm_str = json.dumps({k: v for k, v in fm.items() if v}, indent=2, default=_json_default)
            content_preview = f"Frontmatter:\n{fm_str}\n\nBody:\n{content_preview}"

        # If the file already declares a known type, skip LLM classification entirely.
        # This is cheaper, faster, and prevents misclassification of structured drops
        # (e.g. session-end hook writes type: session explicitly).
        # Normalise legacy types: conversation → session, ai-dialogue → session.
        existing_type = fm.get("type", "")
        if existing_type in ("conversation", "ai-dialogue"):
            existing_type = "session"
            fm["type"] = "session"
        if existing_type and existing_type in KNOWN_TYPES:
            classification = {
                "type": existing_type,
                "name": fm.get("name") or fm.get("subject") or None,
                "status": fm.get("status") or None,
                "tags": fm.get("tags") or [],
            }
            self.log.debug("curator.type_from_frontmatter", type=existing_type, path=inbox_file.name)
        else:
            classification = await self._classify(content_preview)
            if not classification:
                self.log.info("curator.skip_no_classification", path=inbox_file.name)
                return False

        rec_type = classification.get("type", "")
        # Normalise legacy type aliases before validation
        if rec_type in ("conversation", "ai-dialogue"):
            rec_type = "session"
            classification["type"] = "session"
        # Try _TYPE_CORRECTIONS if not directly known
        if rec_type not in KNOWN_TYPES:
            corrected = correct_type(rec_type)
            if corrected:
                rec_type = corrected
                classification["type"] = corrected
        if rec_type not in KNOWN_TYPES:
            self.log.info("curator.skip_unknown_type", type=rec_type, path=inbox_file.name)
            return False

        # Prefer a real heading/title from the source over LLM-generated slug
        extracted_title = (
            fm.get("title") or fm.get("name") or fm.get("subject")
            or _extract_heading(body)
        )
        name = extracted_title or classification.get("name") or inbox_file.stem
        safe_name = _slugify(name)

        set_fields: dict = {}
        if raw_status := classification.get("status"):
            valid_status = correct_status(raw_status, rec_type) or raw_status
            if valid_status in STATUS_BY_TYPE.get(rec_type, {valid_status}):
                set_fields["status"] = valid_status
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
        return True

    async def _classify(self, content: str) -> dict | None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None

        if time.monotonic() < self._classify_paused_until:
            return None

        prompt = _CLASSIFY_PROMPT.format(
            types=", ".join(sorted(KNOWN_TYPES)),
            content=content[:3000],
        )

        try:
            import asyncio
            import anthropic
            client = anthropic.Anthropic()

            def _call():
                resp = client.messages.create(
                    model=self.cfg.anthropic_model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip()

            raw = await asyncio.to_thread(_call)
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(raw)
        except Exception as e:
            err_str = str(e)
            self.log.warning("curator.classify_error", error=err_str)
            if "credit balance is too low" in err_str or "balance is too low" in err_str:
                self._classify_paused_until = time.monotonic() + 3600
                self.log.warning("curator.classify_paused", reason="credit_exhaustion", resume_in_seconds=3600)
            return None


def _slugify(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:80] or "untitled"


def _extract_heading(body: str) -> str:
    """Return text of the first H1 heading in the body, or empty string."""
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()[:120]
    return ""
