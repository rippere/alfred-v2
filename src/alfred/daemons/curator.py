"""CuratorDaemon — inbox watch → LLM classification → vault record creation."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path

import frontmatter
import structlog

from alfred.core.local_llm import (
    LocalLLMBadRequest,
    LocalLLMOutputTruncated,
    LocalLLMRequestTooLarge,
    LocalLLMUnavailable,
    complete_json,
)
from alfred.core.schema import KNOWN_TYPES, STATUS_BY_TYPE, TYPE_DIRECTORY, correct_status, correct_type
from alfred.core.vault_ops import VaultError, vault_create, vault_move
from alfred.daemons.base import BaseDaemon

log = structlog.get_logger()

WATCH_INTERVAL = 10.0   # poll inbox every 10 seconds

# A classification cut off at max_tokens is retried, never marked done: a
# sampled answer that ran long once need not again. Back off per file so a
# persistent one is not re-sent every 10 s: 1 min, doubling, capped at 1 h.
# In memory only, so a restart retries at once.
TRUNCATED_RETRY_BASE_S = 60.0
TRUNCATED_RETRY_MAX_S = 3600.0

# Output budget for one classification. Ollama keeps the 256 it always had,
# so flag-off requests are byte-for-byte unchanged. The OpenAI path gets
# headroom: its schema caps every string (below), so a whole reply is a few
# hundred tokens at most and finish_reason=length should be rare.
CLASSIFY_MAX_TOKENS_OLLAMA = 256
CLASSIFY_MAX_TOKENS_OPENAI = 1024


_CLASSIFY_SYSTEM = """\
You are classifying personal knowledge vault inbox notes.
Given a note, respond with a JSON object (and nothing else).

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

Respond with only a JSON object on a single line. No prose, no markdown fences."""

_CLASSIFY_USER_TEMPLATE = """\
Note content:
---
{content}
---"""

# The reply shape _CLASSIFY_SYSTEM asks for, enforced by the server on the
# OpenAI path (contract §4): `type` can only come back as a known note type,
# and every free-text field is length-capped, so the longest reply the schema
# allows (~1.9K chars, ~600 tokens) fits CLASSIFY_MAX_TOKENS_OPENAI.
#
# The caps are generous on purpose, well past "1-2 sentences". Measured on the
# Spark's vLLM 0.11 (2026-09-24, synthetic prompt): a string that reaches its
# maxLength is cut, but the grammar then allows any whitespace before the
# next token, and a model that wanted to keep writing pads tabs and newlines
# until max_tokens (finish_reason=length). Per-request whitespace controls
# are ignored there. A tight cap would turn a verbose answer into a truncated
# one; these only stop a runaway string, and the curator retries that later.
_CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": sorted(KNOWN_TYPES)},
        "name": {"type": "string", "maxLength": 200},
        "status": {"type": "string", "maxLength": 60},
        "tags": {
            "type": "array",
            "items": {"type": "string", "maxLength": 60},
            "maxItems": 10,
        },
        "summary": {"type": "string", "maxLength": 1000},
    },
    "required": ["type", "name"],
}


def _json_default(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _now() -> float:
    return time.monotonic()


class CuratorDaemon(BaseDaemon):
    name = "curator"

    def __init__(self, cfg, state, events) -> None:
        super().__init__(cfg, state, events)
        # process_key -> (truncated attempts so far, _now() to retry after)
        self._truncated_retry: dict[str, tuple[int, float]] = {}

    async def run(self) -> None:
        self.log.info("curator.start")
        try:
            while not self._stop.is_set():
                await self._process_inbox()
                await asyncio.sleep(WATCH_INTERVAL)
        finally:
            await self.save_state()
            self.log.info("curator.stopped")

    async def tick(self) -> None:
        """One-shot inbox poll — called by APScheduler every WATCH_INTERVAL seconds."""
        try:
            await self._process_inbox()
        except Exception as e:
            self.log.error("curator.tick_error", error=str(e))

    async def _process_inbox(self) -> None:
        inbox_path = self.cfg.vault_path / "inbox"
        if not inbox_path.exists():
            return

        processed_dir = inbox_path / "processed"
        processed_dir.mkdir(exist_ok=True)

        state = self.state.state

        for md_file in sorted(inbox_path.glob("*.md")):
            rel_str = str(md_file.relative_to(self.cfg.vault_path)).replace("\\", "/")
            try:
                content_hash = hashlib.sha256(md_file.read_bytes()).hexdigest()[:16]
            except OSError as e:
                self.log.warning("curator.read_error", path=rel_str, error=str(e))
                continue
            # Keyed on path+content, not path alone: a path can be reused (a new
            # note dropped under a filename an old, already-archived note used).
            # A path-only key marks that path "done" forever and silently
            # swallows every future drop at it — that's how curator lost ~10
            # re-dropped notes over 3 weeks before anyone noticed.
            process_key = f"{rel_str}#{content_hash}"
            if process_key in state.curator_processed:
                continue
            retry = self._truncated_retry.get(process_key)
            if retry is not None and _now() < retry[1]:
                continue  # its classification ran long; waiting out the backoff
            if rel_str in state.curator_processed:
                self.log.warning("curator.redrop_detected", path=rel_str)
            try:
                ingested = await self._ingest_file(md_file, processed_dir, content_hash)
                self._truncated_retry.pop(process_key, None)
                if ingested:
                    state.curator_processed[process_key] = datetime.now(timezone.utc).isoformat()
            except LocalLLMOutputTruncated as e:
                # The answer hit max_tokens. Unlike an oversized prompt, the
                # next sample may fit, so it is NOT marked processed: it stays
                # in inbox/ and is retried after a backoff. The rest of the
                # inbox goes on.
                attempts = (retry[0] if retry else 0) + 1
                delay = min(TRUNCATED_RETRY_MAX_S, TRUNCATED_RETRY_BASE_S * 2 ** (attempts - 1))
                self._truncated_retry[process_key] = (attempts, _now() + delay)
                self.log.warning(
                    "curator.classify_truncated",
                    path=rel_str,
                    attempts=attempts,
                    retry_in_s=delay,
                    error=str(e),
                )
            except LocalLLMRequestTooLarge as e:
                # The same file gets the same answer every 10 s. Mark it done so
                # it is retried only when its content changes; it stays in
                # inbox/, unclassified, for a person to look at.
                self.log.warning("curator.request_too_large", path=rel_str, error=str(e))
                state.curator_processed[process_key] = datetime.now(timezone.utc).isoformat()
            except LocalLLMUnavailable as e:
                # Abandon the whole tick, not just this file. The backend is down
                # for everyone, so continuing would retry it once per inbox entry
                # and bury the one fact that matters under N identical errors.
                # Nothing is marked processed, so the next tick picks up where
                # this one stopped. A LocalLLMBadRequest (a 400 that is not
                # about size: a rejected schema, say) is the setup's fault and
                # would fail every file alike, so it lands here too, at error.
                report = self.log.error if isinstance(e, LocalLLMBadRequest) else self.log.warning
                report(
                    "curator.backend_unavailable",
                    error=str(e),
                    deferred_from=rel_str,
                )
                return
            except Exception as e:
                self.log.warning("curator.ingest_error", path=rel_str, error=str(e))

    async def _ingest_file(self, inbox_file: Path, processed_dir: Path, content_hash: str) -> bool:
        try:
            post = frontmatter.load(str(inbox_file))
            fm = dict(post.metadata)
            body = post.content
        except Exception as e:
            # Malformed frontmatter — ingest the raw text rather than drop the file,
            # but surface it so a bad source isn't silently stripped of metadata.
            self.log.warning("curator.frontmatter_parse_failed", path=str(inbox_file), error=str(e))
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
            # LocalLLMUnavailable propagates deliberately. A backend that is down
            # (ollama-game-guard stops Ollama while a game runs) must leave the
            # file in inbox/ for the next tick, NOT be recorded as "classified as
            # nothing" — that is how the inbox silently stalled behind an expired
            # Anthropic credit balance while every request still returned 200.
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
        tags = classification.get("tags") or []
        if not isinstance(tags, list):
            tags = [tags]
        if _has_content_signal(body):
            if "content" not in tags:
                tags.append("content")
            set_fields["content_signal"] = True
            self.log.debug("curator.content_signal", path=inbox_file.name)
        if tags:
            set_fields["tags"] = tags

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
                # Retry with session_id suffix to avoid same-day slug collisions
                session_id = fm.get("session_id", "")
                sid_suffix = session_id[:8] if session_id else inbox_file.stem[-8:]
                fallback_name = f"{safe_name}-{sid_suffix}"
                try:
                    result = vault_create(
                        self.cfg.vault_path,
                        rec_type,
                        fallback_name,
                        set_fields=set_fields,
                        body=record_body or None,
                    )
                    self.log.info("curator.created_with_suffix", path=result["path"], source=inbox_file.name)
                except VaultError as e2:
                    if "Already exists" in str(e2):
                        self.log.debug("curator.duplicate_skip", path=inbox_file.name)
                        result = {}
                    else:
                        raise
            else:
                raise

        # Auto-link session to its project hub (if one exists)
        created_path = result.get("path", "") if isinstance(result, dict) else ""
        if rec_type in ("session", "conversation", "ai-dialogue") and created_path:
            try:
                _link_session_to_project(self.cfg.vault_path, created_path, body)
            except Exception as e:
                self.log.debug("curator.project_link_failed", path=created_path, error=str(e))

        # Move to processed/. If this filename was already archived (a reused
        # inbox path — see the process_key comment in _process_inbox),
        # disambiguate with the content hash instead of unlinking: the source
        # has already been ingested at this point, but deleting it here would
        # destroy the only raw copy of the newly-recovered content with no
        # archived fallback.
        dest = processed_dir / inbox_file.name
        if dest.exists():
            dest = processed_dir / f"{inbox_file.stem}.{content_hash}{inbox_file.suffix}"
        inbox_file.rename(dest)

        self.emit("curator_ingested", source=inbox_file.name, type=rec_type)
        return True

    async def _classify(self, content: str) -> dict | None:
        """Classify one inbox document. Returns None only when the model had
        nothing useful to say.

        Raises LocalLLMUnavailable when the backend is unreachable. That is not
        a classification outcome and must not be flattened into None: the caller
        leaves the file in inbox/ and retries on the next tick (so does
        LocalLLMBadRequest, its subclass). Raises LocalLLMOutputTruncated when
        the answer hit max_tokens; the caller retries that file after a
        backoff. Raises LocalLLMRequestTooLarge when the prompt itself is over
        the window and retrying cannot help; the caller skips it.
        """
        system_text = _CLASSIFY_SYSTEM.format(types=", ".join(sorted(KNOWN_TYPES)))
        user_text = _CLASSIFY_USER_TEMPLATE.format(content=content[:3000])

        llm = self.cfg.llm
        openai = llm["api"] == "openai"

        def _call() -> dict:
            return complete_json(
                system_text,
                user_text,
                **llm,
                # Ollama keeps plain format=json and its 256, so flag-off
                # requests are unchanged.
                schema=_CLASSIFY_SCHEMA if openai else None,
                max_tokens=CLASSIFY_MAX_TOKENS_OPENAI if openai else CLASSIFY_MAX_TOKENS_OLLAMA,
            )

        # LocalLLMUnavailable intentionally not caught — see docstring.
        parsed = await asyncio.to_thread(_call)
        if not parsed:
            self.log.debug("curator.classify_empty")
            return None
        return parsed


_CONTENT_SIGNAL_TERMS = {
    # Platforms
    "reels", "carousel", "tiktok", "youtube", "newsletter", "substack",
    # Content craft
    "hook", "cta", "script", "angle", "viral", "engagement", "saves", "shares",
    "content idea", "content strategy", "post idea", "going viral",
    # Project-specific
    "tribe-social", "tribe_social", "tribev2", "tribe v2",
    # Creation intent
    "content creation", "create content", "content calendar", "content brief",
}


def _build_project_index(vault_path) -> dict[str, str]:
    """Return {lowercase_name_variant: rel_path} for all project hub files."""
    from pathlib import Path
    import frontmatter as _fm
    index: dict[str, str] = {}
    proj_dir = Path(vault_path) / "project"
    if not proj_dir.exists():
        return index
    for md in proj_dir.glob("*.md"):
        try:
            post = _fm.load(str(md))
            if post.metadata.get("type") != "project":
                continue
            rel = f"project/{md.stem}"
            # Index by file stem, name field, and common abbreviations
            for variant in [md.stem, post.metadata.get("name", "")]:
                if variant and len(variant) > 3:
                    index[variant.lower().replace("-", " ")] = rel
                    index[variant.lower()] = rel
        except Exception as e:
            log.debug("curator.project_index_skip", path=str(md), error=str(e))
            continue
    return index


def _link_session_to_project(vault_path, session_rel: str, body: str) -> None:
    """Add a wikilink to the matching project hub when a session mentions a project."""
    import re as _re
    from alfred.core.vault_ops import vault_edit, vault_read
    proj_index = _build_project_index(vault_path)
    if not proj_index:
        return
    body_lower = body.lower()
    matched_proj = None
    matched_len = 0
    for name_variant, rel in proj_index.items():
        if len(name_variant) > matched_len and name_variant in body_lower:
            matched_proj = rel
            matched_len = len(name_variant)
    if not matched_proj:
        return
    # Add wikilink to session's related field
    vault_edit(vault_path, session_rel, append_fields={"related": f"[[{matched_proj}]]"})
    # Add wikilink back from project hub to this session
    vault_edit(vault_path, matched_proj + ".md", append_fields={"related": f"[[{session_rel.removesuffix('.md')}]]"})


def _has_content_signal(text: str) -> bool:
    """Return True if the text contains strong content-creation signals."""
    lower = text.lower()
    return sum(1 for term in _CONTENT_SIGNAL_TERMS if term in lower) >= 2


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
