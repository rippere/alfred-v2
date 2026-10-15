"""JanitorDaemon — structural scan + deterministic autofix + per-file LLM enrichment.

Architecture:
  Stage 1: Structural scan (pure Python, deterministic) — finds all issues
  Stage 2: Autofix (pure Python) — fixes FM001/FM002/FM003/FM004 without LLM
  Stage 3: LLM enrichment (one call per file, one reference template per call)
            — prompt capped at janitor_max_bytes_per_call
"""
from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import os
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import frontmatter
import structlog

from alfred.core.failures import record_failure
from alfred.core.local_llm import LocalLLMRequestTooLarge, LocalLLMUnavailable, complete
from alfred.core.schema import (
    DIRECTORY_TO_TYPE, KNOWN_TYPES, LIST_FIELDS, NAME_FIELD_BY_TYPE,
    REQUIRED_FIELDS, STATUS_BY_TYPE,
    correct_status, correct_type,
)
from alfred.core.vault import extract_wikilinks, is_sync_conflict
from alfred.core.vault_ops import VaultError, vault_edit, vault_read
from alfred.daemons.base import BaseDaemon

if TYPE_CHECKING:
    from alfred.store.lancedb_store import LanceDBStore

log = structlog.get_logger()

SWEEP_INTERVAL = 3600.0     # structural scan every hour
DEEP_INTERVAL = 86400.0     # LLM enrichment once per day
DEDUP_INTERVAL = 604800.0   # dedup sweep once per week (7 days)
ARCHIVE_INTERVAL = 86400.0  # session archival once per day
FORGET_INTERVAL = 86400.0   # Ebbinghaus retention sweep once per day
REAP_INTERVAL = 86400.0     # orphan-vector reap once per day


class IssueCode(str, Enum):
    MISSING_REQUIRED_FIELD = "FM001"
    INVALID_TYPE_VALUE = "FM002"
    INVALID_STATUS_VALUE = "FM003"
    INVALID_FIELD_TYPE = "FM004"
    WRONG_DIRECTORY = "DIR001"
    BROKEN_WIKILINK = "LINK001"
    STUB_RECORD = "STUB001"
    GARBAGE_CONTENT = "SEM001"


class JanitorDaemon(BaseDaemon):
    name = "janitor"

    def __init__(self, cfg, state, events, store: LanceDBStore) -> None:
        super().__init__(cfg, state, events)
        self.store = store
        self._last_sweep = float("-inf")
        self._last_deep = float("-inf")
        self._last_dedup = float("-inf")
        self._last_archive = float("-inf")
        self._last_forget = float("-inf")
        self._last_reap = float("-inf")
        self._stem_index: dict[str, set[str]] = {}

    async def run(self) -> None:
        self.log.info("janitor.start")
        sweep_interval = float(self.cfg.janitor_sweep_interval_s)
        deep_interval = float(self.cfg.janitor_deep_interval_h * 3600)
        try:
            while not self._stop.is_set():
                now = asyncio.get_event_loop().time()
                if now - self._last_sweep > sweep_interval:
                    await self._structural_sweep()
                    self._last_sweep = now
                if now - self._last_deep > deep_interval:
                    await self._deep_sweep()
                    self._last_deep = now
                dedup_enabled = getattr(self.cfg, "janitor_dedup_enabled", False)
                if dedup_enabled and now - self._last_dedup > DEDUP_INTERVAL:
                    await self._dedup_sweep()
                    self._last_dedup = now
                if now - self._last_archive > ARCHIVE_INTERVAL:
                    await self.session_archive_tick()
                    self._last_archive = now
                forget_enabled = getattr(self.cfg, "janitor_forget_enabled", False)
                if forget_enabled and now - self._last_forget > FORGET_INTERVAL:
                    await self.forget_tick()
                    self._last_forget = now
                reap_enabled = getattr(self.cfg, "janitor_reap_enabled", False)
                if reap_enabled and now - self._last_reap > REAP_INTERVAL:
                    await self.reap_tick()
                    self._last_reap = now
                await asyncio.sleep(60.0)
        finally:
            await self.save_state()
            self.log.info("janitor.stopped")

    async def structural_tick(self) -> None:
        """One-shot structural sweep — called by APScheduler on sweep_interval."""
        try:
            await self._structural_sweep()
        except Exception as e:
            self.log.error("janitor.structural_tick_error", error=str(e))

    async def deep_tick(self) -> None:
        """One-shot LLM enrichment sweep — called by APScheduler on deep_interval."""
        try:
            await self._deep_sweep()
        except Exception as e:
            self.log.error("janitor.deep_tick_error", error=str(e))

    async def dedup_tick(self) -> None:
        """One-shot dedup sweep — called by APScheduler weekly if enabled."""
        if not getattr(self.cfg, "janitor_dedup_enabled", False):
            return
        try:
            await self._dedup_sweep()
        except Exception as e:
            self.log.error("janitor.dedup_tick_error", error=str(e))

    async def session_archive_tick(self) -> None:
        """One-shot session archival — called by APScheduler daily.

        Moves absorbed/completed sessions older than 90 days to _archived/.
        Split out of the structural sweep so a slow archival pass can never
        delay the hourly structural lint (and vice versa).
        """
        try:
            archived = await self._archive_sessions(self.cfg.vault_path)
            if archived:
                self.log.info("janitor.sessions_archived", count=archived)
                await self.save_state()
        except Exception as e:
            self.log.error("janitor.session_archive_tick_error", error=str(e))

    # ── Stage 1: structural scan ───────────────────────────────────────────────

    async def _structural_sweep(self) -> None:
        vault_path = self.cfg.vault_path
        ignore = set(self.cfg.ignore_dirs)
        issues: dict[str, list[dict]] = {}   # rel_path -> [{code, message, fix}]

        self._stem_index = self._build_stem_index(vault_path, ignore)

        for md_file in vault_path.rglob("*.md"):
            rel = md_file.relative_to(vault_path)
            if any(part in ignore for part in rel.parts):
                continue
            if is_sync_conflict(md_file):
                continue
            rel_str = str(rel).replace("\\", "/")
            try:
                file_issues = self._check_file(vault_path, rel_str)
                if file_issues:
                    issues[rel_str] = [i.__dict__ for i in file_issues]
            except Exception as e:
                self.log.warning("janitor.scan_error", path=rel_str, error=str(e))

        # Update state open_issues — clear all tracked files first so resolved issues don't persist
        state = self.state.state
        now_iso = datetime.now(timezone.utc).isoformat()

        # Prune ghost state entries (files deleted from vault but still in state.files).
        # Conflict copies are excluded from live_paths deliberately: any that were
        # indexed before the filter existed now read as ghosts and get their state
        # entry and embeddings dropped here, which is exactly the cleanup wanted.
        # The files themselves are never touched — only the index.
        live_paths = {
            str(md_file.relative_to(vault_path)).replace("\\", "/")
            for md_file in vault_path.rglob("*.md")
            if not is_sync_conflict(md_file)
        }
        ghost_keys = [k for k in state.files if k not in live_paths]
        for k in ghost_keys:
            self._delete_embeddings(state, k)
        if ghost_keys:
            self.log.info("janitor.pruned_ghosts", count=len(ghost_keys))

        for fs in state.files.values():
            fs.open_issues = []
        for rel_path, file_issues in issues.items():
            if rel_path in state.files:
                state.files[rel_path].open_issues = [i["code"] for i in file_issues]
                state.files[rel_path].last_scanned = now_iso

        self.log.info("janitor.sweep_complete", files_with_issues=len(issues))

        # Stage 2: autofix deterministic issues
        fixed = await self._autofix(issues, vault_path)
        if fixed:
            self.log.info("janitor.autofixed", count=len(fixed))

        # Record sweep in state
        state.janitor_sweeps.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "files_with_issues": len(issues),
            "autofixed": len(fixed),
        })
        if len(state.janitor_sweeps) > 50:
            state.janitor_sweeps = state.janitor_sweeps[-50:]

        await self.save_state()

    def _build_stem_index(self, vault_path: Path, ignore: set[str]) -> dict[str, set[str]]:
        """Build a lookup index for wikilink resolution.

        Indexes all vault files (including ignored dirs like ai-dialogue) so that
        wikilinks pointing into those dirs don't appear as false-positive LINK001.
        The ignore list only controls which files are *scanned for issues*, not
        which files are valid link targets.
        """
        index: dict[str, set[str]] = {}
        for md_file in vault_path.rglob("*.md"):
            # A conflict copy is not a legitimate wikilink target — registering
            # one lets a broken link resolve to a stale duplicate and read as fixed.
            if is_sync_conflict(md_file):
                continue
            rel = md_file.relative_to(vault_path)
            rel_str = str(rel).replace("\\", "/")
            stem = md_file.stem
            # Register under stem, path-without-extension, and full path (with .md)
            index.setdefault(stem, set()).add(rel_str)
            rel_no_ext = rel_str.removesuffix(".md")
            index.setdefault(rel_no_ext, set()).add(rel_str)
            index.setdefault(rel_str, set()).add(rel_str)
        return index

    def _check_file(self, vault_path: Path, rel_path: str) -> list:
        fp = vault_path / rel_path
        try:
            post = frontmatter.load(str(fp))
            fm = dict(post.metadata)
            body = post.content
        except Exception:
            return []

        issues = []
        rec_type = fm.get("type", "")

        class _Issue:
            def __init__(self, code, message):
                self.code = code
                self.message = message

        for req in REQUIRED_FIELDS:
            if not fm.get(req):
                issues.append(_Issue(IssueCode.MISSING_REQUIRED_FIELD.value, f"Missing: {req}"))

        if rec_type and rec_type not in KNOWN_TYPES:
            issues.append(_Issue(IssueCode.INVALID_TYPE_VALUE.value, f"Unknown type: {rec_type!r}"))

        status = fm.get("status", "")
        if rec_type and status and rec_type in STATUS_BY_TYPE:
            valid = STATUS_BY_TYPE[rec_type]
            if valid and status not in valid:
                issues.append(_Issue(IssueCode.INVALID_STATUS_VALUE.value, f"Invalid status: {status!r}"))

        for field_name in LIST_FIELDS:
            val = fm.get(field_name)
            if val is not None and not isinstance(val, list):
                if field_name == "project" and isinstance(val, str):
                    continue
                issues.append(_Issue(IssueCode.INVALID_FIELD_TYPE.value, f"Field {field_name!r} must be a list"))

        for link in extract_wikilinks(fp.read_text(encoding="utf-8", errors="replace")):
            # Normalize multiline YAML string artifacts (e.g. "foo\n  bar" → "foo bar")
            normalized = " ".join(link.split())
            if not self._stem_index.get(normalized):
                issues.append(_Issue(IssueCode.BROKEN_WIKILINK.value, f"Broken: [[{normalized}]]"))
                break  # only flag first broken link per file to keep noise down

        # Stub detection
        body_len = len(body.strip())
        if rec_type and body_len < 50:
            issues.append(_Issue(IssueCode.STUB_RECORD.value, f"Stub body ({body_len} chars)"))

        return issues

    # ── Stage 2: deterministic autofix ───────────────────────────────────────

    async def _autofix(self, issues: dict[str, list[dict]], vault_path: Path) -> list[str]:
        fixed: list[str] = []
        for rel_path, file_issues in issues.items():
            codes = {i["code"] for i in file_issues}
            if not (codes & {
                IssueCode.MISSING_REQUIRED_FIELD.value,
                IssueCode.INVALID_TYPE_VALUE.value,
                IssueCode.INVALID_STATUS_VALUE.value,
                IssueCode.INVALID_FIELD_TYPE.value,
            }):
                continue
            try:
                rec = vault_read(vault_path, rel_path)
                fm = rec["frontmatter"]
                edits: dict = {}

                # FM001: fill missing required fields
                if "type" not in fm or not fm["type"]:
                    inferred = self._infer_type(rel_path)
                    if inferred:
                        edits["type"] = inferred
                if "created" not in fm or not fm["created"]:
                    fp = vault_path / rel_path
                    edits["created"] = date.fromtimestamp(fp.stat().st_mtime).isoformat()
                # Set name from filename if missing
                rec_type = edits.get("type", fm.get("type", ""))
                title_field = NAME_FIELD_BY_TYPE.get(rec_type, "name")
                if rec_type and not fm.get(title_field) and not fm.get("name"):
                    edits[title_field] = Path(rel_path).stem

                # FM002: correct type typos
                raw_type = fm.get("type", "")
                if raw_type and raw_type not in KNOWN_TYPES:
                    corrected = correct_type(raw_type)
                    if corrected:
                        edits["type"] = corrected

                # FM003: correct status typos
                raw_status = fm.get("status", "")
                effective_type = edits.get("type", fm.get("type", ""))
                if raw_status and effective_type:
                    valid = STATUS_BY_TYPE.get(effective_type, set())
                    if valid and raw_status not in valid:
                        corrected = correct_status(raw_status, effective_type)
                        if corrected:
                            edits["status"] = corrected

                # FM004: wrap non-list list fields
                for field_name in LIST_FIELDS:
                    val = fm.get(field_name)
                    if val is not None and not isinstance(val, list):
                        if field_name == "project" and isinstance(val, str):
                            continue
                        edits[field_name] = [val]

                if edits:
                    vault_edit(vault_path, rel_path, set_fields=edits)
                    fixed.append(rel_path)
                    self.log.debug("janitor.autofixed", path=rel_path, fields=list(edits.keys()))
            except Exception as e:
                self.log.warning("janitor.autofix_error", path=rel_path, error=str(e))

        return fixed

    def _infer_type(self, rel_path: str) -> str:
        """Infer a record type from the directory a file sits in.

        Walks the containing directories deepest-first so nested layouts
        resolve to the nearest meaningful directory: `_archived/session/x.md`
        is a session, and `session/2026/x.md` is still a session.

        Two lookups per directory part, in order:

        1. The directory name *is* a record type — this is the case for
           directories TYPE_DIRECTORY does not point at because they
           consolidate elsewhere on write (`decision/`, `assumption/`,
           `constraint/`, `contradiction/`, `input/` all sit on disk but
           TYPE_DIRECTORY routes them to topic/). Inverting TYPE_DIRECTORY
           inferred nothing at all for those.
        2. DIRECTORY_TO_TYPE, the collision-resolved inverse, for directories
           whose name differs from the type (`ideas/` -> idea, `drafts/` ->
           script, `hooks/` -> hook).

        Returns "" when nothing resolves, which leaves `type` unset rather
        than stamping a guess.
        """
        parts = rel_path.replace("\\", "/").split("/")
        if len(parts) < 2:
            return ""
        for part in reversed(parts[:-1]):     # directories only, nearest first
            if part in KNOWN_TYPES:
                return part
            resolved = DIRECTORY_TO_TYPE.get(part)
            if resolved:
                return resolved
        return ""

    # ── Stage 3: LLM enrichment (stub records, per-file, per-type prompt) ────

    async def _deep_sweep(self) -> None:
        """LLM enrichment for stub records. One call per file, one template per call."""
        vault_path = self.cfg.vault_path
        ignore = set(self.cfg.ignore_dirs)
        state = self.state.state
        enriched = 0

        for rel_path, fs in list(state.files.items()):
            if IssueCode.STUB_RECORD.value not in fs.open_issues:
                continue
            vault_file = vault_path / rel_path
            if not vault_file.exists():
                continue
            try:
                await self._enrich_file(vault_path, rel_path)
                # Clear stub issue after enrichment attempt
                fs.open_issues = [c for c in fs.open_issues if c != IssueCode.STUB_RECORD.value]
                enriched += 1
                await asyncio.sleep(1.0)   # gentle rate limiting
            except LocalLLMRequestTooLarge as e:
                # The same stub gets the same answer next sweep: drop the issue
                # instead of paying for it every four hours.
                self.log.warning("janitor.request_too_large", path=rel_path, error=str(e))
                fs.open_issues = [c for c in fs.open_issues if c != IssueCode.STUB_RECORD.value]
            except LocalLLMUnavailable as e:
                # Stop rather than retry a dead backend per stub. The STUB_RECORD
                # issue is only cleared on a successful enrichment, so everything
                # unreached stays queued for the next sweep.
                self.log.warning(
                    "janitor.backend_unavailable",
                    error=str(e),
                    deferred_from=rel_path,
                    enriched_before_stop=enriched,
                )
                break
            except Exception as e:
                self.log.warning("janitor.enrich_error", path=rel_path, error=str(e))

        if enriched:
            self.log.info("janitor.enriched", count=enriched)
            await self.save_state()

    async def _enrich_file(self, vault_path: Path, rel_path: str) -> None:
        """Ask LLM to fill in stub body for one file. Prompt capped at max_bytes."""
        rec = vault_read(vault_path, rel_path)
        fm = rec["frontmatter"]
        body = rec["body"]
        rec_type = fm.get("type", "unknown")

        # Build a compact prompt — well under the 8KB cap
        from datetime import date, datetime
        fm_summary = json.dumps(
            {k: v.isoformat() if isinstance(v, (date, datetime)) else v
             for k, v in fm.items() if v},
            indent=2,
        )
        prompt = (
            f"You are enriching a personal knowledge vault record.\n"
            f"Type: {rec_type}\n"
            f"File: {rel_path}\n"
            f"Current frontmatter:\n```json\n{fm_summary}\n```\n"
            f"Current body:\n```\n{body[:500]}\n```\n\n"
            f"Write a concise, factual body (2-4 sentences) for this {rec_type} record based on available context. "
            f"Return ONLY the body text, no headers, no JSON."
        )

        # Enforce byte cap
        prompt_bytes = prompt.encode("utf-8")
        if len(prompt_bytes) > self.cfg.janitor_max_bytes_per_call:
            prompt = prompt[:self.cfg.janitor_max_bytes_per_call].decode("utf-8", errors="replace")

        def _call() -> str:
            return complete(
                "You are a careful editor enriching a knowledge-vault record.",
                prompt,
                **self.cfg.llm,
                max_tokens=512,
            )

        new_body = (await asyncio.to_thread(_call)).strip()
        if new_body and len(new_body) > 20:
            vault_edit(vault_path, rel_path, body_replace=new_body)
            self.log.info("janitor.enriched_file", path=rel_path, chars=len(new_body))

    def _delete_embeddings(self, state, rel_path: str) -> None:
        """Delete a file's vectors from the vector store and prune its state entry.

        Mirrors ``SurveyorDaemon._process_diff``'s deletion path: look up the
        known chunk_ids (if any) before popping state, so ``store.delete_file``
        can use the fast explicit-id delete instead of falling back to a
        prefix scan. Used by ``_archive_sessions`` and ``_dedup_sweep`` so
        moved/removed files never leave orphaned LanceDB embeddings behind.
        """
        fs = state.files.get(rel_path)
        chunk_ids = fs.chunk_ids if fs else None
        try:
            self.store.delete_file(rel_path, chunk_ids)
        except Exception as e:
            self.log.warning("janitor.vector_delete_failed", path=rel_path, error=str(e))
        state.files.pop(rel_path, None)

    # ── Forget sweep: Ebbinghaus retention ────────────────────────────────────

    def forget_candidates(self, now: datetime | None = None) -> list[dict]:
        """Files whose memory has decayed past the retention threshold.

        Pure and side-effect free so `alfred forget --dry-run` and the sweep
        itself can never disagree about what would be evicted — the dry run
        calls exactly the function the sweep does.

        A file qualifies only if ALL of:
          * it currently has vectors (chunk_ids non-empty) and isn't already
            forgotten — otherwise there is nothing to reclaim;
          * it was embedded at least forget_min_age_days ago. This is the
            cold-start guard: with no query history every file looks
            unaccessed, and without an age floor the first sweep would evict
            the entire store;
          * its Ebbinghaus retrievability is below the threshold. Never-
            accessed files score 0.0 and so pass on age alone — which is the
            intent, since "embedded 6 months ago and never once retrieved"
            is the exact profile of dead weight.

        Returned newest-decay-last (weakest memory first) so the per-sweep cap
        evicts the coldest files rather than an arbitrary slice.
        """
        now = now or datetime.now(timezone.utc)
        min_age_days = float(getattr(self.cfg, "janitor_forget_min_age_days", 180))
        threshold = float(getattr(self.cfg, "janitor_forget_retrievability", 0.02))
        state = self.state.state

        candidates: list[dict] = []
        for rel_path, fs in state.files.items():
            if fs.forgotten or not fs.chunk_ids:
                continue
            if not fs.last_embedded:
                continue
            try:
                embedded_at = datetime.fromisoformat(fs.last_embedded)
            except ValueError as e:
                record_failure(
                    "janitor.forget_timestamp_unparsable", error=e, path=rel_path
                )
                continue
            if embedded_at.tzinfo is None:
                embedded_at = embedded_at.replace(tzinfo=timezone.utc)
            age_days = (now - embedded_at).total_seconds() / 86400
            if age_days < min_age_days:
                continue

            strength = state.memory.get(rel_path)
            r = strength.retrievability(now) if strength else 0.0
            if r >= threshold:
                continue
            candidates.append({
                "rel_path": rel_path,
                "retrievability": r,
                "age_days": round(age_days, 1),
                "chunks": len(fs.chunk_ids),
                "access_count": strength.access_count if strength else 0,
            })

        candidates.sort(key=lambda c: (c["retrievability"], -c["age_days"]))
        return candidates

    def _forget_file(self, rel_path: str, now_iso: str) -> bool:
        """Evict one file's vectors, keeping its FileState entry.

        Deliberately NOT _delete_embeddings(), which pops the state entry.
        Popping is right for a file that is genuinely gone; here the file
        still exists in the vault, and an entry popped from state comes back
        as "new" on the surveyor's next diff and is immediately re-embedded.
        That round trip is precisely how the vector store regrew. Keeping the
        entry (md5 intact, chunk_ids cleared, forgotten stamped) makes the
        eviction stick until the file is actually edited.
        """
        state = self.state.state
        fs = state.files.get(rel_path)
        if fs is None:
            return False
        try:
            self.store.delete_file(rel_path, fs.chunk_ids)
        except Exception as e:
            record_failure("janitor.forget_vector_delete_failed", error=e, path=rel_path)
            return False
        fs.chunk_ids = []
        fs.last_embedded = ""
        fs.forgotten = now_iso
        return True

    async def _forget_sweep(self) -> None:
        if not getattr(self.cfg, "janitor_forget_enabled", False):
            return
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        cap = int(getattr(self.cfg, "janitor_forget_max_per_sweep", 500))

        candidates = self.forget_candidates(now)
        if not candidates:
            self.log.info("janitor.forget_sweep_done", evicted=0, candidates=0)
            return

        selected = candidates[:cap]
        evicted = 0
        chunks_freed = 0
        for i, c in enumerate(selected):
            if i % 20 == 0:
                await asyncio.sleep(0)  # yield; this can be a long list
            if self._forget_file(c["rel_path"], now_iso):
                evicted += 1
                chunks_freed += c["chunks"]

        # A cap that silently truncates reads as "that's all there was", so
        # say plainly how many were left for the next sweep.
        self.log.info(
            "janitor.forget_sweep_done",
            evicted=evicted,
            chunks_freed=chunks_freed,
            candidates=len(candidates),
            deferred=max(0, len(candidates) - len(selected)),
        )
        await self.save_state()

    async def forget_tick(self) -> None:
        """One-shot retention sweep — called by APScheduler daily if enabled."""
        if not getattr(self.cfg, "janitor_forget_enabled", False):
            return
        try:
            await self._forget_sweep()
        except Exception as e:
            self.log.error("janitor.forget_tick_error", error=str(e))

    # ── Reap sweep: orphaned vector rows ──────────────────────────────────────

    def reap_plan(self, max_rows: int | None = None):
        """What the reap sweep would delete.  Side-effect free.

        Same contract as ``forget_candidates``: `alfred reap` and the sweep
        call exactly this, so a dry run and the real thing can never disagree
        about what would go.
        """
        from alfred.store.reaper import build_plan

        cap = int(
            max_rows
            if max_rows is not None
            else getattr(self.cfg, "janitor_reap_max_rows_per_sweep", 5000)
        )
        return build_plan(
            self.store,
            self.state.state,
            self.cfg.vault_path,
            max_rows=cap,
            batch_size=int(getattr(self.cfg, "janitor_reap_scan_batch_size", 4096)),
        )

    async def _reap_sweep(self) -> None:
        if not getattr(self.cfg, "janitor_reap_enabled", False):
            return
        from alfred.store.reaper import execute_plan

        plan = await asyncio.to_thread(self.reap_plan)
        if plan.aborted:
            # Not an error the sweep can fix — an unmounted vault or a state
            # file that disagrees with the disk.  Loud, and nothing deleted.
            self.log.warning("janitor.reap_sweep_aborted", reason=plan.aborted)
            return
        if not plan.orphans:
            self.log.info(
                "janitor.reap_sweep_done",
                deleted=0,
                store_rows=plan.store_rows,
                untracked_but_present=plan.untracked_but_present,
            )
            return

        deleted = await asyncio.to_thread(
            execute_plan,
            self.store,
            self.state.state,
            self.cfg.vault_path,
            plan,
            int(getattr(self.cfg, "janitor_reap_scan_batch_size", 4096)),
            int(getattr(self.cfg, "janitor_reap_delete_batch", 500)),
        )
        # untracked_but_present is reported every sweep on purpose: it is the
        # count of rows a naive "not in state.json" reaper would have deleted
        # (2,892 files / 45,352 rows on the live store as of 2026-08-07), and
        # keeping it in the log is how a regression in the vault check shows
        # up as a number instead of as missing data.
        self.log.info(
            "janitor.reap_sweep_done",
            deleted=deleted,
            paths=len(plan.orphans),
            deferred=len(plan.deferred),
            malformed=plan.malformed_count,
            store_rows=plan.store_rows,
            untracked_but_present=plan.untracked_but_present,
        )
        # Flush record_failure() counters into state.error_counts — they only
        # arrive there via StateStore.save() -> drain_failures(). Relying on
        # another daemon to save first works in a full run and silently loses
        # every reap failure when the janitor runs alone (`alfred up --only
        # janitor`).
        await self.save_state()

    async def reap_tick(self) -> None:
        """One-shot orphan reap — called by APScheduler daily if enabled."""
        if not getattr(self.cfg, "janitor_reap_enabled", False):
            return
        try:
            await self._reap_sweep()
        except Exception as e:
            self.log.error("janitor.reap_tick_error", error=str(e))
            record_failure("janitor.reap_tick_error", error=e)

    # ── Dedup sweep: weekly similarity-based deduplication ────────────────────

    async def _dedup_sweep(self) -> None:
        """Compare all vault .md files within each directory and merge near-duplicates.

        Uses difflib.SequenceMatcher on file bodies: if ratio > 0.85 and both files
        are in the same directory, the shorter file's unique content is appended to
        the longer file, then the shorter is deleted.

        Runs at most once per week. Writes a summary to inbox/dedup-report-{date}.md.
        """
        vault_path = self.cfg.vault_path
        ignore = set(self.cfg.ignore_dirs) | {"inbox", "_archived", "_templates", "_bases"}
        state = self.state.state

        # Guard: skip if already ran this week (using state timestamp)
        last_run_iso = getattr(state, "last_dedup", None)
        if last_run_iso:
            try:
                last_run_dt = datetime.fromisoformat(last_run_iso)
                elapsed = (datetime.now(timezone.utc) - last_run_dt).total_seconds()
                if elapsed < DEDUP_INTERVAL:
                    self.log.debug("janitor.dedup_skip", reason="ran recently", elapsed_h=round(elapsed/3600, 1))
                    return
            except Exception as e:
                # A malformed last_dedup makes the interval guard fall through,
                # so dedup runs every tick instead of daily — expensive, and
                # invisible without a count.
                record_failure("janitor.dedup_timestamp_unparsable", error=e, value=last_run_iso)

        self.log.info("janitor.dedup_sweep.start")
        merged_count = 0
        merge_log: list[str] = []

        # Group all markdown files by their parent directory.
        #
        # Conflict copies are excluded, and this exclusion is load-bearing, not
        # tidiness. A conflict lives in the same directory as its original and
        # is near-identical to it: measured against the live vault, all 129
        # conflict/original pairs scored above this sweep's 0.85 threshold, with
        # a median similarity of 1.000. Keeper selection below is "longer body
        # wins", which has no notion of which file is the live one — in 9 of
        # those pairs the conflict was longer, so enabling this sweep would have
        # deleted the real note and promoted a June-22 copy in its place.
        # Conflict resolution belongs to scripts/reconcile_conflicts.py, which
        # is dry-run by default and shows its work.
        dir_files: dict[str, list[Path]] = {}
        for md_file in vault_path.rglob("*.md"):
            rel = md_file.relative_to(vault_path)
            if any(part in ignore for part in rel.parts):
                continue
            if is_sync_conflict(md_file):
                continue
            dir_key = str(rel.parent)
            dir_files.setdefault(dir_key, []).append(md_file)

        # Compare pairs within each directory
        for dir_key, files in dir_files.items():
            await asyncio.sleep(0)  # yield to event loop between directories
            if len(files) < 2:
                continue
            # Sort for determinism; limit to 200 files per dir to avoid O(n²) blowup
            files_sorted = sorted(files)[:200]
            checked: set[str] = set()

            for i, fa in enumerate(files_sorted):
                if i % 20 == 0:
                    await asyncio.sleep(0)  # yield every 20 files
                if str(fa) in checked:
                    continue
                try:
                    post_a = frontmatter.load(str(fa))
                    body_a = post_a.content.strip()
                except Exception as e:
                    # Unparsable file is silently exempt from dedup forever.
                    record_failure("janitor.frontmatter_parse_failed", error=e, path=str(fa))
                    continue
                if len(body_a) < 30:
                    continue

                for fb in files_sorted[i + 1:]:
                    if str(fb) in checked:
                        continue
                    try:
                        post_b = frontmatter.load(str(fb))
                        body_b = post_b.content.strip()
                    except Exception as e:
                        record_failure("janitor.frontmatter_parse_failed", error=e, path=str(fb))
                        continue
                    if len(body_b) < 30:
                        continue

                    ratio = difflib.SequenceMatcher(None, body_a, body_b, autojunk=False).ratio()
                    if ratio < 0.85:
                        continue

                    # Decide keeper (longer body wins)
                    if len(body_a) >= len(body_b):
                        keeper, dupe = fa, fb
                        keeper_body, dupe_body = body_a, body_b
                        keeper_post = post_a
                    else:
                        keeper, dupe = fb, fa
                        keeper_body, dupe_body = body_b, body_a
                        keeper_post = post_b

                    # Extract lines from dupe that are absent from keeper
                    keeper_lines = set(keeper_body.splitlines())
                    unique_lines = [
                        ln for ln in dupe_body.splitlines()
                        if ln.strip() and ln not in keeper_lines
                    ]
                    unique_content = "\n".join(unique_lines).strip()

                    # Append unique content to keeper if meaningful
                    if unique_content and len(unique_content) > 20:
                        try:
                            vault_edit(
                                vault_path,
                                str(keeper.relative_to(vault_path)).replace("\\", "/"),
                                body_append=f"<!-- merged from {dupe.name} -->\n{unique_content}",
                            )
                        except Exception as e:
                            self.log.warning("janitor.dedup_merge_error", keeper=keeper.name, error=str(e))
                            continue

                    # Delete duplicate
                    try:
                        dupe_rel = str(dupe.relative_to(vault_path)).replace("\\", "/")
                        dupe.unlink()
                        # Prune from state and vector store — mirrors surveyor's
                        # deletion path so a deduped file doesn't leave orphaned
                        # LanceDB embeddings behind.
                        self._delete_embeddings(state, dupe_rel)
                        checked.add(str(dupe))
                        merged_count += 1
                        msg = f"merged {dupe.name} → {keeper.name} (ratio={ratio:.2f})"
                        merge_log.append(msg)
                        self.log.info("janitor.dedup_merged", **dict(zip(
                            ["dupe", "keeper", "ratio"],
                            [dupe.name, keeper.name, round(ratio, 3)]
                        )))
                    except Exception as e:
                        self.log.warning("janitor.dedup_delete_error", path=dupe.name, error=str(e))

        # Update last_dedup timestamp on state (field now declared in PipelineState)
        state.last_dedup = datetime.now(timezone.utc).isoformat()

        # Write dedup report to inbox
        report_path = vault_path / "inbox" / f"dedup-report-{date.today().isoformat()}.md"
        report_lines = [
            "---",
            "type: note",
            f"created: '{date.today().isoformat()}'",
            "tags: [janitor, dedup]",
            f"name: dedup-report-{date.today().isoformat()}",
            "---",
            "",
            f"# Dedup Report — {date.today().isoformat()}",
            "",
            f"**Files merged:** {merged_count}",
            "",
        ]
        if merge_log:
            report_lines.append("## Merge Log\n")
            report_lines.extend(f"- {entry}" for entry in merge_log)
        else:
            report_lines.append("No duplicates found.")
        report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

        self.log.info("janitor.dedup_sweep.done", merged=merged_count)
        await self.save_state()

    # ── Session archive: move old absorbed/completed sessions out of active vault ─

    async def _archive_sessions(self, vault_path: Path) -> int:
        """Move session files with status absorbed or completed that are older than
        90 days into _archived/session/ to reduce active vault clutter and free
        Milvus slots on the next rebuild.

        Returns the count of files moved.
        """
        SESSION_ARCHIVE_DAYS = 90
        session_dir = vault_path / "session"
        if not session_dir.exists():
            return 0

        archive_dir = vault_path / "_archived" / "session"
        archive_dir.mkdir(parents=True, exist_ok=True)

        now_ts = datetime.now(timezone.utc).timestamp()
        archived = 0
        state = self.state.state

        for md_file in list(session_dir.glob("*.md")):
            try:
                post = frontmatter.load(str(md_file))
                status = str(post.metadata.get("status", "")).lower()
            except Exception as e:
                # Session file never gets archived — it accumulates forever
                # with nothing reporting that it was skipped.
                record_failure("janitor.session_parse_failed", error=e, path=str(md_file))
                continue

            if status not in {"absorbed", "completed"}:
                continue

            try:
                age_days = (now_ts - md_file.stat().st_mtime) / 86400
            except OSError as e:
                record_failure("janitor.session_stat_failed", error=e, path=str(md_file))
                continue

            if age_days < SESSION_ARCHIVE_DAYS:
                continue

            dest = archive_dir / md_file.name
            rel_path = f"session/{md_file.name}"
            # If destination already exists, skip (idempotent)
            if dest.exists():
                md_file.unlink(missing_ok=True)
                self._delete_embeddings(state, rel_path)
                archived += 1
                continue

            try:
                md_file.rename(dest)
                self._delete_embeddings(state, rel_path)
                archived += 1
                self.log.debug(
                    "janitor.session_archived",
                    file=md_file.name,
                    age_days=round(age_days, 1),
                )
            except Exception as e:
                self.log.warning("janitor.session_archive_error", file=md_file.name, error=str(e))

        return archived
