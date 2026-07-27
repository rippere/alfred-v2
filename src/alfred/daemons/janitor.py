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

from alfred.core.local_llm import LocalLLMUnavailable, complete
from alfred.core.schema import (
    KNOWN_TYPES, LIST_FIELDS, NAME_FIELD_BY_TYPE,
    REQUIRED_FIELDS, STATUS_BY_TYPE, TYPE_DIRECTORY,
    correct_status, correct_type,
)
from alfred.core.vault import extract_wikilinks
from alfred.core.vault_ops import VaultError, vault_edit, vault_read
from alfred.daemons.base import BaseDaemon

if TYPE_CHECKING:
    from alfred.store.lancedb_store import LanceDBStore

log = structlog.get_logger()

SWEEP_INTERVAL = 3600.0     # structural scan every hour
DEEP_INTERVAL = 86400.0     # LLM enrichment once per day
DEDUP_INTERVAL = 604800.0   # dedup sweep once per week (7 days)
ARCHIVE_INTERVAL = 86400.0  # session archival once per day


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

        # Prune ghost state entries (files deleted from vault but still in state.files)
        live_paths = {
            str(md_file.relative_to(vault_path)).replace("\\", "/")
            for md_file in vault_path.rglob("*.md")
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
        parts = rel_path.replace("\\", "/").split("/")
        if len(parts) < 2:
            return ""
        dir_to_type = {v: k for k, v in TYPE_DIRECTORY.items()}
        return dir_to_type.get(parts[0], "")

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
                base_url=self.cfg.ollama_base_url,
                model=self.cfg.ollama_llm_model,
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
            except Exception:
                pass

        self.log.info("janitor.dedup_sweep.start")
        merged_count = 0
        merge_log: list[str] = []

        # Group all markdown files by their parent directory
        dir_files: dict[str, list[Path]] = {}
        for md_file in vault_path.rglob("*.md"):
            rel = md_file.relative_to(vault_path)
            if any(part in ignore for part in rel.parts):
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
                except Exception:
                    continue
                if len(body_a) < 30:
                    continue

                for fb in files_sorted[i + 1:]:
                    if str(fb) in checked:
                        continue
                    try:
                        post_b = frontmatter.load(str(fb))
                        body_b = post_b.content.strip()
                    except Exception:
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
            except Exception:
                continue

            if status not in {"absorbed", "completed"}:
                continue

            try:
                age_days = (now_ts - md_file.stat().st_mtime) / 86400
            except OSError:
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
