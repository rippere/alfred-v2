"""Core vault operations — create, read, edit, search, move, delete.

Ported from personal-alfred vault/ops.py. Obsidian CLI integration removed
for simplicity — filesystem-only. Wikilink updates on move are not automatic.
"""
from __future__ import annotations

import re
import threading
from datetime import date
from pathlib import Path

import frontmatter
import structlog
import yaml

from alfred.core.schema import (
    KNOWN_TYPES, LIST_FIELDS, NAME_FIELD_BY_TYPE,
    REQUIRED_FIELDS, STATUS_BY_TYPE, TYPE_DIRECTORY,
)

log = structlog.get_logger()


class VaultError(Exception):
    pass


# Serializes vault file mutations (create/edit/move) within this process.
# Daemon file I/O already hops threads via asyncio.to_thread (surveyor's
# embedding/HDBSCAN work), so two jobs can otherwise interleave inside a
# check-then-write or read-modify-write window and corrupt/clobber a record.
# threading.RLock rather than asyncio.Lock because these functions are
# synchronous and must stay so — call sites across 5 daemons depend on it.
_write_lock = threading.RLock()


def _resolve(vault_path: Path, rel_path: str) -> Path:
    full = (vault_path / rel_path).resolve()
    if not str(full).startswith(str(vault_path.resolve())):
        raise VaultError(f"Path traversal denied: {rel_path}")
    return full


def _parse(file_path: Path) -> tuple[dict, str]:
    try:
        post = frontmatter.load(str(file_path))
    except yaml.YAMLError as exc:
        raise VaultError(f"Malformed YAML frontmatter in {file_path.name}: {exc}") from exc
    return dict(post.metadata), post.content


def _serialize(fm: dict, body: str) -> str:
    post = frontmatter.Post(body, **fm)
    return frontmatter.dumps(post) + "\n"


def vault_read(vault_path: Path, rel_path: str) -> dict:
    fp = _resolve(vault_path, rel_path)
    if not fp.exists():
        raise VaultError(f"File not found: {rel_path}")
    fm, body = _parse(fp)
    return {"path": rel_path, "frontmatter": fm, "body": body}


def vault_create(
    vault_path: Path,
    record_type: str,
    name: str,
    *,
    set_fields: dict | None = None,
    body: str | None = None,
) -> dict:
    if record_type not in KNOWN_TYPES:
        raise VaultError(f"Unknown type: {record_type!r}")
    set_fields = set_fields or {}

    directory = TYPE_DIRECTORY.get(record_type, record_type)
    rel_path = f"{directory}/{name}.md"
    fp = _resolve(vault_path, rel_path)

    fm: dict = {"type": record_type}
    title_field = NAME_FIELD_BY_TYPE.get(record_type, "name")
    fm[title_field] = name
    fm["created"] = date.today().isoformat()
    fm.update(set_fields)

    status = fm.get("status", "")
    if status:
        valid = STATUS_BY_TYPE.get(record_type, set())
        if valid and status not in valid:
            raise VaultError(f"Invalid status {status!r} for {record_type}")

    final_body = body if body is not None else f"# {name}\n"
    with _write_lock:  # exists-check + write must be one atomic window
        if fp.exists():
            raise VaultError(f"Already exists: {rel_path}")
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(_serialize(fm, final_body), encoding="utf-8")
    return {"path": rel_path}


def vault_edit(
    vault_path: Path,
    rel_path: str,
    *,
    set_fields: dict | None = None,
    append_fields: dict | None = None,
    body_replace: str | None = None,
    body_append: str | None = None,
) -> dict:
    fp = _resolve(vault_path, rel_path)
    with _write_lock:  # hold across the full read-modify-write window
        if not fp.exists():
            raise VaultError(f"File not found: {rel_path}")
        fm, body = _parse(fp)
        changed: list[str] = []

        if set_fields:
            for k, v in set_fields.items():
                fm[k] = v
                changed.append(k)
        if append_fields:
            for k, v in append_fields.items():
                existing = fm.get(k)
                if existing is None:
                    fm[k] = [v] if k in LIST_FIELDS else v
                elif isinstance(existing, list):
                    if v not in existing:
                        existing.append(v)
                else:
                    fm[k] = [existing, v]
                changed.append(k)
        if body_replace is not None:
            body = body_replace
            changed.append("body")
        if body_append:
            body = body.rstrip() + "\n\n" + body_append + "\n"
            changed.append("body_append")

        fp.write_text(_serialize(fm, body), encoding="utf-8")
    return {"path": rel_path, "fields_changed": changed}


def _find_richest_topic_by_tag(vault_path: Path, tags: list[str]) -> str | None:
    """Scan existing topic files for one that already covers any of the given tags.

    Returns the rel_path of the most content-rich (highest line count) matching file,
    or None if no match is found.  This prevents the distiller from spawning a new topic
    file when the concept is already covered under a different slug.
    """
    topic_dir = vault_path / "topic"
    if not topic_dir.exists():
        return None

    tag_set = {t.lower().strip() for t in tags if t}
    if not tag_set:
        return None

    best_path: str | None = None
    best_lines: int = -1

    for md_file in topic_dir.glob("*.md"):
        try:
            post = frontmatter.load(str(md_file))
        except Exception as e:
            log.debug("vault_ops.frontmatter_skip", path=str(md_file), where="find_richest_topic", error=str(e))
            continue
        file_tags_raw = post.metadata.get("tags", [])
        if not isinstance(file_tags_raw, list):
            file_tags_raw = [file_tags_raw] if file_tags_raw else []
        file_tags = {str(t).lower().strip() for t in file_tags_raw}
        if not (tag_set & file_tags):
            continue
        try:
            line_count = md_file.read_text(encoding="utf-8").count("\n")
        except OSError:
            line_count = 0
        if line_count > best_lines:
            best_lines = line_count
            best_path = f"topic/{md_file.name}"

    return best_path


def vault_append_to_topic(
    vault_path: Path,
    topic_slug: str,
    insight_title: str,
    insight_body: str,
    tags: list[str] | None = None,
    source: str | None = None,
) -> dict:
    """Append an insight section to topic/{topic_slug}.md, creating the file if needed.

    Before creating a new topic file, scans existing topic files for one that already
    covers any of the incoming tags.  If found, appends to the richest matching file
    instead of spawning a duplicate.
    """
    tags = tags or []
    rel_path = f"topic/{topic_slug}.md"
    fp = _resolve(vault_path, rel_path)

    # If the canonical slug file does not exist, look for an existing topic that
    # already covers one of these tags — prefer appending there over creating new.
    if not fp.exists():
        existing_rel = _find_richest_topic_by_tag(vault_path, tags + [topic_slug])
        if existing_rel and existing_rel != rel_path:
            rel_path = existing_rel
            fp = _resolve(vault_path, rel_path)

    source_link = (source[:-3] if source and source.endswith(".md") else source) or ""
    section = f"## {insight_title}\n\n{insight_body.strip()}"
    if source_link:
        section += f"\n\nSource: [[{source_link}]]"

    if fp.exists():
        fm, body = _parse(fp)
        existing_tags: list = fm.get("tags", [])
        if not isinstance(existing_tags, list):
            existing_tags = [existing_tags] if existing_tags else []
        for t in tags:
            if t not in existing_tags:
                existing_tags.append(t)
        fm["tags"] = existing_tags
        existing_sources: list = fm.get("sources", [])
        if not isinstance(existing_sources, list):
            existing_sources = [existing_sources] if existing_sources else []
        if source and source not in existing_sources:
            existing_sources.append(source)
        fm["sources"] = existing_sources
        fm["generated_by"] = "llm"
        body = body.rstrip() + "\n\n---\n\n" + section + "\n"
        fp.write_text(_serialize(fm, body), encoding="utf-8")
    else:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fm = {
            "type": "topic",
            "name": topic_slug,
            "tags": tags,
            "sources": [source] if source else [],
            "created": date.today().isoformat(),
            "status": "active",
            "generated_by": "llm",
        }
        body = f"# {topic_slug}\n\n{section}\n"
        fp.write_text(_serialize(fm, body), encoding="utf-8")

    return {"path": rel_path}


def vault_move(vault_path: Path, from_path: str, to_path: str) -> dict:
    src = _resolve(vault_path, from_path)
    dst = _resolve(vault_path, to_path)
    with _write_lock:  # exists-checks + rename must be one atomic window
        if not src.exists():
            raise VaultError(f"Source not found: {from_path}")
        if dst.exists():
            raise VaultError(f"Destination exists: {to_path}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
    return {"from": from_path, "to": to_path}


def vault_search(
    vault_path: Path,
    *,
    glob_pattern: str | None = None,
    grep_pattern: str | None = None,
    ignore_dirs: list[str] | None = None,
) -> list[dict]:
    ignore = set(ignore_dirs or [])
    results: list[dict] = []
    matches = list(vault_path.glob(glob_pattern)) if glob_pattern else list(vault_path.rglob("*.md"))
    for md_file in sorted(matches):
        rel = md_file.relative_to(vault_path)
        if any(part in ignore for part in rel.parts):
            continue
        if grep_pattern:
            try:
                if not re.search(re.escape(grep_pattern), md_file.read_text(encoding="utf-8"), re.IGNORECASE):
                    continue
            except (OSError, UnicodeDecodeError):
                continue
        try:
            post = frontmatter.load(str(md_file))
            fm = post.metadata
        except Exception as e:
            log.debug("vault_ops.frontmatter_skip", path=str(md_file), where="vault_search", error=str(e))
            fm = {}
        rel_str = str(rel).replace("\\", "/")
        results.append({
            "path": rel_str,
            "name": fm.get("name") or fm.get("subject") or md_file.stem,
            "type": fm.get("type", ""),
            "status": fm.get("status", ""),
        })
    return results


def vault_context(vault_path: Path, ignore_dirs: list[str] | None = None) -> dict:
    """Return compact vault summary grouped by type for LLM context."""
    ignore = set(ignore_dirs or []) | {".obsidian", "inbox"}
    by_type: dict[str, list[dict]] = {}
    for md_file in vault_path.rglob("*.md"):
        rel = md_file.relative_to(vault_path)
        if any(p in ignore for p in rel.parts):
            continue
        try:
            post = frontmatter.load(str(md_file))
        except Exception as e:
            log.debug("vault_ops.frontmatter_skip", path=str(md_file), where="vault_context", error=str(e))
            continue
        rec_type = post.metadata.get("type", "")
        if not rec_type:
            continue
        rel_str = str(rel).replace("\\", "/").removesuffix(".md")
        by_type.setdefault(rec_type, []).append({
            "path": rel_str,
            "name": md_file.stem,
            "status": str(post.metadata.get("status", "")),
        })
    return {"records_by_type": by_type, "total": sum(len(v) for v in by_type.values())}
