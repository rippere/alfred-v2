"""Core vault operations — create, read, edit, search, move, delete.

Ported from personal-alfred vault/ops.py. Obsidian CLI integration removed
for simplicity — filesystem-only. Wikilink updates on move are not automatic.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path

import frontmatter
import yaml

from alfred.core.schema import (
    KNOWN_TYPES, LIST_FIELDS, NAME_FIELD_BY_TYPE,
    REQUIRED_FIELDS, STATUS_BY_TYPE, TYPE_DIRECTORY,
)


class VaultError(Exception):
    pass


def compute_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


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
    if fp.exists():
        raise VaultError(f"Already exists: {rel_path}")

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


def vault_delete(vault_path: Path, rel_path: str) -> dict:
    fp = _resolve(vault_path, rel_path)
    if not fp.exists():
        raise VaultError(f"File not found: {rel_path}")
    fp.unlink()
    return {"path": rel_path, "deleted": True}


def vault_move(vault_path: Path, from_path: str, to_path: str) -> dict:
    src = _resolve(vault_path, from_path)
    dst = _resolve(vault_path, to_path)
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
        except Exception:
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
        except Exception:
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
