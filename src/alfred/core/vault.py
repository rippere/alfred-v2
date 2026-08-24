"""Vault file parsing, wikilink extraction, and chunk building.

Migrated from personal-alfred surveyor/parser.py — logic preserved exactly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]+)?\]\]")

MAX_EMBEDDING_CHARS = 6_000
CHUNK_SIZE = 3_000
CHUNK_OVERLAP = 200

EMBEDDING_FM_KEYS = ["type", "status", "name", "description", "intent", "source", "channel"]
EXCLUDE_FM_KEYS = [
    "tags", "alfred_tags", "relationships", "created", "updated", "date",
    "aliases", "cssclass", "cssclasses",
]


@dataclass
class VaultRecord:
    rel_path: str
    frontmatter: dict
    body: str
    record_type: str
    wikilinks: list[str] = field(default_factory=list)


# Syncthing names a conflicted copy `<stem>.sync-conflict-<date>-<time>-<ID><ext>`,
# so a conflicted note ends in `.md` and matches every `rglob("*.md")` in this
# codebase. cfg.ignore_dirs cannot express it — that filter is directory-based.
#
# The result was 162 conflicted copies indexed alongside their live originals,
# 129 of them with the original present too. A live `vault_search "decision"`
# returned 11 conflicts in 40 results: a quarter of the retrieval budget spent
# on June-22 duplicates, 100 of which were byte-identical to the file they
# shadowed. They are Syncthing's bookkeeping, not vault content.
_SYNC_CONFLICT_RE = re.compile(r"\.sync-conflict-\d{8}-\d{6}-[A-Z0-9]+")


def is_sync_conflict(path: Path | str) -> bool:
    """True if *path* is a Syncthing conflict copy.

    Matches on the filename only, so it is safe to call with either an absolute
    Path or a vault-relative string.
    """
    name = path.name if isinstance(path, Path) else str(path).rsplit("/", 1)[-1]
    return bool(_SYNC_CONFLICT_RE.search(name))


def extract_wikilinks(text: str) -> list[str]:
    # Strip Obsidian Bases plugin virtual tables (*.base) — not real files
    return [l for l in WIKILINK_RE.findall(text) if not l.endswith(".base")]


def parse_file(vault_path: Path, rel_path: str) -> VaultRecord:
    full_path = vault_path / rel_path
    raw_text = full_path.read_text(encoding="utf-8")
    post = frontmatter.loads(raw_text)
    fm = dict(post.metadata)
    return VaultRecord(
        rel_path=rel_path,
        frontmatter=fm,
        body=post.content,
        record_type=fm.get("type", "unknown"),
        wikilinks=extract_wikilinks(raw_text),
    )


def build_embedding_text(record: VaultRecord) -> str:
    parts: list[str] = []
    for key in EMBEDDING_FM_KEYS:
        val = record.frontmatter.get(key)
        if val and isinstance(val, str):
            parts.append(f"{key}: {val}")
    fm_text = "\n".join(parts)
    fm_chars = len(fm_text) + 1
    if record.body:
        body = record.body.strip()
        body_budget = MAX_EMBEDDING_CHARS - fm_chars
        if body_budget > 0:
            parts.append(body[:body_budget])
    return "\n".join(parts)


def _safe_chunk_id(rel_path: str, idx: int) -> str:
    """Build a Milvus-safe chunk_id. Strips characters that break Milvus's internal SQL parser."""
    safe = rel_path.replace("'", "")
    return f"{safe}::chunk_{idx:02d}"


def _split_by_headers(body: str, max_size: int) -> list[str]:
    """Split a Markdown body on header boundaries, then sub-split oversized sections."""
    header_re = re.compile(r"^#{1,3} ", re.MULTILINE)
    # Find all header positions
    positions = [m.start() for m in header_re.finditer(body)]
    if not positions:
        positions = []

    # Build sections: text before first header + each header+body block
    sections: list[str] = []
    boundaries = positions + [len(body)]
    prev = 0
    for pos in boundaries[:-1] if positions else []:
        if pos > prev:
            sections.append(body[prev:pos].strip())
        prev = pos
    sections.append(body[prev:].strip())
    sections = [s for s in sections if s]

    # Sub-split any section that exceeds max_size
    result: list[str] = []
    for section in sections:
        if len(section) <= max_size:
            result.append(section)
        else:
            step = max(max_size - CHUNK_OVERLAP, 1)
            start = 0
            while start < len(section):
                result.append(section[start:start + max_size])
                start += step
    return result or [body[:max_size]]


def chunk_record(record: VaultRecord) -> list[tuple[str, str]]:
    """Return (chunk_id, text) pairs using header-aware Markdown chunking."""
    fm_parts: list[str] = []
    for key in EMBEDDING_FM_KEYS:
        val = record.frontmatter.get(key)
        if val and isinstance(val, str):
            fm_parts.append(f"{key}: {val}")
    fm_prefix = "\n".join(fm_parts)
    fm_len = len(fm_prefix) + (1 if fm_prefix else 0)

    body = record.body.strip() if record.body else ""
    body_budget = CHUNK_SIZE - fm_len

    if body_budget <= 0:
        return [(_safe_chunk_id(record.rel_path, 0), fm_prefix)]

    if len(body) <= body_budget:
        text = (fm_prefix + "\n" + body).strip()
        return [(_safe_chunk_id(record.rel_path, 0), text)]

    sections = _split_by_headers(body, body_budget)
    chunks: list[tuple[str, str]] = []
    for idx, section in enumerate(sections):
        text = (fm_prefix + "\n" + section).strip()
        chunks.append((_safe_chunk_id(record.rel_path, idx), text))
    return chunks
