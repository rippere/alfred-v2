"""Vault file parsing, wikilink extraction, and chunk building.

Migrated from personal-alfred surveyor/parser.py — logic preserved exactly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter

WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

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


def extract_wikilinks(text: str) -> list[str]:
    return WIKILINK_RE.findall(text)


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


def chunk_record(record: VaultRecord) -> list[tuple[str, str]]:
    """Return (chunk_id, text) pairs. chunk_id format: rel_path::chunk_NN."""
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
        return [(f"{record.rel_path}::chunk_00", fm_prefix)]

    if len(body) <= body_budget:
        text = (fm_prefix + "\n" + body).strip()
        return [(f"{record.rel_path}::chunk_00", text)]

    chunks: list[tuple[str, str]] = []
    step = max(body_budget - CHUNK_OVERLAP, 1)
    start, idx = 0, 0
    while start < len(body):
        text = (fm_prefix + "\n" + body[start:start + body_budget]).strip()
        chunks.append((f"{record.rel_path}::chunk_{idx:02d}", text))
        start += step
        idx += 1
    return chunks
