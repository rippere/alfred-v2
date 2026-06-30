#!/usr/bin/env python3
"""
Repair broken [[learn/...]] wikilinks in synthesis files.

Builds a lookup map from topic file section headers (## slug → topic-file-stem),
then rewrites every [[learn/slug]] reference in synthesis/*.md to [[topic/slug]].

Run from alfred-v2/ root:
    python scripts/repair_synthesis_links.py [--dry-run]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

VAULT_PATH = Path("/mnt/external/obsidian-vault")
SYNTHESIS_DIR = VAULT_PATH / "synthesis"
TOPIC_DIR = VAULT_PATH / "topic"
DRY_RUN = "--dry-run" in sys.argv

LEARN_LINK_RE = re.compile(r"\[\[learn/([^\]]+)\]\]")


def build_learn_to_topic_map() -> dict[str, str]:
    """Scan topic/*.md files and map each ## section heading → topic file stem."""
    mapping: dict[str, str] = {}
    for fp in TOPIC_DIR.glob("*.md"):
        topic_slug = fp.stem
        text = fp.read_text(encoding="utf-8")
        for heading in re.findall(r"^## (.+)$", text, re.MULTILINE):
            learn_slug = heading.strip()
            if learn_slug not in mapping:
                mapping[learn_slug] = topic_slug
    return mapping


def repair_file(fp: Path, mapping: dict[str, str]) -> tuple[int, int]:
    """Return (replacements_made, unresolved_count)."""
    original = fp.read_text(encoding="utf-8")
    replaced = 0
    unresolved = 0
    result = original

    for match in LEARN_LINK_RE.finditer(original):
        learn_slug = match.group(1)
        if learn_slug in mapping:
            old = f"[[learn/{learn_slug}]]"
            new = f"[[topic/{mapping[learn_slug]}]]"
            result = result.replace(old, new)
            replaced += 1
        else:
            unresolved += 1

    if replaced and not DRY_RUN:
        fp.write_text(result, encoding="utf-8")

    return replaced, unresolved


def main() -> None:
    if not TOPIC_DIR.exists():
        print("[ERROR] topic/ dir not found — run migrate_learn_to_topic.py first")
        sys.exit(1)

    print("Building learn → topic map from section headings...")
    mapping = build_learn_to_topic_map()
    print(f"  Mapped {len(mapping)} learn slugs across {len(list(TOPIC_DIR.glob('*.md')))} topic files")

    synthesis_files = sorted(SYNTHESIS_DIR.glob("*.md"))
    print(f"\nScanning {len(synthesis_files)} synthesis files...")

    total_replaced = 0
    total_unresolved = 0
    files_touched = 0

    for fp in synthesis_files:
        replaced, unresolved = repair_file(fp, mapping)
        if replaced or unresolved:
            status = f"  {fp.name}"
            if replaced:
                status += f"  ✓ {replaced} fixed"
            if unresolved:
                status += f"  ✗ {unresolved} unresolved"
            print(status)
        if replaced:
            files_touched += 1
        total_replaced += replaced
        total_unresolved += unresolved

    print(f"\n{'[DRY RUN] ' if DRY_RUN else ''}Done.")
    print(f"  Synthesis files touched   : {files_touched}")
    print(f"  Links repaired            : {total_replaced}")
    print(f"  Links unresolved          : {total_unresolved}")

    if total_unresolved:
        print("\n  Unresolved links are learn/ slugs with no matching ## heading in topic/.")
        print("  These may be from synthesis pages generated before the migration ran,")
        print("  referencing learn files that were already deleted before migration.")

    if DRY_RUN:
        print("\n  Re-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
