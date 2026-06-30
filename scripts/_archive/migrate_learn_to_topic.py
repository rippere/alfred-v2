#!/usr/bin/env python3
"""
Migrate learn/ atomic notes → topic/ consolidated files.

Groups all learn/*.md files by their primary tag (first tag, or "misc"),
writes each group as a single topic/{slug}.md file with ## sections,
then deletes the originals.

Run from alfred-v2/ root:
    python scripts/migrate_learn_to_topic.py [--dry-run]
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import frontmatter
import yaml

VAULT_PATH = Path("/mnt/external/obsidian-vault")
LEARN_DIR = VAULT_PATH / "learn"
TOPIC_DIR = VAULT_PATH / "topic"
DRY_RUN = "--dry-run" in sys.argv


def slugify(tag: str) -> str:
    s = tag.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "misc"


def serialize(fm: dict, body: str) -> str:
    post = frontmatter.Post(body, **fm)
    return frontmatter.dumps(post) + "\n"


def main() -> None:
    if not LEARN_DIR.exists():
        print(f"[ERROR] learn/ dir not found: {LEARN_DIR}")
        sys.exit(1)

    learn_files = sorted(LEARN_DIR.glob("*.md"))
    print(f"Found {len(learn_files)} learn/ files")

    # Group by primary tag slug
    groups: dict[str, list[dict]] = defaultdict(list)
    skipped = 0

    for fp in learn_files:
        try:
            post = frontmatter.load(str(fp))
        except yaml.YAMLError:
            print(f"  [WARN] bad frontmatter, skipping: {fp.name}")
            skipped += 1
            continue

        fm = dict(post.metadata)
        body = post.content.strip()

        tags = fm.get("tags", [])
        if not isinstance(tags, list):
            tags = [tags] if tags else []
        primary = slugify(tags[0]) if tags else "misc"

        title = fp.stem  # the insight slug is the filename without .md
        source = fm.get("source", "")

        groups[primary].append({
            "title": title,
            "body": body,
            "tags": tags,
            "source": source,
            "path": fp,
        })

    print(f"Grouped into {len(groups)} topics  (skipped {skipped} malformed files)")

    # Write topic files
    TOPIC_DIR.mkdir(parents=True, exist_ok=True)
    total_insights = 0
    topic_files_written = 0

    for slug, items in sorted(groups.items()):
        topic_path = TOPIC_DIR / f"{slug}.md"
        print(f"\n  topic/{slug}.md  ({len(items)} insights)")

        # Collect all tags across items
        all_tags: list[str] = []
        all_sources: list[str] = []
        for item in items:
            for t in item["tags"]:
                if t not in all_tags:
                    all_tags.append(t)
            if item["source"] and item["source"] not in all_sources:
                all_sources.append(item["source"])

        # Build body
        sections: list[str] = []
        for item in items:
            body_text = item["body"]
            # Strip the trailing "Source: [[...]]" line if the distiller already added it
            # (we'll re-add it cleanly below)
            body_text = re.sub(r"\n\nSource: \[\[.*?\]\]\s*$", "", body_text).strip()

            source_link = item["source"]
            if source_link.endswith(".md"):
                source_link = source_link[:-3]

            section = f"## {item['title']}\n\n{body_text}"
            if source_link:
                section += f"\n\nSource: [[{source_link}]]"
            sections.append(section)

        if topic_path.exists():
            # Append to existing topic file
            existing_post = frontmatter.load(str(topic_path))
            existing_fm = dict(existing_post.metadata)
            existing_body = existing_post.content.rstrip()

            merged_tags = existing_fm.get("tags", [])
            if not isinstance(merged_tags, list):
                merged_tags = [merged_tags] if merged_tags else []
            for t in all_tags:
                if t not in merged_tags:
                    merged_tags.append(t)
            existing_fm["tags"] = merged_tags

            merged_sources = existing_fm.get("sources", [])
            if not isinstance(merged_sources, list):
                merged_sources = [merged_sources] if merged_sources else []
            for s in all_sources:
                if s and s not in merged_sources:
                    merged_sources.append(s)
            existing_fm["sources"] = merged_sources

            new_body = existing_body + "\n\n---\n\n" + "\n\n---\n\n".join(sections) + "\n"
            if not DRY_RUN:
                topic_path.write_text(serialize(existing_fm, new_body), encoding="utf-8")
            print(f"    [APPEND] {len(items)} sections to existing file")
        else:
            # Create new topic file
            fm = {
                "type": "topic",
                "name": slug,
                "tags": all_tags,
                "sources": all_sources,
                "created": date.today().isoformat(),
                "status": "active",
            }
            body = f"# {slug}\n\n" + "\n\n---\n\n".join(sections) + "\n"
            if not DRY_RUN:
                topic_path.write_text(serialize(fm, body), encoding="utf-8")
            print(f"    [CREATE] {len(items)} sections")
            topic_files_written += 1

        total_insights += len(items)

    # Delete learn/ files
    deleted = 0
    if not DRY_RUN:
        for fp in learn_files:
            try:
                fm_check = frontmatter.load(str(fp)).metadata
                tags = fm_check.get("tags", [])
                if not isinstance(tags, list):
                    tags = [tags] if tags else []
                slug = slugify(tags[0]) if tags else "misc"
                if (TOPIC_DIR / f"{slug}.md").exists():
                    fp.unlink()
                    deleted += 1
            except Exception as e:
                print(f"  [WARN] could not delete {fp.name}: {e}")
    else:
        deleted = len(learn_files)

    print(f"\n{'[DRY RUN] ' if DRY_RUN else ''}Done.")
    print(f"  learn/ files processed : {len(learn_files)}")
    print(f"  topic/ files created   : {topic_files_written}")
    print(f"  total insights written : {total_insights}")
    print(f"  learn/ files deleted   : {deleted}")
    if DRY_RUN:
        print("\n  Re-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
