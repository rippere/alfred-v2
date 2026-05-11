#!/usr/bin/env python3
"""
Phase 2 — Merge semantic near-duplicate topic files.

Topic files are structured with ## section headings — merging is mechanical
concatenation (no Claude needed). You review the candidate groups from the
audit manifest and confirm which ones to merge.

STEP 1: Run audit to get candidates:
    cat data/consolidation-manifest-YYYY-MM-DD.json | python -c "
    import json,sys
    m = json.load(sys.stdin)
    groups = m['topic']['merge_groups']
    print(json.dumps([{'canonical': g[0], 'merge_in': g[1:]} for g in groups], indent=2))
    " > data/topic-merge-confirmed.json

STEP 2: Edit data/topic-merge-confirmed.json — remove groups you don't want,
         adjust which slug is the canonical (largest/best-named file).

STEP 3: Run:
    uv run python scripts/phase2_topic_merge.py --dry-run
    uv run python scripts/phase2_topic_merge.py

Format of data/topic-merge-confirmed.json:
[
  {
    "canonical": "ai-agents",
    "merge_in": ["agent", "agent-architecture", "agent-design", "agent-systems", "agentic"]
  },
  ...
]

Output:
    data/redirect-log-phase2.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import frontmatter as fm_lib

VAULT_PATH = Path("/mnt/external/obsidian-vault")
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TOPIC_DIR = VAULT_PATH / "topic"

CONFIRMED_PATH = DATA_DIR / "topic-merge-confirmed.json"
COMMIT_EVERY = 30


def merge_topic_files(canonical_slug: str, merge_slugs: list[str], dry_run: bool) -> dict | None:
    """
    Concatenate all merge_slugs sections into canonical_slug.
    Returns {merged_slug → canonical_rel} on success, None on error.
    """
    canonical_rel = f"topic/{canonical_slug}.md"
    canonical_fp = TOPIC_DIR / f"{canonical_slug}.md"

    if not canonical_fp.exists():
        print(f"    [error] Canonical not found: {canonical_rel}")
        return None

    try:
        post = fm_lib.load(str(canonical_fp))
        canonical_fm = dict(post.metadata)
        canonical_body = post.content
    except Exception as e:
        print(f"    [error] Parse failed for {canonical_rel}: {e}")
        return None

    existing_tags: list = canonical_fm.get("tags", [])
    if not isinstance(existing_tags, list):
        existing_tags = [existing_tags] if existing_tags else []

    existing_sources: list = canonical_fm.get("sources", [])
    if not isinstance(existing_sources, list):
        existing_sources = [existing_sources] if existing_sources else []

    appended_sections = ""
    redirects: dict[str, str] = {}

    for slug in merge_slugs:
        fp = TOPIC_DIR / f"{slug}.md"
        if not fp.exists():
            print(f"    [skip] Not found: topic/{slug}.md")
            continue

        try:
            vpost = fm_lib.load(str(fp))
            vfm = dict(vpost.metadata)
            vbody = vpost.content.strip()
        except Exception as e:
            print(f"    [warn] Parse failed for topic/{slug}.md: {e}")
            continue

        # Collect tags and sources from variant
        for t in vfm.get("tags", []):
            if t and t not in existing_tags:
                existing_tags.append(t)
        for s in vfm.get("sources", []):
            if s and s not in existing_sources:
                existing_sources.append(s)

        # Append body sections (skip empty files)
        if vbody:
            appended_sections += f"\n\n---\n\n{vbody}"

        redirects[f"topic/{slug}.md"] = canonical_rel

    if dry_run:
        print(f"    [dry] Would merge {len(redirects)} file(s) into {canonical_rel}")
        return redirects

    if not redirects:
        print(f"    [skip] No merge targets existed on disk")
        return {}

    # Write merged canonical
    new_body = canonical_body.rstrip() + appended_sections
    canonical_fm["tags"] = existing_tags
    canonical_fm["sources"] = existing_sources
    new_post = fm_lib.Post(new_body, **canonical_fm)
    canonical_fp.write_text(fm_lib.dumps(new_post) + "\n", encoding="utf-8")

    # Delete merged-in files
    for slug in merge_slugs:
        fp = TOPIC_DIR / f"{slug}.md"
        if fp.exists():
            fp.unlink()

    return redirects


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirmed", type=Path, default=CONFIRMED_PATH)
    args = parser.parse_args()

    if not args.confirmed.exists():
        print(f"[error] Confirmed groups file not found: {args.confirmed}")
        print()
        print("Generate it from the manifest:")
        print("  cat data/consolidation-manifest-*.json | python3 -c \"")
        print("  import json,sys")
        print("  m = json.load(sys.stdin)")
        print("  groups = m['topic']['merge_groups']")
        print("  out = [{'canonical': g[0], 'merge_in': g[1:]} for g in groups]")
        print("  print(json.dumps(out, indent=2))\" > data/topic-merge-confirmed.json")
        print()
        print("Edit that file, then re-run this script.")
        sys.exit(1)

    print(f"Phase 2 — Topic Merge")
    print(f"Confirmed groups: {args.confirmed}")
    if args.dry_run:
        print("[DRY RUN]\n")

    confirmed_groups = json.loads(args.confirmed.read_text())
    print(f"Groups to process: {len(confirmed_groups)}\n")

    all_redirects: dict[str, str] = {}
    succeeded = 0
    skipped = 0
    errors = 0
    since_last_commit = 0

    for i, group in enumerate(confirmed_groups, 1):
        canonical_slug = group["canonical"]
        merge_in = group.get("merge_in", [])

        if not merge_in:
            print(f"[{i}] {canonical_slug} — no merge_in entries, skipping")
            skipped += 1
            continue

        print(f"[{i}/{len(confirmed_groups)}] {canonical_slug} ← {merge_in}")

        result = merge_topic_files(canonical_slug, merge_in, args.dry_run)

        if result is None:
            errors += 1
        elif not result:
            skipped += 1
        else:
            all_redirects.update(result)
            succeeded += 1
            since_last_commit += 1
            print(f"    → merged {len(result)} file(s)")

        if not args.dry_run and since_last_commit >= COMMIT_EVERY:
            try:
                subprocess.run(["git", "add", "-A"], cwd=VAULT_PATH, check=True, capture_output=True)
                subprocess.run(
                    ["git", "commit", "-m", f"Phase 2: checkpoint {succeeded} topic groups merged"],
                    cwd=VAULT_PATH, check=True, capture_output=True,
                )
                print(f"  [checkpoint] git commit")
            except subprocess.CalledProcessError:
                pass
            since_last_commit = 0

    if not args.dry_run:
        log_path = DATA_DIR / "redirect-log-phase2.json"
        log_path.write_text(json.dumps(all_redirects, indent=2))

        if succeeded > 0:
            try:
                subprocess.run(["git", "add", "-A"], cwd=VAULT_PATH, check=True, capture_output=True)
                msg = f"Phase 2: {succeeded} topic groups merged, {len(all_redirects)} files removed"
                subprocess.run(["git", "commit", "-m", msg], cwd=VAULT_PATH, check=True)
                print(f"\nGit commit: '{msg}'")
            except subprocess.CalledProcessError as e:
                print(f"\n[warn] Git commit failed: {e}")

        print(f"\nDone.")
        print(f"  Succeeded:    {succeeded}")
        print(f"  Skipped:      {skipped}")
        print(f"  Errors:       {errors}")
        print(f"  Redirect log: {log_path}")
    else:
        print(f"\nDry run complete. {succeeded} groups would merge {len(all_redirects)} files.")

    print(f"\nNext: uv run python scripts/phase3_repair_links.py --dry-run")


if __name__ == "__main__":
    main()
