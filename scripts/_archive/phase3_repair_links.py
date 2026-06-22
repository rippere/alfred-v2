#!/usr/bin/env python3
"""
Phase 3 — Repair broken wikilinks after consolidation.

After Phases 1 and 2 deleted ~700+ files, surviving vault files may contain:
  - [[synthesis/foo-358]] links pointing to deleted numbered variants
  - [[topic/agent-design]] links pointing to merged-away topic files
  - cluster_sources: frontmatter entries pointing to deleted files

This script loads the redirect logs from Phases 1a, 1b, and 2, then:
  1. Rewrites body wikilinks using the redirect map
  2. Cleans cluster_sources frontmatter (removes dead entries, updates redirected ones)
  3. Commits the result

Run AFTER phases 1a, 1b, and 2:
    uv run python scripts/phase3_repair_links.py --dry-run
    uv run python scripts/phase3_repair_links.py

Scans ALL vault .md files (not just synthesis/).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import frontmatter as fm_lib

PROJECT_ROOT = Path(__file__).parent.parent

# These are resolved at runtime after arg parsing (see _resolve_paths)
VAULT_PATH: Path = None  # type: ignore[assignment]
DATA_DIR: Path = None    # type: ignore[assignment]
IGNORE_DIRS = {"inbox", "wiki", "_archived", "_templates", "_bases", ".obsidian"}


def _resolve_paths(config_path: Path | None) -> None:
    """Set module-level VAULT_PATH and DATA_DIR from config or hardcoded defaults."""
    global VAULT_PATH, DATA_DIR
    if config_path is not None:
        import yaml
        raw = yaml.safe_load(config_path.read_text())
        VAULT_PATH = Path(raw["vault"]["path"]).expanduser()
        data_dir_raw = raw.get("data_dir", "./data")
        DATA_DIR = (config_path.parent / data_dir_raw).resolve()
    else:
        VAULT_PATH = Path("/mnt/external/obsidian-vault")
        DATA_DIR = PROJECT_ROOT / "data"

# Matches [[synthesis/foo-358]] or [[topic/agent-design]] or [[synthesis/foo-358|display]]
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]+)?\]\]")


def load_redirect_logs() -> dict[str, str | None]:
    """Merge all phase redirect logs into one map: deleted_rel → canonical_rel (or None)."""
    combined: dict[str, str | None] = {}
    # Load old phase logs
    for phase in ["phase1a", "phase1b", "phase2"]:
        log_path = DATA_DIR / f"redirect-log-{phase}.json"
        if log_path.exists():
            data = json.loads(log_path.read_text())
            combined.update(data)
            print(f"  Loaded {log_path.name}: {len(data)} entries")
        else:
            print(f"  [skip] {log_path.name} not found")
    # Load any sledgehammer redirect logs
    for slg in sorted(DATA_DIR.glob("sledgehammer-redirect-log-*.json")):
        data = json.loads(slg.read_text())
        combined.update(data)
        print(f"  Loaded {slg.name}: {len(data)} entries")
    return combined


def _rel_from_wikilink(link_text: str) -> str:
    """Convert wikilink target to rel_path. e.g. 'synthesis/foo-358' → 'synthesis/foo-358.md'"""
    # Strip leading/trailing whitespace
    target = link_text.strip()
    # If it already looks like a path with .md, keep it; otherwise add .md
    if not target.endswith(".md"):
        return target + ".md"
    return target


def repair_body(body: str, redirect_map: dict[str, str | None]) -> tuple[str, int, int]:
    """
    Rewrite wikilinks in body text using redirect_map.
    Returns (new_body, n_rewritten, n_removed).
    """
    rewritten = 0
    removed = 0
    result = body

    for match in WIKILINK_RE.finditer(body):
        link_text = match.group(1)
        # Only process synthesis/ and topic/ links (not [[project/foo]] etc.)
        if not (link_text.startswith("synthesis/") or link_text.startswith("topic/")):
            continue

        rel_path = _rel_from_wikilink(link_text)

        # Is this file still on disk? If yes, no action needed.
        if (VAULT_PATH / rel_path).exists():
            continue

        # File is gone — check redirect map for a canonical target
        full_match = match.group(0)  # e.g. [[synthesis/foo-358]]
        if rel_path in redirect_map:
            target = redirect_map[rel_path]
            if target is not None:
                # Rewrite to canonical (strip .md suffix for wikilink format)
                new_link_target = target.removesuffix(".md")
                new_link = f"[[{new_link_target}]]"
                result = result.replace(full_match, new_link)
                rewritten += 1
            else:
                # No canonical — remove the dead link entirely
                result = result.replace(full_match, "")
                removed += 1
        else:
            # File deleted without a redirect record — remove the dead link
            result = result.replace(full_match, "")
            removed += 1

    return result.strip(), rewritten, removed


def repair_cluster_sources(sources: list, redirect_map: dict[str, str | None]) -> tuple[list, int, int]:
    """
    Clean cluster_sources frontmatter list.
    Returns (new_sources, n_rewritten, n_removed).
    """
    new_sources = []
    rewritten = 0
    removed = 0

    for src in sources:
        if not isinstance(src, str):
            new_sources.append(src)
            continue

        # cluster_sources entries look like: [[synthesis/foo-358]] or synthesis/foo-358
        # Extract the rel_path
        m = WIKILINK_RE.match(src.strip())
        if m:
            link_text = m.group(1).strip()
        else:
            link_text = src.strip()

        rel_path = _rel_from_wikilink(link_text)

        if (VAULT_PATH / rel_path).exists():
            new_sources.append(src)  # still alive
            continue

        if rel_path in redirect_map:
            target = redirect_map[rel_path]
            if target is not None:
                new_wikilink = f"[[{target.removesuffix('.md')}]]"
                new_sources.append(new_wikilink)
                rewritten += 1
            else:
                removed += 1  # drop dead reference
        else:
            # Not in redirect map — file was deleted without redirect record. Remove it.
            removed += 1

    return new_sources, rewritten, removed


def scan_vault_files() -> list[Path]:
    files = []
    for fp in sorted(VAULT_PATH.rglob("*.md")):
        rel = fp.relative_to(VAULT_PATH)
        if any(part in IGNORE_DIRS for part in rel.parts):
            continue
        files.append(fp)
    return files


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config.yaml (overrides hardcoded vault path)")
    args = parser.parse_args()

    _resolve_paths(args.config)

    print("Phase 3 — Wikilink Repair")
    if args.dry_run:
        print("[DRY RUN]\n")

    print("Loading redirect logs...")
    redirect_map = load_redirect_logs()
    print(f"  Total redirect entries: {len(redirect_map)}\n")

    if not redirect_map:
        print("No redirect entries found — nothing to repair. Exiting.")
        sys.exit(0)

    print("Scanning vault files...")
    vault_files = scan_vault_files()
    print(f"  Found {len(vault_files)} files to scan\n")

    total_body_rewritten = 0
    total_body_removed = 0
    total_sources_rewritten = 0
    total_sources_removed = 0
    files_touched = 0
    errors = 0

    for fp in vault_files:
        rel_path = str(fp.relative_to(VAULT_PATH)).replace("\\", "/")
        try:
            post = fm_lib.load(str(fp))
        except Exception as e:
            print(f"  [error] Parse failed {rel_path}: {e}")
            errors += 1
            continue

        original_body = post.content
        original_fm = dict(post.metadata)
        fm = dict(original_fm)
        changed = False
        # Initialize so dry-run prints are always safe to reference
        br = brem = sr = srem = 0

        # Repair body wikilinks
        new_body, br, brem = repair_body(original_body, redirect_map)
        if br or brem:
            total_body_rewritten += br
            total_body_removed += brem
            changed = True

        # Repair cluster_sources frontmatter
        if "cluster_sources" in fm and isinstance(fm["cluster_sources"], list):
            new_sources, sr, srem = repair_cluster_sources(fm["cluster_sources"], redirect_map)
            if sr or srem:
                fm["cluster_sources"] = new_sources
                total_sources_rewritten += sr
                total_sources_removed += srem
                changed = True

        if not changed:
            continue

        files_touched += 1
        if args.dry_run:
            print(f"  [dry] {rel_path}")
            if br:
                print(f"        body: {br} rewritten")
            if brem:
                print(f"        body: {brem} removed")
            if sr:
                print(f"        sources: {sr} rewritten")
            if srem:
                print(f"        sources: {srem} removed")
        else:
            new_post = fm_lib.Post(new_body, **fm)
            fp.write_text(fm_lib.dumps(new_post) + "\n", encoding="utf-8")

    if not args.dry_run and files_touched > 0:
        try:
            subprocess.run(["git", "add", "-A"], cwd=VAULT_PATH, check=True, capture_output=True)
            msg = (
                f"Phase 3: repaired {files_touched} files — "
                f"{total_body_rewritten + total_sources_rewritten} links rewritten, "
                f"{total_body_removed + total_sources_removed} dead links removed"
            )
            subprocess.run(["git", "commit", "-m", msg], cwd=VAULT_PATH, check=True)
            print(f"\nGit commit: '{msg}'")
        except subprocess.CalledProcessError as e:
            print(f"\n[warn] Git commit failed: {e}")

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done.")
    print(f"  Files scanned:          {len(vault_files)}")
    print(f"  Files touched:          {files_touched}")
    print(f"  Body links rewritten:   {total_body_rewritten}")
    print(f"  Body links removed:     {total_body_removed}")
    print(f"  Sources rewritten:      {total_sources_rewritten}")
    print(f"  Sources removed:        {total_sources_removed}")
    print(f"  Parse errors:           {errors}")
    print(f"\nNext: uv run python scripts/phase4_rebuild_milvus.py --dry-run")


if __name__ == "__main__":
    main()
