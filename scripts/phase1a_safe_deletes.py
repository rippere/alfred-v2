#!/usr/bin/env python3
"""
Phase 1a — Delete absorbed/superseded synthesis files (safe, no merging needed).

These files are explicitly marked as replaced — their content was already
incorporated into something else. Deleting them is safe.

Alfred must be stopped:  systemctl --user stop alfred.service
Run after audit:         uv run python scripts/audit_vault.py

Usage:
    uv run python scripts/phase1a_safe_deletes.py --dry-run   (preview only)
    uv run python scripts/phase1a_safe_deletes.py             (execute)
    uv run python scripts/phase1a_safe_deletes.py --manifest data/consolidation-manifest-2026-05-04.json

Output:
    data/redirect-log-phase1a.json  — maps deleted_rel_path → canonical (or null)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

VAULT_PATH = Path("/mnt/external/obsidian-vault")
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SAFE_STATUSES = {"absorbed", "superseded"}


def build_delete_list(synthesis: dict) -> tuple[list[tuple[str, str | None]], list[str]]:
    """
    Returns (to_delete, warnings).
    to_delete: list of (rel_path, redirect_target_or_None)
    warnings:  human-readable issues found
    """
    to_delete: list[tuple[str, str | None]] = []
    warnings: list[str] = []

    # Absorbed/superseded standalone canonicals
    for rel in synthesis["absorbed_canonicals"] + synthesis["superseded_canonicals"]:
        to_delete.append((rel, None))

    # Numbered variants that are absorbed/superseded
    for group in synthesis["numbered_with_canonical"]:
        canonical_rel = group["canonical"]
        canonical_status = group["canonical_status"]
        canonical_being_deleted = canonical_status in SAFE_STATUSES

        for v_rel, v_status in group["variant_statuses"].items():
            if v_status not in SAFE_STATUSES:
                continue

            if canonical_being_deleted:
                # Canonical is also going away — no redirect target
                to_delete.append((v_rel, None))
            else:
                to_delete.append((v_rel, canonical_rel))

        # Warn if a canonical is being deleted but has surviving draft variants
        if canonical_being_deleted:
            draft_variants = [
                v for v, s in group["variant_statuses"].items() if s == "draft"
            ]
            if draft_variants:
                warnings.append(
                    f"WARNING: {canonical_rel} ({canonical_status}) is being deleted "
                    f"but has {len(draft_variants)} draft variant(s) that will become orphans: "
                    f"{draft_variants[:3]}"
                )

    return to_delete, warnings


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    # Find most recent manifest
    if args.manifest:
        manifest_path = args.manifest
    else:
        manifests = sorted(DATA_DIR.glob("consolidation-manifest-*.json"), reverse=True)
        if not manifests:
            print("[error] No manifest found. Run: uv run python scripts/audit_vault.py")
            sys.exit(1)
        manifest_path = manifests[0]

    print(f"Phase 1a — Safe Deletes")
    print(f"Manifest: {manifest_path}")
    if args.dry_run:
        print("[DRY RUN] No files will be modified.\n")
    else:
        print()

    manifest = json.loads(manifest_path.read_text())
    synthesis = manifest["synthesis"]

    to_delete, warnings = build_delete_list(synthesis)

    for w in warnings:
        print(w)
    if warnings:
        print()

    print(f"Files to delete: {len(to_delete)}")

    deleted = 0
    already_gone = 0
    errors = 0
    redirect_log: dict[str, str | None] = {}

    for rel, target in sorted(to_delete, key=lambda x: x[0]):
        fp = VAULT_PATH / rel
        if not fp.exists():
            already_gone += 1
            redirect_log[rel] = target
            continue

        if args.dry_run:
            status_tag = "[->canonical]" if target else "[->none]"
            print(f"  [dry] {status_tag} {rel}")
        else:
            try:
                fp.unlink()
                deleted += 1
                redirect_log[rel] = target
                if deleted % 100 == 0:
                    print(f"  {deleted}/{len(to_delete)} deleted...")
            except Exception as e:
                print(f"  [error] {rel}: {e}")
                errors += 1

    if args.dry_run:
        print(f"\nDry run: {len(to_delete)} files would be deleted.")
        return

    # Write redirect log
    log_path = DATA_DIR / "redirect-log-phase1a.json"
    log_path.write_text(json.dumps(redirect_log, indent=2))

    # Git commit
    try:
        subprocess.run(["git", "add", "-A"], cwd=VAULT_PATH, check=True, capture_output=True)
        msg = f"Phase 1a: delete {deleted} absorbed/superseded synthesis files"
        subprocess.run(["git", "commit", "-m", msg], cwd=VAULT_PATH, check=True)
        print(f"\nGit commit: '{msg}'")
    except subprocess.CalledProcessError as e:
        print(f"\n[warn] Git commit failed: {e}")

    print(f"\nDone.")
    print(f"  Deleted:     {deleted}")
    print(f"  Already gone: {already_gone}")
    print(f"  Errors:      {errors}")
    print(f"  Redirect log: {log_path}")
    print(f"\nNext: uv run python scripts/phase1b_merge_drafts.py --dry-run")


if __name__ == "__main__":
    main()
