#!/usr/bin/env python3
"""
Vault partition migration script.

Reads data/partition-manifest.json and moves files from the main vault into
the appropriate specialized vault (vault-neuroscience, vault-finance, vault-personal).

Phases:
  1. Copy files to target vaults (preserving directory structure)
  2. Verify all copies succeeded (size + hash check)
  3. Delete originals from main vault
  4. Git commit in each affected repo

Safety:
  - Dry-run by default (--execute to actually move)
  - Phase 2 verification required before any delete
  - Aborts if any copy fails

Usage:
  uv run python scripts/partition_vault.py [--dry-run]   # preview only
  uv run python scripts/partition_vault.py --execute     # run for real
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

VAULT_MAP: dict[str, Path] = {
    "neuroscience": Path("/mnt/external/vault-neuroscience"),
    "finance": Path("/mnt/external/vault-finance"),
    "personal": Path("/mnt/external/vault-personal"),
}
MAIN_VAULT = Path("/mnt/external/obsidian-vault")

MANIFEST_PATH = DATA_DIR / "partition-manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(vault_path: Path, message: str) -> None:
    try:
        subprocess.run(["git", "add", "-A"], cwd=vault_path, check=True, capture_output=True)
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=vault_path, capture_output=True
        )
        if result.returncode == 0:
            print(f"  [git] nothing to commit in {vault_path.name}")
            return
        subprocess.run(["git", "commit", "-m", message], cwd=vault_path, check=True,
                       capture_output=True)
        print(f"  [git] committed: {vault_path.name}")
    except subprocess.CalledProcessError as e:
        print(f"  [warn] git commit failed in {vault_path.name}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Actually perform the migration (default: dry-run)")
    args = parser.parse_args()
    dry_run = not args.execute

    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found at {MANIFEST_PATH}")
        print("Run generate_partition_manifest.py first.")
        sys.exit(1)

    manifest: dict[str, str] = json.loads(MANIFEST_PATH.read_text())
    print(f"Partition migration {'[DRY RUN] ' if dry_run else ''}— {len(manifest)} files")
    print()

    # Verify target vaults exist
    for domain, vault_path in VAULT_MAP.items():
        if not vault_path.exists():
            print(f"ERROR: target vault missing: {vault_path}")
            print("Create it with: mkdir -p <path> && git init <path>")
            sys.exit(1)

    # Phase 1: Copy
    print("Phase 1 — Copying files to target vaults...")
    copies: list[tuple[Path, Path]] = []
    copy_errors: list[str] = []

    for rel_path, domain in manifest.items():
        src = MAIN_VAULT / rel_path
        dst = VAULT_MAP[domain] / rel_path

        if not src.exists():
            print(f"  [skip] source missing: {rel_path}")
            continue

        if dry_run:
            copies.append((src, dst))
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            copies.append((src, dst))
        except Exception as e:
            print(f"  [error] copy failed {rel_path}: {e}")
            copy_errors.append(rel_path)

    if copy_errors:
        print(f"\nAborting: {len(copy_errors)} copy errors. Fix before retrying.")
        sys.exit(1)

    print(f"  Copied {len(copies)} files")

    # Phase 2: Verify
    print("\nPhase 2 — Verifying copies...")
    verify_errors: list[str] = []

    if not dry_run:
        for src, dst in copies:
            if not dst.exists():
                print(f"  [error] dst missing after copy: {dst}")
                verify_errors.append(str(dst))
                continue
            if src.stat().st_size != dst.stat().st_size:
                print(f"  [error] size mismatch: {src.name}")
                verify_errors.append(str(dst))
                continue
            if sha256(src) != sha256(dst):
                print(f"  [error] hash mismatch: {src.name}")
                verify_errors.append(str(dst))

        if verify_errors:
            print(f"\nAborting: {len(verify_errors)} verification failures. Originals preserved.")
            sys.exit(1)

        print(f"  All {len(copies)} copies verified OK")
    else:
        print(f"  [dry-run] would verify {len(copies)} copies")

    # Phase 3: Delete originals
    print("\nPhase 3 — Deleting originals from main vault...")
    deleted = 0

    if not dry_run:
        for src, _dst in copies:
            try:
                src.unlink()
                deleted += 1
            except Exception as e:
                print(f"  [warn] delete failed {src}: {e}")
        print(f"  Deleted {deleted} files from main vault")
    else:
        print(f"  [dry-run] would delete {len(copies)} files from main vault")

    # Phase 4: Git commits
    if not dry_run and deleted > 0:
        print("\nPhase 4 — Git commits...")

        # Count per vault
        vault_counts: dict[str, int] = {}
        for rel_path, domain in manifest.items():
            vault_counts[domain] = vault_counts.get(domain, 0) + 1

        # Commit each target vault
        for domain, vault_path in VAULT_MAP.items():
            count = vault_counts.get(domain, 0)
            if count > 0:
                git_commit(vault_path, f"partition: add {count} files from main vault")

        # Commit main vault (removals)
        git_commit(MAIN_VAULT, f"partition: remove {deleted} files migrated to specialized vaults")

    print()
    by_domain: dict[str, int] = {}
    for rel_path, domain in manifest.items():
        by_domain[domain] = by_domain.get(domain, 0) + 1

    print(f"{'[DRY RUN] ' if dry_run else ''}Summary:")
    for domain, count in sorted(by_domain.items()):
        verb = "would migrate" if dry_run else "migrated"
        print(f"  {verb} {count:4d} files → vault-{domain}")
    print(f"  Total: {len(manifest)} files")

    if dry_run:
        print("\nRun with --execute to perform the migration.")


if __name__ == "__main__":
    main()
