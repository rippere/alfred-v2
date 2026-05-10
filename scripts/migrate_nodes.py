#!/usr/bin/env python3
"""
Migrate vault nodes to domain-specific vaults.

Domain mapping:
  vault-neuroscience: project/ nodes tagged neuro/tribe/psych/cognitive/fmri/learning
                      topic/ nodes with neuro/brain/cognit/fmri/roi/psycho/sleep/learning/
                             memory/somatot/spinal/neuroplast/neurochemist/behavioral-neuroscience/
                             nervous-system slugs

  vault-finance:      project/ nodes for memecoin/sector-flow/quant/behavioral-finance
                      topic/ nodes with finance/trading/quant/crypto/market/sentiment/
                             behavioral-finance/covariance/portfolio/investment slugs

  vault-personal:     session/ nodes with "1-1" or "home-rippere" in name or project
                              "Home System Configuration" or "Personal Health"
                      note/ nodes that are class notes or personal reflection
                      topic/ nodes: health/sleep/self-/skill-/career/system-config/
                                    keyboard/hardware/ssh/syncthing slugs

Bridge projects (copy to BOTH vaults):
  tribe-social → vault-neuroscience + vault-finance (project node)
  EMM summary → vault-personal (note copy)

Originals always stay in ai-systems vault (no moves).

Usage:
    uv run python scripts/migrate_nodes.py --dry-run     (default)
    uv run python scripts/migrate_nodes.py --execute
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import NamedTuple

import frontmatter as fm_lib
import yaml

PROJECT_ROOT = Path(__file__).parent.parent

# ── Slug / name matchers ───────────────────────────────────────────────────────

NEURO_PROJECT_TAGS = {
    "neuro", "tribe", "tribe-v2", "psych", "cognitive", "fmri", "learning",
    "neuroscience", "neurophysiology", "behavioral-neuroscience", "neuromarketing",
    "psychology",
}

NEURO_TOPIC_SLUGS = re.compile(
    r"^(neuro|brain|cognit|fmri|roi|psycho|sleep|learning|memory|"
    r"somatot|spinal|neuroplast|neurochemist|behavioral-neuroscience|nervous-system)",
    re.IGNORECASE,
)

FINANCE_PROJECT_KEYWORDS = re.compile(
    r"(memecoin|sector.flow|quant|behavioral.finance|finance|trading|crypto|"
    r"investment|portfolio|sentiment|market)",
    re.IGNORECASE,
)

FINANCE_TOPIC_SLUGS = re.compile(
    r"^(finance|trading|quant|crypto|market|sentiment|behavioral-finance|"
    r"covariance|portfolio|investment)",
    re.IGNORECASE,
)

PERSONAL_SESSION_KEYWORDS = re.compile(
    r"(1-1|1_1|home.rippere|home-rippere|personal.health|Home System Configuration)",
    re.IGNORECASE,
)

PERSONAL_TOPIC_SLUGS = re.compile(
    r"^(health|sleep|self-|skill-|career|system-config|keyboard|hardware|ssh|syncthing)",
    re.IGNORECASE,
)

PERSONAL_NOTE_KEYWORDS = re.compile(
    r"(class.note|lecture|coursework|personal.reflect|1-1|1_1|home.rippere|"
    r"personal.health|daily.journal|weekly.review)",
    re.IGNORECASE,
)

BRIDGE_TRIBE = "tribe-social"
BRIDGE_EMM_NAMES = {"Executive Mind Matrix", "executive-mind-matrix", "ExecutiveMindMatrix"}


class MigrateTarget(NamedTuple):
    src: Path
    dst_vault: Path
    reason: str


def load_frontmatter(fp: Path) -> dict:
    try:
        post = fm_lib.load(str(fp))
        return dict(post.metadata)
    except Exception:
        return {}


def tags_from_fm(fm: dict) -> set[str]:
    tags = fm.get("tags", [])
    if isinstance(tags, list):
        return {str(t).lower() for t in tags}
    if isinstance(tags, str):
        return {tags.lower()}
    return set()


def name_from_fm(fm: dict, fp: Path) -> str:
    return str(fm.get("name", fp.stem)).lower()


def collect_targets(
    vault_path: Path,
    vault_neuro: Path,
    vault_finance: Path,
    vault_personal: Path,
) -> list[MigrateTarget]:
    targets: list[MigrateTarget] = []
    seen: set[tuple[Path, Path]] = set()  # (src, dst) dedup

    def add(src: Path, cat: str, dst_vault: Path, reason: str) -> None:
        dst = dst_vault / src.relative_to(vault_path)
        key = (src, dst_vault)
        if key not in seen:
            seen.add(key)
            targets.append(MigrateTarget(src=src, dst_vault=dst_vault, reason=reason))

    # ── project/ ───────────────────────────────────────────────────────────────
    project_dir = vault_path / "project"
    if project_dir.exists():
        for fp in sorted(project_dir.glob("*.md")):
            fm = load_frontmatter(fp)
            tags = tags_from_fm(fm)
            name = name_from_fm(fm, fp)
            slug = fp.stem.lower()

            # Bridge: tribe-social → both
            if BRIDGE_TRIBE in name or BRIDGE_TRIBE in slug:
                add(fp, "project", vault_neuro, "bridge: tribe-social → neuro")
                add(fp, "project", vault_finance, "bridge: tribe-social → finance")
                continue

            # Neuroscience project tags
            if tags & NEURO_PROJECT_TAGS or NEURO_TOPIC_SLUGS.match(slug):
                add(fp, "project", vault_neuro, f"neuro tags: {tags & NEURO_PROJECT_TAGS or slug}")

            # Finance projects
            if FINANCE_PROJECT_KEYWORDS.search(name) or FINANCE_PROJECT_KEYWORDS.search(slug):
                add(fp, "project", vault_finance, f"finance keyword in name/slug")

            # Personal: Home System Configuration, Personal Health
            if any(k in name or k in slug for k in ["home system", "personal health", "home-system"]):
                add(fp, "project", vault_personal, "personal project (home/health)")

    # ── topic/ ─────────────────────────────────────────────────────────────────
    topic_dir = vault_path / "topic"
    if topic_dir.exists():
        for fp in sorted(topic_dir.glob("*.md")):
            slug = fp.stem.lower()
            if NEURO_TOPIC_SLUGS.match(slug):
                add(fp, "topic", vault_neuro, f"neuro slug: {slug}")
            if FINANCE_TOPIC_SLUGS.match(slug):
                add(fp, "topic", vault_finance, f"finance slug: {slug}")
            if PERSONAL_TOPIC_SLUGS.match(slug):
                add(fp, "topic", vault_personal, f"personal slug: {slug}")

    # ── session/ ───────────────────────────────────────────────────────────────
    session_dir = vault_path / "session"
    if session_dir.exists():
        for fp in sorted(session_dir.glob("*.md")):
            fm = load_frontmatter(fp)
            name = name_from_fm(fm, fp)
            slug = fp.stem.lower()
            project = str(fm.get("project", "")).lower()
            if (PERSONAL_SESSION_KEYWORDS.search(name)
                    or PERSONAL_SESSION_KEYWORDS.search(slug)
                    or "home system configuration" in project
                    or "personal health" in project):
                add(fp, "session", vault_personal, f"personal session: {fp.name[:40]}")

    # ── note/ ─────────────────────────────────────────────────────────────────
    note_dir = vault_path / "note"
    if note_dir.exists():
        for fp in sorted(note_dir.glob("*.md")):
            slug = fp.stem.lower()
            name = slug
            fm = load_frontmatter(fp)
            if PERSONAL_NOTE_KEYWORDS.search(name):
                add(fp, "note", vault_personal, f"personal note keyword: {fp.name[:40]}")

    # ── EMM → personal (summary note) ─────────────────────────────────────────
    for cat in ["project", "note"]:
        cat_dir = vault_path / cat
        if cat_dir.exists():
            for fp in sorted(cat_dir.glob("*.md")):
                fm = load_frontmatter(fp)
                name = str(fm.get("name", fp.stem))
                if any(k in name for k in BRIDGE_EMM_NAMES) or any(k in fp.stem for k in ["executive-mind-matrix", "Executive Mind Matrix", "ExecutiveMindMatrix"]):
                    # Copy EMM summary into personal vault
                    add(fp, cat, vault_personal, "bridge: EMM summary → personal")

    return targets


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Migrate vault nodes to domain vaults")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
    )
    args = parser.parse_args()
    execute = args.execute
    dry_run = not execute

    raw = yaml.safe_load(args.config.read_text())
    vault_path = Path(raw["vault"]["path"]).expanduser()

    vault_neuro = Path("/mnt/external/vault-neuroscience")
    vault_finance = Path("/mnt/external/vault-finance")
    vault_personal = Path("/mnt/external/vault-personal")

    mode_label = "[DRY RUN]" if dry_run else "[EXECUTE]"
    print(f"migrate_nodes {mode_label}")
    print(f"  Source vault: {vault_path}\n")

    targets = collect_targets(vault_path, vault_neuro, vault_finance, vault_personal)

    # Group by destination vault for summary
    by_vault: dict[str, list[MigrateTarget]] = {}
    for t in targets:
        key = t.dst_vault.name
        by_vault.setdefault(key, []).append(t)

    copied = 0
    errors = 0

    for vault_name, items in sorted(by_vault.items()):
        print(f"\n── {vault_name} ({len(items)} files) ──")
        for t in items:
            rel = str(t.src.relative_to(vault_path))
            dst_dir = t.dst_vault / t.src.relative_to(vault_path).parent
            dst_file = dst_dir / t.src.name
            if dry_run:
                print(f"  [dry-copy] {rel}  → {vault_name}/  ({t.reason})")
            else:
                try:
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(t.src), str(dst_file))
                    print(f"  [copy] {rel}  → {vault_name}/{rel}")
                    copied += 1
                except Exception as e:
                    print(f"  [error] {rel}: {e}")
                    errors += 1

    # Summary
    print(f"\n{'─'*60}")
    print(f"{'Vault':<30} {'Files':>6}")
    print(f"{'─'*60}")
    for vault_name, items in sorted(by_vault.items()):
        print(f"{vault_name:<30} {len(items):>6}")
    print(f"{'─'*60}")
    print(f"{'TOTAL':<30} {len(targets):>6}")
    print(f"\nMode: {mode_label}")
    if not dry_run:
        print(f"  Copied: {copied}  Errors: {errors}")
    else:
        print("Re-run with --execute to apply.")


if __name__ == "__main__":
    main()
