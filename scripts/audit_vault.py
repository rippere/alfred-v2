#!/usr/bin/env python3
"""
Phase 0 — Vault consolidation audit.

Scans synthesis/ and topic/ directories, classifies every file, and writes
a manifest JSON + human-readable summary. Does NOT modify any files.

Alfred must be stopped before running surgery:
    systemctl --user stop alfred.service

Run from alfred-v2/ root:
    uv run python scripts/audit_vault.py

Output:
    data/consolidation-manifest-YYYY-MM-DD.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import frontmatter

VAULT_PATH = Path("/mnt/external/obsidian-vault")
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SYNTHESIS_DIR = VAULT_PATH / "synthesis"
TOPIC_DIR = VAULT_PATH / "topic"

# Matches stems like "foo-bar-378" → group 1: "foo-bar", group 2: "378"
NUM_SUFFIX_RE = re.compile(r"^(.+)-(\d+)$")


def parse_fm(fp: Path) -> dict:
    try:
        post = frontmatter.load(str(fp))
        return dict(post.metadata)
    except Exception:
        return {}


def audit_synthesis(synthesis_dir: Path) -> dict:
    files = sorted(synthesis_dir.glob("*.md"))
    print(f"  Found {len(files)} synthesis files")

    canonicals: dict[str, dict] = {}           # slug → info
    numbered: dict[str, list[dict]] = defaultdict(list)  # base_slug → [variant info]

    for fp in files:
        stem = fp.stem
        m = NUM_SUFFIX_RE.match(stem)
        if m:
            base, num = m.group(1), m.group(2)
            fm = parse_fm(fp)
            numbered[base].append({
                "rel_path": f"synthesis/{fp.name}",
                "stem": stem,
                "base": base,
                "num": int(num),
                "status": fm.get("status", ""),
                "cluster_sources": fm.get("cluster_sources", []),
            })
        else:
            fm = parse_fm(fp)
            canonicals[stem] = {
                "rel_path": f"synthesis/{fp.name}",
                "stem": stem,
                "status": fm.get("status", ""),
                "cluster_sources": fm.get("cluster_sources", []),
            }

    # Build groups
    numbered_with_canonical: list[dict] = []
    numbered_orphans: list[str] = []

    for base, variants in sorted(numbered.items()):
        sorted_variants = sorted(variants, key=lambda x: x["num"])
        if base in canonicals:
            numbered_with_canonical.append({
                "canonical": f"synthesis/{base}.md",
                "canonical_status": canonicals[base]["status"],
                "variants": [v["rel_path"] for v in sorted_variants],
                "variant_statuses": {v["rel_path"]: v["status"] for v in sorted_variants},
            })
        else:
            for v in sorted_variants:
                numbered_orphans.append(v["rel_path"])

    # Classify canonicals by status
    superseded_canonicals = []
    absorbed_canonicals = []
    draft_canonicals = []
    active_canonicals = []

    for stem, info in sorted(canonicals.items()):
        rel = info["rel_path"]
        s = info["status"]
        if s == "superseded":
            superseded_canonicals.append(rel)
        elif s == "absorbed":
            absorbed_canonicals.append(rel)
        elif s == "draft":
            draft_canonicals.append(rel)
        else:
            active_canonicals.append(rel)  # "active" or missing status

    return {
        "numbered_with_canonical": numbered_with_canonical,
        "numbered_orphans": numbered_orphans,
        "superseded_canonicals": superseded_canonicals,
        "absorbed_canonicals": absorbed_canonicals,
        "draft_canonicals": draft_canonicals,
        "active_canonicals": active_canonicals,
    }


def audit_topics(topic_dir: Path) -> dict:
    if not topic_dir.exists():
        return {"merge_groups": [], "singletons": [], "total_files": 0}

    files = sorted(topic_dir.glob("*.md"))
    stems = [fp.stem for fp in files]
    print(f"  Found {len(stems)} topic files")

    # Group by first 2 hyphen-delimited tokens — flags obvious clusters for review
    groups: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        parts = stem.split("-")
        key = "-".join(parts[:2]) if len(parts) >= 2 else parts[0]
        groups[key].append(stem)

    merge_groups = []
    singletons = []
    for key, members in sorted(groups.items()):
        if len(members) >= 2:
            merge_groups.append(sorted(members))
        else:
            singletons.extend(members)

    return {
        "merge_groups": merge_groups,
        "singletons": sorted(singletons),
        "total_files": len(stems),
    }


def _count_safe_deletes(synthesis: dict) -> int:
    n = len(synthesis["absorbed_canonicals"]) + len(synthesis["superseded_canonicals"])
    for group in synthesis["numbered_with_canonical"]:
        for vstatus in group["variant_statuses"].values():
            if vstatus in ("absorbed", "superseded"):
                n += 1
    return n


def _count_merge_targets(synthesis: dict) -> int:
    n = 0
    for group in synthesis["numbered_with_canonical"]:
        if group["canonical_status"] not in ("absorbed", "superseded"):
            n += sum(
                1 for s in group["variant_statuses"].values() if s == "draft"
            )
    return n


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    today = date.today().isoformat()
    manifest_path = DATA_DIR / f"consolidation-manifest-{today}.json"

    print("Alfred Vault Consolidation Audit")
    print("=" * 50)

    if not SYNTHESIS_DIR.exists():
        print(f"[error] synthesis/ not found at {SYNTHESIS_DIR}")
        sys.exit(1)

    print(f"\nScanning synthesis/...")
    synthesis = audit_synthesis(SYNTHESIS_DIR)
    print(f"Scanning topic/...")
    topics = audit_topics(TOPIC_DIR)

    manifest = {
        "generated": today,
        "vault_path": str(VAULT_PATH),
        "synthesis": synthesis,
        "topic": topics,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Compute summary numbers
    nwc = synthesis["numbered_with_canonical"]
    n_orphans = len(synthesis["numbered_orphans"])
    n_superseded = len(synthesis["superseded_canonicals"])
    n_absorbed = len(synthesis["absorbed_canonicals"])
    n_draft = len(synthesis["draft_canonicals"])
    n_active = len(synthesis["active_canonicals"])
    total_numbered = sum(len(g["variants"]) for g in nwc) + n_orphans
    total_synthesis = len(nwc) + n_orphans + n_superseded + n_absorbed + n_draft + n_active
    n_safe_delete = _count_safe_deletes(synthesis)
    n_merge = _count_merge_targets(synthesis)

    print(f"""
SYNTHESIS ({total_synthesis} files total)
  Numbered with canonical:   {len(nwc)} groups ({total_numbered} numbered files)
  Numbered orphans:          {n_orphans}
  Superseded canonicals:     {n_superseded}
  Absorbed canonicals:       {n_absorbed}
  Draft canonicals:          {n_draft}
  Active canonicals:         {n_active}

WHAT WILL HAPPEN
  Phase 1a — Safe deletes:   ~{n_safe_delete} files (absorbed/superseded)
  Phase 1b — Claude merges:  ~{n_merge} draft variants → merged into canonicals
  Post-surgery synthesis:    ~{total_synthesis - n_safe_delete - n_merge} files

TOPIC ({topics["total_files"]} files)
  Merge candidate groups:    {len(topics["merge_groups"])}  (review and confirm manually)
  Singletons:                {len(topics["singletons"])}

Manifest: {manifest_path}
""")

    # Show sample numbered groups
    if nwc:
        print("Sample numbered groups (first 15):")
        for group in nwc[:15]:
            canon = group["canonical"]
            cstatus = group["canonical_status"]
            variants = group["variants"]
            print(f"  [{cstatus:12s}] {canon}")
            for v in variants:
                vstatus = group["variant_statuses"][v]
                print(f"    [{vstatus:12s}] {v}")
        if len(nwc) > 15:
            print(f"  ... and {len(nwc) - 15} more groups (see manifest)")

    # Show sample topic groups
    if topics["merge_groups"]:
        print("\nSample topic merge candidates (first 15):")
        for grp in topics["merge_groups"][:15]:
            sizes = []
            for slug in grp:
                fp = TOPIC_DIR / f"{slug}.md"
                size = fp.stat().st_size if fp.exists() else 0
                sizes.append(f"{slug} ({size // 1024}KB)")
            print(f"  {' | '.join(sizes)}")
        if len(topics["merge_groups"]) > 15:
            print(f"  ... and {len(topics['merge_groups']) - 15} more groups (see manifest)")

    print("""
NEXT STEPS
  1. Review the groups above.
  2. For topic merges, copy synthesis.topic.merge_groups from the manifest
     to data/topic-merge-confirmed.json and trim to only the groups you want
     to merge (format: [{"canonical": "slug", "merge_in": ["slug2", ...]}, ...]).
  3. Run: uv run python scripts/_archive/phase1a_safe_deletes.py --dry-run
""")


if __name__ == "__main__":
    main()
