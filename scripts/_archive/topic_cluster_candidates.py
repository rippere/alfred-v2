#!/usr/bin/env python3
"""
B3 — Topic cluster candidates: apply canonicalization map + near-synonym grouping.

Reads all topic slugs from topic/, groups near-synonyms by canonical slug, and
outputs data/topic-merge-candidates-YYYY-MM-DD.json.

Each group includes member file counts and line counts so the user can choose
which to confirm for phase2_topic_merge.py.

Usage:
    uv run python scripts/topic_cluster_candidates.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import frontmatter

VAULT_PATH = Path("/mnt/external/obsidian-vault")
TOPIC_DIR = VAULT_PATH / "topic"
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Mirror of distiller._TAG_CANONICAL — maps variant slug → canonical slug.
# Add entries here to group more topics; remove to keep them separate.
_TAG_CANONICAL: dict[str, str] = {
    # --- ai / agents / llm cluster ---
    "agent":                   "ai-agents",
    "agents":                  "ai-agents",
    "agentic":                 "ai-agents",
    "agentic-ai":              "ai-agents",
    "agentic-systems":         "ai-agents",
    "agentic-workflow":        "ai-agents",
    "agentic-builds":          "ai-agents",
    "agentic-coding":          "ai-agents",
    "agent-systems":           "ai-agents",
    "agent-setup":             "ai-agents",
    "agent-orchestration":     "ai-agents",
    "agent-design":            "ai-agents",
    "agent-architecture":      "ai-agents",
    "multi-agent":             "ai-agents",
    "llm-systems":             "llm",
    "llm-architecture":        "llm",
    "llm-pipelines":           "llm",
    "llm-workflows":           "llm",
    "local-llm":               "llm",
    "local-ai":                "llm",
    "local-inference":         "llm",
    "local-ml":                "llm",
    "foundation-models":       "llm",
    # ai-systems → ai (most content)
    "ai-systems":              "ai",
    "ai-systems-design":       "ai",
    "artificial-intelligence": "ai",
    # --- knowledge cluster ---
    "knowledge-systems":       "knowledge-management",
    "knowledge-organization":  "knowledge-management",
    "knowledge-architecture":  "knowledge-management",
    "knowledge-quality":       "knowledge-management",
    "knowledge-integrity":     "knowledge-management",
    "knowledge-capture":       "knowledge-management",
    "knowledge-distillation":  "knowledge-management",
    "knowledge-extraction":    "knowledge-management",
    "knowledge-transfer":      "knowledge-management",
    "pkm":                     "knowledge-management",
    "personal-knowledge-management": "knowledge-management",
    "second-brain":            "knowledge-management",
    # --- workflow cluster ---
    "workflows":               "workflow",
    "workflow-automation":     "workflow",
    "workflow-design":         "workflow",
    "workflow-evolution":      "workflow",
    "workflow-optimization":   "workflow",
    "workflow-orchestration":  "workflow",
    "workflow-sequencing":     "workflow",
    # --- architecture / system-design cluster ---
    "systems-design":          "system-design",
    "software-design":         "software-architecture",
    "software-engineering":    "software-architecture",
    # --- graph cluster ---
    "knowledge-graphs":        "knowledge-graph",
}


def _line_count(fp: Path) -> int:
    try:
        return fp.read_text(encoding="utf-8").count("\n")
    except OSError:
        return 0


def _file_info(slug: str) -> dict:
    fp = TOPIC_DIR / f"{slug}.md"
    if not fp.exists():
        return {"slug": slug, "exists": False, "lines": 0, "tags": []}
    try:
        post = frontmatter.load(str(fp))
        tags = post.metadata.get("tags", [])
        if not isinstance(tags, list):
            tags = [tags] if tags else []
    except Exception:
        tags = []
    return {
        "slug": slug,
        "exists": True,
        "lines": _line_count(fp),
        "tags": [str(t) for t in tags],
    }


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    if not TOPIC_DIR.exists():
        print(f"[error] topic/ not found at {TOPIC_DIR}")
        return

    all_slugs = sorted(fp.stem for fp in TOPIC_DIR.glob("*.md"))
    print(f"Found {len(all_slugs)} topic files in {TOPIC_DIR}")

    # Group slugs by their canonical target.
    # A slug that IS a canonical will be the key.
    # A slug that maps TO a canonical goes into its group.
    # Slugs with no mapping are standalone (singletons).
    canonical_groups: dict[str, list[str]] = defaultdict(list)

    for slug in all_slugs:
        canonical = _TAG_CANONICAL.get(slug)
        if canonical and canonical != slug:
            canonical_groups[canonical].append(slug)
        # else: this slug is either a canonical or an unmapped singleton

    # Build output: only groups where canonical file exists AND has at least 1 member variant
    output_groups = []
    for canonical, members in sorted(canonical_groups.items()):
        existing_members = [m for m in members if (TOPIC_DIR / f"{m}.md").exists()]
        if not existing_members:
            continue
        if not (TOPIC_DIR / f"{canonical}.md").exists():
            # Canonical slug doesn't exist as a file — pick the richest member as canonical
            richest = max(existing_members, key=lambda s: _line_count(TOPIC_DIR / f"{s}.md"))
            real_canonical = richest
            merge_in = [m for m in existing_members if m != richest]
        else:
            real_canonical = canonical
            merge_in = existing_members

        if not merge_in:
            continue

        canonical_info = _file_info(real_canonical)
        member_infos = [_file_info(m) for m in merge_in]
        total_lines = canonical_info["lines"] + sum(m["lines"] for m in member_infos)

        output_groups.append({
            "canonical": real_canonical,
            "canonical_lines": canonical_info["lines"],
            "merge_in": merge_in,
            "member_lines": {m: _file_info(m)["lines"] for m in merge_in},
            "total_lines_if_merged": total_lines,
            "note": f"Merge {len(merge_in)} slug(s) into {real_canonical}",
        })

    # Sort by total content descending (largest merges first)
    output_groups.sort(key=lambda g: g["total_lines_if_merged"], reverse=True)

    today = date.today().isoformat()
    out_path = DATA_DIR / f"topic-merge-candidates-{today}.json"
    out_path.write_text(json.dumps(output_groups, indent=2))

    print(f"\nCanonical groups found: {len(output_groups)}")
    print(f"Output: {out_path}\n")

    print("Groups (sorted by total content):")
    for g in output_groups:
        print(f"  {g['canonical']:40s}  canonical={g['canonical_lines']} lines | "
              f"merge_in={g['merge_in']}  total={g['total_lines_if_merged']} lines")

    print(f"""
NEXT STEPS
  Review {out_path}
  Copy it to data/topic-merge-confirmed.json (trim groups you want to keep separate).
  Format for confirmed file:
    [{{"canonical": "ai-agents", "merge_in": ["agent", "agents", ...]}}, ...]
  Then run:
    uv run python scripts/phase2_topic_merge.py --dry-run
    uv run python scripts/phase2_topic_merge.py
""")


if __name__ == "__main__":
    main()
