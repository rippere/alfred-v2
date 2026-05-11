#!/usr/bin/env python3
"""
B5 — Absorb short assumption files into related decision records.

For each assumption with body < 50 lines:
  1. Looks for a related decision via `based_on` frontmatter links (preferred)
     or title similarity (fallback).
  2. If found: appends the assumption body as a section to the decision file,
     sets assumption status: absorbed, and moves it to _archived/assumption/.
  3. If no decision match: skips (assumption stays put).

Alfred must be stopped before running:
    systemctl --user stop alfred.service

Usage:
    uv run python scripts/phase_absorb_assumptions.py              # dry-run (default)
    uv run python scripts/phase_absorb_assumptions.py --execute    # commit changes

Output:
    data/absorb-assumptions-report-YYYY-MM-DD.json
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import Optional

import frontmatter as fm_lib

VAULT_PATH = Path("/mnt/external/obsidian-vault")
ASSUMPTION_DIR = VAULT_PATH / "assumption"
DECISION_DIR = VAULT_PATH / "decision"
ARCHIVE_DIR = VAULT_PATH / "_archived" / "assumption"
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

MAX_BODY_LINES = 50       # assumptions with body >= 50 lines are left alone
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]+)?\]\]")


def _load_file(fp: Path) -> tuple[dict, str] | None:
    try:
        post = fm_lib.load(str(fp))
        return dict(post.metadata), post.content
    except Exception as e:
        print(f"  [warn] Parse failed {fp.name}: {e}")
        return None


def _save_file(fp: Path, fm: dict, body: str) -> None:
    post = fm_lib.Post(body, **fm)
    fp.write_text(fm_lib.dumps(post) + "\n", encoding="utf-8")


def _title_tokens(name: str) -> set[str]:
    """Lowercase word tokens from a title, stripping common stopwords."""
    stopwords = {"the", "a", "an", "of", "in", "on", "at", "is", "are", "was",
                 "for", "to", "and", "or", "not", "with", "by", "from", "that"}
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    return {t for t in tokens if t not in stopwords and len(t) > 2}


def _find_decision_via_based_on(based_on: list) -> Optional[Path]:
    """Resolve [[decision/foo]] links in based_on frontmatter."""
    for item in based_on:
        item_str = str(item)
        m = WIKILINK_RE.search(item_str)
        if m:
            link_target = m.group(1).strip()
        else:
            link_target = item_str.strip()

        # Normalize: strip leading "decision/"
        link_target = link_target.removeprefix("decision/").removesuffix(".md")
        candidate = DECISION_DIR / f"{link_target}.md"
        if candidate.exists():
            return candidate
    return None


def _find_decision_via_title(assumption_name: str) -> Optional[Path]:
    """Find the decision file whose title most overlaps with the assumption name."""
    assumption_tokens = _title_tokens(assumption_name)
    if not assumption_tokens:
        return None

    best_path: Optional[Path] = None
    best_score = 0.0

    for dp in DECISION_DIR.glob("*.md"):
        decision_tokens = _title_tokens(dp.stem)
        if not decision_tokens:
            continue
        overlap = assumption_tokens & decision_tokens
        if not overlap:
            continue
        # Jaccard similarity
        score = len(overlap) / len(assumption_tokens | decision_tokens)
        if score > best_score:
            best_score = score
            best_path = dp

    # Only accept if Jaccard >= 0.25 (at least 1-in-4 tokens match)
    if best_score >= 0.25:
        return best_path
    return None


def process_assumption(fp: Path, dry_run: bool) -> dict:
    """
    Returns a status dict:
      {"file": name, "action": "absorbed"|"skipped"|"no_match"|"error", "target": str|None}
    """
    data = _load_file(fp)
    if data is None:
        return {"file": fp.name, "action": "error", "target": None}

    fm, body = data
    body_lines = [ln for ln in body.strip().splitlines() if ln.strip()]

    if len(body_lines) >= MAX_BODY_LINES:
        return {"file": fp.name, "action": "skipped", "target": None,
                "reason": f"body has {len(body_lines)} lines (>= {MAX_BODY_LINES})"}

    assumption_name = fm.get("name", fp.stem)

    # Find target decision
    based_on = fm.get("based_on", [])
    if isinstance(based_on, str):
        based_on = [based_on]

    decision_fp: Optional[Path] = None
    match_method = ""

    if based_on:
        decision_fp = _find_decision_via_based_on(based_on)
        if decision_fp:
            match_method = "based_on"

    if decision_fp is None:
        decision_fp = _find_decision_via_title(assumption_name)
        if decision_fp:
            match_method = "title_similarity"

    if decision_fp is None:
        return {"file": fp.name, "action": "no_match", "target": None}

    # Build the section to append to decision
    section_header = f"## Assumption: {assumption_name}"
    source = fm.get("source", "")
    confidence = fm.get("confidence", "")
    meta_lines = []
    if source:
        meta_lines.append(f"- **Source:** {source}")
    if confidence:
        meta_lines.append(f"- **Confidence:** {confidence}")
    meta_block = "\n".join(meta_lines) + "\n\n" if meta_lines else ""
    section_body = f"\n\n{section_header}\n\n{meta_block}{body.strip()}\n"

    result = {
        "file": fp.name,
        "action": "absorbed",
        "target": str(decision_fp.relative_to(VAULT_PATH)),
        "match_method": match_method,
        "body_lines": len(body_lines),
    }

    if dry_run:
        return result

    # Append to decision file
    try:
        decision_data = _load_file(decision_fp)
        if decision_data is None:
            return {"file": fp.name, "action": "error", "target": str(decision_fp.name)}
        dec_fm, dec_body = decision_data
        dec_body = dec_body.rstrip() + section_body
        _save_file(decision_fp, dec_fm, dec_body)
    except Exception as e:
        return {"file": fp.name, "action": "error", "target": str(decision_fp.name),
                "error": str(e)}

    # Mark assumption absorbed and archive it
    try:
        fm["status"] = "absorbed"
        fm["absorbed_into"] = str(decision_fp.relative_to(VAULT_PATH))
        _save_file(fp, fm, body)

        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        dest = ARCHIVE_DIR / fp.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(fp), str(dest))
    except Exception as e:
        return {"file": fp.name, "action": "error", "target": str(decision_fp.name),
                "error": f"archive step: {e}"}

    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Commit changes (default is dry-run)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N candidates")
    args = parser.parse_args()

    dry_run = not args.execute
    DATA_DIR.mkdir(exist_ok=True)

    if not ASSUMPTION_DIR.exists():
        print(f"[error] assumption/ not found at {ASSUMPTION_DIR}")
        return
    if not DECISION_DIR.exists():
        print(f"[error] decision/ not found at {DECISION_DIR}")
        return

    mode = "[DRY RUN]" if dry_run else "[EXECUTE]"
    print(f"B5 — Absorb Short Assumptions into Decision Records {mode}")
    print(f"Vault: {VAULT_PATH}")
    print()

    candidates = sorted(ASSUMPTION_DIR.glob("*.md"))
    if args.limit:
        candidates = candidates[:args.limit]

    results = []
    for fp in candidates:
        r = process_assumption(fp, dry_run)
        results.append(r)
        action = r["action"]
        if action == "absorbed":
            target = r.get("target", "?")
            method = r.get("match_method", "?")
            lines = r.get("body_lines", "?")
            print(f"  [absorbed] {fp.stem[:50]:50s}  → {target}  (via {method}, {lines} lines)")
        elif action == "no_match":
            print(f"  [no_match] {fp.stem[:50]:50s}")
        elif action == "skipped":
            reason = r.get("reason", "")
            print(f"  [skipped]  {fp.stem[:50]:50s}  {reason}")
        elif action == "error":
            print(f"  [ERROR]    {fp.stem[:50]:50s}  {r.get('error', '')}")

    absorbed = [r for r in results if r["action"] == "absorbed"]
    no_match = [r for r in results if r["action"] == "no_match"]
    skipped  = [r for r in results if r["action"] == "skipped"]
    errors   = [r for r in results if r["action"] == "error"]

    print(f"""
Summary:
  Total candidates:  {len(results)}
  Absorbed:          {len(absorbed)}
  No decision match: {len(no_match)}
  Skipped (large):   {len(skipped)}
  Errors:            {len(errors)}
""")

    today = date.today().isoformat()
    report_path = DATA_DIR / f"absorb-assumptions-report-{today}.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"Report: {report_path}")

    if not dry_run and absorbed:
        try:
            subprocess.run(["git", "add", "-A"], cwd=VAULT_PATH, check=True, capture_output=True)
            msg = f"B5: absorbed {len(absorbed)} short assumptions into decision records"
            subprocess.run(["git", "commit", "-m", msg], cwd=VAULT_PATH, check=True)
            print(f"\nGit commit: '{msg}'")
        except subprocess.CalledProcessError as e:
            print(f"\n[warn] Git commit failed: {e}")


if __name__ == "__main__":
    main()
