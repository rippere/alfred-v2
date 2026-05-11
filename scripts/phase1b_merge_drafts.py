#!/usr/bin/env python3
"""
Phase 1b — Merge draft numbered synthesis variants into their canonicals via Claude.

For each canonical that has draft-numbered variants, this script:
  1. Reads all variants and the canonical
  2. Calls Claude to produce one richer merged body
  3. Writes the merged body back to the canonical (status → active)
  4. Deletes the numbered variants

Run AFTER phase1a:  uv run python scripts/phase1a_safe_deletes.py

Usage:
    uv run python scripts/phase1b_merge_drafts.py --dry-run         (preview)
    uv run python scripts/phase1b_merge_drafts.py                   (execute)
    uv run python scripts/phase1b_merge_drafts.py --limit 10        (process first 10 only)
    uv run python scripts/phase1b_merge_drafts.py --resume          (skip already-merged canonicals)

Output:
    data/redirect-log-phase1b.json  — maps deleted variant → canonical
    data/phase1b-errors.json        — groups that failed (for manual review)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import frontmatter as fm_lib

VAULT_PATH = Path("/mnt/external/obsidian-vault")
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

ANTHROPIC_MODEL = "claude-sonnet-4-6"
MIN_BODY_LEN = 150     # merged body shorter than this is treated as a failed merge
RATE_DELAY = 1.5       # seconds between Claude calls
MAX_RETRIES = 3
COMMIT_EVERY = 20      # git commit after every N successful merges


def _load_dotenv() -> None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_file(rel_path: str) -> tuple[dict, str] | None:
    fp = VAULT_PATH / rel_path
    if not fp.exists():
        return None
    try:
        post = fm_lib.load(str(fp))
        return dict(post.metadata), post.content
    except Exception as e:
        print(f"    [warn] Parse failed {rel_path}: {e}")
        return None


def call_claude(canonical_slug: str, canonical_body: str, variants: list[tuple[str, str]]) -> str | None:
    """Merge via Anthropic API, falling back to Ollama if credits are exhausted."""
    variant_blocks = "\n\n".join(
        f"### Variant: {slug}\n{body.strip()}"
        for slug, body in variants
    )

    prompt = (
        f"You are consolidating duplicate entries in a personal knowledge vault.\n\n"
        f"All files below cover the same topic: \"{canonical_slug}\"\n\n"
        f"CANONICAL (this is the primary file — preserve its structure):\n"
        f"{canonical_body.strip()}\n\n"
        f"VARIANTS TO ABSORB:\n"
        f"{variant_blocks}\n\n"
        f"Write a single merged synthesis page. Rules:\n"
        f"- Keep the four-section structure: ## Insight, ## Evidence, ## Implications, ## Applicability\n"
        f"- Incorporate unique insights from the variants not already in the canonical\n"
        f"- Do not repeat content that is already present\n"
        f"- Do NOT include a ## Sources section (added automatically)\n"
        f"- Target 350-500 words\n"
        f"- Start with ## Insight — no preamble, no frontmatter"
    )

    # Try Anthropic first
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            err = str(e)
            if "credit balance" in err or "insufficient" in err.lower():
                print(f"    [warn] Anthropic credits exhausted — falling back to Ollama")
            else:
                print(f"    [warn] Anthropic error: {err[:120]} — falling back to Ollama")

    # Ollama fallback (mistral:latest)
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.environ.get("OLLAMA_LLM_MODEL", "mistral:latest")
    import httpx
    for attempt in range(MAX_RETRIES):
        try:
            resp = httpx.post(
                f"{ollama_url}/api/generate",
                json={"model": ollama_model, "prompt": prompt, "stream": False},
                timeout=120.0,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"    [error] Ollama (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)

    return None


def process_group(group: dict, dry_run: bool, already_merged: set[str]) -> dict | None:
    """
    Returns dict of {variant_rel → canonical_rel} on success,
    {} if nothing to do, None on failure.
    """
    canonical_rel = group["canonical"]
    canonical_status = group["canonical_status"]

    # Skip if canonical itself was deleted in Phase 1a
    if canonical_status in ("absorbed", "superseded"):
        return {}

    # Skip if already merged in a previous run
    if canonical_rel in already_merged:
        return {}

    # Collect draft variants that still exist on disk
    draft_variants = [
        v_rel
        for v_rel, v_status in group["variant_statuses"].items()
        if v_status == "draft" and (VAULT_PATH / v_rel).exists()
    ]

    if not draft_variants:
        return {}

    canonical_data = load_file(canonical_rel)
    if canonical_data is None:
        print(f"    [skip] Canonical missing on disk: {canonical_rel}")
        return None

    canonical_fm, canonical_body = canonical_data
    canonical_slug = Path(canonical_rel).stem

    # Collect all cluster_sources across canonical + variants
    all_sources: list[str] = list(canonical_fm.get("cluster_sources", []))

    loaded_variants: list[tuple[str, str]] = []
    for v_rel in draft_variants:
        vdata = load_file(v_rel)
        if vdata is None:
            continue
        v_fm, v_body = vdata
        for src in v_fm.get("cluster_sources", []):
            if src not in all_sources:
                all_sources.append(src)
        if v_body.strip():
            loaded_variants.append((Path(v_rel).stem, v_body))

    print(f"    {canonical_slug}: {len(loaded_variants)} variant(s) with content")

    if dry_run:
        return {v: canonical_rel for v in draft_variants}

    # If variants have no body content, just delete them (they're empty)
    if not loaded_variants:
        for v_rel in draft_variants:
            fp = VAULT_PATH / v_rel
            if fp.exists():
                fp.unlink()
        return {v: canonical_rel for v in draft_variants}

    # Call Claude
    merged_body = call_claude(canonical_slug, canonical_body, loaded_variants)
    if not merged_body or len(merged_body) < MIN_BODY_LEN:
        print(f"    [fail] Merge too short or empty for {canonical_slug}")
        return None

    # Build final body with sources section
    source_lines = "\n".join(f"- {s}" for s in all_sources)
    full_body = merged_body + (f"\n\n## Sources\n{source_lines}" if all_sources else "")

    # Overwrite canonical file
    canonical_fm["status"] = "active"
    canonical_fm["cluster_sources"] = all_sources
    post = fm_lib.Post(full_body, **canonical_fm)
    (VAULT_PATH / canonical_rel).write_text(fm_lib.dumps(post) + "\n", encoding="utf-8")

    # Delete variants
    for v_rel in draft_variants:
        fp = VAULT_PATH / v_rel
        if fp.exists():
            fp.unlink()

    time.sleep(RATE_DELAY)
    return {v: canonical_rel for v in draft_variants}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip canonicals already marked active (from a prior run)")
    args = parser.parse_args()

    _load_dotenv()

    # Find manifest
    if args.manifest:
        manifest_path = args.manifest
    else:
        manifests = sorted(DATA_DIR.glob("consolidation-manifest-*.json"), reverse=True)
        if not manifests:
            print("[error] No manifest found. Run: uv run python scripts/audit_vault.py")
            sys.exit(1)
        manifest_path = manifests[0]

    print(f"Phase 1b — Merge Draft Variants via Claude")
    print(f"Manifest: {manifest_path}")
    if args.dry_run:
        print("[DRY RUN]\n")

    manifest = json.loads(manifest_path.read_text())
    groups = manifest["synthesis"]["numbered_with_canonical"]

    # Filter: only groups where canonical survives Phase 1a and has draft variants
    actionable = [
        g for g in groups
        if g["canonical_status"] not in ("absorbed", "superseded")
        and any(s == "draft" for s in g["variant_statuses"].values())
    ]

    if args.limit:
        actionable = actionable[:args.limit]

    # Load existing redirect log to detect already-merged canonicals
    log_path = DATA_DIR / "redirect-log-phase1b.json"
    existing_redirects: dict[str, str] = {}
    if log_path.exists():
        existing_redirects = json.loads(log_path.read_text())

    already_merged: set[str] = set()
    if args.resume:
        # A canonical is "already merged" if its variants all appear in the existing log
        already_merged = {
            g["canonical"]
            for g in actionable
            if all(v in existing_redirects for v in g["variants"] if g["variant_statuses"].get(v) == "draft")
        }
        print(f"Resume mode: skipping {len(already_merged)} already-processed canonicals\n")

    print(f"Groups to process: {len(actionable)}\n")

    all_redirects: dict[str, str] = dict(existing_redirects)
    errors: list[str] = []
    succeeded = 0
    skipped = 0
    since_last_commit = 0

    for i, group in enumerate(actionable, 1):
        canonical = group["canonical"]
        n_draft = sum(1 for s in group["variant_statuses"].values() if s == "draft")
        print(f"[{i}/{len(actionable)}] {canonical} ({n_draft} draft variants)")

        result = process_group(group, args.dry_run, already_merged)

        if result is None:
            errors.append(canonical)
            print(f"    → FAILED (see phase1b-errors.json)")
        elif not result:
            skipped += 1
            print(f"    → skipped")
        else:
            all_redirects.update(result)
            succeeded += 1
            since_last_commit += 1
            print(f"    → merged {len(result)} variants")

        # Periodic commit + checkpoint
        if not args.dry_run and since_last_commit >= COMMIT_EVERY:
            log_path.write_text(json.dumps(all_redirects, indent=2))
            try:
                subprocess.run(["git", "add", "-A"], cwd=VAULT_PATH, check=True, capture_output=True)
                subprocess.run(
                    ["git", "commit", "-m", f"Phase 1b: checkpoint {succeeded} groups merged"],
                    cwd=VAULT_PATH, check=True, capture_output=True,
                )
                print(f"  [checkpoint] git commit at {succeeded} merged")
            except subprocess.CalledProcessError:
                pass
            since_last_commit = 0

    if not args.dry_run:
        # Final writes
        log_path.write_text(json.dumps(all_redirects, indent=2))

        error_path = DATA_DIR / "phase1b-errors.json"
        error_path.write_text(json.dumps(errors, indent=2))

        if succeeded > 0 or (since_last_commit > 0):
            try:
                subprocess.run(["git", "add", "-A"], cwd=VAULT_PATH, check=True, capture_output=True)
                msg = f"Phase 1b: final — {succeeded} groups merged, {len(errors)} failed"
                subprocess.run(["git", "commit", "-m", msg], cwd=VAULT_PATH, check=True)
                print(f"\nGit commit: '{msg}'")
            except subprocess.CalledProcessError as e:
                print(f"\n[warn] Final git commit failed: {e}")

        print(f"\nDone.")
        print(f"  Succeeded:    {succeeded}")
        print(f"  Skipped:      {skipped}")
        print(f"  Failed:       {len(errors)}")
        print(f"  Redirect log: {log_path}")
        if errors:
            print(f"  Error log:    {error_path}")
            print(f"  Review errors manually, then re-run with --resume")
    else:
        print(f"\nDry run: {len(actionable)} groups inspected.")
        print(f"  Would merge: {sum(1 for g in actionable if any(s == 'draft' for s in g['variant_statuses'].values()))}")

    print(f"\nNext: uv run python scripts/phase2_topic_merge.py --dry-run")


if __name__ == "__main__":
    main()
