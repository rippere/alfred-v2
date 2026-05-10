#!/usr/bin/env python3
"""
Sledgehammer — bulk cleanup of the Alfred vault.

Categories processed:
  topic/      delete if body lines < 60 AND source count ([[) < 3
  synthesis/  delete if total file lines < 50
  assumption/ delete if no based_on key or it is empty/null
  session/    archive to _archived/session/ if status != active AND mtime > 60 days
  task/       delete if status in {done, cancelled, completed} AND mtime > 90 days

Usage:
    uv run python scripts/sledgehammer.py --dry-run            (default — no changes)
    uv run python scripts/sledgehammer.py --execute            (live run)
    uv run python scripts/sledgehammer.py --config PATH --dry-run
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import frontmatter as fm_lib
import yaml

PROJECT_ROOT = Path(__file__).parent.parent

TASK_DONE_STATUSES = {"done", "cancelled", "completed"}


def load_config(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text())


def parse_vault_path(raw: dict) -> Path:
    return Path(raw["vault"]["path"]).expanduser()


def resolve_data_dir(raw: dict, config_path: Path) -> Path:
    data_dir_raw = raw.get("data_dir", "./data")
    return (config_path.parent / data_dir_raw).resolve()


def file_age_days(fp: Path) -> float:
    mtime = fp.stat().st_mtime
    now = time.time()
    return (now - mtime) / 86400.0


def count_source_refs(body: str) -> int:
    """Count [[wikilink]] occurrences in body text."""
    return body.count("[[")


def body_lines(content: str) -> int:
    return len([ln for ln in content.splitlines() if ln.strip()])


def total_lines(fp: Path) -> int:
    return len(fp.read_text(encoding="utf-8", errors="replace").splitlines())


def should_delete_topic(fp: Path) -> tuple[bool, str]:
    try:
        post = fm_lib.load(str(fp))
    except Exception as e:
        return False, f"parse error: {e}"
    bl = body_lines(post.content)
    sc = count_source_refs(post.content)
    if bl < 60 and sc < 3:
        return True, f"body_lines={bl} source_refs={sc}"
    return False, f"keep: body_lines={bl} source_refs={sc}"


def should_delete_synthesis(fp: Path) -> tuple[bool, str]:
    tl = total_lines(fp)
    if tl < 50:
        return True, f"total_lines={tl}"
    return False, f"keep: total_lines={tl}"


def should_delete_assumption(fp: Path) -> tuple[bool, str]:
    try:
        post = fm_lib.load(str(fp))
    except Exception as e:
        return False, f"parse error: {e}"
    fm = post.metadata
    based_on = fm.get("based_on", None)
    if based_on is None:
        return True, "no based_on key"
    if not based_on:
        return True, "based_on is empty/null"
    return False, f"keep: based_on present ({len(based_on) if isinstance(based_on, list) else 'scalar'})"


def should_archive_session(fp: Path) -> tuple[bool, str]:
    try:
        post = fm_lib.load(str(fp))
    except Exception as e:
        return False, f"parse error: {e}"
    fm = post.metadata
    status = str(fm.get("status", "")).lower()
    age = file_age_days(fp)
    if status != "active" and age > 60:
        return True, f"status={status!r} age={age:.0f}d"
    return False, f"keep: status={status!r} age={age:.0f}d"


def should_delete_task(fp: Path) -> tuple[bool, str]:
    try:
        post = fm_lib.load(str(fp))
    except Exception as e:
        return False, f"parse error: {e}"
    fm = post.metadata
    status = str(fm.get("status", "")).lower()
    age = file_age_days(fp)
    if status in TASK_DONE_STATUSES and age > 90:
        return True, f"status={status!r} age={age:.0f}d"
    return False, f"keep: status={status!r} age={age:.0f}d"


def run_subprocess(cmd: list[str], label: str) -> None:
    print(f"\n--- Running {label} ---")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[warn] {label} exited with code {result.returncode}")
    else:
        print(f"[ok] {label} complete")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Sledgehammer vault cleanup")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True,
                      help="Show what would be changed (default)")
    mode.add_argument("--execute", action="store_true",
                      help="Actually perform deletions and archives")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Path to config.yaml (default: project root config.yaml)",
    )
    args = parser.parse_args()
    execute = args.execute
    dry_run = not execute

    # Load config
    raw = load_config(args.config)
    vault_path = parse_vault_path(raw)
    data_dir = resolve_data_dir(raw, args.config)
    ignore_dirs = set(raw.get("vault", {}).get("ignore_dirs", [
        "inbox", "wiki", "_archived", "_templates", "_bases", ".obsidian"
    ]))

    mode_label = "[DRY RUN]" if dry_run else "[EXECUTE]"
    print(f"Sledgehammer {mode_label}")
    print(f"  Vault:  {vault_path}")
    print(f"  Config: {args.config}")
    print(f"  Data:   {data_dir}\n")

    # Redirect log path
    today = datetime.now().strftime("%Y-%m-%d")
    log_path = data_dir / f"sledgehammer-redirect-log-{today}.json"
    redirect_log: dict[str, None] = {}

    # Stats per category
    stats: dict[str, dict[str, int]] = {}
    for cat in ["topic", "synthesis", "assumption", "session", "task"]:
        stats[cat] = {"scanned": 0, "deleted": 0, "archived": 0, "skipped": 0}

    # ── topic/ ────────────────────────────────────────────────────────────────
    cat = "topic"
    topic_dir = vault_path / cat
    if topic_dir.exists():
        for fp in sorted(topic_dir.glob("*.md")):
            stats[cat]["scanned"] += 1
            should, reason = should_delete_topic(fp)
            rel = str(fp.relative_to(vault_path))
            if should:
                stats[cat]["deleted"] += 1
                redirect_log[rel] = None
                if dry_run:
                    print(f"  [dry-delete] {rel}  ({reason})")
                else:
                    print(f"  [delete] {rel}  ({reason})")
                    fp.unlink()
            else:
                stats[cat]["skipped"] += 1
    else:
        print(f"  [skip] {cat}/ not found in vault")

    # ── synthesis/ ────────────────────────────────────────────────────────────
    cat = "synthesis"
    synth_dir = vault_path / cat
    if synth_dir.exists():
        for fp in sorted(synth_dir.glob("*.md")):
            stats[cat]["scanned"] += 1
            should, reason = should_delete_synthesis(fp)
            rel = str(fp.relative_to(vault_path))
            if should:
                stats[cat]["deleted"] += 1
                redirect_log[rel] = None
                if dry_run:
                    print(f"  [dry-delete] {rel}  ({reason})")
                else:
                    print(f"  [delete] {rel}  ({reason})")
                    fp.unlink()
            else:
                stats[cat]["skipped"] += 1
    else:
        print(f"  [skip] {cat}/ not found in vault")

    # ── assumption/ ───────────────────────────────────────────────────────────
    cat = "assumption"
    assump_dir = vault_path / cat
    if assump_dir.exists():
        for fp in sorted(assump_dir.glob("*.md")):
            stats[cat]["scanned"] += 1
            should, reason = should_delete_assumption(fp)
            rel = str(fp.relative_to(vault_path))
            if should:
                stats[cat]["deleted"] += 1
                redirect_log[rel] = None
                if dry_run:
                    print(f"  [dry-delete] {rel}  ({reason})")
                else:
                    print(f"  [delete] {rel}  ({reason})")
                    fp.unlink()
            else:
                stats[cat]["skipped"] += 1
    else:
        print(f"  [skip] {cat}/ not found in vault")

    # ── session/ ──────────────────────────────────────────────────────────────
    cat = "session"
    session_dir = vault_path / cat
    archive_dir = vault_path / "_archived" / "session"
    if session_dir.exists():
        if not dry_run:
            archive_dir.mkdir(parents=True, exist_ok=True)
        for fp in sorted(session_dir.glob("*.md")):
            stats[cat]["scanned"] += 1
            should, reason = should_archive_session(fp)
            rel = str(fp.relative_to(vault_path))
            if should:
                stats[cat]["archived"] += 1
                dest_rel = str(("_archived" / Path(rel)))
                if dry_run:
                    print(f"  [dry-archive] {rel} → {dest_rel}  ({reason})")
                else:
                    dest = vault_path / "_archived" / "session" / fp.name
                    print(f"  [archive] {rel} → {dest_rel}  ({reason})")
                    shutil.move(str(fp), str(dest))
            else:
                stats[cat]["skipped"] += 1
    else:
        print(f"  [skip] {cat}/ not found in vault")

    # ── task/ ─────────────────────────────────────────────────────────────────
    cat = "task"
    task_dir = vault_path / cat
    if task_dir.exists():
        for fp in sorted(task_dir.glob("*.md")):
            stats[cat]["scanned"] += 1
            should, reason = should_delete_task(fp)
            rel = str(fp.relative_to(vault_path))
            if should:
                stats[cat]["deleted"] += 1
                redirect_log[rel] = None
                if dry_run:
                    print(f"  [dry-delete] {rel}  ({reason})")
                else:
                    print(f"  [delete] {rel}  ({reason})")
                    fp.unlink()
            else:
                stats[cat]["skipped"] += 1
    else:
        print(f"  [skip] {cat}/ not found in vault")

    # ── Write redirect log ────────────────────────────────────────────────────
    if not dry_run:
        log_path.write_text(json.dumps(redirect_log, indent=2) + "\n")
        print(f"\nRedirect log written: {log_path}  ({len(redirect_log)} entries)")

        # Run repair + rebuild
        python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
        repair_script = str(PROJECT_ROOT / "scripts" / "phase3_repair_links.py")
        rebuild_script = str(PROJECT_ROOT / "scripts" / "phase4_rebuild_milvus.py")
        config_str = str(args.config)

        run_subprocess([python, repair_script, "--config", config_str], "phase3_repair_links")
        run_subprocess([python, rebuild_script, "--config", config_str], "phase4_rebuild_milvus")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"{'Category':<14} {'Scanned':>8} {'Deleted':>8} {'Archived':>9} {'Skipped':>8}")
    print(f"{'─'*62}")
    totals = {"scanned": 0, "deleted": 0, "archived": 0, "skipped": 0}
    for cat, s in stats.items():
        print(f"{cat:<14} {s['scanned']:>8} {s['deleted']:>8} {s['archived']:>9} {s['skipped']:>8}")
        for k in totals:
            totals[k] += s[k]
    print(f"{'─'*62}")
    print(f"{'TOTAL':<14} {totals['scanned']:>8} {totals['deleted']:>8} {totals['archived']:>9} {totals['skipped']:>8}")
    print(f"\nMode: {mode_label}")
    if dry_run:
        print("Re-run with --execute to apply changes.")


if __name__ == "__main__":
    main()
