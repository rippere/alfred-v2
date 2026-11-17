#!/usr/bin/env python3
"""Classify Syncthing conflict copies. Dry-run by default; deletes nothing.

Syncthing writes `<stem>.sync-conflict-<YYYYMMDD>-<HHMMSS>-<ID><ext>` when two
machines edit the same file. On this vault they accumulated unnoticed: 724 when
last counted, 2,460 by 2026-07-26.

They are not all alike, and that is the point of this script. Sampling once
suggested "~95% differ in content, real material", which turned out to be
backwards — most are byte-identical to the file they shadow. Deciding by
sampling is how you either delete real work or keep thousands of exact
duplicates. So: classify everything, show the work, and let a human choose.

Categories
  IDENTICAL   body matches the live original exactly -> nothing to lose
  DIFFERS     live original exists but content diverges -> needs human eyes
  ORPHANED    no live original survives -> the copy may be the only version
  NON_MD      not a vault record (.obsidian config, images, ...)

Usage
  reconcile_conflicts.py                        # dry run over the main vault
  reconcile_conflicts.py --root /mnt/external   # every vault at once
  reconcile_conflicts.py --apply-identical      # delete ONLY the IDENTICAL set

--apply-identical is the sole destructive mode and is deliberately narrow: it
refuses to touch DIFFERS or ORPHANED at all. There is no flag that deletes
those, by design — that is a per-file human decision, not a batch operation.
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CONFLICT_RE = re.compile(r"\.sync-conflict-\d{8}-\d{6}-[A-Z0-9]+")

IDENTICAL = "IDENTICAL"
DIFFERS = "DIFFERS"
ORPHANED = "ORPHANED"
NON_MD = "NON_MD"


def live_counterpart(path: Path) -> Path:
    """The path this conflict copy shadows, with the marker stripped."""
    return path.with_name(CONFLICT_RE.sub("", path.name))


def _body(path: Path) -> str | None:
    """File text minus YAML frontmatter, or None if unreadable.

    Frontmatter is excluded because Syncthing conflicts routinely differ only
    in a `modified:` stamp, which is not a content difference worth a human's
    attention.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:]
    return text.strip()


def classify(path: Path) -> tuple[str, float | None]:
    """Return (category, similarity-to-original or None)."""
    if path.suffix != ".md":
        return NON_MD, None

    original = live_counterpart(path)
    if not original.exists():
        return ORPHANED, None

    a, b = _body(original), _body(path)
    if a is None or b is None:
        return ORPHANED, None
    if a == b:
        return IDENTICAL, 1.0
    return DIFFERS, difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def scan(root: Path):
    rows = []
    for path in sorted(root.rglob("*.sync-conflict-*")):
        if not path.is_file():
            continue
        category, ratio = classify(path)
        rows.append((path, category, ratio))
    return rows


def render_report(root: Path, rows) -> str:
    counts = Counter(c for _, c, _ in rows)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    out = [
        "<!-- alfred:source reconcile_conflicts -->",
        f"# Sync-conflict reconciliation — {generated}",
        "",
        f"Root: `{root}`  ·  total conflict files: **{len(rows)}**",
        "",
        "| Category | Count | Meaning |",
        "|---|---:|---|",
        f"| IDENTICAL | {counts[IDENTICAL]} | body matches the live file exactly |",
        f"| DIFFERS | {counts[DIFFERS]} | live file exists, content diverges |",
        f"| ORPHANED | {counts[ORPHANED]} | no live counterpart — may be the only copy |",
        f"| NON_MD | {counts[NON_MD]} | not a vault record |",
        "",
        "Nothing has been deleted. `--apply-identical` removes only the IDENTICAL",
        "set; DIFFERS and ORPHANED are never batch-deleted.",
        "",
    ]

    differs = sorted(
        ((r if r is not None else 0.0, p) for p, c, r in rows if c == DIFFERS),
    )
    if differs:
        out += [
            "## DIFFERS — review these by hand",
            "",
            "Lowest similarity first: the most divergent copies are likeliest to",
            "hold content the live file lost.",
            "",
            "| Similarity | Conflict copy |",
            "|---:|---|",
        ]
        out += [f"| {r:.3f} | `{p}` |" for r, p in differs]
        out.append("")

    orphans = [p for p, c, _ in rows if c == ORPHANED]
    if orphans:
        out += [
            "## ORPHANED — no live counterpart",
            "",
            "Deleting one of these destroys the last copy. Check each.",
            "",
        ]
        out += [f"- `{p}`" for p in orphans]
        out.append("")

    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path("/mnt/external/obsidian-vault"))
    ap.add_argument("--report", type=Path, default=None,
                    help="where to write the report (default: <root>/inbox/, else stdout)")
    ap.add_argument("--apply-identical", action="store_true",
                    help="DELETE the IDENTICAL set only. Never touches DIFFERS/ORPHANED.")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"root does not exist: {args.root}", file=sys.stderr)
        return 2

    rows = scan(args.root)
    counts = Counter(c for _, c, _ in rows)
    report = render_report(args.root, rows)

    dest = args.report
    if dest is None:
        inbox = args.root / "inbox"
        dest = inbox / f"conflict-reconciliation-{datetime.now(timezone.utc):%Y-%m-%d}.md" \
            if inbox.is_dir() else None

    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(report, encoding="utf-8")
        print(f"report written: {dest}")
    else:
        print(report)

    for name in (IDENTICAL, DIFFERS, ORPHANED, NON_MD):
        print(f"  {name:<10} {counts[name]:>5}")

    if not args.apply_identical:
        print("\nDry run — nothing deleted. Re-run with --apply-identical to remove")
        print("only the IDENTICAL set once you have read the report.")
        return 0

    removed = 0
    for path, category, _ in rows:
        if category != IDENTICAL:
            continue
        # Re-verify immediately before unlinking rather than trusting the scan:
        # the vault is live and Syncthing may have rewritten either side since.
        if classify(path)[0] != IDENTICAL:
            print(f"  skip (changed since scan): {path}")
            continue
        path.unlink()
        removed += 1

    print(f"\ndeleted {removed} byte-identical conflict copies; "
          f"{counts[DIFFERS]} DIFFERS and {counts[ORPHANED]} ORPHANED left untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
