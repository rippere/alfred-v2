"""Write the monthly tripwire verdict to the Alfred inbox as a markdown note.

Mirrors the `<!-- alfred:source ... -->` convention used by the other
inbox-writing paths in this repo (scripts/alfred-alert-inbox.sh,
scripts/alfred-heartbeat-check.sh) so curator ingests it the same way.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from alfred.tripwire import config as C
from alfred.tripwire.watcher import Verdict


def render_note(verdict: Verdict, *, now: datetime | None = None) -> str:
    ts = (now or datetime.now().astimezone()).strftime("%Y-%m-%d %H:%M")
    s = verdict.signals
    reasons_block = "\n".join(f"- {r}" for r in verdict.reasons)
    moves_block = (
        "\n".join(f"- **{k}**: {v}" for k, v in s.competitor_moves.items())
        if s.competitor_moves else "- (none reported)"
    )
    notes_block = f"\n**Notes:** {s.notes}\n" if s.notes else ""
    return f"""<!-- alfred:source x402_tripwire_watcher -->
# x402 Tripwire Verdict — {ts}

## Verdict: **{verdict.verdict}**

{reasons_block}

## Signals this run
- ecosystem volume: ${s.ecosystem_volume_usd:,.0f}/mo (baseline ${s.baseline_volume_usd:,.0f}/mo)
- Coinbase seller-dashboard shipped: {s.coinbase_seller_dashboard_shipped}

### Competitor embedded-middleware moves
{moves_block}
{notes_block}
---
Source: ADR-001 flip/abandon triggers (x402 × machine-readable-interfaces strategy, 2026-07-15).
"""


def write_verdict_note(verdict: Verdict, *, inbox_dir: Path | None = None, now: datetime | None = None) -> Path:
    """Render + write the note; returns the path written."""
    target_dir = inbox_dir or C.INBOX_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now().astimezone()).strftime("%Y%m%d-%H%M%S")
    out_path = target_dir / f"x402-tripwire-{stamp}.md"
    out_path.write_text(render_note(verdict, now=now))
    return out_path
