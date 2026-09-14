"""`alfred tripwire ...` Typer sub-app: check.

Wired into the top-level app in alfred/cli.py via ``app.add_typer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

tripwire_app = typer.Typer(
    name="tripwire",
    help="Monthly x402 tripwire watcher — HOLD / FLIP-TO-GO / ABANDON-WEDGE verdicts.",
    add_completion=False,
)
console = Console()


def _render_verdict(verdict) -> None:
    color = {"HOLD": "yellow", "FLIP-TO-GO": "green", "ABANDON-WEDGE": "red"}.get(verdict.verdict, "white")
    console.print(f"\n[bold {color}]{verdict.verdict}[/bold {color}]\n")

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column(style="dim")
    table.add_column()
    s = verdict.signals
    table.add_row("ecosystem volume", f"${s.ecosystem_volume_usd:,.0f}/mo")
    table.add_row("baseline", f"${s.baseline_volume_usd:,.0f}/mo")
    table.add_row("coinbase seller dashboard shipped", str(s.coinbase_seller_dashboard_shipped))
    console.print(table)

    console.print("\n[bold]Reasons[/bold]")
    for r in verdict.reasons:
        console.print(f"  - {r}")


@tripwire_app.command()
def check(
    signals: Optional[Path] = typer.Option(None, "--signals", help="Path to the signals YAML (default: data/tripwire-signals.yaml)"),
    write_inbox: bool = typer.Option(True, "--write-inbox/--no-write-inbox", help="Write the verdict note to the Alfred inbox"),
):
    """Evaluate this month's x402 signals and write a verdict note to the Alfred inbox."""
    from alfred.tripwire import inbox as I
    from alfred.tripwire import watcher as W

    try:
        loaded = W.load_signals(signals)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    verdict = W.evaluate(loaded)
    _render_verdict(verdict)

    if write_inbox:
        out_path = I.write_verdict_note(verdict)
        console.print(f"\n[dim]Verdict note written: {out_path}[/dim]")
