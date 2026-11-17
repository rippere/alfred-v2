"""`alfred bridge ...` Typer sub-app: enrich.

Wired into the top-level app in alfred/cli.py via ``app.add_typer``.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.table import Table

bridge_app = typer.Typer(
    name="bridge",
    help="Alfred -> NovaCRM deal/contact brief enrichment (Pattern A).",
    add_completion=False,
)
console = Console()

# Resolve relative to repo root regardless of CWD (this file is
# src/alfred/bridge/cli.py -> repo root is three parents up), matching the
# top-level alfred.cli's own `_DEFAULT_CONFIG` convention.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _REPO_ROOT / "config.yaml"


@bridge_app.command()
def enrich(
    config: Path = typer.Option(_DEFAULT_CONFIG, "--config", "-c", help="Path to config.yaml"),
    push: bool = typer.Option(
        False,
        "--push",
        help=(
            "Actually POST synthesized briefs as CRM notes (requires "
            "credentials in ~/.config/alfred-ledger/env). Omitted by "
            "default: resolves contacts, synthesizes briefs, and checks for "
            "duplicates, but makes zero write calls (dry run)."
        ),
    ),
    top_k: int = typer.Option(6, "--top-k", help="Vault retrieval depth per resolved contact"),
):
    """Resolve CRM contacts on active deals to vault entities and post
    synthesized briefs as CRM notes.

    Dry-run by default (equivalent to an explicit ``--dry-run``): every
    resolution, brief synthesis, and duplicate check still runs and is
    printed, but the note-creation POST is never made. Pass ``--push`` to
    perform the actual write.
    """
    from alfred.bridge.enrich_crm import run_enrich
    from alfred.config import AlfredConfig
    from alfred.query.engine import QueryEngine

    dry_run = not push
    mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[red]LIVE PUSH[/red]"
    console.print(f"\n[bold]alfred bridge enrich[/bold] — {mode}\n")

    cfg = AlfredConfig.load(config)
    engine = QueryEngine(cfg)

    summary = run_enrich(cfg, engine, dry_run=dry_run, top_k=top_k)

    if summary.decisions:
        detail_table = Table(title="decisions", box=box.SIMPLE, padding=(0, 1))
        detail_table.add_column("deal", style="cyan", no_wrap=True)
        detail_table.add_column("contact", no_wrap=True)
        detail_table.add_column("outcome", style="magenta")
        detail_table.add_column("brief / detail", overflow="fold")
        for d in summary.decisions:
            outcome = d.get("outcome", "?")
            color = {
                "would_post": "yellow", "posted": "green",
                "skipped_duplicate": "dim", "no_match": "dim",
                "no_brief": "dim", "error": "red",
            }.get(outcome, "white")
            note = d.get("brief_text") or d.get("post_detail") or ""
            detail_table.add_row(
                str(d.get("deal_id")),
                str(d.get("contact_name") or d.get("contact_id")),
                f"[{color}]{outcome}[/{color}]",
                note[:160],
            )
        console.print(detail_table)

    table = Table(title="bridge enrich summary", box=box.SIMPLE, padding=(0, 1))
    table.add_column("metric", style="cyan")
    table.add_column("count", justify="right", style="green")
    for metric, value in summary.as_dict().items():
        table.add_row(metric, str(value))
    console.print(table)

    if dry_run:
        console.print(
            "\n[dim]Dry run only — no notes were posted. "
            "Re-run with --push to write for real.[/dim]"
        )
    else:
        console.print(f"\n[green]Posted {summary.posted} note(s).[/green]")
