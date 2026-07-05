"""`alfred meetily ...` Typer sub-app: sync, status.

Wired into the top-level app in alfred/cli.py via ``app.add_typer``.

    alfred meetily status                 # is the Meetily DB found? how many meetings?
    alfred meetily sync                    # ingest new meetings into the vault inbox
    alfred meetily sync --dry-run          # show what would be ingested, write nothing
    alfred meetily sync --since 2026-06-01 # only meetings created on/after a date
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

meetily_app = typer.Typer(
    name="meetily",
    help="Ingest Meetily meeting notes (transcripts + summaries) into the Alfred vault.",
    add_completion=False,
)
console = Console()

# Same default-config resolution the top-level CLI uses.
_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent.parent / "config.yaml"


def _load_cfg(config: Path):
    from alfred.config import AlfredConfig
    return AlfredConfig.load(config)


def _resolve_db(explicit: Optional[str], config: Path):
    """Resolve the Meetily DB path, reading meetily.db_path from config.yaml if present."""
    import yaml
    from alfred.meetily.config import resolve_db_path

    cfg_db = None
    try:
        raw = yaml.safe_load(Path(config).read_text()) or {}
        cfg_db = (raw.get("meetily") or {}).get("db_path")
    except (OSError, yaml.YAMLError):
        pass
    return resolve_db_path(explicit=explicit, cfg_path=cfg_db)


@meetily_app.command()
def status(
    config: Path = typer.Option(_DEFAULT_CONFIG, "--config", "-c"),
    db: Optional[str] = typer.Option(None, "--db", help="Path to Meetily SQLite DB"),
):
    """Show whether the Meetily DB is reachable and how many meetings it holds."""
    from alfred.meetily.ingest import load_synced
    from alfred.meetily.reader import read_meetings

    cfg = _load_cfg(config)
    db_path = _resolve_db(db, config)

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column()
    if not db_path:
        console.print("[red]Meetily DB not found.[/red] Set $MEETILY_DB or meetily.db_path in config.yaml.")
        raise typer.Exit(1)

    meetings = read_meetings(db_path)
    synced = load_synced(cfg.data_dir)
    pending = sum(1 for m in meetings if m.id not in synced)

    t.add_row("meetily db", str(db_path))
    t.add_row("vault", str(cfg.vault_path))
    t.add_row("meetings in db", str(len(meetings)))
    t.add_row("already synced", str(len(synced)))
    t.add_row("pending sync", f"[green]{pending}[/green]" if pending else "0")
    console.print("[bold cyan]Meetily → Alfred[/bold cyan]")
    console.print(t)


@meetily_app.command()
def sync(
    config: Path = typer.Option(_DEFAULT_CONFIG, "--config", "-c"),
    db: Optional[str] = typer.Option(None, "--db", help="Path to Meetily SQLite DB"),
    since: Optional[str] = typer.Option(None, "--since", help="Only meetings created on/after YYYY-MM-DD"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would happen; write nothing"),
    force: bool = typer.Option(False, "--force", help="Re-ingest even already-synced meetings"),
):
    """Ingest new Meetily meetings into the vault inbox (Curator files them from there)."""
    from alfred.meetily.ingest import sync as run_sync

    cfg = _load_cfg(config)
    db_path = _resolve_db(db, config)
    if not db_path:
        console.print("[red]Meetily DB not found.[/red] Set $MEETILY_DB or meetily.db_path in config.yaml.")
        raise typer.Exit(1)

    res = run_sync(
        db_path, cfg.vault_path, cfg.data_dir,
        since=since, dry_run=dry_run, force=force,
    )

    verb = "would ingest" if dry_run else "ingested"
    console.print(
        f"[green]{res.ingested} {verb}[/green] · "
        f"{res.skipped} already synced · {res.scanned} scanned"
    )
    for title in (res.ingested_titles or [])[:20]:
        console.print(f"  [dim]+[/dim] {title}")
    if dry_run:
        console.print("[yellow]dry-run — no files written[/yellow]")
    elif res.ingested:
        console.print("[dim]Curator will file these into meeting/ within ~10s.[/dim]")
