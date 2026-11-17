"""`alfred ledger ...` Typer sub-app: collect, backfill, show.

Wired into the top-level app in alfred/cli.py via ``app.add_typer``.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

ledger_app = typer.Typer(
    name="ledger",
    help="Daily life-ledger KPI snapshots (vault + git + PM briefs → SQLite, optional CRM push).",
    add_completion=False,
)
console = Console()


def _parse_date(s: Optional[str]) -> str:
    """Validate/normalise a YYYY-MM-DD string; default to today (local)."""
    if s is None:
        return datetime.now().astimezone().strftime("%Y-%m-%d")
    try:
        return datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        console.print(f"[red]Invalid date '{s}' — expected YYYY-MM-DD[/red]")
        raise typer.Exit(2)


def _render_snapshot_table(date_str: str, rows: list[dict]) -> None:
    table = Table(title=f"ledger snapshot — {date_str}", box=box.SIMPLE, padding=(0, 1))
    table.add_column("domain", style="cyan")
    table.add_column("metric")
    table.add_column("value", justify="right", style="green")
    table.add_column("meta", style="dim", overflow="fold")
    for r in rows:
        val = r["value"]
        val_s = f"{val:.4g}" if isinstance(val, float) and val != int(val) else f"{int(val)}"
        meta = r.get("meta")
        meta_s = ", ".join(f"{k}:{v}" for k, v in meta.items()) if isinstance(meta, dict) else ""
        table.add_row(r["domain"], r["metric"], val_s, meta_s)
    console.print(table)


@ledger_app.command()
def collect(
    date: Optional[str] = typer.Option(None, "--date", help="YYYY-MM-DD (default: today, local)"),
    push: bool = typer.Option(False, "--push", help="Push the snapshot to NovaCRM (dry-run if no creds)"),
):
    """Compute and store today's (or --date's) KPI snapshot."""
    from alfred.ledger import db as L_db
    from alfred.ledger import push as L_push
    from alfred.ledger.collect import LedgerComputer

    target = _parse_date(date)
    computer = LedgerComputer()
    rows = computer.snapshot(target)

    conn = L_db.connect()
    n = L_db.upsert_snapshot(conn, target, rows)
    console.print(f"[green]Stored {n} metrics for {target}[/green]")
    _render_snapshot_table(target, rows)

    if push:
        status, detail = L_push.push_snapshot(target, rows)
        L_db.record_push(conn, target, status, detail)
        _print_push_result(status, detail)
    conn.close()

    if push and status == "error":
        # "dry" (no creds configured) is intentional and non-fatal; "error" is a
        # real push failure and must exit non-zero so ledger-collect.service's
        # OnFailure= alert can fire.
        raise typer.Exit(1)


@ledger_app.command()
def backfill(
    since: str = typer.Option(..., "--since", help="Start date YYYY-MM-DD (inclusive)"),
    until: Optional[str] = typer.Option(None, "--until", help="End date YYYY-MM-DD (inclusive; default today)"),
    push: bool = typer.Option(False, "--push", help="Push each day's snapshot to NovaCRM"),
):
    """Recompute and store snapshots for every day in a date range."""
    from alfred.ledger import db as L_db
    from alfred.ledger import push as L_push
    from alfred.ledger.collect import LedgerComputer

    start = _parse_date(since)
    end = _parse_date(until)
    if end < start:
        console.print("[red]--until is before --since[/red]")
        raise typer.Exit(2)

    console.print(f"[dim]Scanning sources…[/dim]")
    computer = LedgerComputer()
    conn = L_db.connect()

    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    days = (d1 - d0).days + 1

    total_rows = 0
    days_with_data = 0
    push_errors = 0
    cur = d0
    while cur <= d1:
        ds = cur.isoformat()
        rows = computer.snapshot(ds)
        n = L_db.upsert_snapshot(conn, ds, rows)
        total_rows += n
        if n:
            days_with_data += 1
        if push:
            status, detail = L_push.push_snapshot(ds, rows)
            L_db.record_push(conn, ds, status, detail)
            if status == "error":
                push_errors += 1
        cur += timedelta(days=1)

    console.print(
        f"[green]Backfilled {days} days ({start} → {end}); "
        f"{days_with_data} with data, {total_rows} rows written.[/green]"
    )
    counts = L_db.domain_row_counts(conn)
    dt = Table(title="rows per domain (whole DB)", box=box.SIMPLE, padding=(0, 1))
    dt.add_column("domain", style="cyan")
    dt.add_column("rows", justify="right", style="green")
    for dom, c in counts.items():
        dt.add_row(dom, str(c))
    console.print(dt)
    conn.close()

    if push_errors:
        # Report-and-continue across the whole range (a single bad day
        # shouldn't abort collection for the rest), but still fail the run
        # overall so ledger-collect.service's OnFailure= alert can fire.
        console.print(f"[red]{push_errors} push(es) failed during backfill[/red]")
        raise typer.Exit(1)


@ledger_app.command()
def show(
    days: int = typer.Option(14, "--days", help="How many recent days to render"),
):
    """Render a compact table of recent KPI history from the ledger DB."""
    from alfred.ledger import db as L_db

    conn = L_db.connect()
    all_dates = L_db.distinct_dates(conn)
    if not all_dates:
        console.print("[yellow]No ledger data yet — run `alfred ledger collect` or `backfill`.[/yellow]")
        conn.close()
        return

    recent = all_dates[-days:]
    rows = L_db.fetch_range(conn, recent[0], recent[-1])
    conn.close()

    # Pivot: metric (row) × date (col). Choose a stable, readable metric order.
    metric_order = [
        "git_commits", "sessions",
        "records.main", "records.neuroscience", "records.content",
        "topics_distilled", "api_cost_usd",
        "crm_users", "crm_status", "tribe_corpus_videos", "tribe_avg_score",
        "records.personal", "records.finance",
    ]
    domain_of: dict[str, str] = {}
    cells: dict[tuple[str, str], float] = {}
    seen_metrics: set[str] = set()
    for r in rows:
        cells[(r["metric"], r["date"])] = r["value"]
        domain_of[r["metric"]] = r["domain"]
        seen_metrics.add(r["metric"])

    ordered = [m for m in metric_order if m in seen_metrics]
    ordered += sorted(m for m in seen_metrics if m not in metric_order)

    table = Table(title=f"ledger — last {len(recent)} days", box=box.SIMPLE, padding=(0, 1))
    table.add_column("metric", style="cyan", no_wrap=True)
    for d in recent:
        table.add_column(d[5:], justify="right")  # MM-DD to keep it compact

    for m in ordered:
        cellvals = []
        for d in recent:
            v = cells.get((m, d))
            if v is None:
                cellvals.append("[dim]·[/dim]")
            elif isinstance(v, float) and v != int(v):
                cellvals.append(f"{v:.4g}")
            else:
                cellvals.append(str(int(v)))
        table.add_row(m, *cellvals)
    console.print(table)


def _print_push_result(status: str, detail: str) -> None:
    color = {"ok": "green", "dry": "yellow", "error": "red"}.get(status, "white")
    console.print(f"[{color}]push[{status}][/{color}] {detail}")
