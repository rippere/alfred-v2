from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

app = typer.Typer(name="alfred", help="Personal agentic knowledge infrastructure.", add_completion=False)
console = Console()

# Resolve relative to repo root regardless of CWD
_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "config.yaml"


def _load(config_path: Path):
    from alfred.config import AlfredConfig
    return AlfredConfig.load(config_path)


def _load_state(cfg):
    from alfred.store.state import StateStore
    store = StateStore(cfg.state_path)
    store.load()
    return store


@app.command()
def status(
    config: Path = typer.Option(_DEFAULT_CONFIG, "--config", "-c", help="Path to config.yaml"),
):
    """Show system status: config, state counts, and daemon readiness."""
    try:
        cfg = _load(config)
    except FileNotFoundError as e:
        console.print(f"[red]Config not found:[/red] {e}")
        raise typer.Exit(1)

    store = _load_state(cfg)

    console.print("\n[bold cyan]Alfred v2[/bold cyan]\n")

    # Config table
    cfg_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    cfg_table.add_column(style="dim")
    cfg_table.add_column()
    cfg_table.add_row("vault", str(cfg.vault_path))
    cfg_table.add_row("data", str(cfg.data_dir))
    cfg_table.add_row("embed model", cfg.ollama_embed_model)
    cfg_table.add_row("llm model", cfg.ollama_llm_model)
    cfg_table.add_row("milvus", cfg.milvus_uri)
    console.print("[bold]Config[/bold]")
    console.print(cfg_table)

    # State table
    state_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    state_table.add_column(style="dim")
    state_table.add_column()
    state_table.add_row("files tracked", str(store.file_count()))
    state_table.add_row("files embedded", str(store.embedded_count()))
    state_table.add_row("chunks", str(store.chunk_count()))
    state_table.add_row("clusters", str(store.cluster_count()))
    state_table.add_row("wiki pages", str(store.wiki_page_count()))
    state_table.add_row("curator processed", str(len(store.state.curator_processed)))
    state_table.add_row("distiller runs", str(len(store.state.distiller_runs)))
    state_table.add_row("last run", store.state.last_run or "never")
    console.print("[bold]State[/bold]")
    console.print(state_table)

    # Vault check
    vault_ok = cfg.vault_path.exists()
    state_ok = cfg.state_path.exists()
    milvus_ok = Path(cfg.milvus_uri).exists()

    checks = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    checks.add_column(style="dim")
    checks.add_column()
    checks.add_row("vault path", "[green]ok[/green]" if vault_ok else "[red]missing[/red]")
    checks.add_row("state.json", "[green]ok[/green]" if state_ok else "[yellow]not yet created[/yellow]")
    checks.add_row("milvus.db", "[green]ok[/green]" if milvus_ok else "[yellow]not yet created[/yellow]")
    console.print("[bold]Checks[/bold]")
    console.print(checks)


@app.command()
def up(
    config: Path = typer.Option(_DEFAULT_CONFIG, "--config", "-c"),
    only: Optional[str] = typer.Option(None, "--only", help="Comma-separated daemon names to start"),
    daemon: bool = typer.Option(False, "--daemon", "-d", help="Fork to background"),
    no_pid_check: bool = typer.Option(False, "--no-pid-check", hidden=True),
):
    """Start all daemons (or a named subset with --only surveyor,janitor)."""
    import asyncio
    import sys

    cfg = _load(config)
    pid_path = cfg.data_dir / "alfred.pid"

    if not no_pid_check and pid_path.exists():
        existing_pid = pid_path.read_text().strip()
        try:
            pid_int = int(existing_pid)
            import os as _os
            _os.kill(pid_int, 0)  # raises if process is dead
            console.print(f"[yellow]Alfred already running (PID {existing_pid}). Use 'alfred down' first.[/yellow]")
            raise typer.Exit(1)
        except ProcessLookupError:
            console.print(f"[dim]Removing stale PID file (PID {existing_pid} is dead).[/dim]")
            pid_path.unlink(missing_ok=True)
        except PermissionError:
            console.print(f"[yellow]Alfred already running (PID {existing_pid}). Use 'alfred down' first.[/yellow]")
            raise typer.Exit(1)
        except ValueError:
            pid_path.unlink(missing_ok=True)

    if daemon:
        # Fork to background — redirect stdout/stderr to log file so process survives terminal close
        import subprocess
        cfg2 = _load(config)
        log_path = cfg2.data_dir / "alfred.log"
        cfg2.data_dir.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, "-m", "alfred.cli", "up", "--config", str(config.resolve()), "--no-pid-check"]
        if only:
            args += ["--only", only]
        with open(log_path, "a") as logf:
            proc = subprocess.Popen(
                args,
                start_new_session=True,
                stdout=logf,
                stderr=logf,
                cwd=str(config.resolve().parent),
            )
        console.print(f"[green]Alfred started in background (PID {proc.pid})[/green]")
        console.print(f"[dim]Logs: {log_path}[/dim]")
        return

    # Write PID
    import os
    pid_path.write_text(str(os.getpid()))
    console.print(f"[green]Alfred starting...[/green] (PID {os.getpid()})")

    selected = set(only.split(",")) if only else None

    from alfred.runner import run_daemons
    try:
        asyncio.run(run_daemons(cfg, only=selected))
    finally:
        pid_path.unlink(missing_ok=True)


@app.command()
def down(
    config: Path = typer.Option(_DEFAULT_CONFIG, "--config", "-c"),
):
    """Stop all running daemons."""
    import os
    import signal

    cfg = _load(config)
    pid_path = cfg.data_dir / "alfred.pid"

    if not pid_path.exists():
        console.print("[yellow]No alfred.pid found — not running?[/yellow]")
        raise typer.Exit(1)

    pid_str = pid_path.read_text().strip()
    try:
        pid = int(pid_str)
        os.kill(pid, signal.SIGTERM)
        console.print(f"[green]Sent SIGTERM to Alfred (PID {pid})[/green]")
        pid_path.unlink(missing_ok=True)
    except ProcessLookupError:
        console.print(f"[yellow]Process {pid_str} not found. Removing stale PID file.[/yellow]")
        pid_path.unlink(missing_ok=True)
    except ValueError:
        console.print(f"[red]Invalid PID in {pid_path}[/red]")
        raise typer.Exit(1)


@app.command()
def query(
    text: str = typer.Argument(..., help="Query string"),
    config: Path = typer.Option(_DEFAULT_CONFIG, "--config", "-c"),
    top_k: int = typer.Option(8, "--top-k", "-k"),
    synthesis: bool = typer.Option(False, "--synthesis", "-s", help="Include LLM synthesis"),
    no_hopfield: bool = typer.Option(False, "--no-hopfield", help="Skip Hopfield refinement"),
    no_graph: bool = typer.Option(False, "--no-graph", help="Skip graph spreading activation"),
    include_inbox: bool = typer.Option(False, "--include-inbox", help="Include inbox/ in results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-step timing"),
):
    """Query the vault with hybrid retrieval + FlashRank reranking."""
    from rich import box
    from rich.rule import Rule
    from rich.table import Table

    cfg = _load(config)

    from alfred.query.engine import QueryEngine, QueryOptions
    engine = QueryEngine(cfg)
    opts = QueryOptions(
        top_k=top_k,
        use_hopfield=not no_hopfield,
        use_graph=not no_graph,
        include_synthesis=synthesis,
        include_inbox=include_inbox,
    )

    console.print()
    console.print(Rule(f'[bold]alfred query[/bold]', style="dim"))
    console.print(f'  [italic]"{text}"[/italic]\n')

    try:
        result = engine.query(text, opts)
    except Exception as e:
        console.print(f"[red]Query failed:[/red] {e}")
        raise typer.Exit(1)

    # ── Wiki hit ───────────────────────────────────────────────────────────────
    if result.wiki_hit:
        console.print(f"[bold cyan]Wiki:[/bold cyan] {result.wiki_hit.rel_path}  [dim](fast-path)[/dim]\n")

    # ── Retrieved chunks ───────────────────────────────────────────────────────
    if result.hits:
        table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        table.add_column("#", style="dim", width=3)
        table.add_column("Score", width=10)
        table.add_column("Type", width=12)
        table.add_column("Source")
        for i, hit in enumerate(result.hits, 1):
            score = hit.rerank_score if hit.rerank_score else hit.score
            color = "green" if score >= 0.70 else "yellow" if score >= 0.40 else "dim"
            table.add_row(
                str(i),
                f"[{color}]{score:.3f}[/{color}]",
                hit.record_type or "—",
                hit.rel_path,
            )
        console.print(table)

    # ── Answer ─────────────────────────────────────────────────────────────────
    if result.answer:
        console.print(Rule("Answer", style="dim"))
        console.print(result.answer)
        console.print(f"\n[dim]via {result.synthesis_backend} ({result.synthesis_model})[/dim]")
    elif synthesis:
        console.print("[yellow]No context to synthesize from.[/yellow]")

    # ── Timing ────────────────────────────────────────────────────────────────
    if verbose and result.elapsed:
        console.print()
        t_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        t_table.add_column(style="dim")
        t_table.add_column()
        for step, ms in result.elapsed.items():
            t_table.add_row(step, f"{ms*1000:.1f}ms")
        console.print("[dim]Timing[/dim]")
        console.print(t_table)


@app.command()
def mcp(
    config: Path = typer.Option(_DEFAULT_CONFIG, "--config", "-c"),
):
    """Start the MCP stdio server (for Claude Code integration)."""
    from alfred.mcp.server import run_server
    run_server(config)
