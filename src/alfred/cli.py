from __future__ import annotations

from pathlib import Path
from typing import Optional

import json

import typer
from rich.console import Console
from rich.table import Table
from rich import box

app = typer.Typer(name="alfred", help="Personal agentic knowledge infrastructure.", add_completion=False)
console = Console()

# Resolve relative to repo root regardless of CWD
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "config.yaml"


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
            _parts = existing_pid.split(":")
            pid_int = int(_parts[0])
            _expected_starttime = _parts[1] if len(_parts) > 1 else None
            import os as _os
            _os.kill(pid_int, 0)  # raises ProcessLookupError if process is dead
            # Verify this is the same process that wrote the PID file, not a recycled PID.
            # /proc/{pid}/stat field 22 (0-indexed 21) is the process start time in jiffies
            # since boot — set at fork, never changes, and is unique per boot cycle.
            # If it doesn't match what we recorded, a different process owns this PID now.
            _stale = False
            try:
                _stat_fields = Path(f"/proc/{pid_int}/stat").read_text().split()
                _actual_starttime = _stat_fields[21]
                if _expected_starttime and _actual_starttime != _expected_starttime:
                    _stale = True
                elif not _expected_starttime:
                    # Legacy PID file (no starttime) — fall back to cmdline heuristic
                    cmdline = Path(f"/proc/{pid_int}/cmdline").read_bytes().replace(b"\x00", b" ")
                    config_bytes = str(config.resolve()).encode()
                    if b"alfred" not in cmdline.lower() or config_bytes not in cmdline:
                        _stale = True
            except OSError:
                pass  # /proc unavailable — conservatively assume Alfred is running
            if _stale:
                console.print(f"[dim]Removing stale PID file (PID {pid_int} start-time mismatch — recycled).[/dim]")
                pid_path.unlink(missing_ok=True)
            else:
                console.print(f"[yellow]Alfred already running (PID {pid_int}). Use 'alfred down' first.[/yellow]")
                raise typer.Exit(1)
        except ProcessLookupError:
            console.print(f"[dim]Removing stale PID file (PID {existing_pid.split(':')[0]} is dead).[/dim]")
            pid_path.unlink(missing_ok=True)
        except PermissionError:
            console.print(f"[yellow]Alfred already running (PID {existing_pid.split(':')[0]}). Use 'alfred down' first.[/yellow]")
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

    # Write PID + process start time — start time is set at fork and never changes,
    # making this file immune to PID recycling across reboots or rapid restarts.
    import os
    _pid = os.getpid()
    try:
        _stat = Path(f"/proc/{_pid}/stat").read_text().split()
        pid_path.write_text(f"{_pid}:{_stat[21]}")
    except OSError:
        pid_path.write_text(str(_pid))
    console.print(f"[green]Alfred starting...[/green] (PID {_pid})")

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
        pid = int(pid_str.split(":")[0])
        os.kill(pid, signal.SIGTERM)
        console.print(f"[green]Sent SIGTERM to Alfred (PID {pid})[/green]")
        pid_path.unlink(missing_ok=True)
    except ProcessLookupError:
        console.print(f"[yellow]Process {pid_str} not found. Removing stale PID file.[/yellow]")
        pid_path.unlink(missing_ok=True)
    except ValueError:
        console.print(f"[red]Invalid PID in {pid_path}[/red]")
        raise typer.Exit(1)


def _query_result_as_dict(cfg, result) -> dict:
    """Shape a QueryResult for --json.

    `preview` is read back off disk rather than sliced out of result.context,
    because context is a single assembled blob with wiki blocks and separators
    mixed in — attributing a slice of it to a specific hit would be guesswork.
    """
    hits = []
    for h in result.hits:
        preview = ""
        try:
            raw = (cfg.vault_path / h.rel_path).read_text(encoding="utf-8", errors="replace")
            if raw.startswith("---"):
                end = raw.find("\n---", 3)
                if end != -1:
                    raw = raw[end + 4:]
            preview = raw.strip()[:400]
        except OSError:
            pass    # a hit whose file moved still deserves its path in the output
        hits.append({
            "rel_path": h.rel_path,
            "name": h.name,
            "record_type": h.record_type,
            "score": h.score,
            "preview": preview,
        })
    return {
        "query": result.query,
        "hits": hits,
        "answer": result.answer,
        "synthesis_backend": result.synthesis_backend,
        "synthesis_model": result.synthesis_model,
    }


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
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON on stdout"),
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

    # --json must keep stdout parseable, so none of the rich chrome below runs.
    if not json_out:
        console.print()
        console.print(Rule('[bold]alfred query[/bold]', style="dim"))
        console.print(f'  [italic]"{text}"[/italic]\n')

    try:
        result = engine.query(text, opts)
    except Exception as e:
        if json_out:
            # Structured failure, so a caller can tell "query broke" from
            # "no hits" instead of parsing an empty result as an answer.
            print(json.dumps({"query": text, "error": str(e), "hits": []}))
            raise typer.Exit(1)
        console.print(f"[red]Query failed:[/red] {e}")
        raise typer.Exit(1)

    if json_out:
        print(json.dumps(_query_result_as_dict(cfg, result), indent=2))
        return

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


@app.command("create-vault")
def create_vault(
    name: str = typer.Argument(..., help="Vault name, e.g. 'research' — becomes config-<name>.yaml, data-<name>/, alfred@<name>"),
    vault_path: Optional[Path] = typer.Option(
        None, "--vault-path",
        help="Markdown/Obsidian vault directory (default: /mnt/external/vault-<name>; created if missing)",
    ),
    root: Path = typer.Option(_REPO_ROOT, "--root", hidden=True, help="Repo root override (tests only)"),
):
    """Scaffold a new vault: config-<name>.yaml, data-<name>/, config-meta.yaml entry.

    config-meta.yaml's vaults: list is the single source of truth for the fleet
    roster — the meta server, watchdog, and game-guard all derive from it (via
    scripts/alfred-roster.sh), so registering here is what makes the new vault
    queryable, monitored, and game-paused. Refuses to touch anything if the
    vault already exists in any form (idempotent); all file writes are atomic
    (tmp + rename) and rolled back together on failure.
    """
    import os
    import re

    import yaml

    # ── validate the name ─────────────────────────────────────────────────────
    # Must be safe as a systemd instance name, a filename suffix, and a YAML
    # scalar. Reserved names collide with non-vault config files.
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        console.print(f"[red]Invalid vault name '{name}'[/red] — use lowercase letters, digits, '-', '_' (must start alphanumeric).")
        raise typer.Exit(1)
    if name in {"base", "meta"}:
        console.print(f"[red]'{name}' is reserved[/red] (config-{name}.yaml is not a vault config).")
        raise typer.Exit(1)

    config_path = (root / f"config-{name}.yaml").resolve()
    data_dir = (root / f"data-{name}").resolve()
    meta_path = (root / "config-meta.yaml").resolve()

    if not meta_path.is_file():
        console.print(f"[red]config-meta.yaml not found at {meta_path}[/red] — wrong --root?")
        raise typer.Exit(1)

    try:
        meta_text = meta_path.read_text()
        meta = yaml.safe_load(meta_text) or {}
    except OSError as e:
        console.print(f"[red]Cannot read config-meta.yaml:[/red] {e}")
        raise typer.Exit(1)
    except yaml.YAMLError as e:
        console.print(f"[red]config-meta.yaml is not valid YAML:[/red] {e}")
        raise typer.Exit(1)
    vaults = meta.get("vaults") or []

    # ── refuse if the vault exists in any form (idempotency) ─────────────────
    conflicts = []
    if config_path.exists():
        conflicts.append(f"config file already exists: {config_path}")
    if data_dir.exists():
        conflicts.append(f"data dir already exists: {data_dir}")
    for v in vaults:
        if v.get("name") == name or Path(v.get("config", "")).name == config_path.name:
            conflicts.append(f"already registered in config-meta.yaml as '{v.get('name')}' -> {v.get('config')}")
    if conflicts:
        console.print(f"[yellow]Vault '{name}' already exists — refusing to scaffold:[/yellow]")
        for c in conflicts:
            console.print(f"  - {c}")
        raise typer.Exit(1)

    vp = (vault_path or Path(f"/mnt/external/vault-{name}")).expanduser()

    # ── stage the new config-meta.yaml text (edit textually to keep the file's
    # formatting/comments; append the entry to the end of the vaults: block) ──
    lines = meta_text.splitlines()
    try:
        v_idx = next(i for i, ln in enumerate(lines) if re.fullmatch(r"vaults:\s*", ln))
    except StopIteration:
        console.print("[red]config-meta.yaml has no 'vaults:' block — cannot register.[/red]")
        raise typer.Exit(1)
    end = v_idx + 1
    while end < len(lines) and (lines[end].startswith(" ") or not lines[end].strip()):
        end += 1
    while end > v_idx + 1 and not lines[end - 1].strip():
        end -= 1  # attach before trailing blank lines, inside the list
    new_meta_text = "\n".join(
        lines[:end] + [f"  - name: {name}", f"    config: {config_path}"] + lines[end:]
    ) + "\n"
    parsed = yaml.safe_load(new_meta_text)
    if not any(v.get("name") == name for v in parsed.get("vaults", [])):
        console.print("[red]Internal error: staged config-meta.yaml edit did not round-trip — aborting, nothing written.[/red]")
        raise typer.Exit(1)

    config_body = (
        f"# {name.capitalize()} vault. Fleet-wide defaults live in config-base.yaml —\n"
        f"# keys here override the base (deep merge, this file wins).\n"
        f"vault:\n"
        f"  path: {vp}\n"
        f"\n"
        f"data_dir: ./data-{name}\n"
    )

    # ── commit: each write is atomic (tmp + rename); registration in
    # config-meta.yaml goes LAST so a partial failure never leaves the roster
    # pointing at files that don't exist. Roll back our own writes on failure. ─
    created_data_dir = created_config = created_vault_dir = False
    try:
        if not vp.exists():
            vp.mkdir(parents=True)
            created_vault_dir = True
        data_dir.mkdir()
        created_data_dir = True

        tmp_cfg = config_path.with_suffix(".yaml.tmp")
        tmp_cfg.write_text(config_body)
        os.replace(tmp_cfg, config_path)
        created_config = True

        tmp_meta = meta_path.with_suffix(".yaml.tmp")
        tmp_meta.write_text(new_meta_text)
        os.replace(tmp_meta, meta_path)
    except OSError as e:
        if created_config:
            config_path.unlink(missing_ok=True)
        if created_data_dir:
            data_dir.rmdir()
        if created_vault_dir:
            vp.rmdir()
        console.print(f"[red]Scaffold failed, rolled back:[/red] {e}")
        raise typer.Exit(1)

    # sanity: the scaffolded config must load through AlfredConfig
    try:
        _load(config_path)
    except Exception as e:  # pragma: no cover — defensive
        console.print(f"[yellow]Warning: scaffolded config did not load cleanly:[/yellow] {e}")

    console.print(f"\n[green]Vault '{name}' scaffolded.[/green]\n")
    console.print(f"  config    {config_path}")
    console.print(f"  data dir  {data_dir}")
    console.print(f"  vault     {vp}" + ("  [dim](created)[/dim]" if created_vault_dir else ""))
    console.print(f"  roster    registered in {meta_path}")
    console.print("            (meta server, watchdog, and game-guard pick it up automatically")
    console.print("             via scripts/alfred-roster.sh — no further edits needed)\n")
    console.print("[bold]Start the daemon fleet for it:[/bold]")
    console.print(f"  systemctl --user enable --now alfred@{name}.service")
    console.print(
        "[dim]  (requires the alfred@.service template unit — deploy/systemd/alfred@.service —\n"
        "   installed in ~/.config/systemd/user/)[/dim]"
    )


@app.command()
def mcp(
    config: Path = typer.Option(_DEFAULT_CONFIG, "--config", "-c"),
):
    """Start the MCP stdio server (for Claude Code integration)."""
    from alfred.mcp.server import run_server
    run_server(config)


# Daily life-ledger KPI snapshots: `alfred ledger collect|backfill|show`.
from alfred.ledger.cli import ledger_app  # noqa: E402
app.add_typer(ledger_app, name="ledger")
