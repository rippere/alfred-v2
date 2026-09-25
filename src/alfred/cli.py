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
    cfg_table.add_row("embed model", f"{cfg.ollama_embed_model} at {cfg.embed_base_url}")
    llm = cfg.llm
    cfg_table.add_row("llm", f"{llm['model']} via {llm['api']} at {llm['base_url']}")
    if cfg.local_only:
        cfg_table.add_row("local only", "yes: chat and embeddings refused off loopback Ollama")
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
    # Orphan vectors: store rows minus state-tracked chunk_ids. Deliberately
    # computed from count_rows() (a manifest read, no scan) rather than the
    # real reap diff, which needs a full id scan plus a vault walk — status
    # must stay instant. That makes this an UPPER BOUND, not the reapable
    # count: most of the delta is files embedded but not yet state-saved
    # (measured 2026-08-07: 33,922 delta, of which 5 rows were genuinely
    # reapable). Labelled as such so nobody reads it as "rows to delete".
    try:
        # Deliberately NOT LanceDBStore(...): its constructor mkdirs, takes a
        # quarantine lock, create_table(mode="create")s when the table is
        # absent, and on a corruption-shaped open error quarantines and
        # RECREATES the table — then status would discard was_recreated, so the
        # runner's re-embed invalidation never fires. An operator running
        # `alfred status` to diagnose a broken store could empty it. A status
        # command must only read.
        import lancedb as _lancedb
        _db = _lancedb.connect(getattr(cfg, "lancedb_uri", str(cfg.data_dir / "lancedb")))
        if cfg.milvus_collection not in _db.table_names():
            raise FileNotFoundError(f"table {cfg.milvus_collection!r} not present")
        _delta = _db.open_table(cfg.milvus_collection).count_rows() - store.chunk_count()
        state_table.add_row(
            "orphan vectors",
            f"{max(0, _delta)} [dim]upper bound — `alfred reap` for the real count[/dim]",
        )
    except Exception as e:  # vector store absent or unopenable — not fatal for status
        from alfred.core.failures import record_failure
        record_failure("cli.status_orphan_count_failed", error=e)
        state_table.add_row("orphan vectors", "[yellow]unavailable[/yellow]")
    # Swallowed failures — the whole point of error_counts is that this row
    # reads a real number instead of nothing at all when handlers have been
    # quietly eating errors. Red when non-zero so it can't be skimmed past.
    _errors = store.state.error_counts
    _total_errors = sum(_errors.values())
    state_table.add_row(
        "swallowed errors",
        "0" if not _total_errors else f"[red]{_total_errors}[/red]",
    )
    console.print("[bold]State[/bold]")
    console.print(state_table)
    if _errors:
        err_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        err_table.add_column(style="dim")
        err_table.add_column(justify="right")
        for _key, _count in sorted(_errors.items(), key=lambda kv: -kv[1]):
            err_table.add_row(_key, str(_count))
        console.print("[bold]Swallowed errors[/bold] [dim](cumulative)[/dim]")
        console.print(err_table)

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
    timeout: float = typer.Option(120.0, "--timeout", help="Seconds to wait for exit"),
):
    """Stop the daemons for this config and WAIT for the process to exit.

    Exits non-zero if the process is still alive at the deadline, so a caller
    doing surgery on the store can branch on it. The old implementation sent
    one SIGTERM, printed success and returned immediately — callers read that
    as "quiesced" while the surveyor kept committing vectors. It also unlinked
    the pidfile before the process died, defeating `up`'s duplicate-start
    guard for the whole (minutes-long) shutdown.

    SIGKILL is deliberately NOT sent on timeout: killing mid-commit is what
    leaves the zero-byte manifests that crash-loop the next open (see the
    comments in store/lancedb_store.py and store/graph.py). Escalating is the
    operator's call, made knowingly.
    """
    import os
    import signal
    import time

    cfg = _load(config)
    pid_path = cfg.data_dir / "alfred.pid"

    if not pid_path.exists():
        console.print("[yellow]No alfred.pid found — not running?[/yellow]")
        raise typer.Exit(1)

    pid_str = pid_path.read_text().strip()
    try:
        pid = int(pid_str.split(":")[0])
    except ValueError:
        console.print(f"[red]Invalid PID in {pid_path}[/red] — content: {pid_str!r}")
        pid_path.unlink(missing_ok=True)
        raise typer.Exit(1)

    def _alive() -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True      # exists, owned by someone else
        return True

    if not _alive():
        console.print(f"[yellow]Process {pid} not running. Removing stale PID file.[/yellow]")
        pid_path.unlink(missing_ok=True)
        return

    os.kill(pid, signal.SIGTERM)
    console.print(f"Sent SIGTERM to Alfred (PID {pid}) — waiting up to {timeout:g}s for exit…")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive():
            # Only now is the pidfile meaningless. Unlinking earlier would let a
            # concurrent `alfred up` start a second instance against the same store.
            pid_path.unlink(missing_ok=True)
            console.print(f"[green]Alfred (PID {pid}) stopped[/green]")
            return
        time.sleep(0.5)

    console.print(
        f"[red]PID {pid} still alive after {timeout:g}s — NOT quiesced.[/red]\n"
        "[dim]The surveyor only checks for shutdown between files, so a large embed\n"
        "backlog delays exit. Do NOT assume the store is idle. Re-run with a longer\n"
        "--timeout, and escalate manually only once the log shows no recent\n"
        "'surveyor.embedded' lines — a mid-commit SIGKILL can corrupt the table.[/dim]"
    )
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
        except OSError as e:
            # A hit whose file moved still deserves its path in the output —
            # but a run of these means the index has drifted from the vault,
            # which is worth being able to count.
            from alfred.core.failures import record_failure
            record_failure("cli.preview_read_failed", error=e, path=h.rel_path)
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
def forget(
    config: Path = typer.Option(_DEFAULT_CONFIG, "--config", "-c"),
    apply: bool = typer.Option(
        False, "--apply", help="Actually evict. Without this, nothing is changed."
    ),
    limit: int = typer.Option(25, "--limit", "-n", help="Rows to show in the preview"),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Show (or apply) the Ebbinghaus retention sweep — evict vectors for files
    whose memory has decayed.

    Dry run by default. The source notes are never touched: only the vector
    embeddings are evicted, and editing a file re-embeds it normally, so this
    is reclaimable storage rather than lost content.
    """
    import asyncio
    import json as _json
    from datetime import datetime, timezone
    from rich import box
    from rich.table import Table

    from alfred.daemons.janitor import JanitorDaemon
    from alfred.store.state import StateStore

    cfg = _load(config)
    store = StateStore(cfg.state_path, cfg=cfg)
    store.load()

    from alfred.store.lancedb_store import LanceDBStore
    vector_store = LanceDBStore(
        uri=getattr(cfg, "lancedb_uri", str(cfg.data_dir / "lancedb")),
        collection=cfg.milvus_collection,
        dims=cfg.embed_dims,
    )
    janitor = JanitorDaemon(cfg, store, asyncio.Queue(), store=vector_store)

    candidates = janitor.forget_candidates()
    total_chunks = sum(c["chunks"] for c in candidates)

    if json_out:
        console.print_json(_json.dumps({
            "candidates": len(candidates),
            "chunks": total_chunks,
            "applied": False,
            "rows": candidates[:limit],
        }))
        if not apply:
            return

    if not json_out:
        console.print(
            f"[bold]{len(candidates)}[/bold] file(s) below the retention threshold "
            f"([dim]R < {cfg.janitor_forget_retrievability}, embedded ≥ "
            f"{cfg.janitor_forget_min_age_days}d ago[/dim]) — "
            f"[bold]{total_chunks}[/bold] chunk(s) of vectors"
        )
        if candidates:
            t = Table(box=box.SIMPLE, padding=(0, 2))
            t.add_column("file", style="dim", overflow="fold")
            t.add_column("R", justify="right")
            t.add_column("age (d)", justify="right")
            t.add_column("reads", justify="right")
            t.add_column("chunks", justify="right")
            for c in candidates[:limit]:
                t.add_row(
                    c["rel_path"], f"{c['retrievability']:.4f}", str(c["age_days"]),
                    str(c["access_count"]), str(c["chunks"]),
                )
            console.print(t)
            if len(candidates) > limit:
                console.print(f"[dim]… and {len(candidates) - limit} more[/dim]")

    if not apply:
        if not json_out:
            console.print("\n[dim]Dry run — nothing changed. Re-run with --apply to evict.[/dim]")
        return

    now_iso = datetime.now(timezone.utc).isoformat()
    cap = int(getattr(cfg, "janitor_forget_max_per_sweep", 500))
    selected = candidates[:cap]
    evicted = sum(1 for c in selected if janitor._forget_file(c["rel_path"], now_iso))
    store.save()
    console.print(
        f"[green]evicted[/green] {evicted} file(s), "
        f"{sum(c['chunks'] for c in selected)} chunk(s)"
        + (f" — {len(candidates) - len(selected)} deferred past the per-sweep cap"
           if len(candidates) > len(selected) else "")
    )


@app.command()
def reap(
    config: Path = typer.Option(_DEFAULT_CONFIG, "--config", "-c"),
    apply: bool = typer.Option(
        False, "--apply", help="Actually delete. Without this, nothing is changed."
    ),
    limit: int = typer.Option(25, "--limit", "-n", help="Rows to show in the preview"),
    max_rows: Optional[int] = typer.Option(
        None, "--max-rows", help="Per-sweep row cap (default: janitor.reap_max_rows_per_sweep)"
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Show (or apply) the orphan reap — delete vector rows whose note is gone.

    A row is reaped only if its rel_path is absent from state.json AND its
    .md is absent from the vault. Both, never either: state.json lags the
    vault by up to 25 files (the surveyor's save interval), and on 2026-08-07
    that lag covered 2,892 files / 45,352 rows whose notes were all still
    present. A state-only reaper would have deleted 29% of the index.

    Dry run by default.
    """
    import asyncio
    import json as _json
    from rich import box
    from rich.table import Table

    from alfred.daemons.janitor import JanitorDaemon
    from alfred.store.state import StateStore

    cfg = _load(config)
    store = StateStore(cfg.state_path, cfg=cfg)
    store.load()

    from alfred.store.lancedb_store import LanceDBStore
    vector_store = LanceDBStore(
        uri=getattr(cfg, "lancedb_uri", str(cfg.data_dir / "lancedb")),
        collection=cfg.milvus_collection,
        dims=cfg.embed_dims,
    )
    janitor = JanitorDaemon(cfg, store, asyncio.Queue(), store=vector_store)

    plan = janitor.reap_plan(max_rows=max_rows)

    if plan.aborted:
        if json_out:
            console.print_json(_json.dumps({"aborted": plan.aborted, "applied": False}))
        else:
            console.print(f"[red]aborted:[/red] {plan.aborted}")
        raise typer.Exit(1)

    if json_out:
        console.print_json(_json.dumps({
            "store_rows": plan.store_rows,
            "store_paths": plan.store_paths,
            "reapable_paths": plan.reapable_paths,
            "reapable_rows": plan.reapable_rows,
            "deferred": len(plan.deferred),
            "malformed": plan.malformed_count,
            "untracked_but_present": plan.untracked_but_present,
            "untracked_but_present_rows": plan.untracked_but_present_rows,
            "applied": False,
            "rows": [{"rel_path": p, "chunks": n} for p, n in plan.orphans[:limit]],
        }))
        if not apply:
            return

    if not json_out:
        console.print(
            f"[bold]{plan.reapable_paths}[/bold] orphaned file(s) — "
            f"[bold]{plan.reapable_rows}[/bold] row(s) of "
            f"{plan.store_rows} in the store"
        )
        if plan.orphans:
            t = Table(box=box.SIMPLE, padding=(0, 2))
            t.add_column("file", style="dim", overflow="fold")
            t.add_column("chunks", justify="right")
            for p, n in plan.orphans[:limit]:
                t.add_row(p, str(n))
            console.print(t)
            if len(plan.orphans) > limit:
                console.print(f"[dim]… and {len(plan.orphans) - limit} more[/dim]")
        # Never hide this: it is the number the safe definition is protecting.
        console.print(
            f"[dim]retained: {plan.untracked_but_present} file(s) / "
            f"{plan.untracked_but_present_rows} row(s) absent from state.json but "
            f"still present in the vault (state lags by up to 25 files)[/dim]"
        )
        if plan.deferred:
            console.print(
                f"[yellow]deferred:[/yellow] {len(plan.deferred)} file(s) / "
                f"{sum(n for _, n in plan.deferred)} row(s) past the per-sweep cap "
                f"— re-run, or raise --max-rows"
            )
        if plan.malformed_count:
            console.print(
                f"[yellow]{plan.malformed_count}[/yellow] id(s) did not parse as "
                f"'<rel_path>::chunk_NN' and were left alone, e.g. "
                f"{plan.malformed[:3]}"
            )

    if not apply:
        if not json_out:
            console.print("\n[dim]Dry run — nothing changed. Re-run with --apply to delete.[/dim]")
        return

    from alfred.store.reaper import execute_plan
    deleted = execute_plan(
        vector_store,
        store.state,
        cfg.vault_path,
        plan,
        batch_size=int(getattr(cfg, "janitor_reap_scan_batch_size", 4096)),
        delete_batch=int(getattr(cfg, "janitor_reap_delete_batch", 500)),
    )
    # record_failure() only reaches state.error_counts via StateStore.save() ->
    # drain_failures(). Without this save, reap.vector_delete_failed,
    # reap.malformed_chunk_id and lancedb.delete_ids_quote_in_id are logged and
    # then lost, and `alfred status` keeps reporting "swallowed errors 0" while
    # deletes are silently failing. `alfred forget` already saves here.
    store.save()
    console.print(f"[green]reaped[/green] {deleted} row(s)")


@app.command("graph-repair")
def graph_repair(
    config: Path = typer.Option(_DEFAULT_CONFIG, "--config", "-c"),
    apply: bool = typer.Option(
        False, "--apply", help="Actually rewrite graph.pkl. Without this, nothing is changed."
    ),
    limit: int = typer.Option(15, "--limit", "-n", help="Sample rows to show in the preview"),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
):
    """Merge legacy raw-link-text nodes in graph.pkl into their rel_path node.

    A graph built before wikilink targets were normalized holds each linked
    file twice — once as `note/foo.md` (the source side) and once as
    `note/foo` (the link-text side). The link-text half collects every inbound
    edge but has no outbound edges, so it is a dead end that halves graph-hop
    recall. This merges the two halves, carrying edges across and summing
    weights.

    Dry run by default. Idempotent — a second run reports 0 to merge.
    """
    import json as _json
    import shutil

    from alfred.store.graph import GraphStore, normalize_node

    cfg = _load(config)
    graph = GraphStore(cfg.graph_path)
    if not graph.load():
        console.print(f"[yellow]no graph at[/yellow] {cfg.graph_path} — nothing to repair")
        raise typer.Exit(0)

    g = graph._graph()
    before_nodes, before_edges = g.number_of_nodes(), g.number_of_edges()
    candidates = [
        n for n in g.nodes()
        if isinstance(n, str) and normalize_node(n) not in ("", n)
    ]
    # A candidate that already has a normalized twin is a genuine split file;
    # one without is a link to a note that does not exist (yet).
    split = [n for n in candidates if g.has_node(normalize_node(n))]

    if json_out:
        console.print_json(_json.dumps({
            "nodes": before_nodes, "edges": before_edges,
            "unnormalized": len(candidates), "split_files": len(split),
            "applied": False, "sample": candidates[:limit],
        }))
    else:
        console.print(
            f"graph [bold]{before_nodes}[/bold] nodes / [bold]{before_edges}[/bold] edges — "
            f"[bold]{len(candidates)}[/bold] un-normalized node(s), of which "
            f"[bold]{len(split)}[/bold] duplicate a file that already exists as a .md node"
        )
        for n in candidates[:limit]:
            marker = "[red]split[/red]" if g.has_node(normalize_node(n)) else "[dim]dangling[/dim]"
            console.print(f"  {marker}  {n}  ->  {normalize_node(n)}")
        if len(candidates) > limit:
            console.print(f"[dim]  … and {len(candidates) - limit} more[/dim]")

    if not apply:
        if not json_out:
            console.print("\n[dim]Dry run — nothing changed. Re-run with --apply to merge.[/dim]")
        return

    if not candidates:
        console.print("[green]nothing to merge[/green]")
        return

    # graph.pkl is the only copy of the wikilink topology and a full rebuild
    # costs a vault-wide re-read, so keep a restorable copy before rewriting.
    backup = cfg.graph_path.with_name(cfg.graph_path.name + ".pre-repair")
    shutil.copy2(cfg.graph_path, backup)

    with graph.transaction():
        graph.load()                      # re-read under the lock, not the preview copy
        merged = graph.merge_link_text_nodes()
        graph.save()

    g = graph._graph()
    console.print(
        f"[green]merged[/green] {merged} node(s) — "
        f"{before_nodes} -> {g.number_of_nodes()} nodes, "
        f"{before_edges} -> {g.number_of_edges()} edges\n"
        f"[dim]backup: {backup}[/dim]"
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

# Deal/contact brief enrichment: `alfred bridge enrich [--push]`.
from alfred.bridge.cli import bridge_app  # noqa: E402
app.add_typer(bridge_app, name="bridge")
