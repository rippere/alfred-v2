#!/usr/bin/env python3
"""Log who reads the Alfred vault, and when.

Two modes:
  hook (default)  Claude Code PostToolUse hook. Reads the hook payload on stdin and,
                  if the tool call touched the vault (or an Alfred MCP tool), appends
                  one JSON line to data/vault_access.jsonl. Never blocks, never fails.
  --report [N]    Summarize the last N days (default 7) by caller and by record type.

Caller attribution, in order:
  subagent:<x>  payload carries agent_type (spawned via the Agent tool)
  agent:<x>     headless `claude -p "Run the <x> agent..."` (systemd timers)
  headless      other `claude -p` runs
  interactive   everything else (a human-driven session)
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

VAULT = "/mnt/external/obsidian-vault"
LOG = Path(__file__).resolve().parent.parent / "data" / "vault_access.jsonl"
AGENT_RE = re.compile(r"Run the ([\w-]+) agent")
WRITE_RE = re.compile(r"(>>?|\btee\b|\bcp\b|\bmv\b|\brm\b)[^|;&]*" + re.escape(VAULT))
PATH_RE = re.compile(re.escape(VAULT) + r"/[^\s'\"|;&)]*")


def _claude_argv() -> list[str]:
    """Walk up the process tree to the claude process and return its argv."""
    pid = os.getppid()
    for _ in range(6):
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            args = [a.decode(errors="ignore") for a in argv if a]
            if args and Path(args[0]).name == "claude":
                return args
            stat = Path(f"/proc/{pid}/stat").read_text()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return []


def _caller(payload: dict) -> str:
    if payload.get("agent_type"):
        return f"subagent:{payload['agent_type']}"
    argv = _claude_argv()
    if "-p" in argv or "--print" in argv:
        m = AGENT_RE.search(" ".join(argv))
        return f"agent:{m.group(1)}" if m else "headless"
    return "interactive"


def _classify(tool: str, inp: dict) -> tuple[str, list[str]] | None:
    """Return (op, vault paths) if this call touched the vault, else None."""
    if tool.startswith("mcp__alfred"):
        q = inp.get("query") or inp.get("rel_path") or inp.get("entity_name") or ""
        return "query", [str(q)[:200]]
    if tool == "Read":
        p = str(inp.get("file_path", ""))
        return ("read", [p]) if p.startswith(VAULT) else None
    if tool in ("Grep", "Glob"):
        p = str(inp.get("path", ""))
        return ("search", [p]) if p.startswith(VAULT) else None
    if tool == "Bash":
        cmd = str(inp.get("command", ""))
        if VAULT not in cmd:
            return None
        op = "write" if WRITE_RE.search(cmd) else "read"
        return op, sorted(set(PATH_RE.findall(cmd)))[:10]
    return None


def _record_type(path: str) -> str:
    rel = path[len(VAULT) + 1:] if path.startswith(VAULT + "/") else ""
    return rel.split("/", 1)[0] if rel else "(root)"


def hook() -> None:
    payload = json.load(sys.stdin)
    hit = _classify(payload.get("tool_name", ""), payload.get("tool_input") or {})
    if not hit:
        return
    op, paths = hit
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caller": _caller(payload),
        "tool": payload.get("tool_name"),
        "op": op,
        "paths": paths,
        "session": payload.get("session_id"),
        "agent_id": payload.get("agent_id"),
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def report(days: int) -> None:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    if LOG.exists():
        for line in LOG.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if datetime.fromisoformat(r["ts"]) >= since:
                rows.append(r)
    kind = Counter(r["caller"].split(":")[0] for r in rows)
    print(f"vault access, last {days}d: {len(rows)} calls, "
          f"{len({r['session'] for r in rows})} sessions")
    print("by caller kind:", dict(kind.most_common()))
    print("by caller:", dict(Counter(r["caller"] for r in rows).most_common(15)))
    print("by op:", dict(Counter(r["op"] for r in rows).most_common()))
    types = Counter(_record_type(p) for r in rows if r["op"] != "query" for p in r["paths"])
    print("by record type:", dict(types.most_common(12)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report(int(sys.argv[2]) if len(sys.argv) > 2 else 7)
    else:
        try:
            hook()
        except Exception:  # a logging hook must never break a tool call
            pass
