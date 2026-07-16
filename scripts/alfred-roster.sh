#!/usr/bin/env bash
# alfred-roster.sh — single source of truth for the vault/service roster
# (audit 2026-07-13, structural improvement #3).
#
# Derives systemd unit names (and PID-file paths) from config-meta.yaml's
# vaults: list, so the watchdog, game-guard, and any future consumer stop
# carrying independently-drifting hardcoded copies of the fleet roster.
#
# Usage:
#   alfred-roster.sh units      one unit name per line, no ".service" suffix
#                               (e.g. alfred, alfred-personal or alfred@personal)
#   alfred-roster.sh watchdog   "<unit>\t<pid_file>" per line — vault daemons
#                               only; the watchdog adds alfred-mcp-http itself
#                               (it is not a vault, so not in config-meta.yaml)
#
# Unit-name derivation (per vault config path):
#   config.yaml        -> alfred                 (main vault, never templated)
#   config-<x>.yaml    -> alfred-<x>   if ~/.config/systemd/user/alfred-<x>.service
#                                      still exists (pre-template-migration), else
#                         alfred@<x>   if the alfred@.service template is installed,
#                                      else alfred-<x> (legacy default)
# This makes the output correct both before and after
# deploy/migrate-to-template-units.sh has been run.
#
# Failure contract: exits non-zero with NO stdout on any parse/read failure,
# so callers can fall back to their hardcoded rosters — a roster failure must
# never take the watchdog down with it. Output is buffered and emitted only
# on full success (no partial rosters).

set -euo pipefail

REPO="${ALFRED_REPO:-/home/rippere/alfred-v2}"
META="$REPO/config-meta.yaml"
PY="$REPO/.venv/bin/python"
MODE="${1:-units}"

case "$MODE" in
    units|watchdog) ;;
    *) echo "usage: $(basename "$0") [units|watchdog]" >&2; exit 2 ;;
esac

[[ -f "$META" && -x "$PY" ]] || exit 1

exec "$PY" - "$META" "$MODE" <<'PYEOF'
import sys
from pathlib import Path

import yaml

meta_path = Path(sys.argv[1])
mode = sys.argv[2]

meta = yaml.safe_load(meta_path.read_text()) or {}
vaults = meta.get("vaults") or []
if not vaults:
    sys.exit(1)

unit_dir = Path.home() / ".config" / "systemd" / "user"
template_installed = (unit_dir / "alfred@.service").is_file()

lines: list[str] = []
for vault in vaults:
    cfg_path = Path(vault["config"])
    stem = cfg_path.stem  # "config" or "config-<x>"
    if stem == "config":
        unit = "alfred"
    else:
        suffix = stem.removeprefix("config-")
        if not suffix:
            sys.exit(1)
        if (unit_dir / f"alfred-{suffix}.service").is_file():
            unit = f"alfred-{suffix}"          # pre-migration named unit
        elif template_installed:
            unit = f"alfred@{suffix}"          # template instance
        else:
            unit = f"alfred-{suffix}"          # legacy default
    if mode == "watchdog":
        # PID file lives in the vault's data_dir (default ./data, relative to
        # the config file — mirrors AlfredConfig.load()).
        raw = yaml.safe_load(cfg_path.read_text()) or {}
        data_dir = (cfg_path.parent / raw.get("data_dir", "./data")).resolve()
        lines.append(f"{unit}\t{data_dir / 'alfred.pid'}")
    else:
        lines.append(unit)

sys.stdout.write("\n".join(lines) + "\n")
PYEOF
