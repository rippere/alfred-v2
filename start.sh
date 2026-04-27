#!/usr/bin/env bash
# Alfred v2 — start script
# Strips Claude Code env vars to avoid auth conflicts and process isolation issues.
# Run from terminal or Hyprland exec-once, NOT from within a Claude Code session.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

env \
  -u CLAUDECODE \
  -u CLAUDE_CODE_ENTRYPOINT \
  -u CLAUDE_CODE_EXECPATH \
  "$SCRIPT_DIR/.venv/bin/alfred" up
