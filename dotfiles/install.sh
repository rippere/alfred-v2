#!/usr/bin/env bash
# Alfred dotfiles installer — run on any machine to install Claude Code hooks.
# Usage: bash /path/to/alfred-v2/dotfiles/install.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Alfred dotfiles installer"
echo "Script dir: $SCRIPT_DIR"

# Install Claude Code hooks
HOOKS_DST="$HOME/.config/claude/hooks"
mkdir -p "$HOOKS_DST"
cp "$SCRIPT_DIR/claude-hooks/"* "$HOOKS_DST/"
chmod +x "$HOOKS_DST/"*.sh
echo "Hooks installed to $HOOKS_DST"

echo ""
echo "NEXT: Merge the following into ~/.claude/settings.json"
echo "(On the laptop, use MCP type 'sse' pointing to the desktop Alfred MCP server)"
echo ""
cat "$SCRIPT_DIR/settings-snippet.json"
echo ""
echo "Done."
