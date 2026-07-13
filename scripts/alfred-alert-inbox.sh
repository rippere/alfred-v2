#!/bin/bash
# alfred-alert-inbox — drop a tier-3 style alert note into the vault inbox
# naming a failed systemd unit. Invoked by alfred-alert@.service, which the
# peripheral oneshot units reference via OnFailure=alfred-alert@%n.service.
#
# Usage: alfred-alert-inbox.sh <failed-unit-name>

set -uo pipefail

INBOX="/mnt/external/obsidian-vault/inbox"
UNIT="${1:-unknown-unit}"

ts=$(date +%Y-%m-%d-%H%M%S)
mkdir -p "$INBOX"
detail=$(systemctl --user status "$UNIT" --no-pager -n 30 2>&1 || true)
cat > "$INBOX/alfred-unit-failure-${ts}.md" << EOF
<!-- alfred:source alfred_watchdog -->
# Alfred Unit Failure Alert — ${ts}

Unit **${UNIT}** failed (OnFailure= escalation).
Manual intervention required.

\`\`\`
${detail}
\`\`\`
EOF
echo "[alert-inbox] Inbox alert written for ${UNIT}: alfred-unit-failure-${ts}.md"
