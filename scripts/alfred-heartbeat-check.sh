#!/bin/bash
# alfred-heartbeat-check — dead-man's switch for the alfred watchdog.
#
# alfred-watchdog.sh touches $HEARTBEAT_FILE at the start of every 5-minute
# run. If that file goes stale (> MAX_AGE_MIN), the watchdog itself has
# stopped running — the one failure it cannot report on its own — so we
# escalate straight to the tier-3 inbox path.
#
# No dedicated timer: wired as ExecStartPre= on ledger-collect.service so it
# piggybacks on the existing daily ledger-collect.timer. Always exits 0 so a
# stale heartbeat never blocks the ledger collection itself.

set -uo pipefail

INBOX="/mnt/external/obsidian-vault/inbox"
HEARTBEAT_FILE="${XDG_STATE_HOME:-$HOME/.local/state}/alfred/watchdog.heartbeat"
MAX_AGE_MIN=15

if [[ ! -f "$HEARTBEAT_FILE" ]]; then
    age_desc="heartbeat file missing (never written)"
    stale=1
else
    now=$(date +%s)
    mtime=$(stat -c %Y "$HEARTBEAT_FILE" 2>/dev/null || echo 0)
    age_min=$(( (now - mtime) / 60 ))
    age_desc="last heartbeat ${age_min} minutes ago"
    stale=$(( age_min > MAX_AGE_MIN ? 1 : 0 ))
fi

if [[ "$stale" -eq 0 ]]; then
    echo "[heartbeat-check] OK — ${age_desc}"
    exit 0
fi

ts=$(date +%Y-%m-%d-%H%M%S)
mkdir -p "$INBOX"
timer_status=$(systemctl --user status alfred-watchdog.timer --no-pager -n 10 2>&1 || true)
cat > "$INBOX/alfred-watchdog-alert-${ts}.md" << EOF
<!-- alfred:source alfred_watchdog -->
# Alfred Watchdog Alert — ${ts}

Watchdog **heartbeat is stale** (${age_desc}, threshold ${MAX_AGE_MIN} min).
The watchdog itself is not running — service failures are currently going
undetected. Manual intervention required (check alfred-watchdog.timer).

\`\`\`
heartbeat file: ${HEARTBEAT_FILE}
${timer_status}
\`\`\`
EOF
echo "[heartbeat-check] STALE — ${age_desc}; inbox alert written: alfred-watchdog-alert-${ts}.md"
exit 0
