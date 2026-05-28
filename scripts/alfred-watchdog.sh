#!/bin/bash
# Alfred watchdog — auto-heals stale PID files and restarts failed/crashed services.
# Runs every 5 minutes via alfred-watchdog.timer.
# On unrecoverable failure, drops an alert to the Alfred inbox for escalation.

set -euo pipefail

INBOX="/mnt/external/obsidian-vault/inbox"
WORK_DIR="/home/rippere/alfred-v2"

declare -A SERVICES=(
    [alfred]="$WORK_DIR/data/alfred.pid"
    [alfred-personal]="$WORK_DIR/data-personal/alfred.pid"
    [alfred-finance]="$WORK_DIR/data-finance/alfred.pid"
    [alfred-neuroscience]="$WORK_DIR/data-neuroscience/alfred.pid"
    [alfred-mcp-http]=""
)

_clear_stale_pid() {
    local pid_file="$1"
    [[ -z "$pid_file" || ! -f "$pid_file" ]] && return 0
    local pid
    pid=$(cut -d: -f1 < "$pid_file" 2>/dev/null || echo "")
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$pid_file"
        echo "[watchdog] Cleared stale PID file: $pid_file (PID $pid was dead)"
    fi
}

_alert_inbox() {
    local svc="$1"
    local detail="$2"
    local ts
    ts=$(date +%Y-%m-%d-%H%M%S)
    mkdir -p "$INBOX"
    cat > "$INBOX/alfred-watchdog-alert-${ts}.md" << EOF
<!-- alfred:source alfred_watchdog -->
# Alfred Watchdog Alert — ${ts}

Service **${svc}** failed to auto-recover.

\`\`\`
${detail}
\`\`\`
EOF
    echo "[watchdog] Alert written to inbox: alfred-watchdog-alert-${ts}.md"
}

HEALED=0
FAILED=0

for svc in "${!SERVICES[@]}"; do
    pid_file="${SERVICES[$svc]}"
    status=$(systemctl --user is-active "${svc}.service" 2>/dev/null || echo "unknown")

    if [[ "$status" == "active" ]]; then
        continue
    fi

    echo "[watchdog] ${svc}: status=${status} — attempting recovery"

    # Step 1: clear stale PID if present
    _clear_stale_pid "$pid_file"

    # Step 2: reset failed counter and restart
    systemctl --user reset-failed "${svc}.service" 2>/dev/null || true
    systemctl --user start "${svc}.service" 2>/dev/null || true
    sleep 4

    # Step 3: verify recovery
    new_status=$(systemctl --user is-active "${svc}.service" 2>/dev/null || echo "unknown")
    if [[ "$new_status" == "active" ]]; then
        echo "[watchdog] ${svc}: recovered successfully"
        HEALED=$((HEALED + 1))
    else
        echo "[watchdog] ${svc}: FAILED to recover — escalating to inbox"
        detail=$(systemctl --user status "${svc}.service" --no-pager -n 20 2>&1 || true)
        _alert_inbox "$svc" "$detail"
        FAILED=$((FAILED + 1))
    fi
done

echo "[watchdog] Done. Healed=${HEALED} Failed=${FAILED}"
