#!/bin/bash
# Alfred watchdog — 3-tier recovery for all alfred services.
# Tier 1: auto-fix (stale PID clear + restart)
# Tier 2: claude healing agent (non-interactive claude -p)
# Tier 3: inbox alert — human escalation only if Claude also fails
# Runs every 5 minutes via alfred-watchdog.timer.

set -euo pipefail

INBOX="/mnt/external/obsidian-vault/inbox"
WORK_DIR="/home/rippere/alfred-v2"
CLAUDE_BIN="$(which claude 2>/dev/null || echo "")"

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

_tier2_claude_heal() {
    local svc="$1"
    local pid_file="$2"

    if [[ -z "$CLAUDE_BIN" ]]; then
        echo "[watchdog] claude CLI not found — skipping tier 2"
        return 1
    fi

    echo "[watchdog] ${svc}: invoking Claude healing agent (tier 2)..."

    local journal
    journal=$(journalctl --user -u "${svc}.service" --no-pager -n 40 2>&1 || true)

    local prompt
    prompt="You are an automated healer for alfred systemd services. The service '${svc}.service' is not active.

Journal (last 40 lines):
${journal}

PID file path (may or may not exist): ${pid_file}

Your job:
1. Read the journal output above and identify the failure cause
2. If a stale PID file exists at the path above, check if the PID is alive: run 'kill -0 \$(cut -d: -f1 < ${pid_file} 2>/dev/null)' — if dead, remove it
3. Run: systemctl --user reset-failed ${svc}.service
4. Run: systemctl --user start ${svc}.service
5. Wait 4 seconds, then check: systemctl --user is-active ${svc}.service
6. If still not active, try one more diagnostic step based on the journal error
7. Final check: output exactly 'HEALED' if the service is now active, or 'FAILED: <reason>' if not

Only use Bash tool. Be concise. This runs unattended."

    local result
    result=$(CLAUDECODE="" "$CLAUDE_BIN" --dangerously-skip-permissions -p "$prompt" 2>&1 | tail -5 || true)
    echo "[watchdog] Claude result: $result"

    # Check if service came up regardless of Claude's output
    local post_status
    post_status=$(systemctl --user is-active "${svc}.service" 2>/dev/null || echo "unknown")
    [[ "$post_status" == "active" ]]
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

Service **${svc}** could not be recovered by auto-fix or Claude healer.
Manual intervention required.

\`\`\`
${detail}
\`\`\`
EOF
    echo "[watchdog] Inbox alert written: alfred-watchdog-alert-${ts}.md"
}

HEALED=0
FAILED=0

for svc in "${!SERVICES[@]}"; do
    pid_file="${SERVICES[$svc]}"
    status=$(systemctl --user is-active "${svc}.service" 2>/dev/null || echo "unknown")

    if [[ "$status" == "active" ]]; then
        continue
    fi

    echo "[watchdog] ${svc}: status=${status}"

    # ── Tier 1: stale PID clear + restart ────────────────────────────────────
    echo "[watchdog] ${svc}: tier 1 — stale PID clear + restart"
    _clear_stale_pid "$pid_file"
    systemctl --user reset-failed "${svc}.service" 2>/dev/null || true
    systemctl --user start "${svc}.service" 2>/dev/null || true
    sleep 4

    tier1_status=$(systemctl --user is-active "${svc}.service" 2>/dev/null || echo "unknown")
    if [[ "$tier1_status" == "active" ]]; then
        echo "[watchdog] ${svc}: healed by tier 1"
        HEALED=$((HEALED + 1))
        continue
    fi

    # ── Tier 2: Claude healing agent ─────────────────────────────────────────
    echo "[watchdog] ${svc}: tier 1 failed — escalating to Claude healer (tier 2)"
    if _tier2_claude_heal "$svc" "$pid_file"; then
        echo "[watchdog] ${svc}: healed by Claude (tier 2)"
        HEALED=$((HEALED + 1))
        continue
    fi

    # ── Tier 3: inbox alert — human escalation ───────────────────────────────
    echo "[watchdog] ${svc}: all recovery tiers failed — alerting inbox (tier 3)"
    detail=$(systemctl --user status "${svc}.service" --no-pager -n 30 2>&1 || true)
    _alert_inbox "$svc" "$detail"
    FAILED=$((FAILED + 1))
done

echo "[watchdog] Done. Healed=${HEALED} Failed=${FAILED}"
