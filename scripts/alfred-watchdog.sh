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

# ── dead-man's switch heartbeat ───────────────────────────────────────────────
# Touched at the start of EVERY run (including game-guard early exits) so a
# separate low-frequency check (scripts/alfred-heartbeat-check.sh, run via
# ledger-collect.service's ExecStartPre=) can alert if the watchdog itself
# stops running — the one failure the watchdog can't report on its own.
HEARTBEAT_FILE="${XDG_STATE_HOME:-$HOME/.local/state}/alfred/watchdog.heartbeat"
mkdir -p "$(dirname "$HEARTBEAT_FILE")"
touch "$HEARTBEAT_FILE"

# ── game-guard awareness ──────────────────────────────────────────────────────
# ollama-game-guard intentionally pauses the alfred fleet while a game runs and
# re-enforces the pause every 5s — healing mid-game just creates the recursive
# heal-vs-pause storm observed 2026-06-04. Skip entirely while the pause is in
# effect. Mode semantics mirror the guard: off → guard disabled, heal normally;
# on → forced GPU-free, fleet is MEANT to be down; auto → detect a live game.
GUARD_MODE_FILE="${XDG_STATE_HOME:-$HOME/.local/state}/ollama-game-guard/mode"
_game_active() {
    local mode
    mode="$(cat "$GUARD_MODE_FILE" 2>/dev/null || echo auto)"
    [[ "$mode" == "off" ]] && return 1
    [[ "$mode" == "on" ]] && return 0
    pgrep -f 'reaper.*SteamLaunch AppId=' >/dev/null 2>&1 && return 0
    gamemoded -s 2>/dev/null | grep -q 'gamemode is active' && return 0
    return 1
}
if _game_active; then
    echo "[watchdog] game active — fleet intentionally paused by game-guard; skipping all tiers"
    exit 0
fi

# ── service roster ────────────────────────────────────────────────────────────
# Populated at runtime from config-meta.yaml via scripts/alfred-roster.sh (the
# single source of truth — audit structural #3), so a vault added with
# `alfred create-vault` is monitored on the next 5-minute run with no edit here.
# If the helper or config-meta.yaml is unreadable we fall back to the hardcoded
# roster below — the watchdog must never die from a roster parse failure.
ROSTER_HELPER="$WORK_DIR/scripts/alfred-roster.sh"
declare -A SERVICES=()
if roster_out="$("$ROSTER_HELPER" watchdog 2>/dev/null)" && [[ -n "$roster_out" ]]; then
    while IFS=$'\t' read -r roster_unit roster_pid_file; do
        [[ -n "$roster_unit" ]] && SERVICES["$roster_unit"]="$roster_pid_file"
    done <<< "$roster_out"
else
    echo "[watchdog] roster helper failed — using hardcoded fallback roster"
    SERVICES=(
        [alfred]="$WORK_DIR/data/alfred.pid"
        [alfred-personal]="$WORK_DIR/data-personal/alfred.pid"
        [alfred-finance]="$WORK_DIR/data-finance/alfred.pid"
        [alfred-neuroscience]="$WORK_DIR/data-neuroscience/alfred.pid"
        [alfred-employment]="$WORK_DIR/data-employment/alfred.pid"
    )
fi
# Not a vault (absent from config-meta.yaml) — always in the watchdog set.
SERVICES[alfred-mcp-http]=""

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
    # Narrowed to alfred-finance only — the one failure signature tier 2 has
    # actually fixed (4/46 successes overall, all alfred-finance). Every other
    # service goes straight tier 1 → tier 3; each tier-2 run is a live costed
    # API call.
    # (matches both the pre-migration named unit and the alfred@ template form)
    if [[ "$svc" == "alfred-finance" || "$svc" == "alfred@finance" ]]; then
        echo "[watchdog] ${svc}: tier 1 failed — escalating to Claude healer (tier 2)"
        if _tier2_claude_heal "$svc" "$pid_file"; then
            echo "[watchdog] ${svc}: healed by Claude (tier 2)"
            HEALED=$((HEALED + 1))
            continue
        fi
    else
        echo "[watchdog] ${svc}: tier 1 failed — tier 2 skipped (claude healer reserved for alfred-finance)"
    fi

    # ── Tier 3: inbox alert — human escalation ───────────────────────────────
    echo "[watchdog] ${svc}: all recovery tiers failed — alerting inbox (tier 3)"
    detail=$(systemctl --user status "${svc}.service" --no-pager -n 30 2>&1 || true)
    _alert_inbox "$svc" "$detail"
    FAILED=$((FAILED + 1))
done

echo "[watchdog] Done. Healed=${HEALED} Failed=${FAILED}"
