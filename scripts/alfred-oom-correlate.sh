#!/usr/bin/env bash
# alfred-oom-correlate.sh — correlate journal kill events with LanceDB quarantines
#
# LanceDB table corruption is caused by a process dying mid-commit. The store
# self-heals (quarantine + recreate, see src/alfred/store/lancedb_store.py) but
# healing costs a full-vault re-embed, so each trip should be *diagnosed*:
# was the killer the kernel OOM-killer, a systemd SIGKILL escalation, or an
# unclean host shutdown? (Historical note: the 2026-05-27 23:50 incident was a
# hard power-off — journal ended mid-stream with no shutdown sequence.)
#
# With no args: scans the last 7 days of the user + system journals for
# kill-shaped events against alfred units, and prints any that land within
# 5 minutes of a `lancedb.table_quarantined` line in the per-vault
# data*/alfred.log files. Also lists boot boundaries in the window, since a
# power-loss kill leaves no journal line at all.
#
# Exit code: 0 always (reporting tool, not a health check).

set -euo pipefail

REPO="${ALFRED_REPO:-/home/rippere/alfred-v2}"
SINCE="${SINCE:-7 days ago}"
WINDOW_S=300  # correlation window: +/- 5 minutes

# Log signature emitted by LanceDBStore._quarantine_and_recreate()
QUARANTINE_SIG='lancedb\.table_quarantined'
# Kill-shaped journal lines (kernel OOM-killer, SIGKILL escalation, killed unit)
KILL_RE='oom.kill|Out of memory|oom_reaper|signal=KILL|SIGKILL|code=killed'

# ---------------------------------------------------------------- collection

# Kill events against alfred units/processes, one per line: "<epoch>\t<line>".
# System journal may be unreadable without privileges — best-effort.
collect_kills() {
  {
    journalctl --user --since "$SINCE" -o short-iso --no-pager -q 2>/dev/null || true
    journalctl        --since "$SINCE" -o short-iso --no-pager -q 2>/dev/null || true
  } | grep -Ei "$KILL_RE" | grep -i 'alfred' | sort -u | while IFS= read -r line; do
    ts=${line%% *}
    if epoch=$(date -d "$ts" +%s 2>/dev/null); then
      printf '%s\t%s\n' "$epoch" "$line"
    fi
  done
}

# Quarantine events from the vault logs: "<epoch>\t<file>: <line>".
# Log lines start "YYYY-MM-DD HH:MM:SS " (local time).
collect_quarantines() {
  grep -H "$QUARANTINE_SIG" "$REPO"/data*/alfred.log* 2>/dev/null |
  while IFS= read -r entry; do
    file=${entry%%:*}
    line=${entry#"$file":}
    ts=$(printf '%s' "$line" | cut -c1-19)
    if epoch=$(date -d "$ts" +%s 2>/dev/null); then
      printf '%s\t%s: %s\n' "$epoch" "${file#"$REPO"/}" "$line"
    fi
  done
}

# ---------------------------------------------------------------- report

kills=$(collect_kills || true)
quarantines=$(collect_quarantines || true)

n_kills=0
n_quarantines=0
if [ -n "$kills" ]; then n_kills=$(printf '%s\n' "$kills" | wc -l); fi
if [ -n "$quarantines" ]; then n_quarantines=$(printf '%s\n' "$quarantines" | wc -l); fi

echo "alfred-oom-correlate: journal window '${SINCE}', correlation +/- ${WINDOW_S}s"
echo "  kill-shaped alfred journal events : ${n_kills}"
echo "  lancedb.table_quarantined lines   : ${n_quarantines} (all of data*/alfred.log*)"
echo

if [ "$n_kills" -gt 0 ]; then
  echo "--- kill events (journal) ---"
  printf '%s\n' "$kills" | cut -f2-
  echo
fi

matched=0
if [ "$n_quarantines" -gt 0 ]; then
  echo "--- quarantine events (vault logs) ---"
  while IFS=$'\t' read -r q_epoch q_line; do
    [ -n "$q_epoch" ] || continue
    echo "$q_line"
    if [ -n "$kills" ]; then
      while IFS=$'\t' read -r k_epoch k_line; do
        [ -n "$k_epoch" ] || continue
        delta=$(( q_epoch - k_epoch ))
        [ "$delta" -lt 0 ] && delta=$(( -delta ))
        if [ "$delta" -le "$WINDOW_S" ]; then
          echo "    CORRELATED (+/-${delta}s): $k_line"
          matched=$(( matched + 1 ))
        fi
      done <<< "$kills"
    fi
  done <<< "$quarantines"
  echo
fi

echo "--- boot boundaries (unclean shutdown = a kill with no journal line) ---"
journalctl --list-boots --no-pager -q 2>/dev/null | tail -8 || echo "  (journal unavailable)"
echo

if [ "$matched" -gt 0 ]; then
  echo "RESULT: ${matched} kill event(s) within ${WINDOW_S}s of a quarantine — likely cause identified above."
elif [ "$n_quarantines" -gt 0 ]; then
  echo "RESULT: quarantine(s) found but no correlated kill in the journal window."
  echo "        Check the boot boundaries above — an unclean shutdown/power loss"
  echo "        (the confirmed cause of the 2026-05-27 corruption) leaves no kill line."
else
  echo "RESULT: no quarantine events in the vault logs; nothing to correlate."
fi

exit 0
