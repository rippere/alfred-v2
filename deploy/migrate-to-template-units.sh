#!/usr/bin/env bash
# migrate-to-template-units.sh — switch the per-vault Alfred daemons from
# copy-pasted named units (alfred-personal.service, ...) to the alfred@.service
# template (audit 2026-07-13, structural improvement #2).
#
# ── DO NOT RUN FROM AN AUTOMATED SESSION ─────────────────────────────────────
# This restarts live vault daemons. It is written for the MAIN (interactive)
# session to run deliberately, once, after reviewing it. It is idempotent:
# re-running after a partial failure is safe.
#
# What it does, in order:
#   1. Pre-flight: template + configs exist, template unit passes verify.
#   2. Stops + disables the four old named units and quarantines their unit
#      files to .trash-migration/ (never deletes).
#   3. Enables + starts alfred@personal alfred@finance alfred@neuroscience
#      alfred@employment.
#   4. Refreshes the hardcoded FALLBACK rosters in the watchdog and game-guard.
#      (The live rosters now populate at runtime from config-meta.yaml via
#      scripts/alfred-roster.sh — audit structural #3 — which auto-detects
#      named-vs-template units, so they need no rewrite; only the last-resort
#      fallback arrays still carry unit names worth keeping current.)
#   5. Verifies all instances are active and prints a post-migration checklist.
#
# What it deliberately does NOT touch:
#   - alfred.service (main vault): its config is plain config.yaml and its data
#     dir is data/ — no -suffix, so it does not fit the %i scheme. It stays a
#     named unit. (Renaming config.yaml -> config-main.yaml + data/ -> data-main/
#     would let it join the template, but that is a separate, riskier migration.)
#   - alfred-mcp-http.service, alfred-watchdog.*, timers — not per-vault daemons.
#   - The content vault: it has a config but no service (audit quick-win #22 is
#     still an open decision). If it is ever wired up: systemctl --user enable
#     --now alfred@content — no new unit file needed. That is the payoff here.

set -euo pipefail

REPO=/home/rippere/alfred-v2
UNIT_DIR="$HOME/.config/systemd/user"
TRASH="$REPO/.trash-migration/systemd-named-units-$(date +%Y%m%d-%H%M%S)"
VAULTS=(personal finance neuroscience employment)

WATCHDOG="$REPO/scripts/alfred-watchdog.sh"
GAME_GUARD="$HOME/.local/bin/ollama-game-guard.sh"

echo "── 1/5 pre-flight ──────────────────────────────────────────────────────"
[[ -f "$UNIT_DIR/alfred@.service" ]] || {
    echo "FATAL: $UNIT_DIR/alfred@.service missing — copy it from $REPO/deploy/systemd/ first" >&2
    exit 1
}
for v in "${VAULTS[@]}"; do
    [[ -f "$REPO/config-$v.yaml" ]] || { echo "FATAL: config-$v.yaml missing" >&2; exit 1; }
    [[ -d "$REPO/data-$v" ]]        || { echo "FATAL: data-$v/ missing" >&2; exit 1; }
done
systemd-analyze --user verify "$UNIT_DIR/alfred@.service"
echo "pre-flight OK"

echo "── 2/5 stop + disable old named units ──────────────────────────────────"
mkdir -p "$TRASH"
for v in "${VAULTS[@]}"; do
    old="alfred-$v.service"
    if systemctl --user list-unit-files --no-legend "$old" | grep -q .; then
        systemctl --user disable --now "$old" || true
        # Quarantine, never delete (repo policy).
        [[ -f "$UNIT_DIR/$old" ]] && mv -v "$UNIT_DIR/$old" "$TRASH/"
    else
        echo "$old already gone — skipping (idempotent re-run)"
    fi
done
systemctl --user daemon-reload

echo "── 3/5 enable + start template instances ───────────────────────────────"
for v in "${VAULTS[@]}"; do
    systemctl --user enable --now "alfred@$v.service"
done

echo "── 4/5 refresh hardcoded fallback rosters (watchdog + game-guard) ──────"
# Live rosters are runtime-derived from config-meta.yaml (alfred-roster.sh);
# these seds only keep the last-resort hardcoded fallbacks in sync with the
# post-migration unit names.
for v in "${VAULTS[@]}"; do
    sed -i "s/\[alfred-$v\]/[alfred@$v]/" "$WATCHDOG"
done
# Game-guard: FALLBACK_USER_UNITS=(alfred.service alfred-finance.service ...)
for v in "${VAULTS[@]}"; do
    sed -i "s/alfred-$v\.service/alfred@$v.service/g" "$GAME_GUARD"
done
echo "fallback rosters updated:"
grep -n "alfred@" "$WATCHDOG" | head -8 || true
grep -n "FALLBACK_USER_UNITS" "$GAME_GUARD" || true
echo "runtime roster now resolves to:"
"$REPO/scripts/alfred-roster.sh" units || echo "  (roster helper failed — fallbacks will be used)"

echo "── 5/5 verify ──────────────────────────────────────────────────────────"
sleep 5
fail=0
for u in alfred.service "${VAULTS[@]/#/alfred@}"; do
    # ${VAULTS[@]/#/alfred@} -> alfred@personal alfred@finance ... (no .service
    # suffix needed; systemctl resolves it)
    if systemctl --user is-active --quiet "$u"; then
        echo "  ACTIVE   $u"
    else
        echo "  FAILED   $u   <-- investigate: journalctl --user -u $u -n 50"
        fail=1
    fi
done

cat <<'EOF'

Post-migration checklist (manual):
  [ ] tail data-personal/alfred.log (and siblings) — daemons cycling normally
  [ ] next watchdog run logs Healed=0 Failed=0 (journalctl --user -u alfred-watchdog)
  [ ] RUNBOOK.md service table: alfred-<vault>.service -> alfred@<vault>.service
  [ ] commit the roster edits this script made to scripts/alfred-watchdog.sh
EOF

exit "$fail"
