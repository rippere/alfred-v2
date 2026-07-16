# Alfred ledger — systemd timer install

These are **templates**. Nothing here is installed or enabled automatically.

The timer fires `alfred ledger collect --push` once a day at **23:45 local**.
`Persistent=true` means a snapshot missed while the machine was off/asleep runs
on the next boot.

## Install (user service — recommended)

```bash
mkdir -p ~/.config/systemd/user
cp /home/rippere/alfred-v2/deploy/systemd/ledger-collect.service ~/.config/systemd/user/
cp /home/rippere/alfred-v2/deploy/systemd/ledger-collect.timer   ~/.config/systemd/user/

systemctl --user daemon-reload
systemctl --user enable --now ledger-collect.timer

# verify
systemctl --user list-timers ledger-collect.timer
systemctl --user start ledger-collect.service   # run once now
journalctl --user -u ledger-collect.service -n 50 --no-pager
```

> A **user** unit only runs while you're logged in. To run headless/always-on,
> enable lingering: `loginctl enable-linger $USER` (or install as a system unit
> with a `User=` line and an absolute `Environment=TZ=...`).

## Push credentials

`--push` reads `~/.config/alfred-ledger/env`. If any required key is missing the
push is a **dry run** (payload summary printed, `push_log.status='dry'`) — the
snapshot is still written to SQLite. Copy the template and fill it in:

```bash
cp ~/.config/alfred-ledger/env.example ~/.config/alfred-ledger/env
$EDITOR ~/.config/alfred-ledger/env
```

Required keys: `LEDGER_SUPABASE_URL`, `LEDGER_SUPABASE_ANON_KEY`,
`LEDGER_API_URL`, `LEDGER_WORKSPACE_ID`, `LEDGER_EMAIL`, `LEDGER_PASSWORD`.

## Backfill history once

```bash
/home/rippere/alfred-v2/.venv/bin/alfred ledger backfill --since 2026-05-01
/home/rippere/alfred-v2/.venv/bin/alfred ledger show --days 35
```

## x402 tripwire watcher

The timer fires `alfred tripwire check` once a month (1st, 09:00 local) and
writes a HOLD / FLIP-TO-GO / ABANDON-WEDGE verdict note to the Alfred inbox,
per the ADR-001 flip/abandon triggers.

```bash
cp /home/rippere/alfred-v2/deploy/systemd/alfred-tripwire.service ~/.config/systemd/user/
cp /home/rippere/alfred-v2/deploy/systemd/alfred-tripwire.timer   ~/.config/systemd/user/

systemctl --user daemon-reload
systemctl --user enable --now alfred-tripwire.timer

# verify
systemctl --user list-timers alfred-tripwire.timer
systemctl --user start alfred-tripwire.service   # run once now
journalctl --user -u alfred-tripwire.service -n 50 --no-pager
```

Before the first run, fill in this month's signals:

```bash
cp /home/rippere/alfred-v2/tripwire-signals.yaml.example /home/rippere/alfred-v2/data/tripwire-signals.yaml
$EDITOR /home/rippere/alfred-v2/data/tripwire-signals.yaml
```

The check exits non-zero (and the timer's `OnFailure=` fires an inbox alert)
if the signals file is missing or malformed — it never silently defaults to
HOLD on bad input.
