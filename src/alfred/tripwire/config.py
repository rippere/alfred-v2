"""Single source of truth for the x402 tripwire watcher's paths and thresholds.

The watcher converts the ADR-001 flip/abandon triggers (see
`x402-machine-readable-interfaces-converged-strategy-grilled-2026-07-15` in the
vault) from a markdown file a human has to remember into a monitored
instrument. It does NOT fetch data itself — ecosystem volume and competitor/
Coinbase news require judgment a script can't safely fabricate — it reads a
human- or agent-maintained signals file and applies the documented criteria
consistently every run.

Remote credentials are never needed here (the watcher only reads a local
signals file and writes a local inbox note), so unlike ledger/config.py there
is no env-file credential path.
"""

from __future__ import annotations

from pathlib import Path

# This file is src/alfred/tripwire/config.py -> repo root is three parents up.
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
DATA_DIR: Path = REPO_ROOT / "data"

# Human/agent-maintained monthly signals. Not committed (real numbers change
# monthly and belong to the operator, not the repo) — see signals.yaml.example
# for the schema.
SIGNALS_FILE: Path = DATA_DIR / "tripwire-signals.yaml"

INBOX_DIR: Path = Path("/mnt/external/obsidian-vault/inbox")

# Baseline + kill criterion sourced from the strategy doc (2026-07-15 grilling
# session, confirmed shared understanding with Ben). Overridable per-run via
# the signals file's `baseline_volume_usd` key — these are just the defaults.
BASELINE_VOLUME_USD: float = 1_600_000.0
# Ecosystem volume at/above this multiple of the baseline is the FLIP-TO-GO
# signal (Act 2 timing bet paying off, not just noise). Conservative default —
# tune in the signals file once real monthly data exists.
FLIP_VOLUME_MULTIPLIER: float = 2.0

# Watched competitors whose embedded-middleware moves count as tripwire input.
WATCHED_COMPETITORS: tuple[str, ...] = ("g402", "monapi", "Bankr")
