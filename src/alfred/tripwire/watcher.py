"""Verdict logic: signals in, HOLD / FLIP-TO-GO / ABANDON-WEDGE out.

Reads the monthly signals file (ecosystem volume, competitor embedded-
middleware moves, whether Coinbase shipped a first-party seller dashboard)
and applies the ADR-001 triggers documented in the strategy doc:

* ABANDON-WEDGE — Coinbase ships a first-party seller dashboard. Explicit
  kill criterion; overrides everything else.
* FLIP-TO-GO — ecosystem volume clears `FLIP_VOLUME_MULTIPLIER` x the
  baseline. The timing bet paying off, not noise.
* HOLD — default. Position is cheap to hold pre-inflection; absence of a
  trigger is not itself a signal.

Never fabricates a signal: a missing/unreadable signals file raises, on the
theory that a stale or absent read should block the verdict rather than
silently defaulting to HOLD.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from alfred.tripwire import config as C

HOLD = "HOLD"
FLIP_TO_GO = "FLIP-TO-GO"
ABANDON_WEDGE = "ABANDON-WEDGE"


@dataclass(frozen=True)
class Signals:
    ecosystem_volume_usd: float
    coinbase_seller_dashboard_shipped: bool
    competitor_moves: dict[str, str] = field(default_factory=dict)
    baseline_volume_usd: float = C.BASELINE_VOLUME_USD
    notes: str = ""


@dataclass(frozen=True)
class Verdict:
    verdict: str
    reasons: list[str]
    signals: Signals


def load_signals(path=None) -> Signals:
    """Read + validate the signals file. Raises on missing/malformed input —
    a tripwire that silently defaults on bad data isn't a tripwire."""
    signals_path = path or C.SIGNALS_FILE
    if not signals_path.exists():
        raise FileNotFoundError(
            f"tripwire signals file not found: {signals_path}\n"
            f"Copy tripwire-signals.yaml.example (repo root) to {signals_path} "
            f"and fill in this month's numbers."
        )
    raw = yaml.safe_load(signals_path.read_text()) or {}

    missing = [k for k in ("ecosystem_volume_usd", "coinbase_seller_dashboard_shipped") if k not in raw]
    if missing:
        raise ValueError(f"tripwire signals file missing required key(s): {', '.join(missing)}")

    return Signals(
        ecosystem_volume_usd=float(raw["ecosystem_volume_usd"]),
        coinbase_seller_dashboard_shipped=bool(raw["coinbase_seller_dashboard_shipped"]),
        competitor_moves={str(k): str(v) for k, v in (raw.get("competitor_moves") or {}).items()},
        baseline_volume_usd=float(raw.get("baseline_volume_usd", C.BASELINE_VOLUME_USD)),
        notes=str(raw.get("notes", "")),
    )


def evaluate(signals: Signals) -> Verdict:
    """Apply the ADR-001 triggers to a signals snapshot. Pure function — no I/O."""
    reasons: list[str] = []

    if signals.coinbase_seller_dashboard_shipped:
        reasons.append(
            "Coinbase shipped a first-party seller dashboard — explicit kill criterion (ADR-001)."
        )
        return Verdict(ABANDON_WEDGE, reasons, signals)

    flip_threshold = signals.baseline_volume_usd * C.FLIP_VOLUME_MULTIPLIER
    if signals.ecosystem_volume_usd >= flip_threshold:
        reasons.append(
            f"ecosystem volume ${signals.ecosystem_volume_usd:,.0f}/mo >= "
            f"{C.FLIP_VOLUME_MULTIPLIER:g}x baseline (${flip_threshold:,.0f}/mo)."
        )
        return Verdict(FLIP_TO_GO, reasons, signals)

    major_moves = {
        k: v for k, v in signals.competitor_moves.items()
        if v and v.strip().lower() not in ("", "no major move", "none")
    }
    if major_moves:
        for name, move in major_moves.items():
            reasons.append(f"{name}: {move}")
        reasons.append("Competitor move(s) noted — does not on its own trigger a flip, see notes below.")

    if not reasons:
        reasons.append(
            f"ecosystem volume ${signals.ecosystem_volume_usd:,.0f}/mo below flip threshold "
            f"(${flip_threshold:,.0f}/mo); no kill criterion hit."
        )

    return Verdict(HOLD, reasons, signals)
