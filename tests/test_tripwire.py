"""x402 tripwire watcher — verdict logic + inbox note writing.

Covers the ADR-001 triggers converted from the strategy doc into code:
Coinbase-ships-a-dashboard kill criterion, volume-vs-baseline flip threshold,
and the HOLD default. Also proves load_signals never silently defaults on
missing/malformed input, and that the inbox note lands with the repo's
standard `alfred:source` header.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from alfred.tripwire import inbox as I
from alfred.tripwire import watcher as W


def _signals(**overrides) -> W.Signals:
    base = dict(
        ecosystem_volume_usd=1_000_000.0,
        coinbase_seller_dashboard_shipped=False,
        competitor_moves={},
        baseline_volume_usd=1_600_000.0,
        notes="",
    )
    base.update(overrides)
    return W.Signals(**base)


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

def test_coinbase_dashboard_is_abandon_wedge_and_overrides_everything():
    s = _signals(
        coinbase_seller_dashboard_shipped=True,
        ecosystem_volume_usd=10_000_000.0,  # would otherwise be a huge flip signal
    )
    v = W.evaluate(s)
    assert v.verdict == W.ABANDON_WEDGE
    assert any("Coinbase" in r for r in v.reasons)


def test_volume_at_flip_multiplier_is_flip_to_go():
    s = _signals(ecosystem_volume_usd=3_200_000.0, baseline_volume_usd=1_600_000.0)  # exactly 2x
    v = W.evaluate(s)
    assert v.verdict == W.FLIP_TO_GO
    assert any("baseline" in r for r in v.reasons)


def test_volume_just_below_flip_multiplier_is_hold():
    s = _signals(ecosystem_volume_usd=3_199_999.0, baseline_volume_usd=1_600_000.0)
    v = W.evaluate(s)
    assert v.verdict == W.HOLD


def test_default_hold_with_no_triggers():
    v = W.evaluate(_signals())
    assert v.verdict == W.HOLD
    assert v.reasons  # never empty — always explains itself


def test_competitor_moves_noted_but_do_not_force_a_flip():
    s = _signals(competitor_moves={"g402": "shipped embedded checkout SDK"})
    v = W.evaluate(s)
    assert v.verdict == W.HOLD
    assert any("g402" in r for r in v.reasons)


def test_boilerplate_competitor_moves_are_not_treated_as_signals():
    s = _signals(competitor_moves={"g402": "no major move", "monapi": "None", "Bankr": ""})
    v = W.evaluate(s)
    assert v.verdict == W.HOLD
    assert not any("g402" in r or "monapi" in r or "Bankr" in r for r in v.reasons)


# ---------------------------------------------------------------------------
# load_signals()
# ---------------------------------------------------------------------------

def test_load_signals_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        W.load_signals(tmp_path / "does-not-exist.yaml")


def test_load_signals_missing_required_key_raises(tmp_path):
    p = tmp_path / "signals.yaml"
    p.write_text("ecosystem_volume_usd: 1000000\n")  # missing coinbase_seller_dashboard_shipped
    with pytest.raises(ValueError):
        W.load_signals(p)


def test_load_signals_round_trip(tmp_path):
    p = tmp_path / "signals.yaml"
    p.write_text(
        "ecosystem_volume_usd: 2500000\n"
        "coinbase_seller_dashboard_shipped: false\n"
        "competitor_moves:\n"
        "  g402: shipped v2\n"
        "notes: test note\n"
    )
    s = W.load_signals(p)
    assert s.ecosystem_volume_usd == 2_500_000.0
    assert s.coinbase_seller_dashboard_shipped is False
    assert s.competitor_moves == {"g402": "shipped v2"}
    assert s.notes == "test note"
    assert s.baseline_volume_usd == 1_600_000.0  # default, not overridden


def test_load_signals_baseline_override(tmp_path):
    p = tmp_path / "signals.yaml"
    p.write_text(
        "ecosystem_volume_usd: 1000000\n"
        "coinbase_seller_dashboard_shipped: false\n"
        "baseline_volume_usd: 500000\n"
    )
    s = W.load_signals(p)
    assert s.baseline_volume_usd == 500_000.0


# ---------------------------------------------------------------------------
# inbox note
# ---------------------------------------------------------------------------

def test_render_note_includes_verdict_and_source_header():
    v = W.evaluate(_signals(ecosystem_volume_usd=5_000_000.0))
    note = I.render_note(v)
    assert "<!-- alfred:source x402_tripwire_watcher -->" in note
    assert "FLIP-TO-GO" in note
    assert "$5,000,000" in note


def test_write_verdict_note_writes_file_in_target_dir(tmp_path):
    v = W.evaluate(_signals())
    out = I.write_verdict_note(v, inbox_dir=tmp_path)
    assert out.exists()
    assert out.parent == tmp_path
    assert out.name.startswith("x402-tripwire-")
    assert "HOLD" in out.read_text()
