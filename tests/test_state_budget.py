"""The API spend-cap and credit-exhaustion pause gates must actually gate.

StateStore is instantiated all over the test suite, but its two production
throttles — the daily spend cap (budget_remaining/can_make_api_call) and the
credit-exhaustion pause (record_api_failure/_is_paused) — are never asserted.
They are the branches that stop the daemon from burning the Anthropic account
after a "credit balance is too low" error or once the daily call budget is
spent (live callers: distiller, curator, consolidator), so a silent regression
here is exactly the kind that only shows up as a surprise bill.

These tests construct a StateStore against a tmp state.json with cfg=None (the
production default of 500 calls/day) and hold the floor under each branch: the
recognized credit-low signature pauses every daemon, an unrecognized error does
not, the budget counts down and clamps at zero, a stale api_calls_date rolls the
daily counter over, and an expired/malformed pause timestamp clears itself. No
source change — pure characterization of the existing gates.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from alfred.store.state import StateStore


def _store(tmp_path) -> StateStore:
    """A loaded StateStore on a fresh tmp state.json, cfg=None (limit 500)."""
    store = StateStore(tmp_path / "state.json", cfg=None)
    store.load()
    return store


def test_credit_low_failure_pauses_all_daemons(tmp_path):
    """A recognized credit-exhaustion error records a future pause and makes
    can_make_api_call() return False for every daemon until it expires —
    retrying is pointless while the account has no credit."""
    store = _store(tmp_path)
    assert store.can_make_api_call() is True

    store.record_api_failure("Your credit balance is too low to run this request")

    assert store.state.api_paused_until, "credit-low error left no pause timestamp"
    resume_at = datetime.fromisoformat(store.state.api_paused_until)
    assert resume_at > datetime.now(timezone.utc), "pause must be in the future"
    assert store.can_make_api_call() is False
    assert store.can_make_api_call(daemon="distiller") is False


def test_unrecognized_error_does_not_pause(tmp_path):
    """A transient/unknown error is not a signature match: no pause is set and
    calls keep flowing, so ordinary faults retry on their normal cadence."""
    store = _store(tmp_path)

    store.record_api_failure("APIConnectionError: connection reset by peer")

    assert store.state.api_paused_until == ""
    assert store.can_make_api_call() is True


def test_budget_remaining_counts_down_from_default_500(tmp_path):
    """With cfg=None the cap is 500/day; budget_remaining() == 500 - N after N
    recorded calls."""
    store = _store(tmp_path)
    assert store.budget_remaining() == 500

    for _ in range(3):
        store.record_api_call(input_tokens=100, output_tokens=50)

    assert store.budget_remaining() == 497


def test_budget_remaining_clamps_at_zero_when_over_limit(tmp_path):
    """Past the cap the remaining count floors at 0 (never negative) and the
    spend-cap gate closes."""
    store = _store(tmp_path)
    store.state.api_calls_today = 510  # over the 500 default, same-day

    assert store.budget_remaining() == 0
    assert store.can_make_api_call() is False


def test_stale_api_calls_date_resets_daily_counter(tmp_path):
    """A budget check on a new day rolls the daily counter over rather than
    carrying yesterday's spend forward."""
    store = _store(tmp_path)
    store.state.api_calls_date = "2000-01-01"
    store.state.api_calls_today = 99
    store.state.api_cost_usd_today = 1.23

    assert store.budget_remaining() == 500
    assert store.state.api_calls_today == 0
    assert store.state.api_cost_usd_today == 0.0
    assert store.state.api_calls_date == datetime.now(timezone.utc).date().isoformat()


def test_expired_pause_clears_and_reopens_calls(tmp_path):
    """Once the recorded pause window has passed, the gate reopens and the
    stale timestamp is cleared."""
    store = _store(tmp_path)
    past = datetime.now(timezone.utc) - timedelta(seconds=1)
    store.state.api_paused_until = past.isoformat()

    assert store.can_make_api_call() is True
    assert store.state.api_paused_until == "", "expired pause should be cleared"


def test_malformed_pause_timestamp_is_cleared(tmp_path):
    """A corrupt api_paused_until must not wedge the daemon shut — it is treated
    as no pause and cleared rather than raising."""
    store = _store(tmp_path)
    store.state.api_paused_until = "not-a-timestamp"

    assert store.can_make_api_call() is True
    assert store.state.api_paused_until == ""
