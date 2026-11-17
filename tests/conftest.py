"""Test bootstrap: make src/ importable without requiring an editable install."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest


@pytest.fixture(autouse=True)
def _no_live_vault_corruption_alerts(tmp_path, monkeypatch):
    """Keep LanceDBStore's quarantine alert out of the real vault inbox.

    The quarantine tests trigger a genuine auto-quarantine, and the alert path
    used to be the live /mnt/external/obsidian-vault/inbox — so every test run
    left a fake "LanceDB corruption" note that the curator then ingested.
    Point it at a per-test directory that does not exist (the alert is a no-op
    when its inbox is missing).
    """
    from alfred.store import lancedb_store

    monkeypatch.setattr(
        lancedb_store, "CORRUPTION_ALERT_INBOX", tmp_path / "no-live-inbox", raising=True
    )
