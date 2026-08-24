"""x402 tripwire watcher.

Converts the ADR-001 flip/abandon triggers for the x402 wedge (see the
2026-07-15 strategy doc) from a markdown file a human has to remember into a
monitored instrument. Reads a monthly, human/agent-maintained signals file and
applies the documented criteria to produce a HOLD / FLIP-TO-GO / ABANDON-WEDGE
verdict, written as a note to the Alfred inbox.

Public surface:
    - ``alfred tripwire check [--signals PATH] [--no-write]``

Read-only against every source except the Alfred inbox, where it writes one
verdict note per run.
"""

from __future__ import annotations

__all__ = ["config", "watcher", "inbox"]
