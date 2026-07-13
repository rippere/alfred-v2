"""Golden-path tests for the unified provenance predicate (audit quick win #7 / #6b).

Covers all four former call-site behaviors that `alfred.core.provenance` replaced:

1. curator/distiller: path-prefix check (learn/, topic/, synthesis/)
2. distiller: frontmatter `generated_by: llm` field check
3. consolidator: record-type tuple gate (learn/topic/synthesis)
4. surveyor: raw-bytes grep fast path on the diff scan
"""
from __future__ import annotations

import pytest

from alfred.core.provenance import (
    DAEMON_OUTPUT_PREFIXES,
    DAEMON_OUTPUT_TYPES,
    is_daemon_generated,
    is_daemon_generated_raw,
)

# ── 1. Path-prefix behavior (curator/distiller/consolidator) ─────────────────

@pytest.mark.parametrize("rel_path", [
    "learn/2026-07-01-distilled.md",
    "topic/quantum-computing.md",
    "synthesis/cluster-042.md",
    "learn/nested/deeper.md",
])
def test_daemon_output_paths_are_flagged(rel_path):
    assert is_daemon_generated(rel_path) is True


@pytest.mark.parametrize("rel_path", [
    "inbox/raw-idea.md",
    "notes/learn-to-cook.md",          # prefix must anchor at path start
    "projects/topic-modeling.md",
    "notes/learn/nested.md",           # daemon dirs are top-level only
    "wiki/alfred.md",
    "",
])
def test_human_authored_paths_are_not_flagged(rel_path):
    assert is_daemon_generated(rel_path) is False


# ── 2. `generated_by` frontmatter behavior (distiller) ──────────────────────

def test_generated_by_llm_is_flagged():
    assert is_daemon_generated(generated_by="llm") is True


@pytest.mark.parametrize("generated_by", ["human", "ben", "", None])
def test_other_generated_by_values_are_not_flagged(generated_by):
    assert is_daemon_generated(generated_by=generated_by) is False


# ── 3. Record-type behavior (consolidator's former tuple gate) ───────────────

@pytest.mark.parametrize("record_type", sorted(DAEMON_OUTPUT_TYPES))
def test_daemon_record_types_are_flagged(record_type):
    assert is_daemon_generated(record_type=record_type) is True


@pytest.mark.parametrize("record_type", ["note", "project", "person", "", None])
def test_human_record_types_are_not_flagged(record_type):
    assert is_daemon_generated(record_type=record_type) is False


# ── Combined signals: any one positive signal wins ───────────────────────────

def test_any_single_positive_signal_flags():
    assert is_daemon_generated("notes/x.md", record_type="note", generated_by="llm") is True
    assert is_daemon_generated("learn/x.md", record_type="note", generated_by=None) is True
    assert is_daemon_generated("notes/x.md", record_type="learn", generated_by=None) is True


def test_all_negative_signals_do_not_flag():
    assert is_daemon_generated("notes/x.md", record_type="note", generated_by="human") is False


def test_no_signals_is_not_flagged():
    assert is_daemon_generated() is False


# ── 4. Raw-bytes fast path (surveyor diff scan) ──────────────────────────────

@pytest.mark.parametrize("raw", [
    b"---\ntype: learn\ngenerated_by: llm\n---\nbody",
    b'---\ngenerated_by: "llm"\n---\nbody',
])
def test_raw_llm_marker_is_flagged(raw):
    assert is_daemon_generated_raw(raw) is True


@pytest.mark.parametrize("raw", [
    b"---\ntype: note\nauthor: ben\n---\nhand-written body",
    b"generated_by: human\n",
    b"",
])
def test_raw_human_bytes_are_not_flagged(raw):
    assert is_daemon_generated_raw(raw) is False


def test_raw_path_agrees_with_parsed_predicate():
    """The byte grep is the byte-level equivalent of generated_by='llm'."""
    llm_doc = b"---\ngenerated_by: llm\n---\n"
    assert is_daemon_generated_raw(llm_doc) == is_daemon_generated(generated_by="llm")


# ── Schema constants sanity (a schema change needs exactly one edit) ─────────

def test_prefixes_and_types_stay_in_sync():
    assert {p.rstrip("/") for p in DAEMON_OUTPUT_PREFIXES} == set(DAEMON_OUTPUT_TYPES)
