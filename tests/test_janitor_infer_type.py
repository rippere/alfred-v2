"""Regression tests for JanitorDaemon._infer_type directory-collision bug.

Several types share a directory (decision/assumption/constraint/contradiction/
learn all live under topic/; session/conversation/ai-dialogue all live under
session/). A naive {v: k for k, v in TYPE_DIRECTORY.items()} reversal picks
whichever type is declared LAST for a shared directory, silently mis-inferring
type for files missing a `type` field.
"""
from __future__ import annotations

from alfred.daemons.janitor import JanitorDaemon

# _infer_type reads no instance state, so it can be called unbound.
_infer_type = JanitorDaemon._infer_type


def test_infer_type_topic_collision_prefers_topic():
    assert _infer_type(None, "topic/some-note.md") == "topic"


def test_infer_type_session_collision_prefers_session():
    assert _infer_type(None, "session/some-note.md") == "session"


def test_infer_type_dedicated_directories_unaffected():
    assert _infer_type(None, "project/some-project.md") == "project"
    assert _infer_type(None, "person/some-person.md") == "person"


def test_infer_type_unknown_directory_returns_empty():
    assert _infer_type(None, "not-a-real-dir/foo.md") == ""


def test_infer_type_top_level_file_returns_empty():
    assert _infer_type(None, "foo.md") == ""
