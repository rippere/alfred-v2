"""Regression tests for graph.pkl node split-brain (audit: janitor/graph split-brain).

Wikilinks are bare stems (``[[foo]]``), but graph nodes for source files are
keyed by full rel_path (``topic/foo.md``). Without resolving link text through
build_wikilink_index, the same file ends up as two disconnected nodes — one
keyed by rel_path (as an edge source), one keyed by raw link text (as an edge
target).
"""
from __future__ import annotations

from alfred.store.graph import GraphStore, build_wikilink_index


def test_build_wikilink_index_maps_stem_and_bare_path_to_rel_path():
    index = build_wikilink_index(["topic/foo.md", "note/bar.md"])
    assert index["foo"] == "topic/foo.md"
    assert index["topic/foo"] == "topic/foo.md"
    assert index["topic/foo.md"] == "topic/foo.md"
    assert index["bar"] == "note/bar.md"


def test_add_edges_from_wikilinks_resolves_target_to_canonical_node():
    store = GraphStore(None)
    index = build_wikilink_index(["topic/foo.md", "note/bar.md"])

    store.add_edges_from_wikilinks("note/bar.md", ["foo"], link_index=index)

    g = store._graph()
    assert g.has_edge("note/bar.md", "topic/foo.md")
    assert not g.has_node("foo")  # no phantom node split from the real file


def test_add_edges_from_wikilinks_without_index_keeps_legacy_behavior():
    """Backward compat: callers that don't pass link_index (e.g. tests that
    already pass full rel_paths as targets) are unaffected."""
    store = GraphStore(None)
    store.add_edges_from_wikilinks("a.md", ["b.md"])
    g = store._graph()
    assert g.has_edge("a.md", "b.md")


def test_build_from_vault_resolves_wikilinks_to_source_node_identity(tmp_path):
    (tmp_path / "topic").mkdir()
    (tmp_path / "note").mkdir()
    (tmp_path / "topic" / "foo.md").write_text(
        "---\ntype: topic\n---\nbody text here.\n", encoding="utf-8"
    )
    (tmp_path / "note" / "bar.md").write_text(
        "---\ntype: note\n---\nSee [[foo]] for details.\n", encoding="utf-8"
    )

    store = GraphStore(tmp_path / "graph.pkl")
    store.build_from_vault(tmp_path)

    g = store._graph()
    assert g.has_node("topic/foo.md")
    assert g.has_node("note/bar.md")
    assert not g.has_node("foo"), "wikilink target must not create a phantom node distinct from the real file"
    assert g.has_edge("note/bar.md", "topic/foo.md")
