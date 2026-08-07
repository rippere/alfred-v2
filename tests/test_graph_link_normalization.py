"""Regression coverage for the graph.pkl split-brain.

add_edges_from_wikilinks added sources as rel_paths (`note/foo.md`) but targets
as raw wikilink text (`note/foo`), so every linked file existed in the graph as
TWO nodes. Measured on the live data/graph.pkl before the fix:

    nodes 25,757   edges 99,479
    .md nodes 16,968   non-.md 8,789   duplicates 6,897
    out-degree of duplicate nodes: [0, 0, 0, 0, 0]

The link-text half collected every inbound edge and had no outbound edges, so
it was a dead end: spreading activation died at hop 2, and the chunk_id
synthesized downstream (f"{rel_path}::chunk_00") matched nothing in LanceDB.

Two things are covered here: the write path never creates the split again, and
merge_link_text_nodes() repairs a graph that already has it.
"""
from __future__ import annotations

import pytest

from alfred.store.graph import GraphStore, normalize_node


@pytest.fixture
def store(tmp_path):
    return GraphStore(tmp_path / "graph.pkl")


# ── normalize_node ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("note/foo", "note/foo.md"),
        ("note/foo.md", "note/foo.md"),
        ("note/foo.MD", "note/foo.MD"),          # already markdown, left alone
        ("foo", "foo.md"),
        ("note\\foo", "note/foo.md"),            # windows separators
        ("  note/foo  ", "note/foo.md"),
        ("/note/foo", "note/foo.md"),            # leading slash is not a new root
        ("note/foo#Heading", "note/foo.md"),     # defensive: heading anchor
        ("note/foo|Alias", "note/foo.md"),       # defensive: display alias
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_node(raw, expected):
    assert normalize_node(raw) == expected


@pytest.mark.parametrize("raw", ["assets/diagram.png", "refs/paper.pdf", "board.canvas"])
def test_normalize_node_leaves_real_attachments_alone(raw):
    """Obsidian keeps the extension for non-note files, so those are already
    real paths — appending .md would invent a file that does not exist."""
    assert normalize_node(raw) == raw


def test_normalize_node_treats_a_dotted_note_name_as_a_note():
    """A note legitimately named `alfred-v2.0-plan` parses as suffix
    '.0-plan'. "Has a suffix" cannot mean "is an attachment"."""
    assert normalize_node("note/alfred-v2.0-plan") == "note/alfred-v2.0-plan.md"


def test_normalize_node_is_idempotent():
    for raw in ["note/foo", "note/foo.md", "assets/diagram.png", "note/a.b.c"]:
        once = normalize_node(raw)
        assert normalize_node(once) == once


# ── write path: the split must not be recreated ──────────────────────────────

def test_link_target_and_its_own_node_are_the_same_node(store):
    """The core defect. `note/alpha.md` links to `[[note/beta]]`; beta then
    gets indexed itself as `note/beta.md`. Before the fix that produced two
    nodes, with all inbound edges on the dead-end half."""
    store.add_edges_from_wikilinks("note/alpha.md", ["note/beta"])
    store.add_edges_from_wikilinks("note/beta.md", ["note/gamma"])

    g = store._graph()
    assert not g.has_node("note/beta"), "raw link text must not survive as a node"
    assert g.has_node("note/beta.md")
    assert store.get_neighbors("note/alpha.md") == [("note/beta.md", 1.0)]


def test_activation_reaches_hop_two(store):
    """The user-visible consequence: with the split, beta had out-degree 0 and
    spreading activation stopped there, so gamma was never reached."""
    store.add_edges_from_wikilinks("note/alpha.md", ["note/beta"])
    store.add_edges_from_wikilinks("note/beta.md", ["note/gamma"])

    activated = store.spreading_activation(["note/alpha.md"], hops=2, decay=0.5)

    assert activated["note/beta.md"] == pytest.approx(0.5)
    assert activated["note/gamma.md"] == pytest.approx(0.25)


def test_repeated_links_reinforce_the_normalized_edge(store):
    """Reinforcement keyed off the raw text would split weight between the two
    spellings of the same target."""
    store.add_edges_from_wikilinks("note/alpha.md", ["note/beta"])
    store.add_edges_from_wikilinks("note/alpha.md", ["note/beta.md"])

    assert store._graph()["note/alpha.md"]["note/beta.md"]["weight"] == pytest.approx(1.1)


def test_self_links_add_no_edge(store):
    """A file linking to itself, spelled either way, is not a traversable
    relationship — and a self-loop would leak activation back into a seed."""
    store.add_edges_from_wikilinks("note/alpha.md", ["note/alpha", "note/alpha.md"])

    g = store._graph()
    assert g.has_node("note/alpha.md")
    assert g.number_of_edges() == 0


def test_empty_targets_are_skipped(store):
    store.add_edges_from_wikilinks("note/alpha.md", ["", "   ", "note/beta"])
    assert sorted(store._graph().nodes()) == ["note/alpha.md", "note/beta.md"]


def test_build_from_vault_produces_no_unnormalized_nodes(tmp_path):
    vault = tmp_path / "vault"
    (vault / "note").mkdir(parents=True)
    (vault / "note" / "alpha.md").write_text("Links to [[note/beta]].\n", encoding="utf-8")
    (vault / "note" / "beta.md").write_text("Links back to [[note/alpha]].\n", encoding="utf-8")

    store = GraphStore(tmp_path / "graph.pkl")
    store.build_from_vault(vault)

    g = store._graph()
    assert sorted(g.nodes()) == ["note/alpha.md", "note/beta.md"]
    assert g.number_of_edges() == 2


# ── repair path: merge_link_text_nodes ───────────────────────────────────────

def _split_brain_graph(store: GraphStore) -> None:
    """Reproduce the pre-fix shape by writing nodes directly, bypassing the
    now-normalizing write path."""
    g = store._graph()
    g.add_node("note/alpha.md")
    g.add_node("note/beta.md")
    g.add_node("note/gamma.md")
    g.add_edge("note/alpha.md", "note/beta", weight=1.0, edge_type="wikilink")
    g.add_edge("note/gamma.md", "note/beta", weight=2.0, edge_type="wikilink")
    g.add_edge("note/beta.md", "note/gamma", weight=1.0, edge_type="wikilink")


def test_merge_collapses_the_duplicate_and_restores_traversal(store):
    _split_brain_graph(store)
    g = store._graph()
    assert g.out_degree("note/beta") == 0, "precondition: the dead end exists"

    merged = store.merge_link_text_nodes()

    assert merged == 2                                    # note/beta and note/gamma
    g = store._graph()
    assert sorted(g.nodes()) == ["note/alpha.md", "note/beta.md", "note/gamma.md"]
    assert g.out_degree("note/beta.md") == 1              # dead end is gone
    activated = store.spreading_activation(["note/alpha.md"], hops=2, decay=0.5)
    assert "note/gamma.md" in activated


def test_merge_sums_weight_instead_of_overwriting(store):
    """If alpha already links to the .md half and also to the link-text half,
    the two link strengths must add, not clobber each other."""
    g = store._graph()
    g.add_edge("note/alpha.md", "note/beta.md", weight=1.0, edge_type="wikilink")
    g.add_edge("note/alpha.md", "note/beta", weight=2.5, edge_type="wikilink")

    store.merge_link_text_nodes()

    assert store._graph()["note/alpha.md"]["note/beta.md"]["weight"] == pytest.approx(3.5)


def test_merge_preserves_edge_type(store):
    _split_brain_graph(store)
    store.merge_link_text_nodes()
    g = store._graph()
    assert g["note/alpha.md"]["note/beta.md"]["edge_type"] == "wikilink"


def test_merge_into_a_cluster_edge_widens_to_wikilink(store):
    """Regression: summing a wikilink's weight into a pre-existing CLUSTER edge
    while leaving edge_type='cluster' hands the merged edge to
    clear_cluster_edges(), which surveyor._recluster() calls on every pass — so
    the wikilink is deleted at the next recluster, and only an md5-gated source
    re-embed could restore it. This mislabelled 241 live edges."""
    g = store._graph()
    g.add_edge("note/alpha.md", "note/beta.md", weight=0.3, edge_type="cluster")
    g.add_edge("note/alpha.md", "note/beta", weight=1.0, edge_type="wikilink")

    store.merge_link_text_nodes()

    edge = store._graph()["note/alpha.md"]["note/beta.md"]
    assert edge["edge_type"] == "wikilink", "must not stay collectable by clear_cluster_edges"
    assert edge["weight"] == pytest.approx(1.3)
    assert store.clear_cluster_edges() == 0, "the merged edge must survive a recluster"


def test_merge_leaves_a_pure_cluster_edge_collectable(store):
    """The widening must not go the other way — an untouched cluster edge stays
    a cluster edge, or reclustering stops being able to clean up after itself."""
    g = store._graph()
    g.add_edge("note/alpha.md", "note/gamma.md", weight=0.3, edge_type="cluster")
    g.add_edge("note/alpha.md", "note/beta", weight=1.0, edge_type="wikilink")

    store.merge_link_text_nodes()

    assert store._graph()["note/alpha.md"]["note/gamma.md"]["edge_type"] == "cluster"


def test_merge_keeps_dangling_links_as_normalized_nodes(store):
    """A link to a note that does not exist yet has no .md twin. It still
    normalizes, so the node is already correct when the file appears."""
    store._graph().add_edge("note/alpha.md", "note/never-written", weight=1.0)

    assert store.merge_link_text_nodes() == 1
    assert store._graph().has_node("note/never-written.md")


def test_merge_leaves_attachments_alone(store):
    store._graph().add_edge("note/alpha.md", "assets/diagram.png", weight=1.0)

    assert store.merge_link_text_nodes() == 0
    assert store._graph().has_node("assets/diagram.png")


def test_merge_is_idempotent(store):
    _split_brain_graph(store)
    store.merge_link_text_nodes()
    before = (store._graph().number_of_nodes(), store._graph().number_of_edges())

    assert store.merge_link_text_nodes() == 0
    assert (store._graph().number_of_nodes(), store._graph().number_of_edges()) == before


def test_merge_survives_a_save_load_round_trip(store, tmp_path):
    _split_brain_graph(store)
    store.merge_link_text_nodes()
    store.save()

    reloaded = GraphStore(tmp_path / "graph.pkl")
    assert reloaded.load()
    assert sorted(reloaded._graph().nodes()) == [
        "note/alpha.md", "note/beta.md", "note/gamma.md"
    ]


def test_merge_on_a_clean_graph_changes_nothing(store):
    store.add_edges_from_wikilinks("note/alpha.md", ["note/beta"])
    before = (store._graph().number_of_nodes(), store._graph().number_of_edges())

    assert store.merge_link_text_nodes() == 0
    assert (store._graph().number_of_nodes(), store._graph().number_of_edges()) == before
