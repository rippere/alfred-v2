"""NetworkX knowledge graph: wikilink edges + spreading activation."""
from __future__ import annotations

import os
import pickle
import threading
from collections.abc import Iterable
from pathlib import Path

import structlog

log = structlog.get_logger()


def build_wikilink_index(rel_paths: Iterable[str]) -> dict[str, str]:
    """Map every way a wikilink can spell a file to its canonical rel_path.

    Wikilinks are bare stems (``[[foo]]``), never full paths, but graph nodes
    for source files are keyed by full rel_path (``topic/foo.md``). Without
    resolving link text through this index, the same file ends up as two
    disconnected nodes — one keyed by rel_path (as an edge source), one keyed
    by raw link text (as an edge target) — a graph.pkl node split-brain.
    """
    index: dict[str, str] = {}
    for rel_str in rel_paths:
        stem = Path(rel_str).stem
        index.setdefault(stem, rel_str)
        index.setdefault(rel_str.removesuffix(".md"), rel_str)
        index.setdefault(rel_str, rel_str)
    return index


class GraphStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._g = None
        # Guards the load/mutate/save path against thread-pool interleaving —
        # daemon file I/O already hops threads via asyncio.to_thread (surveyor's
        # embedding/HDBSCAN work), so an asyncio.Lock alone can't cover it.
        # Reentrant because build_from_vault -> add_edges_from_wikilinks and
        # spreading_activation -> get_neighbors re-acquire under the same lock.
        self._lock: threading.RLock = threading.RLock()

    def _graph(self):
        if self._g is None:
            import networkx as nx
            self._g = nx.DiGraph()
        return self._g

    def is_empty(self) -> bool:
        with self._lock:
            return self._g is None or self._g.number_of_nodes() == 0

    def load(self) -> bool:
        with self._lock:
            if not self.path.exists():
                return False
            try:
                import networkx as nx
                self._g = pickle.loads(self.path.read_bytes())
                return True
            except Exception as e:
                log.warning("graph.load_failed", error=str(e))
                return False

    def save(self) -> None:
        # Temp file + os.replace() so a mid-write kill (OOM, SIGKILL) can never
        # leave a truncated pickle behind — matches StateStore.save().
        with self._lock:
            if self._g is not None:
                tmp = self.path.with_name(self.path.name + ".tmp")
                tmp.write_bytes(pickle.dumps(self._g))
                os.replace(tmp, self.path)

    def node_count(self) -> int:
        with self._lock:
            return self._graph().number_of_nodes()

    def edge_count(self) -> int:
        with self._lock:
            return self._graph().number_of_edges()

    def add_edges_from_wikilinks(
        self,
        source_rel_path: str,
        targets: list[str],
        link_index: dict[str, str] | None = None,
    ) -> None:
        """Add edges from source_rel_path to each of targets.

        targets are raw wikilink text (bare stems) by default. Pass
        link_index (from build_wikilink_index) to resolve them to the
        canonical rel_path of the file they point to — otherwise the same
        file can end up keyed as two disconnected nodes (see
        build_wikilink_index docstring). Unresolvable links are dropped
        rather than added as phantom nodes.
        """
        with self._lock:
            g = self._graph()
            g.add_node(source_rel_path)
            if link_index is not None:
                resolved = [link_index[" ".join(t.split())] for t in targets if " ".join(t.split()) in link_index]
            else:
                resolved = targets
            for t in resolved:
                g.add_node(t)
                if not g.has_edge(source_rel_path, t):
                    g.add_edge(source_rel_path, t, weight=1.0, edge_type="wikilink")
                else:
                    g[source_rel_path][t]["weight"] += 0.1   # reinforce repeated links

    def clear_cluster_edges(self) -> int:
        """Remove all cluster-type edges. Returns count removed."""
        with self._lock:
            g = self._graph()
            to_remove = [(u, v) for u, v, d in g.edges(data=True) if d.get("edge_type") == "cluster"]
            g.remove_edges_from(to_remove)
            return len(to_remove)

    def add_cluster_edges(self, cluster_members: list[str], max_fan: int = 5) -> None:
        """Add weak edges within a semantic cluster.

        Small clusters (≤ max_fan*2): all-pairs.
        Large clusters: ring topology with max_fan forward links per node — O(N*k)
        instead of O(N²), keeps the subgraph connected without flooding spreading activation.
        """
        with self._lock:
            g = self._graph()
            n = len(cluster_members)
            if n < 2:
                return
            if n <= max_fan * 2:
                for i, a in enumerate(cluster_members):
                    for b in cluster_members[i + 1:]:
                        if not g.has_edge(a, b):
                            g.add_edge(a, b, weight=0.3, edge_type="cluster")
                            g.add_edge(b, a, weight=0.3, edge_type="cluster")
            else:
                for i, a in enumerate(cluster_members):
                    for step in range(1, max_fan + 1):
                        b = cluster_members[(i + step) % n]
                        if not g.has_edge(a, b):
                            g.add_edge(a, b, weight=0.3, edge_type="cluster")
                            g.add_edge(b, a, weight=0.3, edge_type="cluster")

    def remove_file(self, rel_path: str) -> None:
        with self._lock:
            g = self._graph()
            if g.has_node(rel_path):
                g.remove_node(rel_path)

    def get_node_degree(self, rel_path: str) -> int:
        """Return total in+out degree for centrality ranking."""
        with self._lock:
            g = self._graph()
            if not g.has_node(rel_path):
                return 0
            return g.in_degree(rel_path) + g.out_degree(rel_path)

    def get_neighbors(self, rel_path: str) -> list[tuple[str, float]]:
        """Return [(neighbor_rel_path, weight)] for direct neighbors."""
        with self._lock:
            g = self._graph()
            if not g.has_node(rel_path):
                return []
            return [
                (nbr, g[rel_path][nbr].get("weight", 1.0))
                for nbr in g.successors(rel_path)
            ]

    def spreading_activation(
        self,
        seeds: list[str],
        hops: int = 2,
        decay: float = 0.5,
    ) -> dict[str, float]:
        """BFS spreading activation from seed nodes.

        Returns {rel_path: activation_score} for nodes activated but not in seeds.
        """
        with self._lock:
            activated: dict[str, float] = {}
            seed_set = set(seeds)

            # Initialize from seeds using their retrieval score as base activation
            frontier: list[tuple[str, float]] = [(s, 1.0) for s in seeds if self._graph().has_node(s)]

            for _ in range(hops):
                next_frontier: list[tuple[str, float]] = []
                for node, activation in frontier:
                    for nbr, weight in self.get_neighbors(node):
                        if nbr in seed_set:
                            continue
                        propagated = activation * decay * weight
                        if nbr not in activated:
                            activated[nbr] = propagated
                            next_frontier.append((nbr, propagated))
                        else:
                            activated[nbr] += propagated
                frontier = next_frontier

            return activated

    def build_from_vault(self, vault_path: Path, ignore_dirs: list[str] | None = None) -> None:
        """Rebuild graph from wikilinks in all vault files."""
        import networkx as nx
        from alfred.core.vault import extract_wikilinks, parse_file

        with self._lock:
            ignore = set(ignore_dirs or [])
            self._g = nx.DiGraph()

            md_files = []
            for md_file in vault_path.rglob("*.md"):
                rel = md_file.relative_to(vault_path)
                if any(part in ignore for part in rel.parts):
                    continue
                md_files.append((md_file, str(rel).replace("\\", "/")))

            link_index = build_wikilink_index(rel_str for _, rel_str in md_files)

            for md_file, rel_str in md_files:
                try:
                    raw = md_file.read_text(encoding="utf-8")
                    links = extract_wikilinks(raw)
                    if links:
                        self.add_edges_from_wikilinks(rel_str, links, link_index=link_index)
                    else:
                        self._g.add_node(rel_str)
                except (OSError, UnicodeDecodeError):
                    continue

            log.info("graph.built", nodes=self._g.number_of_nodes(), edges=self._g.number_of_edges())
