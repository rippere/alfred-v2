"""NetworkX knowledge graph: wikilink edges + spreading activation."""
from __future__ import annotations

import pickle
from pathlib import Path

import structlog

log = structlog.get_logger()


class GraphStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._g = None

    def _graph(self):
        if self._g is None:
            import networkx as nx
            self._g = nx.DiGraph()
        return self._g

    def is_empty(self) -> bool:
        return self._g is None or self._g.number_of_nodes() == 0

    def load(self) -> bool:
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
        if self._g is not None:
            self.path.write_bytes(pickle.dumps(self._g))

    def node_count(self) -> int:
        return self._graph().number_of_nodes()

    def edge_count(self) -> int:
        return self._graph().number_of_edges()

    def add_edges_from_wikilinks(self, source_rel_path: str, targets: list[str]) -> None:
        g = self._graph()
        g.add_node(source_rel_path)
        for t in targets:
            g.add_node(t)
            if not g.has_edge(source_rel_path, t):
                g.add_edge(source_rel_path, t, weight=1.0, edge_type="wikilink")
            else:
                g[source_rel_path][t]["weight"] += 0.1   # reinforce repeated links

    def clear_cluster_edges(self) -> int:
        """Remove all cluster-type edges. Returns count removed."""
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
        g = self._graph()
        if g.has_node(rel_path):
            g.remove_node(rel_path)

    def get_node_degree(self, rel_path: str) -> int:
        """Return total in+out degree for centrality ranking."""
        g = self._graph()
        if not g.has_node(rel_path):
            return 0
        return g.in_degree(rel_path) + g.out_degree(rel_path)

    def get_neighbors(self, rel_path: str) -> list[tuple[str, float]]:
        """Return [(neighbor_rel_path, weight)] for direct neighbors."""
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

        ignore = set(ignore_dirs or [])
        self._g = nx.DiGraph()

        for md_file in vault_path.rglob("*.md"):
            rel = md_file.relative_to(vault_path)
            if any(part in ignore for part in rel.parts):
                continue
            rel_str = str(rel).replace("\\", "/")
            try:
                raw = md_file.read_text(encoding="utf-8")
                links = extract_wikilinks(raw)
                if links:
                    self.add_edges_from_wikilinks(rel_str, links)
                else:
                    self._g.add_node(rel_str)
            except (OSError, UnicodeDecodeError):
                continue

        log.info("graph.built", nodes=self._g.number_of_nodes(), edges=self._g.number_of_edges())
