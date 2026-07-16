"""NetworkX knowledge graph: wikilink edges + spreading activation."""
from __future__ import annotations

import fcntl
import os
import pickle
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import structlog

log = structlog.get_logger()

# Module-level registry of per-path RLocks, keyed by resolved graph file path.
# surveyor.py and consolidator.py each construct their own GraphStore(...) for
# the same on-disk graph file rather than sharing one long-lived instance —
# without this registry each instance would get its own independent RLock,
# so a concurrent load/mutate/save sequence across instances is not mutually
# exclusive and silently last-writer-wins, losing edges. Keying by resolved
# path (instead of one global lock) still lets stores over distinct graph
# files run concurrently.
_locks_guard = threading.Lock()
_path_locks: dict[str, threading.RLock] = {}


def _lock_for_path(path: Path) -> threading.RLock:
    key = str(Path(path).resolve())
    with _locks_guard:
        lock = _path_locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _path_locks[key] = lock
        return lock


class GraphStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._g = None
        # Guards the load/mutate/save path against thread-pool interleaving —
        # daemon file I/O already hops threads via asyncio.to_thread (surveyor's
        # embedding/HDBSCAN work), so an asyncio.Lock alone can't cover it.
        # Reentrant because build_from_vault -> add_edges_from_wikilinks and
        # spreading_activation -> get_neighbors re-acquire under the same lock.
        # Shared across all GraphStore instances pointed at the same path (see
        # _lock_for_path) so independently-constructed instances in different
        # daemons still serialize against each other.
        self._lock: threading.RLock = _lock_for_path(path)
        # Cross-process exclusive lock, held for the duration of transaction().
        # threading.RLock (above) is process-local — it does nothing for
        # surveyor.py and consolidator.py when `alfred up --only <daemon>`
        # runs them as genuinely separate OS processes against the same
        # graph file, which is a real, currently-supported deployment mode.
        # A sidecar ".lock" file (not the graph file itself) mirrors the
        # pattern in StateStore.save() / lancedb_store's quarantine lock:
        # flock() arbitrates across both separate processes and separate
        # file descriptions within one process.
        self._lock_path = path.with_name(path.name + ".lock")

    @contextmanager
    def transaction(self) -> Iterator["GraphStore"]:
        """Hold the path-shared lock across a multi-step load/mutate/save
        sequence so the whole sequence is atomic relative to any other
        GraphStore instance — including independently-constructed ones, in
        this process or another — pointed at the same path.

        Without the in-process RLock, sharing the lock alone is not enough:
        load(), a mutation, and save() each acquire-and-release the lock
        separately, so another instance's full load/mutate/save cycle can
        still interleave between them and clobber this one's save (lost
        writes). Reentrant, so nested load()/save()/mutate() calls inside
        the `with` block re-acquire the same RLock without deadlocking.

        Without the flock, the RLock alone does nothing across real OS
        processes (e.g. `alfred up --only surveyor` and `--only
        consolidator` each running as their own process against the same
        graph file) — each process gets its own independent RLock object,
        so concurrent processes' load/mutate/save sequences interleave and
        lose writes just as badly as if there were no lock at all. The
        flock is acquired for the whole transaction, matching the RLock's
        scope, and held across nested load()/save() calls the same way
        (flock is per-open-file-description, so re-entering while already
        holding it is a harmless no-op, not a deadlock).
        """
        with self._lock:
            lock_fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
                try:
                    yield self
                finally:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)

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
        #
        # The temp file uses a unique per-call name (tempfile.mkstemp in the
        # same directory, so os.replace() stays an atomic same-filesystem
        # rename) rather than a fixed "<path>.tmp". A fixed name is a second,
        # independent collision hazard on top of the missing cross-process
        # lock: two concurrent processes/instances writing "<path>.tmp" at
        # the same time can interleave their writes into the same inode, or
        # have one process's os.replace() race the other's — this showed up
        # as FileNotFoundError from os.replace() when a fixed-name temp file
        # was already renamed away by a sibling process by the time this one
        # tried to replace from it. A unique name means even a lock-
        # acquisition edge case (or a caller that saves outside of
        # transaction()) can't collide on the temp file itself.
        with self._lock:
            if self._g is not None:
                fd, tmp_name = tempfile.mkstemp(
                    dir=str(self.path.parent), prefix=f".{self.path.name}.", suffix=".tmp"
                )
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(pickle.dumps(self._g))
                    os.replace(tmp_name, self.path)
                except BaseException:
                    Path(tmp_name).unlink(missing_ok=True)
                    raise

    def node_count(self) -> int:
        with self._lock:
            return self._graph().number_of_nodes()

    def edge_count(self) -> int:
        with self._lock:
            return self._graph().number_of_edges()

    def add_edges_from_wikilinks(self, source_rel_path: str, targets: list[str]) -> None:
        with self._lock:
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
