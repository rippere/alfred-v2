from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from alfred.core.models import (
    ClusterState,
    FileState,
    MemoryStrength,
    PipelineState,
    WikiPage,
)


def _decode_state(raw: dict) -> PipelineState:
    state = PipelineState(
        version=raw.get("version", 1),
        last_run=raw.get("last_run", ""),
        curator_processed=raw.get("curator_processed", {}),
        distiller_runs=raw.get("distiller_runs", []),
        janitor_sweeps=raw.get("janitor_sweeps", []),
    )
    for rel_path, f in raw.get("files", {}).items():
        state.files[rel_path] = FileState(**{
            k: v for k, v in f.items()
            if k in FileState.__dataclass_fields__
        })
    for key, c in raw.get("clusters", {}).items():
        state.clusters[key] = ClusterState(**{
            k: v for k, v in c.items()
            if k in ClusterState.__dataclass_fields__
        })
    for rel_path, m in raw.get("memory", {}).items():
        state.memory[rel_path] = MemoryStrength(**{
            k: v for k, v in m.items()
            if k in MemoryStrength.__dataclass_fields__
        })
    for key, w in raw.get("wiki_pages", {}).items():
        state.wiki_pages[key] = WikiPage(**{
            k: v for k, v in w.items()
            if k in WikiPage.__dataclass_fields__
        })
    return state


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._state: PipelineState = PipelineState()

    def load(self) -> PipelineState:
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self._state = _decode_state(raw)
        else:
            self._state = PipelineState()
        return self._state

    def save(self) -> None:
        self.path.write_text(json.dumps(asdict(self._state), indent=2))

    @property
    def state(self) -> PipelineState:
        return self._state

    # Convenience counters for status display
    def file_count(self) -> int:
        return len(self._state.files)

    def embedded_count(self) -> int:
        return sum(1 for f in self._state.files.values() if f.last_embedded)

    def cluster_count(self) -> int:
        return len(self._state.clusters)

    def wiki_page_count(self) -> int:
        return len(self._state.wiki_pages)

    def chunk_count(self) -> int:
        return sum(len(f.chunk_ids) for f in self._state.files.values())
