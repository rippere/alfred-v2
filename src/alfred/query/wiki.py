"""Wiki fast-path: entity name → vault/wiki/{name}.md before vector search."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WikiHit:
    entity_name: str
    rel_path: str
    content: str
    score: float = 0.95   # wiki hits get a synthetic high score


_CAPITALIZED = re.compile(r"\b([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*)\b")


class WikiFastPath:
    def __init__(self, vault_path: Path, wiki_dir: str = "wiki") -> None:
        self.wiki_path = vault_path / wiki_dir

    def lookup(self, query: str) -> WikiHit | None:
        """Check for a wiki page matching any capitalized n-gram in the query."""
        if not self.wiki_path.exists():
            return None

        candidates = _CAPITALIZED.findall(query)
        # Longest match first — prefer "John Smith" over "John"
        for name in sorted(set(candidates), key=len, reverse=True):
            safe = _safe_name(name)
            page = self.wiki_path / f"{safe}.md"
            if page.exists():
                content = page.read_text(encoding="utf-8")
                rel = str(page.relative_to(self.wiki_path.parent))
                return WikiHit(entity_name=name, rel_path=rel, content=content)
        return None

    def get_page(self, entity_name: str) -> WikiHit | None:
        """Direct entity name lookup (used by MCP vault_entity_lookup)."""
        safe = _safe_name(entity_name)
        page = self.wiki_path / f"{safe}.md"
        if not page.exists():
            return None
        content = page.read_text(encoding="utf-8")
        rel = str(page.relative_to(self.wiki_path.parent))
        return WikiHit(entity_name=entity_name, rel_path=rel, content=content)


def _safe_name(name: str) -> str:
    """Convert entity name to a safe filename (lowercase, underscores)."""
    return re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_").lower()
