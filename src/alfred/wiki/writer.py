"""WikiWriter — create and update wiki entity pages from vault records."""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import frontmatter
import structlog

from alfred.core.vault_ops import VaultError, vault_create, vault_edit, vault_read
from alfred.store.state import StateStore

log = structlog.get_logger()

_UPDATE_PROMPT = """\
You are updating a wiki page for a personal knowledge vault.

Entity: {entity_name} (type: {entity_type})

Existing known facts:
{existing_facts}

New source records that mention this entity:
{new_sources}

Extract any new, distinct facts about this entity from the new sources. \
Do NOT repeat facts already listed above.

Output a JSON object:
  "new_facts": list of strings (each a concise factual statement, max 5 new facts)
  "related": list of entity names that appear related to {entity_name} (max 5, plain strings)

Respond with only the JSON object."""


class WikiWriter:
    def __init__(self, cfg, state_store: StateStore) -> None:
        self.cfg = cfg
        self.state = state_store

    def ensure_page(self, entity_name: str, entity_type: str, source_rel_path: str) -> str:
        """Create wiki page if it doesn't exist. Returns rel_path."""
        state = self.state.state
        key = entity_name.lower()

        if key in state.wiki_pages:
            page = state.wiki_pages[key]
            if source_rel_path not in page.sources:
                page.sources.append(source_rel_path)
                page.updated = datetime.now(timezone.utc).isoformat()
            return page.rel_path

        slug = _slugify(entity_name)
        rel_path = f"wiki/{slug}.md"
        try:
            vault_create(
                self.cfg.vault_path,
                "wiki",
                slug,
                set_fields={"entity_type": entity_type},
                body=f"# {entity_name}\n\n",
            )
        except VaultError:
            pass

        from alfred.core.models import WikiPage
        now = datetime.now(timezone.utc).isoformat()
        state.wiki_pages[key] = WikiPage(
            entity_name=entity_name,
            entity_type=entity_type,
            rel_path=rel_path,
            created=now,
            updated=now,
            sources=[source_rel_path],
        )
        log.info("wiki.page_created", entity=entity_name, path=rel_path)
        return rel_path

    def enrich_page(self, entity_name: str, new_source_paths: list[str]) -> bool:
        """Call LLM to extract new facts from new sources. Returns True if updated."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return False

        state = self.state.state
        key = entity_name.lower()
        page = state.wiki_pages.get(key)
        if not page:
            return False

        source_texts = []
        for rel_path in new_source_paths[:5]:
            try:
                rec = vault_read(self.cfg.vault_path, rel_path)
                source_texts.append(f"[{rel_path}]:\n{rec['body'][:500]}")
            except Exception:
                pass

        if not source_texts:
            return False

        existing = "\n".join(f"- {f}" for f in page.known_facts) or "(none yet)"
        prompt = _UPDATE_PROMPT.format(
            entity_name=entity_name,
            entity_type=page.entity_type,
            existing_facts=existing,
            new_sources="\n\n".join(source_texts[:3]),
        )

        try:
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=self.cfg.anthropic_model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
        except Exception as e:
            log.warning("wiki.enrich_error", entity=entity_name, error=str(e))
            return False

        new_facts = [f for f in data.get("new_facts", []) if f and f not in page.known_facts]
        new_related = [r for r in data.get("related", []) if r and r not in page.related]

        if not new_facts and not new_related:
            return False

        page.known_facts.extend(new_facts)
        page.related.extend(new_related)
        page.updated = datetime.now(timezone.utc).isoformat()

        # Rebuild wiki page body
        facts_md = "\n".join(f"- {f}" for f in page.known_facts)
        related_md = "\n".join(f"- [[{r}]]" for r in page.related)
        new_body = f"# {page.entity_name}\n\n"
        if facts_md:
            new_body += f"## Facts\n{facts_md}\n\n"
        if related_md:
            new_body += f"## Related\n{related_md}\n"

        try:
            vault_edit(self.cfg.vault_path, page.rel_path, body_replace=new_body)
            log.info("wiki.page_enriched", entity=entity_name, new_facts=len(new_facts))
        except VaultError as e:
            log.warning("wiki.edit_error", entity=entity_name, error=str(e))

        return True


def _slugify(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:80] or "untitled"
