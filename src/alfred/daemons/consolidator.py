"""ConsolidatorDaemon — cluster summarization via Ollama + wiki page generation + synthesis pass."""
from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from datetime import date, datetime, timezone

import httpx
import structlog

from alfred.core.anthropic_client import get_client
from alfred.core.vault_ops import vault_create, vault_edit, vault_read
from alfred.daemons.base import BaseDaemon

log = structlog.get_logger()

CONSOLIDATE_INTERVAL = 1800.0   # run every 30 minutes
MIN_MEMBERS = 3                  # skip clusters with fewer files
MAX_MEMBERS_IN_PROMPT = 8        # cap members sent to LLM
SYNTHESIS_BATCH = 5              # max clusters to synthesize per run

# Record types that should get wiki pages
WIKI_ENTITY_TYPES = {"person", "org"}


_LABEL_PROMPT = """\
You are summarizing a semantic cluster of personal knowledge vault records.

These files have been grouped together by semantic similarity:
{file_list}

In 3-5 words, give a descriptive label for what this cluster is about (e.g. "machine learning research", \
"gym training logs", "reading notes philosophy"). Then on the next line, write a single sentence summarizing \
the cluster's theme.

Format:
LABEL: <3-5 word label>
SUMMARY: <one sentence>"""


class ConsolidatorDaemon(BaseDaemon):
    name = "consolidator"

    def __init__(self, cfg, state, events) -> None:
        super().__init__(cfg, state, events)
        self._last_run = 0.0

    async def run(self) -> None:
        self.log.info("consolidator.start")
        try:
            while not self._stop.is_set():
                now = time.time()
                if now - self._last_run > CONSOLIDATE_INTERVAL:
                    await self._consolidate()
                    self._last_run = now
                await asyncio.sleep(60.0)
        finally:
            await self.save_state()
            self.log.info("consolidator.stopped")

    async def tick(self) -> None:
        """One-shot consolidation — called by APScheduler every consolidator_min_interval_s."""
        try:
            await self._consolidate()
        except Exception as e:
            self.log.error("consolidator.tick_error", error=str(e))

    async def _consolidate(self) -> None:
        state = self.state.state
        vault_path = self.cfg.vault_path
        updated = 0

        for key, cluster in list(state.clusters.items()):
            if len(cluster.member_files) < MIN_MEMBERS:
                continue
            if cluster.label and cluster.last_labeled:
                try:
                    last = datetime.fromisoformat(cluster.last_labeled)
                    age_h = (datetime.now(timezone.utc) - last).total_seconds() / 3600
                    if age_h < 24:
                        continue  # labeled recently enough
                except Exception:
                    pass

            try:
                label = await self._label_cluster(cluster.member_files, vault_path)
                if label:
                    cluster.label = [label]
                    cluster.last_labeled = datetime.now(timezone.utc).isoformat()
                    updated += 1
            except Exception:
                self.log.warning("consolidator.label_error", cluster=key, error=traceback.format_exc())

        if updated:
            self.log.info("consolidator.labeled", clusters=updated)
            await self.save_state()

        # Synthesize high-centrality clusters into synthesis/ pages
        await self._synthesis_pass(vault_path)

        # Generate wiki stub pages for all person/org records (no LLM needed)
        await self._generate_wiki_stubs(vault_path)

    async def _synthesis_pass(self, vault_path) -> None:
        """Synthesize learn/ clusters into synthesis/ pages, ranked by graph centrality."""
        from alfred.store.graph import GraphStore

        state = self.state.state
        graph = GraphStore(self.cfg.graph_path)
        graph.load()

        SYNTHESIS_STALE_DAYS = 30  # only re-synthesize existing pages after 30 days

        ranked: list[tuple[str, object, int]] = []
        for key, cluster in state.clusters.items():
            if len(cluster.member_files) < MIN_MEMBERS:
                continue
            if not cluster.label:
                continue  # not yet labeled — wait for label pass

            # Guard: if consolidated_chunk_id is set, verify the file exists on disk.
            # If it exists and is fresh (< 30 days old), skip — already synthesized.
            # If consolidated_chunk_id is set but the file is gone, allow re-synthesis.
            if cluster.consolidated_chunk_id:
                synthesis_path = vault_path / cluster.consolidated_chunk_id
                if synthesis_path.exists():
                    try:
                        mtime = synthesis_path.stat().st_mtime
                        age_days = (time.time() - mtime) / 86400
                        if age_days < SYNTHESIS_STALE_DAYS:
                            continue  # fresh synthesis exists — skip
                    except OSError:
                        pass  # can't stat — fall through and re-synthesize
                else:
                    # consolidated_chunk_id points to a missing file — reset it
                    cluster.consolidated_chunk_id = ""

            # Additional guard: even without consolidated_chunk_id, check if target
            # synthesis file already exists on disk (race condition / state reset).
            label_slug = "-".join((cluster.label[0] if cluster.label else "cluster").lower().split())[:60]
            synthesis_rel = f"synthesis/{label_slug}.md"
            if not cluster.consolidated_chunk_id and (vault_path / synthesis_rel).exists():
                try:
                    mtime = (vault_path / synthesis_rel).stat().st_mtime
                    age_days = (time.time() - mtime) / 86400
                    if age_days < SYNTHESIS_STALE_DAYS:
                        # File exists and is fresh — adopt it without re-synthesizing
                        cluster.consolidated_chunk_id = synthesis_rel
                        continue
                except OSError:
                    pass

            degree_sum = sum(graph.get_node_degree(f) for f in cluster.member_files)
            ranked.append((key, cluster, degree_sum))

        ranked.sort(key=lambda x: x[2], reverse=True)

        synthesized = 0
        for cluster_key, cluster, _ in ranked[:SYNTHESIS_BATCH]:
            try:
                await self._synthesize_cluster(cluster, vault_path)
                synthesized += 1
                await asyncio.sleep(2.0)  # gentle rate limit between LLM calls
            except Exception as e:
                self.log.warning("consolidator.synthesize_error", cluster=cluster_key, error=str(e))

        if synthesized:
            self.log.info("consolidator.synthesized", clusters=synthesized)
            await self.save_state()

    async def _synthesize_cluster(self, cluster, vault_path) -> None:
        """Build a synthesis/ page from a cluster's learn/ members, then mark them absorbed."""
        learn_entries: list[tuple[str, str, str]] = []  # (rel_path, name, body)
        for rel_path in cluster.member_files[:MAX_MEMBERS_IN_PROMPT]:
            try:
                rec = vault_read(vault_path, rel_path)
                name = rec["frontmatter"].get("name") or rel_path.rsplit("/", 1)[-1].replace(".md", "")
                body = rec["body"].strip()
                if body:
                    learn_entries.append((rel_path, name, body))
            except Exception:
                pass

        if not learn_entries:
            return

        # Guard: if every source is daemon-generated content, skip synthesis.
        # Synthesizing LLM output produces compounding noise, not knowledge.
        daemon_prefixes = ("topic/", "synthesis/", "learn/")
        human_sources = [p for p, _, _ in learn_entries
                         if not any(p.startswith(pfx) for pfx in daemon_prefixes)]
        if not human_sources:
            self.log.info("consolidator.skip_all_daemon_sources",
                          cluster=cluster.label, entries=len(learn_entries))
            return

        label = cluster.label[0] if cluster.label else "cluster"
        synthesis_body = await self._call_synthesis_llm(label, learn_entries)
        if not synthesis_body:
            return

        # Wikilinks pointing back to each learn/ source
        source_links = [
            f"[[{rel.removesuffix('.md')}]]"
            for rel, _, _ in learn_entries
        ]
        full_body = f"{synthesis_body}\n\n## Sources\n" + "\n".join(f"- {lnk}" for lnk in source_links)

        label_slug = "-".join(label.lower().split())[:60]
        synthesis_rel = f"synthesis/{label_slug}.md"
        if (vault_path / synthesis_rel).exists():
            # Update in place: merge source links, replace body with fresh synthesis
            try:
                existing = vault_read(vault_path, synthesis_rel)
                existing_sources = existing["frontmatter"].get("cluster_sources", [])
                merged_sources = list(dict.fromkeys(existing_sources + source_links))
                merged_body = f"{synthesis_body}\n\n## Sources\n" + "\n".join(f"- {lnk}" for lnk in merged_sources)
                vault_edit(
                    vault_path,
                    synthesis_rel,
                    set_fields={"status": "active", "cluster_sources": merged_sources, "generated_by": "llm"},
                    body_replace=merged_body,
                )
            except Exception as e:
                self.log.warning("consolidator.synthesis_update_failed", path=synthesis_rel, error=str(e))
                return
            result = {"path": synthesis_rel}
        else:
            result = vault_create(
                vault_path,
                "synthesis",
                label_slug,
                set_fields={
                    "status": "draft",
                    "cluster_sources": source_links,
                    "confidence": "medium",
                    "created": date.today().isoformat(),
                    "generated_by": "llm",
                },
                body=full_body,
            )

        # Mark each learn/ file absorbed
        for rel_path, _, _ in learn_entries:
            try:
                vault_edit(vault_path, rel_path, set_fields={"status": "absorbed"})
            except Exception as e:
                self.log.debug("consolidator.absorb_skip", path=rel_path, error=str(e))

        cluster.consolidated_chunk_id = result.get("path", f"synthesis/{label_slug}.md")
        self.log.info("consolidator.cluster_synthesized", label=label, path=cluster.consolidated_chunk_id)

    async def _call_synthesis_llm(self, label: str, entries: list[tuple[str, str, str]]) -> str:
        """Call Claude API for synthesis (falls back to Ollama if key absent)."""
        bodies_text = "\n\n".join(f"### {name}\n{body}" for _, name, body in entries)
        prompt = (
            f"You are synthesizing atomic knowledge fragments into a coherent insight document "
            f"for a personal knowledge vault.\n\n"
            f"Cluster topic: {label}\n\n"
            f"Atomic learnings:\n{bodies_text}\n\n"
            f"Write a synthesis in this exact format (300-400 words):\n\n"
            f"## Insight\n<2-3 sentences: the core unified insight across all fragments>\n\n"
            f"## Evidence\n<What the fragments show; reference key points>\n\n"
            f"## Implications\n<What this means practically>\n\n"
            f"## Applicability\n<When and where this insight applies>\n\n"
            f"Return only the formatted synthesis — no preamble."
        )

        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if anthropic_key and self.state.can_make_api_call(daemon="consolidator"):
            try:
                client = get_client()

                def _call():
                    resp = client.messages.create(
                        model=self.cfg.anthropic_model,
                        max_tokens=600,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return resp

                resp = await asyncio.to_thread(_call)
                usage = resp.usage
                self.state.record_api_call(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cached_tokens=getattr(usage, "cache_read_input_tokens", 0),
                )
                return resp.content[0].text.strip()
            except Exception as e:
                self.log.warning("consolidator.claude_error", error=str(e))

        # Ollama fallback
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.cfg.ollama_base_url}/api/generate",
                    json={"model": self.cfg.ollama_llm_model, "prompt": prompt, "stream": False},
                )
                resp.raise_for_status()
                return resp.json().get("response", "").strip()
        except Exception as e:
            self.log.warning("consolidator.ollama_synthesis_error", error=str(e))
            return ""

    async def _generate_wiki_stubs(self, vault_path) -> None:
        """Create wiki stub pages for all person and org records that don't have one yet."""
        try:
            from alfred.wiki.writer import WikiWriter
        except ImportError:
            return

        writer = WikiWriter(self.cfg, self.state)
        state = self.state.state
        created = 0

        for rel_path, fs in list(state.files.items()):
            try:
                rec = vault_read(vault_path, rel_path)
                fm = rec["frontmatter"]
                rec_type = fm.get("type", "")
                if rec_type not in WIKI_ENTITY_TYPES:
                    continue
                name = fm.get("name") or fm.get("subject")
                if not name:
                    continue
                key = name.lower()
                if key not in state.wiki_pages:
                    writer.ensure_page(name, rec_type, rel_path)
                    created += 1
            except Exception as e:
                self.log.debug("consolidator.wiki_skip", path=rel_path, error=str(e))

        if created:
            self.log.info("consolidator.wiki_stubs_created", count=created)
            await self.save_state()

    async def _label_cluster(self, member_files: list[str], vault_path) -> str:
        # Build file list for prompt
        lines = []
        for rel_path in member_files[:MAX_MEMBERS_IN_PROMPT]:
            try:
                rec = vault_read(vault_path, rel_path)
                fm = rec["frontmatter"]
                name = fm.get("name") or fm.get("subject") or rel_path.rsplit("/", 1)[-1]
                rec_type = fm.get("type", "")
                lines.append(f"- [{rec_type}] {name}")
            except Exception:
                lines.append(f"- {rel_path}")

        if len(member_files) > MAX_MEMBERS_IN_PROMPT:
            lines.append(f"  (and {len(member_files) - MAX_MEMBERS_IN_PROMPT} more...)")

        file_list = "\n".join(lines)
        prompt = _LABEL_PROMPT.format(file_list=file_list)

        # ── Primary: Ollama ───────────────────────────────────────────────────
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.cfg.ollama_base_url}/api/generate",
                    json={
                        "model": self.cfg.ollama_llm_model,
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "").strip()

            # Parse LABEL: line
            for line in raw.splitlines():
                if line.upper().startswith("LABEL:"):
                    return line.split(":", 1)[-1].strip()
            if raw:
                return raw[:50].strip()
        except Exception:
            self.log.warning(
                "consolidator.ollama_error",
                error=traceback.format_exc(),
            )

        # ── Fallback 1: Anthropic ─────────────────────────────────────────────
        if not self.state.can_make_api_call(daemon="consolidator"):
            return ""
        try:
            from alfred.core.anthropic_client import get_client

            client = get_client()
            file_list_short = "\n".join(f"- {f}" for f in member_files[:10])
            fallback_prompt = (
                f"These files are in the same knowledge cluster:\n{file_list_short}\n\n"
                "Give a 2-4 word descriptive label for this cluster. "
                "Return only the label, no explanation."
            )

            def _call():
                response = client.messages.create(
                    model=self.cfg.anthropic_model,
                    max_tokens=20,
                    messages=[{"role": "user", "content": fallback_prompt}],
                )
                return response.content[0].text.strip()

            label = await asyncio.to_thread(_call)
            if label:
                return label
        except Exception:
            self.log.warning(
                "consolidator.anthropic_label_error",
                error=traceback.format_exc(),
            )

        # ── Fallback 2: deterministic from file names ─────────────────────────
        from pathlib import Path
        from collections import Counter

        dirs = [Path(f).parent.name for f in member_files if Path(f).parent.name not in (".", "")]
        if dirs:
            most_common_dir = Counter(dirs).most_common(1)[0][0]
            return most_common_dir

        stems = [Path(f).stem for f in member_files[:2]]
        return "/".join(stems) if stems else "unlabeled"
