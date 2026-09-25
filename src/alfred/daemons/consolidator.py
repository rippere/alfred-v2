"""ConsolidatorDaemon — cluster summarization via Ollama + wiki page generation + synthesis pass."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import traceback
from datetime import date, datetime, timezone

import httpx
import structlog

from alfred.core.failures import record_failure
from alfred.core.local_llm import LocalLLMUnavailable, complete
from alfred.core.provenance import is_daemon_generated
from alfred.core.vault_ops import vault_create, vault_edit, vault_read
from alfred.daemons.base import BaseDaemon

log = structlog.get_logger()

CONSOLIDATE_INTERVAL = 1800.0   # run every 30 minutes
MIN_MEMBERS = 3                  # skip clusters with fewer files
MAX_MEMBERS_IN_PROMPT = 8        # cap members sent to LLM
SYNTHESIS_BATCH = 5              # max clusters to synthesize per run
# Member bodies per synthesis prompt. The Spark's window is 32K tokens for
# input + output and it answers 400 above that, where Ollama silently cut the
# prompt; the employment vault's synthesis was already arriving at 31.3K.
SYNTHESIS_INPUT_TOKEN_CAP = 24_000

# Over-counts Qwen tokens on purpose: a run of up to 5 letters, each digit,
# each other visible char. Against the Qwen tokenizer on 2,344 vault notes
# (2026-09-24) it read a median 1.25x the real count and fell short on 3,
# worst 0.72x. A flat chars/4 fell short on a quarter of them — Qwen spends a
# token per digit, and table-heavy notes run 1.5 chars per token. The short
# ones are URL-encoded blobs; capped, the eight densest large notes came to
# 26.5K real tokens — over the nominal cap, still inside the window.
_TOKEN_RE = re.compile(r"[^\W\d_]{1,5}|\d|\S")

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
        """One-shot full consolidation (label + synthesize + stubs).

        Legacy composite entry point, kept for the run() fallback loop and
        ad-hoc invocation.  The runner now schedules label_tick /
        synthesis_tick / stubs_tick as three independent APScheduler jobs so
        a slow pass in one responsibility cannot block the others.
        """
        try:
            await self._consolidate()
        except Exception as e:
            self.log.error("consolidator.tick_error", error=str(e))

    async def label_tick(self) -> None:
        """One-shot cluster-labeling pass — called by APScheduler."""
        try:
            await self._label_pass(self.cfg.vault_path)
        except Exception as e:
            self.log.error("consolidator.label_tick_error", error=str(e))

    async def synthesis_tick(self) -> None:
        """One-shot synthesis pass — called by APScheduler.

        Only synthesizes clusters that already carry a label; a cluster
        labeled after this tick fires is picked up on the next interval.
        """
        try:
            await self._synthesis_pass(self.cfg.vault_path)
        except Exception as e:
            self.log.error("consolidator.synthesis_tick_error", error=str(e))

    async def stubs_tick(self) -> None:
        """One-shot wiki-stub generation pass — called by APScheduler."""
        try:
            await self._generate_wiki_stubs(self.cfg.vault_path)
        except Exception as e:
            self.log.error("consolidator.stubs_tick_error", error=str(e))

    async def _consolidate(self) -> None:
        vault_path = self.cfg.vault_path

        # Label clusters via Ollama (Anthropic / deterministic fallbacks)
        await self._label_pass(vault_path)

        # Synthesize high-centrality clusters into synthesis/ pages
        await self._synthesis_pass(vault_path)

        # Generate wiki stub pages for all person/org records (no LLM needed)
        await self._generate_wiki_stubs(vault_path)

    async def _label_pass(self, vault_path) -> None:
        """Label clusters whose membership changed since their label was made.

        This used to relabel every cluster older than 24h, and the writes were
        mostly lost (see _live_cluster), so each pass relabelled nearly every
        eligible cluster: ~3.5K label calls on 2026-09-23 across the personal,
        finance and neuroscience vaults, 761 in the main one, for memberships
        that had not changed.
        """
        state = self.state.state
        labeled = 0
        reused = 0

        # Model-made labels by the membership they describe, so a cluster whose
        # members only moved to a new key (HDBSCAN renumbers when the vault
        # changes) takes its label along instead of costing a call.
        known = {
            c.labeled_members: list(c.label)
            for c in state.clusters.values()
            if c.label and c.labeled_members
        }

        for key, cluster in list(state.clusters.items()):
            members = list(cluster.member_files)
            if len(members) < MIN_MEMBERS:
                continue
            fingerprint = _members_fingerprint(members)
            if cluster.label and cluster.labeled_members == fingerprint:
                continue  # nothing joined or left since it was labeled

            label = known.get(fingerprint)
            if label is not None:
                reused += 1
            else:
                try:
                    text = await self._label_cluster(members, vault_path)
                except Exception:
                    self.log.warning("consolidator.label_error", cluster=key, error=traceback.format_exc())
                    continue
                if text:
                    label = [text]
                    labeled += 1

            live = self._live_cluster(key, fingerprint)
            if live is None:
                continue  # membership moved on mid-call; the next pass labels it
            if label is None:
                # No model answer (backend paused). A name from the paths beats
                # no label, but it is not stamped, so the model gets another go
                # next pass instead of the placeholder sticking for good — and
                # it never replaces a real label the cluster already has.
                if not live.label:
                    live.label = [_label_from_paths(members)]
                continue
            live.label = list(label)
            live.last_labeled = datetime.now(timezone.utc).isoformat()
            live.labeled_members = fingerprint

        if labeled or reused:
            self.log.info("consolidator.labeled", clusters=labeled, reused=reused)
            await self.save_state()

    def _live_cluster(self, key: str, fingerprint: str):
        """The cluster holding exactly these members now, or None.

        Looked up at write time, never taken from the start of the pass: the
        surveyor reclusters on the same 30-min cadence while a pass is waiting
        on the model, and a label or synthesis path written into a cluster it
        has since moved on from is written for nobody.
        """
        clusters = self.state.state.clusters
        cluster = clusters.get(key)
        if cluster is not None and _members_fingerprint(cluster.member_files) == fingerprint:
            return cluster
        for cluster in clusters.values():
            if _members_fingerprint(cluster.member_files) == fingerprint:
                return cluster
        return None

    async def _synthesis_pass(self, vault_path) -> None:
        """Synthesize learn/ clusters into synthesis/ pages, ranked by graph centrality."""
        from alfred.store.graph import GraphStore

        state = self.state.state
        graph = GraphStore(self.cfg.graph_path)

        SYNTHESIS_STALE_DAYS = 30  # only re-synthesize existing pages after 30 days

        # Synthesis pages by the membership they were written from; same idea
        # as the label pass's `known`, for members that moved to a new key.
        done_by_members = {
            c.synthesized_members: c.consolidated_chunk_id
            for c in state.clusters.values()
            if c.synthesized_members and c.consolidated_chunk_id
        }

        ranked: list[tuple[str, object, int]] = []
        # Hold the path-shared lock across load() + the degree reads below so
        # this never observes a graph mid-mutation by a surveyor GraphStore
        # instance operating on the same file (see GraphStore.transaction()).
        with graph.transaction():
            graph.load()
            for key, cluster in state.clusters.items():
                if len(cluster.member_files) < MIN_MEMBERS:
                    continue
                if not cluster.label:
                    continue  # not yet labeled — wait for label pass

                # Unchanged since its last synthesis (or since it was found to
                # hold nothing to synthesize): skip, whatever the page's age.
                # The age check below alone re-ran one 31K-token synthesis 40+
                # times a day: a cluster pointing at a >30-day-old page was
                # re-synthesized every pass because the new page's path never
                # reached the cluster the next pass read.
                fingerprint = _members_fingerprint(cluster.member_files)
                if cluster.labeled_members != fingerprint:
                    # Label not yet made for these members (new, changed, or a
                    # placeholder while the backend was paused). The page's
                    # path comes from the label, so wait for the label pass
                    # rather than write a synthesis under a name about to change.
                    continue
                if cluster.synthesized_members == fingerprint and (
                    not cluster.consolidated_chunk_id
                    or (vault_path / cluster.consolidated_chunk_id).exists()
                ):
                    continue
                moved = done_by_members.get(fingerprint)
                if moved and (vault_path / moved).exists():
                    cluster.consolidated_chunk_id = moved
                    cluster.synthesized_members = fingerprint
                    continue

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
                        except OSError as e:
                            # Can't stat — fall through and re-synthesize (an
                            # LLM call). Cheap to do once, expensive if it's
                            # every tick, so count it.
                            record_failure(
                                "consolidator.synthesis_stat_failed",
                                error=e,
                                path=str(synthesis_path),
                            )
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
                    except OSError as e:
                        record_failure(
                            "consolidator.synthesis_stat_failed",
                            error=e,
                            path=str(vault_path / synthesis_rel),
                        )

                degree_sum = sum(graph.get_node_degree(f) for f in cluster.member_files)
                ranked.append((key, cluster, degree_sum))

        ranked.sort(key=lambda x: x[2], reverse=True)

        synthesized = 0
        settled = 0
        for cluster_key, cluster, _ in ranked[:SYNTHESIS_BATCH]:
            fingerprint = _members_fingerprint(cluster.member_files)
            try:
                path = await self._synthesize_cluster(cluster, vault_path)
            except Exception as e:
                self.log.warning("consolidator.synthesize_error", cluster=cluster_key, error=str(e))
                continue
            if path is None:
                continue  # no answer from the model — retry next pass
            live = self._live_cluster(cluster_key, fingerprint)
            if live is not None:
                live.synthesized_members = fingerprint
                if path:
                    live.consolidated_chunk_id = path
                settled += 1
            if path:
                synthesized += 1
                await asyncio.sleep(2.0)  # gentle rate limit between LLM calls

        if synthesized or settled:
            self.log.info("consolidator.synthesized", clusters=synthesized)
            await self.save_state()

    async def _synthesize_cluster(self, cluster, vault_path) -> str | None:
        """Build a synthesis/ page from a cluster's learn/ members, then mark them absorbed.

        Returns the page's path; "" when the members hold nothing to
        synthesize (none readable, or all daemon output), which stays true
        until the membership changes; None when the model gave nothing back,
        so the next pass retries. The caller records the result on the live
        cluster (_live_cluster) — `cluster` may be stale by the time the model
        answers.
        """
        learn_entries: list[tuple[str, str, str]] = []  # (rel_path, name, body)
        for rel_path in cluster.member_files[:MAX_MEMBERS_IN_PROMPT]:
            try:
                rec = vault_read(vault_path, rel_path)
                name = rec["frontmatter"].get("name") or rel_path.rsplit("/", 1)[-1].replace(".md", "")
                body = rec["body"].strip()
                if body:
                    learn_entries.append((rel_path, name, body))
            except Exception as e:
                # Member silently missing from the synthesis prompt — the
                # resulting summary is quietly built from partial input.
                record_failure("consolidator.member_read_failed", error=e, path=rel_path)

        if not learn_entries:
            return ""

        # Guard: if every source is daemon-generated content, skip synthesis.
        # Synthesizing LLM output produces compounding noise, not knowledge.
        human_sources = [p for p, _, _ in learn_entries if not is_daemon_generated(p)]
        if not human_sources:
            self.log.info("consolidator.skip_all_daemon_sources",
                          cluster=cluster.label, entries=len(learn_entries))
            return ""

        label = cluster.label[0] if cluster.label else "cluster"
        synthesis_body = await self._call_synthesis_llm(label, learn_entries)
        if not synthesis_body:
            return None

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
                return None
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

        path = result.get("path", f"synthesis/{label_slug}.md")
        self.log.info("consolidator.cluster_synthesized", label=label, path=path)
        return path

    async def _call_synthesis_llm(self, label: str, entries: list[tuple[str, str, str]]) -> str:
        """Call Claude API for synthesis (falls back to Ollama if key absent)."""
        entries, estimated = _fit_to_budget(entries, SYNTHESIS_INPUT_TOKEN_CAP)
        if estimated > SYNTHESIS_INPUT_TOKEN_CAP:
            self.log.info(
                "consolidator.synthesis_input_capped",
                label=label,
                estimated_tokens=estimated,
                cap=SYNTHESIS_INPUT_TOKEN_CAP,
            )
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

        # Local backend only. The Anthropic leg that used to sit here was removed
        # with the rest of the cloud chain; the Ollama path below was already the
        # de-facto backend anyway, since the cloud call had been 400-ing on an
        # exhausted credit balance and falling through silently.
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(
                    f"{self.cfg.ollama_base_url}/api/generate",
                    json={"model": self.cfg.ollama_llm_model, "prompt": prompt, "stream": False, "think": False},
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
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(
                    f"{self.cfg.ollama_base_url}/api/generate",
                    json={
                        "model": self.cfg.ollama_llm_model,
                        "prompt": prompt,
                        "stream": False,
                        "think": False,
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

        # ── Fallback 1: local model ───────────────────────────────────────────
        try:
            file_list_short = "\n".join(f"- {f}" for f in member_files[:10])
            fallback_prompt = (
                f"These files are in the same knowledge cluster:\n{file_list_short}\n\n"
                "Give a 2-4 word descriptive label for this cluster. "
                "Return only the label, no explanation."
            )

            def _call() -> str:
                return complete(
                    "You label clusters of related documents concisely.",
                    fallback_prompt,
                    base_url=self.cfg.ollama_base_url,
                    model=self.cfg.ollama_llm_model,
                    max_tokens=20,
                ).strip()

            label = await asyncio.to_thread(_call)
            if label:
                return label
        except LocalLLMUnavailable as e:
            # Deliberately non-fatal here, unlike curator/distiller: fallback 2
            # below derives a label from filenames, so a paused backend costs
            # label quality, not correctness, and never blocks clustering.
            self.log.info("consolidator.label_backend_unavailable", error=str(e))
        except Exception:
            self.log.warning(
                "consolidator.local_label_error",
                error=traceback.format_exc(),
            )

        # No model answer. _label_pass falls back to _label_from_paths, and
        # knows not to treat that name as the cluster's real label.
        return ""


def _label_from_paths(member_files: list[str]) -> str:
    """A placeholder label from the members' folders or file names — no LLM."""
    from pathlib import Path
    from collections import Counter

    dirs = [Path(f).parent.name for f in member_files if Path(f).parent.name not in (".", "")]
    if dirs:
        most_common_dir = Counter(dirs).most_common(1)[0][0]
        return most_common_dir

    stems = [Path(f).stem for f in member_files[:2]]
    return "/".join(stems) if stems else "unlabeled"


def _members_fingerprint(member_files) -> str:
    """Order-independent identity of a cluster's membership."""
    return hashlib.sha1("\n".join(sorted(member_files)).encode()).hexdigest()


def _approx_tokens(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _fit_to_budget(
    entries: list[tuple[str, str, str]], budget: int
) -> tuple[list[tuple[str, str, str]], int]:
    """Trim member bodies so together they come to at most `budget` tokens.

    Returns the entries and their estimated size before trimming. Short bodies
    go in whole; the long ones split what is left evenly, each cut at a token
    boundary and marked, so no single note crowds the others out.
    """
    sizes = [_approx_tokens(body) for _, _, body in entries]
    total = sum(sizes)
    if total <= budget:
        return entries, total

    allowed = [0] * len(entries)
    left = budget
    order = sorted(range(len(entries)), key=sizes.__getitem__)
    for n, i in enumerate(order):
        allowed[i] = min(sizes[i], left // (len(order) - n))
        left -= allowed[i]

    fitted = []
    for (rel, name, body), size, cap in zip(entries, sizes, allowed, strict=True):
        if cap < size:
            cut = 0
            for count, match in enumerate(_TOKEN_RE.finditer(body), start=1):
                if count > cap:
                    break
                cut = match.end()
            body = body[:cut] + "\n[…truncated]"
        fitted.append((rel, name, body))
    return fitted, total
