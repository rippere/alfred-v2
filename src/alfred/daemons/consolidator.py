"""ConsolidatorDaemon — cluster summarization via Ollama."""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone

import httpx
import structlog

from alfred.core.vault_ops import vault_read
from alfred.daemons.base import BaseDaemon

log = structlog.get_logger()

CONSOLIDATE_INTERVAL = 1800.0   # run every 30 minutes
MIN_MEMBERS = 3                  # skip clusters with fewer files
MAX_MEMBERS_IN_PROMPT = 8        # cap members sent to Ollama


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
            except Exception as e:
                self.log.warning("consolidator.label_error", cluster=key, error=str(e))

        if updated:
            self.log.info("consolidator.labeled", clusters=updated)
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
        except Exception as e:
            self.log.warning("consolidator.ollama_error", error=str(e))
            return ""

        # Parse LABEL: line
        for line in raw.splitlines():
            if line.upper().startswith("LABEL:"):
                return line.split(":", 1)[-1].strip()
        return raw[:50].strip()
