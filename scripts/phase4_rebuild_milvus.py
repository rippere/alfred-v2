#!/usr/bin/env python3
"""
Phase 4 — Drop and rebuild Milvus from scratch after consolidation.

After the vault has been condensed (Phases 1–3), this script:
  1. Verifies Alfred is stopped and Ollama is reachable
  2. Loads old state.json to preserve MemoryStrength records
  3. Drops the vault_v2 Milvus collection
  4. Scans all vault files, chunks them, fits a fresh BM25 index
  5. Embeds every chunk via Ollama (nomic-embed-text)
  6. Upserts into the new Milvus collection
  7. Writes a fresh state.json carrying forward memory strengths

Alfred MUST be stopped: systemctl --user stop alfred.service
Ollama MUST be running: curl http://localhost:11434/api/tags

Usage:
    uv run python scripts/phase4_rebuild_milvus.py --dry-run   (count files, no changes)
    uv run python scripts/phase4_rebuild_milvus.py             (execute full rebuild)

Expected time: ~10 minutes for 2,000 vault files at 0.15s/chunk.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).parent.parent

# Resolved at runtime after arg parsing via _resolve_paths()
DATA_DIR: Path = None   # type: ignore[assignment]
VAULT_PATH: Path = None  # type: ignore[assignment]
CONFIG_PATH: Path = None  # type: ignore[assignment]


def _resolve_paths(config_arg: Path | None) -> None:
    """Set module-level paths from --config or defaults."""
    global DATA_DIR, VAULT_PATH, CONFIG_PATH
    import yaml
    if config_arg is not None:
        CONFIG_PATH = config_arg.resolve()
    else:
        CONFIG_PATH = PROJECT_ROOT / "config.yaml"
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    VAULT_PATH = Path(raw["vault"]["path"]).expanduser()
    data_dir_raw = raw.get("data_dir", "./data")
    DATA_DIR = (CONFIG_PATH.parent / data_dir_raw).resolve()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

IGNORE_DIRS = {"inbox", "wiki", "_archived", "_templates", "_bases", ".obsidian"}
CHECKPOINT_EVERY = 50   # save progress log every N files
THROTTLE = 0.15         # seconds between Ollama embed calls


def load_config() -> dict:
    import yaml
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)  # CONFIG_PATH set by _resolve_paths()


def check_alfred_stopped() -> None:
    pid_path = DATA_DIR / "alfred.pid"
    if pid_path.exists():
        pid_str = pid_path.read_text().strip()
        try:
            import os
            os.kill(int(pid_str), 0)
            print(f"[error] Alfred is still running (PID {pid_str}).")
            print(f"  Stop it: systemctl --user stop alfred.service")
            sys.exit(1)
        except ProcessLookupError:
            pass  # stale PID — fine


def check_ollama(base_url: str) -> None:
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except Exception as e:
        print(f"[error] Ollama not reachable at {base_url}: {e}")
        print(f"  Start Ollama, then re-run this script.")
        sys.exit(1)


def scan_vault() -> list[str]:
    """Return sorted list of rel_paths for all .md files to embed."""
    rel_paths = []
    for fp in sorted(VAULT_PATH.rglob("*.md")):
        rel = fp.relative_to(VAULT_PATH)
        if any(part in IGNORE_DIRS for part in rel.parts):
            continue
        rel_paths.append(str(rel).replace("\\", "/"))
    return rel_paths


def embed_sync(text: str, ollama_url: str, model: str) -> list[float] | None:
    """Synchronous embed call to Ollama with retry."""
    for attempt in range(4):
        try:
            resp = httpx.post(
                f"{ollama_url}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except httpx.HTTPStatusError as e:
            if "input length exceeds" in e.response.text:
                return None  # skip oversized chunk
            delay = 2.0 * (2 ** attempt)
            print(f"    [warn] Ollama HTTP error (attempt {attempt + 1}): {e}")
            time.sleep(delay)
        except Exception as e:
            delay = 2.0 * (2 ** attempt)
            print(f"    [warn] Ollama error (attempt {attempt + 1}): {e}")
            time.sleep(delay)
    return None


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Count files and chunks without modifying Milvus or state.json")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config.yaml (overrides default config.yaml location)")
    args = parser.parse_args()

    _resolve_paths(args.config)

    print("Phase 4 — Milvus Rebuild")
    if args.dry_run:
        print("[DRY RUN]\n")

    cfg = load_config()
    ollama_url = cfg.get("ollama", {}).get("base_url", "http://localhost:11434")
    embed_model = cfg.get("ollama", {}).get("embed_model", "nomic-embed-text")
    milvus_uri = str(DATA_DIR / "milvus.db")
    bm25_path = DATA_DIR / "bm25_index.pkl"

    # Preflight checks
    print("Preflight checks...")
    check_alfred_stopped()
    check_ollama(ollama_url)
    print(f"  Alfred: stopped  ✓")
    print(f"  Ollama: reachable at {ollama_url}  ✓\n")

    # Scan vault
    print("Scanning vault...")
    rel_paths = scan_vault()
    print(f"  Found {len(rel_paths)} files to embed\n")

    # Parse and chunk all files
    print("Parsing and chunking all files...")
    from alfred.core.vault import chunk_record, parse_file

    all_chunks: list[tuple[str, str, str, str]] = []  # (chunk_id, text, record_type, name)
    parse_errors = 0

    for i, rel_path in enumerate(rel_paths):
        try:
            record = parse_file(VAULT_PATH, rel_path)
            chunks = chunk_record(record)
            rec_type = record.frontmatter.get("type", "unknown")
            name = (
                record.frontmatter.get("name")
                or record.frontmatter.get("subject")
                or Path(rel_path).stem
            )
            for chunk_id, text in chunks:
                all_chunks.append((chunk_id, text, rec_type, name))
        except Exception as e:
            parse_errors += 1
            if parse_errors <= 10:
                print(f"  [warn] Parse failed {rel_path}: {e}")

        if (i + 1) % 500 == 0:
            print(f"  Parsed {i + 1}/{len(rel_paths)} files ({len(all_chunks)} chunks so far)...")

    print(f"  Total chunks: {len(all_chunks)} from {len(rel_paths) - parse_errors} files")
    if parse_errors:
        print(f"  Parse errors: {parse_errors}")

    if args.dry_run:
        print(f"\nDry run complete.")
        print(f"  Files:   {len(rel_paths)}")
        print(f"  Chunks:  {len(all_chunks)}")
        print(f"  Errors:  {parse_errors}")
        print(f"\nRe-run without --dry-run to execute the rebuild.")
        return

    # Load old state to preserve memory strengths
    print("\nLoading old state (preserving memory strengths)...")
    from alfred.store.state import StateStore
    from alfred.core.models import FileState, MemoryStrength, PipelineState

    old_memory: dict = {}
    old_curator: dict = {}
    old_distiller: list = []

    state_json = DATA_DIR / "state.json"
    # Also check for most recent backup (if previous run crashed after renaming)
    if not state_json.exists():
        backups = sorted(DATA_DIR.glob("state.json.backup-*"), reverse=True)
        if backups:
            state_json = backups[0]
            print(f"  state.json missing — loading from backup: {state_json.name}")

    if state_json.exists():
        state_store = StateStore(state_json)
        old_state = state_store.load()
        old_memory = {k: v for k, v in old_state.memory.items()}
        old_curator = dict(old_state.curator_processed)
        old_distiller = list(old_state.distiller_runs[-20:])
        print(f"  Preserved {len(old_memory)} memory strength records")
    else:
        print(f"  No existing state found — starting fresh")

    # Back up if state.json (not a backup) is present
    active_state = DATA_DIR / "state.json"
    if active_state.exists():
        backup_path = DATA_DIR / f"state.json.backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        active_state.rename(backup_path)
        print(f"  State backed up to: {backup_path.name}")

    # Fit BM25 on corpus — use fit_and_store() to also save corpus matrix for
    # BM25-only mode on the second machine (adds ~5–15 MB to the pkl).
    print("\nFitting BM25 index on new corpus...")
    from alfred.store.bm25 import BM25Store

    corpus_texts = [text for _, text, _, _ in all_chunks]
    corpus_chunk_ids = [chunk_id for chunk_id, _, _, _ in all_chunks]
    bm25 = BM25Store(bm25_path)
    bm25.fit_and_store(corpus_texts, corpus_chunk_ids)
    bm25.save()
    print(f"  Vocabulary size: {bm25.vocab_size():,} terms")
    print(f"  BM25 index: {bm25_path}  (includes corpus for offline search)")

    # Compute sparse vectors for all chunks
    print("\nComputing sparse vectors...")
    chunk_sparse: dict[str, dict[int, float]] = {}
    for chunk_id, text, _, _ in all_chunks:
        chunk_sparse[chunk_id] = bm25.encode(text)
    print(f"  Done: {len(chunk_sparse)} sparse vectors")

    # Drop and recreate Milvus collection
    print("\nDropping old Milvus collection...")
    from alfred.store.milvus import MilvusStore, COLLECTION
    from pymilvus import MilvusClient

    raw_client = MilvusClient(uri=milvus_uri)
    if raw_client.has_collection(COLLECTION):
        raw_client.drop_collection(COLLECTION)
        print(f"  Dropped: {COLLECTION}")
    raw_client.close()

    # Detect embedding dimensions from first chunk
    print(f"\nDetecting embedding dimensions (calling Ollama)...")
    first_text = all_chunks[0][1] if all_chunks else "test"
    sample_vec = embed_sync(first_text, ollama_url, embed_model)
    if sample_vec is None:
        print("[error] Could not get embedding from Ollama — aborting.")
        print("  Check: ollama run nomic-embed-text")
        sys.exit(1)
    embed_dims = len(sample_vec)
    print(f"  Embedding dims: {embed_dims}")

    store = MilvusStore(uri=milvus_uri, embed_dims=embed_dims)
    print(f"  New collection created: {COLLECTION}")

    # Embed and upsert all chunks
    print(f"\nEmbedding {len(all_chunks)} chunks (this takes ~{len(all_chunks) * THROTTLE / 60:.0f} min)...")
    now_iso = datetime.now(timezone.utc).isoformat()

    # Track chunk_ids per file for state.json
    file_chunk_ids: dict[str, list[str]] = {}
    file_md5s: dict[str, str] = {}

    upserted = 0
    skipped = 0
    failed = 0
    checkpoint_log: dict[str, bool] = {}  # chunk_id → success

    # Pre-compute MD5s
    for rel_path in rel_paths:
        fp = VAULT_PATH / rel_path
        try:
            file_md5s[rel_path] = hashlib.md5(fp.read_bytes()).hexdigest()
        except Exception:
            file_md5s[rel_path] = ""

    for i, (chunk_id, text, record_type, name) in enumerate(all_chunks):
        rel_path = chunk_id.rsplit("::", 1)[0]
        try:
            chunk_index = int(chunk_id.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            chunk_index = 0

        # First chunk uses the already-fetched sample_vec; rest call Ollama
        if i == 0:
            dense = sample_vec
        else:
            dense = embed_sync(text, ollama_url, embed_model)
            time.sleep(THROTTLE)

        if dense is None:
            skipped += 1
            continue

        sparse = chunk_sparse.get(chunk_id, {})
        try:
            store.upsert(
                chunk_id=chunk_id,
                dense=dense,
                sparse=sparse,
                record_type=record_type,
                name=name,
                chunk_index=chunk_index,
            )
            file_chunk_ids.setdefault(rel_path, []).append(chunk_id)
            upserted += 1
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  [warn] Upsert failed {chunk_id}: {e}")

        if (i + 1) % CHECKPOINT_EVERY == 0:
            print(f"  {upserted} upserted / {i + 1} total chunks ({skipped} skipped, {failed} failed)...")

    print(f"\n  Upserted: {upserted} | Skipped: {skipped} | Failed: {failed}")
    print(f"  Milvus count: {store.count()}")

    # Build fresh state.json
    print("\nBuilding fresh state.json...")
    new_state = PipelineState()
    new_state.last_run = now_iso
    new_state.curator_processed = old_curator
    new_state.distiller_runs = old_distiller

    for rel_path in rel_paths:
        chunk_ids = file_chunk_ids.get(rel_path, [])
        if not chunk_ids:
            continue  # file had no successful embeddings

        new_state.files[rel_path] = FileState(
            md5=file_md5s.get(rel_path, ""),
            last_embedded=now_iso,
            chunk_ids=sorted(chunk_ids),
            semantic_cluster_id=-1,      # will be recomputed by Surveyor
            structural_community_id=-1,
        )

        # Preserve memory strength if file still exists
        if rel_path in old_memory:
            new_state.memory[rel_path] = old_memory[rel_path]

    new_state_store = StateStore(DATA_DIR / "state.json")
    new_state_store._state = new_state
    new_state_store.save()

    print(f"  Files tracked:  {len(new_state.files)}")
    print(f"  Memory entries: {len(new_state.memory)} (preserved from old state)")
    print(f"  State written:  {DATA_DIR / 'state.json'}")

    print(f"""
Phase 4 complete.

  Milvus:    {milvus_uri}  ({store.count()} vectors)
  BM25:      {bm25_path}
  State:     {DATA_DIR / 'state.json'}
  Backup:    {backup_path.name}

Next:
  1. systemctl --user start alfred.service
  2. Watch logs: tail -f {DATA_DIR / 'alfred.log'}
  3. Test: .venv/bin/alfred query "What did I learn about agent architecture?"
""")


if __name__ == "__main__":
    main()
