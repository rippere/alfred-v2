#!/usr/bin/env python3
"""Migrate embeddings from personal-alfred (vault_embeddings) to alfred-v2 (vault_v2).

What this does:
  1. Reads all 1,939 chunks from the old Milvus DB (reuses dense embeddings — no re-embedding)
  2. Re-parses vault files to get chunk texts (needed for BM25 sparse vectors)
  3. Fits TF-IDF over the corpus, computes sparse vectors
  4. Upserts everything into the new vault_v2 collection
  5. Writes state.json with FileState.chunk_ids populated

Run from the alfred-v2 project root:
    uv run python scripts/migrate_milvus.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

OLD_DB = Path("/mnt/external/Projects/personal-alfred/data/milvus_lite.db")
OLD_COLLECTION = "vault_embeddings"
VAULT_PATH = Path("/mnt/external/obsidian-vault")

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    if not OLD_DB.exists():
        print(f"[error] Old Milvus DB not found: {OLD_DB}")
        sys.exit(1)

    print(f"Opening old DB: {OLD_DB}")
    from pymilvus import MilvusClient
    old_client = MilvusClient(uri=str(OLD_DB))

    if not old_client.has_collection(OLD_COLLECTION):
        print(f"[error] Collection '{OLD_COLLECTION}' not found in old DB")
        print(f"  Collections available: {old_client.list_collections()}")
        sys.exit(1)

    # ── Step 1: Read all old chunks ────────────────────────────────────────────
    print("Reading old chunks...")
    PAGE = 16_000
    old_rows: list[dict] = []
    offset = 0
    while True:
        page = old_client.query(
            collection_name=OLD_COLLECTION,
            filter="",
            output_fields=["id", "embedding", "record_type", "name"],
            limit=PAGE,
            offset=offset,
        )
        if not page:
            break
        old_rows.extend(page)
        if len(page) < PAGE:
            break
        offset += PAGE

    print(f"  Found {len(old_rows)} chunks")

    # ── Step 2: Re-parse vault files for chunk texts ───────────────────────────
    print("Re-parsing vault files for chunk texts...")
    from alfred.core.vault import chunk_record, parse_file

    # Build lookup: chunk_id → text
    chunk_texts: dict[str, str] = {}
    missing_files: set[str] = set()

    # Group chunk_ids by rel_path
    by_file: dict[str, list[str]] = defaultdict(list)
    for row in old_rows:
        rel_path = row["id"].rsplit("::", 1)[0]
        by_file[rel_path].append(row["id"])

    for rel_path, chunk_ids in by_file.items():
        vault_file = VAULT_PATH / rel_path
        if not vault_file.exists():
            missing_files.add(rel_path)
            continue
        try:
            record = parse_file(VAULT_PATH, rel_path)
            chunks = chunk_record(record)
            for cid, text in chunks:
                chunk_texts[cid] = text
        except Exception as e:
            print(f"  [warn] Failed to parse {rel_path}: {e}")

    print(f"  Parsed {len(chunk_texts)} chunks from {len(by_file) - len(missing_files)} files")
    if missing_files:
        print(f"  Skipping {len(missing_files)} chunks from deleted files")

    # ── Step 3: Fit BM25/TF-IDF on all corpus texts ───────────────────────────
    print("Fitting BM25 index...")
    from alfred.store.bm25 import BM25Store

    bm25 = BM25Store(DATA_DIR / "bm25_index.pkl")
    corpus_texts = list(chunk_texts.values())
    if not corpus_texts:
        print("[error] No texts to fit BM25 on — check vault path")
        sys.exit(1)
    bm25.fit(corpus_texts)
    bm25.save()
    print(f"  Vocabulary size: {bm25.vocab_size():,} terms")

    # ── Step 4: Compute sparse vectors for all chunks ─────────────────────────
    print("Computing sparse vectors...")
    chunk_sparse: dict[str, dict[int, float]] = {}
    for chunk_id, text in chunk_texts.items():
        chunk_sparse[chunk_id] = bm25.encode(text)
    print(f"  Done: {len(chunk_sparse)} sparse vectors")

    # ── Step 5: Upsert to new vault_v2 collection ─────────────────────────────
    print("Opening new DB and creating vault_v2 collection...")
    from alfred.store.milvus import MilvusStore

    new_db_path = DATA_DIR / "milvus.db"
    embed_dims = len(old_rows[0]["embedding"]) if old_rows else 768
    store = MilvusStore(uri=str(new_db_path), embed_dims=embed_dims)

    print(f"Upserting {len(old_rows)} chunks (reusing dense embeddings)...")
    upserted = 0
    skipped = 0
    failed: list[str] = []
    for row in old_rows:
        chunk_id: str = row["id"]
        rel_path = chunk_id.rsplit("::", 1)[0]

        if chunk_id not in chunk_texts:
            skipped += 1
            continue

        # Parse chunk_index from id suffix (chunk_NN)
        try:
            chunk_index = int(chunk_id.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            chunk_index = 0

        sparse = chunk_sparse.get(chunk_id, {})
        try:
            store.upsert(
                chunk_id=chunk_id,
                dense=row["embedding"],
                sparse=sparse,
                record_type=row.get("record_type", "unknown"),
                name=row.get("name", rel_path),
                chunk_index=chunk_index,
            )
            upserted += 1
        except Exception as e:
            failed.append(chunk_id)
            print(f"  [warn] upsert failed for {chunk_id!r}: {e}")
            continue

        if upserted % 200 == 0:
            print(f"  {upserted}/{len(old_rows) - skipped}...")

    if failed:
        print(f"  Failed chunks ({len(failed)}):")
        for cid in failed[:10]:
            print(f"    {cid!r}")

    print(f"  Upserted: {upserted} | Skipped (deleted files): {skipped}")
    print(f"  New collection count: {store.count()}")

    # ── Step 6: Build state.json ───────────────────────────────────────────────
    print("Building state.json...")
    from alfred.core.models import FileState, PipelineState
    from alfred.store.state import StateStore

    state_store = StateStore(DATA_DIR / "state.json")
    state = state_store.load()  # empty on first run

    # Build FileState.chunk_ids from what we just upserted
    file_chunks: dict[str, list[str]] = defaultdict(list)
    for row in old_rows:
        chunk_id = row["id"]
        rel_path = chunk_id.rsplit("::", 1)[0]
        if chunk_id in chunk_texts:
            file_chunks[rel_path].append(chunk_id)

    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()

    for rel_path, cids in file_chunks.items():
        vault_file = VAULT_PATH / rel_path
        if not vault_file.exists():
            continue
        import hashlib
        md5 = hashlib.md5(vault_file.read_bytes()).hexdigest()
        state.files[rel_path] = FileState(
            md5=md5,
            last_embedded=now_iso,
            chunk_ids=sorted(cids),
        )

    state.last_run = now_iso
    state_store.save()
    print(f"  state.json written: {len(state.files)} files tracked")

    # ── Done ───────────────────────────────────────────────────────────────────
    print()
    print("Migration complete.")
    print(f"  New DB:      {new_db_path}")
    print(f"  BM25 index:  {DATA_DIR / 'bm25_index.pkl'}")
    print(f"  State:       {DATA_DIR / 'state.json'}")
    print()
    print("Run 'alfred status' to verify.")


if __name__ == "__main__":
    main()
