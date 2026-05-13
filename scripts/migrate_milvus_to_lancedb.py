"""Migrate vector data from Milvus Lite to LanceDB.

Usage
-----
    uv run python scripts/migrate_milvus_to_lancedb.py [--config config.yaml]

What it does
------------
1. Tries to open the existing Milvus database and read all rows.
2. Writes every row into the new LanceDB store.
3. Prints a summary.

Notes
-----
- If the alfred daemon is running and holding the Milvus file-lock, this
  script will block/fail on step 1.  Stop the daemon first:

      systemctl --user stop alfred.service

- Alternatively, skip step 1 entirely by passing --fresh.  The surveyor
  will re-embed all vault files on the next daemon startup.

- The LanceDB directory is NEVER wiped by this script — existing rows in
  LanceDB are kept and incoming Milvus rows are upserted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the project src is importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Milvus → LanceDB")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to Alfred config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Skip Milvus read — just initialise an empty LanceDB table and exit.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[1] / config_path

    print(f"Loading config from {config_path}")
    from alfred.config import AlfredConfig
    cfg = AlfredConfig.load(config_path)

    print(f"LanceDB URI : {cfg.lancedb_uri}")
    print(f"Milvus URI  : {cfg.milvus_uri}")

    # Always open/create the LanceDB store so the table is ready.
    from alfred.store.lancedb_store import LanceDBStore
    lance_store = LanceDBStore(
        uri=cfg.lancedb_uri,
        collection=cfg.milvus_collection,
        dims=cfg.embed_dims,
    )
    print(f"LanceDB rows before migration : {lance_store.count()}")

    if args.fresh:
        print("--fresh specified — skipping Milvus read.  LanceDB table is initialised.")
        print("The surveyor will re-embed all vault files on next startup.")
        return

    # Try to open Milvus.
    milvus_path = Path(cfg.milvus_uri)
    if not milvus_path.exists():
        print(f"Milvus database not found at {milvus_path}.")
        print("If this is a fresh install, start the daemon and it will embed everything into LanceDB.")
        return

    print(f"Opening Milvus at {cfg.milvus_uri} ...")
    try:
        from alfred.store.milvus import MilvusStore
    except ImportError:
        print("pymilvus not installed — cannot read Milvus data.")
        print("Run with --fresh to skip migration, or install: uv pip install pymilvus")
        sys.exit(1)

    try:
        milvus_store = MilvusStore(
            uri=cfg.milvus_uri,
            embed_dims=cfg.embed_dims,
            collection=cfg.milvus_collection,
        )
    except Exception as e:
        print(f"Could not open Milvus: {e}")
        print("If the daemon is running, stop it first: systemctl --user stop alfred.service")
        print("Alternatively, run with --fresh to skip Milvus and start clean.")
        sys.exit(1)

    print("Reading all rows from Milvus (this can take a minute for large vaults)...")
    try:
        rows = milvus_store.query_all(output_fields=["id", "embedding", "record_type", "name", "chunk_index"])
    except Exception as e:
        print(f"query_all failed: {e}")
        sys.exit(1)

    total = len(rows)
    print(f"Found {total} rows in Milvus.")

    if total == 0:
        print("Nothing to migrate.")
        return

    migrated = 0
    errors = 0
    for i, row in enumerate(rows, 1):
        chunk_id    = row.get("id", "")
        dense       = row.get("embedding") or []
        record_type = row.get("record_type", "") or ""
        name        = row.get("name", "") or ""
        chunk_index = int(row.get("chunk_index", 0) or 0)

        if not chunk_id or not dense:
            errors += 1
            continue

        try:
            lance_store.upsert(
                chunk_id=chunk_id,
                dense=dense,
                sparse={},          # BM25 is managed separately by BM25Store
                record_type=record_type,
                name=name,
                chunk_index=chunk_index,
            )
            migrated += 1
        except Exception as e:
            print(f"  [WARN] upsert failed for {chunk_id}: {e}")
            errors += 1

        if i % 500 == 0:
            print(f"  {i}/{total} rows processed...")

    print()
    print("Migration complete.")
    print(f"  Migrated : {migrated}")
    print(f"  Errors   : {errors}")
    print(f"  LanceDB rows after migration : {lance_store.count()}")


if __name__ == "__main__":
    main()
