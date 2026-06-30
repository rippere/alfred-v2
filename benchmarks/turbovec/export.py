"""Read-only export of vault embeddings from Alfred's LanceDB → /tmp/tvbench/."""
import numpy as np
import lancedb

db = lancedb.connect("/home/rippere/alfred-v2/data/lancedb")
tbl = db.open_table("vault_v2")
print(f"table version: {tbl.version}")

df = tbl.to_pandas()
print(f"rows: {len(df)}  columns: {list(df.columns)}")

vecs = np.stack(df["vector"].to_numpy()).astype(np.float32)
ids = df["id"].astype(str).tolist()
print(f"vector matrix: {vecs.shape}  dtype: {vecs.dtype}")
print(f"raw float32 bytes: {vecs.nbytes:,} ({vecs.nbytes/1e6:.1f} MB)")

np.save("/tmp/tvbench/vecs.npy", vecs)
with open("/tmp/tvbench/ids.txt", "w") as f:
    f.write("\n".join(ids))
print("exported OK")
