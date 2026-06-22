"""turbovec 2-bit/4-bit vs exact float32 search on the real Alfred vault embeddings.

Ground truth: exact cosine (inner product on L2-normalized vectors).
Queries: 500 held-out vault vectors (self-hit excluded) + optional real text
queries from qtext.npy (embedded via Ollama to mirror live usage).
"""
import json
import os
import time

import numpy as np
from turbovec import TurboQuantIndex

vecs = np.load("/tmp/tvbench/vecs.npy")
N, d = vecs.shape
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
norms[norms == 0] = 1.0
V = np.ascontiguousarray(vecs / norms, dtype=np.float32)

rng = np.random.default_rng(42)
q_idx = rng.choice(N, size=min(500, N), replace=False)
Q = V[q_idx]

# ---- exact ground truth (top-10 excluding self) ----
t0 = time.perf_counter()
S = Q @ V.T
batch_ms_per_q = (time.perf_counter() - t0) * 1000 / len(Q)

gt = []
for i in range(len(Q)):
    row = np.argpartition(-S[i], 12)[:12]
    row = row[np.argsort(-S[i][row])]
    gt.append([int(j) for j in row if j != q_idx[i]][:10])

# exact per-query latency (fair single-query comparison)
lat_exact = []
for i in range(0, len(Q), 25):  # 20 samples
    q = Q[i]
    t0 = time.perf_counter()
    s = V @ q
    np.argpartition(-s, 10)[:10]
    lat_exact.append((time.perf_counter() - t0) * 1000)

out = {
    "N": int(N), "dim": int(d),
    "raw_mb": round(V.nbytes / 1e6, 1),
    "exact": {"p50_ms": round(float(np.median(lat_exact)), 3),
              "batch_ms_per_q": round(batch_ms_per_q, 3),
              "recall@10": 1.0},
}

# ---- optional real text queries ----
qtext = None
if os.path.exists("/tmp/tvbench/qtext.npy"):
    qt = np.load("/tmp/tvbench/qtext.npy").astype(np.float32)
    qt = qt / np.linalg.norm(qt, axis=1, keepdims=True)
    qtext = np.ascontiguousarray(qt)
    gt_text = []
    St = qtext @ V.T
    for i in range(len(qtext)):
        row = np.argpartition(-St[i], 10)[:10]
        gt_text.append(set(int(j) for j in row[np.argsort(-St[i][row])]))

# ---- turbovec at each bit width ----
for bw in (2, 4):
    idx = TurboQuantIndex(dim=d, bit_width=bw)
    t0 = time.perf_counter()
    idx.add(V)
    build_s = time.perf_counter() - t0

    lat, rec = [], []
    for i, q in enumerate(Q):
        qb = np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32)
        t0 = time.perf_counter()
        scores, I = idx.search(qb, k=12)
        lat.append((time.perf_counter() - t0) * 1000)
        I = [int(j) for j in np.asarray(I).ravel().tolist() if int(j) != q_idx[i]][:10]
        rec.append(len(set(I) & set(gt[i])) / 10.0)

    path = f"/tmp/tvbench/idx_{bw}bit.tq"
    idx.write(path)
    entry = {
        "recall@10_vs_exact": round(float(np.mean(rec)), 4),
        "p50_ms": round(float(np.median(lat)), 3),
        "build_s": round(build_s, 2),
        "quantized_mb_theoretical": round(N * d * bw / 8 / 1e6, 2),
        "index_file_mb": round(os.path.getsize(path) / 1e6, 2),
    }
    if qtext is not None:
        trec = []
        for i, q in enumerate(qtext):
            qb = np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32)
            _, I = idx.search(qb, k=10)
            I = set(int(j) for j in np.asarray(I).ravel().tolist())
            trec.append(len(I & gt_text[i]) / 10.0)
        entry["recall@10_real_queries"] = round(float(np.mean(trec)), 4)
    out[f"turbovec_{bw}bit"] = entry

print(json.dumps(out, indent=2))
